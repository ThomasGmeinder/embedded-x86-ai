"""The demo's session: commands in, picks run one at a time, one state for the picture.

``DemoSession`` owns the state the picture is drawn from: an immutable
:class:`Snapshot` replaced whole, the transcript, the voice level, and one
``changed`` event the renderer blocks on. One worker thread runs requests on
the shared pick-and-place loop (``pick_place.PickPlaceScene``), one at a time
with at most one waiting behind it (a newer request replaces the waiting one):
a typed sentence is first given to Gemma on that thread, a spoken one arrives
with Gemma's fields already; a new layout is a request too, so only the worker
ever touches the scene. Nothing here sleeps to wait: the worker blocks on a
condition and ends when the session closes, callers of ``wait_idle`` block
once on the ``idle`` event.

``DemoStack`` builds the real parts (the Gemma server and client, SAM 3, SmolVLA,
the SAPIEN scene, the microphone) around a session and closes them.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.demo_console import Snapshot, VoiceLevel, cubes_in_bin
from so101_smolvla.pick_place import (
    DEFAULT_STEPS_PER_PICK,
    PickCommand,
    PickPlaceScene,
    PlanView,
    StepView,
    command_from_fields,
)
from so101_smolvla.voice.commands import SPOKEN, TYPED, CommandFields
from so101_smolvla.voice.intents import ParsedCommand

TRANSCRIPT_LINES = 40


@dataclass(frozen=True)
class CommandRequest:
    """A sentence to run: typed (Gemma is asked on the worker) or spoken (Gemma's fields came with it)."""

    sentence: str
    how: str  # TYPED or SPOKEN
    parsed: ParsedCommand | None = None
    seconds: float = 0.0  # Gemma's time for a spoken sentence


@dataclass(frozen=True)
class LayoutRequest:
    seed: int


def fields_from_parsed(parsed: ParsedCommand, sentence: str, how: str, seconds: float) -> CommandFields:
    """Gemma's answer as the fields the pick-and-place loop reads."""
    return CommandFields(
        sentence=sentence,
        how=how,
        command=parsed.intent.value,
        object=parsed.params.get("object", ""),
        place=parsed.params.get("place", ""),
        heard=parsed.transcript,
        seconds=seconds,
    )


class DemoSession:
    """The state of the demo and the one worker that changes the scene."""

    def __init__(self, scene: PickPlaceScene, parse: Callable[[str], ParsedCommand], *, seed: int) -> None:
        self.scene = scene
        self.parse = parse  # a typed sentence to Gemma's answer
        self.seed = seed
        self.snapshot = Snapshot(
            seed=seed, status="starting", outline_source=scene.outline_source,
            steps_per_pick=scene.steps_per_pick,
        )
        self.voice = VoiceLevel()
        self._transcript: deque = deque(maxlen=TRANSCRIPT_LINES)
        self.changed = threading.Event()  # something to draw
        self.idle = threading.Event()  # no request runs and none waits
        self.results: list[dict] = []
        # One request runs, at most one waits behind it; a newer one replaces the waiting one.
        # ``_pending`` holds the requests not yet claimed by the worker: the one it is about to
        # take and, at most, one more. All of it lives under one condition, so a request is
        # never lost before it started and ``idle`` is never set while one waits.
        self._state = threading.Condition()
        self._current = None
        self._pending: list = []
        self._stopping = False
        self._worker = threading.Thread(target=self._serve, daemon=True, name="demo-picks")

    # -- what the picture reads ------------------------------------------------

    @property
    def transcript(self) -> tuple:
        return tuple(self._transcript)

    def say(self, text: str, kind: str = "note") -> None:
        self._transcript.append((text, kind))
        self.changed.set()

    def _update(self, **fields) -> None:
        self.snapshot = replace(self.snapshot, **fields)
        self.changed.set()

    # -- the microphone's callbacks (other threads) -----------------------------

    def on_level(self, level: float, speaking: bool) -> None:
        self.voice = replace(self.voice, available=True, level=level, speaking=speaking, time=time.monotonic())
        self.changed.set()

    def on_busy(self, busy: bool) -> None:
        self.voice = replace(self.voice, busy=busy)
        self.changed.set()

    def microphone_failed(self, error: str) -> None:
        self.voice = replace(self.voice, available=False, error=error)
        self.say("no microphone: type a command", "error")

    def on_spoken(self, parsed: ParsedCommand, seconds: float = 0.0) -> None:
        """A command the Listener found in the microphone's sound."""
        self._request(CommandRequest(parsed.transcript or "(spoken)", SPOKEN, parsed, seconds))

    # -- requests --------------------------------------------------------------

    def start(self) -> "DemoSession":
        observation = self.scene.observation
        if observation is None:
            raise RuntimeError("start the scene's layout first")
        self._show_layout(observation)
        self.idle.set()
        self._worker.start()
        return self

    def submit(self, sentence: str) -> None:
        """A typed sentence; Gemma is asked on the worker thread."""
        sentence = sentence.strip()
        if sentence:
            self._request(CommandRequest(sentence, TYPED))

    def new_layout(self, seed: int) -> None:
        self._request(LayoutRequest(seed))

    def wait_idle(self, timeout_s: float) -> bool:
        """One bounded wait until no request runs or waits."""
        return self.idle.wait(timeout_s)

    def close(self) -> None:
        with self._state:
            self._stopping = True
            self._state.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=600)

    def _request(self, request) -> None:
        dropped = None
        with self._state:
            # Behind the running request, or the one the worker is about to take, one may wait.
            waiting = 1 if self._current is None and self._pending else 0
            if len(self._pending) > waiting:
                dropped = self._pending.pop()
            self._pending.append(request)
            self.idle.clear()
            self._state.notify_all()
        if dropped is not None:
            self.say("replaced the waiting command", "note")
        self.changed.set()

    # -- the worker --------------------------------------------------------------

    def _serve(self) -> None:
        while True:
            with self._state:
                # Blocks until a request waits or the session closes.
                self._state.wait_for(lambda: self._pending or self._stopping)
                if self._stopping:
                    return
                request = self._pending.pop(0)
                self._current = request
            try:
                if isinstance(request, LayoutRequest):
                    self._run_layout(request)
                else:
                    self._run_command(request)
            except Exception as exc:  # the picture tells; the worker lives on
                self.say(f"error: {exc}", "error")
                self._update(status=f"error: {exc}"[:60])
            with self._state:
                self._current = None
                if not self._pending:
                    self.idle.set()

    def _show_layout(self, observation) -> None:
        facts = self.scene.env.facts
        self.snapshot = Snapshot(
            seed=self.seed, status="idle; say or type a command",
            outline_source=self.scene.outline_source, steps_per_pick=self.scene.steps_per_pick,
            pictures={"top": observation.top, "wrist": observation.wrist},
            opening=float(observation.state[5]), cubes_in_bin=cubes_in_bin(facts),
        )
        self.changed.set()

    def _run_layout(self, request: LayoutRequest) -> None:
        self.seed = request.seed
        self._update(status=f"new layout, seed {request.seed}...", command=None, cube=None, step=None, plans={})
        observation = self.scene.start_layout(request.seed, request.seed)
        self.say(f"new layout, seed {request.seed}", "note")
        self._show_layout(observation)

    def _run_command(self, request: CommandRequest) -> None:
        if request.parsed is None:
            self._update(status="asking Gemma...")
            began = time.perf_counter()
            parsed = self.parse(request.sentence)
            seconds = time.perf_counter() - began
        else:
            parsed, seconds = request.parsed, request.seconds
        fields = fields_from_parsed(parsed, request.sentence, request.how, seconds)
        pick = command_from_fields(fields, request.sentence)
        self.say(f"you: {fields.heard or request.sentence}", "you")
        self.say(f"-> {pick.command}  ({seconds:.2f} s)", "call")
        if pick.cube is None:
            self.say(f"refused: {pick.refusal}; nothing moves", "error")
            self._update(status=f"refused: {pick.refusal}"[:60])
            result = self.scene.run_command(pick)
            self.results.append(result)
            return
        self.say(f"picking the {pick.cube} cube", "note")
        self._update(status=f"picking the {pick.cube} cube", command=pick.command, cube=pick.cube, step=0,
                     placed_at=None, in_bin=False, contact=False, grasped=False)
        result = self.scene.run_command(pick, observer=self)
        self.results.append(result)
        placed = result["first_success_step"]
        self.say(
            f"{pick.cube} cube: {'placed, rule held at step ' + str(placed) if placed is not None else 'not placed'}; "
            f"in the bin at the end: {'yes' if result['in_bin_at_end'] else 'no'}",
            "note",
        )
        self._update(status="idle; say or type a command", command=None, cube=None, step=None,
                     cubes_in_bin=cubes_in_bin(self.scene.env.facts))

    # -- the loop's observer (worker thread) ----------------------------------------

    def on_plan(self, view: PlanView) -> None:
        self._update(plans=dict(view.feeds))

    def on_step(self, view: StepView) -> None:
        observation = view.observation
        placed = self.snapshot.placed_at
        if view.success and placed is None:
            placed = view.step
        self._update(
            step=view.step,
            pictures={"top": observation.top, "wrist": observation.wrist},
            opening=float(observation.state[5]),
            contact=view.contact, grasped=view.grasped, in_bin=view.in_bin, placed_at=placed,
            status=f"picking the {view.cube} cube: step {view.step + 1} of {view.steps_per_pick}"
            + ("; in the bin" if view.in_bin else ""),
        )


class DemoStack:
    """The real parts around a session: Gemma, SAM 3, SmolVLA, the SAPIEN scene, the microphone."""

    def __init__(
        self,
        seed: int = 0,
        *,
        pictures: str = "others-painted",
        steps_per_pick: int = DEFAULT_STEPS_PER_PICK,
        log_dir: Path | str = "logs",
        layouts_per_scene: int = 3,
    ) -> None:
        self.seed = seed
        self.pictures = pictures
        self.steps_per_pick = steps_per_pick
        self.log_dir = Path(log_dir)
        self.layouts_per_scene = layouts_per_scene
        self.server = self.client = self.sam3 = self.env = self.scene = self.session = self.listener = None

    def start(self, say: Callable[[str], None] = print) -> "DemoStack":
        from so101_smolvla.policy_env import create_policy_environment
        from so101_smolvla.sam3_outlines import Sam3Outlines
        from so101_smolvla.sapien_config import THREE_CUBE_TASK
        from so101_smolvla.sapien_rollout import _action, _load_policy
        from so101_smolvla.voice.client import GemmaClient
        from so101_smolvla.voice.server import GemmaServer
        from so101_smolvla.voice.settings import load_settings

        settings = load_settings()
        began = time.perf_counter()
        self.server = GemmaServer(self.log_dir, settings=settings).start()
        self.client = GemmaClient(self.server.base, settings)
        say(f"Gemma 4 server at {self.server.base} after {time.perf_counter() - began:.1f} s; log {self.server.log_path}")
        began = time.perf_counter()
        self.sam3 = Sam3Outlines()
        say(f"SAM 3 worker ready after {time.perf_counter() - began:.1f} s")
        began = time.perf_counter()
        checkpoint, policy, pre, post = _load_policy()
        say(f"SmolVLA {checkpoint.name} loaded after {time.perf_counter() - began:.1f} s")
        began = time.perf_counter()
        # The horizon only sets the simulator's truncation flag; room for several commands.
        self.env = create_policy_environment(
            "sapien",
            horizon=self.steps_per_pick * len(BIN_CUBE_COLORS) * self.layouts_per_scene,
            task_mode=THREE_CUBE_TASK,
        )
        self.scene = PickPlaceScene(
            self.env, policy, pre, post, pictures=self.pictures, sam3=self.sam3,
            steps_per_pick=self.steps_per_pick, act=_action, record=False,
        )
        self.scene.start_layout(self.seed, self.seed)
        say(f"SAPIEN scene, layout seed {self.seed}, after {time.perf_counter() - began:.1f} s")
        self.session = DemoSession(self.scene, self.client.parse, seed=self.seed).start()
        return self

    def listen(self, device=None, wait_s: float = 10.0) -> bool:
        """Start the microphone; False, with the reason in the picture, when it cannot be opened."""
        from so101_smolvla.voice.microphone import Listener

        session = self.session
        self.listener = Listener(self.client, device=device).start(
            on_command=session.on_spoken, on_level=session.on_level, on_busy=session.on_busy,
        )
        if not self.listener.opened.wait(wait_s) or self.listener.error:
            session.microphone_failed(self.listener.error or f"not opened within {wait_s:.0f} s")
            return False
        session.say("listening; say a command", "note")
        return True

    def close(self) -> None:
        for part, close in (
            (self.listener, lambda p: p.stop()),
            (self.session, lambda p: p.close()),
            (self.sam3, lambda p: p.close()),
            (self.env, lambda p: p.close()),
            (self.client, lambda p: p.close()),
            (self.server, lambda p: p.close()),
        ):
            if part is not None:
                close(part)
        self.listener = self.session = self.sam3 = self.env = self.client = self.server = None
