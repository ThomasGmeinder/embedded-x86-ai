# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""How a sentence becomes a behavior: the context, the registry, the gauntlet, the queue.

:func:`handle_transcript` runs one transcript through the gauntlet on the
calling thread (the workshop's TODO 3.2):

1. stop fast path - a stop keyword stops the running behavior at once,
   without an LLM round trip;
2. noise / hallucination gate;
3. parse - the grammar-constrained LLM picks the intent;
4. dispatch - the registry's handler runs, unless a stop came in since the
   transcript arrived.

:class:`Dispatcher` puts a queue in front of it for the always-hot listener
(Whisper) or the ``--text-commands`` stdin loop. ``on_transcript`` takes the
startup grace window, the stop fast path and the noise gate at once, on the
listener's thread, and queues the rest; a new command while a behavior runs
stops it so the new one takes over. ``run`` (main thread) hands each queued
transcript to :func:`handle_transcript`, draining a backlog to the newest
command so stale behaviors never replay. It blocks on the queue;
``Dispatcher.stop`` wakes it to see its ``shutdown`` event.

Every step is reported to ``ctx.listeners`` (``fn(event, **data)``) so the
console and the digital-twin tests can timeline it.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from vla_pipeline.llm.intents import CommandRegistry, Intent, ParsedCommand, is_stop, make_noise_gate

logger = logging.getLogger(__name__)

_STOP = object()  # queued by Dispatcher.stop: run() wakes and checks its shutdown event


@dataclass
class PipelineContext:
    """Everything a behavior handler may need, in one bag.

    Handlers receive ``(ctx, cmd)``: the arm, the models, the cameras, the
    mimic mapper, shared state and the stop event. ``stop_check`` is what
    every behavior loop polls. The stop counter lets a stop heard while a
    command was queued or parsed cancel that command.
    """

    cfg: dict
    arm: object
    pose: object | None = None  # PoseEstimator (NPU)
    detector: object | None = None  # ObjectDetector (NPU)
    hands: object | None = None  # HandTracker (CPU)
    mount_camera: object | None = None  # camera client, mount role (faces the person)
    arm_camera: object | None = None  # camera client, on the wrist
    mapper: object | None = None  # MimicMapper; mimic keeps its smoothing state here
    registry: CommandRegistry | None = None
    shared_state: dict = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    listeners: list = field(default_factory=list)  # fn(event, **data) per gauntlet step
    headless: bool = False
    max_frames: int = 0  # mimic: 0 = until stopped; a test runs a few frames
    _stops: int = field(default=0, repr=False)  # stops requested so far
    _stop_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def stop_check(self) -> bool:
        """True while a stop is in effect."""
        return self.stop_event.is_set()

    def request_stop(self) -> None:
        """Stop the running behavior, and any command queued or parsed before now."""
        with self._stop_lock:
            self._stops += 1
            self.stop_event.set()

    def stops_heard(self) -> int:
        """How many stops so far. Read it when a transcript arrives and hand it to :meth:`begin`."""
        with self._stop_lock:
            return self._stops

    def begin(self, stops_before: int) -> bool:
        """Clear the stop event for a new behavior, unless a stop came in since
        ``stops_before``: then return False, and the behavior must not run."""
        with self._stop_lock:
            if self._stops != stops_before:
                return False
            self.stop_event.clear()
            return True

    def say(self, message: str) -> None:
        """Print (and log) one line of robot speech."""
        print(f"{message}")
        logger.info("[say] %s", message)

    def emit(self, event: str, **data) -> None:
        """Report one gauntlet step to every listener."""
        for fn in list(self.listeners):
            try:
                fn(event, **data)
            except Exception as e:  # pragma: no cover - listener bugs stay local
                logger.debug("dispatch listener failed: %s", e)


def build_registry() -> CommandRegistry:
    """Register every behavior the pipeline ships on a fresh registry.

    STOP is deliberately not a handler: it is an interrupt, handled before
    dispatch ever happens.
    """
    from vla_pipeline.behaviors.dance import run_dance
    from vla_pipeline.behaviors.gesture_mimic import run_mimic
    from vla_pipeline.behaviors.manipulation import register_manipulation
    from vla_pipeline.behaviors.simple import run_home, run_wave

    registry = CommandRegistry()
    registry.add(Intent.GESTURE_MIMIC, run_mimic)
    registry.add(Intent.DANCE, run_dance)
    registry.add(Intent.WAVE, run_wave)
    registry.add(Intent.HOME, run_home)
    register_manipulation(registry)
    return registry


def handle_transcript(ctx: PipelineContext, text: str, parse, stops_before: int | None = None) -> ParsedCommand | None:
    """Route one raw transcript through the gauntlet; return what it parsed.

    The order is a safety property:

    1. **Stop fast path**: a stop keyword in the RAW text stops at once
       (``ctx.request_stop()``) and returns ``ParsedCommand(STOP)``. A stop
       never goes through the LLM.
    2. **Noise gate**: a known hallucination does nothing and returns
       ``ParsedCommand(UNKNOWN)``.
    3. **Parse**: ``parse(text)``. ``stops_before`` is the stop count when the
       transcript arrived (default: now); a stop since then cancels it.
    4. **Dispatch**: UNKNOWN tells the user nothing was caught; no handler
       says so; otherwise ``ctx.begin(stops_before)`` clears the stop event so
       a stale stop never kills the *next* behavior, unless a stop came in:
       then the command does not run. The handler runs, and the stop event
       is cleared again afterwards. A crash holds position.
    """
    if is_stop(text):
        ctx.request_stop()
        ctx.say("Stopping.")
        ctx.emit("stop", text=text)
        return ParsedCommand(Intent.STOP, transcript=text)
    if make_noise_gate(ctx.cfg)(text):
        logger.info("Ignoring %r (noise/hallucination gate).", text)
        ctx.emit("ignored", text=text, reason="noise")
        return ParsedCommand(Intent.UNKNOWN, transcript=text)

    heard = ctx.stops_heard() if stops_before is None else stops_before
    if ctx.stops_heard() != heard:
        logger.info("Stop heard before %r was dispatched - command not parsed.", text)
        ctx.emit("cancelled", text=text)
        return None
    try:
        cmd = parse(text)
    except Exception:
        logger.exception("Intent parse failed for %r.", text)
        ctx.say("I couldn't understand that command because the language service is unavailable.")
        ctx.emit("parse_error", text=text)
        return None
    ctx.emit("parsed", text=text, intent=cmd.intent.value, params=dict(cmd.params or {}))
    if cmd.intent in (Intent.UNKNOWN, Intent.STOP):
        if cmd.intent is Intent.UNKNOWN:
            ctx.say("I didn't catch a command in that.")
        return cmd
    handler = ctx.registry.get(cmd.intent)
    if handler is None:
        ctx.say(f"No behavior registered for '{cmd.intent.value}'.")
        return cmd
    if not ctx.begin(heard):
        logger.info("Stop heard while %r was parsed - %s not run.", text, cmd.intent.value)
        ctx.emit("cancelled", text=text, intent=cmd.intent.value)
        return cmd

    logger.info("=== %s %s ===", cmd.intent.value, cmd.params or "")
    ctx.emit("behavior_start", intent=cmd.intent.value, params=dict(cmd.params or {}))
    ok = True
    try:
        # a handler that returns False ran but didn't achieve its goal
        ok = handler(ctx, cmd) is not False
    except Exception as e:
        ok = False
        logger.exception("Behavior %s crashed - holding position.", cmd.intent.value)
        ctx.emit("behavior_crash", intent=cmd.intent.value, error=repr(e))
    finally:
        stopped = ctx.stop_event.is_set()
        ctx.stop_event.clear()
        logger.info("=== %s ended (%s) ===", cmd.intent.value, "stopped" if stopped else "ok" if ok else "failed")
        ctx.emit("behavior_end", intent=cmd.intent.value, ok=ok, stopped=stopped)
    return cmd


class Dispatcher:
    """Owns the transcript queue and runs behaviors from it through :func:`handle_transcript`."""

    def __init__(self, ctx: PipelineContext, parse, *, startup_grace_s: float = 0.0, clock=time.monotonic):
        self.ctx = ctx
        self.parse = parse
        self.is_noise = make_noise_gate(ctx.cfg)
        self.startup_grace_s = float(startup_grace_s)
        self._clock = clock
        self.start_time = clock()
        self.transcripts: queue.Queue[tuple[int, str] | object] = queue.Queue()
        self.current: ParsedCommand | None = None
        self.busy = threading.Event()  # set while a behavior runs
        self.holding = threading.Event()  # set while the console is resetting the scene
        self._work = threading.Condition()
        self._pending = 0  # transcripts queued and not yet run or dropped, the running one included
        ctx.listeners.append(self._track)

    @property
    def listeners(self) -> list:
        """The context's listeners: ``fn(event, **data)`` per gauntlet step."""
        return self.ctx.listeners

    def _track(self, event: str, **data) -> None:
        if event == "behavior_start":
            self.current = ParsedCommand(Intent.from_string(data["intent"]), params=dict(data.get("params") or {}))
            self.busy.set()
        elif event == "behavior_end":
            self.current = None
            self.busy.clear()

    # ---------------------------------------------------------------- input
    def on_transcript(self, text: str) -> None:
        """Gate one transcript (grace, stop, noise) and queue it for dispatch."""
        logger.info("Heard: %r", text)
        self.ctx.emit("heard", text=text)
        if self._clock() - self.start_time < self.startup_grace_s:
            logger.info("Ignoring %r (startup grace period).", text)
            self.ctx.emit("ignored", text=text, reason="grace")
            return
        if is_stop(text):
            self.ctx.request_stop()
            self.ctx.say("Stopping.")
            self.ctx.emit("stop", text=text)
            return
        if self.is_noise(text):
            logger.info("Ignoring %r (noise/hallucination gate).", text)
            self.ctx.emit("ignored", text=text, reason="noise")
            return
        with self._work:
            self._pending += 1
        self.transcripts.put((self.ctx.stops_heard(), text))
        self.ctx.emit("queued", text=text)
        if self.busy.is_set():
            # A new command switches behaviors: stop the running one gracefully;
            # run() then dispatches the newest queued command.
            logger.info("New command while %s runs - stopping it to switch.",
                        self.current.intent.value if self.current else "a behavior")
            self.ctx.stop_event.set()

    # ------------------------------------------------------------- dispatch
    def dispatch(self, text: str, generation: int | None = None) -> ParsedCommand | None:
        """Parse and run one transcript on the calling thread (:func:`handle_transcript`)."""
        return handle_transcript(self.ctx, text, self.parse, stops_before=generation)

    def _take_all(self) -> list[tuple[int, str]]:
        """Every queued transcript, oldest first; a queued stop stays queued."""
        texts, stops = [], 0
        while True:
            try:
                item = self.transcripts.get_nowait()
            except queue.Empty:
                break
            if item is _STOP:
                stops += 1
            else:
                texts.append(item)
        for _ in range(stops):
            self.transcripts.put(_STOP)
        return texts

    def _finished(self, count: int) -> None:
        if count:
            with self._work:
                self._pending -= count
                self._work.notify_all()

    def drain_to_newest(self) -> None:
        """Drop queued transcripts except the newest one."""
        texts = self._take_all()
        if texts:
            self.transcripts.put(texts[-1])
        self._finished(max(len(texts) - 1, 0))

    def discard(self) -> None:
        """Drop every queued transcript."""
        self._finished(len(self._take_all()))

    def wait_idle(self, timeout: float | None) -> bool:
        """True once no transcript is queued and no behavior runs; False when ``timeout`` passes first
        (``None``: no limit)."""
        with self._work:
            return self._work.wait_for(lambda: self._pending == 0, timeout)

    def stop(self) -> None:
        """Wake :meth:`run` so it sees its ``shutdown`` event (set that first)."""
        self.transcripts.put(_STOP)

    def run(self, shutdown: threading.Event) -> None:
        """Dispatch loop until ``shutdown`` is set and :meth:`stop` is called."""
        while not shutdown.is_set():
            item = self.transcripts.get()
            if item is _STOP:
                continue
            generation, text = item
            try:
                if not self.holding.is_set():
                    self.dispatch(text, generation)
            finally:
                self._finished(1)
            if self.holding.is_set():
                self.discard()
                continue
            self.drain_to_newest()
