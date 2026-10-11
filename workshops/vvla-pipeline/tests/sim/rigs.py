# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Test rigs: the real pipeline / workshop stacks wired to the digital twin.

A rig owns one scenario: it builds the stack's own context (arm transport,
cameras, models, LLM client, dispatcher) with only the sensors swapped for
the scripted sources, records joint/goal/current traces for assertions, and
films the run with :class:`so101_sim.SimRecorder`.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from scripted import (
    Clock,
    ScriptedMountCamera,
    ScriptedTranscripts,
    SignalBoard,
    SimDetector,
    TimelineHands,
    TimelinePose,
)

logger = logging.getLogger(__name__)

MOTORS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


@dataclass
class Sample:
    t: float
    joints: dict
    goal: dict
    current: float  # gripper Present_Current (raw counts)
    grasped: dict = field(default_factory=dict)
    heights: dict = field(default_factory=dict)


class Trace:
    """Samples the twin at ``hz`` on a thread (scenario clock)."""

    def __init__(self, world, clock: Clock, hz: float = 50.0, watch=()):
        self.world, self.clock, self.hz = world, clock, hz
        self.watch = tuple(watch)  # object bodies to track
        self.samples: list[Sample] = []
        self._run = False
        self._thread = None

    def start(self) -> None:
        if self._run:
            return
        self._run = True
        self._thread = threading.Thread(target=self._loop, name="trace", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._run = False
        if self._thread is not None:
            self._thread.join(2.0)

    def _loop(self) -> None:
        w = self.world
        while self._run:
            snap = w.snapshot()
            cur = snap["current_a"]["gripper"] * 1000.0 / w.current.ma_per_unit
            grasped = {b: w.is_grasped(b) for b in self.watch}
            heights = {b: float(w.body_pos(b)[2]) for b in self.watch}
            self.samples.append(Sample(self.clock(), snap["joints"], snap["goal"], cur, grasped, heights))
            time.sleep(1.0 / self.hz)

    # -- queries ------------------------------------------------------------
    def window(self, t0: float = 0.0, t1: float = float("inf")) -> list[Sample]:
        return [s for s in self.samples if t0 <= s.t <= t1]

    def series(self, joint: str, t0: float = 0.0, t1: float = float("inf"), goal: bool = False):
        win = self.window(t0, t1)
        key = "goal" if goal else "joints"
        return np.array([s.t for s in win]), np.array([getattr(s, key)[joint] for s in win])

    def currents(self, t0: float = 0.0, t1: float = float("inf")) -> np.ndarray:
        return np.array([s.current for s in self.window(t0, t1)])

    def heights(self, body: str) -> np.ndarray:
        return np.array([s.heights[body] for s in self.samples])

    def grasped_seconds(self, body: str) -> float:
        """Longest continuous stretch with ``body`` held between both jaws."""
        best = run = 0.0
        prev = None
        for s in self.samples:
            if s.grasped.get(body):
                run += 0.0 if prev is None else s.t - prev
                best = max(best, run)
            else:
                run = 0.0
            prev = s.t
        return best


class _Rig:
    """Common scenario plumbing; subclasses build the stack's context."""

    target = "?"

    def __init__(self, twin, cfg: dict, recorder_path=None, title: str = ""):
        self.world = twin.world
        self.renderer = twin.renderer
        self.cfg = copy.deepcopy(cfg)
        self.clock = Clock()
        self.board = SignalBoard()
        self.events: list[tuple[float, str, dict]] = []
        self.said: list[str] = []
        self.pose = TimelinePose(self.clock)
        self.hands = TimelineHands(self.clock, self._hand_obs_cls())
        self.mount = ScriptedMountCamera(self.pose, self.hands, self.board)
        self.world.reset()
        for obj in self.world.objects:
            self.world.hide(obj)
        self.trace = Trace(self.world, self.clock, watch=self.world.objects)
        self.recorder = None
        if recorder_path is not None:
            from so101_sim.record import SimRecorder

            self.recorder = SimRecorder(
                self.world, recorder_path, renderer=self.renderer,
                user_source=self.mount.signals_pane, user_label="scripted user signals",
                title=title, load_threshold=self._load_threshold(),
            ).start()
        self._closers: list = []

    # -- hooks ----------------------------------------------------------------
    def _hand_obs_cls(self):
        raise NotImplementedError

    def _load_threshold(self) -> float:
        return 120.0

    # -- events ---------------------------------------------------------------
    def on_event(self, event: str, **data) -> None:
        self.events.append((self.clock(), event, data))
        self.board.on_event(event, **data)
        if self.recorder is not None:
            self.recorder.event(event, **data)
            if event == "behavior_start":
                self.recorder.label(f"behavior: {data.get('intent')}")
            elif event in ("behavior_end", "handled"):
                self.recorder.label("idle")

    def say(self, message: str) -> None:
        self.said.append(message)
        logger.info("[say] %s", message)
        self.on_event("say", text=message)

    def event_times(self, name: str, **match) -> list[float]:
        return [t for t, e, d in self.events if e == name and all(d.get(k) == v for k, v in match.items())]

    def said_any(self, phrase: str) -> bool:
        return any(phrase.lower() in m.lower() for m in self.said)

    # -- scene ------------------------------------------------------------------
    def stage(self, obj: str, pose: dict, stand: str = "stand_a"):
        p = self.world.stage_object(obj, {k: v for k, v in pose.items() if k != "gripper"}, stand=stand)
        self.on_event("stage", object=obj, at=[round(float(v), 3) for v in p])
        return p

    def wait_until(self, pred, timeout: float, poll: float = 0.02) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pred():
                return True
            time.sleep(poll)
        return False

    def joints(self) -> dict:
        return self.world.joint_cal()

    def near(self, pose: dict, tol: float = 3.0, joints=None) -> bool:
        cur = self.world.joint_cal()
        keys = joints or [k for k in pose if k != "gripper"]
        return all(abs(cur[k] - pose[k]) <= tol for k in keys)

    def when_near(self, pose: dict, action, tol: float = 3.0, delay: float = 0.0, timeout: float = 20.0):
        """Run ``action()`` on a thread once the arm reaches ``pose`` (+ ``delay``)."""

        def run():
            if self.wait_until(lambda: self.near(pose, tol), timeout):
                time.sleep(delay)
                action()

        th = threading.Thread(target=run, daemon=True)
        th.start()
        return th

    # -- lifecycle --------------------------------------------------------------
    def finish(self, status: str, message: str = "", keep: bool = True):
        self.trace.stop()
        path = None
        if self.recorder is not None:
            self.recorder.stamp(status, message)
            path = self.recorder.stop(keep=keep)
        for close in reversed(self._closers):
            try:
                close()
            except Exception as e:  # pragma: no cover
                logger.debug("rig close: %s", e)
        return path


def _llm_host_port(base: str) -> tuple[str, int]:
    host, port = base.rsplit("//", 1)[1].split(":")
    return host, int(port)


# ---------------------------------------------------------------------------
# pipeline (vla_pipeline)
# ---------------------------------------------------------------------------
class PipelineRig(_Rig):
    """``vla_pipeline``: SimArmClient + real Dispatcher/registry/LLM client."""

    target = "pipeline"

    def _hand_obs_cls(self):
        from vla_pipeline.vision.mediapipe_hands import HandObservation

        return HandObservation

    def _load_threshold(self) -> float:
        return float(self.cfg["behaviors"]["grip"].get("load_threshold", 120.0))

    def __init__(self, twin, cfg: dict, llm_base: str, recorder_path=None, title: str = "",
                 arm=None, arm_camera=None):
        super().__init__(twin, cfg, recorder_path, title)
        from so101_sim.camera import SimCamera
        from so101_sim.detect import SegDetector

        from vla_pipeline.llm.llama_intent import LlamaIntentParser
        from vla_pipeline.main import PipelineContext
        from vla_pipeline.robot.sim_arm import SimArmClient
        from vla_pipeline.vision.yolo_detect_npu import Detection, resolve_class

        host, port = _llm_host_port(llm_base)
        self.cfg["llm"].update(host=host, port=port, autostart=False)
        self.cfg["robot"]["backend"] = "sim"
        self.arm = arm if arm is not None else SimArmClient(self.cfg, world=self.world)
        # the detector always reads the twin's segmentation of the wrist camera
        self.seg_camera = SimCamera(self.world, "arm", renderer=self.renderer, segmentation=True)
        self._closers.append(self.seg_camera.release)
        self.arm_camera = arm_camera if arm_camera is not None else self.seg_camera
        self.detector = SimDetector(SegDetector(self.seg_camera), Detection, resolve_class)
        self.parser = LlamaIntentParser(self.cfg)
        self._closers.append(self.parser.close)
        self.ctx = PipelineContext(
            cfg=self.cfg, arm=self.arm, pose=self.pose, detector=self.detector, hands=self.hands,
            mount_camera=self.mount, arm_camera=self.arm_camera, headless=True,
        )
        self.ctx.say = self.say
        self.dispatcher = None

    def run(self, script, timeout: float = 30.0, settle: float = 0.6, startup_grace_s: float = 0.0,
            until=None) -> "PipelineRig":
        """Play ``script`` through the real Dispatcher; return once idle (or ``until()``)."""
        from vla_pipeline.dispatch import Dispatcher
        from vla_pipeline.main import is_stop, make_noise_gate, registry

        self.clock.start()
        self.trace.start()
        d = Dispatcher(
            self.ctx, self.parser.parse, registry, is_stop=is_stop,
            is_noise=make_noise_gate(self.cfg), startup_grace_s=startup_grace_s, clock=self.clock,
        )
        d.listeners.append(self.on_event)
        self.dispatcher = d
        shutdown = threading.Event()
        loop = threading.Thread(target=d.run, args=(shutdown,), name="dispatch", daemon=True)
        loop.start()
        tx = ScriptedTranscripts(self.clock, script, d.on_transcript).start()
        self.transcripts = tx

        idle_since = None
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"scenario did not finish within {timeout}s")
                if until is not None and until():
                    break
                idle = tx.done.is_set() and d.transcripts.empty() and not d.busy.is_set()
                if idle:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since >= settle:
                        break
                else:
                    idle_since = None
                time.sleep(0.02)
        finally:
            self.ctx.stop_event.set()
            shutdown.set()
            loop.join(15.0)
            self.ctx.stop_event.clear()
        return self


# ---------------------------------------------------------------------------
# workshop (workshop/<project>)
# ---------------------------------------------------------------------------
class WorkshopRig(_Rig):
    """Workshop: common.motion.SimArmClient + integration.handle_transcript."""

    target = "workshop"

    def _hand_obs_cls(self):
        from models.hands import HandObservation

        return HandObservation

    def __init__(self, twin, cfg: dict, llm_base: str, recorder_path=None, title: str = "", arm=None):
        super().__init__(twin, cfg, recorder_path, title)
        from common.motion import SimArmClient, mapper_from_config
        from integration.dispatch import WorkshopContext, build_registry
        from models.llm import LlamaIntentParser

        host, port = _llm_host_port(llm_base)
        self.cfg["llm"].update(host=host, port=port, autostart=False)
        self.cfg["robot"]["backend"] = "sim"
        self.arm = arm if arm is not None else SimArmClient(self.cfg, world=self.world)
        self.parser = LlamaIntentParser(self.cfg, autostart=False)
        self._closers.append(self.parser.close)
        self.ctx = WorkshopContext(
            cfg=self.cfg, arm=self.arm, pose=self.pose, hands=self.hands, camera=self.mount,
            mapper=mapper_from_config(self.cfg), registry=build_registry(), headless=True,
        )
        self.ctx.say = self.say

    def _deliver(self, text: str) -> None:
        from common.intents import Intent
        from integration.dispatch import handle_transcript

        self.on_event("heard", text=text)

        def parse(t):
            cmd = self.parser.parse(t)
            self.on_event("parsed", text=t, intent=cmd.intent.value, params=dict(cmd.params or {}))
            if cmd.intent not in (Intent.UNKNOWN, Intent.STOP) and self.ctx.registry.get(cmd.intent):
                self.on_event("behavior_start", intent=cmd.intent.value)
            return cmd

        try:
            cmd = handle_transcript(self.ctx, text, parse)
        except Exception as e:
            self.on_event("behavior_crash", error=repr(e))
            raise
        if cmd.intent is Intent.STOP:
            self.on_event("stop", text=text)
        self.on_event("handled", text=text, intent=cmd.intent.value)

    def run(self, script, timeout: float = 30.0, settle: float = 0.3) -> "WorkshopRig":
        self.clock.start()
        self.trace.start()
        tx = ScriptedTranscripts(self.clock, script, self._deliver, concurrent=True).start()
        self.transcripts = tx
        deadline = time.monotonic() + timeout
        try:
            while True:
                if time.monotonic() > deadline:
                    raise TimeoutError(f"scenario did not finish within {timeout}s")
                if tx.done.is_set() and not any(w.is_alive() for w in tx.workers):
                    break
                time.sleep(0.02)
            time.sleep(settle)
        finally:
            self.ctx.stop_event.set()
            tx.join(10.0)
            self.ctx.stop_event.clear()
        return self
