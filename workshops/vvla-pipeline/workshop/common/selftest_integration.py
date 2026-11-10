# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Component 3 self-test (PROVIDED) - run: ``python -m vvla_workshop.selftest_integration``

Exercises YOUR integration wiring (TODOs 3.1-3.4 in
``workshop/project/integration/``) end to end with synthetic backends - a
synthetic camera, a scripted "person" (pose) and hand source, and a dry-run
arm - so the whole compose-and-dispatch path runs with ZERO hardware: no NPU,
no ROS 2, no llama-server, no webcam. ``--reference`` runs the same checks on
the pipeline's finished functions.

- **3.1 build_registry** - the four behaviors are registered to the right
  handlers (and STOP deliberately is NOT - it's an interrupt, not a behavior).
- **3.2 handle_transcript** - the safety gauntlet in order: the stop fast
  path (sets the stop event, bypasses the parser), the noise gate (drops
  Whisper's silence hallucinations before the LLM), normal dispatch, the
  stop-event hygiene that keeps a stale stop from killing the next behavior,
  and a stop said while the LLM parses keeping that command from running.
- **3.3 mimic_tick** - one camera frame → pose + hands → mapper → a full,
  clamped arm command; it follows the NEAREST person (largest box) and
  drives the gripper from the pinch.
- **3.4 run_dance** - beat-paced choreography that stays stoppable and
  always parks at the rest pose.
"""

from __future__ import annotations


import numpy as np

from vla_pipeline.utils.config import load_config
from vvla_workshop.feedback import ARTIFACTS, Reporter, implementation, reference_flag

# COCO-17 keypoint indices used by the two-person fixture below.
R_ELBOW, R_WRIST = 8, 10


class _TwoPeoplePose:
    """Two people in frame: a FAR (small-box) one first, a NEAR (large-box)
    one second - so 'just take index 0' follows the wrong person on purpose."""

    provider = "TwoPeople(synthetic)"
    kpt_threshold = 0.3

    def infer(self, frame_bgr):
        """Return two people: index 0 far/right/small, index 1 near/left/large."""
        h, w = frame_bgr.shape[:2]
        kpts = np.zeros((2, 17, 3), dtype=np.float32)
        kpts[0, R_ELBOW] = (w * 0.80, h * 0.42, 0.9)
        kpts[0, R_WRIST] = (w * 0.80, h * 0.55, 0.95)
        kpts[1, R_ELBOW] = (w * 0.20, h * 0.42, 0.9)
        kpts[1, R_WRIST] = (w * 0.20, h * 0.55, 0.95)
        boxes = np.array(
            [[w * 0.70, h * 0.35, w * 0.90, h * 0.70],  # far  → small area
             [w * 0.05, h * 0.10, w * 0.45, h * 0.95]],  # near → large area
            dtype=np.float32,
        )
        scores = np.array([0.90, 0.92], dtype=np.float32)
        return boxes, scores, kpts


def _make_ctx(cfg, *, registry=None):
    """A fully synthetic context: scripted pose and hands, joints recorded in memory."""
    from vla_pipeline.dispatch import PipelineContext
    from vla_pipeline.robot.arm_interface import REST_POSE, clamp_joints
    from vla_pipeline.utils.gesture_map import mapper_from_config
    from vvla_workshop.fixtures import ScriptedHands, ScriptedPose, SyntheticCamera

    class _OfflineArm:
        """Records joint commands. The self-test freezes ``time.sleep`` and
        compares degree poses, so it does not start the MoveIt bringup."""

        def __init__(self):
            self._joints = dict(REST_POSE)
            self.gripper_joint = "gripper_joint"

        def send_joints(self, joints):
            self._joints.update(clamp_joints(joints))

        def send_gripper(self, position):
            self.send_joints({self.gripper_joint: float(position)})

        def get_joints(self):
            return dict(self._joints)

        def go_to(self, target, duration_s=2.0, stop_event=None, **_kwargs):
            if stop_event is not None and stop_event.is_set():
                return "stopped"
            self.send_joints(target)
            return "arrived"

        def go_to_rest(self):
            self.send_joints(REST_POSE)
            return "arrived"

    return PipelineContext(cfg=cfg, arm=_OfflineArm(), pose=ScriptedPose(), hands=ScriptedHands(),
                           mount_camera=SyntheticCamera(), mapper=mapper_from_config(cfg), registry=registry,
                           headless=True)


class _FixedPinch:
    """Scripted hands with the pinch held at one value."""

    def __init__(self, pinch: float):
        from vvla_workshop.fixtures import ScriptedHands

        self._inner, self._pinch = ScriptedHands(), float(pinch)

    def process(self, frame):
        obs, raw = self._inner.process(frame)
        if obs is not None:
            obs.pinch = self._pinch
        return obs, raw


def check_registry(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODO 3.1 - the behavior registry maps each intent to its handler."""
    from vla_pipeline.behaviors.simple import run_home, run_wave
    from vla_pipeline.llm.intents import Intent

    dispatch = implementation("3.1", reference)
    want = {
        Intent.GESTURE_MIMIC: implementation("3.3", reference).run_mimic,
        Intent.DANCE: implementation("3.4", reference).run_dance,
        Intent.WAVE: run_wave,
        Intent.HOME: run_home,
    }

    def build():
        """Verify build_registry wires the four behaviors (and never STOP)."""
        reg = dispatch.build_registry()
        for intent, fn in want.items():
            got = reg.get(intent)
            assert got is not None, (
                f"{intent.value} has no handler - map it with registry.add(Intent.{intent.name}, {fn.__name__})")
            assert got is fn, (
                f"{intent.value} → {getattr(got, '__name__', got)!r}, expected {fn.__name__} - each intent points "
                "at its own behavior")
        assert reg.get(Intent.STOP) is None, (
            "STOP has a handler - stop is an INTERRUPT handled before dispatch, never a registered behavior")

    rep.run("3.1", "build_registry wires the four behaviors (and not STOP)", build,
            hint="registry.add(Intent.GESTURE_MIMIC, run_mimic) and the same for DANCE/WAVE/HOME")


def check_dispatch(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODO 3.2 - the transcript gauntlet: stop, noise, parse, dispatch."""
    from vla_pipeline.llm.intents import CommandRegistry, Intent, keyword_parse

    handle_transcript = implementation("3.2", reference).handle_transcript
    # A stub registry so we test the GAUNTLET, not the real behaviors: each
    # handler just records that it ran and whether the stop event was clear
    # at the time (a stale stop must never pre-empt the next behavior).
    dispatched: list = []

    def make_handler(name):
        """Build a recording stand-in handler for ``name``."""

        def handler(ctx, cmd):
            """Record the dispatch and the stop-event state when it fired."""
            dispatched.append((name, ctx.stop_check()))

        return handler

    reg = CommandRegistry()
    for intent, name in ((Intent.GESTURE_MIMIC, "mimic"), (Intent.DANCE, "dance"), (Intent.WAVE, "wave"),
                         (Intent.HOME, "home")):
        reg.add(intent, make_handler(name))
    ctx = _make_ctx(cfg, registry=reg)
    parsed: list = []

    def parse(text: str):
        """Keyword stand-in for the LLM parser that records every call."""
        parsed.append(text)
        return keyword_parse(text)

    def fresh():
        dispatched.clear()
        parsed.clear()
        ctx.stop_event.clear()

    def stop_fast_path():
        """Verify a stop transcript sets the event and bypasses the parser."""
        fresh()
        cmd = handle_transcript(ctx, "whoa STOP right now", parse)
        assert cmd.intent is Intent.STOP, f"stop transcript returned {cmd.intent.value}, expected stop"
        assert ctx.stop_event.is_set(), "a stop transcript must SET ctx.stop_event so a running behavior is interrupted"
        assert not parsed, ("the parser was invoked on a STOP - the safety stop must bypass the model "
                            "(keyword match on the RAW text first)")
        assert not dispatched, "a stop must not dispatch a behavior handler"

    rep.run("3.2", "stop fast path sets the event and skips the parser", stop_fast_path,
            hint="is_stop(text) FIRST → ctx.request_stop(); return STOP")

    def noise_gate():
        """Verify a silence-hallucination transcript is dropped before the LLM."""
        fresh()
        cmd = handle_transcript(ctx, "Thank you.", parse)
        assert cmd.intent is Intent.UNKNOWN, f"noise returned {cmd.intent.value}, expected unknown (dropped)"
        assert not parsed, "noise must be dropped BEFORE the parser (is_noise)"
        assert not dispatched, "noise must not dispatch a behavior"

    rep.run("3.2", "noise gate drops silence-hallucinations before the LLM", noise_gate,
            hint="is_noise(text) after the stop check → return UNKNOWN, do nothing")

    def dispatch_and_clear():
        """Verify a normal command dispatches with a freshly cleared stop event."""
        fresh()
        ctx.stop_event.set()  # a STALE stop from a prior behavior
        cmd = handle_transcript(ctx, "show me a dance", parse)
        assert parsed == ["show me a dance"], "the parser should see a normal transcript"
        assert cmd.intent is Intent.DANCE, f"routed to {cmd.intent.value}, expected dance"
        assert dispatched == [("dance", False)], (
            f"dispatch record {dispatched} - the DANCE handler must RUN, and the stale stop must be CLEARED "
            "before it (so it isn't pre-empted)")
        assert not ctx.stop_event.is_set(), (
            "stop event must be clear AFTER a behavior too - a leftover stop would kill the NEXT behavior")

    rep.run("3.2", "normal command dispatches with a clean stop event", dispatch_and_clear,
            hint="ctx.begin(heard) before the handler (it clears a stale stop), ctx.stop_event.clear() after it")

    def stop_during_parse():
        """Verify a stop said while the LLM parses keeps the parsed command from running."""
        fresh()

        def parse_then_hear_stop(text: str):
            """The LLM parses; meanwhile the always-hot listener hears 'stop'."""
            cmd = parse(text)
            handle_transcript(ctx, "stop", parse)
            return cmd

        handle_transcript(ctx, "show me a dance", parse_then_hear_stop)
        assert not dispatched, (
            f"dispatch record {dispatched} - a stop said while the LLM parsed must keep that command from "
            "running: read ctx.stops_heard() before parse(), and run the handler only if ctx.begin(heard) "
            "returns True")
        assert ctx.stop_event.is_set(), "the stop must stay in effect"

    rep.run("3.2", "a stop said while the LLM parses cancels that command", stop_during_parse,
            hint="heard = ctx.stops_heard(); cmd = parse(text); ...; if not ctx.begin(heard): return cmd")

    def unknown_reported():
        """Verify an unroutable transcript returns UNKNOWN and dispatches nothing."""
        fresh()
        cmd = handle_transcript(ctx, "the weather is nice today", parse)
        assert cmd.intent is Intent.UNKNOWN, f"unroutable transcript returned {cmd.intent.value}, expected unknown"
        assert not dispatched, "an unknown command must not dispatch a behavior"

    rep.run("3.2", "unrecognized command → UNKNOWN, dispatches nothing", unknown_reported)


def check_mimic(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODO 3.3 - one frame becomes one arm command (nearest person, pinch)."""
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, MOTOR_NAMES
    from vla_pipeline.vision.yolo_pose_npu import draw_skeleton

    mimic_tick = implementation("3.3", reference).mimic_tick
    ctx = _make_ctx(cfg)
    _, frame = ctx.mount_camera.read()

    def full_command():
        """Verify mimic_tick turns a frame into a full, clamped, sent arm command."""
        sig = mimic_tick(ctx, frame)
        assert sig is not None and hasattr(sig, "joints"), "mimic_tick must return a MimicSignals"
        assert sig.wrist_xy is not None, (
            "the scripted person has a visible wrist but wrist_xy is None - pick the nearest person's arm "
            "with pick_nearest + pick_arm")
        assert isinstance(sig.joints, dict), (
            "no joints commanded - with a wrist AND a hand in frame the mapper must produce a joint dict")
        for m in MOTOR_NAMES:
            assert m in sig.joints, f"commanded joints missing '{m}'"
        commanded = ctx.arm.get_joints()
        from vla_pipeline.robot.arm_interface import radians_from_calibrated

        expected = radians_from_calibrated(ctx.cfg, sig.joints)
        for name, value in expected.items():
            assert abs(commanded[name] - value) < 1e-6, (
                "the joints you returned were not the joints you sent to the arm - "
                "convert them with radians_from_calibrated and call ctx.arm.send_joints")
        import cv2

        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS / "mimic_tick.png"
        cv2.imwrite(str(path), draw_skeleton(frame, sig.boxes, sig.scores, sig.kpts, ctx.pose.kpt_threshold))
        rep.artifact(path)

    ok_tick, _ = rep.run("3.3", "mimic_tick: frame → pose+hands → full arm command", full_command,
                         hint="infer pose → nearest person's arm → read the hand → mapper.step(...) → "
                         "ctx.arm.send_joints(joints)")

    def follows_nearest():
        """Verify mimic_tick follows the NEAREST (largest-box) of two people."""
        near_ctx = _make_ctx(cfg)
        near_ctx.pose = _TwoPeoplePose()
        sig = mimic_tick(near_ctx, frame)
        assert sig.wrist_xy is not None, "no wrist chosen with two people in frame"
        assert sig.wrist_xy[0] < frame.shape[1] * 0.5, (
            f"wrist_xy={tuple(round(v, 1) for v in sig.wrist_xy)} is on the far (small-box) person - select "
            "the LARGEST box (nearest person) with pick_nearest, not just index 0")

    def pinch_drives_gripper():
        """Verify the pinch reaches the gripper: fingers together close the jaws."""
        grips = {}
        for label, pinch in (("together", 0.05), ("apart", 1.4)):
            pinch_ctx = _make_ctx(cfg)
            pinch_ctx.hands = _FixedPinch(pinch)
            sig = mimic_tick(pinch_ctx, frame)
            assert sig.joints is not None, "pinch: a command should be sent"
            grips[label] = sig.joints["gripper"]
        assert grips["together"] <= GRIPPER_CLOSED_FLOOR + 1.0 < grips["apart"] - 30.0, (
            f"gripper {grips['together']:.1f} with the fingers together and {grips['apart']:.1f} with them "
            "apart - pass obs.pinch to ctx.mapper.step(pinch=...)")

    for name, check in (("mimic_tick follows the NEAREST (largest-box) person", follows_nearest),
                        ("mimic_tick drives the gripper from the pinch", pinch_drives_gripper)):
        if ok_tick:
            rep.run("3.3", name, check)
        else:
            rep.add("3.3", name, "SKIP", "needs mimic_tick first")


def check_dance(rep: Reporter, cfg: dict, reference: bool) -> None:
    """TODO 3.4 - the dance loop is beat-paced, stoppable, and parks at rest."""
    from unittest import mock

    from vla_pipeline.robot.arm_interface import REST_POSE

    run_dance = implementation("3.4", reference).run_dance

    def dances_and_parks():
        """Verify run_dance runs to its deadline and parks at the rest pose."""
        ctx = _make_ctx(cfg)
        with mock.patch("time.sleep", lambda *a, **k: None):
            run_dance(ctx, duration_s=0.05, seed=0)
        joints = ctx.arm.get_joints()
        for k, v in REST_POSE.items():
            assert abs(joints[k] - v) < 1.0, (
                f"after the dance {k}={joints[k]:.1f}, expected rest {v:.1f} - always park with "
                "ctx.arm.go_to_rest() in a finally")

    rep.run("3.4", "run_dance runs a beat clock and parks at rest", dances_and_parks,
            hint="loop until the deadline, go_to each keyframe, finally go_to_rest()")

    def stays_stoppable():
        """Verify a stop asserted up front halts the dance before it moves."""
        ctx = _make_ctx(cfg)
        ctx.stop_event.set()  # stop asserted before it starts
        max_dev = {"v": 0.0}
        real_send = ctx.arm.send_joints

        def watched_send(joints):
            """Send joints and remember the largest deviation from rest seen."""
            real_send(joints)
            cur = ctx.arm.get_joints()
            max_dev["v"] = max(max_dev["v"], max(abs(cur[k] - REST_POSE[k]) for k in REST_POSE))

        ctx.arm.send_joints = watched_send
        with mock.patch("time.sleep", lambda *a, **k: None):
            run_dance(ctx, duration_s=0.3, seed=0)  # long clock; the stop must win
        assert max_dev["v"] < 2.0, (
            f"the arm moved {max_dev['v']:.0f}° into a dance keyframe despite a stop - poll ctx.stop_check() "
            "in the loop (a dance you can't stop is a safety bug, not a feature)")

    rep.run("3.4", "run_dance stays stoppable (a set stop halts it immediately)", stays_stoppable,
            hint="check ctx.stop_check() between keyframes; still park in a finally")


def run(reference: bool = False) -> int:
    """Run every component 3 check on your functions (or the reference); return the exit code."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    rep = Reporter("Component 3 - integration: composing models into behavior" + (" (reference)" if reference else ""))
    print("  backends: synthetic camera + scripted pose/hands + recorded joints (no NPU / ROS 2 / llama-server needed)")
    check_registry(rep, cfg, reference)
    check_dispatch(rep, cfg, reference)
    check_mimic(rep, cfg, reference)
    check_dance(rep, cfg, reference)
    return rep.summary()


if __name__ == "__main__":
    raise SystemExit(run(reference_flag("Component 3 self-test")))
