# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Gesture-mimic behavior: the robot follows the human hand around the camera
frame, via **direct camera-frame motion mapping** (see
:mod:`vla_pipeline.utils.gesture_map`).

One tick of mimic is the whole pipeline in miniature::

    camera frame ──► YOLO-pose (NPU) ──► wrist x/y + forearm ─┐
                └──► MediaPipe (CPU) ──► size/roll/pinch ─────┴──► mapper ──► arm

- **YOLOv26s-pose (NPU)** → the nearest person's right wrist position in the
  frame: horizontal → left/right (shoulder_pan), vertical → up/down
  (shoulder_lift and elbow_flex as a locked pair). The right elbow gives the
  forearm direction for wrist_flex. The YOLO wrist is also the **fallback
  controller**: when MediaPipe loses the hand the robot keeps following the
  wrist for x/y until the hand returns.
- **MediaPipe Hands (CPU)** →
    * hand **size** (how near the hand is) → forward/back,
    * wrist→knuckle direction measured against the forearm (the wrist bend)
      → wrist_flex,
    * thumb↔index **line** (with the knuckle-line reference, then the
      wrist→knuckle angle, as reliability-weighted fallbacks) → wrist_roll,
    * thumb↔index **distance** → gripper, as a continuous ramp.

:func:`mimic_tick` is one tick (the workshop's TODO 3.3); :func:`run_mimic`
is the behavior around it: the loop, the preview and the HUD that attributes
each joint to the TODO that feeds it. The pinch always drives the gripper: a
grip locked by *grip* or *pick_place* is released when mimic starts, so a
held object drops when the fingers open.

Depth neutral (optional, one-time per camera position):
    python -m vla_pipeline.behaviors.gesture_mimic --calibrate-depth
Hold your hand at a comfortable mid-distance; paste the printed
``hand_size_neutral`` into ``mimic`` in ``config/pipeline.yaml``.
With it left at 0 the neutral auto-calibrates from the first few frames.

Gripper distance (recommended, one-time per operator):
    python -m vla_pipeline.behaviors.gesture_mimic --calibrate-grip
Pinch your thumb & index together, then spread them wide when prompted; paste
the printed ``grip_min`` / ``grip_max`` into ``mimic`` in ``config/pipeline.yaml``.

Independent test
----------------
    python -m vla_pipeline.behaviors.gesture_mimic --dry-run     # no robot
Press 'q' in the preview window (or Ctrl-C) to stop.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from vla_pipeline.user_robot_control import PipelineContext
from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, make_arm
from vla_pipeline.utils.config import load_config
from vla_pipeline.utils.gesture_map import mapper_from_config
from vla_pipeline.vision.camera import wait_for_frames
from vla_pipeline.vision.mediapipe_hands import HandTracker
from vla_pipeline.vision.yolo_pose_npu import PoseEstimator, draw_skeleton, pick_arm, pick_nearest

logger = logging.getLogger(__name__)

WINDOW = "gesture mimic"


@dataclass
class MimicSignals:
    """One tick's worth of evidence for the HUD."""

    joints: dict = None  # what was commanded (None = nothing sent)
    wrist_xy: tuple = None  # pose model's wrist pick
    elbow_xy: tuple = None  # same arm's elbow (forearm → wrist_flex)
    boxes: object = None  # raw pose outputs, for the skeleton overlay
    scores: object = None
    kpts: object = None
    obs: object = None  # HandObservation (or None)
    raw: object = None  # raw MediaPipe results, for the hand overlay


def _run_mimic_from_observations(ctx) -> MimicSignals:
    """Follow the perception process until a stop, or until the camera goes stale."""
    from vla_pipeline.behaviors.manipulation import release_to_pinch

    stale = float(((ctx.cfg.get("cameras") or {}).get("mount") or {}).get("stale_after_s", 1.0))
    threshold = float((ctx.cfg.get("yolo_pose") or {}).get("kpt_threshold", 0.3))
    if release_to_pinch(ctx.shared_state):
        ctx.say("Your pinch drives the gripper now. Open your fingers to let go.")
    last_seq = 0
    frames = 0
    missing = 0
    last = MimicSignals()
    try:
        while not ctx.stop_check():
            observation = ctx.mailbox.wait_new("mount", last_seq, stale, ctx.stop_event)
            if observation is None:
                break
            last_seq = observation.seq
            last = _mimic_from_observation(ctx, observation, threshold)
            frames += 1
            if not observation.kpts:
                missing += 1
                if missing == 5:
                    ctx.mapper.reset()
            else:
                missing = 0
            if ctx.max_frames and frames >= ctx.max_frames:
                break
    finally:
        logger.info("Mimic stopped after %d frames - parking the arm.", frames)
        park_after_mimic(ctx)
    return last


def park_after_mimic(ctx) -> None:
    """Park at rest after mimic, without consulting the stop event.

    The stream ends first. Its last setpoint can be further than the arm
    travels before the park gives up waiting for it, and the park replaces it.
    """
    ctx.arm.stop()
    ctx.arm.go_to_rest()


def mimic_tick(ctx, frame) -> MimicSignals:
    """Turn ONE camera frame into ONE arm command.

    The composition, in order:

    1. ``ctx.pose.infer(frame)`` → boxes/scores/kpts; keep only the NEAREST
       person (largest box, :func:`pick_nearest`), then pick a wrist AND its
       elbow with :func:`pick_arm` (threshold: ``ctx.pose.kpt_threshold``).
    2. ``ctx.hands.process(frame)`` → a HandObservation (or None) carrying
       hand_size / roll_ti / roll_ref / roll_ti_weight / roll / pinch.
    3. ``ctx.mapper.step(...)`` with everything from 1-2 (the elbow as
       ``elbow_xy=``: wrist_flex measures the hand's direction against it;
       without it the forearm is assumed vertical) → a clamped joint dict,
       then ``ctx.arm.send_joints(joints)``. If NEITHER a wrist NOR a hand
       was seen, send nothing (hold pose).

    Returns a :class:`MimicSignals` with everything used, for the HUD,
    including the RAW MediaPipe results for the hand overlay.
    """
    boxes, scores, kpts = ctx.pose.infer(frame)
    wrist_xy = elbow_xy = None
    if len(boxes) > 0:
        wrist_xy, elbow_xy = pick_arm(kpts[pick_nearest(boxes)], ctx.pose.kpt_threshold)

    obs, raw = ctx.hands.process(frame) if ctx.hands else (None, None)
    height, width = frame.shape[:2]
    joints = apply_mimic(ctx, wrist_xy, elbow_xy, obs, width, height)
    return MimicSignals(joints=joints, wrist_xy=wrist_xy, elbow_xy=elbow_xy, boxes=boxes,
                        scores=scores, kpts=kpts, obs=obs, raw=raw)


def apply_mimic(ctx, wrist_xy, elbow_xy, hand, width: int, height: int) -> dict | None:
    """One mapper step and one arm command. None when neither a wrist nor a hand was seen."""
    if wrist_xy is None and hand is None:
        return None
    joints = ctx.mapper.step(
        wrist_xy,
        width,
        height,
        elbow_xy=elbow_xy,
        hand_size=hand.hand_size if hand else None,
        roll_ti=hand.roll_ti if hand else None,
        roll_ref=hand.roll_ref if hand else None,
        roll_ti_weight=hand.roll_ti_weight if hand else 0.0,
        roll_fallback=hand.roll if hand else None,
        pinch=hand.pinch if hand else None,
    )
    from vla_pipeline.robot.arm_interface import radians_from_calibrated

    command = radians_from_calibrated(ctx.cfg, joints)
    ctx.arm.send_joints(command)
    jaw = command.get(ctx.arm.gripper_joint)
    if jaw is not None:
        ctx.arm.send_gripper(jaw)
    return joints


def _mimic_from_observation(ctx, observation, threshold: float) -> MimicSignals:
    """One observation already measured by the perception process."""
    import numpy as np

    from vla_pipeline.vision.yolo_pose_npu import pick_arm

    wrist_xy = elbow_xy = None
    kpts = None
    if observation.kpts:
        kpts = np.asarray(observation.kpts, dtype=float)
        wrist_xy, elbow_xy = pick_arm(kpts, threshold)
    joints = apply_mimic(
        ctx, wrist_xy, elbow_xy, observation.hand, observation.width, observation.height,
    )
    return MimicSignals(joints=joints, wrist_xy=wrist_xy, elbow_xy=elbow_xy, kpts=kpts, obs=observation.hand)


# =============================================================================
# The loop, the preview, and the HUD around the tick
# =============================================================================

HUD_ATTRIBUTION = [
    "wrist x  -> shoulder_pan            [TODO 1.2 session -> TODO 3.3 tick]",
    "wrist y  -> lift+elbow (locked)     [TODO 1.2 session -> TODO 3.3 tick]",
    "wrist bend -> wrist_flex            [TODO 1.2+1.4     -> TODO 3.3 tick]",
    "hand size-> fwd reach (lift+elbow)  [TODO 1.4 hands   -> TODO 3.3 tick]",
    "thumb-index angle -> wrist_roll     [TODO 1.4 hands   -> TODO 3.3 tick]",
    "pinch    -> gripper                 [TODO 1.4 hands   -> TODO 3.3 tick]",
]


def annotate_lines(frame, lines, origin=(12, 30), color=(0, 255, 255), scale=0.55):
    """Stack text lines onto a frame (in place); returns the frame."""
    x, y = origin
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)
        y += int(28 * scale / 0.55)
    return frame


_L1, _L2, _L3 = 70.0, 62.0, 34.0  # link lengths in panel pixels


def draw_arm_panel(joints, width: int = 300, height: int = 480):
    """Render a schematic SO-101 from a joint dict (degrees).

    Top half: side view of the shoulder_lift → elbow_flex → wrist_flex chain.
    Bottom half: top-down shoulder_pan needle + gripper jaw opening.
    Pure visualization - the real kinematics live on the robot.
    """
    panel = np.full((height, width, 3), 24, dtype=np.uint8)
    j = {k: float(joints.get(k, 0.0))
         for k in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")}
    label = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(160, 160, 160), thickness=1)

    # side view; shoulder_lift 0 = upright link, positive leans forward (screen right)
    base = (width // 2, int(height * 0.42))
    cv2.putText(panel, "side view", (12, 22), **label)
    a1 = np.radians(j["shoulder_lift"] + 90.0)  # -90 (rest) → straight up
    p1 = (int(base[0] + _L1 * np.cos(a1)), int(base[1] - _L1 * np.sin(a1)))
    a2 = a1 - np.radians(j["elbow_flex"] - 90.0)
    p2 = (int(p1[0] + _L2 * np.cos(a2)), int(p1[1] - _L2 * np.sin(a2)))
    a3 = a2 - np.radians(j["wrist_flex"])
    p3 = (int(p2[0] + _L3 * np.cos(a3)), int(p2[1] - _L3 * np.sin(a3)))
    cv2.rectangle(panel, (base[0] - 26, base[1]), (base[0] + 26, base[1] + 14), (90, 90, 90), -1)
    for a, b, c in ((base, p1, (0, 200, 255)), (p1, p2, (0, 255, 160)), (p2, p3, (255, 200, 0))):
        cv2.line(panel, a, b, c, 5)
        cv2.circle(panel, b, 5, (255, 255, 255), -1)

    # top view: shoulder_pan needle
    c2 = (width // 2, int(height * 0.68))
    cv2.putText(panel, "top view (pan)", (12, c2[1] - 46), **label)
    cv2.circle(panel, c2, 40, (90, 90, 90), 2)
    pan = np.radians(j["shoulder_pan"])
    cv2.line(panel, c2, (int(c2[0] + 40 * np.sin(pan)), int(c2[1] - 40 * np.cos(pan))), (0, 200, 255), 4)

    # gripper jaws
    gy = int(height * 0.86)
    cv2.putText(panel, "gripper", (12, gy - 26), **label)
    half = 6 + int(44 * np.clip(j["gripper"] / 100.0, 0.0, 1.0))
    for sign in (-1, 1):
        x = width // 2 + sign * half
        cv2.line(panel, (x, gy - 18), (x, gy + 18), (0, 160, 255), 6)

    # numbers
    y = height - 74
    for k in j:
        cv2.putText(panel, f"{k:>13s} {j[k]:+7.1f}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        y += 18
    return panel


def _grip_label(grip_deg: float) -> str:
    """Describe the continuous gripper opening for the HUD."""
    if grip_deg <= GRIPPER_CLOSED_FLOOR + 2.0:
        return "closed"
    if grip_deg >= 85.0:
        return "open"
    return f"{grip_deg:.0f}%"


def _preview(ctx, frame, signals: MimicSignals, provider: str):
    """The camera frame with the pose and hand overlays, the HUD, and the schematic arm beside it."""
    vis = frame.copy()
    if signals.kpts is not None and len(signals.kpts) > 0:
        vis = draw_skeleton(frame, signals.boxes, signals.scores, signals.kpts, ctx.pose.kpt_threshold)
    if ctx.hands is not None:
        vis = ctx.hands.draw(vis, signals.obs, signals.raw)
    if signals.wrist_xy is not None:
        wrist = (int(signals.wrist_xy[0]), int(signals.wrist_xy[1]))
        cv2.circle(vis, wrist, 9, (0, 255, 0), 2)
        if signals.elbow_xy is not None:  # the forearm wrist_flex follows
            elbow = (int(signals.elbow_xy[0]), int(signals.elbow_xy[1]))
            cv2.line(vis, elbow, wrist, (0, 255, 255), 2)
            cv2.circle(vis, elbow, 6, (0, 255, 255), 2)
    grip = (signals.joints or {}).get("gripper")
    status = f"gesture mimic - pose on {provider}"
    status += "" if grip is None else f"  grip {grip:3.0f} ({_grip_label(grip)})"
    annotate_lines(vis, [status] + HUD_ATTRIBUTION, scale=0.5)
    panel = draw_arm_panel({**ctx.arm.get_joints(), **(signals.joints or {})}, width=300, height=vis.shape[0])
    return np.hstack([vis, panel])


def run_mimic(ctx, cmd=None, tick=mimic_tick) -> MimicSignals | bool:
    """Mimic until 'q', a stop, or ``ctx.max_frames``; then park.

    ``tick`` turns one frame into one arm command (default :func:`mimic_tick`;
    a lesson passes its own).
    """
    from vla_pipeline.behaviors.manipulation import release_to_pinch

    if ctx.mailbox is None and (ctx.pose is None or ctx.mount_camera is None):
        ctx.say("Gesture mimic needs the mount camera and pose model.")
        return False
    if ctx.mapper is None:
        ctx.mapper = mapper_from_config(ctx.cfg)
    if ctx.mailbox is not None:
        return _run_mimic_from_observations(ctx)
    provider = getattr(ctx.pose, "provider", "?")
    if release_to_pinch(ctx.shared_state):
        ctx.say("Your pinch drives the gripper now. Open your fingers to let go.")
    logger.info("Mimic running (pose on %s) - move your hand; 'q' quits.", provider)
    frames = 0
    missing = 0  # consecutive frames with no person (→ reset to avoid a jump)
    last = MimicSignals()
    try:
        while not ctx.stop_check():
            ok, frame = ctx.mount_camera.read()
            if not ok:
                # No frame: wait on the camera's own event (its next frame, or its reopen
                # time, which a stop cuts short), not on a sleep.
                wait_for_frames(ctx.mount_camera, ctx.stop_event)
                continue
            last = tick(ctx, frame)
            frames += 1
            # Person gone a while → drop stale smoothing/roll-center state so
            # the arm doesn't lurch when they reappear.
            if last.kpts is None or len(last.kpts) == 0:
                missing += 1
                if missing == 5:
                    ctx.mapper.reset()
            else:
                missing = 0
            if not ctx.headless or ctx.max_frames:
                canvas = _preview(ctx, frame, last, provider)
                if ctx.headless:
                    ctx.shared_state["last_ui_frame"] = canvas
                else:
                    cv2.imshow(WINDOW, canvas)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if ctx.max_frames and frames >= ctx.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if not ctx.headless:
            try:
                cv2.destroyWindow(WINDOW)
            except cv2.error:
                pass
        logger.info("Mimic stopped after %d frames - parking the arm.", frames)
        # Park home as one goal, deliberately without stop_check: on a voice
        # "stop" the stop event is already set, and honoring it would abort
        # this park and leave the arm wherever it was.
        park_after_mimic(ctx)
    return last


# =============================================================================
# Depth-neutral calibration mode
# =============================================================================


def calibrate_depth(
    cfg: dict, hands: HandTracker, camera, sample_s: float = 2.5
) -> None:
    """Measure the neutral hand size at a comfortable mid-distance and print the
    ``hand_size_neutral`` value to paste into ``config/pipeline.yaml``."""
    print(
        "\nHold your open hand toward the mount camera at a comfortable "
        "mid-distance (the pose you want to be 'neither forward nor back')."
    )
    input("Press Enter when in position... ")
    samples: list[float] = []
    never = threading.Event()  # this calibration has no stop; the sampling time ends it
    deadline = time.time() + sample_s
    while time.time() < deadline:
        ok, frame = camera.read()
        if not ok:
            wait_for_frames(camera, never, bound_s=max(deadline - time.time(), 0.0))
            continue
        obs, _ = hands.process(frame)
        if obs is not None and obs.hand_size > 0:
            samples.append(obs.hand_size)
    if len(samples) < 5:
        raise SystemExit("Calibration failed - hand not visible long enough.")
    neutral = float(np.median(samples))
    print(
        f"\nMeasured neutral hand size: {neutral:.4f} "
        f"(n={len(samples)}, std {np.std(samples):.4f})"
    )
    print("Paste into config/pipeline.yaml under mimic:\n")
    print(f"    hand_size_neutral: {neutral:.4f}")


def calibrate_grip(
    cfg: dict, hands: HandTracker, camera, sample_s: float = 2.5
) -> None:
    """Measure the thumb↔index ``pinch`` at fully-closed and fully-open and print
    the ``grip_min`` / ``grip_max`` values to paste into ``config/pipeline.yaml``.

    These bound the normalized pinch the continuous gripper ramp uses, so a closed
    pinch maps to fully closed and a wide spread to fully open.
    """

    def _sample(prompt: str, reducer) -> float:
        """Prompt, sample obs.pinch for sample_s seconds, and reduce the readings to one value."""
        print(prompt)
        input("Press Enter when in position... ")
        vals: list[float] = []
        never = threading.Event()  # this calibration has no stop; the sampling time ends it
        deadline = time.time() + sample_s
        while time.time() < deadline:
            ok, frame = camera.read()
            if not ok:
                wait_for_frames(camera, never, bound_s=max(deadline - time.time(), 0.0))
                continue
            obs, _ = hands.process(frame)
            if obs is not None and obs.pinch >= 0:
                vals.append(obs.pinch)
        if len(vals) < 5:
            raise SystemExit("Calibration failed - hand not visible long enough.")
        return float(reducer(vals))

    print("\nGripper distance calibration - thumb↔index distance min/max.")
    grip_min = _sample(
        "\n1/2: Pinch your thumb and index finger together (gripper CLOSED).", np.median
    )
    grip_max = _sample(
        "\n2/2: Spread your thumb and index finger as wide apart as is "
        "comfortable (gripper WIDE OPEN).",
        np.median,
    )
    if grip_max <= grip_min:
        raise SystemExit(
            f"Calibration failed - max ({grip_max:.3f}) not greater "
            f"than min ({grip_min:.3f}). Re-run and spread wider."
        )
    print(f"\nMeasured grip_min={grip_min:.4f}, grip_max={grip_max:.4f}")
    print("Paste into config/pipeline.yaml under mimic:\n")
    print(f"    grip_min: {grip_min:.4f}")
    print(f"    grip_max: {grip_max:.4f}")


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run gesture mimic, or one of the calibration modes."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Gesture mimic behavior - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true", help="no robot hardware")
    parser.add_argument(
        "--calibrate-depth",
        action="store_true",
        help="measure hand_size_neutral at a comfortable distance and exit",
    )
    parser.add_argument(
        "--calibrate-grip",
        action="store_true",
        help="measure thumb↔index grip_min/grip_max and exit",
    )
    parser.add_argument(
        "--device", choices=["npu", "cpu"], default=None, help="YOLO device override"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    from vla_pipeline.vision.camera import make_camera

    hands = HandTracker(cfg)
    camera = make_camera(cfg, "mount")
    if args.calibrate_depth or args.calibrate_grip:
        try:
            if args.calibrate_depth:
                calibrate_depth(cfg, hands, camera)
            if args.calibrate_grip:
                calibrate_grip(cfg, hands, camera)
        finally:
            camera.close()
            hands.close()
        return

    pose = PoseEstimator(cfg, device=args.device)
    arm = make_arm(cfg)
    ctx = PipelineContext(cfg=cfg, arm=arm, pose=pose, hands=hands, mount_camera=camera,
                          mapper=mapper_from_config(cfg))
    try:
        run_mimic(ctx)
    finally:
        camera.close()
        hands.close()
        arm.close()


if __name__ == "__main__":
    main()
