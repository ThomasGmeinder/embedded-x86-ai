# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Manipulation behaviors: grip, pick_place, fetch_block (PROVIDED).

The pipeline's remaining voice commands, in the workshop's handler shape
``(ctx, cmd)`` so the registry can dispatch them like dance or wave:

- **grip** - present the open jaws, close until contact (gripper current, or
  a position stall), confirm with the arm camera, lock the grip. A following
  gesture_mimic keeps the lock; "grip" again releases.
- **pick_place** - ground the spoken object name to a COCO class, find it with
  YOLO detection on the arm camera, center it with the pan joint, grasp, then
  hand it over.
- **fetch_block** - fetch the block from its taught pose, then hand it over.

The handover finds you with YOLO-pose on the operator camera, aims the pan
joint at you, extends, opens the jaws and parks. The same sequence the
pipeline runs (``vla_pipeline/behaviors/``); the workshop is standalone, so
it lives here rather than being imported. Settings: ``grip``, ``pick_place``
and ``fetch_block`` in ``config/workshop.yaml``.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from common.imaging import KP
from common.intents import Intent
from common.motion import GRIPPER_CLOSED_FLOOR

logger = logging.getLogger(__name__)

# Pose presenting the gripper toward the user, jaws open.
PRESENT_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.0,
    "elbow_flex": 30.0,
    "wrist_flex": 10.0,
    "wrist_roll": 0.0,
    "gripper": 80.0,
}

TICK_S = 0.05


# ----------------------------------------------------------------------------
# Grasping: contact from current or position stall, confirmed by the camera
# ----------------------------------------------------------------------------


def _jaw_roi(frame: np.ndarray, roi) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi
    return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def object_in_jaws(reference: np.ndarray, frame: np.ndarray, roi, threshold: float) -> bool:
    """True if the jaw region differs enough from the empty-jaws reference."""
    a = cv2.cvtColor(_jaw_roi(reference, roi), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(_jaw_roi(frame, roi), cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape or a.size == 0:
        return False
    diff = float(np.mean(cv2.absdiff(cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0))))
    logger.debug("[grip] jaw ROI diff %.3f (threshold %.3f)", diff / 255.0, threshold)
    return diff / 255.0 >= threshold


def close_with_feedback(arm, g: dict, stop_check=None, start_from: float | None = None):
    """Close the gripper until contact. Returns ``(gripped, hold_position)``.

    Contact is sustained current above ``load_threshold`` (when the arm reports
    it) or the measured position lagging the command (a stall). After the
    command reaches the closed floor, sampling continues for ``stall_grace_s``
    so a thin object (a pen) is still caught.
    """
    step = float(g["close_step_deg"])
    eps = float(g["stall_eps_deg"])
    need_ticks = int(g["stall_ticks"])
    extra = float(g["settle_extra_deg"])
    load_threshold = float(g.get("load_threshold", 120.0))
    load_ticks_needed = int(g.get("load_ticks", 3))
    grace_s = float(g.get("stall_grace_s", 0.6))
    floor = GRIPPER_CLOSED_FLOOR  # the arm clamps below this; a lower command would read as a stall

    use_load = arm.get_gripper_load() is not None
    logger.info("[grip] closing with %s feedback", "current" if use_load else "position-stall")

    commanded = (
        start_from if start_from is not None else float(arm.get_joints().get("gripper", PRESENT_POSE["gripper"]))
    )
    stall_count = load_count = 0
    gripped = False
    grace_deadline = None
    while True:
        if stop_check and stop_check():
            break
        if commanded > floor:
            commanded = max(commanded - step, floor)
            arm.send_joints({"gripper": commanded})
            if commanded == floor:
                grace_deadline = time.time() + grace_s
        elif grace_deadline is not None and time.time() >= grace_deadline:
            break  # fully closed and the grace window is over: nothing between the jaws
        time.sleep(TICK_S)

        if use_load:
            load = arm.get_gripper_load()
            if load is None:  # current readback died mid-grasp: fall back to the stall
                use_load = False
            elif load >= load_threshold:
                load_count += 1
                if load_count >= load_ticks_needed:
                    gripped = True
                    break
            else:
                load_count = 0

        measured = float(arm.get_joints().get("gripper", commanded))
        if measured - commanded > eps:
            stall_count += 1
            if stall_count >= need_ticks:
                gripped = True
                break
        else:
            stall_count = 0

    if not gripped:
        return False, floor
    measured = float(arm.get_joints().get("gripper", commanded))
    hold = max(min(measured, commanded) - extra, floor)  # a little tighter than contact
    arm.send_joints({"gripper": hold})
    return True, hold


def run_grip(ctx, cmd=None, wait_s: float = 4.0) -> bool:
    """Take an object from the user and lock the grip; "grip" again releases."""
    g = ctx.cfg["grip"]
    state = ctx.shared_state
    roi = g.get("jaw_roi", [0.30, 0.45, 0.70, 0.95])
    diff_thr = float(g.get("jaw_diff_threshold", 0.06))
    cam = ctx.arm_camera

    if state.get("grip_lock"):
        ctx.say("Releasing the object.")
        state.pop("grip_lock", None)
        state.pop("grip_lock_pos", None)
        ctx.arm.go_to({**PRESENT_POSE, "gripper": 80.0}, duration_s=2.0, stop_check=ctx.stop_check)
        time.sleep(1.5)
        ctx.arm.go_to_rest()
        return False

    ctx.say("Place the object between the jaws.")
    ctx.arm.go_to(PRESENT_POSE, duration_s=2.5, stop_check=ctx.stop_check)

    reference = None  # the empty jaws, before the user reaches in
    if cam is not None:
        ok, ref = cam.read()
        reference = ref if ok else None

    for remaining in range(int(wait_s), 0, -1):
        if ctx.stop_check():
            ctx.arm.go_to_rest()
            return False
        logger.info("[grip] closing in %d...", remaining)
        time.sleep(1.0)

    if reference is not None:
        ok, now = cam.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            ctx.say("I don't see anything between the jaws yet - closing carefully.")

    gripped, hold = close_with_feedback(ctx.arm, g, stop_check=ctx.stop_check)

    if gripped and reference is not None:
        ok, now = cam.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            logger.warning("[grip] contact detected but the arm camera sees empty jaws")
            gripped = False

    if gripped:
        state["grip_lock"] = True
        state["grip_lock_pos"] = hold
        ctx.say(
            f"Object gripped at {hold:.1f} degrees - grip locked. "
            "Say 'gesture mimic' to use it; say 'grip' again to release."
        )
    else:
        ctx.say("No object detected.")
        ctx.arm.go_to({"gripper": GRIPPER_CLOSED_FLOOR}, duration_s=0.8)
    return gripped


# ----------------------------------------------------------------------------
# Handover: find the person, extend, release, park
# ----------------------------------------------------------------------------


def person_pan_target(pose, frame: np.ndarray) -> float | None:
    """The pan angle facing the nearest person's torso, or None."""
    person = pose.primary_person(frame)
    if person is None:
        return None
    kpts, _ = person
    xs = [kpts[i, 0] for i in (KP["l_shoulder"], KP["r_shoulder"], KP["nose"]) if kpts[i, 2] > pose.kpt_threshold]
    if not xs:
        return None
    cx = (float(np.mean(xs)) / frame.shape[1]) * 2.0 - 1.0  # [-1, 1]
    return float(np.clip(-cx * 60.0, -80.0, 80.0))  # mirrored: person on the left -> pan left


def find_person_pan(ctx, seek_s: float = 4.0) -> tuple[float, bool]:
    """Track the person for ``seek_s`` and aim the pan joint at them."""
    pan, found = 0.0, False
    if ctx.pose is None or ctx.camera is None:
        logger.info("[handoff] no operator camera - handing over straight ahead")
        return pan, found
    deadline = time.time() + seek_s
    while time.time() < deadline and not ctx.stop_check():
        ok, frame = ctx.camera.read()
        if not ok:
            time.sleep(TICK_S)
            continue
        target = person_pan_target(ctx.pose, frame)
        if target is not None:
            pan, found = target, True
            ctx.arm.send_joints({"shoulder_pan": pan})
        time.sleep(TICK_S)
    logger.info("[handoff] person %s (pan=%.1f)", "found" if found else "NOT found", pan)
    return pan, found


def hand_over(ctx, section: dict, lift_pose: dict) -> None:
    """Find the person, extend toward them, open the jaws, park."""
    pan, _ = find_person_pan(ctx)
    logger.info("[handoff] extend toward the person")
    ctx.arm.go_to(
        {
            "shoulder_pan": pan,
            "shoulder_lift": float(lift_pose["shoulder_lift"]) + float(section["handoff_reach"]),
            "elbow_flex": max(float(lift_pose["elbow_flex"]) - 40.0, 0.0),
            "wrist_flex": 20.0,
        },
        duration_s=2.5,
        stop_check=ctx.stop_check,
    )
    logger.info("[handoff] release - take it!")
    time.sleep(min(float(section["release_timeout_s"]), 6.0) * 0.5)
    ctx.arm.go_to({"gripper": 75.0}, duration_s=1.0)
    time.sleep(1.0)
    ctx.arm.go_to_rest()


# ----------------------------------------------------------------------------
# fetch_block and pick_place
# ----------------------------------------------------------------------------


def run_fetch_block(ctx, cmd=None) -> bool:
    """Grasp the block at its taught pose, lift it, hand it over."""
    fb = ctx.cfg["fetch_block"]
    block_pose, lift_pose = dict(fb["block_pose"]), dict(fb["lift_pose"])

    logger.info("[fetch] approach the block (jaws open)")
    ctx.arm.go_to({**block_pose, "gripper": 70.0}, duration_s=2.5, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    logger.info("[fetch] grasp")
    ctx.arm.go_to({"gripper": float(block_pose.get("gripper", GRIPPER_CLOSED_FLOOR))}, duration_s=1.2)
    time.sleep(0.3)
    logger.info("[fetch] lift")
    ctx.arm.go_to(lift_pose, duration_s=2.5, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    hand_over(ctx, fb, lift_pose)
    return True


def _best(dets, frame_area: float, min_area_frac: float):
    dets = [d for d in dets if d.area / frame_area >= min_area_frac]
    return dets[0] if dets else None


def _detect(ctx, camera, name: str, duration_s: float = 2.0):
    """Sample frames for up to ``duration_s``: ``(matches, class)``; class None = unknown name."""
    deadline = time.time() + duration_s
    resolved = ""
    while time.time() < deadline and not ctx.stop_check():
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue
        dets, resolved = ctx.detector.find(frame, name)
        if resolved is None or dets:
            return dets, resolved
        time.sleep(0.1)
    return [], resolved


def run_pick_place(ctx, cmd) -> bool:
    """Find the named object with the arm camera, grasp it, hand it over."""
    pp, g = ctx.cfg["pick_place"], ctx.cfg["grip"]
    name = (cmd.object_name if cmd is not None else "") or ""
    if ctx.detector is None:
        ctx.say("Object detection isn't available - can't do pick and place.")
        return False
    if not name:
        ctx.say("What should I pick up? Try: pick up the ball.")
        return False
    camera = ctx.arm_camera or ctx.camera
    if camera is None:
        ctx.say("I can't see anything - both cameras are offline.")
        return False
    if camera is not ctx.arm_camera:
        ctx.say("Arm camera is offline - using the operator camera (coarser aim).")

    ctx.say(f"Looking for a {name}.")
    ctx.arm.go_to({**pp["scan_pose"]}, duration_s=2.5, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    dets, resolved = _detect(ctx, camera, name)
    if resolved is None:
        ctx.say(f"Sorry - I don't know what a {name} looks like.")
        ctx.arm.go_to_rest()
        return False
    if not dets:
        ctx.say(f"I can't see a {name} in the workspace.")
        ctx.arm.go_to_rest()
        return False
    if len(dets) > 1 and dets[0].score / max(dets[1].score, 1e-6) < float(pp["ambiguous_ratio"]):
        ctx.say(f"I see {len(dets)} of those - taking the clearest one.")

    ok, frame = camera.read()
    area = float(frame.shape[0] * frame.shape[1]) if ok else 640.0 * 480.0
    if _best(dets, area, float(pp["min_box_area_frac"])) is None:
        ctx.say(f"The {resolved} I can see is too small to grab safely.")
        ctx.arm.go_to_rest()
        return False

    # Center the object horizontally: the camera rides on the arm, so pan opposite the error.
    gain, tol = float(pp["servo_gain_pan"]), float(pp["servo_tolerance"])
    deadline = time.time() + float(pp["servo_timeout_s"])
    centered = False
    while time.time() < deadline and not ctx.stop_check():
        ok, frame = camera.read()
        if not ok:
            time.sleep(TICK_S)
            continue
        dets, _ = ctx.detector.find(frame, name)
        d = _best(dets, frame.shape[0] * frame.shape[1], float(pp["min_box_area_frac"]))
        if d is None:
            time.sleep(TICK_S)
            continue
        err = (d.center[0] / frame.shape[1]) * 2.0 - 1.0
        if abs(err) <= tol:
            centered = True
            break
        ctx.arm.send_joints({"shoulder_pan": ctx.arm.get_joints()["shoulder_pan"] - gain * err})
        time.sleep(0.1)
    if ctx.stop_check():
        return False
    if not centered:
        ctx.say("I couldn't line up on it - grabbing where I last saw it.")

    gripped = False
    attempts = 1 + int(pp.get("retries", 1))
    for attempt in range(attempts):
        if ctx.stop_check():
            return False
        ctx.arm.go_to({**pp["approach_pose"], "gripper": 75.0}, duration_s=1.5, stop_check=ctx.stop_check)
        ctx.arm.go_to({"shoulder_lift": float(pp["descend_lift"])}, duration_s=1.5, stop_check=ctx.stop_check)
        gripped, _ = close_with_feedback(ctx.arm, g, stop_check=ctx.stop_check, start_from=75.0)
        if gripped:
            break
        if attempt + 1 < attempts:
            ctx.say("The grasp slipped - trying once more.")
            ctx.arm.go_to({**pp["scan_pose"]}, duration_s=1.5, stop_check=ctx.stop_check)
            dets, _ = _detect(ctx, camera, name, duration_s=1.5)
            if not dets:
                break
    if not gripped:
        ctx.say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        ctx.arm.go_to_rest()
        return False

    ctx.say(f"Got the {resolved} - bringing it to you.")
    lift_pose = {
        "shoulder_pan": ctx.arm.get_joints()["shoulder_pan"],
        "shoulder_lift": -80.0,
        "elbow_flex": 85.0,
        "wrist_flex": 50.0,
        "wrist_roll": 0.0,
    }
    ctx.arm.go_to(lift_pose, duration_s=2.0, stop_check=ctx.stop_check)
    if ctx.stop_check():
        return False
    hand_over(ctx, pp, lift_pose)
    ctx.say("Done.")
    return True


def register_manipulation(registry) -> None:
    """Add grip, pick_place and fetch_block to ``registry``."""
    registry.add(Intent.GRIP, run_grip)
    registry.add(Intent.PICK_PLACE, run_pick_place)
    registry.add(Intent.FETCH_BLOCK, run_fetch_block)
