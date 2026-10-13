# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Voice-grounded pick-and-place: "pick up the ball" → find it → grasp →
lift it clear → hold it out. The jaws stay closed.

Pipeline
--------
1. **GROUND** - the object name parsed from the voice command
   (``ParsedCommand.object_name``) is resolved to a COCO class
   (:func:`vla_pipeline.vision.yolo_detect_npu.resolve_class`). Unknown names
   fail fast with a spoken explanation instead of guessing.
2. **SCAN** - move to ``scan_pose`` and run YOLOv26s detection (NPU) on the
   **arm camera** (falls back to the operator camera if the arm camera is down).
3. **SERVO** - proportional visual servoing on ``shoulder_pan`` turns the
   base until the detection sits at ``servo_target_x`` in the image, where
   an object between the jaws appears (``servo_gain_pan`` deg per unit of
   normalized error).
4. **LOCATE + REACH** (``reach: ik``) - the box of a fresh detection is
   turned into the object's position on the table
   (:meth:`so101_sim.kinematics.Kinematics.locate_on_table`: FK gives the
   camera pose, the ray through the box bottom meets the table plane); IK
   solves a top-down grasp, the jaws open ``pregrasp_above_m`` above it and
   come straight down in joint-space steps. A standing object with no
   solution stops; the ball and the other short objects, or ``reach: fixed``,
   use the taught ``approach_pose`` + ``descend_lift``. The camera model is
   ``cameras.arm.intrinsics`` / ``mount`` (``scripts/calibrate_arm_camera.py``),
   the twin's until calibrated.
5. **GRASP** - close with the same feedback-based loop the grip behavior
   uses (:func:`vla_pipeline.behaviors.grip.close_with_feedback`):
   effort/current when available, position stall + thin-object grace window
   otherwise; then settle and check the grip holds.
6. **OFFER** - lift the grasp point above the other props and hold the
   object out toward the operator. The jaws stay closed, so the object can
   be taken or left.

Failure handling: every stage reports through the ``feedback`` callback
("I can't see a ball", "I see two cups, taking the clearest one", "the grasp
slipped, trying once more") and the behavior retries the grasp
``retries`` times from the servoed pan before going home. It returns False
whenever it gives up, so the dispatcher reports the command as failed.

Independent test
----------------
    python -m vla_pipeline.behaviors.pick_place --object ball --dry-run-arm
    python -m vla_pipeline.behaviors.pick_place --object "cell phone"
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vla_pipeline.behaviors.fetch_block import park_open
from vla_pipeline.behaviors.grip import close_with_feedback, holding
from vla_pipeline.robot.arm_interface import (
    GRIPPER_CLOSED_FLOOR,
    JOINT_LIMITS,
    ArmClient,
    DryRunArm,
    make_arm,
)
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_detect_npu import Detection, ObjectDetector

logger = logging.getLogger(__name__)

GRASP_SETTLE_S = 0.4


def _best_detection(
    dets: list[Detection], frame_area: float, min_area_frac: float
) -> Detection | None:
    """Return the highest-scoring detection at or above min_area_frac of the frame, or None."""
    dets = [d for d in dets if d.area / frame_area >= min_area_frac]
    return dets[0] if dets else None


def _detect_with_retry(
    detector: ObjectDetector,
    camera,
    object_name: str,
    duration_s: float = 2.0,
    stop_check=None,
) -> tuple[list[Detection], str | None, np.ndarray | None]:
    """Sample frames for up to ``duration_s``; return the first hit and its frame."""
    deadline = time.time() + duration_s
    resolved: str | None = ""
    while time.time() < deadline and not (stop_check and stop_check()):
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue
        dets, resolved = detector.find(frame, object_name)
        if resolved is None:
            return [], None, frame
        if dets:
            return dets, resolved, frame
        time.sleep(0.1)
    return [], resolved, None


# Grasp point high enough that a carried ball (its centre is the grasp
# point) clears the bottle, the tallest prop on the table.
_CARRY_Z = 0.12
# Out toward the operator, who stands in front of the arm.
_OFFER_XY = (0.30, 0.0)
_OFFER_ERR_M = 0.015


def _offer(arm, hold, stop_check, limits, say, resolved, still_holding) -> str:
    """Lift above the other props and hold the object out. The jaws stay closed.

    Two joint-space legs from the pose that is holding the object, keeping
    its wrist roll. Returns ``offered``, ``dropped``, ``stopped``, or
    ``unreachable``.
    """
    try:
        from so101_sim.kinematics import joint_line, shared_kinematics
    except Exception as e:
        logger.warning("offer unavailable: %s", e)
        return "unreachable"

    def stopped() -> bool:
        return bool(stop_check and stop_check())

    def follow(steps, duration_s: float) -> bool:
        for step in steps:
            arm.go_to(step, duration_s=duration_s / len(steps), stop_check=stop_check)
            if stopped():
                return False
        return True

    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    here = arm.get_joints()
    roll = float(here.get("wrist_roll", 0.0))
    origin = kin.world.grasp_point(here, 0.04)
    targets = (
        np.array([origin[0], origin[1], _CARRY_Z]),
        np.array([_OFFER_XY[0], _OFFER_XY[1], _CARRY_Z]),
    )
    logger.info(
        "offer: lift to %.2f m, then out to (%.2f, %.2f)",
        _CARRY_Z, _OFFER_XY[0], _OFFER_XY[1],
    )
    seed = here
    for target, duration_s in zip(targets, (2.5, 2.0)):
        pose, err, _tilt = kin.ik(
            target, 0.04, hold, seed, roll, limits=limits, one_branch=True,
        )
        if err > _OFFER_ERR_M:
            logger.info("offer: no pose within 15 mm (err %.0f mm)", 1000 * err)
            return "unreachable"
        if not follow(joint_line(seed, {**pose, "gripper": hold}), duration_s):
            return "stopped"
        time.sleep(0.3)
        if not still_holding():
            return "dropped"
        seed = arm.get_joints()
    say(f"Got the {resolved}.")
    return "offered"


def run_pick_place(
    cfg: dict,
    arm: ArmClient,
    detector: ObjectDetector,
    pose,  # PoseEstimator | None - kept for the dispatcher; the offer does not search for a person
    arm_camera,  # CameraClient | None
    operator_camera,  # CameraClient | None
    object_name: str,
    shared_state: dict | None = None,
    stop_check=None,
    feedback=None,
) -> bool:
    """Pick the named object and hold it out. The jaws stay closed."""
    pp = cfg["behaviors"]["pick_place"]
    g = cfg["behaviors"]["grip"]
    say = feedback or (lambda m: logger.info("[pick] %s", m))
    shared_state = shared_state if shared_state is not None else {}

    def stopped() -> bool:
        """Return True if a stop has been requested."""
        return bool(stop_check and stop_check())

    if not object_name:
        say("What should I pick up? Try: pick up the ball.")
        return False

    camera = arm_camera if (arm_camera and arm_camera.ok()) else operator_camera
    if camera is None or not camera.ok():
        say("I can't see anything - both cameras are offline.")
        return False
    if camera is not arm_camera:
        say("Arm camera is offline - using the operator camera (coarser aim).")

    # ---------------- SCAN ----------------
    say(f"Looking for a {object_name}.")
    base_scan = dict(pp["scan_pose"])
    scan_pans = pp.get("scan_pans") if camera is arm_camera else None
    scan_pans = list(scan_pans or [base_scan.get("shoulder_pan", 0.0)])
    scan_sets = [(base_scan, scan_pans)]
    if camera is arm_camera and pp.get("scan_near_pose"):
        near = dict(pp["scan_near_pose"])
        scan_sets.append((near, list(pp.get("scan_near_pans") or [near.get("shoulder_pan", 0.0)])))
    dets, resolved, detection_frame = [], "", None
    for scan_index, (scan_pose, pans) in enumerate(scan_sets):
        for i, pan in enumerate(pans):
            arm.go_to(
                {**scan_pose, "shoulder_pan": float(pan)},
                duration_s=2.5 if scan_index == 0 and i == 0 else 1.2,
                stop_check=stop_check,
            )
            if stopped():
                return False
            dets, resolved, detection_frame = _detect_with_retry(
                detector, camera, object_name, duration_s=0.8, stop_check=stop_check
            )
            if resolved is None or dets:
                break
        if resolved is None or dets:
            break
    if resolved is None:
        say(f"Sorry - I don't know what a {object_name} looks like.")
        arm.go_to_rest()
        return False
    if not dets:
        say(f"I can't see a {object_name} in the workspace.")
        arm.go_to_rest()
        return False
    if len(dets) > 1 and dets[0].score / max(dets[1].score, 1e-6) < float(
        pp["ambiguous_ratio"]
    ):
        say(f"I see {len(dets)} of those - taking the clearest one.")

    # ---------------- SERVO (center horizontally with pan) ----------------
    ok, frame = camera.read()
    frame_area = float(frame.shape[0] * frame.shape[1]) if ok else 640.0 * 480.0
    target = _best_detection(dets, frame_area, float(pp["min_box_area_frac"]))
    if target is None:
        say(f"The {resolved} I can see is too small to grab safely.")
        arm.go_to_rest()
        return False

    # The scan frame, paired with the joints that took it. Used only when the
    # pan servo cannot centre the object. A centred view is one solve; it is
    # not a second solve of a plan we already kept.
    wrist_align = pp.get("wrist_align_classes", [])
    standing = _standing(resolved, wrist_align)
    scan_observation = (detection_frame, dets, resolved, arm.get_joints())
    centered = _servo_pan(pp, arm, detector, camera, object_name, stop_check)
    if stopped():
        return False
    if not centered:
        say("I couldn't line up on it - grabbing where I last saw it.")
    initial_ik = None
    if (
        pp.get("reach", "fixed") == "ik"
        and camera is arm_camera
        and resolved not in wrist_align
    ):
        if centered:
            initial_ik = _plan_ik_reach(
                cfg, arm, detector, camera, object_name, resolved,
            )
        if initial_ik is None and (not standing or not centered):
            # No centred plan: either the servo never lined up, or a short
            # object ended up behind the jaws. The scan frame is the plan
            # then. A standing object does not take a second, different angle.
            initial_ik = _plan_ik_reach(
                cfg, arm, detector, camera, object_name, resolved,
                observation=scan_observation,
            )
    if standing and pp.get("reach", "fixed") == "ik" and initial_ik is None:
        say(f"The {resolved} is out of reach.")
        arm.go_to_rest()
        return False
    held = [initial_ik]

    # ---------------- GRASP (with one retry of the same plan) ----------------
    attempts = 1 + int(pp.get("retries", 1))
    gripped, hold = False, 0.0
    for attempt in range(attempts):
        if stopped():
            return False
        aligned_pan = arm.get_joints()["shoulder_pan"]
        way_up = None
        if pp.get("reach", "fixed") == "ik" and camera is arm_camera:
            way_up = _reach_ik(
                cfg, arm, detector, camera, object_name, resolved, stop_check,
                down=held[0], held=held,
            )
        if stopped():
            return False
        if way_up is None and resolved in wrist_align:
            say(f"I couldn't align my gripper with the {resolved}.")
            arm.go_to_rest()
            return False
        if way_up is None and standing:
            say(f"The {resolved} is out of reach.")
            arm.go_to_rest()
            return False
        if way_up is None:
            logger.info("IK reach not possible - using the fixed approach")
            way_up = _reach_fixed(pp, arm, stop_check)
        if stopped():
            return False
        gripped, hold = close_with_feedback(
            arm, g, stop_check=stop_check, start_from=75.0
        )
        if stopped():
            return False
        if gripped:
            hold = _settle_grip(arm, g)
            if resolved in pp.get("wrist_align_classes", []):
                hold = max(
                    hold - float(pp.get("wrist_align_settle_extra_deg", 8.0)),
                    GRIPPER_CLOSED_FLOOR,
                )
                arm.send_joints({"gripper": hold})
                time.sleep(0.2)
        if gripped and holding(arm, g, hold):
            break
        gripped = False
        arm.go_to({"gripper": 75.0}, duration_s=0.6, stop_check=stop_check)
        if attempt + 1 < attempts:
            say("The grasp slipped - trying once more.")
            arm.go_to(
                {**pp["scan_pose"], "shoulder_pan": aligned_pan},
                duration_s=1.5,
                stop_check=stop_check,
            )
            dets, _, _ = _detect_with_retry(
                detector, camera, object_name, duration_s=1.5, stop_check=stop_check
            )
            if not dets:
                break
            _servo_pan(pp, arm, detector, camera, object_name, stop_check)
    if stopped():
        return False
    if not gripped:
        say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        park_open(arm)
        return False

    status = _offer(
        arm, hold, stop_check, JOINT_LIMITS, say, resolved,
        lambda: holding(arm, g, hold),
    )
    if status == "stopped":
        return False
    if status == "dropped":
        say(f"I dropped the {resolved}.")
        park_open(arm)
        return False
    if status != "offered":
        say(f"I can't lift the {resolved} clear of the other objects.")
        return False
    shared_state["grip_lock"] = True
    shared_state["grip_lock_pos"] = hold
    say("Done.")
    return True


def _servo_pan(pp: dict, arm: ArmClient, detector, camera, object_name: str, stop_check=None) -> bool:
    """Turn the base until the object sits at ``servo_target_x`` in the image. True once centered."""
    gain = float(pp["servo_gain_pan"])
    tol = float(pp["servo_tolerance"])
    target_x = float(pp.get("servo_target_x", 0.0))
    deadline = time.time() + float(pp["servo_timeout_s"])
    while time.time() < deadline and not (stop_check and stop_check()):
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.05)
            continue
        dets, _ = detector.find(frame, object_name)
        d = _best_detection(
            dets, frame.shape[0] * frame.shape[1], float(pp["min_box_area_frac"])
        )
        if d is None:
            time.sleep(0.05)
            continue
        # [-1, 1], relative to where the object sits in the image when it is between the jaws
        err = (d.center[0] / frame.shape[1]) * 2.0 - 1.0 - target_x
        if abs(err) <= tol:
            return True
        # the sign of servo_gain_pan follows how the camera is mounted on the wrist
        arm.send_joints({"shoulder_pan": arm.get_joints()["shoulder_pan"] - gain * err})
        time.sleep(0.1)
    return False


def _axis_observation(pp: dict, detector, camera, object_name: str):
    """Fresh ``(principal-axis, (width, height))`` for an elongated object."""
    from so101_sim.kinematics import principal_axis_in_box

    camera.read()  # discard a frame captured before the last joint command
    ok, frame = camera.read()
    if not ok:
        return None
    dets, _ = detector.find(frame, object_name)
    d = _best_detection(
        dets, frame.shape[0] * frame.shape[1], float(pp["min_box_area_frac"])
    )
    if d is None:
        return None
    axis = principal_axis_in_box(
        frame,
        d.box,
        min_elongation=float(pp.get("wrist_axis_min_elongation", 1.6)),
    )
    return None if axis is None else (axis, (frame.shape[1], frame.shape[0]))


def _align_view(
    cfg: dict, arm: ArmClient, detector, camera, object_name: str, stop_check=None
) -> bool:
    """Turn the jaws across an elongated object while the camera keeps it in view.

    The camera rides on the gripper, so rolling the wrist alone swings the
    object out of the image. Each round measures the object's long axis
    (PCA inside the box, carried onto the table plane by camera rays),
    solves the wrist roll that closes the jaws across it, and moves the whole
    arm there in roll steps along look-at poses that keep the object centred
    (:meth:`so101_sim.kinematics.Kinematics.view_path`). Converged once a
    fresh view asks for less than ``wrist_axis_tolerance_deg`` more roll.
    """
    from so101_sim.kinematics import shared_kinematics

    pp = cfg["behaviors"]["pick_place"]
    cam = cfg["cameras"]["arm"]
    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    table_z = float(pp.get("table_z", -0.005))
    width = float(pp.get("wrist_align_grasp_width_m", 0.024))
    tolerance = float(pp.get("wrist_axis_tolerance_deg", 6.0))
    distance = float(pp.get("wrist_view_distance_m", 0.17))
    look_down = float(pp.get("wrist_view_look_down", 0.85))
    roll_step = float(pp.get("wrist_view_roll_step_deg", 15.0))
    step_s = float(pp.get("wrist_view_step_s", 0.5))
    above_m = float(pp.get("wrist_align_pregrasp_above_m", 0.10))
    for round_ in range(int(pp.get("wrist_axis_max_rounds", 4))):
        if stop_check and stop_check():
            return False
        time.sleep(0.2)
        joints = arm.get_joints()
        obs = _axis_observation(pp, detector, camera, object_name)
        if obs is None:
            logger.info("wrist alignment: lost sight of the %s", object_name)
            return False
        axis, size = obs
        K, dist = kin.intrinsics(cam.get("intrinsics"), *size)
        found = kin.table_axis(
            axis["center"], axis["direction"], axis["length_px"], joints, K, dist,
            cam.get("mount"), table_z + 0.006,
        )
        if found is None:
            return False
        p, world_axis = found
        grasp_p = np.array([p[0], p[1], table_z + 0.011])
        roll_now = float(joints.get("wrist_roll", 0.0))
        grasp_plan = kin.solve_grasp(
            grasp_p, above_m=above_m, width=width, table_z=table_z,
            wrist_roll=roll_now, limits=JOINT_LIMITS, object_axis=world_axis,
            seed=joints,
        )
        if grasp_plan is None:
            logger.info("wrist alignment: no reachable roll for the %s", object_name)
            return False
        target = float(grasp_plan[1]["wrist_roll"])
        change = target - roll_now
        logger.info(
            "wrist alignment: %.1f deg roll, %.1f deg to go, %.1f deg image axis",
            roll_now, change, axis["angle_deg"],
        )
        if round_ > 0 and abs(change) <= tolerance:
            return True
        start, centre_err = kin.look_at(
            p, roll_now, distance, look_down, joints, cam.get("mount"), limits=JOINT_LIMITS
        )
        path = kin.view_path(
            p, start, target, distance, look_down, roll_step, cam.get("mount"), limits=JOINT_LIMITS
        )
        if centre_err > 0.05 or path is None:
            logger.info("wrist alignment: no view-keeping path to %.1f deg", target)
            return False
        gripper = joints.get("gripper", 75.0)
        for pose in [start, *path]:
            arm.go_to({**pose, "gripper": gripper}, duration_s=step_s, stop_check=stop_check)
            if stop_check and stop_check():
                return False
    return False


def _standing(resolved: str, wrist_align) -> bool:
    """A tall object the hand has to come down around. The ball and the
    toothbrush are not: they keep the short-object reach."""
    from so101_sim.kinematics import FLAT_CLASSES, ROUND_CLASSES

    return (
        resolved not in ROUND_CLASSES
        and resolved not in FLAT_CLASSES
        and resolved not in wrist_align
    )


def _reach_fixed(pp: dict, arm: ArmClient, stop_check=None) -> list[dict]:
    """Jaws open over the object at the configured reach, then down onto it.
    Returns the way back up (poses without the gripper)."""
    above = {**pp["approach_pose"], "shoulder_pan": arm.get_joints()["shoulder_pan"]}
    arm.go_to({**above, "gripper": 75.0}, duration_s=1.5, stop_check=stop_check)
    arm.go_to(
        {"shoulder_lift": float(pp["descend_lift"])},
        duration_s=1.5,
        stop_check=stop_check,
    )
    return [above]


def _plan_ik_reach(
    cfg: dict, arm: ArmClient, detector, camera, object_name: str, resolved_name: str,
    observation=None,
):
    """Locate the object on the table from a fresh arm-camera detection and
    solve the reach: joint poses from the pre-grasp straight down to the
    grasp, or None."""
    pp = cfg["behaviors"]["pick_place"]
    try:
        from so101_sim.kinematics import (
            FLAT_CLASSES,
            ROUND_CLASSES,
            principal_axis_in_box,
            shared_kinematics,
        )

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    except Exception as e:  # so101_sim / mujoco / model assets missing
        logger.warning("IK reach unavailable: %s", e)
        return None
    if observation is not None:
        frame, dets, resolved, joints = observation
        d = None if frame is None else _best_detection(
            dets, frame.shape[0] * frame.shape[1], float(pp["min_box_area_frac"])
        )
    else:
        time.sleep(0.3)  # let the servo's last pan step settle
        camera.read()  # drop a frame that may predate it
        joints = arm.get_joints()
        deadline = time.time() + 0.8
        frame = d = resolved = None
        while time.time() < deadline:
            ok, candidate = camera.read()
            if ok:
                dets, resolved = detector.find(candidate, object_name)
                best = _best_detection(
                    dets,
                    candidate.shape[0] * candidate.shape[1],
                    float(pp["min_box_area_frac"]),
                )
                if best is not None:
                    frame, d = candidate, best
                    break
            time.sleep(0.05)
    if d is None:
        return None
    h, w = frame.shape[:2]
    cam = cfg["cameras"]["arm"]
    K, dist = kin.intrinsics(cam.get("intrinsics"), w, h)
    table_z = float(pp.get("table_z", -0.005))
    shape = (
        "sphere" if resolved in ROUND_CLASSES
        else "flat" if resolved in FLAT_CLASSES
        else "bottle" if resolved == "bottle"
        else "upright"
    )
    loc = kin.locate_on_table(
        d.box, joints, K, dist, cam.get("mount"), table_z, (w, h),
        shape=shape,
    )
    if loc is None:
        logger.info("IK reach: no table fix for the %s box %s", resolved, d.box)
        return None
    p = loc["xyz"].copy()
    p[2] = min(p[2], table_z + float(pp.get("max_grasp_height_m", 0.03)))
    # a standing object's own volume: the hand has to come down around it
    obstacle = (
        (*loc["xyz"][:2], 0.5 * loc["width"], table_z + loc["height"])
        if shape in ("bottle", "upright")
        else None
    )
    oriented = resolved_name in pp.get("wrist_align_classes", [])
    if oriented:
        axis = principal_axis_in_box(
            frame,
            d.box,
            min_elongation=float(pp.get("wrist_axis_min_elongation", 1.6)),
        )
        if axis is not None:
            centre_px = np.asarray(axis["center"])
            along_px = centre_px + (
                np.asarray(axis["direction"]) * axis["length_px"] * 0.25
            )
            origin, rays = kin.rays(
                [centre_px, along_px], joints, K, dist, cam.get("mount")
            )
            top_z = table_z + 0.010
            if origin[2] > top_z and (rays[:, 2] < -1e-6).all():
                points = origin + rays * ((top_z - origin[2]) / rays[:, 2])[:, None]
                direction = points[1, :2] - points[0, :2]
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
                offset_m = (
                    float(pp.get("wrist_align_center_offset_m", 0.018))
                    if abs(float(joints.get("wrist_roll", 0.0))) < 30.0
                    else 0.0
                )
                p[:2] = points[0, :2] + direction * offset_m
        # A few millimetres above the thin handle keeps the jaw tips clear of
        # the table; the close still spans the full handle thickness.
        p[2] = max(p[2], table_z + 0.011)
    grasp_width = (
        float(pp.get("wrist_align_grasp_width_m", 0.024))
        if oriented
        else loc["width"]
    )
    above_m = float(
        pp.get("wrist_align_pregrasp_above_m", 0.10)
        if oriented
        else pp.get("pregrasp_above_m", 0.06)
    )
    # the arm clamps any pose; an aligned grasp, or one that has to clear
    # the object, is only collision-free if solved within its limits
    limits = JOINT_LIMITS if oriented or obstacle else None
    wrist_roll = float(arm.get_joints().get("wrist_roll", 0.0)) if oriented else 0.0
    plan = kin.solve_grasp(
        p,
        above_m=above_m,
        width=grasp_width,
        table_z=table_z,
        wrist_roll=wrist_roll,
        limits=limits,
        obstacle=obstacle,
    )
    if plan is not None and oriented and abs(wrist_roll) >= 30.0:
        # Start a thin object against the fixed finger. The moving finger can
        # then close across it instead of pushing it through empty space.
        jaw = kin.world.jaw_axis(plan[1])
        p = p + jaw * float(pp.get("wrist_align_fixed_jaw_bias_m", 0.024))
        plan = kin.solve_grasp(
            p,
            above_m=above_m,
            width=grasp_width,
            table_z=table_z,
            wrist_roll=wrist_roll,
            limits=limits,
        )
    logger.info(
        "IK reach: %s at (%.3f, %.3f, %.3f) m, %.0f mm wide -> %s",
        resolved, *p, 1000 * grasp_width, "solved" if plan else "out of reach",
    )
    if plan is None:
        return None
    pregrasp, grasp = plan
    from so101_sim.kinematics import joint_line

    # Joint-space steps, including the grasp. One pair of poses, one path.
    return [pregrasp, *joint_line(pregrasp, grasp)]


def _reach_ik(
    cfg: dict,
    arm: ArmClient,
    detector,
    camera,
    object_name: str,
    resolved_name: str,
    stop_check=None,
    down=None,
    held=None,
):
    """Jaws open over the located object, then straight down onto it. Returns
    the way back up (poses without the gripper), or None when there is no plan.

    ``down`` is that plan. A retry passes the same list so the arm repeats it.
    ``held`` stores the plan from an aligned object, which is solved here on
    the first attempt.
    """
    pp = cfg["behaviors"]["pick_place"]
    # The scan frame already produced the one plan. Solving again here is only
    # for an aligned object, whose roll is known after the wrist has turned.
    if down is None and resolved_name in pp.get("wrist_align_classes", []):
        if not _align_view(cfg, arm, detector, camera, object_name, stop_check):
            logger.info("wrist alignment: no confident solution")
            return None
        down = _plan_ik_reach(
            cfg, arm, detector, camera, object_name, resolved_name
        )
    if held is not None:
        held[:] = [down]
    if down is None:
        return None
    from so101_sim.kinematics import joint_line

    transfer = joint_line(arm.get_joints(), down[0])
    for pose in transfer:
        arm.go_to(pose, duration_s=1.5 / len(transfer), stop_check=stop_check)
    for pose in down[1:]:
        arm.go_to(pose, duration_s=1.5 / max(len(down) - 1, 1), stop_check=stop_check)
    return [{k: v for k, v in p.items() if k != "gripper"} for p in reversed(down[:-1])]


def _settle_grip(arm: ArmClient, g: dict) -> float:
    """Contact can register while one jaw is still pushing the object: let the
    jaws finish closing, then hold a little tighter than where they stopped."""
    time.sleep(GRASP_SETTLE_S)
    hold = max(arm.get_joints()["gripper"] - float(g["settle_extra_deg"]), GRIPPER_CLOSED_FLOOR)
    arm.send_joints({"gripper": hold})
    time.sleep(0.2)
    return hold


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run pick-and-place standalone for a named object."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Voice-grounded pick-and-place - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--object", required=True, help='spoken object name, e.g. "ball"'
    )
    parser.add_argument(
        "--dry-run-arm",
        action="store_true",
        help="real cameras + detection, simulated arm",
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    from vla_pipeline.vision.camera import make_camera
    from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

    arm: ArmClient = DryRunArm() if args.dry_run_arm else make_arm(cfg)
    detector = ObjectDetector(cfg, device=args.device)
    pose = PoseEstimator(cfg, device=args.device)
    arm_cam = make_camera(cfg, "arm")
    operator_cam = make_camera(cfg, "operator")
    try:
        ok = run_pick_place(
            cfg,
            arm,
            detector,
            pose,
            arm_cam,
            operator_cam,
            args.object,
            feedback=lambda m: print(f"{m}"),
        )
        print(f"\npick_place {'PASSED' if ok else 'FAILED'}")
        raise SystemExit(0 if ok else 1)
    finally:
        arm_cam.close()
        operator_cam.close()
        arm.close()


if __name__ == "__main__":
    main()
