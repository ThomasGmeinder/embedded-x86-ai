# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Voice-grounded pick-and-place: "pick up the ball" → find it → grasp →
hand it to the person.

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
   come straight down. No fix or out of reach - or ``reach: fixed`` - uses
   the taught ``approach_pose`` + ``descend_lift``. The camera model is
   ``cameras.arm.intrinsics`` / ``mount`` (``scripts/calibrate_arm_camera.py``),
   the twin's until calibrated.
5. **GRASP** - close with the same feedback-based loop the grip behavior
   uses (:func:`vla_pipeline.behaviors.grip.close_with_feedback`):
   effort/current when available, position stall + thin-object grace window
   otherwise; then settle and check the grip holds.
6. **HANDOFF** - straight back up, lift, check the grip again, find the
   person with YOLO-pose on the operator camera, extend, release, home
   (shared with fetch_block).

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

from vla_pipeline.behaviors.fetch_block import track_person_and_handoff
from vla_pipeline.behaviors.grip import close_with_feedback, holding
from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, ArmClient, DryRunArm, make_arm
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
) -> tuple[list[Detection], str | None]:
    """Sample frames for up to ``duration_s``; return the first hit."""
    deadline = time.time() + duration_s
    resolved: str | None = ""
    while time.time() < deadline and not (stop_check and stop_check()):
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue
        dets, resolved = detector.find(frame, object_name)
        if resolved is None:
            return [], None
        if dets:
            return dets, resolved
        time.sleep(0.1)
    return [], resolved


def run_pick_place(
    cfg: dict,
    arm: ArmClient,
    detector: ObjectDetector,
    pose,  # PoseEstimator | None - person finding
    arm_camera,  # CameraClient | None
    operator_camera,  # CameraClient | None
    object_name: str,
    shared_state: dict | None = None,
    stop_check=None,
    feedback=None,
) -> bool:
    """Pick the named object and hand it to the person. True on success."""
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
    arm.go_to({**pp["scan_pose"]}, duration_s=2.5, stop_check=stop_check)
    if stopped():
        return False

    dets, resolved = _detect_with_retry(
        detector, camera, object_name, stop_check=stop_check
    )
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

    centered = _servo_pan(pp, arm, detector, camera, object_name, stop_check)
    if stopped():
        return False
    if not centered:
        say("I couldn't line up on it - grabbing where I last saw it.")

    # ---------------- GRASP (with one retry) ----------------
    attempts = 1 + int(pp.get("retries", 1))
    gripped, hold = False, 0.0
    for attempt in range(attempts):
        if stopped():
            return False
        aligned_pan = arm.get_joints()["shoulder_pan"]
        way_up = None
        if pp.get("reach", "fixed") == "ik" and camera is arm_camera:
            way_up = _reach_ik(cfg, arm, detector, camera, object_name, stop_check)
        if stopped():
            return False
        if way_up is None:
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
            dets, _ = _detect_with_retry(
                detector, camera, object_name, duration_s=1.5, stop_check=stop_check
            )
            if not dets:
                break
            _servo_pan(pp, arm, detector, camera, object_name, stop_check)
    if stopped():
        return False
    if not gripped:
        say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        arm.go_to_rest()
        return False

    # ---------------- LIFT + HANDOFF ----------------
    # straight back up the way the jaws came down, so the object isn't dragged
    for step in way_up:
        arm.go_to({**step, "gripper": hold}, duration_s=1.5 / len(way_up), stop_check=stop_check)
        if stopped():
            return False
    lift_pose = {
        "shoulder_pan": arm.get_joints()["shoulder_pan"],
        "shoulder_lift": -80.0,
        "elbow_flex": 85.0,
        "wrist_flex": 50.0,
        "wrist_roll": 0.0,
        "gripper": hold,
    }
    arm.go_to(lift_pose, duration_s=2.0, stop_check=stop_check)
    if stopped():
        return False
    time.sleep(0.3)
    if not holding(arm, g, hold):
        say(f"I dropped the {resolved}.")
        arm.go_to({"gripper": 75.0}, duration_s=0.6)
        arm.go_to_rest()
        return False
    say(f"Got the {resolved} - bringing it to you.")
    track_person_and_handoff(pp, arm, pose, operator_camera, lift_pose, stop_check)
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


def _plan_ik_reach(cfg: dict, arm: ArmClient, detector, camera, object_name: str):
    """Locate the object on the table from a fresh arm-camera detection and
    solve the reach: joint poses from the pre-grasp straight down to the
    grasp, or None."""
    pp = cfg["behaviors"]["pick_place"]
    try:
        from so101_sim.kinematics import ROUND_CLASSES, shared_kinematics

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    except Exception as e:  # so101_sim / mujoco / model assets missing
        logger.warning("IK reach unavailable: %s", e)
        return None
    time.sleep(0.3)  # let the servo's last pan step settle
    camera.read()  # drop a frame that may predate it
    joints = arm.get_joints()
    ok, frame = camera.read()
    if not ok:
        return None
    dets, resolved = detector.find(frame, object_name)
    d = _best_detection(dets, frame.shape[0] * frame.shape[1], float(pp["min_box_area_frac"]))
    if d is None:
        return None
    h, w = frame.shape[:2]
    cam = cfg["cameras"]["arm"]
    K, dist = kin.intrinsics(cam.get("intrinsics"), w, h)
    table_z = float(pp.get("table_z", -0.005))
    loc = kin.locate_on_table(
        d.box, joints, K, dist, cam.get("mount"), table_z, (w, h),
        shape="sphere" if resolved in ROUND_CLASSES else "upright",
    )
    if loc is None:
        logger.info("IK reach: no table fix for the %s box %s", resolved, d.box)
        return None
    p = loc["xyz"].copy()
    p[2] = min(p[2], table_z + float(pp.get("max_grasp_height_m", 0.03)))
    plan = kin.solve_grasp(
        p, above_m=float(pp.get("pregrasp_above_m", 0.06)), width=loc["width"], table_z=table_z
    )
    logger.info(
        "IK reach: %s at (%.3f, %.3f, %.3f) m, %.0f mm wide -> %s",
        resolved, *loc["xyz"], 1000 * loc["width"], "solved" if plan else "out of reach",
    )
    if plan is None:
        return None
    pregrasp, grasp = plan
    return [pregrasp, *kin.waypoints(pregrasp, grasp, loc["width"])]


def _reach_ik(cfg: dict, arm: ArmClient, detector, camera, object_name: str, stop_check=None):
    """Jaws open over the located object, then straight down onto it. Returns
    the way back up (poses without the gripper), or None to use the fixed reach."""
    down = _plan_ik_reach(cfg, arm, detector, camera, object_name)
    if down is None:
        logger.info("IK reach not possible - using the fixed approach")
        return None
    arm.go_to(down[0], duration_s=1.5, stop_check=stop_check)
    for pose in down[1:]:
        arm.go_to(pose, duration_s=1.5 / (len(down) - 1), stop_check=stop_check)
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
