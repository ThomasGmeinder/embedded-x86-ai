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
   **arm camera** (falls back to the mount camera if the arm camera is down).
3. **SERVO** - proportional visual servoing on ``shoulder_pan`` centers the
   detection horizontally (``servo_gain_pan`` deg per unit of normalized
   error) before committing to the approach.
4. **GRASP** - descend (``descend_lift``) and close with the same
   feedback-based loop the grip behavior uses
   (:func:`vla_pipeline.behaviors.grip.close_with_feedback`): effort/current
   when available, position stall + thin-object grace window otherwise.
5. **HANDOFF** - lift, find the person with YOLO-pose on the mount camera,
   extend, release, home (shared with fetch_block).

Failure handling: every stage reports through the ``feedback`` callback
("I can't see a ball", "I see two cups, taking the clearest one", "the grasp
slipped, trying once more") and the behavior retries the grasp
``retries`` times before going home.

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
from vla_pipeline.behaviors.grip import close_with_feedback
from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, make_arm
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_detect_npu import Detection, ObjectDetector

logger = logging.getLogger(__name__)


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
    mount_camera,  # CameraClient | None
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

    camera = arm_camera if (arm_camera and arm_camera.ok()) else mount_camera
    if camera is None or not camera.ok():
        say("I can't see anything - both cameras are offline.")
        return False
    if camera is not arm_camera:
        say("Arm camera is offline - using the mount camera (coarser aim).")

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

    gain = float(pp["servo_gain_pan"])
    tol = float(pp["servo_tolerance"])
    deadline = time.time() + float(pp["servo_timeout_s"])
    centered = False
    while time.time() < deadline and not stopped():
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
        err = (d.center[0] / frame.shape[1]) * 2.0 - 1.0  # [-1, 1]
        if abs(err) <= tol:
            centered = True
            break
        pan = (
            arm.get_joints()["shoulder_pan"] - gain * err
        )  # camera on EE: move opposite the error
        arm.send_joints({"shoulder_pan": pan})
        time.sleep(0.1)
    if stopped():
        return False
    if not centered:
        say("I couldn't line up on it - grabbing where I last saw it.")

    # ---------------- GRASP (with one retry) ----------------
    attempts = 1 + int(pp.get("retries", 1))
    gripped = False
    for attempt in range(attempts):
        if stopped():
            return False
        arm.go_to(
            {**pp["approach_pose"], "gripper": 75.0},
            duration_s=1.5,
            stop_check=stop_check,
        )
        arm.go_to(
            {"shoulder_lift": float(pp["descend_lift"])},
            duration_s=1.5,
            stop_check=stop_check,
        )
        gripped, hold = close_with_feedback(
            arm, g, stop_check=stop_check, start_from=75.0
        )
        if gripped:
            break
        if attempt + 1 < attempts:
            say("The grasp slipped - trying once more.")
            arm.go_to({**pp["scan_pose"]}, duration_s=1.5, stop_check=stop_check)
            dets, _ = _detect_with_retry(
                detector, camera, object_name, duration_s=1.5, stop_check=stop_check
            )
            if not dets:
                break
    if not gripped:
        say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        arm.go_to_rest()
        return False

    # ---------------- LIFT + HANDOFF ----------------
    say(f"Got the {resolved} - bringing it to you.")
    lift_pose = {
        "shoulder_pan": arm.get_joints()["shoulder_pan"],
        "shoulder_lift": -80.0,
        "elbow_flex": 85.0,
        "wrist_flex": 50.0,
        "wrist_roll": 0.0,
    }
    arm.go_to(lift_pose, duration_s=2.0, stop_check=stop_check)
    if stopped():
        return False
    track_person_and_handoff(pp, arm, pose, mount_camera, lift_pose, stop_check)
    say("Done.")
    return True


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
    mount_cam = make_camera(cfg, "mount")
    try:
        ok = run_pick_place(
            cfg,
            arm,
            detector,
            pose,
            arm_cam,
            mount_cam,
            args.object,
            feedback=lambda m: print(f"{m}"),
        )
        print(f"\npick_place {'PASSED' if ok else 'FAILED'}")
        raise SystemExit(0 if ok else 1)
    finally:
        arm_cam.close()
        mount_cam.close()
        arm.close()


if __name__ == "__main__":
    main()
