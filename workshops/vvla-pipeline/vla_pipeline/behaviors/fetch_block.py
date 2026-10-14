# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Fetch-block behavior: pick the block from its fixed location and hand it
to the user, who is located with YOLOv26s-pose on the **operator camera**.

State machine: APPROACH → GRASP → LIFT → FIND (person) → EXTEND → RELEASE →
HOME. The fixed block location is configured in joint space
(``behaviors.fetch_block.block_pose``).

This revision takes a :class:`~vla_pipeline.vision.camera.CameraClient`
instead of owning a ``cv2.VideoCapture``, and exports
:func:`find_person_pan` / :func:`track_person_and_handoff`. ``fetch_block``
hands the block to the person and lets go. ``pick_place`` holds its object
out and keeps the jaws closed.

Independent test
----------------
    python -m vla_pipeline.behaviors.fetch_block --dry-run
    python -m vla_pipeline.behaviors.fetch_block
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vla_pipeline.robot.arm_interface import (
    GRIPPER_TORQUE_LIMIT,
    JOINT_LIMITS,
    REST_POSE,
    ArmClient,
    DryRunArm,
    make_arm,
)
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_pose_npu import KP, PoseEstimator

logger = logging.getLogger(__name__)

# The lift runs this long at the close limit, so the block hangs from the
# fingers before the grip eases to the hold limit.
_BREAKAWAY_S = 0.3


def person_pan_target(pose: PoseEstimator, frame: np.ndarray) -> float | None:
    """Map the primary person's horizontal torso position to a pan angle."""
    person = pose.primary_person(frame)
    if person is None:
        return None
    kpts, _ = person
    anchors = [KP["l_shoulder"], KP["r_shoulder"], KP["nose"]]
    xs = [kpts[i, 0] for i in anchors if kpts[i, 2] > pose.kpt_threshold]
    if not xs:
        return None
    cx_norm = (float(np.mean(xs)) / frame.shape[1]) * 2.0 - 1.0  # [-1, 1]
    return float(
        np.clip(-cx_norm * 60.0, -80.0, 80.0)
    )  # mirror: person left → pan left


def find_person_pan(
    pose: PoseEstimator | None,
    operator_camera,
    arm: ArmClient,
    seek_s: float = 4.0,
    stop_check=None,
) -> tuple[float, bool]:
    """Track the person for up to ``seek_s`` and aim ``shoulder_pan`` at them.

    Returns ``(pan_deg, found)``; pan 0.0 / not-found when no camera or person.
    """
    pan, found = 0.0, False
    if pose is None or operator_camera is None:
        logger.info("[handoff] no operator camera - handing off straight ahead")
        return pan, found
    deadline = time.time() + seek_s
    while time.time() < deadline and not (stop_check and stop_check()):
        ok, frame = operator_camera.read()
        if not ok:
            time.sleep(0.05)
            continue
        target = person_pan_target(pose, frame)
        if target is not None:
            pan, found = target, True
            arm.send_joints({"shoulder_pan": pan})
        time.sleep(0.05)
    logger.info(
        "[handoff] person %s (pan=%.1f)", "found" if found else "NOT found", pan
    )
    return pan, found


def track_person_and_handoff(
    cfg_section: dict,
    arm: ArmClient,
    pose: PoseEstimator | None,
    operator_camera,
    lift_pose: dict,
    stop_check=None,
    still_holding=lambda: True,
) -> bool:
    """Shared EXTEND → RELEASE → HOME tail used by fetch_block and pick_place.

    Returns False, with the jaws left closed, when ``still_holding()`` finds
    them empty after the extend: there is nothing to hand over.
    """
    pan, _found = find_person_pan(pose, operator_camera, arm, stop_check=stop_check)

    logger.info("[handoff] EXTEND toward person")
    arm.go_to(
        {
            "shoulder_pan": pan,
            "shoulder_lift": float(lift_pose["shoulder_lift"])
            + float(cfg_section["handoff_reach"]),
            "elbow_flex": max(float(lift_pose["elbow_flex"]) - 40.0, 0.0),
            "wrist_flex": 20.0,
        },
        duration_s=2.5,
        stop_check=stop_check,
    )
    if not still_holding():
        return False

    logger.info("[handoff] RELEASE - take it!")
    time.sleep(min(float(cfg_section["release_timeout_s"]), 6.0) * 0.5)
    arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    arm.go_to({"gripper": JOINT_LIMITS["gripper"][1]}, duration_s=1.0)
    time.sleep(1.0)

    logger.info("[handoff] HOME")
    park_open(arm)
    return True


def park_open(arm: ArmClient, stop_check=None) -> None:
    """Go to the rest pose with the jaws wide open. Closing them on the way
    home can catch the object just let go of and drag it across the table."""
    arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
    arm.go_to({**REST_POSE, "gripper": JOINT_LIMITS["gripper"][1]}, duration_s=3.0, stop_check=stop_check)


def run_fetch_block(
    cfg: dict,
    arm: ArmClient,
    pose: PoseEstimator | None,
    operator_camera,
    stop_check=None,
    feedback=None,
) -> bool:
    """Execute the fetch sequence. Returns True once the block was handed over.

    ``block_pose`` is open loop - nothing looks for the block - so the grasp
    is checked twice, after closing and after the lift; if the block isn't
    held, the arm says so and parks instead of handing over empty jaws.
    """
    from vla_pipeline.behaviors.grip import close_with_feedback, ease_grip, holding

    fb = cfg["behaviors"]["fetch_block"]
    g = cfg["behaviors"]["grip"]
    block_pose = dict(fb["block_pose"])
    lift_pose = dict(fb["lift_pose"])
    say = feedback or (lambda m: logger.info("[fetch] %s", m))

    def stopped() -> bool:
        """Return True if a stop has been requested."""
        return bool(stop_check and stop_check())

    logger.info("[fetch] APPROACH block_pose (gripper open)")
    arm.go_to({**block_pose, "gripper": 70.0}, duration_s=2.5, stop_check=stop_check)
    if stopped():
        return False

    logger.info("[fetch] GRASP")
    gripped, hold = close_with_feedback(arm, g, stop_check=stop_check, start_from=70.0, ease=False)
    if stopped():
        return False
    if not (gripped and holding(arm, g, hold)):
        say("I couldn't get a grip on the block - is it in its usual spot?")
        arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
        arm.go_to({"gripper": 70.0}, duration_s=0.8)
        arm.go_to_rest()
        return False

    logger.info("[fetch] LIFT")
    arm.send_joints({**lift_pose, "gripper": hold})
    time.sleep(_BREAKAWAY_S)
    ease_grip(arm, g)
    arm.go_to({**lift_pose, "gripper": hold}, duration_s=2.5, stop_check=stop_check)
    if stopped():
        return False
    time.sleep(0.3)
    if not holding(arm, g, hold):
        say("I dropped the block.")
        park_open(arm)
        return False

    if not track_person_and_handoff(
        fb, arm, pose, operator_camera, lift_pose, stop_check, lambda: holding(arm, g, hold),
    ):
        if stopped():
            return False
        say("I dropped the block.")
        park_open(arm)
        return False
    return True


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run the fetch-block behavior standalone."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Fetch-block behavior - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-camera", action="store_true", help="skip the person-finding step"
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm: ArmClient = DryRunArm() if args.dry_run else make_arm(cfg)
    cam = None
    pose = None
    if not args.no_camera:
        from vla_pipeline.vision.camera import make_camera

        pose = PoseEstimator(cfg, device=args.device)
        cam = make_camera(cfg, "operator")
        if not cam.ok():
            logger.warning(
                "Operator camera unavailable - continuing without person finding."
            )
    try:
        ok = run_fetch_block(cfg, arm, pose, cam)
        print(f"\nfetch_block {'PASSED' if ok else 'FAILED (no block held, or stopped)'}")
    finally:
        if cam is not None:
            cam.close()
        arm.close()


if __name__ == "__main__":
    main()
