# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Fetch-block behavior: pick the block from its fixed location and hand it
to the user, who is located with YOLOv26s-pose on the **mount camera**.

State machine: APPROACH → GRASP → LIFT → FIND (person) → EXTEND → RELEASE →
HOME. The fixed block location is configured in joint space
(``behaviors.fetch_block.block_pose``).

This revision takes a :class:`~vla_pipeline.vision.camera.CameraClient`
instead of owning a ``cv2.VideoCapture``, and exports
:func:`find_person_pan` / :func:`track_person_and_handoff` for reuse by
``pick_place`` (both behaviors end with the same "hand it to the person"
sequence).

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

from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, make_arm
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_pose_npu import KP, PoseEstimator

logger = logging.getLogger(__name__)


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
    mount_camera,
    arm: ArmClient,
    seek_s: float = 4.0,
    stop_check=None,
) -> tuple[float, bool]:
    """Track the person for up to ``seek_s`` and aim ``shoulder_pan`` at them.

    Returns ``(pan_deg, found)``; pan 0.0 / not-found when no camera or person.
    """
    pan, found = 0.0, False
    if pose is None or mount_camera is None:
        logger.info("[handoff] no mount camera - handing off straight ahead")
        return pan, found
    deadline = time.time() + seek_s
    while time.time() < deadline and not (stop_check and stop_check()):
        ok, frame = mount_camera.read()
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
    mount_camera,
    lift_pose: dict,
    stop_check=None,
) -> None:
    """Shared EXTEND → RELEASE → HOME tail used by fetch_block and pick_place."""
    pan, _found = find_person_pan(pose, mount_camera, arm, stop_check=stop_check)

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

    logger.info("[handoff] RELEASE - take it!")
    time.sleep(min(float(cfg_section["release_timeout_s"]), 6.0) * 0.5)
    arm.go_to({"gripper": 75.0}, duration_s=1.0)
    time.sleep(1.0)

    logger.info("[handoff] HOME")
    arm.go_to_rest()


def run_fetch_block(
    cfg: dict,
    arm: ArmClient,
    pose: PoseEstimator | None,
    mount_camera,
    stop_check=None,
) -> bool:
    """Execute the fetch sequence. Returns True on (best-effort) success."""
    fb = cfg["behaviors"]["fetch_block"]
    block_pose = dict(fb["block_pose"])
    lift_pose = dict(fb["lift_pose"])

    def stopped() -> bool:
        """Return True if a stop has been requested."""
        return bool(stop_check and stop_check())

    logger.info("[fetch] APPROACH block_pose (gripper open)")
    arm.go_to({**block_pose, "gripper": 70.0}, duration_s=2.5, stop_check=stop_check)
    if stopped():
        return False

    logger.info("[fetch] GRASP")
    arm.go_to({"gripper": float(block_pose.get("gripper", 30.0))}, duration_s=1.2)
    time.sleep(0.3)

    logger.info("[fetch] LIFT")
    arm.go_to(lift_pose, duration_s=2.5, stop_check=stop_check)
    if stopped():
        return False

    track_person_and_handoff(fb, arm, pose, mount_camera, lift_pose, stop_check)
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
        cam = make_camera(cfg, "mount")
        if not cam.ok():
            logger.warning(
                "Mount camera unavailable - continuing without person finding."
            )
    try:
        ok = run_fetch_block(cfg, arm, pose, cam)
        print(f"\nfetch_block {'PASSED' if ok else 'INTERRUPTED'}")
    finally:
        if cam is not None:
            cam.close()
        arm.close()


if __name__ == "__main__":
    main()
