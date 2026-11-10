# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MoveIt round-trip against a running bringup.

    # Terminal 1
    ros2 launch launch/so101_control.launch.py hardware_type:=mujoco

    # Terminal 2
    python -m vla_pipeline.robot.robot_node --client-test
"""

from __future__ import annotations

import argparse
import logging

from vla_pipeline.utils.config import load_config


def _run_client_test(cfg: dict) -> None:
    """MoveIt plans to ``ready`` and back to ``home``."""
    from vla_pipeline.robot.moveit_arm import MoveItArm

    arm = MoveItArm(cfg, sim=cfg["robot"].get("backend", "real") == "sim")
    try:
        assert arm.go_to_named("ready") == "arrived"
        assert arm.go_to_rest() == "arrived"
        print("ROS 2 round-trip PASSED")
    finally:
        arm.close()


def main() -> None:
    """CLI entry point: the MoveIt round-trip."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SO-101 MoveIt round-trip")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--client-test",
        action="store_true",
        required=True,
        help="plan to ready and back to home on a running bringup",
    )
    args = parser.parse_args()
    _run_client_test(load_config(args.config))


if __name__ == "__main__":
    main()
