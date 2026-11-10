# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The SO-101 MuJoCo scene.

The running twin is ``mujoco_ros2_control`` from the bringup, which loads
``models/scene_vvla.xml``. This package holds that scene and the joint-name tables.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "load_sim_config",
    "MOTOR_NAMES",
    "MJCF_JOINTS",
    "MOTOR_SHORT",
    "PROP_BODIES",
]

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
# The MJCF joint and actuator each motor drives: ros2_so_arm's joint names.
MJCF_JOINTS = {m: f"{m}_joint" for m in MOTOR_NAMES}
# Free-joint bodies in models/scene_vvla.xml. The bringup tests check each one is published.
PROP_BODIES = ("sports_ball", "bottle", "book", "toothbrush")
# Labels for HUDs and log lines.
MOTOR_SHORT = {
    "shoulder_pan": "pan",
    "shoulder_lift": "lift",
    "elbow_flex": "elbow",
    "wrist_flex": "flex",
    "wrist_roll": "roll",
    "gripper": "grip",
}

_LAZY = {
    "load_sim_config": "so101_sim.config",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module 'so101_sim' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod), name)
