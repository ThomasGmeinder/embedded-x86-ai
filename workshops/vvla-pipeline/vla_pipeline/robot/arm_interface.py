# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Arm constants in LeRobot calibrated degrees, their conversion to URDF radians, and ``make_arm``."""

from __future__ import annotations

import math

from so101_sim import MOTOR_NAMES

# The jaws' physical closed point, in calibrated degrees.
GRIPPER_CLOSED_FLOOR = 38.0

# Safe rest pose (degrees).
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -96.5,
    "elbow_flex": 95.9,
    "wrist_flex": 22.5,
    "wrist_roll": 86.8,
    "gripper": GRIPPER_CLOSED_FLOOR,
}

# Software joint limits (degrees) — a hard backstop every streamed command is
# clamped to. These are the mechanical ranges pulled in by 25%, so mimic stops
# short of the desks and monitors around the workshop rigs. Three bounds are
# relaxed just enough (rest + 2 deg) to keep REST_POSE reachable. The gripper
# span ends at the SRDF open state, 1.7 rad. The controller manager's
# enforce_command_limits stays at its default of false.
JOINT_LIMITS = {
    "shoulder_pan": (-86.0, 86.0),
    "shoulder_lift": (-98.5, 82.5),
    "elbow_flex": (-75.0, 98.0),
    "wrist_flex": (-82.5, 67.5),
    "wrist_roll": (-75.0, 89.0),  # NOT range-swept by LeRobot - keep tight
    "gripper": (GRIPPER_CLOSED_FLOOR, 95.0),
}


def clamp_joints(joints: dict, limits: dict | None = None) -> dict:
    """Clamp every joint to ``limits`` (default :data:`JOINT_LIMITS`; unknown names pass ±180)."""
    lims = JOINT_LIMITS if limits is None else limits
    out = {}
    for name, value in joints.items():
        lo, hi = lims.get(name, (-180.0, 180.0))
        out[name] = float(min(max(value, lo), hi))
    return out


def _maps(cfg: dict) -> tuple[dict, dict]:
    from so101_sim.config import load_sim_config

    sim = load_sim_config(cfg)
    return sim["joint_map"], sim["gripper_map"]


def _to_rad(joint_map: dict, gripper_map: dict, name: str, cal: float) -> float:
    if name == "gripper":
        closed_cal = float(gripper_map["closed_cal"])
        open_cal = float(gripper_map["open_cal"])
        span = float(gripper_map["open_rad"]) - float(gripper_map["closed_rad"])
        return float(gripper_map["closed_rad"]) + (float(cal) - closed_cal) * span / (open_cal - closed_cal)
    entry = joint_map.get(name, {})
    return float(entry.get("sign", 1.0)) * math.radians(float(cal) - float(entry.get("offset_deg", 0.0)))


def _to_cal(joint_map: dict, gripper_map: dict, name: str, rad: float) -> float:
    if name == "gripper":
        closed_rad = float(gripper_map["closed_rad"])
        span = float(gripper_map["open_rad"]) - closed_rad
        return float(gripper_map["closed_cal"]) + (float(rad) - closed_rad) / span * (
            float(gripper_map["open_cal"]) - float(gripper_map["closed_cal"])
        )
    entry = joint_map.get(name, {})
    sign = float(entry.get("sign", 1.0))
    return math.degrees(float(rad)) * sign + float(entry.get("offset_deg", 0.0))


def radians_from_calibrated(cfg: dict, pose: dict) -> dict:
    """``pose`` in calibrated degrees, as radians under the URDF joint names."""
    joint_map, gripper_map = _maps(cfg)
    return {f"{name}_joint": _to_rad(joint_map, gripper_map, name, value) for name, value in pose.items()}


def calibrated_from_radians(cfg: dict, joints: dict) -> dict:
    """URDF joint radians back to calibrated degrees, without the ``_joint`` suffix."""
    joint_map, gripper_map = _maps(cfg)
    return {
        name.removesuffix("_joint"): _to_cal(joint_map, gripper_map, name.removesuffix("_joint"), rad)
        for name, rad in joints.items()
    }


def make_arm(cfg: dict):
    """The arm is MoveIt. ``cfg['robot']`` chooses the sim or the real bringup."""
    from vla_pipeline.robot.moveit_arm import MoveItArm

    return MoveItArm(cfg, sim=cfg["robot"].get("backend", "real") == "sim")
