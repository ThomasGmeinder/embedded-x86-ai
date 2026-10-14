# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``DirectArm`` on the MuJoCo digital twin instead of the Feetech bus.

The only thing swapped is the LeRobot ``SO101Follower``: :class:`SimArmClient`
gives ``DirectArm`` a :class:`so101_sim.SimFollower`, so every code path the
real transport runs - the ``max_relative_target`` clamp, direct-goal
``move_to``, the speed/acceleration register writes, coordinated speeds, the
``Present_Current`` effort probe and sign-magnitude decode, the park on
``close()`` - is exercised unchanged against the simulated servos.

Selected with ``robot.backend: sim`` (see ``make_arm`` and the ROS server
node). The world is per process (``so101_sim.shared_world``) so the arm and
an in-process sim arm camera see the same physics.
"""

from __future__ import annotations

import logging

from vla_pipeline.robot.arm_interface import (
    DIRECT_MODE_ACCEL_RAW,
    DIRECT_MODE_SPEED_RAW,
    GRIPPER_CLOSED_FLOOR,
    GRIPPER_TORQUE_LIMIT,
    REST_POSE,
    DirectArm,
)

logger = logging.getLogger(__name__)


def sim_world(cfg: dict):
    """The process-wide twin, configured from ``cfg['sim']`` with the pipeline rest pose."""
    from so101_sim import shared_world

    sim = dict(cfg.get("sim") or {})
    gm = dict(sim.get("gripper_map") or {})
    gm.setdefault("closed_cal", GRIPPER_CLOSED_FLOOR)
    sim["gripper_map"] = gm
    return shared_world({"sim": sim}, rest_pose=REST_POSE)


class SimArmClient(DirectArm):
    """The pipeline's ``DirectArm`` transport bound to the digital twin."""

    def __init__(self, cfg: dict, world=None):  # noqa: D107 - see class docstring
        from so101_sim import SimFollower

        r = cfg["robot"]
        max_relative_target = r.get("max_relative_target", 8.0)
        self.world = world or sim_world(cfg)
        self._robot = SimFollower(
            self.world,
            max_relative_target=max_relative_target,
            robot_id=r.get("robot_id", "sim"),
        )
        self._robot.config.disable_torque_on_disconnect = False
        self._robot.connect()
        logger.info("SO101 digital twin connected (MuJoCo, %s)", self.world.scene_path.name)
        self._effort_register = False
        self._coordinated_joints = []
        self._direct_goal = max_relative_target is None
        if self._direct_goal:
            self.set_motion_profile(
                speed=DIRECT_MODE_SPEED_RAW, acceleration=DIRECT_MODE_ACCEL_RAW
            )
        self.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)

    def grip_report(self) -> str:
        return self.world.grip_report()
