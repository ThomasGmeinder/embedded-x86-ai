# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODOs 2.7-2.9: MoveIt plans and the gripper action.

The pipeline's functions in ``vla_pipeline/robot/moveit_arm.py`` plan a named
state, plan a joint goal, and close the jaws. These stubs are those functions
with your body.

Reference: ``vla_pipeline/robot/moveit_arm.py``.
Notebook: ``notebooks/02_ros2/02_robot.ipynb``
"""

from __future__ import annotations


def go_to_named(robot, name: str) -> str:
    """Plan from the current state to the SRDF named state ``name`` and execute it.

    Return MoveIt's execution status, or ``"NO_PLAN"`` when there is no plan.
    """
    # >>> TODO 2.7: go_to_named - notebooks/02_ros2/02_robot.ipynb
    raise NotImplementedError(
        "TODO 2.7 (go_to_named): implement me - see notebooks/02_ros2/02_robot.ipynb")
    # <<< TODO 2.7


def go_to_joints(robot, joints: dict) -> str:
    """Plan to ``joints`` (radians, URDF names) and execute it.

    Joints left out of ``joints`` stay where they are. Return MoveIt's
    execution status, or ``"NO_PLAN"``.
    """
    # >>> TODO 2.8: go_to_joints - notebooks/02_ros2/02_robot.ipynb
    raise NotImplementedError(
        "TODO 2.8 (go_to_joints): implement me - see notebooks/02_ros2/02_robot.ipynb")
    # <<< TODO 2.8


def grip(node, position: float, max_effort: float) -> bool:
    """Drive the jaws to ``position`` (radians) with at most ``max_effort`` (N·m).

    ``node`` is already spinning on another thread. Return True when the jaws
    stall on something before the goal.
    """
    # >>> TODO 2.9: grip - notebooks/02_ros2/02_robot.ipynb
    raise NotImplementedError(
        "TODO 2.9 (grip): implement me - see notebooks/02_ros2/02_robot.ipynb")
    # <<< TODO 2.9
