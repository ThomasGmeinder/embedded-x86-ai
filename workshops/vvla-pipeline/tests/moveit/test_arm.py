# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MoveItArm on the twin: named states, planned and timed moves, streaming,
stop and the gripper action, judged on /joint_states and the trajectory
controller's own state."""

from __future__ import annotations

import threading

import pytest

import conftest
from conftest import next_message

pytestmark = pytest.mark.sim
# The session's one bringup, named through conftest: a fixture imported here would be a second one.
LAUNCH = pytest.mark.launch(fixture=conftest.twin_bringup)
PAN = "shoulder_pan_joint"


def _off(joints: dict, target: dict) -> float:
    return max(abs(joints[j] - v) for j, v in target.items())


def _held(ros_node) -> dict:
    """The trajectory controller's reference: what it holds the arm at."""
    from control_msgs.msg import JointTrajectoryControllerState

    state = next_message(ros_node, JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state")
    return dict(zip(state.joint_names, state.reference.positions))


@LAUNCH
def test_a_named_state_is_planned_executed_and_reached(arm):
    assert arm.go_to_named("ready") == "arrived"
    assert _off(arm.get_joints(), arm.named_state("ready")) < 0.01


@LAUNCH
def test_a_timed_keyframe_takes_its_duration_on_the_arms_clock(arm):
    """Dance and wave keyframes keep their timing: the controller plays the
    point over its ``duration_s`` and reports when it has."""
    now = arm.get_joints()
    target = {PAN: now[PAN] + 0.3, "wrist_flex_joint": now["wrist_flex_joint"] - 0.3}
    started = arm.state_time()
    assert arm.go_to(target, duration_s=0.9) == "arrived"
    took = arm.state_time() - started
    assert took >= 0.9, f"the keyframe ended {took:.3f} s after it was sent"
    assert _off(arm.get_joints(), target) < 0.02


@LAUNCH
def test_a_streamed_setpoint_is_reached(arm, ros_node):
    from sensor_msgs.msg import JointState

    goal = arm.get_joints()[PAN] - 0.15
    arm.send_joints({PAN: goal})
    reached = next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - goal) < 0.01)
    assert abs(dict(zip(reached.name, reached.position))[PAN] - goal) < 0.01


@LAUNCH
def test_a_planned_move_right_after_streaming_waits_for_the_arm_to_rest(arm):
    """MoveIt rejects a start state the arm has already moved away from."""
    arm.send_joints({PAN: arm.get_joints()[PAN] + 0.3})
    assert arm.go_to_named("ready") == "arrived"
    assert _off(arm.get_joints(), arm.named_state("ready")) < 0.01


@pytest.mark.parametrize("how", ["planned", "timed"])
@LAUNCH
def test_a_stop_cancels_the_move_and_the_arm_holds_where_it_stopped(arm, ros_node, how):
    from sensor_msgs.msg import JointState

    assert arm.go_to_named("ready") == "arrived"
    start = arm.get_joints()[PAN]
    far = arm.named_state("scan_4")
    result: list = []
    move = (lambda: arm.go_to_named("scan_4")) if how == "planned" else (lambda: arm.go_to(far, duration_s=4.0))
    mover = threading.Thread(target=lambda: result.append(move()), daemon=True)
    mover.start()
    next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - start) > 0.1)
    arm.stop()
    mover.join()
    assert result == ["stopped"]
    held = _held(ros_node)[PAN]
    assert abs(held - far[PAN]) > 0.3, f"the controller holds pan at {held:.3f}: the stop came after the move ended"
    next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - held) < 0.01)


@LAUNCH
def test_the_jaws_close_and_open_on_the_gripper_action(arm):
    jaws = arm.gripper_joint
    close, open_ = arm.named_state("close")[jaws], arm.named_state("open")[jaws]
    assert arm.grip(close) is False, "the empty jaws stalled"
    assert abs(arm.get_joints()[jaws] - close) < 0.05
    assert arm.grip(open_) is False
    assert abs(arm.get_joints()[jaws] - open_) < 0.05


@LAUNCH
def test_rest_is_home_with_the_jaws_closed(arm):
    assert arm.go_to_rest() == "arrived"
    joints = arm.get_joints()
    assert _off(joints, arm.named_state("home")) < 0.01
    assert abs(joints[arm.gripper_joint] - arm.named_state("close")[arm.gripper_joint]) < 0.05
