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
from conftest import PAN, next_message, off

pytestmark = pytest.mark.sim
# The session's one bringup, named through conftest: a fixture imported here would be a second one.
launch = pytest.mark.launch(fixture=conftest.twin_bringup)


def _jaw(ros_node, name: str) -> float:
    """The gripper joint on the next ``/joint_states``. MoveIt's copy can still read the URDF zero."""
    from sensor_msgs.msg import JointState

    msg = next_message(ros_node, JointState, "/joint_states")
    return dict(zip(msg.name, msg.position))[name]


def _held(ros_node) -> dict:
    """The trajectory controller's reference: what it holds the arm at."""
    from control_msgs.msg import JointTrajectoryControllerState

    state = next_message(ros_node, JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state")
    return dict(zip(state.joint_names, state.reference.positions))


@launch
def test_a_named_state_is_planned_executed_and_reached(arm):
    assert arm.go_to_named("ready") == "arrived"
    assert off(arm.get_joints(), arm.named_state("ready")) < 0.01


@launch
@pytest.mark.parametrize("joint, distance", [(PAN, 1.5), ("shoulder_lift_joint", -0.2)])
def test_an_untimed_move_runs_at_the_joint_speed_limit(arm, ros_node, joint, distance):
    """An untimed move cruises at ``max_velocity`` x ``default_velocity_scaling_factor``
    from joint_limits.yaml, and the twin moves at that speed.

    The commanded speed is the trajectory controller's reference velocity.
    MoveIt time-parameterizes waypoints 0.1 s apart; between two cruise
    waypoints the controller's spline is exactly the limit, so the most common
    fast reference is the limit itself. The peak reference and peak measured
    speed may differ by up to the controller's existing 5% tolerance. The
    twin's measured speed (rad/s on its clock, from /joint_states) over that
    cruise matches the command. Both distances cruise for several waypoints."""
    from control_msgs.msg import JointTrajectoryControllerState
    from sensor_msgs.msg import JointState

    from vla_pipeline.robot.moveit_arm import moveit_configs, planning_scales

    limits = moveit_configs().joint_limits["robot_description_planning"]["joint_limits"][joint]
    velocity_scale, acceleration_scale = planning_scales()
    cruise = velocity_scale * float(limits["max_velocity"])
    ramp = cruise * cruise / (acceleration_scale * float(limits["max_acceleration"]))
    assert abs(distance) > ramp, "the move must be long enough to reach cruise"
    samples: list[tuple[float, float]] = []
    commanded: list[float] = []
    heard = (threading.Event(), threading.Event())

    def record(msg):
        position = dict(zip(msg.name, msg.position))
        if joint in position:
            samples.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, position[joint]))
            heard[0].set()

    def record_reference(msg):
        velocity = dict(zip(msg.joint_names, msg.reference.velocities))
        if joint in velocity:
            commanded.append(abs(velocity[joint]))
            heard[1].set()

    # The SRDF ready pose leaves room on both joints; home has shoulder_lift near its limit.
    assert arm.go_to_mtc(arm.named_state("ready")) == "arrived"
    start = arm.get_joints()[joint]
    subs = [
        ros_node.create_subscription(JointState, "/joint_states", record, 100),
        ros_node.create_subscription(
            JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state", record_reference, 100,
        ),
    ]
    try:
        assert all(event.wait(5.0) for event in heard), "a subscription heard nothing before the move"
        samples.clear()
        commanded.clear()
        assert arm.go_to({joint: start + distance}) == "arrived"
    finally:
        for sub in subs:
            ros_node.destroy_subscription(sub)
    try:
        import statistics

        # Measured speed over a 0.05 s window of the arm's clock, so one late sample does not read as a spike.
        speeds = []
        j = 0
        for i, (t, q) in enumerate(samples):
            while j < i and t - samples[j][0] > 0.05:
                j += 1
            if j < i and t - samples[j][0] >= 0.04:
                speeds.append(abs(q - samples[j][1]) / (t - samples[j][0]))
        fast_commanded = [v for v in commanded if v > 0.9 * cruise]
        fast_measured = [v for v in speeds if v > 0.9 * cruise]
        assert fast_commanded and fast_measured, f"{joint} never came within 10% of {cruise:.3f} rad/s"
        reference = statistics.median(fast_commanded)
        measured = statistics.median(fast_measured)
        peak_commanded = max(commanded)
        peak_measured = max(speeds)
        print(
            f"{joint} {distance:+.2f} rad cruise commanded {reference:.4f} measured {measured:.4f} "
            f"peak commanded {peak_commanded:.4f} measured {peak_measured:.4f} limit {cruise:.4f} rad/s"
        )
        assert reference == pytest.approx(cruise, rel=0.01), f"{joint} cruised at {reference:.4f} rad/s, not {cruise:.4f}"
        assert measured == pytest.approx(reference, rel=0.05), f"the twin ran {measured:.4f} rad/s for {reference:.4f} commanded"
        assert peak_commanded <= cruise * 1.05, (
            f"{joint} controller reference peaked at {peak_commanded:.4f} rad/s, over the "
            f"{cruise:.4f} rad/s limit plus controller tolerance"
        )
        assert peak_measured <= cruise * 1.05, (
            f"{joint} twin motion peaked at {peak_measured:.4f} rad/s, over the "
            f"{cruise:.4f} rad/s limit plus controller tolerance"
        )
    finally:
        assert arm.go_to({joint: start}) == "arrived"


@launch
def test_a_timed_keyframe_takes_its_duration_on_the_arms_clock(arm):
    """Dance and wave keyframes keep their timing: the controller plays the
    point over its ``duration_s``. The clock is the stamp of the joint state
    that is at the target, not whichever state this process has handled when
    the action result arrives."""
    from sensor_msgs.msg import JointState

    now = arm.get_joints()
    target = {PAN: now[PAN] + 0.3, "wrist_flex_joint": now["wrist_flex_joint"] - 0.3}
    started = arm.state_time()
    assert arm.go_to(target, duration_s=0.9) == "arrived"
    msg = next_message(
        arm.node, JointState, "/joint_states", lambda m: off(dict(zip(m.name, m.position)), target) < 0.02
    )
    done = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    took = done - started
    assert took >= 0.9, f"the keyframe ended {took:.3f} s after it was sent"


@launch
def test_a_streamed_setpoint_is_reached(arm, ros_node):
    from sensor_msgs.msg import JointState

    goal = arm.get_joints()[PAN] - 0.15
    arm.send_joints({PAN: goal})
    reached = next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - goal) < 0.01)
    assert abs(dict(zip(reached.name, reached.position))[PAN] - goal) < 0.01


@launch
def test_reset_discards_a_streamed_target(arm, ros_node):
    """A reset replaces both the physical controller target and Servo's own
    last command, so the next stream starts from home rather than from the
    stream before the reset. Ready is about 1 rad from home on the elbow; the
    joints a stream does not name sag about 0.01 rad under it."""
    from sensor_msgs.msg import JointState

    assert arm.go_to_named("ready") == "arrived"
    arm.send_joints({PAN: arm.get_joints()[PAN] + 0.3})
    arm.reset()
    home = arm.named_state("home")
    goal = home[PAN] + 0.1
    arm.send_joints({PAN: goal})
    reached = next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - goal) < 0.01)
    others = {name: value for name, value in home.items() if name != PAN}
    assert off(dict(zip(reached.name, reached.position)), others) < 0.05
    assert arm.go_to_named("ready") == "arrived"
    assert off(arm.get_joints(), arm.named_state("ready")) < 0.01


@launch
def test_a_streamed_step_runs_no_faster_than_the_joint_speed_limit(arm, ros_node):
    """A streamed step stays within ``max_velocity`` from joint_limits.yaml.

    MoveIt Servo applies that limit and the joint's ``max_acceleration``.
    Timed on the arm's clock, as the planned-move speed test measures it."""
    from sensor_msgs.msg import JointState

    from vla_pipeline.robot.moveit_arm import moveit_configs, planning_scales

    limits = moveit_configs().joint_limits["robot_description_planning"]["joint_limits"][PAN]
    cruise = planning_scales()[0] * float(limits["max_velocity"])
    distance = 1.0
    assert arm.go_to_named("ready") == "arrived"
    start = arm.get_joints()[PAN]
    goal = start + distance
    samples: list[tuple[float, float]] = []
    heard = threading.Event()

    def record(msg):
        position = dict(zip(msg.name, msg.position))
        if PAN in position:
            samples.append((msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, position[PAN]))
            heard.set()

    sub = ros_node.create_subscription(JointState, "/joint_states", record, 100)
    try:
        assert heard.wait(5.0), "/joint_states heard nothing before the step"
        started = arm.state_time()
        arm.send_joints({PAN: goal})
        reached = next_message(ros_node, JointState, "/joint_states", lambda m: abs(dict(zip(m.name, m.position))[PAN] - goal) < 0.01)
    finally:
        ros_node.destroy_subscription(sub)
    try:
        took = reached.header.stamp.sec + reached.header.stamp.nanosec * 1e-9 - started
        speeds = []
        j = 0
        for i, (t, q) in enumerate(samples):
            while j < i and t - samples[j][0] > 0.05:
                j += 1
            if j < i and t - samples[j][0] >= 0.04:
                speeds.append(abs(q - samples[j][1]) / (t - samples[j][0]))
        assert speeds, "no 0.05 s window of /joint_states during the step"
        peak = max(speeds)
        print(f"{PAN} {distance:+.2f} rad streamed in {took:.3f} s, peak {peak:.4f} limit {cruise:.4f} rad/s")
        assert took >= distance / cruise, f"the step took {took:.3f} s, under {distance / cruise:.3f} s at the limit"
        assert peak <= cruise * 1.05, f"the twin ran {peak:.4f} rad/s, over the {cruise:.4f} rad/s limit"
    finally:
        assert arm.go_to_named("ready") == "arrived"


@launch
def test_a_planned_move_right_after_streaming_waits_for_the_arm_to_rest(arm):
    """MoveIt rejects a start state the arm has already moved away from."""
    arm.send_joints({PAN: arm.get_joints()[PAN] + 0.3})
    assert arm.go_to_named("ready") == "arrived"
    assert off(arm.get_joints(), arm.named_state("ready")) < 0.01


@launch
def test_a_planned_move_after_a_setpoint_servo_halts_short_of(arm, ros_node, pipeline_cfg):
    """Mimic's elbow range ends past the URDF limit. Servo holds the elbow
    short of that setpoint and reports why on its status, so the setpoint is
    never reached, and the next planned move starts from where Servo holds the arm."""
    from moveit_msgs.msg import ServoStatus

    from vla_pipeline.robot.arm_interface import JOINT_LIMITS, radians_from_calibrated
    from vla_pipeline.robot.moveit_arm import PLANNING_GROUP

    elbow = "elbow_flex_joint"
    group = arm.robot.get_robot_model().get_joint_model_group(PLANNING_GROUP)
    upper = dict(zip(group.active_joint_model_names, group.active_joint_model_bounds))[elbow][0].max_position
    farthest = radians_from_calibrated(pipeline_cfg, {"elbow_flex": JOINT_LIMITS["elbow_flex"][1]})[elbow]
    assert farthest > upper, "mimic's elbow range no longer passes the URDF limit"
    assert arm.go_to_named("ready") == "arrived"
    arm.send_joints({elbow: farthest})
    # From ready, Servo reports DECELERATE_FOR_COLLISION and holds the elbow near 1.32 rad.
    held = next_message(ros_node, ServoStatus, "/servo_node/status", lambda m: m.code > ServoStatus.NO_WARNING)
    print(f"Servo holds the elbow stream: {held.code} {held.message}")
    assert arm.go_to_named("ready") == "arrived"
    assert off(arm.get_joints(), arm.named_state("ready")) < 0.01


@pytest.mark.parametrize("how", ["planned", "timed"])
@launch
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


@launch
def test_the_jaws_close_and_open_on_the_gripper_action(arm, ros_node):
    jaws = arm.gripper_joint
    close, open_ = arm.named_state("close")[jaws], arm.named_state("open")[jaws]
    assert arm.grip(close) is False, "the empty jaws stalled"
    # goal_tolerance on the gripper action is 0.01 rad; the yaml does not change it.
    assert abs(_jaw(ros_node, jaws) - close) < 0.01
    assert arm.grip(open_) is False
    assert abs(_jaw(ros_node, jaws) - open_) < 0.01


@launch
def test_rest_is_home_with_the_jaws_closed(arm, ros_node):
    assert arm.go_to_rest() == "arrived"
    joints = arm.get_joints()
    assert off(joints, arm.named_state("home")) < 0.01
    assert abs(_jaw(ros_node, arm.gripper_joint) - arm.named_state("close")[arm.gripper_joint]) < 0.01


@launch
def test_the_arm_feeds_the_gripper_current_meter_from_the_twins_effort(arm):
    """MuJoCo's gripper effort on ``/joint_states`` reaches the console through
    one call: ``on_cached_joints(joints, gripper_load)``, in milliamps.
    ``n/a`` on the bar means that call never carried a current."""
    from vla_pipeline.robot.moveit_arm import torque_nm_to_milliamps

    fed: list = []
    carried = threading.Event()

    def feed(joints: dict, gripper_load) -> None:
        fed.append((joints, gripper_load))
        if gripper_load is not None:
            carried.set()

    arm.on_cached_joints = feed
    try:
        assert carried.wait(5.0), f"{len(fed)} joint samples, none with a gripper current"
    finally:
        arm.on_cached_joints = None
    joints, load = next((j, g) for j, g in fed if g is not None)
    print(f"meter fed {load:.0f} mA with {len(joints)} joints")
    assert arm.gripper_joint in joints
    assert load >= torque_nm_to_milliamps(0.0)
