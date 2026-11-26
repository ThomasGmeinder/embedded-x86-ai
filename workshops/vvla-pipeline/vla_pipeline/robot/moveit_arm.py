# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MoveIt 2 on the SO-101: the arm every behavior drives, on ``so_arm101_moveit_config``.

``launch/so101_control.launch.py hardware_type:=mujoco`` (or ``real``) runs
ros2_so_arm's controllers, ``move_group`` and MoveIt Servo; :class:`MoveItArm`
plans and executes with its own MoveItPy (:func:`moveit_py`), plays timed
keyframes on the trajectory controller, streams setpoints through Servo, and
moves the jaws with the gripper action. The controllers are the ones in the config's
``trajectory_execution.yaml``. Joint positions are radians under the URDF
joint names. :func:`go_to_named`, :func:`go_to_joints` and :func:`grip` are
the workshop's TODOs 2.7-2.9.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import math
import threading
from typing import NamedTuple

import numpy as np

logger = logging.getLogger(__name__)


def clear_rcutils_error() -> None:
    """Drop a handled rcutils error so the next ROS call does not overwrite it.

    ``PipelinePlanner`` stores "this typesupport identifier is not supported"
    and returns. The next ``rcl`` call then stores "Expecting token or wildcard"
    on top of that, and librcutils prints the overwrite to stderr. Jupyter
    shows that line in red. The probe already failed closed; it is not a
    failure still waiting to be handled.
    """
    lib = getattr(clear_rcutils_error, "_lib", None)
    if lib is None:
        import ctypes

        lib = ctypes.CDLL("librcutils.so")
        clear_rcutils_error._lib = lib
    lib.rcutils_reset_error()


PLANNING_GROUP = "manipulator"
# MoveIt Servo's publish period (the shipped default). Far from its target a jog velocity is
# the position error divided by this; near it, the speed the joint can stop from (jog_velocities).
# The launch file sets the servo node's publish_period and its acceleration
# filter's update_period from this same value.
SERVO_PERIOD_S = 0.01
SERVO_NODE = "servo_node"
# moveit_servo CommandType::JOINT_JOG. The node leaves this unset until
# ~/switch_command_type is called, and a jog sent before that is ignored.
SERVO_JOINT_JOG = 0
GRIPPER_GROUP = "gripper"
TIP_LINK = "gripper_frame_link"
HOME_KEYFRAME = "home"
# MoveIt steps a Cartesian path by MaxEEFStep and stops at the first waypoint
# whose inverse kinematics fails (cartesian_interpolator.hpp). Task Constructor
# exposes that limit as CartesianPath.step_size, default 0.01 m. This grasp has
# no inverse kinematics at 0.01 m, so that single waypoint travels 0 m.
# CartesianPrecision only splits a segment that has left the straight line; it
# does not recover a failed inverse-kinematics waypoint. 0.004 m is the shortest
# step that still solves the grasp approach. A 4 mm straight raise from the
# bottle's place start has no inverse kinematics, so the place moves in joint
# space to the pose ComputeIK solves.
CARTESIAN_MIN_M = 0.004
# The raise after the pick: this much shoulder_lift lifts the grasp about
# 45 mm. A straight 30 mm Cartesian raise stops at 3.8 mm for the bottle,
# whose grasp has wrist_flex at its stop.
RAISE_SHOULDER_RAD = -0.2


def grasp_frame(size):
    """The object's centre in :data:`TIP_LINK` for a pinch.

    The jaw closes along X. ``fixed_jaw_box3``'s surface is at x=-0.0019 m
    (its centre at -0.0008 m, radius 0.0011 m), z=0.017 m. That is also the
    deepest point of the URDF ``gripper_link`` collision mesh, which does not
    move when the jaws open. The centre is one half-width from that surface
    toward the moving jaw, at that fingertip height. The orientation is the
    tip joint's half turn about Y.
    """
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.orientation.y = 1.0
    pose.orientation.w = 0.0
    pose.position.x = -0.0019 - 0.5 * float(size[0])
    pose.position.z = 0.017
    return pose


# The finger pads in the twin's MJCF: the URDF has no pad geometry or friction.
JAW_MODEL = "so101.xml"
FIXED_PAD = "fixed_jaw_box4"
MOVING_PAD = "moving_jaw_box2"


@functools.lru_cache(maxsize=1)
def _jaw_model():
    import mujoco
    from so101_sim.config import MODELS_DIR

    return mujoco.MjModel.from_xml_path(str(MODELS_DIR / JAW_MODEL))


# gripper_frame_joint in so_arm101_macro.xacro: xyz="-0.0098 0 -0.086" rpy="0 pi 0".
# Ry(pi) sends a gripper-link point into the grasp frame. Tip z is the object's height.
_TIP_ORIGIN_Z = -0.086


def _tip_z(body_z: float) -> float:
    """Gripper-link z, in the grasp frame."""
    return -(float(body_z) - _TIP_ORIGIN_Z)


class JawContact(NamedTuple):
    """The jaw where the pads meet one object side."""

    q: float
    angle: float
    fixed_z: tuple[float, float]  # the fixed pad's face, in the grasp frame
    moving_z: tuple[float, float]  # the moving pad's face, in the grasp frame


def jaw_contact(width: float) -> JawContact | None:
    """Where the finger pads close on ``width`` metres, or ``None``.

    The pads are :data:`FIXED_PAD` on the gripper and :data:`MOVING_PAD` on
    the jaw, both thin along their X. The fixed pad's +X face meets the
    object, and the moving pad's -X face swings about the jaw hinge (Z). The
    gap along the gripper's X is ``A cos q + B sin q + C``, solved on the
    rising side. The pad angle is the angle between the two faces: zero is
    parallel, and the gap widens toward the fingertips as it grows. Each
    face's z interval is that face in the grasp frame, which is the object's
    height.
    """
    import mujoco

    m = _jaw_model()
    fixed, moving = m.geom(FIXED_PAD), m.geom(MOVING_PAD)
    jaw = m.body(int(moving.bodyid[0]))
    joint = m.joint(int(jaw.jntadr[0]))
    if int(jaw.parentid[0]) != int(fixed.bodyid[0]) or not np.allclose(joint.axis, (0.0, 0.0, 1.0)) or np.any(joint.pos):
        raise RuntimeError(f"{JAW_MODEL}: the jaw is not a Z hinge at its body origin on the fixed pad's body")
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, jaw.quat)
    axes = rotation.reshape(3, 3)
    u = axes[0]  # the jaw frame's components of the gripper's X
    face = np.array(moving.pos) - np.array((moving.size[0], 0.0, 0.0))
    fixed_x = float(fixed.pos[0] + fixed.size[0])
    a = u[0] * face[0] + u[1] * face[1]
    b = u[1] * face[0] - u[0] * face[1]
    c = float(jaw.pos[0] + u[2] * face[2])
    reach = math.hypot(a, b)
    want = (float(width) + fixed_x - c) / reach
    if abs(want) > 1.0:
        return None
    q = math.atan2(b, a) - math.acos(want)
    lo, hi = (float(v) for v in joint.range)
    if not lo <= q <= hi:
        return None
    cosine, sine = math.cos(q), math.sin(q)

    def parent_z(point) -> float:
        turned = np.array((cosine * point[0] - sine * point[1], sine * point[0] + cosine * point[1], point[2]))
        return float((np.array(jaw.pos) + axes @ turned)[2])

    def face_z(centre, half_y: float, half_z: float, at) -> tuple[float, float]:
        zs = [
            _tip_z(at(centre + np.array((0.0, y, z))))
            for y in (-half_y, half_y)
            for z in (-half_z, half_z)
        ]
        return min(zs), max(zs)

    fixed_centre = np.array((fixed_x, float(fixed.pos[1]), float(fixed.pos[2])))
    return JawContact(
        q,
        math.acos(max(-1.0, min(1.0, u[0] * cosine + u[1] * sine))),
        face_z(fixed_centre, float(fixed.size[1]), float(fixed.size[2]), lambda point: float(point[2])),
        face_z(face, float(moving.size[1]), float(moving.size[2]), parent_z),
    )


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return min(a[1], b[1]) > max(a[0], b[0])


def jaw_grips(width: float, open_rad: float, height: float) -> bool:
    """Whether the jaws, opened to ``open_rad``, hold a side ``width`` by ``height`` metres.

    The pads must meet it no wider than open. Two pads hold only while the
    angle between them is within twice the friction angle (the two-contact
    friction-cone condition); the friction is the pads', whose MJCF priority
    makes it the contact's. Both faces have to overlap the object's side.
    That side is ``height`` tall and centred at :func:`grasp_frame`'s z.
    """
    contact = jaw_contact(width)
    if contact is None or contact.q > float(open_rad):
        return False
    mu = float(_jaw_model().geom(MOVING_PAD).friction[0])
    if contact.angle > 2.0 * math.atan(mu):
        return False
    centre = grasp_frame((width, 0.0, height)).position.z
    side = (centre - 0.5 * float(height), centre + 0.5 * float(height))
    return _overlaps(contact.fixed_z, side) and _overlaps(contact.moving_z, side)


def grasp_box(size, yaw: float, open_rad: float):
    """``(size, yaw)`` turned so the jaws close across the box's X, or ``None``.

    The jaws may close across a horizontal side only where :func:`jaw_grips`
    holds. The narrower side that passes goes on X: :func:`grasp_frame`
    centres the object on X's width, and Task Constructor's grasp generator
    has one IK frame. A box whose sides both fail is too wide to pick.
    """
    x, y, z = (float(v) for v in size)
    fits_x, fits_y = jaw_grips(x, open_rad, z), jaw_grips(y, open_rad, z)
    if fits_y and (not fits_x or y < x):
        return (y, x, z), float(yaw) + math.pi / 2
    if fits_x:
        return (x, y, z), float(yaw)
    return None


RESET_SERVICE = "/mujoco_ros2_control_node/reset_world"
SWITCH_SERVICE = "/controller_manager/switch_controller"
# A controller accepts a goal and reports its result within this beyond the
# goal's own duration; it only catches a controller that stopped answering.
REPLY_S = 5.0
# The arm waits this long for the controllers at start: a spawner whose
# list_controllers reply DDS drops tries again 10 s later; this allows two.
BRING_UP_S = 30.0
# STS3215 datasheet line, the same numbers as the Gemma 4 branch's CurrentModel.
# The meter draws milliamps and is full at stall.
_IDLE_A = 0.006
_RATED_NM = 0.49
_RATED_A = 0.65
_STALL_A = 2.5
_STALL_NM = 1.91
GRIPPER_STALL_MA = _STALL_A * 1000.0
# The contact tick. 120 Present_Current counts at the datasheet 6.5 mA per count.
GRIPPER_CONTACT_MA = 120.0 * 6.5


def torque_nm_to_milliamps(torque_nm: float) -> float:
    """N·m to gripper current in milliamps, along the STS3215 datasheet line."""
    tau = float(torque_nm)
    if tau <= _RATED_NM:
        amps = _IDLE_A + tau / _RATED_NM * (_RATED_A - _IDLE_A)
    else:
        amps = _RATED_A + (tau - _RATED_NM) / (_STALL_NM - _RATED_NM) * (_STALL_A - _RATED_A)
    return amps * 1000.0


def _gripper_current(msg, joint: str) -> float | None:
    """Gripper current in milliamps from this joint-state sample, or None when it has no effort."""
    try:
        index = list(msg.name).index(joint)
    except ValueError:
        return None
    if index >= len(msg.effort):
        return None
    effort = float(msg.effort[index])
    if effort != effort:
        return None
    return torque_nm_to_milliamps(abs(effort))
# MoveIt's execution status, as the arm reports a move.
# A plan starts from a still arm: MoveIt rejects a start state more than its
# allowed_start_tolerance (0.01 rad) from where the arm is when it executes.
STILL_RAD_S = 0.01
# The share of a joint's max_acceleration a streamed approach brakes with (see jog_velocity).
STREAM_BRAKE = 0.5
# The share of its max_velocity the fastest joint of a stream is asked for. The arm's speed
# swings around what Servo commands: asked for the limit itself, the twin cruised near
# 2.0 rad/s with peaks of 2.24 to 2.29 against a limit of 2.14 (2026-11-25/26). The margin
# keeps the peaks inside it.
STREAM_SPEED = 0.9


def jog_velocity(error: float, max_acceleration: float) -> float:
    """The jog velocity for a joint that is ``error`` rad short of its streamed target,
    before the speed limit (:func:`jog_velocities` applies that).

    It is the velocity that closes the error in one Servo period, or, when
    that is less, the speed the joint can still stop from: Servo also limits
    acceleration, so a joint asked for full speed until the last period
    arrives too fast, runs past the target and comes back (a 1 rad step took
    1.7 to 5.5 s to settle, 2026-11-26). Braking at ``STREAM_BRAKE`` of
    ``max_acceleration`` from distance ``error`` allows
    ``sqrt(2 * STREAM_BRAKE * max_acceleration * |error|)``.
    """
    reach = abs(error) / SERVO_PERIOD_S
    stop = math.sqrt(2.0 * STREAM_BRAKE * max_acceleration * abs(error))
    return math.copysign(min(reach, stop), error)


def jog_velocities(
    errors: dict[str, float], max_acceleration: dict[str, float], max_velocity: dict[str, float],
) -> dict[str, float]:
    """The jog velocities for the joints of one stream, by joint name.

    Each joint's :func:`jog_velocity`, all multiplied by one factor so that
    no joint is asked for more than ``STREAM_SPEED`` of its speed limit. One
    factor for all, as Servo scales a command that is over a limit: the
    joints keep their proportions and arrive together. Limiting each joint
    on its own let every joint run at its own pace; mimic then froze at a
    joint's end stop in 9 of 12 runs of its range test (2026-11-26).
    """
    wanted = {name: jog_velocity(error, max_acceleration[name]) for name, error in errors.items()}
    factor = min(
        (STREAM_SPEED * max_velocity[name] / abs(speed) for name, speed in wanted.items() if speed),
        default=1.0,
    )
    factor = min(1.0, factor)
    return {name: speed * factor for name, speed in wanted.items()}
# The gripper action's goal_tolerance. The controller yaml leaves the default.
# A repeat command inside this band is the same target, and a jaw reading
# inside it is still on the model limit.
JAW_GOAL_RAD = 0.01


class Controller(NamedTuple):
    name: str
    action: str
    joints: list


_CONFIGS = None
_CONFIGS_LOCK = threading.Lock()


def moveit_configs():
    """so_arm101_moveit_config, loaded once. Loading runs xacro, which is not thread-safe."""
    global _CONFIGS
    with _CONFIGS_LOCK:
        if _CONFIGS is None:
            from so_arm_utils.launch_utils import MoveItConfigs

            _CONFIGS = MoveItConfigs("so_arm101")
        return _CONFIGS


def planning_scales(velocity_scale: float | None = None, acceleration_scale: float | None = None, limits: dict | None = None) -> tuple[float, float]:
    """The global speed, as fractions of each joint's ``max_velocity`` and ``max_acceleration``.

    Both defaults are ``default_velocity_scaling_factor`` and
    ``default_acceleration_scaling_factor`` in ``joint_limits.yaml`` — the file
    MoveIt loads. A move passes a fraction to override that one request.
    ``limits`` is that file's mapping; the default reads it from the MoveIt config.
    """
    doc = limits if limits is not None else moveit_configs().joint_limits["robot_description_planning"]
    velocity = doc["default_velocity_scaling_factor"] if velocity_scale is None else velocity_scale
    acceleration = doc["default_acceleration_scaling_factor"] if acceleration_scale is None else acceleration_scale
    return float(velocity), float(acceleration)


def _pipeline(node):
    """OMPL for one Task Constructor stage.

    The workspace is the cube ``ValidateWorkspaceBounds`` substitutes when the
    request leaves it at zero, so that adapter does not warn and the volume
    does not change.
    """
    from moveit.task_constructor import core
    from moveit_msgs.msg import WorkspaceParameters

    planner = _at_speed(core.PipelinePlanner(node, "ompl"))
    bounds = WorkspaceParameters()
    bounds.header.frame_id = "base_link"
    half = 5.0e11
    bounds.min_corner.x = bounds.min_corner.y = bounds.min_corner.z = -half
    bounds.max_corner.x = bounds.max_corner.y = bounds.max_corner.z = half
    planner.workspace_parameters = bounds
    # The planner's typesupport probe stays set. The next rcl call would overwrite it.
    clear_rcutils_error()
    return planner


def _at_speed(solver, velocity_scale: float | None = None, acceleration_scale: float | None = None):
    """Set a Task Constructor solver's scales from :func:`planning_scales` and return it."""
    velocity, acceleration = planning_scales(velocity_scale, acceleration_scale)
    solver.max_velocity_scaling_factor = velocity
    solver.max_acceleration_scaling_factor = acceleration
    return solver


def _cartesian_move(name: str, direction, min_m: float, max_m: float):
    """A Cartesian ``MoveRelative`` of :data:`TIP_LINK`, stepped at :data:`CARTESIAN_MIN_M`.

    ``direction`` is a ``Vector3Stamped``. The distance is a range, so the
    solver's ``min_fraction`` stays at its default.
    """
    from geometry_msgs.msg import PoseStamped
    from moveit.task_constructor import core, stages

    path = _at_speed(core.CartesianPath())
    path.step_size = CARTESIAN_MIN_M
    move = stages.MoveRelative(name, path)
    move.group = PLANNING_GROUP
    frame = PoseStamped()
    frame.header.frame_id = TIP_LINK
    frame.pose.orientation.w = 1.0
    move.ik_frame = frame
    move.setDirection(direction)
    move.min_distance = float(min_m)
    move.max_distance = float(max_m)
    return move


def controllers() -> dict:
    """``{controller: action}`` for every controller MoveIt executes on."""
    return {c.name: c.action for c in _controllers().values()}


def _controllers() -> dict:
    """``{action type: Controller}`` from the config's trajectory_execution.yaml."""
    manager = moveit_configs().trajectory_execution["moveit_simple_controller_manager"]
    return {
        manager[name]["type"]: Controller(name, f"/{name}/{manager[name]['action_ns']}", list(manager[name]["joints"]))
        for name in manager["controller_names"]
    }



def _result(client, name: str, goal, timeout_s: float, on_handle=None):
    """Send ``goal`` on the action ``client`` (``name``) and return ``(status, result)``;
    ``on_handle`` receives the goal handle (to cancel it) once the goal is accepted."""
    accepted = threading.Event()
    future = client.send_goal_async(goal)
    future.add_done_callback(lambda _f: accepted.set())
    if not accepted.wait(REPLY_S):
        raise RuntimeError(f"{name} did not accept the goal within {REPLY_S} s")
    handle = future.result()
    if not handle.accepted:
        raise RuntimeError(f"{name} rejected the goal")
    if on_handle is not None:
        on_handle(handle)
    done = threading.Event()
    result = handle.get_result_async()
    result.add_done_callback(lambda _f: done.set())
    if not done.wait(timeout_s + REPLY_S):
        raise RuntimeError(f"{name} reported no result within {timeout_s + REPLY_S:.1f} s")
    return result.result().status, result.result().result


def _execute(robot, arm, velocity_scale: float | None = None, acceleration_scale: float | None = None) -> str:
    """Plan with the global scales (or this call's override) and execute."""
    from moveit.planning import PlanRequestParameters

    # "" is moveit_cpp.yaml's top-level plan_request_params (pipeline, planner, time).
    params = PlanRequestParameters(robot, "")
    velocity, acceleration = planning_scales(velocity_scale, acceleration_scale)
    params.max_velocity_scaling_factor = velocity
    params.max_acceleration_scaling_factor = acceleration
    plan = arm.plan(single_plan_parameters=params)
    if not plan:
        return "NO_PLAN"
    return robot.execute(plan.trajectory, controllers=[]).status


def go_to_named(robot, name: str) -> str:
    """Plan the arm from where it is to the SRDF named state ``name`` and execute it.

    Returns MoveIt's execution status (``"SUCCEEDED"``; ``"PREEMPTED"`` when a
    stop cancelled it), or ``"NO_PLAN"``. The workshop's TODO 2.7.
    """
    arm = robot.get_planning_component(PLANNING_GROUP)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name=name)
    return _execute(robot, arm)


def go_to_joints(robot, joints: dict) -> str:
    """Plan the arm to ``joints`` (radians by URDF joint name; joints left out
    stay where they are) and execute it. Returns MoveIt's execution status, or
    ``"NO_PLAN"``. The workshop's TODO 2.8.
    """
    from moveit.core.robot_state import RobotState

    now = current_joints(robot)
    arm = robot.get_planning_component(PLANNING_GROUP)
    arm.set_start_state_to_current_state()
    goal = RobotState(robot.get_robot_model())
    goal.joint_positions = {**now, **joints}
    goal.update()
    arm.set_goal_state(robot_state=goal)
    return _execute(robot, arm)


def _gripper_action(node):
    """The one gripper action client for ``node``.

    ``node`` is already being spun. Destroying the client after a goal returns
    races that thread's status take (``action client pointer is invalid``) and
    the thread dies, so every later goal waits out its accept bound. The client
    stays until the node does.
    """
    from control_msgs.action import ParallelGripperCommand
    from rclpy.action import ActionClient

    gripper = _controllers()["ParallelGripperCommand"]
    client = getattr(node, "_so101_gripper_client", None)
    if client is None:
        client = ActionClient(node, ParallelGripperCommand, gripper.action)
        node._so101_gripper_client = client
    if not client.wait_for_server(timeout_sec=REPLY_S):
        raise RuntimeError(f"{gripper.action} is not available")
    return client, gripper


def grip(node, position: float, max_effort: float) -> bool:
    """Drive the jaws to ``position`` (radians) pushing with at most
    ``max_effort`` (N·m) on the gripper action; True when they stalled on
    something before reaching it. ``node`` is spun on another thread.
    The workshop's TODO 2.9.
    """
    from control_msgs.action import ParallelGripperCommand

    client, gripper = _gripper_action(node)
    goal = ParallelGripperCommand.Goal()
    goal.command.name = list(gripper.joints)
    goal.command.position = [float(position)]
    goal.command.effort = [float(max_effort)]
    _status, result = _result(client, gripper.action, goal, REPLY_S)
    return bool(result.stalled)


def _arm_now(sim: bool, timeout_s: float):
    """This moment on the arm's clock: the twin's next ``/clock``, or the system clock."""
    import queue
    import time

    import rclpy
    from rclpy.clock import ClockType
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.time import Time
    from rosgraph_msgs.msg import Clock

    if not sim:
        return Time(nanoseconds=time.time_ns(), clock_type=ClockType.ROS_TIME)
    # rclpy finalizes rcl logging when its last logging context shuts down. That
    # drops every rclcpp node's /rosout entry, MoveIt's included, and each MoveIt
    # log then prints rcutils' "Failed to get logger entry" banner.
    context = rclpy.Context()
    context.init(initialize_logging=False)
    try:
        clear_rcutils_error()
        node = rclpy.create_node("moveit_arm_clock", context=context)
        ticks: queue.Queue = queue.Queue()
        node.create_subscription(Clock, "/clock", ticks.put, 1)
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        try:
            return Time.from_msg(ticks.get(timeout=timeout_s).clock)
        except queue.Empty:
            raise RuntimeError(f"no /clock from the twin within {timeout_s} s") from None
        finally:
            executor.shutdown()
            spinner.join()
            node.destroy_node()
    finally:
        context.shutdown()


def _sensors_3d() -> dict:
    """Occupancy-map parameters for this arm: one sensor the monitor skips.

    The monitor errors when ``sensors`` is missing or empty, and when a named
    sensor has no plugin. A plugin whose name starts with ``~`` is skipped.
    The resolution is the value it otherwise substitutes, and the frame is the
    URDF root link. The file is ``so_arm101_moveit_config``'s ``sensors_3d.yaml``.
    """
    from pathlib import Path

    import yaml
    from ament_index_python.packages import get_package_share_directory

    path = Path(get_package_share_directory("so_arm101_moveit_config")) / "config" / "sensors_3d.yaml"
    loaded = yaml.safe_load(path.read_text())
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{path} did not load a parameter map")
    return loaded


def _wrist_optical_frame() -> str:
    """The wrist camera frame ``config/mujoco_plugins.yaml`` names."""
    from pathlib import Path

    import yaml

    path = Path(__file__).resolve().parents[2] / "config" / "mujoco_plugins.yaml"
    loaded = yaml.safe_load(path.read_text())
    try:
        frame = loaded["/**"]["ros__parameters"]["mujoco_plugins"]["cameras"]["wrist_cam"]["frame_name"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"{path} has no wrist camera frame") from exc
    if not isinstance(frame, str) or not frame:
        raise RuntimeError(f"{path} has no wrist camera frame")
    return frame


def _sim_bringup(timeout_s: float) -> None:
    """Return once ``/joint_states`` has arrived and the wrist camera frame is in the ``world`` tree.

    The planning scene copies every TF frame that is not a robot link, and warns
    while that camera frame is a tree of its own. The state monitor warns once a
    second until the first joint state. Each wait returns when its message
    arrives. ``timeout_s`` is the error when that message does not.
    """
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.parameter import Parameter
    from rclpy.time import Time
    from sensor_msgs.msg import JointState
    from tf2_ros import Buffer, TransformListener

    frame = _wrist_optical_frame()
    context = rclpy.Context()
    context.init(initialize_logging=False)
    try:
        clear_rcutils_error()
        node = rclpy.create_node(
            "moveit_arm_bringup",
            context=context,
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, True)],
        )
        state = threading.Event()
        node.create_subscription(JointState, "/joint_states", lambda _msg: state.set(), 10)
        buffer = Buffer()
        listener = TransformListener(buffer, node, spin_thread=False)
        tf_done = threading.Event()
        future = buffer.wait_for_transform_async("world", frame, Time())
        future.add_done_callback(lambda _done: tf_done.set())
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        spinner = threading.Thread(target=executor.spin, daemon=True)
        spinner.start()
        try:
            if not state.wait(timeout_s):
                raise RuntimeError(f"no /joint_states within {timeout_s} s")
            if not tf_done.wait(timeout_s):
                raise RuntimeError(f"{frame} is not connected to world within {timeout_s} s")
            error = future.exception()
            if error is not None:
                raise RuntimeError(f"{frame} is not connected to world") from error
        finally:
            listener.unregister()
            executor.shutdown()
            spinner.join()
            node.destroy_node()
    finally:
        context.shutdown()


def moveit_py(node_name: str, *, sim: bool, state_timeout_s: float = BRING_UP_S):
    """MoveItPy on so_arm101_moveit_config, on the twin's clock when ``sim``,
    returned once MoveIt holds a state of the arm from now or later.

    The wait is :data:`BRING_UP_S`: the spawner's first ``list_controllers``
    reply can be dropped, and the controller then comes up on the retry.
    On the twin, that same wait first sees a joint state and the wrist camera
    frame in the ``world`` tree, so MoveIt does not warn about a bringup that
    is still starting.

    The parameters go in as a ``/**`` parameter file: MoveItPy's ``config_dict``
    keys them by node name, and with ``use_sim_time`` that aborts on /clock's QoS
    override parameters (moveit/moveit2#2940).
    """
    import moveit.core  # registers the RobotModel caster before MoveItPy returns one
    from moveit.planning import MoveItPy
    from moveit.utils import create_params_file_from_dict

    if sim:
        _sim_bringup(state_timeout_s)
    config = moveit_configs().to_dict() | {"use_sim_time": sim} | _sensors_3d()
    robot = MoveItPy(node_name=node_name, launch_params_filepaths=[create_params_file_from_dict(config, "/**")])
    now = _arm_now(sim, state_timeout_s)
    if not robot.get_planning_scene_monitor().wait_for_current_robot_state(now, state_timeout_s):
        robot.shutdown()
        raise RuntimeError(f"MoveIt received no state of the arm within {state_timeout_s} s")
    return robot


def current_joints(robot) -> dict:
    """Every joint's position now: the first state MoveIt receives that is at
    least as new as this moment.

    A planning-scene read right after :func:`moveit_py` can still hold the
    default zeros; the start state waits for the current one.
    """
    arm = robot.get_planning_component(PLANNING_GROUP)
    arm.set_start_state_to_current_state()
    state = arm.get_start_state()
    if state is None:
        raise RuntimeError("MoveIt received no current state of the arm")
    return dict(state.joint_positions)


def _call(node, srv_type, name: str, request, timeout_s: float):
    """One service call from ``node``, which an executor spins on another thread."""
    # A MoveIt publish leaves the typesupport probe set. Resolving ``name``
    # then stores a lexer miss on top of it, and librcutils prints that.
    clear_rcutils_error()
    client = node.create_client(srv_type, name)
    try:
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"{name} is not available")
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(timeout_s):
            raise RuntimeError(f"{name} did not answer within {timeout_s} s")
        return future.result()
    finally:
        node.destroy_client(client)


class MoveItArm:
    """The arm behaviors drive. Joint positions are radians under the URDF joint names.

    - :meth:`go_to` without a duration, :meth:`go_to_named` and
      :meth:`go_to_rest` are MoveIt plans, executed;
    - :meth:`go_to` with ``duration_s`` plays one timed keyframe on the
      trajectory controller's action;
    - :meth:`send_joints` streams a setpoint through MoveIt Servo;
    - :meth:`grip` moves the jaws on the gripper action;
    - :meth:`stop` cancels whatever moves the arm; a cancelled move reports ``"stopped"``.

    Moves report ``"arrived"``, ``"stopped"`` or ``"blocked"``.
    """

    def __init__(self, cfg: dict, node_name: str = "vla_arm", *, sim: bool | None = None):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from control_msgs.msg import JointJog
        from sensor_msgs.msg import JointState

        robot_cfg = cfg["robot"]
        self.sim = robot_cfg.get("backend") == "sim" if sim is None else sim
        self.robot = moveit_py(f"{node_name}_moveit", sim=self.sim)
        model = self.robot.get_robot_model()
        self.arm_joints = list(model.get_joint_model_group(PLANNING_GROUP).active_joint_model_names)
        by_type = _controllers()
        self._trajectory_controller = by_type["FollowJointTrajectory"]
        self.gripper_joint = by_type["ParallelGripperCommand"].joints[0]
        self._cond = threading.Condition()
        self._state = None  # (stamp seconds, {joint: position})
        self._speeds: dict = {}  # {joint: velocity} of the latest state
        self._goal = None  # the trajectory goal a stop cancels
        self._stream_goal = None  # the last streamed arm setpoint, until the controller has taken it
        self._stream_done = False  # the controller's reference reached that setpoint
        self._servo_target = None  # the joints the jog timer is steering toward
        self._servo_timer = None
        self._servo_client = None
        self._servo_type = None  # ~/switch_command_type
        self._servo_joint_jog = False
        self._servo_limited = False  # Servo's last ~/status slows or holds the stream short of its setpoint
        # Each arm joint's limits from joint_limits.yaml, the file Servo limits with.
        limits = moveit_configs().joint_limits["robot_description_planning"]["joint_limits"]
        self._max_acceleration = {name: float(limits[name]["max_acceleration"]) for name in self.arm_joints}
        self._max_velocity = {name: float(limits[name]["max_velocity"]) for name in self.arm_joints}
        self._jaw_sent = None
        self._attach_link = None  # the SRDF end effector's parent link, read once
        self._stops = 0
        self._active: set[str] = set()  # controllers the manager's activity says are active
        # Called with (joint positions, gripper current in mA or None) for every
        # /joint_states, outside the state lock. The console's only feed.
        self.on_cached_joints = None
        self._gripper_load = None  # gripper current in mA, or None

        # The node keeps the system clock: it times nothing itself (waits use the
        # states' stamps), and on the twin's clock it would take every /clock
        # message, one per 2 ms physics step, in Python. MoveItPy has the twin's clock.
        self._context = rclpy.Context()
        rclpy.init(context=self._context)
        self.node = rclpy.create_node(node_name, context=self._context)
        self._executor = MultiThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        # spin() does not return on executor.shutdown(). A stop event plus the
        # guard is what ends the thread, the same way Spinner.close does.
        self._spin_stop = threading.Event()
        self._spin_wake = self.node.create_guard_condition(lambda: None)
        self._spinner = threading.Thread(target=self._spin, name=node_name, daemon=True)
        self._spinner.start()
        self._trajectory = ActionClient(self.node, FollowJointTrajectory, self._trajectory_controller.action)
        self._jog_pub = self.node.create_publisher(JointJog, f"/{SERVO_NODE}/delta_joint_cmds", 10)
        from controller_manager_msgs.msg import ControllerManagerActivity
        from lifecycle_msgs.msg import State
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

        self._active_state = State.PRIMARY_STATE_ACTIVE
        # Latched: a subscriber that starts after the switch still gets the last activity.
        activity = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", self._on_activity, activity)
        self.node.create_subscription(JointState, "/joint_states", self._on_state, 10)
        from control_msgs.msg import JointTrajectoryControllerState

        self.node.create_subscription(
            JointTrajectoryControllerState, f"/{self._trajectory_controller.name}/controller_state",
            self._on_controller_state, 10,
        )
        from moveit_msgs.msg import ServoStatus

        # Volatile: a latched status from an earlier stream is not this stream's.
        self.node.create_subscription(
            ServoStatus, f"/{SERVO_NODE}/status", self._on_servo_status,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE),
        )
        if not self._trajectory.wait_for_server(timeout_sec=BRING_UP_S):
            self.close()
            raise RuntimeError(f"{self._trajectory_controller.action} is not available after {BRING_UP_S} s")
        with self._cond:
            if not self._cond.wait_for(lambda: self._state is not None, BRING_UP_S):
                self.close()
                raise RuntimeError(f"no /joint_states within {BRING_UP_S} s")

    # -- state ----------------------------------------------------------------
    def _on_activity(self, msg) -> None:
        """The controllers the manager currently reports as active."""
        active = {item.name for item in msg.controllers if item.state.id == self._active_state}
        with self._cond:
            self._active = active
            self._cond.notify_all()

    def _on_controller_state(self, msg) -> None:
        """The trajectory controller has taken the streamed setpoint once its reference is on it."""
        with self._cond:
            goal = self._stream_goal
            if goal is None or self._stream_done:
                return
            reference = dict(zip(msg.joint_names, msg.reference.positions))
            if all(name in reference and abs(reference[name] - value) < 0.01 for name, value in goal.items()):
                self._stream_done = True
                self._cond.notify_all()

    def _on_servo_status(self, msg) -> None:
        """Servo slows or holds the stream near a collision or a singularity, and at a joint bound.

        Every code above ``NO_WARNING`` is one of those. Near a collision Servo
        reports ``DECELERATE_FOR_COLLISION`` with the arm already stopped.
        """
        from moveit_msgs.msg import ServoStatus

        with self._cond:
            self._servo_limited = msg.code > ServoStatus.NO_WARNING
            self._cond.notify_all()

    def _on_state(self, msg) -> None:
        load = _gripper_current(msg, self.gripper_joint)
        with self._cond:
            joints = dict(self._state[1]) if self._state else {}
            joints.update(zip(msg.name, msg.position))
            self._speeds = dict(zip(msg.name, msg.velocity))
            self._state = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, joints)
            self._gripper_load = load
            cached = dict(joints)
            hook = self.on_cached_joints
            self._cond.notify_all()
        if hook is not None:
            hook(cached, load)

    def get_gripper_load(self):
        """The gripper current in milliamps, from the latest joint-state effort."""
        with self._cond:
            return self._gripper_load

    def _settle(self) -> bool:
        """Wait until the arm is still, and until the controller has taken a streamed setpoint.

        A reported speed of zero is also the state before a stream starts. The
        trajectory controller's reference reaching the setpoint is what shows
        the stream is not still on its way. A setpoint Servo holds the arm
        short of never gets there; a still arm under Servo's status for that
        hold ends the stream instead. Returns whether a stream was in flight:
        the next action goal has to replace that hold.
        """
        def rested() -> bool:
            if any(abs(self._speeds.get(j, 0.0)) >= STILL_RAD_S for j in self.arm_joints):
                return False
            return self._stream_goal is None or self._stream_done or self._servo_limited

        with self._cond:
            streamed = self._stream_goal is not None
            still = self._cond.wait_for(rested, REPLY_S)
            if still:
                self._stream_goal = None
                self._stream_done = False
        if not still:
            raise RuntimeError(f"the arm did not come to rest within {REPLY_S} s")
        return streamed

    def get_joints(self) -> dict:
        """Every joint's position now, from MoveIt's current state monitor.

        This process's own ``/joint_states`` subscription can run behind it while
        Python is busy; it only paces :meth:`wait_state`.
        """
        return current_joints(self.robot)

    def cached_joints(self) -> dict | None:
        """The joint positions the state subscription last stored, or None before the first one."""
        with self._cond:
            if self._state is None:
                return None
            return dict(self._state[1])

    def state_time(self) -> float:
        """The stamp (seconds, the arm's clock) of the latest ``/joint_states``."""
        with self._cond:
            return self._state[0]

    def wait_state(self, t: float, stop_check=None) -> float | None:
        """Block until a ``/joint_states`` stamped at or after ``t``; its stamp,
        or None when a stop came first."""
        stops = self._stops
        budget = max(t - self.state_time(), 0.0) + REPLY_S
        with self._cond:
            ok = self._cond.wait_for(lambda: self._state[0] >= t or self._stops != stops, budget)
            if self._stops != stops or (stop_check is not None and stop_check()):
                return None
            if not ok:
                raise RuntimeError(f"no /joint_states at or after {t:.3f} s within {budget:.1f} s")
            return self._state[0]

    def named_state(self, name: str) -> dict:
        """The SRDF named state ``name`` of the arm or the gripper: ``{joint: position}``."""
        from moveit.core.robot_state import RobotState

        state = RobotState(self.robot.get_robot_model())
        for group, joints in ((PLANNING_GROUP, self.arm_joints), (GRIPPER_GROUP, [self.gripper_joint])):
            if state.set_to_default_values(group, name):
                return {j: state.joint_positions[j] for j in joints}
        raise KeyError(f"no named state {name!r} in the SRDF")

    def scan_states(self) -> list[str]:
        """``scan_1`` through ``scan_8``, then ``scan_near_1`` through ``scan_near_5``.

        Each name is an SRDF state :meth:`named_state` can load.
        """
        names = [f"scan_{i}" for i in range(1, 9)] + [f"scan_near_{i}" for i in range(1, 6)]
        for name in names:
            self.named_state(name)
        return names

    # -- motion ---------------------------------------------------------------
    def go_to_named(self, name: str) -> str:
        """The SRDF named state ``name``: a joint-interpolation Task Constructor move for the arm, the gripper action for the jaws."""
        clear_rcutils_error()
        target = self.named_state(name)
        if self.gripper_joint in target:
            self.grip(target[self.gripper_joint])
            return "arrived"
        return self.go_to_mtc(target)

    def go_to(self, target: dict, duration_s: float | None = None, stop_event=None,
              velocity_scale: float | None = None, acceleration_scale: float | None = None) -> str:
        """Move to ``target`` (partial dicts allowed).

        With no ``duration_s`` this is a joint-interpolation plan. Its speed is
        :func:`planning_scales`: ``joint_limits.yaml``'s default scales times
        each joint's max, unless this call passes a scale. ``duration_s``
        instead plays one keyframe the trajectory controller reaches in that
        time (dance and wave). The jaws move on the gripper action at the same
        time. Setting ``stop_event`` cancels the move and it reports
        ``"stopped"``.
        """
        if stop_event is not None and stop_event.is_set():
            return "stopped"
        if stop_event is not None:
            bound = (duration_s if duration_s is not None else 30.0) + REPLY_S
            threading.Thread(target=self._stop_on, args=(stop_event, bound), daemon=True).start()
        arm = {j: float(v) for j, v in target.items() if j in self.arm_joints}
        jaws = target.get(self.gripper_joint)
        failed: list = []

        def move_jaws():
            try:
                self.grip(float(jaws))
            except Exception as e:  # handed to the caller after the arm's move
                failed.append(e)

        gripper = threading.Thread(target=move_jaws, daemon=True) if jaws is not None else None
        if gripper is not None:
            gripper.start()
        how = "arrived"
        if arm and duration_s is not None:
            how = self._play(arm, duration_s)
        elif arm:
            # Joint interpolation. MoveItPy.plan segfaults in this process after a
            # pick task, and a stream can still be moving when this is called.
            self._settle()
            measured = self.get_joints()
            how = self.go_to_mtc(
                {name: float(arm.get(name, measured[name])) for name in self.arm_joints},
                velocity_scale, acceleration_scale,
            )
        if gripper is not None:
            gripper.join()
            if failed:
                raise failed[0]
        return how

    def _cpp_node(self):
        """The ``rclcpp`` node MoveIt's pipeline planner takes.

        MoveItPy's own node is not this object, and the ``rclpy`` node is a
        different type. One node for every task this arm plans.
        """
        node = getattr(self, "_mtc_node", None)
        if node is not None:
            return node
        import rclcpp

        try:
            node = rclcpp.Node("so101_mtc")
        except Exception:
            rclcpp.init()
            node = rclcpp.Node("so101_mtc")
        self._mtc_node = node
        return node

    @property
    def attach_link(self) -> str:
        """The link SimpleGrasp attaches to: the SRDF end effector's parent link."""
        if self._attach_link is None:
            import xml.etree.ElementTree as ET

            text = moveit_configs().robot_description_semantic["robot_description_semantic"]
            parent = next(
                (eef.get("parent_link") for eef in ET.fromstring(text).iter("end_effector") if eef.get("name") == "gripper"),
                None,
            )
            if not parent:
                raise RuntimeError("the SRDF gripper end effector has no parent link")
            self._attach_link = parent
        return self._attach_link

    def _in_attach_link(self, pose):
        """``pose``, given in :data:`TIP_LINK`, expressed in :attr:`attach_link`.

        The robot model holds the fixed joint between those links.
        """
        from geometry_msgs.msg import Pose

        state = __import__("moveit.core.robot_state", fromlist=["RobotState"]).RobotState(
            self.robot.get_robot_model()
        )
        state.update()
        parent = np.asarray(state.get_global_link_transform(self.attach_link), dtype=float)
        tip = np.asarray(state.get_global_link_transform(TIP_LINK), dtype=float)
        into_parent = np.linalg.inv(parent) @ tip
        quarter = np.array((
            (1 - 2 * (pose.orientation.y * pose.orientation.y + pose.orientation.z * pose.orientation.z),
             2 * (pose.orientation.x * pose.orientation.y - pose.orientation.z * pose.orientation.w),
             2 * (pose.orientation.x * pose.orientation.z + pose.orientation.y * pose.orientation.w)),
            (2 * (pose.orientation.x * pose.orientation.y + pose.orientation.z * pose.orientation.w),
             1 - 2 * (pose.orientation.x * pose.orientation.x + pose.orientation.z * pose.orientation.z),
             2 * (pose.orientation.y * pose.orientation.z - pose.orientation.x * pose.orientation.w)),
            (2 * (pose.orientation.x * pose.orientation.z - pose.orientation.y * pose.orientation.w),
             2 * (pose.orientation.y * pose.orientation.z + pose.orientation.x * pose.orientation.w),
             1 - 2 * (pose.orientation.x * pose.orientation.x + pose.orientation.y * pose.orientation.y)),
        ))
        point = into_parent[:3, :3] @ np.array((pose.position.x, pose.position.y, pose.position.z)) + into_parent[:3, 3]
        turned = into_parent[:3, :3] @ quarter
        trace = float(turned[0, 0] + turned[1, 1] + turned[2, 2])
        out = Pose()
        out.position.x, out.position.y, out.position.z = (float(v) for v in point)
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * scale
            x = (turned[2, 1] - turned[1, 2]) / scale
            y = (turned[0, 2] - turned[2, 0]) / scale
            z = (turned[1, 0] - turned[0, 1]) / scale
        elif turned[0, 0] > turned[1, 1] and turned[0, 0] > turned[2, 2]:
            scale = math.sqrt(1.0 + turned[0, 0] - turned[1, 1] - turned[2, 2]) * 2.0
            w = (turned[2, 1] - turned[1, 2]) / scale
            x = 0.25 * scale
            y = (turned[0, 1] + turned[1, 0]) / scale
            z = (turned[0, 2] + turned[2, 0]) / scale
        elif turned[1, 1] > turned[2, 2]:
            scale = math.sqrt(1.0 + turned[1, 1] - turned[0, 0] - turned[2, 2]) * 2.0
            w = (turned[0, 2] - turned[2, 0]) / scale
            x = (turned[0, 1] + turned[1, 0]) / scale
            y = 0.25 * scale
            z = (turned[1, 2] + turned[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + turned[2, 2] - turned[0, 0] - turned[1, 1]) * 2.0
            w = (turned[1, 0] - turned[0, 1]) / scale
            x = (turned[0, 2] + turned[2, 0]) / scale
            y = (turned[1, 2] + turned[2, 1]) / scale
            z = 0.25 * scale
        out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = x, y, z, w
        return out

    def _mtc(self, populate, already=None, scene_object=None, attach_link=None, must_hold=False) -> str:
        """Plan one Task Constructor task from the current state and execute it.

        ``populate(task)`` adds the stages after the fixed start. The start
        scene is the monitored planning scene, with the measured joints.
        ``scene_object`` is a collision object in that start state.
        ``attach_link`` names the link it is already held by; otherwise it
        lies in the world. ``already()`` true means the arm is already there
        when no solution exists. Every sub-trajectory is executed in order; a
        jaw-only one goes to the gripper action. Once one has run, its scene
        change goes into the monitored scene (:meth:`_apply_effect`). With
        ``must_hold``, a close that reaches its goal instead of stalling
        returns ``"missed"``.
        """
        # A previous plan leaves the C++ typesupport probe set. This task's
        # first lookup would overwrite it, and librcutils prints that.
        clear_rcutils_error()
        import copy

        from moveit.core.robot_state import RobotState
        from moveit.task_constructor import core, stages

        self._settle()
        model = self.robot.get_robot_model()
        state = RobotState(model)
        # A joint a fraction of a milliradian outside its URDF limit (the jaw
        # reads -0.0005 at the stop) makes the interpolation planner reject
        # the whole start state. The limits are the model's.
        positions = self.get_joints()
        for group in model.joint_model_groups:
            names = group.active_joint_model_names
            for name, bounds in zip(names, group.active_joint_model_bounds):
                if name not in positions:
                    continue
                for bound in bounds:
                    if not bound.position_bounded:
                        continue
                    limited = min(max(positions[name], bound.min_position), bound.max_position)
                    excess = abs(positions[name] - limited)
                    if excess == 0.0:
                        continue
                    measured = positions[name]
                    positions[name] = limited
                    # The gripper action already treats JAW_GOAL_RAD as the same position.
                    if name == self.gripper_joint and excess < JAW_GOAL_RAD:
                        continue
                    logger.warning(
                        "%s at %.6f rad is outside its limit; planning from %.6f",
                        name, measured, limited,
                    )
        state.joint_positions = positions
        state.update()
        monitor = self.robot.get_planning_scene_monitor()
        with monitor.read_only() as monitored:
            scene = copy.copy(monitored)
        scene.current_state = state
        if scene_object is not None:
            if attach_link:
                from moveit_msgs.msg import AttachedCollisionObject

                attached = AttachedCollisionObject()
                attached.link_name = attach_link
                attached.object = scene_object
                attached.touch_links = ["gripper_link", "jaw_link"]
                scene.process_attached_collision_object(attached)
            else:
                scene.apply_collision_object(scene_object)
            # Pick plans the approach before SimpleGrasp allows this contact.
            # CartesianPath rejects the grasp pose until the start scene allows it.
            for link in ("gripper_link", "jaw_link"):
                scene.allowed_collision_matrix.set_entry(scene_object.id, link, True)
        task = core.Task()
        task.setRobotModel(model)
        start = stages.FixedState("now")
        start.setState(scene)
        task.add(start)
        populate(task)
        task.init()
        clear_rcutils_error()
        task.plan(1)
        clear_rcutils_error()
        solutions = list(task.solutions)
        if not solutions:
            logger.warning("task constructor: %s", [item.comment for item in task.failures])
            if scene.is_state_colliding(state, PLANNING_GROUP, True):
                logger.warning("the start state is in collision; the contacts are in MoveIt's log above")
            if already is not None and already():
                clear_rcutils_error()
                return "arrived"
            clear_rcutils_error()
            return "NO_PLAN"
        subs = list(solutions[0].toMsg().sub_trajectory)
        if not any(sub.trajectory.joint_trajectory.points for sub in subs):
            clear_rcutils_error()
            return "NO_PLAN"
        for sub in subs:
            chosen = sub.trajectory
            if not chosen.joint_trajectory.points:
                self._apply_effect(sub.scene_diff)
                continue
            names = list(chosen.joint_trajectory.joint_names)
            if names == [self.gripper_joint]:
                goal = float(chosen.joint_trajectory.points[-1].positions[0])
                clear_rcutils_error()
                stalled = self.grip(goal)
                clear_rcutils_error()
                close = self.named_state("close")[self.gripper_joint]
                if must_hold and not stalled and abs(goal - close) < JAW_GOAL_RAD:
                    clear_rcutils_error()
                    return "missed"
                self._apply_effect(sub.scene_diff)
                continue
            self._settle()
            now = self.get_joints()
            points = chosen.joint_trajectory.points
            # A one-point trajectory's only point is the goal. Replacing it
            # with the measured joints makes the controller succeed without
            # moving. Prepend a start instead.
            if len(points) == 1:
                from trajectory_msgs.msg import JointTrajectoryPoint

                start = JointTrajectoryPoint()
                start.positions = [float(now[name]) for name in names]
                start.velocities = [0.0] * len(names)
                start.accelerations = [0.0] * len(names)
                goal = points[0]
                if goal.time_from_start.sec == 0 and goal.time_from_start.nanosec == 0:
                    goal.time_from_start.nanosec = 100_000_000
                chosen.joint_trajectory.points.insert(0, start)
            else:
                # The start is measured; the velocities and accelerations stay
                # MoveIt's. Zeroing the start acceleration made the controller's
                # spline overshoot the joint speed limit by 18% on a short ramp.
                points[0].positions = [float(now[name]) for name in names]
            last = chosen.joint_trajectory.points[-1]
            # The trajectory controller rejects a last point that still moves.
            last.velocities = [0.0] * len(names)
            logger.info(
                "execute joints %s points %d last %s",
                names,
                len(chosen.joint_trajectory.points),
                [round(float(value), 4) for value in last.positions],
            )
            # The controller's action result ends the move. MoveItCpp.execute
            # then waits for the sim clock, and that wait does not return when
            # the clock is not advancing, after the controller has already
            # reported the goal reached.
            with self._trajectory_turn():
                clear_rcutils_error()
                how = self._follow_stream(chosen.joint_trajectory)
                clear_rcutils_error()
            if how != "arrived":
                logger.warning("task constructor execute status %r", how)
                clear_rcutils_error()
                return how
            self._apply_effect(sub.scene_diff)
        clear_rcutils_error()
        return "arrived"

    def _apply_effect(self, diff) -> None:
        """One executed sub-trajectory's scene change, in the monitored scene.

        This is what Task Constructor's ExecuteTaskSolution capability does after
        each sub-trajectory: the attached objects, then the world objects, in
        ``setPlanningSceneDiffMsg``'s order. The joints come from the arm. The
        monitor takes no collision-matrix change; every task here ends with the
        object's entries where its start scene had them.
        """
        monitor = self.robot.get_planning_scene_monitor()
        for attached in diff.robot_state.attached_collision_objects:
            monitor.process_attached_collision_object(attached)
        for obj in diff.world.collision_objects:
            monitor.process_collision_object(obj)

    def _follow_stream(self, joint_trajectory) -> str:
        """Send this trajectory on the trajectory action and wait for its result.

        A planned point keeps the velocities and accelerations MoveIt wrote.
        Dropping them makes the controller invent a spline, and that spline
        runs past the joint speed. A point with none keeps positions and
        ``time_from_start`` only. The header stamp stays unset, as
        :meth:`_play` does. The wait is the trajectory's own last time.
        """
        from action_msgs.msg import GoalStatus
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        with self._cond:
            self._stream_goal = None
            self._stream_done = False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(joint_trajectory.joint_names)
        for src in joint_trajectory.points:
            point = JointTrajectoryPoint(
                positions=list(src.positions),
                time_from_start=src.time_from_start,
            )
            if src.velocities:
                point.velocities = list(src.velocities)
            if src.accelerations:
                point.accelerations = list(src.accelerations)
            goal.trajectory.points.append(point)
        if not goal.trajectory.points:
            return "NO_PLAN"
        # A position-only point at time zero is the arm's current position.
        # The controller already holds that. A planned start point carries the
        # acceleration, and dropping it makes the spline run past the joint speed.
        first = goal.trajectory.points[0]
        bare = not first.velocities and not first.accelerations
        if (
            bare
            and first.time_from_start.sec == 0
            and first.time_from_start.nanosec == 0
            and len(goal.trajectory.points) > 1
        ):
            del goal.trajectory.points[0]
        last = goal.trajectory.points[-1].time_from_start
        timeout_s = last.sec + last.nanosec * 1e-9
        stops = self._stops

        def track(handle):
            with self._cond:
                self._goal = handle
                if self._stops != stops:
                    handle.cancel_goal_async()

        status, result = _result(
            self._trajectory, self._trajectory_controller.action, goal, timeout_s, on_handle=track,
        )
        with self._cond:
            self._goal = None
        if status == GoalStatus.STATUS_CANCELED:
            return "stopped"
        return "arrived" if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL else "blocked"

    def go_to_mtc(self, joints: dict, velocity_scale: float | None = None, acceleration_scale: float | None = None,
                  held=None) -> str:
        """Plan from the current state to ``joints`` with MoveIt Task Constructor and execute it.

        The task is a fixed start state and one joint-interpolation ``MoveTo``.
        ``held`` is a collision object attached to :attr:`attach_link` (see
        :meth:`_held`): the jaws touch it, and it moves with them.
        """
        from moveit.task_constructor import core, stages

        goal = {name: float(joints[name]) for name in self.arm_joints if name in joints}

        def populate(task):
            move = stages.MoveTo("goal", _at_speed(core.JointInterpolationPlanner(), velocity_scale, acceleration_scale))
            move.group = PLANNING_GROUP
            move.setGoal(goal)
            task.add(move)

        def already():
            now = self.get_joints()
            return bool(goal) and max(abs(now[name] - value) for name, value in goal.items()) < 0.02

        if held is None:
            return self._mtc(populate, already)
        return self._mtc(populate, already, scene_object=held, attach_link=self.attach_link)

    def _held(self, name: str, size):
        """``name`` held in the jaws, at :func:`grasp_frame` expressed in :attr:`attach_link`."""
        from vla_pipeline.vision.yolo_detect_npu import collision_object

        frame = self._in_attach_link(grasp_frame(size))
        held = collision_object(
            name, (frame.position.x, frame.position.y, frame.position.z), size, frame=self.attach_link,
        )
        held.primitive_poses[0].orientation = frame.orientation
        return held

    def move_tip(self, direction, distance_m: float) -> str:
        """Move the planning tip ``distance_m`` along ``direction`` in ``base_link``.

        ``direction`` is a 3-vector. The task is one Cartesian ``MoveRelative``
        of :data:`TIP_LINK`, which is the fixed-jaw pad. The whole distance
        has to be planned. The task starts from the robot alone: a remembered
        object in this scene stops the lower, and allowing the contact left
        the trajectory controller without a result.
        """
        from geometry_msgs.msg import PoseStamped, Vector3Stamped
        from moveit.task_constructor import core, stages

        if distance_m <= 0.0:
            return "arrived"

        def populate(task):
            path = _at_speed(core.CartesianPath())
            path.min_fraction = 1.0
            move = stages.MoveRelative("tip", path)
            move.group = PLANNING_GROUP
            frame = PoseStamped()
            frame.header.frame_id = TIP_LINK
            frame.pose.orientation.w = 1.0
            move.ik_frame = frame
            stamped = Vector3Stamped()
            stamped.header.frame_id = "base_link"
            stamped.vector.x, stamped.vector.y, stamped.vector.z = (float(v) for v in direction)
            move.setDirection(stamped)
            move.min_distance = float(distance_m)
            move.max_distance = float(distance_m)
            task.add(move)

        return self._mtc(populate)

    def pick_up(self, name: str, centre, size, lift_m: float, yaw: float) -> str:
        """Pick ``name`` with a Task Constructor pick and leave it held.

        The task is the current state, an OMPL connection, a ``Pick``
        (approach down, close, a short lift), then a ``MoveRelative`` of
        :data:`RAISE_SHOULDER_RAD` on ``shoulder_lift`` with the object still
        attached. The generated
        poses sit on the object and turn about its vertical axis. ``size``
        and ``yaw`` come from :meth:`grasp_box`: the jaws close across X, and
        across Y only when that side is the same. The jaw pad is
        ``TIP_LINK``. That link's fixed joint is half a turn about Y from the
        gripper link, so the inverse-kinematics frame carries the same half
        turn: without it the generator asks the jaw to point up. ``yaw``
        rotates the object's box about the vertical.
        """
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from moveit.task_constructor import core, stages

        from vla_pipeline.vision.yolo_detect_npu import collision_object

        obj = collision_object(name, centre, size, yaw=yaw)
        logger.info("pick %s centre %s size %s lift %s", name, [round(float(v), 4) for v in centre], size, lift_m)
        node = self._cpp_node()

        def populate(task):
            planner = _pipeline(node)
            task.add(stages.Connect("connect", [(PLANNING_GROUP, planner)]))
            generator = stages.GenerateGraspPose("grasp pose")
            # Angle 0 closes the jaws across the box's X, and a half turn swaps
            # the jaws. A right angle closes them across Y, which grasp_frame
            # centres only when Y is as wide as X.
            generator.angle_delta = math.pi / 2 if float(size[1]) == float(size[0]) else math.pi
            generator.pregrasp = "open"
            generator.grasp = "close"
            generator.eef = "gripper"
            generator.object = name
            generator.setMonitoredStage(task["now"])
            grasp = stages.SimpleGrasp(generator, "grasp")
            ik_frame = PoseStamped()
            ik_frame.header.frame_id = TIP_LINK
            ik_frame.pose = grasp_frame(size)
            grasp.setIKFrame(ik_frame)
            for stage in grasp:
                if stage.name == "compute ik":
                    # The next stage allows contact with the object. IK runs first and
                    # would otherwise reject the pad-on-object solution for that contact.
                    stage.ignore_collisions = True
                    break
            pick = stages.Pick(grasp, "pick")
            pick.eef = "gripper"
            pick.object = name
            pick.eef_frame = TIP_LINK
            down = TwistStamped()
            down.header.frame_id = "base_link"
            down.twist.linear.z = -1.0
            # The solver's step is 0.01 m. One step of that length has no IK
            # for this grasp, so the travelled distance is 0 and the 4 mm
            # minimum never sees the segment that does. Ask for that segment.
            pick.setApproachMotion(down, CARTESIAN_MIN_M, CARTESIAN_MIN_M)
            up = TwistStamped()
            up.header.frame_id = "base_link"
            up.twist.linear.z = 1.0
            pick.setLiftMotion(up, CARTESIAN_MIN_M, CARTESIAN_MIN_M)
            task.add(pick)
            raise_up = stages.MoveRelative("raise", _at_speed(core.JointInterpolationPlanner()))
            raise_up.group = PLANNING_GROUP
            raise_up.setDirection({"shoulder_lift_joint": RAISE_SHOULDER_RAD})
            task.add(raise_up)

        return self._mtc(populate, scene_object=obj, must_hold=True)

    def place_down(self, name: str, pose, size) -> str:
        """Set ``name`` down at ``pose``, open the jaw, and return to the SRDF ``home`` pose.

        ``pose`` is the object's target in ``base_link``. The pick left the
        object raised by :data:`RAISE_SHOULDER_RAD` on ``shoulder_lift``, and
        this starts from there. The stages are the MoveIt Task Constructor
        demo's (``demo/src/pick_place_task.cpp``): a joint-space connection,
        the lower, the place pose's inverse kinematics, the jaw opening,
        forbidding the hand contact, detaching, a Cartesian retreat, then an
        OMPL move to ``home``. The demo lowers in a straight line along
        world -z. From this grasp that line reaches 0.0117 m and the stage's
        minimum is 0.03 m, so the lower is the pick's shoulder move in the
        other direction. The retreat is stepped at :data:`CARTESIAN_MIN_M`,
        because the demo's 0.01 m step has no inverse kinematics for this
        grasp. The retreat is along :data:`TIP_LINK`'s +x, which is away from
        the object :func:`grasp_frame` puts on -x; the fixed jaw does not
        move when the jaw opens. The object starts attached to
        :attr:`attach_link`. A plan that does not arrive returns that status,
        and the object stays held.
        """
        from geometry_msgs.msg import PoseStamped, Vector3Stamped
        from moveit.task_constructor import core, stages

        held = self._held(name, size)
        frame = grasp_frame(size)
        node = self._cpp_node()
        # The gripper group lists the moving jaw only. The fixed jaw is the end
        # effector's parent link, and it is the one left against the object.
        hand_links = list(self.robot.get_robot_model().get_joint_model_group(GRIPPER_GROUP).link_model_names)
        if self.attach_link not in hand_links:
            hand_links.append(self.attach_link)

        def populate(task):
            task.add(stages.Connect(
                "connect", [(PLANNING_GROUP, _at_speed(core.JointInterpolationPlanner()))],
            ))
            # The demo's straight lower has no inverse kinematics here
            # (0.0117 m of the 0.03 m minimum). Undo the pick's shoulder raise.
            lower = stages.MoveRelative("lower", _at_speed(core.JointInterpolationPlanner()))
            lower.group = PLANNING_GROUP
            lower.setDirection({"shoulder_lift_joint": -RAISE_SHOULDER_RAD})
            task.add(lower)
            generator = stages.GeneratePlacePose("place pose")
            generator.object = name
            generator.eef = "gripper"
            generator.pose = pose
            generator.setMonitoredStage(task["now"])
            ik = stages.ComputeIK("place pose IK", generator)
            ik.group = PLANNING_GROUP
            ik.eef = "gripper"
            ik_frame = PoseStamped()
            ik_frame.header.frame_id = TIP_LINK
            ik_frame.pose = frame
            ik.ik_frame = ik_frame
            # The pose comes from the generator's solution. Without this the
            # stage raises "Property 'target_pose': undefined".
            ik.properties.configureInitFrom(core.Stage.PropertyInitializerSource.INTERFACE, ["target_pose"])
            task.add(ik)
            open_hand = stages.MoveTo("open hand", _at_speed(core.JointInterpolationPlanner()))
            open_hand.group = GRIPPER_GROUP
            open_hand.setGoal("open")
            task.add(open_hand)
            forbid = stages.ModifyPlanningScene("forbid collision")
            forbid.allowCollisions(name, hand_links, False)
            task.add(forbid)
            detach = stages.ModifyPlanningScene("detach")
            detach.detachObject(name, self.attach_link)
            task.add(detach)
            away = Vector3Stamped()
            away.header.frame_id = TIP_LINK
            away.vector.x = 1.0
            # The demo retreats up to 0.25 m back along its approach axis. 0.03 m
            # clears this fixed jaw.
            task.add(_cartesian_move("retreat", away, 0.03, 0.25))
            home = stages.MoveTo("move home", _pipeline(node))
            home.group = PLANNING_GROUP
            home.setGoal("home")
            home.restrictDirection(stages.MoveTo.Direction.FORWARD)
            task.add(home)

        return self._mtc(populate, scene_object=held, attach_link=self.attach_link)

    def _stop_on(self, event, bound_s: float) -> None:
        """Cancel the current move once ``event`` is set, within ``bound_s``."""
        if event.wait(timeout=bound_s):
            self.stop()

    def _play(self, arm: dict, duration_s: float) -> str:
        from action_msgs.msg import GoalStatus
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        # This action goal replaces any streamed setpoint on the same controller.
        # Servo publishes on that same topic, so it stays paused until the goal ends.
        with self._trajectory_turn():
            with self._cond:
                self._stream_goal = None
                self._stream_done = False
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = list(self.arm_joints)
            # Leave the header stamp unset. The controller then starts the point
            # when it receives the goal, on its own clock. This process's latest
            # joint-state stamp runs behind that clock, and a header taken from it
            # makes a short point end in the past, so the controller rejects it.
            now = self.get_joints()
            point = JointTrajectoryPoint(positions=[arm.get(j, now[j]) for j in self.arm_joints])
            whole = int(duration_s)
            point.time_from_start.sec = whole
            point.time_from_start.nanosec = int(round((duration_s - whole) * 1e9))
            goal.trajectory.points = [point]
            stops = self._stops

            def track(handle):
                with self._cond:
                    self._goal = handle
                    if self._stops != stops:
                        handle.cancel_goal_async()

            status, result = _result(self._trajectory, self._trajectory_controller.action, goal, duration_s, on_handle=track)
            with self._cond:
                self._goal = None
            if status == GoalStatus.STATUS_CANCELED:
                return "stopped"
            return "arrived" if result.error_code == FollowJointTrajectory.Result.SUCCESSFUL else "blocked"

    @contextlib.contextmanager
    def _trajectory_turn(self):
        """Pause Servo around one trajectory, then let streaming resume.

        Servo and this action both command the trajectory controller. A jog
        published during the action replaces it.
        """
        self._pause_servo(True)
        try:
            yield
        finally:
            self._pause_servo(False)

    def _pause_servo(self, paused: bool) -> None:
        """Stop the jog, and tell Servo to publish nothing while ``paused``."""
        from std_srvs.srv import SetBool

        if paused:
            with self._cond:
                self._servo_target = None
                timer = self._servo_timer
                self._servo_timer = None
            if timer is not None:
                timer.cancel()
        if self._servo_client is None:
            self._servo_client = self.node.create_client(SetBool, f"/{SERVO_NODE}/pause_servo")
        if not self._servo_client.service_is_ready() and not self._servo_client.wait_for_service(BRING_UP_S):
            raise RuntimeError(f"/{SERVO_NODE}/pause_servo is not available after {BRING_UP_S} s")
        done = threading.Event()
        future = self._servo_client.call_async(SetBool.Request(data=paused))
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(REPLY_S):
            raise RuntimeError(f"/{SERVO_NODE}/pause_servo did not answer within {REPLY_S} s")
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(getattr(result, "message", None) or f"/{SERVO_NODE}/pause_servo failed")

    def _select_joint_jog(self) -> None:
        """Tell Servo the next commands are joint velocities.

        The node starts with no command type. A jog that arrives before
        ``switch_command_type`` selects joint jog is dropped.
        """
        from moveit_msgs.srv import ServoCommandType

        if self._servo_joint_jog:
            return
        if self._servo_type is None:
            self._servo_type = self.node.create_client(
                ServoCommandType, f"/{SERVO_NODE}/switch_command_type"
            )
        if not self._servo_type.service_is_ready() and not self._servo_type.wait_for_service(BRING_UP_S):
            raise RuntimeError(f"/{SERVO_NODE}/switch_command_type is not available after {BRING_UP_S} s")
        done = threading.Event()
        future = self._servo_type.call_async(ServoCommandType.Request(command_type=SERVO_JOINT_JOG))
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(REPLY_S):
            raise RuntimeError(f"/{SERVO_NODE}/switch_command_type did not answer within {REPLY_S} s")
        result = future.result()
        if result is None or not result.success:
            raise RuntimeError(f"/{SERVO_NODE}/switch_command_type rejected joint jog")
        self._servo_joint_jog = True

    def _publish_jog(self) -> None:
        """One Servo jog toward the streamed target, at a speed each joint can still stop from
        (:func:`jog_velocities`).

        Servo integrates ``velocity * publish_period`` and then keeps that step
        inside the joint speed and acceleration limits. The stamp is the latest
        joint state's, which is the twin's clock; this node stays on the system
        clock. Publishing stops once the named joints are at the setpoint, and
        Servo halts on its own.
        """
        from control_msgs.msg import JointJog

        with self._cond:
            target = None if self._servo_target is None else dict(self._servo_target)
            wanted = None if self._stream_goal is None else dict(self._stream_goal)
            state = self._state
            speeds = dict(self._speeds)
        if target is None or state is None:
            return
        stamp_s, measured = state
        # The same 0.01 rad the controller-state check uses for "the reference is there",
        # and :data:`STILL_RAD_S`. Only the joints this stream named: the others hold.
        arrived = not wanted or all(
            abs(wanted[name] - measured.get(name, wanted[name])) < 0.01
            and abs(speeds.get(name, 0.0)) < STILL_RAD_S
            for name in wanted
        )
        if arrived:
            with self._cond:
                if self._servo_target != target:
                    return
                timer = self._servo_timer
                self._servo_timer = None
                self._servo_target = None
            if timer is not None:
                timer.cancel()
            return
        # Only the joints this call named. A JointJog leaves every other joint at
        # zero delta, so Servo holds it. Commanding those joints toward a frozen
        # position drives one that is already on its stop through the velocity
        # scale, and the named joint then crawls.
        names = list(wanted)
        asked = jog_velocities(
            {name: target[name] - measured.get(name, target[name]) for name in names},
            self._max_acceleration, self._max_velocity,
        )
        velocities = [asked[name] for name in names]
        whole = int(stamp_s)
        nanosec = int(round((stamp_s - whole) * 1e9))
        if nanosec >= 1_000_000_000:
            whole += 1
            nanosec -= 1_000_000_000
        msg = JointJog()
        msg.header.stamp.sec = whole
        msg.header.stamp.nanosec = nanosec
        msg.joint_names = names
        msg.velocities = velocities
        self._jog_pub.publish(msg)

    def send_joints(self, joints: dict) -> None:
        """Stream one setpoint through MoveIt Servo; joints left out are not in the jog.

        Servo applies each joint's ``max_velocity`` and ``max_acceleration``
        from joint_limits.yaml. ``_settle`` waits for the controller's
        reference to reach the joints this call named; the joints filled in
        from the measured state are not part of that check.
        """
        self._select_joint_jog()
        wanted = {name: float(value) for name, value in joints.items() if name in self.arm_joints}
        with self._cond:
            measured = dict(self._state[1]) if self._state is not None else {}
        if len(measured) < len(self.arm_joints):
            measured = {**self.get_joints(), **measured}
        commanded = {name: float(wanted.get(name, measured[name])) for name in self.arm_joints}
        with self._cond:
            self._servo_target = commanded
            self._stream_goal = wanted
            self._stream_done = False
            self._servo_limited = False
            if self._servo_timer is None:
                self._servo_timer = self.node.create_timer(SERVO_PERIOD_S, self._publish_jog)
            self._cond.notify_all()

    def send_gripper(self, position: float) -> None:
        """Stream the jaws toward ``position`` (radians) on the gripper action.

        The trajectory controller does not own the jaw. A target within
        :data:`JAW_GOAL_RAD` of the last one sent is left alone, so a stream
        does not stack a goal per frame.
        """
        from control_msgs.action import ParallelGripperCommand

        position = float(position)
        if self._jaw_sent is not None and abs(position - self._jaw_sent) < JAW_GOAL_RAD:
            return
        self._jaw_sent = position
        client, gripper = _gripper_action(self.node)
        goal = ParallelGripperCommand.Goal()
        goal.command.name = list(gripper.joints)
        goal.command.position = [position]
        goal.command.effort = [0.0]
        client.send_goal_async(goal)

    def grip(self, position: float, max_effort: float = 0.0) -> bool:
        """The jaws to ``position`` on the gripper action; True when they stalled.

        The action succeeds as soon as the jaw is inside its goal tolerance,
        still moving: an open from the close passes 1.69 rad at 3.1 rad/s and
        swings to 1.734 rad. A jaw that did not stall is back within
        :data:`JAW_GOAL_RAD` of ``position`` and still before this returns, so
        the next plan starts where it stops. Its speed alone is zero at the
        top of that swing. A stalled jaw presses on what it holds and is not
        waited for.
        """
        position = float(position)
        if grip(self.node, position, max_effort):
            return True
        jaw = self.gripper_joint

        def at_rest() -> bool:
            return (
                abs(self._state[1][jaw] - position) < JAW_GOAL_RAD
                and abs(self._speeds.get(jaw, 0.0)) < STILL_RAD_S
            )

        with self._cond:
            if not self._cond.wait_for(at_rest, REPLY_S):
                raise RuntimeError(f"the jaw did not come to rest at {position:.3f} rad within {REPLY_S} s")
        return False

    def go_to_rest(self) -> str:
        """The SRDF ``home`` pose with the jaws closed.

        Home is a joint-interpolation plan.
        """
        how = self.go_to_mtc(self.named_state("home"))
        self.go_to_named("close")
        return how

    def stop(self) -> None:
        """Cancel the move in progress: MoveIt's execution, the trajectory controller's goal, or the stream.

        A cancelled stream leaves no setpoint for :meth:`_settle` to wait for;
        Servo halts the arm wherever the jog ended.
        """
        with self._cond:
            self._stops += 1
            self._servo_target = None
            self._stream_goal = None
            self._stream_done = False
            timer = self._servo_timer
            self._servo_timer = None
            goal = self._goal
            self._cond.notify_all()
        if timer is not None:
            timer.cancel()
        self.robot.get_trajectory_execution_manager().stop_execution(True)
        if goal is not None:
            goal.cancel_goal_async()

    def grasp_box(self, size, yaw: float):
        """:func:`grasp_box` with the jaws opened to the SRDF ``open`` state.

        The pick, the planning scene and the set-down all take the box this returns.
        """
        return grasp_box(size, yaw, self.named_state("open")[self.gripper_joint])

    def remember(self, name: str, centre, size, yaw: float = 0.0) -> None:
        """Put the located object into this arm's planning scene.

        ``centre`` and ``size`` are metres. ``size`` is ``(x, y, z)``. ``yaw``
        is the rotation about the vertical. The same name replaces the previous
        object. A pick plans from this scene, so every prop the camera sees is
        here before the arm moves.
        """
        from vla_pipeline.vision.yolo_detect_npu import collision_object

        monitor = self.robot.get_planning_scene_monitor()
        with monitor.read_write() as scene:
            scene.apply_collision_object(collision_object(name, centre, size, yaw=yaw))

    def let_go(self, name: str) -> None:
        """Drop ``name`` from the planning scene: the jaws opened where they are.

        Detaching puts the object back in the world at the jaws, which is not
        where it lands. The next scan :meth:`remember`\\ s it again.
        """
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject

        monitor = self.robot.get_planning_scene_monitor()
        detach = AttachedCollisionObject()
        detach.object.id = name
        detach.object.operation = CollisionObject.REMOVE
        monitor.process_attached_collision_object(detach)
        gone = CollisionObject()
        gone.id = name
        gone.operation = CollisionObject.REMOVE
        monitor.process_collision_object(gone)

    def reset(self, timeout_s: float = 15.0) -> None:
        """Put the arm back at the start.

        The same call on the real arm and on the twin. The Servo jog is
        cancelled first: a target from before the reset can never be
        acknowledged, and a jog that keeps publishing moves the arm out from
        under the next plan. Remembered and attached objects leave the planning
        scene. On the twin the world then snaps to its ``home`` keyframe, arm
        and props together, with the controllers off so the trajectory
        controller holds that pose instead of driving back to its last goal.
        Servo is paused across it: a stream it was still halting would publish
        its last pose from before the reset, and unpausing makes the reset pose
        its last command. On the real arm there is no world to snap, so the arm plans home and
        closes the jaws, which is :meth:`go_to_rest`.
        """
        from controller_manager_msgs.srv import SwitchController
        from moveit_msgs.msg import AttachedCollisionObject, CollisionObject
        from mujoco_ros2_control_msgs.srv import ResetWorld

        needed = set(controllers())
        with self._cond:
            self._cond.wait_for(lambda: needed <= self._active)
            self._stream_goal = None
            self._stream_done = False
            self._servo_target = None
            self._servo_limited = False
            timer = self._servo_timer
            self._servo_timer = None
        if timer is not None:
            timer.cancel()
        monitor = self.robot.get_planning_scene_monitor()
        detach = AttachedCollisionObject()
        detach.object.operation = CollisionObject.REMOVE
        monitor.process_attached_collision_object(detach)
        with monitor.read_write() as scene:
            scene.remove_all_collision_objects()
        if not self.sim:
            self.go_to_rest()
            return
        names = list(needed)
        strict = SwitchController.Request.STRICT
        self._pause_servo(True)
        try:
            stop = SwitchController.Request(deactivate_controllers=names, strictness=strict)
            if not _call(self.node, SwitchController, SWITCH_SERVICE, stop, timeout_s).ok:
                raise RuntimeError(f"could not stop {names}")
            try:
                reply = _call(self.node, ResetWorld, RESET_SERVICE, ResetWorld.Request(keyframe=HOME_KEYFRAME), timeout_s)
                if not reply.success:
                    raise RuntimeError(f"reset_world: {reply.message}")
            finally:
                start = SwitchController.Request(activate_controllers=names, strictness=strict)
                if not _call(self.node, SwitchController, SWITCH_SERVICE, start, timeout_s).ok:
                    raise RuntimeError(f"could not start {names}")
            now = _arm_now(True, timeout_s)
            if not self.robot.get_planning_scene_monitor().wait_for_current_robot_state(now, timeout_s):
                raise RuntimeError(f"MoveIt received no state of the arm within {timeout_s} s of the reset")
        finally:
            self._pause_servo(False)

    def _spin(self) -> None:
        """Run callbacks until :meth:`close` sets the stop event and wakes this wait."""
        from rclpy.executors import ExternalShutdownException, ShutdownException

        while not self._spin_stop.is_set():
            try:
                self._executor.spin_once()
            except (ExternalShutdownException, ShutdownException):
                return
            except Exception:
                logger.exception("arm callback")

    def close(self) -> None:
        import rclpy

        with self._cond:
            self._servo_target = None
            timer = self._servo_timer
            self._servo_timer = None
        if timer is not None:
            timer.cancel()
        self._spin_stop.set()
        self._spin_wake.trigger()
        self._spinner.join(timeout=2.0)
        self._executor.shutdown(timeout_sec=0, wait_for_threads=False)
        try:
            self.node.destroy_node()
        except Exception:
            pass
        # MoveIt logs while it shuts down. This context can be the last one
        # holding rcl logging (see _arm_now), so it goes after MoveItPy.
        self.robot.shutdown()
        rclpy.shutdown(context=self._context)
