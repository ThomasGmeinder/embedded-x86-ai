# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MoveIt 2 on the SO-101: MoveItPy on ``so_arm101_moveit_config``, and the twin's reset.

``launch/so101_control.launch.py hardware_type:=mujoco`` (or ``real``) runs
ros2_so_arm's controllers and ``move_group``; a process plans and executes
with its own :func:`moveit_py`. The controllers MoveIt drives are the ones in
the config's ``trajectory_execution.yaml``.
"""

from __future__ import annotations

import threading

PLANNING_GROUP = "manipulator"
GRIPPER_GROUP = "gripper"
TIP_LINK = "gripper_frame_link"
HOME_KEYFRAME = "home"
RESET_SERVICE = "/mujoco_ros2_control_node/reset_world"
SWITCH_SERVICE = "/controller_manager/switch_controller"


def moveit_configs():
    from so_arm_utils.launch_utils import MoveItConfigs

    return MoveItConfigs("so_arm101")


def controllers() -> dict:
    """``{controller: action}`` for every controller MoveIt executes on."""
    manager = moveit_configs().trajectory_execution["moveit_simple_controller_manager"]
    return {name: f"/{name}/{manager[name]['action_ns']}" for name in manager["controller_names"]}


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
    context = rclpy.Context()
    rclpy.init(context=context)
    try:
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
        rclpy.shutdown(context=context)


def moveit_py(node_name: str, *, sim: bool, state_timeout_s: float = 10.0):
    """MoveItPy on so_arm101_moveit_config, on the twin's clock when ``sim``,
    returned once MoveIt holds a state of the arm from now or later.

    The parameters go in as a ``/**`` parameter file: MoveItPy's ``config_dict``
    keys them by node name, and with ``use_sim_time`` that aborts on /clock's QoS
    override parameters (moveit/moveit2#2940).
    """
    from moveit.planning import MoveItPy
    from moveit.utils import create_params_file_from_dict

    config = moveit_configs().to_dict() | {"use_sim_time": sim}
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


def reset_twin(node, timeout_s: float = 15.0) -> None:
    """Put the twin back at its ``home`` keyframe: the arm at home with the jaws
    open, and every prop where the scene file has it.

    MoveIt's controllers are switched off around ``reset_world``, so the
    trajectory controller holds the reset pose when it starts again instead of
    driving the arm back to its last goal.
    """
    from controller_manager_msgs.srv import SwitchController
    from mujoco_ros2_control_msgs.srv import ResetWorld

    names = list(controllers())
    strict = SwitchController.Request.STRICT
    stop = SwitchController.Request(deactivate_controllers=names, strictness=strict)
    if not _call(node, SwitchController, SWITCH_SERVICE, stop, timeout_s).ok:
        raise RuntimeError(f"could not stop {names}")
    reply = _call(node, ResetWorld, RESET_SERVICE, ResetWorld.Request(keyframe=HOME_KEYFRAME), timeout_s)
    if not reply.success:
        raise RuntimeError(f"reset_world: {reply.message}")
    start = SwitchController.Request(activate_controllers=names, strictness=strict)
    if not _call(node, SwitchController, SWITCH_SERVICE, start, timeout_s).ok:
        raise RuntimeError(f"could not start {names}")
