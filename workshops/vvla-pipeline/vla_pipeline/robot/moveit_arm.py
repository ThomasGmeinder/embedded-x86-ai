# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""MoveIt 2 on the SO-101: the arm every behavior drives, on ``so_arm101_moveit_config``.

``launch/so101_control.launch.py hardware_type:=mujoco`` (or ``real``) runs
ros2_so_arm's controllers and ``move_group``; :class:`MoveItArm` plans and
executes with its own MoveItPy (:func:`moveit_py`), plays timed keyframes and
streamed setpoints on the trajectory controller, and moves the jaws with the
gripper action. The controllers are the ones in the config's
``trajectory_execution.yaml``. Joint positions are radians under the URDF
joint names. :func:`go_to_named`, :func:`go_to_joints` and :func:`grip` are
the workshop's TODOs 2.7-2.9.
"""

from __future__ import annotations

import threading
from typing import NamedTuple

PLANNING_GROUP = "manipulator"
GRIPPER_GROUP = "gripper"
TIP_LINK = "gripper_frame_link"
HOME_KEYFRAME = "home"
RESET_SERVICE = "/mujoco_ros2_control_node/reset_world"
SWITCH_SERVICE = "/controller_manager/switch_controller"
# A controller accepts a goal and reports its result within this beyond the
# goal's own duration; it only catches a controller that stopped answering.
REPLY_S = 5.0
# The arm waits this long for the controllers at start: a spawner whose
# list_controllers reply DDS drops tries again 10 s later; this allows two.
BRING_UP_S = 30.0
# MoveIt's execution status, as the arm reports a move.
MOVE_RESULT = {"SUCCEEDED": "arrived", "PREEMPTED": "stopped"}  # any other status: "blocked"
# A plan starts from a still arm: MoveIt rejects a start state more than its
# allowed_start_tolerance (0.01 rad) from where the arm is when it executes.
STILL_RAD_S = 0.01


class Controller(NamedTuple):
    name: str
    action: str
    joints: list


def moveit_configs():
    from so_arm_utils.launch_utils import MoveItConfigs

    return MoveItConfigs("so_arm101")


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


def _execute(robot, arm) -> str:
    plan = arm.plan()
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


def grip(node, position: float, max_effort: float) -> bool:
    """Drive the jaws to ``position`` (radians) pushing with at most
    ``max_effort`` (N·m) on the gripper action; True when they stalled on
    something before reaching it. ``node`` is spun on another thread.
    The workshop's TODO 2.9.
    """
    from control_msgs.action import ParallelGripperCommand
    from rclpy.action import ActionClient

    gripper = _controllers()["ParallelGripperCommand"]
    client = ActionClient(node, ParallelGripperCommand, gripper.action)
    try:
        if not client.wait_for_server(timeout_sec=REPLY_S):
            raise RuntimeError(f"{gripper.action} is not available")
        goal = ParallelGripperCommand.Goal()
        goal.command.name = list(gripper.joints)
        goal.command.position = [float(position)]
        goal.command.effort = [float(max_effort)]
        _status, result = _result(client, gripper.action, goal, REPLY_S)
        return bool(result.stalled)
    finally:
        client.destroy()


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


class MoveItArm:
    """The arm behaviors drive. Joint positions are radians under the URDF joint names.

    - :meth:`go_to` without a duration, :meth:`go_to_named` and
      :meth:`go_to_rest` are MoveIt plans, executed;
    - :meth:`go_to` with ``duration_s`` plays one timed keyframe on the
      trajectory controller's action;
    - :meth:`send_joints` streams a setpoint on its ``~/joint_trajectory`` topic;
    - :meth:`grip` moves the jaws on the gripper action;
    - :meth:`stop` cancels whatever moves the arm; a cancelled move reports ``"stopped"``.

    Moves report ``"arrived"``, ``"stopped"`` or ``"blocked"``.
    """

    def __init__(self, cfg: dict, node_name: str = "vla_arm", *, sim: bool | None = None):
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import MultiThreadedExecutor
        from sensor_msgs.msg import JointState
        from trajectory_msgs.msg import JointTrajectory

        robot_cfg = cfg["robot"]
        self.sim = robot_cfg.get("backend") == "sim" if sim is None else sim
        self.robot = moveit_py(f"{node_name}_moveit", sim=self.sim)
        model = self.robot.get_robot_model()
        self.arm_joints = list(model.get_joint_model_group(PLANNING_GROUP).active_joint_model_names)
        by_type = _controllers()
        self._trajectory_controller = by_type["FollowJointTrajectory"]
        self.gripper_joint = by_type["ParallelGripperCommand"].joints[0]
        self._stream_s = 1.0 / float(robot_cfg.get("control_fps", 60.0))
        self._cond = threading.Condition()
        self._state = None  # (stamp seconds, {joint: position})
        self._speeds: dict = {}  # {joint: velocity} of the latest state
        self._goal = None  # the trajectory goal a stop cancels
        self._stops = 0

        # The node keeps the system clock: it times nothing itself (waits use the
        # states' stamps), and on the twin's clock it would take every /clock
        # message, one per 2 ms physics step, in Python. MoveItPy has the twin's clock.
        self._context = rclpy.Context()
        rclpy.init(context=self._context)
        self.node = rclpy.create_node(node_name, context=self._context)
        self._executor = MultiThreadedExecutor(context=self._context)
        self._executor.add_node(self.node)
        self._spinner = threading.Thread(target=self._executor.spin, name=node_name, daemon=True)
        self._spinner.start()
        self._trajectory = ActionClient(self.node, FollowJointTrajectory, self._trajectory_controller.action)
        self._stream = self.node.create_publisher(
            JointTrajectory, f"/{self._trajectory_controller.name}/joint_trajectory", 1,
        )
        self.node.create_subscription(JointState, "/joint_states", self._on_state, 10)
        if not self._trajectory.wait_for_server(timeout_sec=BRING_UP_S):
            self.close()
            raise RuntimeError(f"{self._trajectory_controller.action} is not available after {BRING_UP_S} s")
        with self._cond:
            if not self._cond.wait_for(lambda: self._state is not None, BRING_UP_S):
                self.close()
                raise RuntimeError(f"no /joint_states within {BRING_UP_S} s")

    # -- state ----------------------------------------------------------------
    def _on_state(self, msg) -> None:
        with self._cond:
            joints = dict(self._state[1]) if self._state else {}
            joints.update(zip(msg.name, msg.position))
            self._speeds = dict(zip(msg.name, msg.velocity))
            self._state = (msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9, joints)
            self._cond.notify_all()

    def _settle(self) -> None:
        """Wait for a ``/joint_states`` in which every arm joint is still."""
        with self._cond:
            still = self._cond.wait_for(
                lambda: all(abs(self._speeds.get(j, 0.0)) < STILL_RAD_S for j in self.arm_joints), REPLY_S,
            )
        if not still:
            raise RuntimeError(f"the arm did not come to rest within {REPLY_S} s")

    def get_joints(self) -> dict:
        """Every joint's position now, from MoveIt's current state monitor.

        This process's own ``/joint_states`` subscription can run behind it while
        Python is busy; it only paces :meth:`wait_state`.
        """
        return current_joints(self.robot)

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

    # -- motion ---------------------------------------------------------------
    def go_to_named(self, name: str) -> str:
        """The SRDF named state ``name``: a MoveIt plan for the arm, the gripper action for the jaws."""
        target = self.named_state(name)
        if self.gripper_joint in target:
            self.grip(target[self.gripper_joint])
            return "arrived"
        self._settle()
        return MOVE_RESULT.get(go_to_named(self.robot, name), "blocked")

    def go_to(self, target: dict, duration_s: float | None = None) -> str:
        """Move to ``target`` (partial dicts allowed): MoveIt's plan, or with
        ``duration_s`` one keyframe the trajectory controller reaches in that time.
        The jaws move on the gripper action at the same time."""
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
            self._settle()
            how = MOVE_RESULT.get(go_to_joints(self.robot, arm), "blocked")
        if gripper is not None:
            gripper.join()
            if failed:
                raise failed[0]
        return how

    def _play(self, arm: dict, duration_s: float) -> str:
        from action_msgs.msg import GoalStatus
        from control_msgs.action import FollowJointTrajectory
        from trajectory_msgs.msg import JointTrajectoryPoint

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(self.arm_joints)
        now = self.get_joints()
        point = JointTrajectoryPoint(positions=[arm.get(j, now[j]) for j in self.arm_joints])
        point.time_from_start.sec, point.time_from_start.nanosec = int(duration_s), int(duration_s % 1 * 1e9)
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

    def send_joints(self, joints: dict) -> None:
        """Stream one setpoint to the trajectory controller (reached in one
        ``robot.control_fps`` period); arm joints left out hold their position."""
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

        now = self.get_joints()
        point = JointTrajectoryPoint(positions=[float(joints.get(j, now[j])) for j in self.arm_joints])
        point.time_from_start.nanosec = int(self._stream_s * 1e9)
        self._stream.publish(JointTrajectory(joint_names=list(self.arm_joints), points=[point]))

    def grip(self, position: float, max_effort: float = 0.0) -> bool:
        """The jaws to ``position`` on the gripper action; True when they stalled."""
        return grip(self.node, position, max_effort)

    def go_to_rest(self) -> str:
        """The SRDF ``home`` pose with the jaws closed."""
        how = self.go_to_named("home")
        self.go_to_named("close")
        return how

    def stop(self) -> None:
        """Cancel the move in progress: MoveIt's execution, or the trajectory controller's goal."""
        with self._cond:
            self._stops += 1
            goal = self._goal
            self._cond.notify_all()
        self.robot.get_trajectory_execution_manager().stop_execution(True)
        if goal is not None:
            goal.cancel_goal_async()

    def reset(self) -> None:
        """The twin back at its ``home`` keyframe (:func:`reset_twin`)."""
        if not self.sim:
            raise RuntimeError("only the twin resets")
        reset_twin(self.node)

    def close(self) -> None:
        import rclpy

        self._executor.shutdown()
        self._spinner.join()
        self.node.destroy_node()
        rclpy.shutdown(context=self._context)
        self.robot.shutdown()
