# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The arm over ros2_control (PROVIDED).

``launch/so101_control.launch.py`` runs the controllers on top of the arm
server. :class:`ControllerArm` drives them; a subclass wires its ROS
endpoints and message conversions (``ros/arm_client.py``: TODO 2.7-2.9):

- a move (:meth:`~ControllerArm.move_to`, and ``go_to``) is one
  ``FollowJointTrajectory`` goal to ``arm_controller``, timed by the servo
  profile planner (:mod:`common.servo_profile`) and carrying the caller's
  tolerance; the controller's result says whether the joints got there;
- a streamed setpoint (:meth:`~ControllerArm.send_joints`: gesture mimic,
  person tracking) goes to ``arm_forward_controller``, switched in on the
  first one; the next move switches ``arm_controller`` back;
- the gripper is ``gripper_forward_controller``;
- the state is ``/joint_states`` from ``joint_state_broadcaster``.

Joint values are radians on the ROS side and calibrated units for behaviors,
through the twin's joint map. The gripper's effort is N·m on the wire and
``Present_Current`` counts for behaviors, through the datasheet line in
:mod:`so101_sim.servo`. Its torque limit is the arm server's
``gripper_torque_limit_nm`` parameter.
"""

from __future__ import annotations

import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from abc import abstractmethod

from common.motion import MOTOR_NAMES, ArmClient, MotionTimeout, clamp_joints, joints_text
from common.ros_helpers import Spinner

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCH_FILE = REPO_ROOT / "launch" / "so101_control.launch.py"
# topic_based_ros2_control, built by bootstrap.sh into the colcon workspace ros_ws/
BRIDGE_PREFIX = REPO_ROOT / "ros_ws" / "install" / "topic_based_ros2_control"
# The arm server's topics as the hardware side sees them; the launch remaps them there.
HARDWARE_COMMANDS = "/so101/hardware/joint_commands"
HARDWARE_STATES = "/so101/hardware/joint_states"
BODY = tuple(MOTOR_NAMES[:5])
ARM_CONTROLLER = "arm_controller"
STREAM_CONTROLLER = "arm_forward_controller"
# The launch's ``gripper:=`` choices; the client drives the forward one.
GRIPPER_CONTROLLERS = {"forward": "gripper_forward_controller", "action": "gripper_action_controller"}
GRIPPER_CONTROLLER = GRIPPER_CONTROLLERS["forward"]
SERVER_NODE = "so101_server_node"
TORQUE_PARAMETER = "gripper_torque_limit_nm"
# How long the hardware side waits, on activation, for the arm server's first
# joint state. The twin's server published it 0.7-0.95 s after its process
# started (scripts/ros2_control_poc.py); the real server also connects the bus.
SERVER_START_S = 10.0
# From the server's first state until every controller is loaded: 1.0 s with
# strix_vla.launch.py on the twin, on an idle host. Five times that covers a
# host that loads the pipeline's models at the same time.
CONTROLLERS_UP_S = 5.0
# How long a client started with the stack waits for it to come up.
BRING_UP_S = SERVER_START_S + CONTROLLERS_UP_S
# A controller's or service's reply takes one controller update and a local DDS
# hop (1.3 ms at most, scripts/ros2_control_poc.py): this bound only catches a
# controller manager or arm server that is gone.
REPLY_S = 2.0
# The arm server writes the controllers' setpoints as they come; LeRobot's
# per-tick clamp stays its backstop, at LeRobot's default when the config sets none.
STREAM_CLAMP_DEG = 8.0


def server_transport_cfg(cfg: dict) -> dict:
    """``cfg`` with the per-tick clamp the arm server's bus transport streams under."""
    robot = dict(cfg.get("robot") or {})
    if robot.get("max_relative_target") is None:
        robot["max_relative_target"] = STREAM_CLAMP_DEG
    return {**cfg, "robot": robot}


def bridge_prefix_path() -> str:
    """``AMENT_PREFIX_PATH`` with the bridge first, for the process that loads its plugin."""
    if not BRIDGE_PREFIX.is_dir():
        raise RuntimeError(f"{BRIDGE_PREFIX} is missing: run ./bootstrap.sh (section 4e builds it)")
    prefixes = [str(BRIDGE_PREFIX), *filter(None, os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep))]
    return os.pathsep.join(dict.fromkeys(prefixes))


def start_controllers(config_path, stack: str, gripper: str = "forward", log=None) -> subprocess.Popen:
    """``launch/so101_control.launch.py`` with this Python, in a process group of
    its own, shut down (SIGINT) when this process exits however it exits."""
    import shutil

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 is not on PATH: source /opt/ros/jazzy/setup.bash first")
    return subprocess.Popen(
        [sys.executable, "-m", "so101_sim.parent_death", sys.executable, ros2, "launch", str(LAUNCH_FILE),
         f"config:={config_path}", f"stack:={stack}", f"gripper:={gripper}"],
        cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True,
    )


def stop_controllers(proc: subprocess.Popen, budget_s: float = 10.0) -> None:
    """SIGINT the launch's process group; kill it when it has not exited within ``budget_s``."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise RuntimeError(f"the controllers' launch did not exit within {budget_s:.0f} s of SIGINT") from None


def _goal_time_s() -> float:
    """``arm_controller``'s ``constraints.goal_time`` from the controllers file."""
    import yaml

    from so101_sim.description import CONTROLLERS_FILE

    params = yaml.safe_load(CONTROLLERS_FILE.read_text())[ARM_CONTROLLER]["ros__parameters"]
    return float(params["constraints"]["goal_time"])


class ControllerArm(ArmClient):
    """:class:`~common.motion.ArmClient` over ros2_control; spins a private node.

    A subclass provides:

    - :meth:`_setup_ros`: ``self._trajectory``, the ``FollowJointTrajectory``
      action client of ``arm_controller``; ``self._stream_pub`` and
      ``self._gripper_pub``, ``std_msgs/Float64MultiArray`` publishers on the
      forward controllers' ``commands`` topics; and a ``/joint_states``
      subscription to :meth:`_on_state`;
    - :meth:`_on_state`: one ``/joint_states`` message to :meth:`_update_state`;
    - :meth:`send_joints`: a streamed setpoint, through :meth:`_command_gripper`
      and :meth:`_stream_setpoint`.
    """

    is_live = True
    # go_to() is one trajectory goal: the controller interpolates the move.
    _direct_goal = True

    def __init__(self, cfg: dict, node_name: str = "so101_arm_client", state_timeout_s: float = BRING_UP_S):
        from control_msgs.action import FollowJointTrajectory
        from controller_manager_msgs.msg import ControllerManagerActivity
        from controller_manager_msgs.srv import SwitchController
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float64MultiArray
        from std_srvs.srv import Trigger

        from so101_sim.config import load_sim_config
        from so101_sim.mapping import JointMap
        from so101_sim.node import RESET_SERVICE

        sim = load_sim_config(cfg)
        self._jmap = JointMap(sim["joint_map"], sim["gripper_map"])
        self._current_model = sim["current_model"]
        self._stall_nm = float(sim["current_model"]["stall_torque_nm"])
        control_fps = float(cfg["robot"]["control_fps"])
        # The server's position-difference velocity reads one encoder step per
        # tick while still; below one and a half of those the jaw has stopped.
        self._jaw_still_rad_s = 1.5 * self._jmap.steps_to_rad("gripper", 1.0) * control_fps
        self._goal_time_s = _goal_time_s()
        self._FollowJointTrajectory = FollowJointTrajectory
        self._SwitchController = SwitchController
        self._Float64MultiArray = Float64MultiArray

        self._cond = threading.Condition()
        self._state: dict | None = None  # calibrated units
        self._jaw_speed = 0.0  # rad/s
        self._effort_nm: float | None = None
        self._controllers: dict[str, str] = {}  # name -> state label, from the controller manager
        self._streaming = False
        self._stream: dict | None = None  # the last streamed body setpoint
        self._trajectory = self._stream_pub = self._gripper_pub = None
        self._spin = Spinner(node_name)
        node = self._spin.node
        self._switch = node.create_client(SwitchController, "/controller_manager/switch_controller")
        self._params = AsyncParameterClient(node, SERVER_NODE)
        # Only a twin serves it; made now, so it is discovered with the server.
        self._reset = node.create_client(Trigger, RESET_SERVICE)
        self._Trigger = Trigger
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        node.create_subscription(ControllerManagerActivity, "/controller_manager/activity", self._on_activity, latched)
        try:
            self._setup_ros(node)
            self._wait(
                lambda: self._state is not None and self._controllers_ready() and self._trajectory.server_is_ready()
                and self._switch.service_is_ready() and self._params.services_are_ready(),
                state_timeout_s,
                "/joint_states, the controllers, the trajectory action, the controller switch "
                "and the arm server's parameters",
            )
        except BaseException:
            self.close()
            raise
        self._streaming = self._controllers[STREAM_CONTROLLER] == "active"
        logger.info("ControllerArm: state stream and controllers live.")

    # -- the subclass's part --------------------------------------------------
    @abstractmethod
    def _setup_ros(self, node) -> None:
        """Create the trajectory action client, the forward controllers'
        publishers and the ``/joint_states`` subscription on ``node``."""

    @abstractmethod
    def _on_state(self, msg) -> None:
        """Convert one ``/joint_states`` message and pass it to :meth:`_update_state`."""

    @abstractmethod
    def send_joints(self, joints: dict) -> None:
        """Stream a setpoint: the body to the forward controller, the gripper to its own."""

    # -- state ---------------------------------------------------------------
    def _update_state(self, positions: dict, jaw_speed: float, effort_nm: float | None) -> None:
        """Store a state (calibrated units; the jaw's speed in rad/s; the gripper's
        effort in N·m, None when unreadable) and wake every wait."""
        with self._cond:
            self._state = {**(self._state or {}), **positions}
            self._jaw_speed, self._effort_nm = jaw_speed, effort_nm
            self._cond.notify_all()

    def _on_activity(self, msg) -> None:
        with self._cond:
            self._controllers = {c.name: c.state.label for c in msg.controllers}
            self._cond.notify_all()

    def _controllers_ready(self) -> bool:
        """The gripper controller active, one arm controller active and the other loaded."""
        arm = (self._controllers.get(ARM_CONTROLLER), self._controllers.get(STREAM_CONTROLLER))
        return self._controllers.get(GRIPPER_CONTROLLER) == "active" and arm in (
            ("active", "inactive"), ("inactive", "active"),
        )

    def _wait(self, pred, budget_s: float, what: str, stop_check=None) -> bool:
        """Wake on each joint state until ``pred()``; False when ``stop_check`` fires first."""
        with self._cond:
            ok = self._cond.wait_for(lambda: pred() or bool(stop_check and stop_check()), budget_s)
            if not ok:
                raise MotionTimeout(f"no {what} within {budget_s:.2f} s")
            return pred()

    def get_joints(self) -> dict:
        with self._cond:
            if self._state is None:
                raise RuntimeError("No arm state received yet.")
            return dict(self._state)

    def get_gripper_load(self):
        """The gripper's ``Present_Current`` counts, from its effort in N·m."""
        from so101_sim.servo import torque_nm_to_current

        with self._cond:
            effort = self._effort_nm
        return None if effort is None else torque_nm_to_current(effort, self._current_model)

    # -- controllers ---------------------------------------------------------
    def _call(self, future, what: str):
        done = threading.Event()
        future.add_done_callback(lambda _f: done.set())
        if not done.wait(REPLY_S):
            raise MotionTimeout(f"no reply to {what} within {REPLY_S:.1f} s")
        return future.result()

    def _use_stream(self, streaming: bool) -> None:
        """Switch between the trajectory controller and the forward controller."""
        if streaming == self._streaming:
            return
        on, off = (STREAM_CONTROLLER, ARM_CONTROLLER) if streaming else (ARM_CONTROLLER, STREAM_CONTROLLER)
        request = self._SwitchController.Request(
            activate_controllers=[on], deactivate_controllers=[off],
            strictness=self._SwitchController.Request.STRICT, activate_asap=True,
        )
        if not self._call(self._switch.call_async(request), f"switching to {on}").ok:
            raise RuntimeError(f"the controller manager refused to switch to {on}")
        self._streaming = streaming
        self._stream = None

    def _command_gripper(self, cal: float) -> None:
        """Command the gripper's forward controller (calibrated units)."""
        self._gripper_pub.publish(self._Float64MultiArray(data=[self._jmap.to_rad("gripper", cal)]))

    def _stream_setpoint(self, joints: dict) -> dict | None:
        """The whole body setpoint, in :data:`BODY` order, that ``joints`` makes:
        the body joints it names, the last streamed (or measured) value for the
        rest; None when it names none. Switches the forward controller in."""
        body = {k: v for k, v in joints.items() if k in BODY}
        if not body:
            return None
        self._use_stream(True)
        base = self._stream or self.get_joints()
        self._stream = {k: body.get(k, base[k]) for k in BODY}
        return dict(self._stream)

    def move_to(
        self, target: dict, tolerance_deg: float = 2.5, stop_check=None, limits: dict | None = None,
    ) -> str:
        """One trajectory goal for the body, a command for the gripper; returns
        ``"arrived"``, ``"blocked"`` (the controller ended the goal outside
        ``tolerance_deg``) or ``"stopped"`` (``stop_check`` fired; the arm holds
        where it is). A gripper-only move ends when the jaw stops."""
        from builtin_interfaces.msg import Duration
        from control_msgs.msg import JointTolerance
        from trajectory_msgs.msg import JointTrajectoryPoint

        from common.servo_profile import shared_profiler

        previous = self._send_limits
        self._send_limits = limits
        try:
            target = clamp_joints(target, limits)
            start = self.get_joints()
            body = {k: v for k, v in target.items() if k in BODY}
            eta_s = float(shared_profiler().plan(start, body or target).eta_s)
            sent_at = time.time()
            if "gripper" in target:
                self._command_gripper(target["gripper"])
            if body:
                self._use_stream(False)
                goal = self._FollowJointTrajectory.Goal()
                goal.trajectory.joint_names = list(body)
                point = JointTrajectoryPoint()
                point.positions = [self._jmap.to_rad(k, v) for k, v in body.items()]
                point.time_from_start = Duration(
                    sec=int(eta_s), nanosec=int(round((eta_s - int(eta_s)) * 1e9)) % 1_000_000_000,
                )
                goal.trajectory.points = [point]
                goal.goal_tolerance = [JointTolerance(name=k, position=math.radians(tolerance_deg)) for k in body]
                how = self._run(goal, eta_s, stop_check)
            else:
                how = self._settle_jaw(start["gripper"], target["gripper"], eta_s, tolerance_deg, stop_check)
            joints = self.get_joints()
            worst = max(body or list(target), key=lambda k: abs(joints.get(k, target[k]) - target[k]))
            logger.log(
                logging.WARNING if how == "blocked" else logging.INFO,
                "move_to %s in %.2f s (servo profile %.2f s): %.1f deg left on %s%s",
                how, time.time() - sent_at, eta_s,
                abs(joints.get(worst, target[worst]) - target[worst]), worst,
                "" if how == "arrived" else f" | at {joints_text(joints)} | goal {joints_text(target)}",
            )
            return how
        finally:
            self._send_limits = previous

    def _run(self, goal, eta_s: float, stop_check) -> str:
        """Send ``goal`` and wait for the controller's result, cancelling it on a stop."""
        from control_msgs.action import FollowJointTrajectory

        handle = self._call(self._trajectory.send_goal_async(goal), "the trajectory goal")
        if not handle.accepted:
            raise RuntimeError("arm_controller rejected the trajectory goal")
        result = handle.get_result_async()
        finished = threading.Event()

        def done(_f) -> None:
            finished.set()
            with self._cond:
                self._cond.notify_all()

        result.add_done_callback(done)
        budget = eta_s + self._goal_time_s + REPLY_S
        if not self._wait(finished.is_set, budget, "trajectory result", stop_check):
            self._call(handle.cancel_goal_async(), "cancelling the trajectory")
            return "stopped"
        code = result.result().result.error_code
        if code == FollowJointTrajectory.Result.SUCCESSFUL:
            return "arrived"
        if code == FollowJointTrajectory.Result.GOAL_TOLERANCE_VIOLATED:
            return "blocked"
        raise RuntimeError(f"arm_controller ended the trajectory with error code {code}: {result.result().result.error_string}")

    def _settle_jaw(self, start: float, goal: float, eta_s: float, tolerance_deg: float, stop_check) -> str:
        """Wait until the jaw is within ``tolerance_deg`` of ``goal``, or has come
        ``tolerance_deg`` closer to it and stopped (on an object, or short of a
        readback that need not converge). A jaw squeezing an object jitters by an
        encoder step a tick, so its speed alone says nothing until it has moved."""
        toward = math.copysign(1.0, goal - start)

        def still() -> bool:
            jaw = self._state["gripper"]
            if abs(jaw - goal) <= tolerance_deg:
                return True
            return (jaw - start) * toward >= tolerance_deg and abs(self._jaw_speed) < self._jaw_still_rad_s

        if not self._wait(still, eta_s + self._goal_time_s + REPLY_S, "end of the gripper move", stop_check):
            self._command_gripper(self.get_joints()["gripper"])
            return "stopped"
        return "arrived"

    def set_gripper_torque_limit(self, value: int) -> None:
        """Cap the gripper at ``value`` (0-1000) of stall torque: the arm server's
        ``gripper_torque_limit_nm`` parameter, written on its next control tick."""
        from rclpy.parameter import Parameter

        value = int(min(max(int(value), 0), 1000))
        nm = value / 1000.0 * self._stall_nm
        results = self._call(self._params.set_parameters([Parameter(TORQUE_PARAMETER, value=nm)]), TORQUE_PARAMETER)
        if not all(r.successful for r in results.results):
            raise RuntimeError(f"the arm server refused {TORQUE_PARAMETER}={nm:.3f}")
        self.gripper_torque_limit = value

    def reset_twin(self) -> bool:
        """Park through the controllers, then ask the twin to put the props back
        (``/sim/reset_scene``); False when no twin serves it. The controller
        must end up holding the parked pose: a twin snapped there behind its
        back is pulled back to the controller's last goal."""
        from common.motion import parked_pose

        if not self._reset.service_is_ready():
            return False
        parked = parked_pose()
        how = self.move_to(parked)
        result = self._call(self._reset.call_async(self._Trigger.Request()), "the twin reset")
        if how != "arrived":
            # The controller holds where the park move ended, not the parked pose.
            self.move_to(parked)
        return bool(result.success)

    def close(self) -> None:
        self._spin.close()
