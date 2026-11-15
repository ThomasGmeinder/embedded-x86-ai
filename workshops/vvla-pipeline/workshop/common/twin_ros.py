# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The digital twin behind ROS 2, for one notebook kernel (PROVIDED).

Section 5 of ``notebooks/03_integration/02_digital_twin.ipynb`` drives the
twin the way the pipeline does. :class:`TwinSession` starts
``launch/so101_control.launch.py hardware_type:=mujoco`` on a ROS domain of
its own. :class:`~vla_pipeline.robot.moveit_arm.MoveItArm` is constructed
before this process initializes rclpy: doing that first makes ``MoveItPy``
not return. Props are ``/free_joint_states``.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from so101_sim.control import LAUNCH_FILE, sensor_qos
from vla_pipeline.robot.ros2_control_arm import Spinner
from vla_pipeline.utils.config import DEFAULT_CONFIG
from vla_pipeline.vision.camera import from_image_msg

logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_ID = 87  # override with TWIN_ROS_DOMAIN_ID
# Discovery stays on this machine. The profile does not turn builtin transports
# off, so FASTDDS_BUILTIN_TRANSPORTS from scripts/ryzen_ai_env.sh still applies.
_LOCALHOST_PROFILE = Path(__file__).with_name("twin_localhost.xml")
_domain_node = None  # kept so the DDS participant outlives the call
_domain_context = None  # the private context that holds _domain_node, when used


def twin_domain_id() -> int:
    """The ROS domain the twin runs on: ``TWIN_ROS_DOMAIN_ID``, else 87."""
    return int(os.environ.get("TWIN_ROS_DOMAIN_ID", DEFAULT_DOMAIN_ID))


def use_private_domain(domain_id: int, *, private_context: bool = False) -> None:
    """Put this process's ROS 2 on ``domain_id``, discovery on this machine only.

    ROS reads the domain when ``rclpy.init()`` runs, so this must run before
    anything in this kernel starts ROS. A kernel already running ROS on
    another domain has to be restarted.

    ``twin_localhost.xml`` keeps discovery on this host (no multicast, peers
    at 127.0.0.1, other hosts ignored). ``ROS_AUTOMATIC_DISCOVERY_RANGE`` is
    not set to ``LOCALHOST``: that range disables Fast DDS builtin transports,
    and ``FASTDDS_BUILTIN_TRANSPORTS`` would no longer apply.

    Call this before the first Vitis AI session in the process. ``LARGE_DATA``
    does not discover peers if that session is opened before the DDS
    participant. ``rclpy.init`` alone does not create one; this function
    holds a node so the participant exists when it returns. The localhost
    discovery range used to hide the order, because it builds its own UDP
    transport afterwards. The pipeline's arm and camera clients start DDS
    before the models for the same reason.

    ``private_context`` holds that node on its own context and leaves the
    default context uninitialized. ``MoveItPy`` has to be the first init of
    the default context, so a kernel that will construct it passes True.
    """
    global _domain_context, _domain_node
    import rclpy

    if not private_context and rclpy.ok():
        current = rclpy.get_default_context().get_domain_id()
        if current != domain_id:
            raise RuntimeError(
                f"ROS 2 already runs in this kernel on domain {current}; the twin needs "
                f"domain {domain_id}. Restart the kernel and run the notebook from the top."
            )
    else:
        os.environ["ROS_DOMAIN_ID"] = str(domain_id)
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST":
            del os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(_LOCALHOST_PROFILE)
        if private_context:
            if _domain_node is None:
                _domain_context = rclpy.Context()
                rclpy.init(context=_domain_context)
                _domain_node = rclpy.create_node("twin_domain", context=_domain_context)
            return
        rclpy.init()
    if _domain_node is None:
        _domain_node = rclpy.create_node("twin_domain")


class TwinCamera:
    """Newest frame of a twin camera topic, with the time its scene was captured.

    Same ``read() -> (ok, frame)`` as the other cameras. ``last_stamp`` is
    the capture time of the frame the last ``read()`` returned, so
    :class:`~vla_pipeline.console.FanoutCamera` and :meth:`read_after` can
    tell a frame of the settled arm from one rendered just before.
    """

    def __init__(self, role: str, topic: str, stale_after_s: float = 1.0):
        from sensor_msgs.msg import Image

        self.role = role
        self.topic = topic
        self._stale_after = float(stale_after_s)
        self._cond = threading.Condition()
        self._frame = None
        self._stamp = 0.0
        self._arrived = 0.0
        self._closed = False
        self.last_stamp: float | None = None
        self._spin = Spinner(f"twin_camera_{role}")
        self._spin.node.create_subscription(Image, topic, self._on_image, sensor_qos())

    def _on_image(self, msg) -> None:
        frame = from_image_msg(msg)
        if frame is None:
            return
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        with self._cond:
            self._frame, self._stamp, self._arrived = frame, stamp, time.time()
            self._cond.notify_all()

    def read(self):
        """Newest frame, or ``(False, None)`` when the stream is stale or missing."""
        with self._cond:
            if self._frame is None or time.time() - self._arrived > self._stale_after:
                return False, None
            self.last_stamp = self._stamp
            return True, self._frame

    def read_after(self, settled_at: float):
        """First frame whose scene was captured after ``settled_at``, within the stream's staleness limit."""
        ok, frame, stamp = self.next_frame(settled_at, max(self._stale_after, 0.5))
        if ok:
            self.last_stamp = stamp
        return ok, frame

    def next_frame(self, after: float, timeout: float | None = None):
        """``(ok, frame, captured)``: the first frame captured after ``after``;
        not ok once closed, or when ``timeout`` passes first."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._frame is not None and self._stamp > after), timeout)
            if self._closed or self._frame is None or self._stamp <= after:
                return False, None, after
            return True, self._frame, self._stamp

    def frame_arrived_after(self, wall: float, timeout: float):
        """``(ok, frame, captured)`` for the first image that arrives after ``wall``.

        ``wall`` is ``time.time()`` from before the wait. The capture stamp can
        sit behind the latest joint stamp; arrival is what shows the stream
        is still publishing.
        """
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._arrived > wall, timeout)
            if self._closed or self._frame is None or self._arrived <= wall:
                return False, None, self._stamp
            self.last_stamp = self._stamp
            return True, self._frame, self._stamp

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        self._spin.close()

    release = close


class TwinScene:
    """Props from ``/free_joint_states``: ``xyz``, ``quat`` (w, x, y, z).

    The plugin does not report contact, so ``grasped`` is false. ``settled``
    is every body slower than 2 cm/s.
    """

    def __init__(self):
        from mujoco_ros2_control_msgs.msg import FreeJointStateArray

        self._cond = threading.Condition()
        self._latest: dict | None = None
        self._spin = Spinner("twin_scene")
        self._spin.node.create_subscription(FreeJointStateArray, "/free_joint_states", self._on_props, 10)

    def _on_props(self, msg) -> None:
        stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        bodies = {}
        settled = True
        for joint in msg.free_joints:
            pose, linear = joint.pose.pose, joint.twist.twist.linear
            speed = (linear.x ** 2 + linear.y ** 2 + linear.z ** 2) ** 0.5
            settled = settled and speed < 0.02
            rotation = pose.orientation
            bodies[joint.name] = {
                "xyz": [pose.position.x, pose.position.y, pose.position.z],
                "quat": [rotation.w, rotation.x, rotation.y, rotation.z],
                "grasped": False,
            }
        with self._cond:
            self._latest = {
                "stamp": stamp,
                "bodies": bodies,
                "settled": settled,
                "gripper": {"torque_limit": 0, "overload_torque": 0, "tripped": False},
            }
            self._cond.notify_all()

    def latest(
        self, after: float | None = None, timeout: float | None = 2.0, settled: bool = False,
    ) -> dict | None:
        """The newest scene, or the first one stamped after ``after`` (with
        ``settled``, with every prop at rest); None when ``timeout`` passes
        first (``None``: no limit)."""

        def ready() -> bool:
            s = self._latest
            return (
                s is not None
                and (after is None or s["stamp"] > after)
                and (not settled or s["settled"])
            )

        with self._cond:
            if not self._cond.wait_for(ready, timeout):
                return None
            return self._latest

    def reading(self, body: str, after: float | None = None, timeout: float = 2.0) -> dict | None:
        """``{"xyz", "quat", "grasped", "stamp"}`` for ``body``, or None."""
        scene = self.latest(after=after, timeout=timeout)
        if scene is None or body not in scene["bodies"]:
            return None
        return {**scene["bodies"][body], "stamp": scene["stamp"]}

    def close(self) -> None:
        self._spin.close()


def _tail(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return "(no log)"


class TwinSession:
    """The MuJoCo MoveIt bringup and its clients, owned by this process.

    ``MoveItPy`` is created before this process calls ``rclpy.init``. A
    default context that already exists makes that constructor not return.
    The launch dies with this process, so a crashed kernel leaves no twin.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        config_path: str | os.PathLike | None = None,
        scene: dict | None = None,
        domain_id: int | None = None,
        ready_timeout_s: float = 60.0,
    ):
        self.domain_id = twin_domain_id() if domain_id is None else int(domain_id)
        self.config_path = Path(config_path or DEFAULT_CONFIG)
        self._arm_topic = (cfg.get("cameras") or {}).get("arm", {}).get(
            "topic", "/cameras/arm/image_raw"
        )
        self._scene_topic = (cfg.get("cameras") or {}).get("scene", {}).get(
            "topic", "/cameras/scene/image_raw"
        )
        self.arm = self.armcam = self.view = self.scene = None
        self._proc = None
        self._closed = False
        self._scene = scene
        self._domain_lock = None

        # Environment only. rclpy.init before MoveItPy makes that constructor not return.
        os.environ["ROS_DOMAIN_ID"] = str(self.domain_id)
        if os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") == "LOCALHOST":
            del os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"]
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = str(_LOCALHOST_PROFILE)
        self._refuse_if_owned()
        from so101_sim.domain_lock import acquire

        self._domain_lock = acquire(self.domain_id)

        import shutil

        ros2 = shutil.which("ros2")
        if ros2 is None:
            raise RuntimeError("ros2 is not on PATH: source /opt/ros/jazzy/setup.bash first")
        cmd = [
            sys.executable, "-m", "so101_sim.parent_death", sys.executable, ros2, "launch",
            str(LAUNCH_FILE), f"config:={self.config_path}", "hardware_type:=mujoco",
        ]
        fd, log_name = tempfile.mkstemp(prefix="twin_node_", suffix=".log")
        self.log_path = Path(log_name)
        self.controllers_log_path = self.log_path
        with os.fdopen(fd, "w") as log:
            self._proc = subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                start_new_session=True,
            )
        atexit.register(self.close)
        try:
            self._wait_ready(ready_timeout_s, cfg)
        except BaseException:
            self.close()
            raise

    @property
    def pid(self) -> int | None:
        return None if self._proc is None else self._proc.pid

    @property
    def controllers_pid(self) -> int | None:
        """The bringup is one process, so this is the launch pid."""
        return self.pid

    def _refuse_if_owned(self) -> None:
        """The twin node holds its domain's lock (``so101_sim.domain_lock``) while it runs."""
        from so101_sim.domain_lock import DomainTaken, owner

        pid = owner(self.domain_id)
        if pid is not None:
            raise DomainTaken(self.domain_id, pid)

    def _exited(self) -> RuntimeError | None:
        if self._proc.poll() is None:
            return None
        return RuntimeError(f"the twin node exited with code {self._proc.returncode}:\n{_tail(self.log_path)}")

    def _wait_ready(self, timeout_s: float, cfg: dict) -> None:
        """The node's first joint state, then the arm client (the controllers
        up), then the first arm frame, within ``timeout_s``."""
        from sensor_msgs.msg import JointState

        deadline = time.time() + timeout_s

        def remaining() -> float:
            return max(deadline - time.time(), 0.0)

        def not_ready(what: str) -> Exception:
            return self._exited() or TimeoutError(
                f"the twin node sent no {what} within {timeout_s:.0f} s:\n{_tail(self.log_path)}"
            )

        self._wait_bringup(timeout_s, not_ready, remaining)
        try:
            from vla_pipeline.robot.moveit_arm import MoveItArm

            self.arm = MoveItArm(cfg, sim=True)
        except RuntimeError as e:
            raise TimeoutError(f"{e}\n{_tail(self.log_path)}") from None
        if self._scene is not None:
            self._place(self._scene)
        self.armcam = TwinCamera("arm", self._arm_topic)
        self.view = TwinCamera("view", self._scene_topic)
        self.scene = TwinScene()
        if not self.armcam.next_frame(0.0, remaining())[0]:
            raise not_ready("arm camera frame")
        logger.info(
            "twin up on ROS domain %d (pid %d, log %s)", self.domain_id, self.pid, self.log_path
        )

    def _wait_bringup(self, timeout_s: float, not_ready, remaining) -> None:
        """Joint states, then the trajectory action, on a private context.

        The default context stays uninitialized so ``MoveItPy`` can take it.
        """
        import rclpy
        from control_msgs.action import FollowJointTrajectory
        from rclpy.action import ActionClient
        from rclpy.executors import SingleThreadedExecutor
        from sensor_msgs.msg import JointState

        context = rclpy.Context()
        rclpy.init(context=context)
        node = rclpy.create_node("twin_bringup_wait", context=context)
        seen = threading.Event()
        node.create_subscription(JointState, "/joint_states", lambda _m: seen.set(), sensor_qos())
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        spin = threading.Thread(target=executor.spin, daemon=True, name="twin-bringup-wait")
        spin.start()
        try:
            if not seen.wait(timeout_s) or self._exited() is not None:
                raise not_ready("joint state")
            trajectory = ActionClient(
                node, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory",
            )
            try:
                if not trajectory.wait_for_server(timeout_sec=remaining()) or self._exited() is not None:
                    raise not_ready("trajectory action")
            finally:
                trajectory.destroy()
        finally:
            executor.shutdown()
            spin.join(timeout=2.0)
            node.destroy_node()
            rclpy.shutdown(context=context)

    def _place(self, scene: dict) -> None:
        """Put each free body where ``scene`` recorded it (x, y, z, then w x y z)."""
        from geometry_msgs.msg import Pose
        from mujoco_ros2_control_msgs.msg import FreeJointState
        from mujoco_ros2_control_msgs.srv import SetFreeJointState

        from vla_pipeline.robot.moveit_arm import _call

        request = SetFreeJointState.Request()
        for name, qpos in (scene.get("bodies") or {}).items():
            joint = FreeJointState()
            joint.name = name
            joint.pose.pose = Pose()
            joint.pose.pose.position.x, joint.pose.pose.position.y, joint.pose.pose.position.z = qpos[:3]
            joint.pose.pose.orientation.w = qpos[3]
            joint.pose.pose.orientation.x, joint.pose.pose.orientation.y, joint.pose.pose.orientation.z = qpos[4:7]
            request.free_joints.append(joint)
        if not request.free_joints:
            return
        reply = _call(
            self.arm.node, SetFreeJointState, "/mujoco_ros2_control_node/set_free_joint_state", request, 15.0,
        )
        if not reply.success:
            raise RuntimeError(f"set_free_joint_state: {reply.message}")

    def reset_scene(self) -> bool:
        """The ``home`` keyframe: the arm parked and every prop back on the scene."""
        self.arm.reset()
        return True

    def close(self) -> None:
        """Stop the clients and the twin node. Safe to call more than once."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        for part in (self.arm, self.armcam, self.view, self.scene):
            if part is None:
                continue
            try:
                part.close()
            except Exception as e:
                logger.debug("twin client close: %s", e)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            # parent_death ignores SIGINT and, on SIGTERM, signals its process group.
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=5.0)
        if proc is not None and proc.returncode == 0:
            self.log_path.unlink(missing_ok=True)
        if self._domain_lock is not None:
            self._domain_lock.close()
            self._domain_lock = None
