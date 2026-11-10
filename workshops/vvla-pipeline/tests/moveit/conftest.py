# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The twin bringup, once for the session, and what its tests share.

One twin serves every module: a second one would start its clock at zero
beside the first. A test module marks its launch tests
``pytest.mark.launch(fixture=conftest.twin_bringup)`` after ``import conftest``:
importing the fixture by name would define it again in that module, and a
second bringup would start. Tests reset the twin where they need the home scene.
"""

from __future__ import annotations

import os
import queue
import threading

import pytest

launch_pytest = pytest.importorskip("launch_pytest")  # ROS 2 Jazzy

PAN = "shoulder_pan_joint"
OPTICAL = "wrist_cam_optical_frame"


def off(joints: dict, target: dict) -> float:
    """The largest absolute joint error between ``joints`` and ``target``."""
    return max(abs(joints[name] - value) for name, value in target.items())


@pytest.fixture(scope="session")
def ros_domain():
    """A ROS domain no other process holds, for this session's twin, nodes and kernels."""
    import domain_coordinator

    with domain_coordinator.domain_id() as domain:
        os.environ["ROS_DOMAIN_ID"] = str(domain)
        yield domain


@pytest.fixture(scope="session")
def pipeline_cfg():
    from vla_pipeline.utils.config import load_config

    return load_config()


@launch_pytest.fixture(scope="session")
def twin_bringup(ros_domain):
    """``launch/so101_control.launch.py hardware_type:=mujoco``."""
    from launch import LaunchDescription
    from launch.actions import IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    from so101_sim.control import LAUNCH_FILE

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(LAUNCH_FILE)),
            launch_arguments={"hardware_type": "mujoco"}.items(),
        ),
        launch_pytest.actions.ReadyToTest(),
    ])


@pytest.fixture(scope="session")
def ros_node(twin_bringup):
    """An rclpy node in its own context, spun on its own thread. It keeps the
    system clock: on the twin's it would take a /clock message every 2 ms in Python."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    context = rclpy.Context()
    rclpy.init(context=context)
    node = rclpy.create_node("moveit_test", context=context)
    executor = MultiThreadedExecutor(context=context)
    executor.add_node(node)
    spinner = threading.Thread(target=executor.spin, name="moveit-test", daemon=True)
    spinner.start()
    yield node
    executor.shutdown()
    spinner.join()
    node.destroy_node()
    rclpy.shutdown(context=context)


@pytest.fixture(scope="session")
def arm(ros_node, pipeline_cfg):
    """The session's MoveItArm, and with it the process's one MoveItPy: a second
    MoveItPy after the first shuts rclcpp down loses the twin's joint states."""
    from vla_pipeline.robot.moveit_arm import MoveItArm

    robot = MoveItArm(pipeline_cfg, node_name="moveit_test_arm", sim=True)
    yield robot
    robot.close()


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Remember the test result so the bag fixture can keep a failed run."""
    outcome = yield
    report = outcome.get_result()
    setattr(item, "rep_" + report.when, report)


class Bag:
    """An MCAP bag of the twin topics a test is watching. ``close`` deletes it when the test passed."""

    def __init__(self, path, node):
        import threading

        import rosbag2_py
        from mujoco_ros2_control_msgs.msg import FreeJointStateArray
        from sensor_msgs.msg import JointState

        self.path = path
        self._node = node
        self._lock = threading.Lock()
        self._counts: dict[str, int] = {}
        self._open = True
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(uri=path, storage_id="mcap"),
            rosbag2_py.ConverterOptions("cdr", "cdr"),
        )
        topics = (
            ("/joint_states", "sensor_msgs/msg/JointState", JointState),
            ("/free_joint_states", "mujoco_ros2_control_msgs/msg/FreeJointStateArray", FreeJointStateArray),
        )
        self._subs = []
        for index, (name, type_name, msg_type) in enumerate(topics):
            self._writer.create_topic(rosbag2_py.TopicMetadata(index, name, type_name, "cdr"))
            self._subs.append(node.create_subscription(msg_type, name, self._on(name), 10))

    def count(self, topic: str) -> int:
        with self._lock:
            return self._counts.get(topic, 0)

    def _on(self, topic: str):
        from rclpy.serialization import serialize_message

        def on_msg(msg, topic=topic):
            stamp = msg.header.stamp
            when = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            with self._lock:
                if not self._open:
                    return
                self._writer.write(topic, serialize_message(msg), when)
                self._counts[topic] = self._counts.get(topic, 0) + 1

        return on_msg

    def close(self, passed: bool) -> None:
        import shutil

        with self._lock:
            if not self._open:
                return
            self._open = False
            self._writer.close()
        for sub in self._subs:
            self._node.destroy_subscription(sub)
        if passed:
            shutil.rmtree(self.path, ignore_errors=True)
        else:
            print(f"bag kept: {self.path}")


@pytest.fixture(autouse=True)
def bag(request, ros_node):
    """Record ``/joint_states`` and ``/free_joint_states`` for this test. The file stays when the test fails."""
    import os
    import re

    name = re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)
    recorder = Bag(f"/tmp/vvla-bag-{name}-{os.getpid()}", ros_node)
    try:
        yield recorder
    finally:
        report = getattr(request.node, "rep_call", None)
        recorder.close(report is not None and report.passed)


def next_message(node, msg_type, topic: str, accept=lambda _m: True, timeout_s: float = 30.0):
    """The first message on ``topic`` that ``accept`` takes."""
    got: queue.Queue = queue.Queue()
    sub = node.create_subscription(msg_type, topic, lambda m: accept(m) and got.put(m), 10)
    try:
        return got.get(timeout=timeout_s)
    except queue.Empty:
        pytest.fail(f"nothing on {topic} within {timeout_s} s")
    finally:
        node.destroy_subscription(sub)
