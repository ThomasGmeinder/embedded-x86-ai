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
