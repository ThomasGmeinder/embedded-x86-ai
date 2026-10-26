# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: the SO-101 bringup.

``hardware_type:=mujoco|real|mock_components`` runs ros2_so_arm's controllers
bringup (``so_arm101_description``) on the MuJoCo twin, the Feetech bus or mock
hardware, and MoveIt's ``move_group`` (``so_arm101_moveit_config``). The twin
loads the scene in the config's ``sim:`` section with its ``home`` keyframe,
and ``config/mujoco_plugins.yaml`` publishes its cameras and props.

``hardware_type:=topic`` is ros2_control over the arm server's topics:
robot_state_publisher publishes the description, ``ros2_control_node`` runs
the controllers at ``robot.control_fps``, and ``topic_based_ros2_control``
talks to the arm server (or the twin) on ``/so101/hardware/joint_commands`` and
``/so101/hardware/joint_states``, in radians; the controller manager finds it
in ``ros_ws/`` (bootstrap.sh builds it). Run it with the venv's Python, after
sourcing ROS and ``ros_ws/install/setup.bash``:

    .venv/bin/python $(which ros2) launch launch/so101_control.launch.py
    ... launch/so101_control.launch.py gripper:=action
    ... launch/so101_control.launch.py hardware_type:=mujoco
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vla_pipeline.utils.config import DEFAULT_CONFIG

PLUGINS_FILE = Path(__file__).resolve().parents[1] / "config" / "mujoco_plugins.yaml"
HARDWARE_TYPES = ("topic", "mujoco", "real", "mock_components")


def _bringup(cfg: dict, hardware_type: str) -> list:
    """ros2_so_arm's controllers bringup and move_group."""
    from ament_index_python.packages import get_package_share_path

    from so101_sim.config import load_sim_config, scene_path

    sim = load_sim_config(cfg)
    mujoco = hardware_type == "mujoco"
    description = get_package_share_path("so_arm101_description")
    moveit = get_package_share_path("so_arm101_moveit_config")
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(description / "launch" / "controllers_bringup.launch.py")),
            launch_arguments={
                "hardware_type": hardware_type,
                "usb_port": cfg["robot"]["motor_port"],
                "mujoco_model": str(scene_path(sim)),
                "mujoco_headless": str(not sim["viewer"]).lower(),
                "mujoco_initial_keyframe": "home",
                "ros2_control_params_file": str(PLUGINS_FILE) if mujoco else "",
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(moveit / "launch" / "move_group.launch.py")),
            launch_arguments={"use_sim_time": str(mujoco)}.items(),
        ),
    ]


def _nodes(context):
    import yaml

    from vla_pipeline.robot.arm_interface import JOINT_LIMITS

    from so101_sim.control import (
        ARM_CONTROLLER,
        GRIPPER_CONTROLLERS,
        HARDWARE_COMMANDS,
        HARDWARE_STATES,
        SERVER_START_S,
        STREAM_CONTROLLER,
        bridge_prefix_path,
    )
    from so101_sim.description import CONTROLLERS_FILE, robot_description

    config = LaunchConfiguration("config").perform(context)
    gripper = LaunchConfiguration("gripper").perform(context)
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if hardware_type != "topic":
        return _bringup(cfg, hardware_type)
    urdf = robot_description(
        cfg, JOINT_LIMITS, joint_commands_topic=HARDWARE_COMMANDS, joint_states_topic=HARDWARE_STATES,
        initial_state_timeout=SERVER_START_S,
    )
    controllers = str(CONTROLLERS_FILE)
    spawn = ["--controller-manager", "/controller_manager", "--param-file", controllers]
    inactive = [STREAM_CONTROLLER, *(name for key, name in GRIPPER_CONTROLLERS.items() if key != gripper)]
    return [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            parameters=[{"robot_description": urdf}],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="ros2_control_node",
            parameters=[controllers, {"update_rate": int(round(float(cfg["robot"]["control_fps"])))}],
            remappings=[("~/robot_description", "/robot_description")],
            additional_env={"AMENT_PREFIX_PATH": bridge_prefix_path()},
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["joint_state_broadcaster", ARM_CONTROLLER, GRIPPER_CONTROLLERS[gripper], *spawn],
            output="screen",
        ),
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=[*inactive, "--inactive", *spawn],
            output="screen",
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    """robot_state_publisher, the controller manager, its controllers and, on ros2_so_arm, move_group."""
    from so101_sim.control import GRIPPER_CONTROLLERS

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(DEFAULT_CONFIG),
                description="the pipeline config",
            ),
            DeclareLaunchArgument(
                "hardware_type",
                default_value="topic",
                choices=HARDWARE_TYPES,
                description="topic: ros2_control over the arm server's topics; mujoco|real|mock_components: ros2_so_arm and move_group",
            ),
            DeclareLaunchArgument(
                "gripper",
                default_value="forward",
                choices=sorted(GRIPPER_CONTROLLERS),
                description="the active gripper controller; the other is loaded inactive, as is the streaming arm one",
            ),
            OpaqueFunction(function=_nodes),
        ]
    )
