# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: the cameras, and the SO-101 under ros2_control.

A real webcam is ``v4l2_camera``, with the device and topic from
``config/pipeline.yaml``. The arm is ``launch/so101_control.launch.py``:
``hardware_type:=real`` on the Feetech bus, ``mock_components`` for
``dry_run:=true``, and ``mujoco`` for ``digital_twin:=true`` (the wrist and
scene images come from the MuJoCo camera plugins; the mount webcam still starts).

    ros2 launch launch/strix_vla.launch.py
    ros2 launch launch/strix_vla.launch.py dry_run:=true     # no motors
    ros2 launch launch/strix_vla.launch.py digital_twin:=true  # MuJoCo twin, no arm
    ros2 launch launch/strix_vla.launch.py config:=/path/to/pipeline.yaml
    ros2 launch launch/strix_vla.launch.py cameras:=false    # the arm only

Run it with the venv's Python, after sourcing ROS
(``.venv/bin/python $(which ros2) launch ...``).

The orchestrator (``python -m vla_pipeline.main``) is intentionally NOT part
of this launch: it wants a TTY + preview windows. Start it in its own
terminal once the nodes are up.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

from vla_pipeline.utils.config import DEFAULT_CONFIG


def _arm(context):
    """ros2_so_arm's bringup and move_group: MuJoCo, mock hardware, or the Feetech bus."""
    if LaunchConfiguration("arm").perform(context) != "true":
        return []
    from so101_sim.control import LAUNCH_FILE

    config = LaunchConfiguration("config").perform(context)
    twin = LaunchConfiguration("digital_twin").perform(context) == "true"
    dry_run = LaunchConfiguration("dry_run").perform(context) == "true"
    hardware = "mujoco" if twin else "mock_components" if dry_run else "real"
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(LAUNCH_FILE)),
            launch_arguments={
                "config": config,
                "hardware_type": hardware,
                "control_core": LaunchConfiguration("control_core").perform(context),
                "twin_core": LaunchConfiguration("twin_core").perform(context),
            }.items(),
        ),
    ]


def _cameras(context):
    """``v4l2_camera`` for each real webcam.

    On the twin, only the mount webcam starts. The wrist image and the scene
    overview come from the MuJoCo camera plugins.
    """
    if LaunchConfiguration("cameras").perform(context) != "true":
        return []
    twin = LaunchConfiguration("digital_twin").perform(context) == "true"
    import yaml

    from launch_ros.actions import Node

    config = LaunchConfiguration("config").perform(context)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    core = str(LaunchConfiguration("perception_core").perform(context) or "").strip()
    prefix = f"taskset -c {core}" if core else ""
    nodes = []
    roles = ("mount",) if twin else ("mount", "arm")
    for role in roles:
        cam = ((cfg.get("cameras") or {}).get(role) or {})
        if not cam.get("use_ros2", False) or cam.get("source", "device") != "device":
            continue
        node = dict(
            package="v4l2_camera",
            executable="v4l2_camera_node",
            name=f"camera_{role}",
        )
        if prefix:
            node["prefix"] = prefix
        nodes.append(Node(
            **node,
            parameters=[{
                "video_device": str(cam.get("device", "/dev/video0")),
                "image_size": [int(cam.get("width", 640)), int(cam.get("height", 480))],
            }],
            remappings=[("image_raw", cam.get("topic", f"/cameras/{role}/image_raw"))],
            output="screen",
        ))
    return nodes


def generate_launch_description() -> LaunchDescription:
    """The webcam nodes and the arm bringup."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(DEFAULT_CONFIG),
                description="pipeline YAML with cameras/robot parameters",
            ),
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="mock hardware, no motors",
            ),
            DeclareLaunchArgument(
                "digital_twin",
                default_value="false",
                description="MuJoCo digital twin: controllers, move_group, and the camera plugins",
            ),
            DeclareLaunchArgument(
                "arm",
                default_value="true",
                description="start the arm bringup (robot.use_ros2)",
            ),
            DeclareLaunchArgument(
                "cameras",
                default_value="true",
                description="run the camera nodes (cameras.<role>.use_ros2)",
            ),
            DeclareLaunchArgument("twin_core", default_value="", description="CPU core for the digital twin"),
            DeclareLaunchArgument("control_core", default_value="", description="CPU core for robot control"),
            DeclareLaunchArgument("perception_core", default_value="", description="CPU core for the cameras"),
            OpaqueFunction(function=_cameras),
            OpaqueFunction(function=_arm),
        ]
    )
