# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS 2 launch: the SO-101 bringup.

``hardware_type:=mujoco|real|mock_components`` runs ros2_so_arm's controllers
bringup (``so_arm101_description``) on the MuJoCo twin, the Feetech bus or mock
hardware, MoveIt's ``move_group`` (``so_arm101_moveit_config``), and MoveIt
Servo. Servo runs as a user service: this desktop session inherited rtkit's
200 ms realtime budget, which a child cannot raise, and Servo uses realtime
scheduling while it loads the arm model. The user service manager still has
an unlimited budget. The twin
loads the scene in the config's ``sim:`` section with its ``home`` keyframe,
and ``config/mujoco_plugins.yaml`` publishes its cameras and props.

Run it with the venv's Python, after sourcing ROS and ``ros_ws/install/setup.bash``:

    .venv/bin/python $(which ros2) launch launch/so101_control.launch.py
    ... launch/so101_control.launch.py hardware_type:=real
    ... launch/so101_control.launch.py hardware_type:=mock_components
"""

import os
import shlex
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vla_pipeline.utils.config import DEFAULT_CONFIG

PLUGINS_FILE = Path(__file__).resolve().parents[1] / "config" / "mujoco_plugins.yaml"
HARDWARE_TYPES = ("mujoco", "real", "mock_components")
# scripts/servo_user_service.sh starts Servo. A user service does not inherit
# this process's environment; the script forwards the ROS domain and the
# library path, and it stops the unit when launch signals it.
_SERVO_SERVICE = Path(__file__).resolve().parents[1] / "scripts" / "servo_user_service.sh"


def _ensure_servo_package() -> None:
    """Use an unpacked MoveIt Servo prefix when the apt package is not on the path.

    ``ros-jazzy-moveit-servo`` is the package ``bootstrap.sh`` installs. This
    machine's copy is the unpacked deb until that install is done.
    ``MOVEIT_SERVO_PREFIX`` names a different unpack.
    """
    prefixes = [item for item in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep) if item]
    if any((Path(item) / "share" / "moveit_servo").is_dir() for item in prefixes):
        return
    fallback = Path(os.environ.get(
        "MOVEIT_SERVO_PREFIX",
        Path.home() / ".local/opt/ros-jazzy-moveit-servo-2.12.4",
    ))
    if not (fallback / "share" / "moveit_servo").is_dir():
        return
    os.environ["AMENT_PREFIX_PATH"] = os.pathsep.join([str(fallback), *prefixes])
    lib = str(fallback / "lib")
    current = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = lib if not current else os.pathsep.join([lib, current])


def _user_service_prefix() -> str:
    """Start the following command as a user service, then wait until it exits."""
    return shlex.quote(str(_SERVO_SERVICE))


def _taskset(core: str) -> str:
    core = str(core or "").strip()
    return f"taskset -c {core} " if core else ""


def _servo(mujoco: bool, control_core: str = ""):
    """MoveIt Servo, publishing to the trajectory controller within joint_limits.yaml.

    The shipped panda config is the parameter file. This overrides the group,
    the output topic and the publish period, and it is not the primary planning
    scene: ``move_group`` is. A joint at its position bound holds while the
    others keep streaming: with Servo's default every joint halts, and mimic's
    pan stops wherever the elbow met its bound. Servo slows a stream once a
    self-collision is nearer than ``self_collision_proximity_threshold``. The
    folded SRDF ``home`` pose has the wrist 4.7 mm from the shoulder and the
    upper arm, and 3.7 mm with the elbow on its stop, so the panda config's
    10 mm starts every stream from home at a few percent of its speed. Those
    pairs do collide in other poses, so they stay checked. The process is a user service
    so its realtime budget is the user service manager's, not this session's.
    """
    from launch_param_builder import ParameterBuilder
    from so_arm_utils.launch_utils import MoveItConfigs

    from vla_pipeline.robot.moveit_arm import PLANNING_GROUP, SERVO_NODE, SERVO_PERIOD_S

    _ensure_servo_package()
    configs = MoveItConfigs("so_arm101")
    manager = configs.trajectory_execution["moveit_simple_controller_manager"]
    arm_controller = next(
        name for name in manager["controller_names"] if manager[name]["type"] == "FollowJointTrajectory"
    )
    servo = (
        ParameterBuilder("moveit_servo")
        .yaml("config/panda_simulated_config.yaml")
        .parameter("move_group_name", PLANNING_GROUP)
        .parameter("command_out_topic", f"/{arm_controller}/joint_trajectory")
        .parameter("is_primary_planning_scene_monitor", False)
        .parameter("publish_period", SERVO_PERIOD_S)
        .parameter("halt_all_joints_in_joint_mode", False)
        .parameter("self_collision_proximity_threshold", 0.003)
        .to_dict()
    )
    return Node(
        package="moveit_servo",
        executable="servo_node",
        name=SERVO_NODE,
        prefix=_taskset(control_core) + _user_service_prefix(),
        parameters=[
            {"moveit_servo": servo},
            {"update_period": SERVO_PERIOD_S},
            {"planning_group_name": PLANNING_GROUP},
            configs.robot_description,
            configs.robot_description_semantic,
            configs.robot_description_kinematics,
            configs.joint_limits,
            {"use_sim_time": mujoco},
        ],
        output="screen",
    )


def _bringup(cfg: dict, hardware_type: str, control_core: str = "", twin_core: str = "") -> list:
    """ros2_so_arm's controllers bringup, move_group and MoveIt Servo."""
    from ament_index_python.packages import get_package_share_path

    from so101_sim.config import load_sim_config, scene_path

    sim = load_sim_config(cfg)
    mujoco = hardware_type == "mujoco"
    sim_core = twin_core if mujoco else control_core
    description = get_package_share_path("so_arm101_description")
    moveit = get_package_share_path("so_arm101_moveit_config")
    nodes = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(description / "launch" / "controllers_bringup.launch.py")),
            launch_arguments={
                "hardware_type": hardware_type,
                "usb_port": cfg["robot"]["motor_port"],
                "mujoco_model": str(scene_path(sim)),
                "mujoco_headless": str(not sim["viewer"]).lower(),
                "mujoco_initial_keyframe": "home",
                "ros2_control_params_file": str(PLUGINS_FILE) if mujoco else "",
                "node_prefix": _taskset(sim_core).strip(),
            }.items(),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(moveit / "launch" / "move_group.launch.py")),
            launch_arguments={"use_sim_time": str(mujoco), "prefix": _taskset(control_core).strip()}.items(),
        ),
        _servo(mujoco, control_core),
    ]
    if mujoco:
        # The wrist camera lives in the MuJoCo model, on the gripper. This is that
        # camera's optical frame (x right, y down, z forward) in gripper_link:
        # so101.xml wrist_cam, pos 0 0.055 -0.045, xyaxes "0 -0.8419 0.5396 1 0 0".
        nodes.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            arguments=[
                "--x", "0", "--y", "0.055", "--z", "-0.045",
                "--qx", "0.678586", "--qy", "-0.678586", "--qz", "0.198799", "--qw", "0.198799",
                "--frame-id", "gripper_link",
                "--child-frame-id", "wrist_cam_optical_frame",
            ],
            parameters=[{"use_sim_time": True}],
            **({"prefix": _taskset(twin_core).strip()} if twin_core else {}),
        ))
    return nodes


def _nodes(context):
    import yaml

    config = LaunchConfiguration("config").perform(context)
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    with open(config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return _bringup(
        cfg, hardware_type,
        LaunchConfiguration("control_core").perform(context),
        LaunchConfiguration("twin_core").perform(context),
    )


def generate_launch_description() -> LaunchDescription:
    """ros2_so_arm's controllers, MoveIt's move_group and MoveIt Servo."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=str(DEFAULT_CONFIG),
                description="the pipeline config",
            ),
            DeclareLaunchArgument(
                "hardware_type",
                default_value="mujoco",
                choices=HARDWARE_TYPES,
                description="mujoco: the digital twin; real: the Feetech bus; mock_components: a dry run",
            ),
            DeclareLaunchArgument("control_core", default_value="", description="CPU core for move_group and Servo"),
            DeclareLaunchArgument("twin_core", default_value="", description="CPU core for MuJoCo when hardware_type is mujoco"),
            OpaqueFunction(function=_nodes),
        ]
    )
