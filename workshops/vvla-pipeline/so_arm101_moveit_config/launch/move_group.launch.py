from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from so_arm_utils.launch_utils import MoveItConfigs


def _flat_parameters(tree, prefix=""):
    """Dotted parameter names. A launch parameter dict cannot hold a nested map."""
    flat = {}
    for key, value in tree.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flat_parameters(value, name))
        else:
            flat[name] = value
    return flat


def generate_launch_description():
    from launch.actions import OpaqueFunction

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="False",
        description="Whether to use simulation time to launch the MoveIt node.",
    )
    prefix_arg = DeclareLaunchArgument(
        "prefix",
        default_value="",
        description="Optional command prefix, for example taskset -c N",
    )

    def _node(context):
        sensors = Path(get_package_share_directory("so_arm101_moveit_config")) / "config" / "sensors_3d.yaml"
        moveit_configs = MoveItConfigs(robot_name="so_arm101")
        prefix = str(LaunchConfiguration("prefix").perform(context) or "").strip()
        kwargs = {"prefix": prefix} if prefix else {}
        return [Node(
            package="moveit_ros_move_group",
            executable="move_group",
            parameters=[
                moveit_configs.to_dict(),
                _flat_parameters(yaml.safe_load(sensors.read_text())),
                {"capabilities": "move_group/ExecuteTaskSolutionCapability"},
                {"use_sim_time": LaunchConfiguration("use_sim_time")},
            ],
            **kwargs,
        )]

    return LaunchDescription(
        [
            use_sim_time_arg,
            prefix_arg,
            OpaqueFunction(function=_node),
        ],
    )
