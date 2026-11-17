# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""run_pipeline.sh starts strix_vla.launch.py twice: ``arm:=false cameras:=true``
and ``arm:=true cameras:=false``. ros2 launch ends every part of a launch when
one part raises, so each of those launches must hold only its own nodes.

The launch description is evaluated with the real ``launch`` library. No
process starts.
"""

from __future__ import annotations

import importlib.util

import pytest
import yaml

from notebook_harness import PIPE_ROOT

pytestmark = pytest.mark.sim

LAUNCH = PIPE_ROOT / "launch" / "strix_vla.launch.py"


def _entities(tmp_path, **arguments):
    """What each OpaqueFunction of the launch returns for these arguments."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction

    config = tmp_path / "pipeline.yaml"
    config.write_text(yaml.safe_dump({"cameras": {
        "mount": {"use_ros2": True, "device": "/dev/video0", "topic": "/cameras/mount/image_raw"},
        "arm": {"use_ros2": True, "device": "/dev/video2", "topic": "/cameras/arm/image_raw"},
    }}), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("strix_vla_launch", LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()

    context = LaunchContext()
    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            context.launch_configurations[entity.name] = entity.default_value[0].perform(context)
    context.launch_configurations.update({"config": str(config), **arguments})
    return [
        entity.visit(context) or []
        for entity in description.entities
        if isinstance(entity, OpaqueFunction)
    ]


def _kinds(entities) -> list[str]:
    return sorted(type(e).__name__ for group in entities for e in group)


def test_the_camera_launch_holds_only_the_webcam_nodes(tmp_path):
    from launch_ros.actions import Node

    entities = _entities(tmp_path, arm="false", cameras="true", digital_twin="false", dry_run="false")
    nodes = [e for group in entities for e in group]
    assert _kinds(entities) == ["Node", "Node"], _kinds(entities)
    assert all(isinstance(n, Node) and n.node_package == "v4l2_camera" for n in nodes)


def test_the_arm_launch_holds_only_the_arm_bringup(tmp_path):
    entities = _entities(tmp_path, arm="true", cameras="false", digital_twin="false", dry_run="true")
    assert _kinds(entities) == ["IncludeLaunchDescription"], _kinds(entities)


def _node_name(node) -> str:
    """The name passed to ``Node`` before the action is executed."""
    return node._Node__node_name


def _prefix(node) -> str:
    return "".join(getattr(part, "text", "") for part in node._ExecuteLocal__process_description.prefix)


def test_the_twin_camera_launch_starts_only_the_mount_webcam(tmp_path):
    """``arm:=false cameras:=true digital_twin:=true`` is the camera half of ``run_pipeline.sh``."""
    from launch_ros.actions import Node

    entities = _entities(tmp_path, arm="false", cameras="true", digital_twin="true", dry_run="false")
    nodes = [e for group in entities for e in group]
    assert _kinds(entities) == ["Node"], _kinds(entities)
    assert len(nodes) == 1 and isinstance(nodes[0], Node)
    assert nodes[0].node_package == "v4l2_camera"
    assert _node_name(nodes[0]) == "camera_mount"
    assert "so101_sim.device_stop" in _prefix(nodes[0])


def test_the_twin_arm_launch_with_cameras_starts_the_mount_webcam_not_the_arm(tmp_path):
    from launch_ros.actions import Node

    entities = _entities(tmp_path, arm="true", cameras="true", digital_twin="true", dry_run="false")
    nodes = [e for group in entities for e in group if isinstance(e, Node)]
    assert _kinds(entities) == ["IncludeLaunchDescription", "Node"], _kinds(entities)
    assert len(nodes) == 1
    assert nodes[0].node_package == "v4l2_camera"
    assert _node_name(nodes[0]) == "camera_mount"
