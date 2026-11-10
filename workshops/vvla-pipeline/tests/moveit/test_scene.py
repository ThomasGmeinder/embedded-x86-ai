# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A detection box becomes a point on the table and a planning-scene object.

The wrist camera is a MuJoCo camera. The bring-up publishes its optical frame
on the gripper, the camera info names that frame, and ``table_point`` meets
the table where the MuJoCo camera's ray through the ball does.
"""

from __future__ import annotations

import numpy as np
import pytest

import conftest
from conftest import OPTICAL, next_message

pytestmark = pytest.mark.sim
launch = pytest.mark.launch(fixture=conftest.twin_bringup)
BALL = "sports_ball"


def _optical(rotation):
    """MuJoCo camera axes (x right, y up, z back) as a ROS optical frame."""
    return np.column_stack([rotation[:, 0], -rotation[:, 1], -rotation[:, 2]])


@launch
def test_the_ball_ray_lands_on_the_table_and_in_the_planning_scene(arm, ros_node, pipeline_cfg):
    import mujoco
    from image_geometry import PinholeCameraModel
    from moveit_msgs.msg import CollisionObject
    from sensor_msgs.msg import CameraInfo

    from so101_sim.config import DEFAULT_SCENE
    from vla_pipeline.behaviors.manipulation import _optical_in_base
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE, collision_object, table_point

    # go_to_named plans with MoveItPy.plan, which segfaults in this process
    # after a pick task. The scan is a named joint pose, so interpolate it.
    assert arm.go_to_mtc(arm.named_state("scan_1")) == "arrived"
    info = next_message(ros_node, CameraInfo, "/cameras/arm/camera_info")
    assert info.header.frame_id == OPTICAL

    rotation, origin = _optical_in_base(ros_node)
    # A pick looks this up once per scan view on the arm's node.
    assert not {"/tf", "/tf_static"} & {sub.topic_name for sub in ros_node.subscriptions}

    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(model)
    data.qpos[:] = model.key("home").qpos
    for name, value in arm.get_joints().items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint >= 0:
            data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BALL)
    camera_rotation = _optical(data.cam_xmat[camera].reshape(3, 3))
    camera_origin = data.cam_xpos[camera].copy()
    ball = data.xpos[body].copy()
    assert np.allclose(origin, camera_origin, atol=0.005)
    assert np.allclose(rotation, camera_rotation, atol=0.02)

    ahead = camera_rotation.T @ (ball - camera_origin)
    assert ahead[2] > 0.0
    pixel = PinholeCameraModel()
    pixel.from_camera_info(info)
    u, v = pixel.project_3d_to_pixel(ahead)
    table_z = float(pipeline_cfg["pick_place"]["table_z"])
    hit = table_point((u, v), info, (rotation, origin), table_z)
    direction = ball - camera_origin
    expected = camera_origin + direction * ((table_z - camera_origin[2]) / direction[2])
    assert hit == pytest.approx(expected, abs=0.005)

    size = CLASS_SIZE["sports ball"]
    centre = np.array([hit[0], hit[1], table_z + 0.5 * size[2]])
    monitor = arm.robot.get_planning_scene_monitor()
    try:
        arm.remember(BALL, centre, size)
        with monitor.read_only() as scene:
            assert scene.knows_frame_transform(BALL)
            placed = scene.get_frame_transform(BALL)[:3, 3]
        assert placed == pytest.approx(centre, abs=1e-4)
    finally:
        gone = collision_object(BALL, centre, size)
        gone.operation = CollisionObject.REMOVE
        with monitor.read_write() as scene:
            scene.apply_collision_object(gone)


def test_a_box_at_the_grasp_frame_is_clear_of_the_open_hand(tmp_path):
    """The place's forbid stage checks the hand against the released object with the
    jaws open. The fixed jaw does not move, so every class box at :func:`grasp_frame`
    has to be clear of it, and of the rest of the robot."""
    from moveit.core.planning_scene import PlanningScene
    from moveit.core.robot_model import RobotModel
    from moveit.core.robot_state import RobotState

    from vla_pipeline.robot.moveit_arm import PLANNING_GROUP, TIP_LINK, grasp_frame, moveit_configs
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE, collision_object

    configs = moveit_configs()
    urdf, srdf = tmp_path / "so101.urdf", tmp_path / "so101.srdf"
    urdf.write_text(configs.robot_description["robot_description"])
    srdf.write_text(configs.robot_description_semantic["robot_description_semantic"])
    model = RobotModel(urdf_xml_path=str(urdf), srdf_xml_path=str(srdf))
    state = RobotState(model)
    state.set_to_default_values(PLANNING_GROUP, "home")
    state.set_to_default_values("gripper", "open")
    state.update()
    colliding = []
    for name, size in CLASS_SIZE.items():
        scene = PlanningScene(model)
        scene.current_state = state
        frame = grasp_frame(size)
        held = collision_object(name, (frame.position.x, frame.position.y, frame.position.z), size, frame=TIP_LINK)
        held.primitive_poses[0].orientation = frame.orientation
        scene.apply_collision_object(held)
        if scene.is_state_colliding(state, PLANNING_GROUP, True):
            colliding.append(name)
    assert colliding == []


def test_object_axis_measures_a_tilted_bar():
    """The long axis comes from the pixels inside the box, at the bar's angle."""
    import math

    import cv2

    from vla_pipeline.vision.yolo_detect_npu import object_axis

    image = np.full((480, 640, 3), 255, np.uint8)
    corners = cv2.boxPoints(((320, 240), (120, 20), 30)).astype(np.int32)
    cv2.fillConvexPoly(image, corners, (0, 0, 0))
    box = (int(corners[:, 0].min()), int(corners[:, 1].min()),
           int(corners[:, 0].max()), int(corners[:, 1].max()))
    ends = object_axis(image, box, 1.6)
    assert ends is not None
    (x0, y0), (x1, y1) = ends
    measured = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180
    assert abs((measured - 30 + 90) % 180 - 90) <= 2

    disc = np.full((480, 640, 3), 255, np.uint8)
    cv2.circle(disc, (320, 240), 20, (0, 0, 0), -1)
    assert object_axis(disc, (300, 220, 340, 260), 1.6) is None
