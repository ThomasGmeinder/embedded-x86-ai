# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The description the bringup loads: upstream ``so_arm101``, same joint axes as the twin."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

pytestmark = pytest.mark.sim
pytest.importorskip("xacro")

from so101_sim import MJCF_JOINTS  # noqa: E402

ARM = tuple(f"{name}_joint" for name in MJCF_JOINTS)


def _urdf() -> ET.Element:
    import xacro
    from ament_index_python.packages import get_package_share_path

    path = get_package_share_path("so_arm101_description") / "urdf" / "so_arm101.urdf.xacro"
    return ET.fromstring(xacro.process_file(str(path)).toxml())


def _origin(element) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    m = np.eye(4)
    m[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in element.get("rpy").split()]).as_matrix()
    m[:3, 3] = [float(v) for v in element.get("xyz").split()]
    return m


def _joint_frames(root, q: dict) -> dict:
    """Each joint's frame in ``base_link`` at the joint angles ``q`` (rad)."""
    from scipy.spatial.transform import Rotation

    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {"base_link": np.eye(4)}
    frames = {}

    def link(name):
        if name not in links:
            j = joints[name]
            frame = link(j.find("parent").get("link")) @ _origin(j.find("origin"))
            frames[j.get("name")] = frame
            turn = np.eye(4)
            turn[:3, :3] = Rotation.from_rotvec([0, 0, q.get(j.get("name"), 0.0)]).as_matrix()
            links[name] = frame @ turn
        return links[name]

    for child in joints:
        link(child)
    return frames


def test_the_bringup_description_is_not_the_topic_bridge():
    """The arm joints are the twin's, and the hardware plugin is not the topic bridge."""
    root = _urdf()
    names = {j.get("name") for j in root.findall("joint") if j.get("type") == "revolute"}
    assert set(ARM) <= names, names
    plugins = [p.text or "" for p in root.findall("ros2_control/hardware/plugin")]
    assert plugins and all("TopicBasedSystem" not in p for p in plugins), plugins


def test_the_upstream_controllers_drive_those_joints():
    import yaml
    from ament_index_python.packages import get_package_share_path

    path = get_package_share_path("so_arm101_description") / "control" / "ros2_controllers.yaml"
    controllers = yaml.safe_load(path.read_text())
    arm = [
        name.removeprefix("<robot_namespace>")
        for name in controllers["joint_trajectory_controller"]["ros__parameters"]["joints"]
    ]
    jaw = controllers["gripper_controller"]["ros__parameters"]["joint"].removeprefix("<robot_namespace>")
    assert tuple(arm) == ARM[:5]
    assert jaw == "gripper_joint"


@pytest.mark.parametrize("q", [
    {},
    {"shoulder_pan_joint": 0.3, "shoulder_lift_joint": -0.4, "elbow_flex_joint": 0.5,
     "wrist_flex_joint": -0.2, "wrist_roll_joint": 0.6, "gripper_joint": 0.8},
], ids=["zero", "bent"])
def test_every_joint_turns_about_the_twin_axis(q):
    import mujoco

    from so101_sim.assets import require
    from so101_sim.config import DEFAULT_SCENE

    frames = _joint_frames(_urdf(), q)
    require()
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    d = mujoco.MjData(m)
    d.qpos[:] = 0.0
    for name, angle in q.items():
        d.qpos[m.jnt_qposadr[m.joint(name).id]] = angle
    mujoco.mj_forward(m, d)
    base = m.body("base").id
    rot, pos = d.xmat[base].reshape(3, 3), d.xpos[base]
    for name in ARM:
        jid = m.joint(name).id
        anchor = rot.T @ (d.xanchor[jid] - pos)
        axis = rot.T @ d.xaxis[jid]
        frame = frames[name]
        assert np.linalg.norm(frame[:3, 3] - anchor) < 1e-4, f"{name} anchor off by {frame[:3, 3] - anchor} m"
        angle = np.degrees(np.arccos(np.clip(frame[:3, 2] @ axis, -1.0, 1.0)))
        assert angle < 0.2, f"{name} axis off by {angle:.2f} deg"


def test_the_toothbrush_the_camera_sees_lies_along_the_box_the_jaws_grip():
    """The wrist camera yaws the grasp by the visible handle; the jaws close on the collision box.

    The handle's footprint on the table, by area, has its long axis on the box's Y.
    """
    import cv2
    import mujoco

    from so101_sim.assets import require
    from so101_sim.config import DEFAULT_SCENE

    require()
    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    body = m.body("toothbrush").id
    geom = m.geom("toothbrush_visual").id
    mesh = m.geom_dataid[geom]
    vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]]
    faces = m.mesh_face[m.mesh_faceadr[mesh]:m.mesh_faceadr[mesh] + m.mesh_facenum[mesh]]
    rotation = d.xmat[body].reshape(3, 3).T @ d.geom_xmat[geom].reshape(3, 3)
    offset = d.xmat[body].reshape(3, 3).T @ (d.geom_xpos[geom] - d.xpos[body])
    xy = (vertices @ rotation.T + offset)[:, :2]
    pixel_m = 1e-4
    low = xy.min(0)
    image = np.zeros(tuple(np.ceil((xy.max(0) - low) / pixel_m).astype(int)[::-1] + 1), np.uint8)
    corners = np.round((xy - low) / pixel_m).astype(np.int32)
    for face in faces:
        cv2.fillConvexPoly(image, corners[face], 255)
    rows, cols = np.nonzero(image)
    points = np.stack([cols, rows], 1).astype(float)
    _values, vectors = np.linalg.eigh(np.cov((points - points.mean(0)).T))
    major = vectors[:, -1]
    angle = np.degrees(np.arccos(abs(float(major[1]))))
    assert angle < 1.0, f"the toothbrush handle is {angle:.2f} deg off its collision box's long axis"
