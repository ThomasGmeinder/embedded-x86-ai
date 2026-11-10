# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Which box side the jaws close across: the pad gap and angle against MuJoCo's own kinematics."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from vla_pipeline.robot.moveit_arm import (
    FIXED_PAD, MOVING_PAD, _jaw_model, grasp_box, jaw_contact, jaw_grips,
)
from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE

pytestmark = pytest.mark.sim

OPEN_RAD = 1.7  # the SRDF "open" state
# Tall enough that both pad faces overlap the side, so a width test stays a width test.
TALL_M = 1.0
# gripper_frame_joint origin z in so_arm101_macro.xacro. Ry(pi) flips it into the grasp frame.
TIP_ORIGIN_Z = -0.086


def _forward(q: float) -> tuple[float, float]:
    """The pad gap along the gripper's X and the angle between the pads, at jaw angle ``q``."""
    m = _jaw_model()
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("gripper_joint").id]] = q
    mujoco.mj_forward(m, d)
    fixed, moving = m.geom(FIXED_PAD), m.geom(MOVING_PAD)
    body = int(fixed.bodyid[0])
    rotation, origin = d.xmat[body].reshape(3, 3), d.xpos[body]
    pad = d.geom_xmat[moving.id].reshape(3, 3)
    face = d.geom_xpos[moving.id] - pad[:, 0] * moving.size[0]
    gap = (rotation.T @ (face - origin))[0] - (fixed.pos[0] + fixed.size[0])
    return float(gap), math.acos(float((rotation.T @ pad[:, 0])[0]))


def _face_z(data, model, geom: str, body: int, along_x: float) -> tuple[float, float]:
    """The pad face's z interval in the grasp frame, from MuJoCo's positions."""
    g = model.geom(geom)
    centre = data.geom_xpos[g.id]
    axes = data.geom_xmat[g.id].reshape(3, 3)
    rotation, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
    zs = []
    for y in (-g.size[1], g.size[1]):
        for z in (-g.size[2], g.size[2]):
            world = centre + axes @ np.array((along_x * g.size[0], y, z))
            body_z = (rotation.T @ (world - origin))[2]
            zs.append(-(float(body_z) - TIP_ORIGIN_Z))
    return min(zs), max(zs)


@pytest.mark.parametrize("width", [0.024, 0.04, 0.06, 0.09])
def test_the_closed_form_contact_matches_the_simulated_jaw(width):
    contact = jaw_contact(width)
    gap, simulated = _forward(contact.q)
    assert gap == pytest.approx(width, abs=1e-6)
    assert contact.angle == pytest.approx(simulated, abs=1e-6)
    m = _jaw_model()
    d = mujoco.MjData(m)
    d.qpos[m.jnt_qposadr[m.joint("gripper_joint").id]] = contact.q
    mujoco.mj_forward(m, d)
    body = int(m.geom(FIXED_PAD).bodyid[0])
    assert contact.fixed_z == pytest.approx(_face_z(d, m, FIXED_PAD, body, 1.0), abs=1e-6)
    assert contact.moving_z == pytest.approx(_face_z(d, m, MOVING_PAD, body, -1.0), abs=1e-6)


def test_a_side_wider_than_the_open_jaw_has_no_grip():
    gap, _angle = _forward(OPEN_RAD)
    assert jaw_grips(gap - 0.001, OPEN_RAD, TALL_M)
    assert not jaw_grips(gap + 0.001, OPEN_RAD, TALL_M)


@pytest.fixture
def pad_friction():
    """Set the pads' friction for one test; the model is cached, so it goes back after."""
    m = _jaw_model()
    pad = m.geom(MOVING_PAD).id
    saved = float(m.geom_friction[pad][0])

    def set_mu(mu: float) -> None:
        m.geom_friction[pad][0] = mu

    yield set_mu
    m.geom_friction[pad][0] = saved


def test_the_pads_must_stay_within_twice_the_friction_angle(pad_friction):
    pad_friction(0.3)  # limit 2·atan(0.3) = 33.4°
    assert math.degrees(jaw_contact(0.04).angle) < 33.4 < math.degrees(jaw_contact(0.06).angle)
    assert jaw_grips(0.04, OPEN_RAD, TALL_M)
    assert not jaw_grips(0.06, OPEN_RAD, TALL_M)


def test_the_narrower_side_that_fits_goes_on_x():
    assert grasp_box((0.024, 0.12, 0.02), 0.3, OPEN_RAD) == ((0.024, 0.12, 0.02), 0.3)
    size, yaw = grasp_box((0.12, 0.024, 0.02), 0.3, OPEN_RAD)
    assert size == (0.024, 0.12, 0.02)
    assert yaw == pytest.approx(0.3 + math.pi / 2)


def test_a_box_too_wide_both_ways_is_refused():
    q = jaw_contact(0.03).q
    assert grasp_box((0.04, 0.04, 0.085), 0.0, q) is None


def test_the_book_does_not_meet_both_pads():
    assert grasp_box(CLASS_SIZE["book"], 0.0, OPEN_RAD) is None


def test_the_ball_bottle_and_toothbrush_meet_both_pads():
    for name in ("sports ball", "bottle", "toothbrush"):
        assert grasp_box(CLASS_SIZE[name], 0.0, OPEN_RAD) is not None
