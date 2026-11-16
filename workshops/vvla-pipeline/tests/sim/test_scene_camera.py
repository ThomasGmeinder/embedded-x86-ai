# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The published scene camera sits on the free overview the console used to render."""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.sim


def test_the_scene_camera_matches_the_free_overview():
    import mujoco

    from so101_sim.config import DEFAULTS, DEFAULT_SCENE

    view = DEFAULTS["cameras"]["view"]
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = view["lookat"]
    camera.distance = view["distance"]
    camera.azimuth = view["azimuth"]
    camera.elevation = view["elevation"]
    scene = mujoco.MjvScene(model, maxgeom=10)
    mujoco.mjv_updateCamera(model, data, camera, scene)
    center = 0.5 * (np.array(scene.camera[0].pos) + np.array(scene.camera[1].pos))
    scene_id = model.camera("scene_cam").id
    assert data.cam_xpos[scene_id] == pytest.approx(center, abs=1e-4)
    look = data.xpos[model.body("scene_look").id]
    direction = look - data.cam_xpos[scene_id]
    direction /= np.linalg.norm(direction)
    assert direction == pytest.approx(np.array(scene.camera[0].forward), abs=1e-4)
