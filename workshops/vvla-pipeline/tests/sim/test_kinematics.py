# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``so101_sim.kinematics``: locating objects from arm-camera boxes, grasp IK,
and the hand-eye solve behind ``scripts/calibrate_arm_camera.py``.

Localization runs on the twin: the arm looks from ``scan_pose`` (panned the
way the servo would), the box comes from the twin's segmentation, and the
result is compared with where the object really is.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

pytestmark = pytest.mark.sim

BALL_SPOTS = [(0.28, 0.06), (0.22, -0.04), (0.32, 0.0), (0.25, 0.10)]


@pytest.fixture(scope="module")
def kin():
    from so101_sim.kinematics import Kinematics

    return Kinematics()


def _aimed_scan_pose(kin, cfg, p) -> dict:
    """scan_pose with the pan that puts ``p`` where the servo would (servo_target_x)."""
    pp = cfg["behaviors"]["pick_place"]
    best = None
    for pan in np.arange(-60.0, 40.0, 0.5):
        pose = {**pp["scan_pose"], "shoulder_pan": float(pan)}
        u = kin.project(p, pose)[0, 0]
        err = abs(u / 640 * 2 - 1 - pp["servo_target_x"])
        if best is None or err < best[0]:
            best = (err, pose)
    return best[1]


def _look(twin, obj, pos, pose):
    """Arm at ``pose``, only ``obj`` on the table at ``pos``: ``(class, box, (w, h), true centre)``."""
    from so101_sim.camera import SimCamera
    from so101_sim.detect import SegDetector

    world = twin.world
    world.reset(pose)
    for o in world.objects:
        if o != obj:
            world.hide(o)
    world.place(obj, pos)
    cam = SimCamera(world, "arm", renderer=twin.renderer, segmentation=True)
    try:
        time.sleep(0.3)  # let the object settle on the table
        ok, frame = cam.read(fresh=True)
        assert ok
        dets = [d for d in SegDetector(cam).detect(frame) if d[3] == obj]
        assert dets, f"{obj} not in view"
        return dets[0][0], dets[0][2], frame.shape[1::-1], world.body_pos(obj)
    finally:
        cam.release()


@pytest.mark.parametrize(
    "obj,xy",
    [("sports_ball", s) for s in BALL_SPOTS]
    + [("bottle", (0.2, 0.2)), ("book", (0.22, -0.15)), ("toothbrush", (0.12, -0.24))],
)
def test_locate_on_table(twins, pipeline_cfg, kin, obj, xy):
    from so101_sim.kinematics import FLAT_CLASSES, ROUND_CLASSES

    twin = twins("pipeline", pipeline_cfg)
    pos = np.array([xy[0], xy[1], twin.world.model.body(obj).pos[2]])
    pose = _aimed_scan_pose(kin, pipeline_cfg, pos)
    cls, box, size, truth = _look(twin, obj, pos, pose)

    shape = "sphere" if cls in ROUND_CLASSES else "flat" if cls in FLAT_CLASSES else "bottle" if cls == "bottle" else "upright"
    loc = kin.locate_on_table(box, pose, image_size=size, shape=shape)
    assert loc is not None, f"no fix for box {box}"
    err = loc["xyz"] - truth
    xy_tol = 0.025 if shape == "flat" else 0.015 if shape == "bottle" else 0.01
    assert np.linalg.norm(err[:2]) < xy_tol, (
        f"{obj} at {np.round(truth, 3)} located at {np.round(loc['xyz'], 3)}"
    )
    if obj == "sports_ball":
        assert abs(err[2]) < 0.01
        assert abs(loc["width"] - 0.04) < 0.005


def test_box_cut_by_the_image_side_gives_no_fix(kin):
    assert kin.locate_on_table((0, 100, 80, 200), {}, image_size=(640, 480)) is None
    assert kin.locate_on_table((200, 400, 300, 480), {}, image_size=(640, 480)) is None


@pytest.mark.parametrize("r,bearing", [(0.16, 0), (0.20, 25), (0.24, -20), (0.28, 10), (0.30, -5)])
def test_ik_round_trip(kin, r, bearing):
    sx, sy = kin.world.fk({}).body("shoulder").xpos[:2]
    p = np.array([sx + r * math.cos(math.radians(bearing)), sy + r * math.sin(math.radians(bearing)), 0.03])
    plan = kin.solve_grasp(p, above_m=0.06, width=0.04)
    assert plan is not None, f"{np.round(p, 3)} should be in reach"
    pre, grasp = plan
    w = kin.world
    assert np.linalg.norm(w.grasp_point(grasp, 0.04) - p) < 0.003
    assert np.linalg.norm(w.grasp_point(pre, 0.04) - (p + [0, 0, 0.06])) < 0.003
    assert abs(w.jaw_axis(grasp)[2]) < 0.1, "closing axis not horizontal at the grasp"
    # at the edge of reach the wrist limit tilts the pre-grasp a little
    assert abs(w.jaw_axis(pre)[2]) <= 0.3
    for pose in (pre, grasp):
        assert kin.lowest_point(pose) > -0.005, "arm below the table"


def test_descent_waypoints_stay_on_the_vertical(kin):
    p = np.array([0.22, -0.04, 0.021])
    pre, grasp = kin.solve_grasp(p, above_m=0.06, width=0.04)
    path = kin.waypoints(pre, grasp, 0.04)
    assert len(path) >= 4 and path[-1] == grasp
    top = kin.world.grasp_point(pre, 0.04)
    for pose in path:
        q = kin.world.grasp_point(pose, 0.04)
        assert np.linalg.norm(q[:2] - top[:2]) < 0.003, "descent leaves the vertical line"


def test_ik_out_of_reach(kin):
    assert kin.solve_grasp([0.50, 0.0, 0.02]) is None
    assert kin.solve_grasp([0.25, 0.0, -0.05]) is None  # below the table


def test_mount_config_round_trip(kin):
    from so101_sim.kinematics import matrix_to_rpy

    R0, t0 = kin.default_mount
    R, t = kin.mount({"pos": t0.tolist(), "rpy": matrix_to_rpy(R0)})
    assert np.allclose(R, R0, atol=1e-9) and np.allclose(t, t0)
    assert kin.mount({"pos": None, "rpy": None}) is kin.default_mount


def test_default_camera_model_projects_like_the_twin(twins, pipeline_cfg, kin):
    twin = twins("pipeline", pipeline_cfg)
    pos = np.array([0.28, 0.06, 0.015])
    pose = _aimed_scan_pose(kin, pipeline_cfg, pos)
    _, box, _, truth = _look(twin, "sports_ball", pos, pose)
    u, v = kin.project(truth, pose)[0]
    assert abs(u - 0.5 * (box[0] + box[2])) < 5 and abs(v - 0.5 * (box[1] + box[3])) < 5


def test_hand_eye_recovers_a_known_mount(kin):
    from so101_sim.kinematics import rpy_to_matrix, solve_mount

    rng = np.random.default_rng(0)
    R0, t0 = kin.default_mount
    R_true = R0 @ rpy_to_matrix([3.0, -2.0, 4.0])
    t_true = t0 + [0.004, -0.003, 0.005]
    board_R, board_t = rpy_to_matrix([0.0, 0.0, 90.0]), np.array([0.25, -0.04, -0.005])
    grip, board = [], []
    for _ in range(10):
        joints = {"shoulder_pan": rng.uniform(-10, 10), "shoulder_lift": rng.uniform(-62, -40),
                  "elbow_flex": rng.uniform(50, 80), "wrist_flex": rng.uniform(30, 65),
                  "wrist_roll": rng.uniform(-25, 25)}
        Rg, tg = kin.gripper_pose(joints)
        Rc, tc = Rg @ R_true, tg + Rg @ t_true
        # board in camera, with ~0.3 mm / 0.1 deg of PnP noise
        Rb = rpy_to_matrix(rng.normal(0, 0.1, 3)) @ Rc.T @ board_R
        tb = Rc.T @ (board_t - tc) + rng.normal(0, 3e-4, 3)
        grip.append((Rg, tg))
        board.append((Rb, tb))
    R, t, spread = solve_mount(grip, board)
    angle = math.degrees(math.acos(np.clip((np.trace(R_true.T @ R) - 1) / 2, -1, 1)))
    assert np.linalg.norm(t - t_true) < 0.002
    assert angle < 0.5
    assert spread < 0.003
