import time

import mujoco
import numpy as np

from so101_smolvla.sim import HOLD_STEPS, PHYSICS_HZ, TICK_S, TICK_STEPS, ArmSim


def test_tick_is_one_thirtieth_and_under_budget():
    sim = ArmSim()
    try:
        assert abs(sim.model.opt.timestep * TICK_STEPS - TICK_S) < 1e-9
        assert abs(TICK_S - 1.0 / 30.0) < 1e-9
        start = time.perf_counter()
        sim.step(TICK_STEPS)
        assert time.perf_counter() - start < 1.0 / 30.0
    finally:
        sim.close()


def test_images_and_reset():
    sim = ArmSim()
    try:
        sim.reset((0.22, 0.08))
        first = sim.cube_pos().copy()
        top = sim.render("top")
        wrist = sim.render("wrist")
        assert top.shape == (480, 640, 3)
        assert wrist.shape == (480, 640, 3)
        sim.step(50)
        sim.reset((0.22, 0.08))
        assert np.allclose(sim.cube_pos(), first)
    finally:
        sim.close()


def test_each_arm_actuator_moves():
    sim = ArmSim()
    try:
        sim.reset()
        start = sim.read_state().copy()
        target = start.copy()
        target[:5] = start[:5] + np.array([8.0, -6.0, 6.0, -5.0, 10.0])
        target[5] = 40.0
        sim.command(target)
        sim.step(PHYSICS_HZ)
        moved = sim.read_state()
        assert np.all(np.abs(moved[:5] - start[:5]) > 1.0)
        assert moved[5] < 80.0
    finally:
        sim.close()


def test_render_refuses_a_cube_center_inside_a_jaw():
    sim = ArmSim()
    try:
        sim.reset((0.22, 0.08))
        sim.render("wrist")
        jaw = next(iter(sim.jaw_geoms))
        sim.data.qpos[sim.cube_qadr : sim.cube_qadr + 3] = sim.data.geom_xpos[jaw]
        mujoco.mj_forward(sim.model, sim.data)
        assert sim.jaw_contains_cube_center() is not None
        try:
            sim.render("top")
            raised = False
        except RuntimeError:
            raised = True
        assert raised
    finally:
        sim.close()


def test_empty_close_has_no_contact_and_small_force():
    sim = ArmSim()
    try:
        sim.reset((0.22, 0.08))
        state = sim.read_state()
        state[5] = 100.0
        sim.command(state)
        sim.step(PHYSICS_HZ // 5)
        state[5] = 0.0
        sim.command(state)
        sim.step(PHYSICS_HZ)
        assert sim.jaw_contact() is False
        assert abs(sim.gripper_force()) < 0.2
    finally:
        sim.close()
