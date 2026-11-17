import numpy as np

from so101_smolvla.adapter import gripper_example_radians
from so101_smolvla.sim import ArmSim


def test_gripper_15_is_not_degrees():
    sim = ArmSim()
    try:
        radians = gripper_example_radians(sim, 15.0)
        span = sim.open_rad - sim.closed_rad
        expected = sim.closed_rad + 0.15 * span
        assert abs(radians - expected) < 1e-9
        assert abs(radians - np.deg2rad(15.0)) > 0.05
    finally:
        sim.close()
