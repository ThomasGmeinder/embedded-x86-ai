"""Joint-contract conversion. Degrees for the arm, 0-100 for the gripper."""

from __future__ import annotations

import numpy as np

from so101_smolvla.sim import ARM, ArmSim


def state_from_sim(sim: ArmSim) -> np.ndarray:
    return sim.read_state()


def command_to_sim(sim: ArmSim, action: np.ndarray) -> None:
    sim.command(action)


def gripper_example_radians(sim: ArmSim, percent: float = 15.0) -> float:
    return sim.gripper_to_rad(percent)
