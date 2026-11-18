"""Backend-neutral environment boundary for checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

BACKEND_DIRECTORY_NAMES = {"mujoco": "MuJoCo", "sapien": "sapien"}


def random_output_directory(root: Path, backend: str, stamp: str) -> Path:
    """Return the backend-labelled directory for one random regression."""
    try:
        label = BACKEND_DIRECTORY_NAMES[backend]
    except KeyError as exc:
        raise ValueError(f"unknown policy backend {backend!r}") from exc
    return root / f"random-{label}-{stamp}"


@dataclass(frozen=True)
class PolicyObservation:
    state: np.ndarray
    top: np.ndarray
    wrist: np.ndarray


@dataclass(frozen=True)
class PolicyStep:
    observation: PolicyObservation
    commanded: np.ndarray
    achieved: np.ndarray
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    facts: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class PolicyEnvironment(Protocol):
    backend: str
    provenance: dict[str, object]

    def reset(self, environment_seed: int) -> PolicyObservation: ...

    def step(self, action: np.ndarray) -> PolicyStep: ...

    def close(self) -> None: ...


class MujocoPolicyEnvironment:
    """Policy-facing adapter around the existing MuJoCo ``ArmSim``."""

    backend = "mujoco"

    def __init__(self, initial_pose: np.ndarray) -> None:
        from so101_smolvla.sim import ArmSim

        self._sim = ArmSim()
        self._initial_pose = np.asarray(initial_pose, dtype=np.float64).reshape(6)
        self.provenance = {"backend": self.backend}

    @property
    def sim(self):
        return self._sim

    def _observation(self) -> PolicyObservation:
        return PolicyObservation(
            state=self._sim.read_state(),
            top=self._sim.render("top"),
            wrist=self._sim.render("wrist"),
        )

    def reset(self, environment_seed: int) -> PolicyObservation:
        del environment_seed
        self._sim.reset()
        self._sim.hold_pose(self._initial_pose)
        return self._observation()

    def step(self, action: np.ndarray) -> PolicyStep:
        from so101_smolvla.sim import TICK_STEPS

        self._sim.command(action)
        self._sim.step(TICK_STEPS)
        observation = self._observation()
        return PolicyStep(
            observation=observation,
            commanded=self._sim.commanded(),
            achieved=observation.state,
            success=self._sim.in_box(),
            facts={
                "cube_position": self._sim.cube_pos().tolist(),
                "jaw_contact": self._sim.jaw_contact(),
                "item_lifted": self._sim.above_table(),
                "in_box": self._sim.in_box(),
            },
        )

    def close(self) -> None:
        self._sim.close()


def create_policy_environment(
    backend: str,
    *,
    initial_pose: np.ndarray | None = None,
    horizon: int = 300,
):
    """Construct the sole selected policy environment backend."""
    if backend == "mujoco":
        if initial_pose is None:
            raise ValueError("MuJoCo policy environment requires initial_pose")
        return MujocoPolicyEnvironment(initial_pose)
    if backend == "sapien":
        from so101_smolvla.sapien_env_client import SapienPolicyEnvironment

        return SapienPolicyEnvironment(horizon=horizon)
    raise ValueError(f"unknown policy backend {backend!r}; expected 'mujoco' or 'sapien'")
