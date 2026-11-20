"""Measured frame-zero robot states for the real stacking task."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CONFIG_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "sapien_stack_initial_states.json"
)


def _load_states() -> np.ndarray:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    states = np.asarray(document["states"], dtype=np.float64)
    if states.shape != (200, 6) or not np.isfinite(states).all():
        raise RuntimeError(
            f"invalid measured stacking states in {CONFIG_PATH}: {states.shape}"
        )
    return states


TRAINING_INITIAL_STATES = _load_states()


def draw_stack_initial_state(index: int) -> np.ndarray:
    """Return one measured real-units state by deterministic row index."""
    if not 0 <= index < len(TRAINING_INITIAL_STATES):
        raise ValueError(f"invalid stacking state index {index}")
    return TRAINING_INITIAL_STATES[index].copy()
