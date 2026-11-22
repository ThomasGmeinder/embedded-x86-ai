"""Single reachable-workspace authority for randomized object placement."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# The proven 10/10 SAPIEN pickup workspace from so101_sim.envs.ReachableSpawnMixin.
REACHABLE_CENTER = np.array([0.27, 0.045], dtype=np.float64)
REACHABLE_HALF_EXTENT = np.array([0.10, 0.10], dtype=np.float64)
REACHABLE_LOW = REACHABLE_CENTER - REACHABLE_HALF_EXTENT
REACHABLE_HIGH = REACHABLE_CENTER + REACHABLE_HALF_EXTENT


def draw_reachable_layout(
    rng: np.random.Generator,
    clearances: np.ndarray,
    minimum_distance: float,
    candidate_count: int = 100,
    also_valid: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Draw the first valid layout from one finite seeded candidate batch.

    A layout is valid when every object's footprint is reachable and the
    centres of every two objects are at least the minimum distance apart.
    ``also_valid`` adds a condition of the scene: it gets the candidate
    layouts, shaped (candidates, objects, 2), and says for each whether it
    may be used. The candidates drawn do not depend on it.
    """
    margins = np.asarray(clearances, dtype=np.float64)
    if margins.ndim != 1 or len(margins) < 1 or np.any(margins < 0.0):
        raise ValueError(f"invalid clearances {margins}")
    if minimum_distance < 0.0 or candidate_count < 1:
        raise ValueError(
            f"invalid placement limits distance={minimum_distance}, "
            f"candidates={candidate_count}"
        )

    low = REACHABLE_LOW + margins[:, None]
    high = REACHABLE_HIGH - margins[:, None]
    if np.any(low > high):
        raise ValueError("object footprint is wider than the reachable workspace")
    candidates = rng.uniform(
        low=low,
        high=high,
        size=(candidate_count, len(margins), 2),
    )
    separation = candidates[:, :, None, :] - candidates[:, None, :, :]
    distances = np.linalg.norm(separation, axis=-1)
    distinct = ~np.eye(len(margins), dtype=bool)
    valid = np.all(distances[:, distinct] >= minimum_distance, axis=1)
    if also_valid is not None:
        allowed = np.asarray(also_valid(candidates))
        if allowed.shape != valid.shape or allowed.dtype != bool:
            raise ValueError(
                f"the scene's condition returned {allowed.dtype} of shape "
                f"{allowed.shape}, expected bool of shape {valid.shape}"
            )
        valid &= allowed
    if not np.any(valid):
        raise RuntimeError("reachable workspace produced no non-overlapping layout")
    return candidates[np.flatnonzero(valid)[0]]


def is_reachable_xy(
    xy: np.ndarray | tuple[float, float],
    clearance: float = 0.0,
) -> bool:
    """Return whether an object center and its radial clearance are reachable."""
    point = np.asarray(xy, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        return False
    margin = float(clearance)
    if margin < 0.0:
        raise ValueError(f"clearance must be non-negative, got {clearance}")
    return bool(
        np.all(point >= REACHABLE_LOW + margin)
        and np.all(point <= REACHABLE_HIGH - margin)
    )


def translate_layout_into_reach(
    points: np.ndarray,
    clearances: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Rigidly translate a sampled layout so every object footprint is reachable."""
    layout = np.asarray(points, dtype=np.float64)
    margins = np.asarray(clearances, dtype=np.float64)
    if layout.ndim != 2 or layout.shape[1] != 2:
        raise ValueError(f"expected an N×2 layout, got {layout.shape}")
    if margins.shape != (len(layout),) or np.any(margins < 0.0):
        raise ValueError(f"invalid clearances {margins}")

    offsets = layout - layout.mean(axis=0)
    center_low = np.max(REACHABLE_LOW + margins[:, None] - offsets, axis=0)
    center_high = np.min(REACHABLE_HIGH - margins[:, None] - offsets, axis=0)
    if np.any(center_low > center_high):
        raise ValueError("layout is wider than the reachable workspace")

    translated = offsets + rng.uniform(center_low, center_high)
    if not all(
        is_reachable_xy(point, margin)
        for point, margin in zip(translated, margins, strict=True)
    ):
        raise RuntimeError("translated layout escaped the reachable workspace")
    return translated
