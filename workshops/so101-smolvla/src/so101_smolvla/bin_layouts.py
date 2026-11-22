"""Where the bin and the cubes of the three-cube bin test stand."""

from __future__ import annotations

import numpy as np

from so101_smolvla.reachable_workspace import (
    REACHABLE_HIGH,
    REACHABLE_LOW,
    draw_reachable_layout,
)
from so101_smolvla.stack_layouts import COLOR_CUBE_CLEARANCES, COLOR_CUBE_DISTANCE

# The cubes of the test, in the order a layout lists them.
BIN_CUBE_COLORS = ("red", "green", "blue")
# The bin of the test is the simulator's trained bin, this many times as long
# and as wide, and as deep as it was.
BIN_SCALE = (1.5, 1.5, 1.0)
# Bin positions one draw tries. With bin and cubes kept off the table under
# the resting gripper, 100 give a layout for 129 of 200 seeds, 300 for 193
# and 1,000 for all 200 (measured 22 Nov 2026; the slowest draw took 107 ms).
BIN_POSITIONS_TRIED = 1000


def _in_bin_frame(
    points: np.ndarray, bin_xy: np.ndarray, bin_yaw: float
) -> np.ndarray:
    offset = np.asarray(points, dtype=np.float64) - np.asarray(bin_xy, dtype=np.float64)
    cos, sin = np.cos(-bin_yaw), np.sin(-bin_yaw)
    return np.stack(
        [
            cos * offset[..., 0] - sin * offset[..., 1],
            sin * offset[..., 0] + cos * offset[..., 1],
        ],
        axis=-1,
    )


def bin_footprint_distance(
    points: np.ndarray,
    bin_xy: np.ndarray,
    bin_yaw: float,
    bin_half_sizes: np.ndarray,
) -> np.ndarray:
    """Distance of each table point from the bin's footprint; zero inside it."""
    beyond = np.abs(_in_bin_frame(points, bin_xy, bin_yaw)) - np.asarray(
        bin_half_sizes, dtype=np.float64
    )
    return np.linalg.norm(np.maximum(beyond, 0.0), axis=-1)


def bin_footprint_corners(
    bin_xy: np.ndarray, bin_yaw: float, bin_half_sizes: np.ndarray
) -> np.ndarray:
    """The four corners of the bin's footprint on the table."""
    half = np.asarray(bin_half_sizes, dtype=np.float64)
    local = half * np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0]])
    cos, sin = np.cos(bin_yaw), np.sin(bin_yaw)
    turned = np.stack(
        [
            cos * local[:, 0] - sin * local[:, 1],
            sin * local[:, 0] + cos * local[:, 1],
        ],
        axis=-1,
    )
    return turned + np.asarray(bin_xy, dtype=np.float64)


def draw_bin_layout(
    rng: np.random.Generator,
    bin_half_sizes: np.ndarray,
    outreach: float,
    keep_clear: float,
    candidate_count: int = BIN_POSITIONS_TRIED,
    keep_off: np.ndarray | None = None,
    bin_keep_off: float = 0.0,
    cube_keep_off: float = 0.0,
) -> dict[str, object]:
    """Draw the first valid bin position and cube layout of one seeded batch.

    The bin's centre lies in the reachable workspace and no corner of its
    footprint lies farther than ``outreach`` outside it. The cubes stand as
    in the three-cube stacking task, and the centre of each is at least
    ``keep_clear`` from the bin's footprint. ``keep_off`` lists table points
    nothing may stand on, such as those under the low parts of the resting
    arm: the bin's footprint stays farther than ``bin_keep_off`` from each
    and every cube's centre farther than ``cube_keep_off``.
    Fails when no bin position of the batch is free and leaves that room.
    The bin positions tried do not depend on ``keep_off``.
    """
    half = np.asarray(bin_half_sizes, dtype=np.float64)
    if half.shape != (2,) or not np.isfinite(half).all() or np.any(half <= 0.0):
        raise ValueError(f"invalid bin half sizes {half}")
    if (
        outreach < 0.0
        or keep_clear < 0.0
        or candidate_count < 1
        or bin_keep_off < 0.0
        or cube_keep_off < 0.0
    ):
        raise ValueError(
            f"invalid bin placement limits outreach={outreach}, "
            f"keep_clear={keep_clear}, candidates={candidate_count}, "
            f"bin_keep_off={bin_keep_off}, cube_keep_off={cube_keep_off}"
        )
    occupied = np.empty((0, 2)) if keep_off is None else np.asarray(keep_off, dtype=np.float64)
    if occupied.ndim != 2 or occupied.shape[1] != 2 or not np.isfinite(occupied).all():
        raise ValueError(
            f"the points to keep off are of shape {occupied.shape}, expected (points, 2)"
        )
    centres = rng.uniform(REACHABLE_LOW, REACHABLE_HIGH, size=(candidate_count, 2))
    yaws = rng.uniform(-np.pi, np.pi, size=candidate_count)
    for centre, yaw in zip(centres, yaws, strict=True):
        corners = bin_footprint_corners(centre, float(yaw), half)
        if np.any(corners < REACHABLE_LOW - outreach) or np.any(
            corners > REACHABLE_HIGH + outreach
        ):
            continue
        if len(occupied) and np.any(
            bin_footprint_distance(occupied, centre, float(yaw), half)
            <= bin_keep_off
        ):
            continue

        def free_and_clear_of_bin(candidates: np.ndarray) -> np.ndarray:
            distance = bin_footprint_distance(candidates, centre, float(yaw), half)
            valid = np.all(distance >= keep_clear, axis=1)
            if len(occupied):
                nearest = np.linalg.norm(
                    candidates[:, :, None, :] - occupied, axis=-1
                ).min(axis=2)
                valid &= np.all(nearest > cube_keep_off, axis=1)
            return valid

        try:
            cubes = draw_reachable_layout(
                rng,
                COLOR_CUBE_CLEARANCES,
                COLOR_CUBE_DISTANCE,
                also_valid=free_and_clear_of_bin,
            )
        except RuntimeError:
            # This bin position leaves no room; the next one of the batch is tried.
            continue
        return {
            "bin_xy": centre.tolist(),
            "bin_yaw": float(yaw),
            "cubes": {
                color: cubes[index].tolist()
                for index, color in enumerate(BIN_CUBE_COLORS)
            },
        }
    raise RuntimeError(
        f"no bin position of {candidate_count} is free and leaves room for the cubes"
    )
