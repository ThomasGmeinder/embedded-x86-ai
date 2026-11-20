"""Measured placement authority for SAPIEN stacking tasks."""

from __future__ import annotations

import numpy as np

from so101_smolvla.reachable_workspace import draw_reachable_layout

# Exact edge lengths of the real KIT cube_2 and cube_4 assets.
TRAINING_CUBE_SIZES = {"red": 0.020, "blue": 0.040}
# A cube's clearance is the distance from its centre to a corner of its
# footprint, so it holds however the cube is turned.
TRAINING_CUBE_CLEARANCES = np.sqrt(2.0) * np.array(
    [TRAINING_CUBE_SIZES["red"] / 2, TRAINING_CUBE_SIZES["blue"] / 2]
)
# The three equal 40 mm cubes of the colour task.
COLOR_CUBE_CLEARANCES = np.full(3, np.sqrt(2.0) * 0.020)
# Smallest distance between the centres of two cubes. With 40 mm the random
# stacking tests of seeds 0-19 gave 10 stacks; a wider distance that leaves
# room for the open fingers (73.3 mm, vla_testing/diagnostics/
# finger_room_rule.py) gave 6 (DEVELOPMENT_LEDGER.md, 2026-11-20).
TRAINING_CUBE_DISTANCE = 0.040
COLOR_CUBE_DISTANCE = 0.060


def draw_stack_layout(task_mode: str, seed: int) -> dict[str, list[float]]:
    """Draw one deterministic layout inside the proven pickup workspace."""
    rng = np.random.default_rng(seed)
    if task_mode == "stack2-training":
        points = draw_reachable_layout(
            rng,
            TRAINING_CUBE_CLEARANCES,
            minimum_distance=TRAINING_CUBE_DISTANCE,
        )
        return {"red": points[0].tolist(), "blue": points[1].tolist()}
    if task_mode == "stack3-color":
        points = draw_reachable_layout(
            rng,
            COLOR_CUBE_CLEARANCES,
            minimum_distance=COLOR_CUBE_DISTANCE,
        )
        return {
            color: points[index].tolist()
            for index, color in enumerate(("red", "green", "blue"))
        }
    raise ValueError(f"unsupported stacking task {task_mode!r}")
