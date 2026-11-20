"""The placement rule that leaves room for the open fingers between two cubes.

It was the product's rule for part of 2026-11-20 and is kept here for the
diagnostics that measured it. The product places two cubes at least
stack_layouts.TRAINING_CUBE_DISTANCE (40 mm) apart again: with this rule the
random stacking tests of seeds 0-19 gave 6, 5 and 6 stacks in 20 (policy seed
bases 0, 100, 200), with 40 mm they gave 10, 7 and 7 (DEVELOPMENT_LEDGER.md).
"""

from __future__ import annotations

import numpy as np

# How far the open fingers reach from the centre of the cube they come down
# on, below 40 mm above the table (the top of the largest cube): 45 mm or less
# in 95 % of the 187 human grasps of the real stacking recordings, 37 mm in
# the median (hand_room_at_the_grasp.py).
FINGER_REACH = 0.045


def room_for_the_fingers(clearances: np.ndarray) -> np.ndarray:
    """Smallest distance between the centres of every two objects.

    Row i, column j. The fingers that come down on one object must not reach
    the other, whichever of the two is gripped and however it is turned: the
    fingers' reach plus the larger of the two clearances.
    """
    margins = np.asarray(clearances, dtype=np.float64)
    return FINGER_REACH + np.maximum(margins[:, None], margins[None, :])
