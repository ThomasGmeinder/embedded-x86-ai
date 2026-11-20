"""Does a stacking test go differently when only the large cube is moved away?

Run with .venv, PYTHONPATH = <workshop>/src : <workshop>/vla_testing/diagnostics,
from the workshop folder:
    same_tests_with_more_room.py <earlier run folder> <seeds, comma separated> <seed to repeat as it was>

The placement rule tried on 2026-11-20 kept the two cubes' centres apart by
finger_room_rule.room_for_the_fingers. A seed whose cubes stood closer in
an earlier run then got another layout altogether: both cubes lay elsewhere.
That does not show what the room itself does. Here each named seed is run
once more with the red cube where the earlier run had it and the large cube
pushed away from it, along the line from the red cube through the large cube,
to the smallest distance the rule allows. Where that place is out of reach the
line is turned about the red cube by the smallest angle that brings it back.

Everything else is the random stacking test of that seed: same scene, same
sentence, same start pose and same turn of each cube (both are drawn from the
environment seed by the scene, see chosen_layout_worker.py), same policy seed,
same episode runner (sapien_rollout.run_stack_episode), same number of steps,
same stack check.

The last argument checks the method: that seed is run with the earlier layout
unchanged, and its stored arm readings, commands and cube places must equal
the earlier run's.

Writes vla_testing/random-stacking-more-room-sapien-<time>/ with the same
files as a random stacking run; the repeated test is in seed-<n>-as-before/.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from chosen_layout_tests import ChosenTest, mark_complete, run_chosen_layouts, went
from finger_room_rule import room_for_the_fingers
from so101_smolvla.reachable_workspace import is_reachable_xy
from so101_smolvla.sapien_rollout import VLA_TESTING
from so101_smolvla.stack_layouts import TRAINING_CUBE_CLEARANCES

# Added to the smallest allowed distance so that rounding cannot put the cubes below it.
OVER_THE_RULE = 0.0001
TURN_STEP_DEG = 0.5
STORED = ("target", "commanded", "achieved", "cube_position_red", "cube_position_blue")


def with_room(layout: dict[str, list[float]]) -> tuple[dict[str, list[float]], float]:
    """The layout with the large cube pushed away from the red one, and the turn of the line in degrees."""
    red, large = np.asarray(layout["red"]), np.asarray(layout["blue"])
    large_clearance = TRAINING_CUBE_CLEARANCES[1]
    needed = float(room_for_the_fingers(TRAINING_CUBE_CLEARANCES)[0, 1])
    line = large - red
    if np.linalg.norm(line) >= needed:
        raise ValueError(f"the cubes of {layout} already leave room for the fingers")
    direction = np.arctan2(line[1], line[0])
    turns = np.radians(np.arange(-180.0, 180.0, TURN_STEP_DEG))
    for turn in sorted(turns, key=abs):
        place = red + (needed + OVER_THE_RULE) * np.array(
            [np.cos(direction + turn), np.sin(direction + turn)])
        if is_reachable_xy(place, large_clearance):
            return {"red": red.tolist(), "blue": place.tolist()}, float(np.degrees(turn))
    raise RuntimeError(f"no reachable place for the large cube around the red cube of {layout}")


def equal_steps(folder: Path, earlier: Path) -> tuple[int, int]:
    """Steps both tests lasted, and how many of them have every stored number equal."""
    now, before = np.load(folder / "trajectory.npz"), np.load(earlier / "trajectory.npz")
    common = min(len(now["target"]), len(before["target"]))
    same = np.ones(common, dtype=bool)
    for name in STORED:
        same &= np.all(now[name][:common] == before[name][:common], axis=tuple(
            range(1, now[name].ndim)))
    return common, int(same.sum())


def apart_mm(layout: dict[str, list[float]]) -> float:
    return round(1000.0 * float(np.linalg.norm(np.subtract(layout["blue"], layout["red"]))), 1)


def chosen_test(row: dict, name: str, kind: str, layout: dict, turned: float) -> ChosenTest:
    """A test of the earlier run's seed and policy seed with the given cube places."""
    earlier_layout = row["initial_layout"]
    return ChosenTest(
        name=name,
        environment_seed=row["environment_seed"],
        policy_seed=row["policy_seed"],
        layout=layout,
        notes={
            "test": kind,
            "earlier_layout": earlier_layout,
            "large_cube_moved_mm": round(1000.0 * float(np.linalg.norm(
                np.subtract(layout["blue"], earlier_layout["blue"]))), 1),
            "cube_centres_apart_mm": [apart_mm(earlier_layout), apart_mm(layout)],
            "line_turned_deg": turned,
            "earlier": went(row),
        },
    )


def main(earlier_run: str, seeds: str, repeated_seed: str) -> None:
    earlier = Path(earlier_run)
    before = {row["environment_seed"]: row
              for row in json.loads((earlier / "summary.json").read_text())}
    tests = []
    for seed in map(int, seeds.split(",")):
        layout, turned = with_room(before[seed]["initial_layout"])
        tests.append(chosen_test(
            before[seed], f"seed-{seed:05d}", "large cube moved away", layout, turned))
    seed = int(repeated_seed)
    tests.append(chosen_test(
        before[seed], f"seed-{seed:05d}-as-before", "as in the earlier run",
        before[seed]["initial_layout"], 0.0))

    destination = VLA_TESTING / f"random-stacking-more-room-sapien-{datetime.now():%Y%m%d-%H%M%S}"
    rows = run_chosen_layouts(tests, destination)
    common, same = equal_steps(destination / tests[-1].name, earlier / f"seed-{seed:05d}")
    print(f"the repeated test equals the earlier run on {same} of {common} steps", flush=True)
    mark_complete(destination, rows, earlier_run=earlier.name,
                  repeated_test_equal_steps=[same, common])
    print(f"stacked {sum(bool(row['success']) for row in rows[:-1])} of {len(rows) - 1} "
          f"moved tests in {destination}", flush=True)
    if same != common:
        raise SystemExit("the repeated test does not equal the earlier run: the method is not valid")


if __name__ == "__main__":
    main(*sys.argv[1:])
