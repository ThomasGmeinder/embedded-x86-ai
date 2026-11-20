"""What happens to the red cube once it has landed on the large cube?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    after_the_landing.py <result .json in analysis/> <run folder of stack2-recording-rig> ...

Nothing is simulated and no model is run. Read from the stored steps of
finished tests (one step is 1/30 s): the two cubes' places, and the gripper
tip (the point halfway between the fingertips) from the stored joint
readings posed on the robot of the stack2-recording-rig scene.

Only tests in which the red cube was lifted. The landing is the first step
after the lift at which the red cube's underside is within ON_TOP_MM of the
large cube's top after having been higher. Per test:

- the fall: how high the underside was above the large cube's top over the
  steps before the landing, and how fast the cube moves sideways in the step
  into the landing (in all, and the part away from the large cube's centre);
- the landing: how far the red cube's centre is from the large cube's
  centre. The scene turns each cube by a random angle that is not stored
  with the steps, so the edge of the large cube's top face is only known to
  be between half its side (20 mm) and half its diagonal (28.3 mm) from its
  centre: a red cube whose centre is nearer than the first is over the top
  face whatever the turn, one farther than the second is not;
- afterwards: for how many steps it stays on top, its distance from the
  centre at each of the next steps, where the tip is meanwhile (above the
  cube's top or not, how far to the side), and whether the large cube moved.

Writes analysis/<result>.json and refuses to overwrite it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from random_run_grasp_and_release import ANALYSIS, episode_facts
from release_aim_check import recording_rig_scene, tip_at

STEPS_PER_SECOND = 30.0
# The red cube's underside counts as on the large cube's top this close to it.
ON_TOP_MM = 3.0
FALL_STEPS = 6
STEPS_AFTER = 10


def over_the_top_face(distance: float, side: float) -> str:
    """Whether a centre this far from the large cube's centre is over its top face."""
    if distance < side / 2:
        return "yes, whatever the turn"
    if distance > side / np.sqrt(2.0):
        return "no, whatever the turn"
    return "depends on the turn"


def landing_facts(inner, folder: Path) -> dict | None:
    episode = episode_facts(inner, folder)
    if "lifted_at_step" not in episode:
        return None
    steps = np.load(folder / "trajectory.npz")
    red, large = 1000.0 * steps["cube_position_red"], 1000.0 * steps["cube_position_blue"]
    sizes = {colour: 1000.0 * size for colour, size in inner.cube_sizes.items()}
    lifted = episode["lifted_at_step"]
    above_top = red[:, 2] - sizes["red"] / 2 - large[:, 2] - sizes["blue"] / 2
    offset = red - large
    distance = np.linalg.norm(offset[:, :2], axis=1)
    on_top = (np.abs(above_top) < ON_TOP_MM) & (distance < sizes["blue"])
    row: dict = {
        "run": folder.parent.name,
        "test": folder.name,
        "stacked": bool(json.loads((folder / "result.json").read_text())["success"]),
        "steps_stored": len(red),
        "lifted_at_step": lifted,
    }
    came_down = np.flatnonzero(on_top[lifted + 1:] & (above_top[lifted:-1] >= ON_TOP_MM))
    if not len(came_down):
        row["landed_on_the_large_cube"] = False
        return row
    landing = lifted + 1 + int(came_down[0])
    stays = np.flatnonzero(~on_top[landing:])
    steps_on_top = int(stays[0]) if len(stays) else len(red) - landing
    before = slice(max(landing - FALL_STEPS, lifted), landing)
    travel = (red[landing] - red[landing - 1])[:2] * STEPS_PER_SECOND
    outward = offset[landing, :2] / distance[landing]
    after = slice(landing, min(landing + STEPS_AFTER + 1, len(red)))
    tips = 1000.0 * np.array([tip_at(inner, reading) for reading in steps["achieved"][after]])
    tip_over_cube_top = tips[:, 2] - (red[after, 2] + sizes["red"] / 2)
    tip_to_the_side = np.linalg.norm((tips - red[after])[:, :2], axis=1)
    row.update({
        "landed_on_the_large_cube": True,
        "landing_step": landing,
        "underside_above_the_top_before_mm": [round(float(v), 1) for v in above_top[before]],
        "fell_from_mm": round(float(above_top[before].max()), 1),
        "sideways_speed_into_the_landing_mm_s": round(float(np.linalg.norm(travel)), 1),
        "of_that_away_from_the_centre_mm_s": round(float(travel @ outward), 1),
        "from_the_centre_at_landing_mm": round(float(distance[landing]), 1),
        "centre_over_the_top_face_at_landing": over_the_top_face(
            float(distance[landing]), sizes["blue"]),
        "steps_on_top": steps_on_top,
        "still_on_top_at_the_last_step": bool(on_top[-1]),
        "from_the_centre_after_mm": [round(float(v), 1) for v in distance[after]],
        "underside_above_the_top_after_mm": [round(float(v), 1) for v in above_top[after]],
        "tip_above_the_cube_top_after_mm": [round(float(v), 1) for v in tip_over_cube_top],
        "tip_to_the_side_of_the_cube_after_mm": [round(float(v), 1) for v in tip_to_the_side],
        "large_cube_moved_after_mm": round(
            float(np.linalg.norm((large[after][-1] - large[landing])[:2])), 1),
    })
    return row


def spread(rows: list[dict], key: str) -> dict | None:
    values = [row[key] for row in rows if key in row]
    if not values:
        return None
    low, median, high = np.percentile(values, [25, 50, 75])
    return {"tests": len(values), "quarter": round(float(low), 1),
            "median": round(float(median), 1), "three_quarters": round(float(high), 1)}


def main(output: str, *runs: str) -> None:
    result = ANALYSIS / output
    if result.exists():
        raise SystemExit(f"{result} exists; choose another name")
    if not runs:
        raise SystemExit("name at least one run folder")
    env = recording_rig_scene()
    rows = []
    try:
        inner = env._env.unwrapped
        for run in runs:
            folders = sorted(Path(run).glob("seed-*"))
            if not folders:
                raise SystemExit(f"no test folders in {run}")
            for folder in folders:
                row = landing_facts(inner, folder)
                if row is not None:
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    finally:
        env.close()
    landed = [row for row in rows if row["landed_on_the_large_cube"]]
    groups = {
        "lifted": rows,
        "lifted and stacked": [row for row in rows if row["stacked"]],
        "lifted, not stacked": [row for row in rows if not row["stacked"]],
        "stacked, landing found": [row for row in landed if row["stacked"]],
        "not stacked, landed on the large cube": [row for row in landed if not row["stacked"]],
    }
    keys = ("fell_from_mm", "sideways_speed_into_the_landing_mm_s",
            "of_that_away_from_the_centre_mm_s", "from_the_centre_at_landing_mm",
            "steps_on_top", "large_cube_moved_after_mm")
    summary = {name: {"tests": len(chosen), **{key: spread(chosen, key) for key in keys}}
               for name, chosen in groups.items()}
    for name, values in summary.items():
        print(name, json.dumps(values), flush=True)
    result.write_text(
        json.dumps({"runs": list(runs), "on_top_mm": ON_TOP_MM, "summary": summary, "tests": rows},
                   indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
