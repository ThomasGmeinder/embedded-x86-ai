"""Where the gripper closes and where it lets go in the episodes of one random stacking run.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    random_run_grasp_and_release.py <run folder of stack2-recording-rig> <result .json in analysis/>

Nothing is simulated. The joint readings stored with every step of a finished
episode are posed on the robot of the stack2-recording-rig scene to find the
gripper tip (the point halfway between the two fingertips); the cube places
are those stored with every step.

Per episode:
- the grasp: the step at which the gripper has closed for the first time
  after having been open, where the tip is then against the red cube's place
  at that step (along the line from the robot base through the cube, across
  it, and above the table), and how far the red cube had been pushed by then;
- the lift: how high the red cube rose and at which step it was 10 mm up;
- the release: the first step after the lift at which the gripper reads
  open again, where the red cube is then against the large cube (along and
  across the line from the robot base through the large cube, and its
  underside above the large cube's top), and where the red cube lies at the
  end of the episode.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from release_aim_check import recording_rig_scene, tip_at

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
# Gripper readings in per cent of its range.
OPEN_PCT = 25.0
CLOSED_PCT = 16.0
LIFTED_MM = 10.0


def along_and_across(offset: np.ndarray, place: np.ndarray) -> tuple[float, float]:
    """Split a horizontal offset into the part away from the robot base and the part to its left."""
    outward = place[:2] / np.linalg.norm(place[:2])
    leftward = np.array([-outward[1], outward[0]])
    return float(offset[:2] @ outward), float(offset[:2] @ leftward)


def first(condition: np.ndarray, start: int = 0) -> int | None:
    found = np.flatnonzero(condition[start:])
    return int(found[0]) + start if len(found) else None


def episode_facts(inner, folder: Path) -> dict:
    steps = np.load(folder / "trajectory.npz")
    reading = steps["achieved"]
    red, large = 1000.0 * steps["cube_position_red"], 1000.0 * steps["cube_position_blue"]
    table = 1000.0 * float(inner.TABLE_SURFACE_Z)
    sizes = {colour: 1000.0 * size for colour, size in inner.cube_sizes.items()}
    gripper = reading[:, 5]
    facts: dict = {"seed": int(folder.name.split("-")[1])}

    opened = first(gripper > OPEN_PCT)
    closed = first(gripper < CLOSED_PCT, opened) if opened is not None else None
    if closed is not None:
        tip = 1000.0 * tip_at(inner, reading[closed])
        outward, left = along_and_across(tip - red[closed], red[closed])
        facts["grasp"] = {
            "step": closed,
            "tip_past_red_cube_mm": round(outward, 1),
            "tip_left_of_red_cube_mm": round(left, 1),
            "tip_above_table_mm": round(float(tip[2] - table), 1),
            "red_cube_pushed_before_mm": round(float(np.linalg.norm((red[closed] - red[0])[:2])), 1),
            "gripper_reading_20_steps_later_pct": round(float(gripper[min(closed + 20, len(gripper) - 1)]), 1),
        }
    rise = red[:, 2] - red[0, 2]
    facts["red_cube_highest_mm"] = round(float(rise.max()), 1)
    lifted = first(rise > LIFTED_MM)
    if lifted is not None:
        facts["lifted_at_step"] = lifted
        carried = gripper[lifted:]
        let_go = first(carried > np.minimum.accumulate(carried) + 10.0)
        if let_go is not None:
            step = lifted + let_go
            outward, left = along_and_across(red[step] - large[step], large[step])
            facts["release"] = {
                "step": step,
                "red_cube_past_large_cube_mm": round(outward, 1),
                "red_cube_left_of_large_cube_mm": round(left, 1),
                "red_cube_from_large_cube_centre_mm": round(float(np.hypot(outward, left)), 1),
                "red_cube_underside_above_large_cube_top_mm": round(float(
                    red[step, 2] - sizes["red"] / 2 - large[step, 2] - sizes["blue"] / 2), 1),
                "gripper_reading_while_carrying_pct": round(float(np.median(gripper[lifted:step])), 1),
            }
    facts["end"] = {
        "red_cube_from_large_cube_centre_mm": round(float(np.linalg.norm((red[-1] - large[-1])[:2])), 1),
        "red_cube_underside_above_table_mm": round(float(red[-1, 2] - sizes["red"] / 2 - table), 1),
        "large_cube_moved_mm": round(float(np.linalg.norm((large[-1] - large[0])[:2])), 1),
    }
    return facts


def main(run: str, output: str) -> None:
    folders = sorted(Path(run).glob("seed-*"))
    if not folders:
        raise SystemExit(f"no episode folders in {run}")
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        episodes = [episode_facts(inner, folder) for folder in folders]
    finally:
        env.close()
    for facts in episodes:
        print(json.dumps(facts), flush=True)
    (ANALYSIS / output).write_text(
        json.dumps({"run": run, "episodes": episodes}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
