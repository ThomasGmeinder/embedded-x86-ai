"""How high is the red cube carried on its way to the large cube: the tests against the humans.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    carry_path_against_the_humans.py <result .json in analysis/> <run folder of stack2-recording-rig> ...

Nothing is simulated. Joint readings (a test's stored "achieved", a
recording's stored state) are posed on the robot of the stack2-recording-rig
scene; the gripper tip is the point halfway between the two fingertips.

A carry runs from the lift to the letting go:
  test       from the step the red cube is 10 mm up to the step the gripper
             opens again (random_run_grasp_and_release.episode_facts), or to
             the test's last step if it does not open;
  recording  from 10 frames after the human closed the gripper to the release
             frame (analysis/cube-placement.npz).
The large cube's centre is the stored cube place of the same step in a test.
In a recording it is taken to be under the tip at the release frame, as the
replay of the recordings does; a recording's distance at the release is
therefore 0 by definition and says nothing about the human's aim.

Per carry: its length in steps, the highest tip, the tip's height above the
table in bands of its horizontal distance from the large cube's centre (the
lowest height in the band), the distance at which the tip starts to come
down for the last time by 5 mm, and height and distance at the letting go.
For a test also where the red cube itself is then.

Over all carries of a group (recordings, tests that stack, tests that lift
and do not stack): medians and quarter marks.

Writes analysis/<result>.json and refuses to overwrite it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from random_run_grasp_and_release import ANALYSIS, episode_facts
from release_aim_check import recorded, recording_rig_scene, tip_at

# Frames after the human's closing at which the cube counts as held.
HELD_AFTER_CLOSING = 10
# Horizontal distance of the tip from the large cube's centre, in millimetres.
BANDS = ((90, 120), (60, 90), (45, 60), (30, 45), (20, 30), (10, 20), (0, 10))
# The red cube's edge is over the large cube's edge when their centres are this close.
OVER_THE_LARGE_CUBE_MM = 30.0
COMES_DOWN_MM = 5.0


def band_name(band: tuple[int, int]) -> str:
    return f"tip {band[0]} to {band[1]} mm from the large cube's centre: lowest tip above the table, mm"


def carry_facts(tips: np.ndarray, large: np.ndarray, table_mm: float) -> dict:
    """The tip's way during one carry. "tips" and "large" are in millimetres, one row per step."""
    height = tips[:, 2] - table_mm
    distance = np.linalg.norm(tips[:, :2] - large[:, :2], axis=1)
    facts: dict = {
        "steps": len(tips),
        "highest_tip_above_table_mm": round(float(height.max()), 1),
        "tip_above_table_at_letting_go_mm": round(float(height[-1]), 1),
        "tip_from_large_cube_centre_at_letting_go_mm": round(float(distance[-1]), 1),
        "nearest_the_tip_came_to_the_large_cube_centre_mm": round(float(distance.min()), 1),
    }
    for band in BANDS:
        inside = (distance >= band[0]) & (distance < band[1])
        facts[band_name(band)] = round(float(height[inside].min()), 1) if inside.any() else None
    # The last descent: going back from the letting go, the step up to which
    # the tip was never more than COMES_DOWN_MM above where it is let go.
    above = np.flatnonzero(height > height[-1] + COMES_DOWN_MM)
    if len(above):
        start = int(above[-1])
        facts["last_descent"] = {
            "steps_before_letting_go": len(tips) - 1 - start,
            "tip_from_large_cube_centre_when_it_starts_mm": round(float(distance[start]), 1),
            "tip_above_table_when_it_starts_mm": round(float(height[start]), 1),
        }
    first_over = np.flatnonzero(distance < OVER_THE_LARGE_CUBE_MM)
    if len(first_over):
        facts["tip_above_table_when_first_within_30_mm_mm"] = round(float(height[first_over[0]]), 1)
    return facts


def test_carries(inner, runs: list[str], table_mm: float) -> list[dict]:
    rows = []
    for run in runs:
        folders = sorted(Path(run).glob("seed-*"))
        if not folders:
            raise SystemExit(f"no test folders in {run}")
        for folder in folders:
            episode = episode_facts(inner, folder)
            if "lifted_at_step" not in episode:
                continue
            steps = np.load(folder / "trajectory.npz")
            lifted = episode["lifted_at_step"]
            let_go = episode.get("release", {}).get("step", len(steps["achieved"]) - 1)
            chosen = slice(lifted, let_go + 1)
            tips = 1000.0 * np.array([tip_at(inner, reading) for reading in steps["achieved"][chosen]])
            red = 1000.0 * steps["cube_position_red"][chosen]
            large = 1000.0 * steps["cube_position_blue"][chosen]
            row = {
                "run": Path(run).name,
                "test": folder.name,
                "stacked": bool(json.loads((folder / "result.json").read_text())["success"]),
                "gripper_opened_again": "release" in episode,
                **carry_facts(tips, large, table_mm),
                "red_cube_from_large_cube_centre_at_letting_go_mm": round(
                    float(np.linalg.norm((red[-1] - large[-1])[:2])), 1),
                "red_cube_from_the_tip_at_letting_go_mm": round(
                    float(np.linalg.norm((red[-1] - tips[-1])[:2])), 1),
                "red_cube_centre_below_the_tip_at_letting_go_mm": round(float(tips[-1, 2] - red[-1, 2]), 1),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def recording_carries(inner, table_mm: float) -> list[dict]:
    events = np.load(ANALYSIS / "cube-placement.npz")
    episode_of, state = recorded()["episode"], recorded()["state"]
    rows = []
    for number, close, release in zip(
        events["training_episode"].tolist(),
        events["training_close_frame"].tolist(),
        events["training_release_frame"].tolist(),
    ):
        readings = state[episode_of == number][close + HELD_AFTER_CLOSING : release + 1]
        if len(readings) < 2:
            continue
        tips = 1000.0 * np.array([tip_at(inner, reading) for reading in readings])
        large = np.repeat(tips[-1:], len(tips), axis=0)
        row = {"recording": int(number), **carry_facts(tips, large, table_mm)}
        rows.append(row)
    return rows


def spread(rows: list[dict], *keys: str) -> dict | None:
    """Quarter mark, median, quarter mark and count of one number over the rows that have it."""
    values = []
    for row in rows:
        value = row
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        if value is not None:
            values.append(value)
    if not values:
        return None
    low, median, high = np.percentile(values, [25, 50, 75])
    return {"carries": len(values), "quarter": round(float(low), 1),
            "median": round(float(median), 1), "three_quarters": round(float(high), 1)}


def summary(rows: list[dict]) -> dict:
    numbers = [
        ("steps",),
        ("highest_tip_above_table_mm",),
        *[(band_name(band),) for band in BANDS],
        ("tip_above_table_when_first_within_30_mm_mm",),
        ("last_descent", "steps_before_letting_go"),
        ("last_descent", "tip_from_large_cube_centre_when_it_starts_mm"),
        ("last_descent", "tip_above_table_when_it_starts_mm"),
        ("tip_above_table_at_letting_go_mm",),
        ("tip_from_large_cube_centre_at_letting_go_mm",),
        ("nearest_the_tip_came_to_the_large_cube_centre_mm",),
        ("red_cube_from_large_cube_centre_at_letting_go_mm",),
        ("red_cube_from_the_tip_at_letting_go_mm",),
        ("red_cube_centre_below_the_tip_at_letting_go_mm",),
    ]
    return {"carries": len(rows), **{": ".join(keys): spread(rows, *keys) for keys in numbers}}


def main(output: str, *runs: str) -> None:
    result = ANALYSIS / output
    if result.exists():
        raise SystemExit(f"{result} exists; choose another name")
    if not runs:
        raise SystemExit("name at least one run folder")
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        table_mm = 1000.0 * float(inner.TABLE_SURFACE_Z)
        tests = test_carries(inner, list(runs), table_mm)
        humans = recording_carries(inner, table_mm)
    finally:
        env.close()
    groups = {
        "recordings": humans,
        "tests that stack": [row for row in tests if row["stacked"]],
        "tests that lift and do not stack": [row for row in tests if not row["stacked"]],
    }
    summaries = {name: summary(rows) for name, rows in groups.items()}
    for name, values in summaries.items():
        print(name, json.dumps(values), flush=True)
    result.write_text(
        json.dumps({"runs": list(runs), "summary": summaries, "tests": tests, "recordings": humans},
                   indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
