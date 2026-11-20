"""Where do tests with the cubes close together gain: at the grasp or after the lift?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    stacks_by_cube_distance.py <result .json in analysis/> <run folder of stack2-recording-rig> ...

Nothing is simulated. Every test of the named run folders is read once: a
test that a later folder repeats (same environment seed, same policy seed,
same cube places) is counted the first time only. Grasp, lift and release of
a test are those of random_run_grasp_and_release.episode_facts.

Per test: how far apart the cubes' centres started, how far the red cube
started from the robot base, whether the red cube rose 10 mm, whether the
scene reported the stack, and, where the gripper let go after a lift, how far
the red cube then was from the large cube's centre and how much of the way
from its own start to the large cube it had been carried.

Over all tests, grouped by the cubes' distance (below the smallest distance
of the finger-room rule, finger_room_rule.room_for_the_fingers, up to 100 mm,
beyond) and by the red cube's distance from the robot: tests, lifts, stacks,
stacks per lift, and the median miss when the gripper lets go.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from finger_room_rule import room_for_the_fingers
from random_run_grasp_and_release import ANALYSIS, LIFTED_MM, episode_facts
from release_aim_check import recording_rig_scene
from so101_smolvla.stack_layouts import TRAINING_CUBE_CLEARANCES

RULE_MM = 1000.0 * float(room_for_the_fingers(TRAINING_CUBE_CLEARANCES)[0, 1])
FAR_APART_MM = 100.0
# Red cubes nearer than this are the ones the gripper closes above (DEVELOPMENT_LEDGER.md).
NEAR_THE_ROBOT_MM = 265.0


def test_row(inner, folder: Path) -> dict:
    result = json.loads((folder / "result.json").read_text())
    facts = episode_facts(inner, folder)
    steps = np.load(folder / "trajectory.npz")
    red, large = 1000.0 * steps["cube_position_red"], 1000.0 * steps["cube_position_blue"]
    way = (large[0] - red[0])[:2]
    apart = float(np.linalg.norm(way))
    row = {
        "run": folder.parent.name,
        "test": folder.name,
        "environment_seed": result["environment_seed"],
        "policy_seed": result["policy_seed"],
        "layout": result["initial_layout"],
        "cubes_apart_mm": round(apart, 1),
        "red_cube_from_robot_mm": round(float(np.linalg.norm(red[0, :2])), 1),
        "lifted": facts["red_cube_highest_mm"] > LIFTED_MM,
        "stacked": bool(result["success"]),
        "failure_stage": result["failure_stage"],
    }
    release = facts.get("release")
    if release is not None:
        carried = (red[release["step"]] - red[0])[:2] @ way / apart
        row.update({
            "let_go_at_step": release["step"],
            "let_go_from_large_cube_centre_mm": release["red_cube_from_large_cube_centre_mm"],
            "let_go_underside_above_large_cube_top_mm": release[
                "red_cube_underside_above_large_cube_top_mm"],
            "carried_towards_large_cube_mm": round(float(carried), 1),
            "left_to_carry_mm": round(float(apart - carried), 1),
        })
    return row


def counts(rows: list[dict]) -> dict:
    lifted = [row for row in rows if row["lifted"]]
    let_go = [row for row in lifted if "let_go_at_step" in row]
    summary = {
        "tests": len(rows),
        "lifted": len(lifted),
        "stacked": sum(row["stacked"] for row in rows),
        "stacked_without_lift": sum(row["stacked"] and not row["lifted"] for row in rows),
        "lifted_and_let_go": len(let_go),
    }
    if let_go:
        summary.update({
            "let_go_from_large_cube_centre_mm_median": round(float(np.median(
                [row["let_go_from_large_cube_centre_mm"] for row in let_go])), 1),
            "left_to_carry_mm_median": round(float(np.median(
                [row["left_to_carry_mm"] for row in let_go])), 1),
            "let_go_within_20_mm": sum(
                row["let_go_from_large_cube_centre_mm"] <= 20.0 for row in let_go),
            "let_go_below_large_cube_top": sum(
                row["let_go_underside_above_large_cube_top_mm"] < -5.0 for row in let_go),
        })
    return summary


def main(output: str, *runs: str) -> None:
    path = ANALYSIS / output
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        rows, seen = [], set()
        for run in runs:
            for folder in sorted(Path(run).glob("seed-*")):
                row = test_row(inner, folder)
                key = (row["environment_seed"], row["policy_seed"], json.dumps(row["layout"]))
                if key not in seen:
                    seen.add(key)
                    rows.append(row)
    finally:
        env.close()
    for row in rows:
        print(json.dumps({name: value for name, value in row.items() if name != "layout"}),
              flush=True)

    close = [row for row in rows if row["cubes_apart_mm"] < RULE_MM]
    middle = [row for row in rows if RULE_MM <= row["cubes_apart_mm"] < FAR_APART_MM]
    far = [row for row in rows if row["cubes_apart_mm"] >= FAR_APART_MM]
    near_robot = [row for row in rows if row["red_cube_from_robot_mm"] < NEAR_THE_ROBOT_MM]
    away = [row for row in rows if row["red_cube_from_robot_mm"] >= NEAR_THE_ROBOT_MM]
    let_go = [row for row in rows if "let_go_at_step" in row]
    result = {
        "tests": len(rows),
        f"cubes closer than {RULE_MM:.1f} mm": counts(close),
        f"cubes {RULE_MM:.1f} to {FAR_APART_MM:.0f} mm apart": counts(middle),
        f"cubes {FAR_APART_MM:.0f} mm apart or more": counts(far),
        f"red cube nearer than {NEAR_THE_ROBOT_MM:.0f} mm from the robot": counts(near_robot),
        "red cube farther from the robot": counts(away),
        f"closer than {RULE_MM:.1f} mm, red cube far from the robot": counts(
            [row for row in close if row in away]),
        f"{RULE_MM:.1f} mm or more, red cube far from the robot": counts(
            [row for row in middle + far if row in away]),
        f"closer than {RULE_MM:.1f} mm, red cube near the robot": counts(
            [row for row in close if row in near_robot]),
        f"{RULE_MM:.1f} mm or more, red cube near the robot": counts(
            [row for row in middle + far if row in near_robot]),
        "tests in which the gripper let go after a lift": len(let_go),
        "correlation of the cubes' distance with the miss when letting go": round(float(
            np.corrcoef([row["cubes_apart_mm"] for row in let_go],
                        [row["let_go_from_large_cube_centre_mm"] for row in let_go])[0, 1]), 2),
        "correlation of the cubes' distance with the way left to carry": round(float(
            np.corrcoef([row["cubes_apart_mm"] for row in let_go],
                        [row["left_to_carry_mm"] for row in let_go])[0, 1]), 2),
        "episodes": rows,
    }
    for name, value in result.items():
        if name != "episodes":
            print(name, json.dumps(value), flush=True)
    path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
