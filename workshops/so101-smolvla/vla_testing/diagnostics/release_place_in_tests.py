"""Where is the gripper tip above the large cube when a random stacking test lets go, or holds on?

Run with .venv-sapien from the workshop folder, PYTHONPATH = <workshop>/src :
/tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics:
    release_place_in_tests.py <result .json in analysis/> <run folder in vla_testing/> ...

Reads the stored random stacking tests; nothing is simulated forward. Every
test that lifted the red cube is measured at one moment:
  stacked, dropped   the first target after the lift that opens the gripper
                     (release_aim_check.opening_index);
  still held         the test ended with the red cube in the air: the median
                     over all steps from the first one within 45 mm of the
                     large cube's centre to the end.
At that moment the simulated robot is posed at the test's joint readings and
the gripper tip is measured against the large cube, as release_aim_check.py
measures the model's answers: past the cube (along the line from the robot's
base through the cube), left of it, and above the table.

The humans of the real recordings are measured the same way, on the
recordings of analysis/release-aim-pictures.npz: at the frame the human lets
go, and at the last frame before it at which the tip was still 30 mm higher.

A run folder with ASKED_EVERY.json (random_stacking_asked_every.py) is
grouped under that number of steps, any other under the checkpoint's 50.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from release_aim_check import (
    ANALYSIS, FRAMES_BEFORE_RELEASE, PICTURES, opening_index, recorded, tip_at)

VLA_TESTING = Path(__file__).resolve().parents[1]
LIFTED_M = 0.010
OVER_THE_LARGE_CUBE_MM = 45.0
STILL_HIGHER_MM = 30.0
# How far back from the human's letting go the tip is followed (frames of 1/30 s).
FRAMES_LOOKED_BACK = 200
RED_HALF_MM, LARGE_HALF_MM = 10.0, 20.0


def along_and_across(offset_mm: np.ndarray, large_xy: np.ndarray) -> tuple[float, float]:
    """Offset past the large cube and left of it, as in release_aim_check.report (lines 300-301)."""
    outward = large_xy / np.linalg.norm(large_xy)
    leftward = np.array([-outward[1], outward[0]])
    return float(offset_mm @ outward), float(offset_mm @ leftward)


def measured(inner, table_mm: float, readings: np.ndarray, red: np.ndarray, large: np.ndarray) -> list[float]:
    """Tip past, left of and above the large cube's resting place; red cube the same; cube against tip."""
    tip = 1000.0 * tip_at(inner, readings)
    red, large = 1000.0 * red, 1000.0 * large
    tip_past, tip_left = along_and_across(tip[:2] - large[:2], large[:2])
    red_past, red_left = along_and_across(red[:2] - large[:2], large[:2])
    resting_tip = table_mm + 2.0 * LARGE_HALF_MM + RED_HALF_MM
    return [tip_past, tip_left, float(tip[2] - resting_tip),
            red_past, red_left, float(red[2] - large[2] - LARGE_HALF_MM - RED_HALF_MM),
            float(tip[2] - red[2])]


NAMES = ("tip past the large cube", "tip left of it", "tip above the red cube's resting centre",
         "red cube past the large cube", "red cube left of it", "red cube's underside above the large cube's top",
         "tip above the red cube's centre")


def tests(inner, table_mm: float, folders: list[str]) -> dict[str, list]:
    groups: dict[str, list] = {}
    for folder in folders:
        run = VLA_TESTING / folder
        asked = run / "ASKED_EVERY.json"
        every = json.loads(asked.read_text())["model_asked_every_steps"] if asked.exists() else 50
        for test in sorted(run.glob("seed-*")):
            result = json.loads((test / "result.json").read_text())
            stored = np.load(test / "trajectory.npz")
            red, large = stored["cube_position_red"], stored["cube_position_blue"]
            up = np.flatnonzero(red[:, 2] - red[0, 2] > LIFTED_M)
            if not len(up):
                continue
            lift = int(up[0])
            if result["failure_stage"] == "final stack unstable":
                off = 1000.0 * np.linalg.norm(red[:, :2] - large[:, :2], axis=1)
                over = np.flatnonzero((off <= OVER_THE_LARGE_CUBE_MM) & (np.arange(len(off)) >= lift))
                if not len(over) or 1000.0 * (red[-1, 2] - large[-1, 2]) < 2.0 * RED_HALF_MM:
                    kind, steps = "lifted, other end", []
                else:
                    kind, steps = "still held at the end", list(range(int(over[0]), len(red)))
            elif result["failure_stage"] in (None, "dropped cube"):
                opening = opening_index(stored["target"][lift:])
                kind = "stacked" if result["success"] else "dropped"
                steps = [] if opening is None else [lift + opening]
                if steps and red[steps[0], 2] - red[0, 2] <= LIFTED_M:
                    kind, steps = f"{kind}, cube back on the table before the gripper opened", []
            else:
                kind, steps = f"lifted, {result['failure_stage']}", []
            row = [folder[-6:], int(result["environment_seed"]), len(steps)]
            if steps:
                values = np.array([measured(inner, table_mm, stored["achieved"][s], red[s], large[s])
                                   for s in steps])
                row += np.median(values, axis=0).round(1).tolist()
            groups.setdefault(f"asked every {every}: {kind}", []).append(row)
    return groups


def humans(inner, table_mm: float) -> dict[str, list]:
    stored = np.load(PICTURES)
    data = recorded()
    at_release, higher = [], []
    for number, frame, large in zip(stored["episode"].tolist(), stored["frame"].tolist(),
                                    stored["large_cube"]):
        state = data["state"][data["episode"] == number]
        first = max(0, frame + FRAMES_BEFORE_RELEASE - FRAMES_LOOKED_BACK)
        release = frame + FRAMES_BEFORE_RELEASE - first
        tips = 1000.0 * np.array([tip_at(inner, row) for row in state[first: first + release + 1]])
        resting_tip = table_mm + 2.0 * LARGE_HALF_MM + RED_HALF_MM

        def row(index: int, frames_before: int) -> list:
            past, left = along_and_across(tips[index, :2] - 1000.0 * large[:2], 1000.0 * large[:2])
            return [int(number), frames_before, round(past, 1), round(left, 1),
                    round(float(tips[index, 2] - resting_tip), 1)]

        at_release.append(row(release, 0))
        above = np.flatnonzero(tips[:release, 2] >= tips[release, 2] + STILL_HIGHER_MM)
        if len(above):
            higher.append(row(int(above[-1]), release - int(above[-1])))
    return {"human: at the frame of letting go": at_release,
            f"human: last frame with the tip {STILL_HIGHER_MM:g} mm higher than at letting go": higher}


def main(output: str, *folders: str) -> None:
    destination = ANALYSIS / output
    if destination.exists():
        raise SystemExit(f"{destination} exists; choose another name")
    missing = [f for f in folders if not (VLA_TESTING / f / "COMPLETE.json").exists()]
    if missing or not folders:
        raise SystemExit(f"run folders without COMPLETE.json: {missing}")

    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        table_mm = 1000.0 * float(inner.TABLE_SURFACE_Z)
        result = {"columns of a test": ["run", "seed", "steps measured", *NAMES],
                  "columns of a recording": ["recording", "frames before letting go", "tip past the large cube",
                                             "tip left of it", "tip above the red cube's resting centre"],
                  **tests(inner, table_mm, list(folders)), **humans(inner, table_mm)}
    finally:
        env.close()

    np.set_printoptions(linewidth=220, suppress=True)
    for name, rows in result.items():
        if name.startswith("columns"):
            continue
        full = [row for row in rows if len(row) > 3]
        print(f"\n{name}: {len(rows)}, measured {len(full)}")
        if not full:
            continue
        human = name.startswith("human")
        values = np.array([row[2:] if human else row[3:] for row in full], dtype=float)
        for column, label in enumerate(result["columns of a recording"][2:] if human else NAMES):
            print(f"   {label:48s} median {np.median(values[:, column]):6.1f}   sorted "
                  f"{np.sort(values[:, column]).round().astype(int)}")
        if human:
            print(f"   {'frames before letting go':48s} median {np.median([r[1] for r in full]):6.1f}   sorted "
                  f"{np.sort([r[1] for r in full])}")
    destination.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {destination}")


if __name__ == "__main__":
    main(*sys.argv[1:])
