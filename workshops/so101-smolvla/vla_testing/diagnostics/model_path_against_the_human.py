"""Where does the model's way to the red cube leave the human's, in episodes started as recordings started?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after stacking_at_recorded_places.py (paired) and
random_run_grasp_and_release.py on its run:
    model_path_against_the_human.py <run folder> <its grasp-and-release .json in analysis/> <result .json in analysis/>

Nothing is simulated. An episode of that run starts with the cubes and the
arm where one recording had them, so the model's way and the human's way to
the same red cube can be laid side by side. Both are joint readings (the
model's from the stored episode, the human's from the recording) posed on the
robot of the stack2-recording-rig scene; the gripper tip is the point halfway
between the two fingertips.

Per episode, for the model and for the human:
- the step at which the gripper has closed for the first time after having
  been open (the model's from the grasp-and-release result, the human's from
  analysis/cube-placement.npz);
- the tip against the red cube's starting place at that step and at steps
  0, 50 and 100: along the line from the robot base through the cube
  ("past"), across it ("left") and above the table;
- how far the tip travels per step on the way there (median over the steps
  between leaving the start and closing).

Writes analysis/<result>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from random_run_grasp_and_release import along_and_across
from release_aim_check import recorded, recording_rig_scene, tip_at

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
STEPS_LOOKED_AT = (0, 50, 100)
# The arm counts as having left its start once the tip is this far from where it began.
LEFT_THE_START_MM = 10.0


def way(inner, readings: np.ndarray, closed: int, red: np.ndarray, table_mm: float) -> dict:
    """The tip against the red cube's place along one way to the cube."""
    tips = 1000.0 * np.array([tip_at(inner, reading) for reading in readings[: closed + 1]])

    def against_cube(step: int) -> dict:
        past, left = along_and_across(tips[step] - red, red)
        return {"past_mm": round(past, 1), "left_mm": round(left, 1),
                "above_table_mm": round(float(tips[step, 2] - table_mm), 1)}

    away = np.flatnonzero(np.linalg.norm(tips - tips[0], axis=1) > LEFT_THE_START_MM)
    left_start = int(away[0]) if len(away) else closed
    per_step = np.linalg.norm(np.diff(tips[left_start:], axis=0), axis=1)
    return {
        "closes_at_step": closed,
        "leaves_the_start_at_step": left_start,
        "tip_travel_mm_per_step_median": round(float(np.median(per_step)), 2) if len(per_step) else None,
        "at_closing": against_cube(closed),
        **{f"at_step_{step}": against_cube(step) for step in STEPS_LOOKED_AT if step <= closed},
    }


def main(run: str, facts_file: str, output: str) -> None:
    facts = {episode["seed"]: episode
             for episode in json.loads((ANALYSIS / facts_file).read_text())["episodes"]}
    events = np.load(ANALYSIS / "cube-placement.npz")
    human_close = dict(zip(events["training_episode"].tolist(),
                           events["training_close_frame"].tolist()))
    episode_of, state = recorded()["episode"], recorded()["state"]
    env = recording_rig_scene()
    rows = []
    try:
        inner = env._env.unwrapped
        table_mm = 1000.0 * float(inner.TABLE_SURFACE_Z)
        for number, episode in sorted(facts.items()):
            if "grasp" not in episode:
                continue
            folder = Path(run) / f"seed-{number:05d}"
            steps = np.load(folder / "trajectory.npz")
            red = 1000.0 * steps["cube_position_red"][0]
            result = json.loads((folder / "result.json").read_text())
            row = {
                "recording": number,
                "lifted": episode["red_cube_highest_mm"] > 10.0,
                "stacked": bool(result["success"]),
                "model": way(inner, steps["achieved"], episode["grasp"]["step"], red, table_mm),
                "human": way(inner, state[episode_of == number], human_close[number], red, table_mm),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        env.close()

    def median(who: str, *keys: str, chosen=lambda row: True) -> float | None:
        values = []
        for row in rows:
            value = row[who]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            if value is not None and chosen(row):
                values.append(value)
        return round(float(np.median(values)), 1) if values else None

    groups = {"all": lambda row: True, "lifted": lambda row: row["lifted"],
              "not lifted": lambda row: not row["lifted"]}
    summary = {
        name: {
            "episodes": sum(1 for row in rows if chosen(row)),
            **{f"{who}: {' '.join(keys)}": median(who, *keys, chosen=chosen)
               for who in ("model", "human")
               for keys in (("closes_at_step",), ("leaves_the_start_at_step",),
                            ("tip_travel_mm_per_step_median",),
                            ("at_closing", "past_mm"), ("at_closing", "left_mm"),
                            ("at_closing", "above_table_mm"),
                            ("at_step_50", "past_mm"), ("at_step_100", "past_mm"),
                            ("at_step_100", "above_table_mm"))},
        }
        for name, chosen in groups.items()
    }
    for name, values in summary.items():
        print(name, json.dumps(values), flush=True)
    (ANALYSIS / output).write_text(
        json.dumps({"run": run, "summary": summary, "episodes": rows}, indent=1) + "\n",
        encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
