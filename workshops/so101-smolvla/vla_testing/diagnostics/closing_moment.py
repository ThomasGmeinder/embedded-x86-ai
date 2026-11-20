"""Where is the gripper when it starts to close: the model in a test run against the humans?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    closing_moment.py <task> <run folder> <label>

Nothing is simulated forward and no model is run. Every joint row is posed in
the task's own scene (its conversion from model numbers to joint angles and
its table height). Gripper tip = midpoint between the two fingertips. The
closing moment is the first one grip_events.grips finds in the gripper
targets, the same rule for the model and for the humans.

For the test run (model) and for the 187 real stacking recordings (humans),
at the closing moment and at the lowest point of the episode:
  tip height above the scene's table, from the readings and from the targets
  target minus reading of the lift, elbow and wrist-flex joints
  how far the tip is from the red cube, along and across the direction from
  the robot base (test run only; the humans' cube places are defined by
  their grasp)
Writes analysis/closing-moment-<label>.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import so101_smolvla.sapien_stack_env  # noqa: F401  registers the tasks
from grip_events import grips
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
JOINTS = ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper")


def middle(rows: list[dict], key: str) -> list[float]:
    values = np.array([row[key] for row in rows if row.get(key) is not None], dtype=np.float64)
    return [round(float(v), 1) for v in np.percentile(values, [10, 50, 90])]


def main() -> None:
    task_mode, run, label = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    env = So101SimEnv(**environment_arguments(task_mode), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        table = float(inner.TABLE_SURFACE_Z)

        def tips(rows: np.ndarray) -> np.ndarray:
            found = []
            for row in rows:
                qpos = inner._simulated_qpos(row.astype(np.float64))
                inner.agent.robot.set_qpos(
                    torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
                found.append(np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0))
            return 1000.0 * np.array(found)

        def moment(reading: np.ndarray, target: np.ndarray) -> dict | None:
            """Facts of one episode; None if the gripper never closes on something."""
            pairs = grips(target[:, 5])
            if not pairs:
                return None
            close = pairs[0][0]
            reading_tips, target_tips = tips(reading), tips(target)
            lowest = int(reading_tips[:, 2].argmin())
            gap = target[close] - reading[close]
            return {
                "close_step": close,
                "tip_above_table_at_closing_mm": float(reading_tips[close, 2] - 1000 * table),
                "target_tip_above_table_at_closing_mm": float(target_tips[close, 2] - 1000 * table),
                "lowest_tip_above_table_mm": float(reading_tips[lowest, 2] - 1000 * table),
                "lowest_target_tip_above_table_mm": float(target_tips[:, 2].min() - 1000 * table),
                "steps_from_lowest_to_closing": close - lowest,
                **{f"reading_{name}_at_closing": float(reading[close, index])
                   for index, name in enumerate(JOINTS)},
                **{f"target_minus_reading_{name}_at_closing": float(gap[index])
                   for index, name in enumerate(JOINTS[:5])},
                "tip_xy_at_closing": reading_tips[close, :2],
            }

        model = []
        for directory in sorted(run.glob("seed-*")):
            data = np.load(directory / "trajectory.npz")
            row = moment(data["achieved"], data["target"])
            seed = int(directory.name.split("-")[1])
            if row is None:
                print(f"seed {seed}: the gripper never closed on anything", flush=True)
                continue
            cube = 1000.0 * data["cube_position_red"][row["close_step"], :2]
            outward = cube / np.linalg.norm(cube)
            off = row.pop("tip_xy_at_closing") - cube
            row["tip_past_red_cube_mm"] = float(off @ outward)
            row["tip_left_of_red_cube_mm"] = float(off @ np.array([-outward[1], outward[0]]))
            row["red_cube_lifted_mm"] = float(
                1000.0 * (data["cube_position_red"][:, 2] - data["cube_position_red"][0, 2]).max())
            model.append(row)
            print(
                f"seed {seed}: closes at step {row['close_step']}, tip "
                f"{row['tip_above_table_at_closing_mm']:.0f} mm above the table "
                f"(target {row['target_tip_above_table_at_closing_mm']:.0f}), "
                f"{row['tip_past_red_cube_mm']:+.0f} mm past and "
                f"{row['tip_left_of_red_cube_mm']:+.0f} mm left of the red cube, "
                f"cube lifted {row['red_cube_lifted_mm']:.0f} mm",
                flush=True,
            )

        data = np.load(ANALYSIS / "states-real-stacking.npz")
        humans = []
        for number in np.unique(data["episode"]):
            chosen = data["episode"] == number
            row = moment(data["state"][chosen], data["action"][chosen])
            if row is not None:
                row.pop("tip_xy_at_closing")
                humans.append(row)
    finally:
        env.close()

    keys = [key for key in model[0] if key != "close_step"]
    report = {
        "task": task_mode,
        "run": run.name,
        "percentiles": "10th, median, 90th",
        "model": {"episodes": len(model), **{key: middle(model, key) for key in keys}},
        "humans": {
            "episodes": len(humans),
            **{key: middle(humans, key) for key in keys if key in humans[0]},
        },
    }
    print(json.dumps(report, indent=1), flush=True)
    (ANALYSIS / f"closing-moment-{label}.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
