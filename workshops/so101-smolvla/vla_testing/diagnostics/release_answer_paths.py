"""Does the model open the gripper early on the right path, or is the path itself too high?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, with file names in analysis/:
    release_answer_paths.py [<pictures .npz> <answers .npz> <result .json>]
Without arguments: the files of release_aim_check.py.

release_aim_check.py measures the gripper tip at the one target that opens
the gripper. Here all 50 targets of every answer are posed on the simulated
robot, so the whole path of the answer is seen: how low it goes, when, and
where on that path the gripper opens. The human's own next 50 commands are
measured the same way. Nothing is simulated forward.

Per set of answers, medians over the recordings:
  height at the first target, lowest height and its target number, height at
  the last target (gripper tip above the table);
  target number at which the gripper opens, height there, and how far above
  the answer's own lowest point that is;
  distance from the large cube's centre at the lowest point.

Writes analysis/release-answer-paths.json (or the result file named).
"""

from __future__ import annotations

import json
import sys

import numpy as np

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
from release_aim_check import ANALYSIS, ANSWER_LENGTH, opening_index, pairs_in, tip_at
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments


def main(
    pictures: str = "release-aim-pictures.npz",
    predictions: str = "release-aim-predictions.npz",
    output: str = "release-answer-paths.json",
) -> None:
    stored = np.load(ANALYSIS / pictures)
    answers = np.load(ANALYSIS / predictions)
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    large = 1000.0 * stored["large_cube"][:, :2]
    table = 1000.0 * stored["table_height"]

    human = []
    for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist()):
        commands = data["action"][data["episode"] == number][frame:frame + ANSWER_LENGTH]
        # The recording ends inside the answer's reach: the last command stays.
        human.append(np.concatenate(
            [commands, np.repeat(commands[-1:], ANSWER_LENGTH - len(commands), axis=0)]))
    sets = {"human": np.array(human)}
    sets.update({name: answers[name] for name in pairs_in(stored)})

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    result: dict[str, dict] = {"recordings": len(large)}
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        for name, chunks in sets.items():
            rows = []
            for index, chunk in enumerate(chunks):
                tips = 1000.0 * np.array([tip_at(inner, target) for target in chunk])
                height = tips[:, 2] - table[index]
                lowest = int(np.argmin(height))
                opening = opening_index(chunk)
                rows.append((
                    height[0], height[lowest], lowest, height[-1],
                    np.linalg.norm(tips[lowest, :2] - large[index]),
                    np.nan if opening is None else opening,
                    np.nan if opening is None else height[opening],
                    np.nan if opening is None else height[opening] - height[lowest],
                ))
            columns = np.array(rows).T
            result[name] = {
                "height_at_first_target_mm": round(float(np.median(columns[0])), 1),
                "lowest_height_mm": round(float(np.median(columns[1])), 1),
                "lowest_height_target_number": float(np.median(columns[2])),
                "height_at_last_target_mm": round(float(np.median(columns[3])), 1),
                "from_large_cube_centre_at_lowest_mm": round(float(np.median(columns[4])), 1),
                "answers_that_open_the_gripper": int(np.isfinite(columns[5]).sum()),
                "opening_target_number": float(np.nanmedian(columns[5])),
                "height_at_opening_mm": round(float(np.nanmedian(columns[6])), 1),
                "opening_above_own_lowest_mm": round(float(np.nanmedian(columns[7])), 1),
                "answers_that_open_within_5_mm_of_own_lowest": int(
                    (columns[7][np.isfinite(columns[7])] <= 5.0).sum()),
            }
            print(name, json.dumps(result[name]), flush=True)
    finally:
        env.close()
    (ANALYSIS / output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
