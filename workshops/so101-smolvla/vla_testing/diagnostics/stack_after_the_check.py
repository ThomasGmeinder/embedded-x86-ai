"""Does the stack stand when the model keeps running after the stack check has passed?

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder:
    stack_after_the_check.py <run folder of a random stacking run> <result .json in analysis/> <seed> [<seed> ...]

The random stacking test ends a command at the first step at which the scene
reports the stack: the red cube resting on the large cube, the hand off the
cube and the arm at rest (sapien_stack_session.py). At that step the open
hand can still be around the cube. Here the same episodes (same environment
seed, policy seed and checkpoint as the run) are played for the run's full
number of steps and the scene's report of every step is only written down.
Up to the step at which the run ended, the arm must read as it did in the
run; that is checked.

Every fact is the scene's own, as it reports it with every step. Per seed:
the first step with the stack reported, how many steps report it, whether the
red cube still rests on the large cube at the last step, and what the hand
and the arm do then.

Writes analysis/<result>.json and, per step, analysis/<result>.npz.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from so101_smolvla.policy_env import create_policy_environment
from so101_smolvla.sapien_config import task_spec
from so101_smolvla.sapien_rollout import STACK_COMMANDS, _action, _load_policy
from so101_smolvla.sapien_stack_session import parse_stack_command

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
# The replayed arm must read as in the run, in degrees and per cent of the gripper's range.
SAME_READING = 1e-3


def main(run: str, output: str, *seeds: str) -> None:
    folders = [Path(run) / f"seed-{int(seed):05d}" for seed in seeds]
    recorded = [json.loads((folder / "result.json").read_text()) for folder in folders]
    task_mode = recorded[0]["task_mode"]
    steps = int(recorded[0]["steps_per_command"])
    (sentence,) = STACK_COMMANDS[task_mode]
    command = parse_stack_command(sentence)
    interface = f"{command.source}_on_{command.target}"
    text = task_spec(task_mode).instruction or command.canonical

    checkpoint, policy, pre, post = _load_policy()
    print(f"checkpoint revision {checkpoint.name}, task {task_mode}, {steps} steps", flush=True)
    env = create_policy_environment("sapien", horizon=steps, task_mode=task_mode)
    result: dict[str, dict] = {}
    per_step: dict[str, np.ndarray] = {}
    try:
        for folder, episode in zip(folders, recorded):
            in_run = np.load(folder / "trajectory.npz")["achieved"]
            # The order of run_stack_episode and StackCommandSession.execute.
            torch.manual_seed(episode["policy_seed"])
            torch.cuda.manual_seed_all(episode["policy_seed"])
            observation = env.reset(episode["environment_seed"])
            env.set_command(command.source, command.target)
            policy.reset()

            rows: dict[str, list] = {
                name: [] for name in
                ("achieved", "reported", "resting", "hand_off", "arm_at_rest", "from_centre_mm")}
            for _ in range(steps):
                outcome = env.step(_action(policy, pre, post, observation, text))
                observation = outcome.observation
                facts = outcome.facts
                rows["achieved"].append(outcome.achieved)
                rows["reported"].append(bool(outcome.success))
                rows["resting"].append(bool(facts["interfaces"][interface]["supported"]))
                rows["hand_off"].append(bool(facts["source_released"]))
                rows["arm_at_rest"].append(bool(facts["robot_static"]))
                rows["from_centre_mm"].append(1000.0 * facts["interfaces"][interface]["xy_distance"])
            arrays = {name: np.asarray(values) for name, values in rows.items()}

            apart = float(np.abs(arrays["achieved"][: len(in_run)] - in_run).max())
            if apart > SAME_READING:
                raise RuntimeError(
                    f"{folder.name}: the replayed arm is {apart} off the run's readings")
            reported = np.flatnonzero(arrays["reported"])
            resting = arrays["resting"]
            # The last step at which the cube was not resting; it has rested ever since.
            unsettled = np.flatnonzero(~resting)
            rests_since = int(unsettled[-1]) + 1 if len(unsettled) else 0
            result[folder.name] = {
                "run_ended_at_step": episode["commands"][0]["success_step"],
                "steps_played": steps,
                "arm_off_the_runs_readings": apart,
                "first_step_with_the_stack_reported": int(reported[0]) if len(reported) else None,
                "last_step_with_the_stack_reported": int(reported[-1]) if len(reported) else None,
                "steps_with_the_stack_reported": int(len(reported)),
                "steps_with_the_red_cube_resting_on_the_large_cube": int(resting.sum()),
                "at_the_last_step": {
                    "stack_reported": bool(arrays["reported"][-1]),
                    "red_cube_rests_on_the_large_cube": bool(resting[-1]),
                    "resting_without_a_break_since_step": rests_since if resting[-1] else None,
                    "hand_off_the_cube": bool(arrays["hand_off"][-1]),
                    "arm_at_rest": bool(arrays["arm_at_rest"][-1]),
                    "red_cube_from_large_cube_centre_mm": round(float(arrays["from_centre_mm"][-1]), 1),
                    "gripper_reading_pct": round(float(arrays["achieved"][-1, 5]), 1),
                },
            }
            print(folder.name, json.dumps(result[folder.name]), flush=True)
            for name, values in arrays.items():
                per_step[f"{folder.name}: {name}"] = values
    finally:
        env.close()
    (ANALYSIS / output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed((ANALYSIS / output).with_suffix(".npz"), **per_step)


if __name__ == "__main__":
    main(*sys.argv[1:])
