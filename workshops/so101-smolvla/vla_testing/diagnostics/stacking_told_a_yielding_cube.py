"""Does the model stack more often when its gripper reading is what a cube that gives way would leave?

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder:
    stacking_told_a_yielding_cube.py <number of tests> <policy seed base>

The recorded red cube is sponge. While a human carried it the recording
robot's gripper read 5.6 per cent (median of 186 recordings) with a command of
2.7: the reading stayed 3.0 above the command. The scene's red cube is rigid:
in 55 lifted tests of 20 November the gripper read 11.9 to 12.2 per cent with
a command of 3.1 to 3.9, 8.8 above the command. The gripper reading is one of
the six numbers the model is given with every pair of pictures.

Here the random stacking test of the recording-rig scene is run as the
product runs it (same layouts, start poses, runner, steps, stack check), with
one change on the policy side: at every step at which a jaw touches the red
cube and the gripper reads wider than it is commanded, the model is told a
reading that lies RECORDED_GAP / SCENE_GAP of the way from the command to the
true reading. The physics, the pictures and the stored readings ("achieved"
in trajectory.npz) are the scene's own; what the model was told is stored per
test in told_gripper_reading.npy.

This is a test of a cause, not a fix: it tells the model a number the scene
does not produce.

Writes vla_testing/random-stacking-told-a-yielding-cube-sapien-<time>/ with
the same files as a random stacking run.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import datetime

import numpy as np

from chosen_layout_tests import mark_complete, went
from so101_smolvla.policy_env import PolicyStep, create_policy_environment
from so101_smolvla.record import accumulate
from so101_smolvla.sapien_rollout import (
    DEFAULT_STACK_STEPS,
    VLA_TESTING,
    _load_policy,
    _write_stack_report,
    run_stack_episode,
)
from so101_smolvla.stack_recording_rig import ARM_JOINTS

TASK = "stack2-recording-rig"
# The state is the arm joints followed by the gripper.
GRIPPER = len(ARM_JOINTS)
# Reading minus command while the red cube is carried, per cent of the gripper's range.
RECORDED_GAP = 3.0
SCENE_GAP = 8.8


class ToldAYieldingCube:
    """The policy environment, with the gripper reading of a red cube that gives way."""

    def __init__(self, environment) -> None:
        self._environment = environment
        self.told: list[float] = []

    def __getattr__(self, name: str):
        return getattr(self._environment, name)

    def reset(self, environment_seed: int, **options):
        self.told = []
        return self._environment.reset(environment_seed, **options)

    def step(self, action: np.ndarray) -> PolicyStep:
        outcome = self._environment.step(action)
        reading = float(outcome.observation.state[GRIPPER])
        command = float(action[GRIPPER])
        if not (outcome.facts["grip"]["red"]["contact"] and reading > command):
            self.told.append(reading)
            return outcome
        state = outcome.observation.state.copy()
        state[GRIPPER] = command + RECORDED_GAP / SCENE_GAP * (reading - command)
        self.told.append(float(state[GRIPPER]))
        # "achieved" keeps the scene's own reading.
        return replace(outcome, observation=replace(outcome.observation, state=state))


def main(count: str, policy_seed_base: str) -> None:
    destination = VLA_TESTING / (
        f"random-stacking-told-a-yielding-cube-sapien-{datetime.now():%Y%m%d-%H%M%S}")
    destination.mkdir(parents=True)
    checkpoint, policy, pre, post = _load_policy()
    env = ToldAYieldingCube(
        create_policy_environment("sapien", horizon=DEFAULT_STACK_STEPS, task_mode=TASK))
    rows, folders = [], []
    try:
        for seed in range(int(count)):
            folders.append(destination / f"seed-{seed:05d}")
            row = run_stack_episode(
                env, policy, pre, post, checkpoint, folders[-1],
                task_mode=TASK,
                environment_seed=seed,
                policy_seed=int(policy_seed_base) + seed,
                steps_per_command=DEFAULT_STACK_STEPS,
            )
            told = np.asarray(env.told)
            np.save(folders[-1] / "told_gripper_reading.npy", told)
            scene = np.load(folders[-1] / "trajectory.npz")["achieved"][:, GRIPPER]
            row["steps_told_another_gripper_reading"] = int((told != scene).sum())
            rows.append(row)
            print(f"seed {seed}: {went(row)}; told another reading on "
                  f"{row['steps_told_another_gripper_reading']} of {len(told)} steps", flush=True)
    finally:
        env.close()
    (destination / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    accumulate(folders, destination, "overall")
    _write_stack_report(rows, destination)
    mark_complete(destination, rows, policy_seed_base=int(policy_seed_base),
                  recorded_gap=RECORDED_GAP, scene_gap=SCENE_GAP)
    print(f"stacked {sum(bool(row['success']) for row in rows)} of {len(rows)} "
          f"in {destination}", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
