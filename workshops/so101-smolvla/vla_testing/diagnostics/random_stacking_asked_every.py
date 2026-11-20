"""The random stacking test with the model asked more often than every 50 steps.

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder:
    random_stacking_asked_every.py <number of tests> <asked every> [<policy seed base>]

The model answers with 50 joint targets. The checkpoint's own setting plays
all 50 before the model sees new pictures, so the arm acts for up to 1.7
seconds on pictures it no longer sees; the humans of the recordings saw every
frame. In the stored random tests the gripper first closes on the cube at
target 25 or later of an answer in 51 of 60 tests and opens after the lift at
target 25 or later in 30 of 36. Here only the first <asked every> targets of
each answer are played, then the model is asked again with new pictures.

Nothing else differs from the product's random stacking test of the
stack2-recording-rig scene: this calls the product's own batch runner
(sapien_rollout.run_stack_batch: same scene, sentence, seeds, step limit,
stack check and files) and changes one number of the loaded policy, the one
its reset() builds the queue of targets from.

Writes vla_testing/random-stacking-recording-rig-asked-every-<n>-sapien-<time>/
with the files of a random stacking run and ASKED_EVERY.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

import so101_smolvla.sapien_rollout as rollout

TASK = "stack2-recording-rig"


def main(count: str, asked_every: str, policy_seed_base: str = "0") -> None:
    played = int(asked_every)
    load_policy = rollout._load_policy
    own_setting: list[int] = []

    def load_policy_asked_more_often():
        checkpoint, policy, pre, post = load_policy()
        own_setting.append(int(policy.config.n_action_steps))
        if not 1 <= played <= own_setting[0]:
            raise SystemExit(
                f"asked every {played}: the checkpoint answers with {own_setting[0]} targets")
        policy.config.n_action_steps = played
        print(f"the model is asked every {played} steps; the checkpoint's own setting is "
              f"{own_setting[0]}", flush=True)
        return checkpoint, policy, pre, post

    rollout._load_policy = load_policy_asked_more_often
    destination = rollout.VLA_TESTING / (
        f"random-stacking-recording-rig-asked-every-{played}-sapien-{datetime.now():%Y%m%d-%H%M%S}")
    rows = rollout.run_stack_batch(
        int(count), task_mode=TASK, destination=destination,
        policy_seed_base=int(policy_seed_base))
    (destination / "ASKED_EVERY.json").write_text(json.dumps(
        {"model_asked_every_steps": played, "checkpoint_setting": own_setting[0],
         "policy_seed_base": int(policy_seed_base)}, indent=2) + "\n", encoding="utf-8")
    print(f"stacked {sum(bool(row['success']) for row in rows)} of {len(rows)} in {destination}",
          flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
