"""The random stacking test with the robot's base not drawn in the model's pictures.

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder:
    random_stacking_base_not_drawn.py <number of tests> [<policy seed base>]
    random_stacking_base_not_drawn.py check        only the check below, writes nothing

The table picture of the stack2-recording-rig scene holds the recording
robot's own base; the scene draws its robot's base on top of it. Offline
(top_picture_parts.py, scene_without_drawn_base.py) the drawn base is the
part of the drawn robot that moves the model's answers away from those it
gives to the real pictures, during the reach, at the grasp and at the
release. Here the product's random stacking test is run with the base link
not drawn.

Nothing else differs from the product's test: this calls the product's own
batch runner (sapien_rollout.run_stack_batch: same scene, sentence, seeds,
step limit, stack check and files) and starts the simulator worker through
base_not_drawn_python, whose one change is described in
base_not_drawn_worker.py.

Check before any test: the first top picture of environment seed 0 is taken
from the product's worker and from that worker. If they do not differ, the
base is still drawn and nothing is run.

Writes vla_testing/random-stacking-recording-rig-base-not-drawn-sapien-<time>/
with the files of a random stacking run and BASE_NOT_DRAWN.json.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import so101_smolvla.sapien_env_client as client
import so101_smolvla.sapien_rollout as rollout
from so101_smolvla.policy_env import create_policy_environment

TASK = "stack2-recording-rig"
WORKER_LAUNCHER = Path(__file__).with_name("base_not_drawn_python")
CHECK = "check"


def first_top_picture() -> np.ndarray:
    """The top picture at the start of environment seed 0, from the worker in use."""
    env = create_policy_environment(
        "sapien", horizon=rollout.DEFAULT_STACK_STEPS, task_mode=TASK)
    try:
        return env.reset(0).top.copy()
    finally:
        env.close()


def main(count: str, policy_seed_base: str = "0") -> None:
    as_built = first_top_picture()
    client.SAPIEN_PYTHON = WORKER_LAUNCHER
    changed = int(np.any(as_built != first_top_picture(), axis=2).sum())
    if not changed:
        raise SystemExit("the base is still drawn: the first top picture is the product's")
    print(f"base not drawn: {changed} pixels of the first top picture of seed 0 differ "
          f"from the product's", flush=True)
    if count == CHECK:
        return

    destination = rollout.VLA_TESTING / (
        f"random-stacking-recording-rig-base-not-drawn-sapien-{datetime.now():%Y%m%d-%H%M%S}")
    rows = rollout.run_stack_batch(
        int(count), task_mode=TASK, destination=destination,
        policy_seed_base=int(policy_seed_base))
    (destination / "BASE_NOT_DRAWN.json").write_text(json.dumps(
        {"simulator_worker": "base_not_drawn_worker.py",
         "top_picture_pixels_changed_at_start_of_seed_0": changed,
         "policy_seed_base": int(policy_seed_base)}, indent=2) + "\n", encoding="utf-8")
    print(f"stacked {sum(bool(row['success']) for row in rows)} of {len(rows)} in {destination}",
          flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
