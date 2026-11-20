"""Does the model stack when the test starts as a recording started?

Run with .venv, PYTHONPATH = <workshop>/src, from the workshop folder, after
test_places_against_recordings.py has written analysis/recorded-cube-places.npz:
    stacking_at_recorded_places.py <number of episodes> <paired|unpaired> [<steps> [<asked every>]]

The random stacking test draws both cubes evenly in the reachable workspace.
The recordings had the large cube in a much smaller patch. Here an episode of
the stack2-recording-rig scene starts with the two cubes where one recording
had them (the scene's own replay start), and the unchanged model does the
rest. Nothing else differs from the random stacking test: same scene, same
sentence, same episode runner (sapien_rollout.run_stack_episode), same
number of steps, same stack check.

"paired": the arm starts in the start pose of the same recording.
"unpaired": the arm starts in the start pose of another recording, drawn at
random, as the random stacking test pairs cube places and start poses.

Recordings are drawn at random (seed 0, the same recordings for both modes)
from those whose two cubes lie inside the central reachable workspace
(reachable_workspace.is_reachable_xy with the cubes' clearances). The
environment seed of an episode is the recording's number (it turns the cubes
about their upright axis), the policy seed is the episode's position in the run.

<asked every>: the model answers with 50 joint targets. The checkpoint's own
setting plays all 50 before the model sees new pictures. With a number here
only that many targets of each answer are played, then the model is asked
again. Left out, the checkpoint's setting stays.

Writes vla_testing/random-stacking-recorded-places-<mode>[-asked-every-<n>]-sapien-<time>/
with the same files as a random stacking run.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from so101_smolvla.policy_env import create_policy_environment
from so101_smolvla.reachable_workspace import is_reachable_xy
from so101_smolvla.record import accumulate
from so101_smolvla.sapien_rollout import (
    VLA_TESTING,
    _load_policy,
    _write_stack_report,
    run_stack_episode,
)
from so101_smolvla.stack_initial_states import TRAINING_INITIAL_STATES
from so101_smolvla.stack_layouts import TRAINING_CUBE_CLEARANCES

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
TASK = "stack2-recording-rig"
DRAW_SEED = 0


def recordings_in_reach() -> list[tuple[int, list[float], list[float]]]:
    """(recording, red cube place, large cube place) in metres, for every recording in reach."""
    places = np.load(ANALYSIS / "recorded-cube-places.npz")
    red_clearance, large_clearance = TRAINING_CUBE_CLEARANCES
    return [
        (int(number), red.tolist(), large.tolist())
        for number, red, large in zip(
            places["episode"], places["red_mm"] / 1000.0, places["large_mm"] / 1000.0)
        if is_reachable_xy(red, red_clearance) and is_reachable_xy(large, large_clearance)
    ]


def main(count: str, mode: str, steps: str = "600", asked_every: str | None = None) -> None:
    if mode not in ("paired", "unpaired"):
        raise SystemExit(f"mode is paired or unpaired, not {mode!r}")
    in_reach = recordings_in_reach()
    rng = np.random.default_rng(DRAW_SEED)
    chosen = sorted(rng.choice(len(in_reach), size=int(count), replace=False).tolist())
    # Drawn after the recordings, so both modes run the same recordings.
    other_starts = rng.integers(0, len(TRAINING_INITIAL_STATES), size=int(count)).tolist()
    print(f"{len(in_reach)} recordings have both cubes in reach; {count} drawn, {mode}", flush=True)

    asking = "" if asked_every is None else f"-asked-every-{int(asked_every)}"
    destination = VLA_TESTING / (
        f"random-stacking-recorded-places-{mode}{asking}-sapien-{datetime.now():%Y%m%d-%H%M%S}")
    destination.mkdir(parents=True)
    checkpoint, policy, pre, post = _load_policy()
    if asked_every is not None:
        # The policy's own queue length; every episode's policy.reset() builds the queue from it.
        policy.config.n_action_steps = int(asked_every)
    print(f"the model is asked every {policy.config.n_action_steps} steps", flush=True)
    env = create_policy_environment("sapien", horizon=int(steps), task_mode=TASK)
    rows, folders = [], []
    try:
        for index, row in enumerate(chosen):
            number, red, large = in_reach[row]
            start = number if mode == "paired" else int(other_starts[index])
            folders.append(destination / f"seed-{number:05d}")
            rows.append(run_stack_episode(
                env, policy, pre, post, checkpoint, folders[-1],
                task_mode=TASK,
                environment_seed=number,
                policy_seed=index,
                steps_per_command=int(steps),
                stack_replay={"state_index": start, "layout": {"red": red, "blue": large}},
            ))
            rows[-1]["start_pose_of_recording"] = start
    finally:
        env.close()
    (destination / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    accumulate(folders, destination, "overall")
    _write_stack_report(rows, destination)
    (destination / "COMPLETE.json").write_text(json.dumps(
        {"complete": True, "episodes": len(rows), "task_mode": TASK, "mode": mode,
         "model_asked_every_steps": int(policy.config.n_action_steps),
         "recordings": [row["environment_seed"] for row in rows]}, indent=2) + "\n",
        encoding="utf-8")
    print(f"stacked {sum(bool(row['success']) for row in rows)} of {len(rows)} in {destination}",
          flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
