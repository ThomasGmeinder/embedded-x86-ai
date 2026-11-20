"""Run random stacking tests of the recording-rig scene with chosen cube places.

Imported by same_tests_with_more_room.py and earlier_layouts_with_other_noise.py
(.venv, PYTHONPATH = <workshop>/src). A test here is the random stacking test
of its environment seed in everything but the cube places: same scene, same
sentence, same start pose and same turn of each cube (the scene draws both
from the environment seed, see chosen_layout_worker.py), same episode runner
(sapien_rollout.run_stack_episode), same number of steps, same stack check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import so101_smolvla.sapien_env_client as client
from chosen_layout import choose
from so101_smolvla.policy_env import create_policy_environment
from so101_smolvla.record import accumulate
from so101_smolvla.sapien_rollout import (
    DEFAULT_STACK_STEPS,
    _load_policy,
    _write_stack_report,
    run_stack_episode,
)

TASK = "stack2-recording-rig"
WORKER_LAUNCHER = Path(__file__).with_name("chosen_layout_python")


@dataclass(frozen=True)
class ChosenTest:
    name: str
    environment_seed: int
    policy_seed: int
    layout: dict[str, list[float]]
    # Written into the test's row of summary.json.
    notes: dict[str, object] = field(default_factory=dict)


def went(row: dict) -> str:
    """One line on how a test of a summary.json ended."""
    stage = row["commands"][0]
    if row["success"]:
        return f"stack at step {stage['success_step']}"
    return f"{row['failure_stage']} (red cube lifted: {stage['lifted']})"


def run_chosen_layouts(tests: list[ChosenTest], destination: Path) -> list[dict]:
    """Run the tests into a new folder with the files of a random stacking run, COMPLETE.json excepted."""
    destination.mkdir(parents=True)
    client.SAPIEN_PYTHON = WORKER_LAUNCHER
    checkpoint, policy, pre, post = _load_policy()
    rows = []
    for test in tests:
        # One simulator worker per test: it reads the chosen layout when it starts.
        choose(test.layout)
        env = create_policy_environment("sapien", horizon=DEFAULT_STACK_STEPS, task_mode=TASK)
        try:
            row = run_stack_episode(
                env, policy, pre, post, checkpoint, destination / test.name,
                task_mode=TASK,
                environment_seed=test.environment_seed,
                policy_seed=test.policy_seed,
                steps_per_command=DEFAULT_STACK_STEPS,
            )
        finally:
            env.close()
        if row["initial_layout"] != test.layout:
            raise RuntimeError(
                f"{test.name}: the scene placed {row['initial_layout']}, not {test.layout}")
        row.update(test.notes)
        rows.append(row)
        print(f"{test.name}: {json.dumps(test.notes)}; now: {went(row)}", flush=True)
    (destination / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    accumulate([destination / test.name for test in tests], destination, "overall")
    _write_stack_report(rows, destination)
    return rows


def mark_complete(destination: Path, rows: list[dict], **facts: object) -> None:
    (destination / "COMPLETE.json").write_text(json.dumps(
        {"complete": True, "episodes": len(rows), "task_mode": TASK, **facts},
        indent=2) + "\n", encoding="utf-8")
