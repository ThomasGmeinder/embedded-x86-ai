"""What touches the red cube around its landing on the large cube?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    replay_landings_with_contacts.py <result .json in analysis/> <after_the_landing.py's result .json in analysis/> [<run folder name>/<test folder name> ...]

No model is run. A finished test is played again from its stored commands:
the environment is built as the test worker builds it, started from the
test's environment seed, given the same stacking goal, and stepped with the
stored command of every step, up to STEPS_AFTER steps past the landing that
after_the_landing.py found. Without test names, every test with a landing
is replayed.

The replay is checked against the test: after every step the two cubes must
be where the test stored them. The largest difference is reported; a test
whose replay differs by more than REPEATS_MM is marked and not to be read.

For the steps from STEPS_BEFORE before the landing to the end of the replay:
- the gripper's command and reading;
- the red cube: distance of its centre from the large cube's centre, its
  underside above the large cube's top, its tilt (0 is lying flat on a
  face), its speed sideways and its turning speed;
- which robot links press on the red cube and how hard, and how hard the red
  cube presses on the large cube (the scene's contact forces, newton);
- where the red cube's tilt comes from. "The grip" is the last step before
  the landing at which both jaws press on the red cube while it still
  stands on the table. Since then: how far the hand has tipped (the tilt a
  cube would have that is held rigidly in the hand), and how far the cube
  has turned inside the hand.

Writes analysis/<result>.json and refuses to overwrite it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
from random_run_grasp_and_release import ANALYSIS
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_env_worker import _new_environment
from so101_smolvla.stack_recording_rig import ARM_JOINTS

TASK = "stack2-recording-rig"
SOURCE, TARGET = "red", "blue"
STEPS_BEFORE = 8
STEPS_AFTER = 10
# The same threshold the robot's own "is touching" uses.
TOUCHING_NEWTON = 1e-2
REPEATS_MM = 0.5
# The red cube still stands on the table while it is no higher than this above its start.
ON_THE_TABLE_MM = 1.0
# The state is the arm joints followed by the gripper.
GRIPPER = len(ARM_JOINTS)


def tilt_deg(rotation: np.ndarray) -> float:
    """Angle between the vertical and the cube axis nearest to it."""
    return float(np.degrees(np.arccos(np.clip(np.abs(rotation[2]).max(), -1.0, 1.0))))


def tipped_deg(now: np.ndarray, then: np.ndarray) -> float:
    """How far a body has tipped: the angle of what was vertical then to the vertical now."""
    return float(np.degrees(np.arccos(np.clip((now @ then.T)[2, 2], -1.0, 1.0))))


def turned_deg(now: np.ndarray, then: np.ndarray) -> float:
    """The angle of the one turn that takes a body from its orientation then to that of now."""
    return float(np.degrees(np.arccos(np.clip((np.trace(now @ then.T) - 1.0) / 2.0, -1.0, 1.0))))


def replay(test: Path, landing: int) -> dict:
    import torch

    result = json.loads((test / "result.json").read_text())
    if result["task_mode"] != TASK:
        raise SystemExit(f"{test} is a test of {result['task_mode']}, not of {TASK}")
    steps = np.load(test / "trajectory.npz")
    commands = steps["commanded"]
    stored = {colour: 1000.0 * steps[f"cube_position_{colour}"] for colour in (SOURCE, TARGET)}
    last = min(landing + STEPS_AFTER, len(commands) - 1)
    env = _new_environment(environment_arguments(TASK), int(result["steps_per_command"]) + 10)
    rows = []
    differs = 0.0
    started = time.monotonic()
    try:
        env.reset(seed=int(result["environment_seed"]))
        inner = env._env.unwrapped
        inner.set_stack_goal(SOURCE, TARGET)
        red, large = inner.cubes[SOURCE], inner.cubes[TARGET]
        sizes = {colour: 1000.0 * size for colour, size in inner.cube_sizes.items()}
        links = list(inner.agent.robot.links)
        hand = inner.agent.robot.links_map["gripper_link"]
        jaws = (inner.agent.finger1_link, inner.agent.finger2_link)

        def numbers(value) -> np.ndarray:
            return value[0].detach().cpu().numpy().astype(np.float64)

        def orientation(body) -> np.ndarray:
            return numbers(body.pose.to_transformation_matrix())[:3, :3]

        def presses(link) -> float:
            return float(torch.linalg.norm(
                inner.scene.get_pairwise_contact_forces(link, red), axis=1)[0])

        start_height = 1000.0 * float(numbers(red.pose.p)[2])
        grip: dict | None = None
        for step in range(last + 1):
            observation, _, _, _, _ = env.step(commands[step].astype(np.float32))
            place = {SOURCE: 1000.0 * numbers(red.pose.p), TARGET: 1000.0 * numbers(large.pose.p)}
            differs = max(differs, *(float(np.abs(place[colour] - stored[colour][step]).max())
                                     for colour in place))
            if (step < landing and place[SOURCE][2] - start_height < ON_THE_TABLE_MM
                    and all(presses(jaw) >= TOUCHING_NEWTON for jaw in jaws)):
                grip = {"step": step, "hand": orientation(hand), "cube": orientation(red)}
            if step < landing - STEPS_BEFORE:
                continue
            since_the_grip = {}
            if grip is not None:
                since_the_grip = {
                    "hand_tipped_since_the_grip_deg": round(
                        tipped_deg(orientation(hand), grip["hand"]), 1),
                    "cube_turned_in_the_hand_since_the_grip_deg": round(turned_deg(
                        orientation(hand).T @ orientation(red), grip["hand"].T @ grip["cube"]), 1),
                }
            pressing = {}
            for link in links:
                force = presses(link)
                if force >= TOUCHING_NEWTON:
                    pressing[link.name] = round(force, 2)
            offset = place[SOURCE] - place[TARGET]
            rows.append({
                "step": step,
                "steps_after_landing": step - landing,
                "gripper_command_pct": round(float(commands[step][GRIPPER]), 1),
                "gripper_reading_pct": round(float(np.asarray(observation["agent_pos"])[GRIPPER]), 1),
                "from_the_centre_mm": round(float(np.linalg.norm(offset[:2])), 1),
                "underside_above_the_top_mm": round(
                    float(offset[2] - sizes[SOURCE] / 2 - sizes[TARGET] / 2), 1),
                "tilt_deg": round(tilt_deg(orientation(red)), 1),
                **since_the_grip,
                "sideways_speed_mm_s": round(
                    float(1000.0 * np.linalg.norm(numbers(red.linear_velocity)[:2])), 1),
                "turning_speed_deg_s": round(
                    float(np.degrees(np.linalg.norm(numbers(red.angular_velocity)))), 1),
                "robot_links_pressing_on_the_red_cube_newton": pressing,
                "red_cube_on_large_cube_newton": round(float(torch.linalg.norm(
                    inner.scene.get_pairwise_contact_forces(red, large), axis=1)[0]), 2),
            })
    finally:
        env.close()
    return {
        "run": test.parent.name,
        "test": test.name,
        "stacked": bool(result["success"]),
        "landing_step": landing,
        "grip_step": grip["step"] if grip is not None else None,
        "cube_tilt_at_the_grip_deg": round(tilt_deg(grip["cube"]), 1) if grip is not None else None,
        "steps_replayed": last + 1,
        "seconds": round(time.monotonic() - started, 1),
        "replay_differs_from_the_test_by_mm": round(differs, 3),
        "replay_repeats_the_test": differs <= REPEATS_MM,
        "steps": rows,
    }


def main(output: str, landings: str, *chosen: str) -> None:
    result_file = ANALYSIS / output
    if result_file.exists():
        raise SystemExit(f"{result_file} exists; choose another name")
    read = json.loads((ANALYSIS / landings).read_text())
    # Run folders as after_the_landing.py was given them, from the workshop folder.
    folder_of = {Path(run).name: Path(run) for run in read["runs"]}
    found = {
        f"{row['run']}/{row['test']}": (folder_of[row["run"]] / row["test"], row["landing_step"])
        for row in read["tests"]
        if row["landed_on_the_large_cube"]
    }
    unknown = sorted(set(chosen) - set(found))
    if unknown:
        raise SystemExit(f"no landing in {landings} for {unknown}")
    tests = []
    for name in chosen or sorted(found):
        row = replay(*found[name])
        tests.append(row)
        print(f"{name}: {'stacked' if row['stacked'] else 'not stacked'}, landing at step "
              f"{row['landing_step']}, {row['steps_replayed']} steps in {row['seconds']} s, "
              f"replay differs by {row['replay_differs_from_the_test_by_mm']} mm", flush=True)
        for step in row["steps"]:
            print("  " + json.dumps(step), flush=True)
    result_file.write_text(
        json.dumps({"landings": landings, "steps_before": STEPS_BEFORE, "steps_after": STEPS_AFTER,
                    "tests": tests}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
