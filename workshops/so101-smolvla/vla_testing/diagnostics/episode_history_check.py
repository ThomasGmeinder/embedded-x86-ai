"""Does a stacking test run the same way whatever ran before it in the simulator?

Finished random stacking tests store the arm command of every step. This
replays those commands, without the model, through the scene the product
worker builds - one test after the other in one simulator process - and
reports for every test

  - the first step at which the replayed arm reading or a cube's place
    differs from what the run stored,
  - with ``--pictures``, the first step at which a camera picture differs
    from the stored one, and
  - whether the replay ran exactly like the earlier replays of the same test
    in this process.

``--start`` chooses what the simulator does before every test:

  kept             move the robot and the cubes back (the simulator's reset)
  rebuilt-scene    have the simulator build its physics scene anew
  new-environment  close the environment and make a new one

Usage:
  episode_history_check.py [--start <way>] [--pictures] --save <result .json>
      <run folder>:<seed>[:<first steps only>] ...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

TASK_MODE = "stack2-recording-rig"
CUBE_COLOURS = ("red", "blue")
PHYSICS_ARRAYS = ("achieved", *(f"cube_position_{colour}" for colour in CUBE_COLOURS))
CAMERAS = ("top", "wrist")
STARTS = ("kept", "rebuilt-scene", "new-environment")


def listed_test(entry: str) -> tuple[Path, int, int | None]:
    """``<run folder>:<seed>``, or with ``:<steps>`` only that many first steps."""
    folder, *numbers = entry.split(":")
    if len(numbers) not in (1, 2):
        raise ValueError(f"expected <run folder>:<seed>[:<steps>], got {entry!r}")
    return Path(folder), int(numbers[0]), int(numbers[1]) if numbers[1:] else None


def stored_test(
    folder: Path, seed: int, steps: int | None, pictures: bool
) -> dict[str, np.ndarray]:
    episode = folder / f"seed-{seed:05d}"
    result = json.loads((episode / "result.json").read_text(encoding="utf-8"))
    if result["task_mode"] != TASK_MODE or result["environment_seed"] != seed:
        raise ValueError(f"{episode} is not seed {seed} of {TASK_MODE}")
    if len(result["commands"]) != 1:
        raise ValueError(f"{episode} holds {len(result['commands'])} commands, not 1")
    with np.load(episode / "trajectory.npz") as arrays:
        stored = {name: arrays[name] for name in ("commanded", *PHYSICS_ARRAYS)}
    if pictures:
        with np.load(episode / "policy-observations.npz") as arrays:
            stored.update({name: arrays[name] for name in CAMERAS})
    if steps is None:
        return stored
    if not 1 <= steps <= len(stored["commanded"]):
        raise ValueError(f"{episode} has {len(stored['commanded'])} steps, not {steps}")
    return {name: rows[:steps] for name, rows in stored.items()}


def first_difference(one: np.ndarray, other: np.ndarray) -> int | None:
    """First step at which two courses differ; None when they are equal."""
    if one.shape != other.shape:
        raise ValueError(f"courses of {one.shape} and {other.shape} steps")
    steps = np.flatnonzero(np.any(one.reshape(len(one), -1) != other.reshape(len(other), -1), axis=1))
    return int(steps[0]) if len(steps) else None


def make_environment(episode_length: int):
    """The environment as the product worker makes it."""
    import so101_smolvla.sapien_stack_env  # noqa: F401
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _new_environment

    return _new_environment(environment_arguments(TASK_MODE), episode_length)


def start(env, seed: int, rebuild_scene: bool) -> None:
    """Start one test as the product worker does."""
    from so101_smolvla.sapien_env_worker import (
        _configure_robot_appearance,
        _scene_facts,
    )

    if rebuild_scene:
        _, info = env._env.reset(seed=seed, options={"reconfigure": True})
        if not info["reconfigure"]:
            raise RuntimeError("the simulator did not build its scene anew")
        _configure_robot_appearance(env)
    else:
        env.reset(seed=seed)
    _scene_facts(env)
    env._env.unwrapped.set_stack_goal(*CUBE_COLOURS)
    _scene_facts(env)


def replay(env, commands: np.ndarray, pictures: bool) -> dict[str, np.ndarray]:
    """Play stored commands; return what the scene shows after every step."""
    from so101_smolvla.sapien_env_worker import _scene_facts

    names = (*PHYSICS_ARRAYS, *(CAMERAS if pictures else ()))
    course: dict[str, list] = {name: [] for name in names}
    for command in commands:
        obs, *_ = env.step(np.asarray(command, dtype=np.float32))
        facts = _scene_facts(env)
        course["achieved"].append(np.asarray(obs["agent_pos"], dtype=np.float32))
        for colour in CUBE_COLOURS:
            course[f"cube_position_{colour}"].append(facts["cube_positions"][colour])
        if pictures:
            for camera in CAMERAS:
                course[camera].append(obs["pixels"][camera])
    arrays = {
        name: np.asarray(course[name], dtype=np.float32)
        for name in PHYSICS_ARRAYS
    }
    arrays["achieved"] = arrays["achieved"].astype(np.float64)
    arrays.update({camera: np.stack(course[camera]) for camera in names if camera in CAMERAS})
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", choices=STARTS, default="kept")
    parser.add_argument("--pictures", action="store_true")
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("tests", nargs="+", metavar="<run folder>:<seed>[:<steps>]")
    args = parser.parse_args()
    if args.save.exists():
        raise FileExistsError(f"refusing to overwrite {args.save}")

    tests = [listed_test(entry) for entry in args.tests]
    stored = {test: stored_test(*test, args.pictures) for test in set(tests)}
    episode_length = max(len(arrays["commanded"]) for arrays in stored.values()) + 10
    compared = (*PHYSICS_ARRAYS, *(CAMERAS if args.pictures else ()))

    rows = []
    earlier: dict[tuple, list[tuple[int, dict[str, np.ndarray]]]] = {}
    env = None
    try:
        for position, test in enumerate(tests):
            folder, seed, _ = test
            began = time.perf_counter()
            if env is None or args.start == "new-environment":
                if env is not None:
                    env.close()
                    env = None
                env = make_environment(episode_length)
            start(env, seed, args.start == "rebuilt-scene")
            start_seconds = time.perf_counter() - began
            began = time.perf_counter()
            course = replay(env, stored[test]["commanded"], args.pictures)
            replay_seconds = time.perf_counter() - began
            row = {
                "position": position,
                "run": folder.name,
                "seed": seed,
                "steps": int(len(stored[test]["commanded"])),
                "start_seconds": start_seconds,
                "replay_seconds": replay_seconds,
                "first_difference_from_stored": {
                    name: first_difference(course[name], stored[test][name])
                    for name in compared
                },
                "first_difference_from_earlier_replay": {
                    str(other): {
                        name: first_difference(course[name], before[name])
                        for name in compared
                    }
                    for other, before in earlier.get(test, [])
                },
            }
            earlier.setdefault(test, []).append((position, course))
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        if env is not None:
            env.close()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.save.write_text(
        json.dumps({"start": args.start, "tests": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
