"""Does a random layout put a cube where the resting hand already is?

For each task and seed: start the test, hold the arm at its start pose for
one step, and read from the scene's own facts how far the cube's centre is
from the midpoint of the fingertips, whether the jaws touch it, and how far
the first step pushed it.

Usage (model environment, it starts the simulator worker itself):
    PYTHONPATH=src .venv/bin/python -u \
        vla_testing/diagnostics/cube_in_resting_hand.py <result.json> [seeds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from so101_smolvla.sapien_env_client import SapienPolicyEnvironment

TASKS = ("stack2-recording-rig", "pick-place", "pick-place-cube20")
# A cube that the first step moves farther than this was placed into the hand.
PUSHED = 0.0005


def cubes(facts: dict) -> dict[str, tuple[np.ndarray, dict]]:
    """Place and grip facts of every cube of a scene, by name."""
    if "item_position" in facts:
        return {"cube": (np.asarray(facts["item_position"]), facts["grip"])}
    return {
        color: (np.asarray(place), facts["grip"][color])
        for color, place in facts["cube_positions"].items()
    }


def measure(task: str, seeds: range) -> dict:
    distances, in_hand = [], []
    env = SapienPolicyEnvironment(horizon=2, task_mode=task)
    try:
        for seed in seeds:
            first = env.reset(environment_seed=seed)
            placed = cubes(dict(env.facts))
            held = cubes(env.step(first.state).facts)
            for name, (place, grip) in placed.items():
                after, grip_after = held[name]
                pushed = float(np.linalg.norm(after - place))
                distances.append(grip["distance"])
                if pushed > PUSHED or grip["contact"] or grip_after["contact"]:
                    in_hand.append(
                        {
                            "seed": seed,
                            "cube": name,
                            "pushed_mm": round(1000 * pushed, 1),
                            "from_fingertip_midpoint_mm": round(
                                1000 * grip["distance"], 1
                            ),
                            "placed_at": np.round(place[:2], 3).tolist(),
                        }
                    )
            print(f"{task} seed {seed}: {len(in_hand)} in the hand so far", flush=True)
    finally:
        env.close()
    low, one, five, median = np.percentile(distances, [0, 1, 5, 50])
    return {
        "seeds": len(seeds),
        "cubes": len(distances),
        "fingertip_midpoint_to_cube_centre_mm": {
            "min": round(1000 * low, 1),
            "1 %": round(1000 * one, 1),
            "5 %": round(1000 * five, 1),
            "median": round(1000 * median, 1),
        },
        "cubes_in_the_resting_hand": in_hand,
    }


def main() -> None:
    output = Path(sys.argv[1])
    seeds = range(int(sys.argv[2]) if len(sys.argv) > 2 else 200)
    result = {task: measure(task, seeds) for task in TASKS}
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for task, facts in result.items():
        print(
            f"{task}: {len(facts['cubes_in_the_resting_hand'])} of {facts['cubes']} "
            f"cubes in the resting hand; nearest "
            f"{facts['fingertip_midpoint_to_cube_centre_mm']['min']} mm"
        )


if __name__ == "__main__":
    main()
