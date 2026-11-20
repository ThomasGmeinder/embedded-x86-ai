"""Does the model roll the wrist so that the jaws come square onto the red cube?

For every test of a finished random stacking run the stored arm commands are
replayed, without the model, up to the step at which the jaws first touch the
red cube (the run's own ``grip_step``). There the scene is asked three times
for the angle between the fingertip line and the nearest cube face:

  as it is            the arm as the test left it - must equal the stored value
  set again           the same joint angles set once more - must not change it
  starting roll       the same joint angles, wrist roll put back to where it
                      was at the start of the test

The third is the angle the jaws would have had if the model had moved the arm
exactly as it did and never rolled the wrist. If the model rolls towards a
cube face, "as it is" is smaller than "starting roll".

Usage (simulator environment):
    PYTHONPATH=src:/tmp/Xbotics-SO101-Sim:vla_testing/diagnostics \
        .venv-sapien/bin/python -u vla_testing/diagnostics/wrist_roll_at_the_grip.py \
        <run folder> <result .json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from episode_history_check import make_environment, stored_test, start

SOURCE = "red"
# The replay must give the angle the run stored (two decimals).
STORED_ANGLE_TOLERANCE_DEG = 0.01
# Setting the same joint angles again must leave the angle where it was.
SAME_POSE_TOLERANCE_DEG = 0.01


def angle_now(inner) -> float:
    from so101_smolvla.sapien_grip import grip_facts

    return float(grip_facts(inner.agent, inner.cubes[SOURCE])["finger_face_deg"])


def angle_with(inner, qpos) -> float:
    """The finger-to-face angle with the arm set to these joint angles."""
    inner.agent.robot.set_qpos(qpos)
    return angle_now(inner)


def measure(folder: Path, row: dict) -> dict:
    from so101_smolvla.sapien_env_worker import _scene_facts
    from so101_smolvla.stack_recording_rig import ARM_JOINTS

    roll = ARM_JOINTS.index("wrist_roll")
    seed = row["environment_seed"]
    command = row["commands"][0]
    grip_step = command["grip_step"]
    commands = stored_test(folder, seed, grip_step + 1, pictures=False)["commanded"]

    env = make_environment(len(commands) + 10)
    try:
        start(env, seed, rebuild_scene=False)
        inner = env._env.unwrapped
        at_start = inner.agent.robot.get_qpos().clone()
        angle_at_start = angle_now(inner)
        for arm_command in commands:
            env.step(np.asarray(arm_command, dtype=np.float32))
        facts = _scene_facts(env)["grip"][SOURCE]
        as_it_is = float(facts["finger_face_deg"])
        if abs(as_it_is - command["finger_face_deg"]) > STORED_ANGLE_TOLERANCE_DEG:
            raise RuntimeError(
                f"seed {seed}: the replay gives {as_it_is:.3f} degrees at step "
                f"{grip_step}, the run stored {command['finger_face_deg']}"
            )
        if bool(facts["contact"]) != command["grip"]:
            raise RuntimeError(f"seed {seed}: jaw contact differs from the stored test")

        at_grip = inner.agent.robot.get_qpos().clone()
        set_again = angle_with(inner, at_grip)
        if abs(set_again - as_it_is) > SAME_POSE_TOLERANCE_DEG:
            raise RuntimeError(
                f"seed {seed}: setting the same joint angles again changes the "
                f"angle from {as_it_is:.3f} to {set_again:.3f} degrees"
            )
        unrolled = at_grip.clone()
        unrolled[..., roll] = at_start[..., roll]
        starting_roll = angle_with(inner, unrolled)
    finally:
        env.close()

    rolled = float(np.rad2deg(at_grip[0, roll] - at_start[0, roll]))
    return {
        "seed": seed,
        "stacked": bool(row["success"]),
        "jaw_contact": bool(command["grip"]),
        "grip_step": grip_step,
        "wrist_rolled_deg": round(rolled, 2),
        "angle_at_start_deg": round(angle_at_start, 2),
        "angle_at_grip_deg": round(as_it_is, 2),
        "angle_at_grip_with_starting_roll_deg": round(starting_roll, 2),
    }


def spread(values) -> dict:
    numbers = np.asarray(list(values), dtype=np.float64)
    return {
        "count": int(len(numbers)),
        "mean": round(float(numbers.mean()), 1),
        "median": round(float(np.median(numbers)), 1),
        "max": round(float(numbers.max()), 1),
        "below_10": int((numbers < 10.0).sum()),
    }


def main() -> None:
    folder, output = Path(sys.argv[1]), Path(sys.argv[2])
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    rows = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    tests = []
    for row in rows:
        tests.append(measure(folder, row))
        print(json.dumps(tests[-1]), flush=True)

    actual = [test["angle_at_grip_deg"] for test in tests]
    unrolled = [test["angle_at_grip_with_starting_roll_deg"] for test in tests]
    gain = np.subtract(unrolled, actual)
    summary = {
        "run": folder.name,
        "angle_at_grip_deg": spread(actual),
        "angle_at_grip_with_starting_roll_deg": spread(unrolled),
        "wrist_rolled_abs_deg": spread(np.abs([t["wrist_rolled_deg"] for t in tests])),
        "tests_where_the_roll_brought_the_jaws_squarer": int((gain > 1.0).sum()),
        "tests_where_the_roll_made_no_difference_of_1_deg": int(
            (np.abs(gain) <= 1.0).sum()
        ),
        "tests_where_the_roll_took_the_jaws_off_square": int((gain < -1.0).sum()),
    }
    output.write_text(
        json.dumps({"summary": summary, "tests": tests}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {output}", flush=True)


if __name__ == "__main__":
    main()
