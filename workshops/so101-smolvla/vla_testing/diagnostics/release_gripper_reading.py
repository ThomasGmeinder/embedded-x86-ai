"""Does the gripper opening at which the scene carries the red cube change where the model lets go?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    release_gripper_reading.py
then, for <name> = release-carrying-as-scene and release-jaws-as-scene, with
file names in analysis/:
    release_aim_check.py predict <name>-pictures.npz <name>-predictions.npz
    release_aim_check.py report  <name>-pictures.npz <name>-predictions.npz <name>.json

Moment and recordings are those of release_aim_check.py: 25 frames before the
human lets the red cube go. The recording robot carried its red cube with the
gripper reading about 6 per cent; the scene's jaws stop on its red cube at
about 12 per cent (red_cube_in_hand.py, gripper_from_wrist_pictures.py).
release_aim_check.py drew the scene's hand at the recorded reading and gave
the model the recorded reading. Here the scene's hand is drawn at the opening
the replayed grasp left it with, as a test run draws it:

  release-jaws-as-scene       the model is still told the recorded reading
  release-carrying-as-scene   the model is told the scene's reading, as in a
                              test run

Each file holds the scene's top and wrist picture, so release_aim_check.py
asks about its four pairs of real and scene pictures.

Writes analysis/release-jaws-as-scene-pictures.npz and
analysis/release-carrying-as-scene-pictures.npz. A word given as argument is
put behind "as-scene" in both names, so that a later state of the scene does
not replace the pictures of an earlier one; existing files are not replaced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from release_aim_check import recording_rig_scene, scene_facts, scenes_before_release

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
GRIPPER = 5


def spread(values) -> list[float]:
    """10 %, median and 90 % of the values."""
    return [round(float(v), 1) for v in np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90])]


def main(state_of_scene: str = "") -> None:
    jaws = ANALYSIS / f"release-jaws-as-scene{state_of_scene}-pictures.npz"
    carrying = ANALYSIS / f"release-carrying-as-scene{state_of_scene}-pictures.npz"
    for path in (jaws, carrying):
        if path.exists():
            raise SystemExit(f"refusing to overwrite {path}")
    env = recording_rig_scene()
    rows: dict[str, list] = {
        "episode": [], "frame": [], "recorded_state": [], "scene_gripper": [], "top": [], "wrist": [],
        "large_cube": [], "red_cube": [], "table_height": [],
    }
    try:
        inner = env._env.unwrapped
        # The gripper reading of the scene after the last replayed command.
        scene_gripper: dict[int, float] = {}

        def watch(number: int) -> None:
            scene_gripper[number] = float(
                env._from_sim(inner.agent.robot.get_qpos()[0].cpu().numpy())[GRIPPER])

        for number, frame, reading in scenes_before_release(env, watch):
            # The camera sits on the fixed jaw, and the cube keeps its place
            # against that jaw; only the moving jaw is set back.
            as_scene = np.asarray(reading, dtype=np.float64).copy()
            as_scene[GRIPPER] = scene_gripper[number]
            inner.agent.robot.set_qpos(torch.as_tensor(
                inner._simulated_qpos(as_scene)[None], dtype=torch.float32, device=inner.device))
            seen = env._format_raw_obs(inner.get_obs())["pixels"]
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["recorded_state"].append(reading)
            rows["scene_gripper"].append(scene_gripper[number])
            rows["top"].append(seen["top"].copy())
            rows["wrist"].append(seen["wrist"].copy())
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            print(f"recording {number}: gripper reads {reading[GRIPPER]:.1f} per cent in the "
                  f"recording, {scene_gripper[number]:.1f} in the scene", flush=True)
    finally:
        env.close()
    recorded = np.array(rows.pop("recorded_state"))
    scene = np.array(rows.pop("scene_gripper"))
    print(f"gripper reading while carrying, per cent (10 %, median, 90 %): recordings "
          f"{spread(recorded[:, GRIPPER])}, scene {spread(scene)}", flush=True)
    told_as_scene = recorded.copy()
    told_as_scene[:, GRIPPER] = scene
    shared = {name: np.array(values) for name, values in rows.items()}
    np.savez_compressed(jaws, state=recorded, **shared)
    np.savez_compressed(carrying, state=told_as_scene, **shared)


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
