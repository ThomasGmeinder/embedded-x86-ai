"""Table height and joint zero points of the recording robot, from its own grasps.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    geometry_from_grasps.py

No model, nothing simulated forward. In every real stacking recording the
small cube stands on the table when the human closes the gripper on it, and
on the 40 mm cube when the human lets it go. So over all recordings the
gripper tip must be at one height at the grasp and exactly 40 mm higher at
the release, wherever on the table that happens. The recorded joint readings
are posed on the simulated robot model and three explanations are fitted:
  table      only the table height is unknown
  table+lift table height and the lift joint's zero point
  joints     table height and the zero points of lift, elbow and wrist flex
Each fit is made on the even-numbered recordings and checked on the odd ones.
Grasp and release frames are those of grip_events.grips (cube-placement.npz).
Writes analysis/geometry-from-grasps.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the scenes)
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
LARGE_CUBE = 0.040
SHIFTED = {"table": (), "table+lift": (1,), "joints": (1, 2, 3)}
NAMES = {1: "lift", 2: "elbow", 3: "wrist_flex"}


def main() -> None:
    places = np.load(ANALYSIS / "cube-placement.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state = data["episode"], data["state"]
    grasp_rows, release_rows, numbers = [], [], []
    for number, close, release in zip(
        places["training_episode"].tolist(),
        places["training_close_frame"].tolist(),
        places["training_release_frame"].tolist(),
    ):
        chosen = np.flatnonzero(episode == number)
        grasp_rows.append(state[chosen[min(int(close) + 10, int(release) - 1)]])
        release_rows.append(state[chosen[int(release)]])
        numbers.append(int(number))
    grasp_rows = np.array(grasp_rows, dtype=np.float64)
    release_rows = np.array(release_rows, dtype=np.float64)
    numbers = np.array(numbers)

    # Only the robot model of the scene is used. Tip heights are read in the
    # robot base frame, so the scene's own table height plays no part.
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def tips(readings: np.ndarray) -> np.ndarray:
            out = []
            for reading in readings:
                qpos = inner._simulated_qpos(reading)
                inner.agent.robot.set_qpos(
                    torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
                out.append(np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0))
            return np.array(out)

        def heights(readings: np.ndarray, joints: tuple[int, ...], shift: np.ndarray):
            moved = readings.copy()
            for joint, value in zip(joints, shift):
                moved[:, joint] += value
            tip = tips(moved)
            return tip[:, 2], np.linalg.norm(tip[:, :2], axis=1)

        def residuals(params, joints, chosen):
            grasp_height, shift = params[0], params[1:]
            grasp_z, _ = heights(grasp_rows[chosen], joints, shift)
            release_z, _ = heights(release_rows[chosen], joints, shift)
            return np.concatenate([
                grasp_z - grasp_height, release_z - grasp_height - LARGE_CUBE])

        def describe(params, joints, chosen) -> dict:
            grasp_height, shift = params[0], params[1:]
            grasp_z, grasp_reach = heights(grasp_rows[chosen], joints, shift)
            release_z, release_reach = heights(release_rows[chosen], joints, shift)
            error = np.concatenate([
                grasp_z - grasp_height, release_z - grasp_height - LARGE_CUBE])
            slope_grasp = np.polyfit(grasp_reach, grasp_z, 1)[0]
            slope_release = np.polyfit(release_reach, release_z, 1)[0]
            return {
                "recordings": int(len(grasp_z)),
                "grasp_tip_height_mm_median": round(1000 * float(np.median(grasp_z)), 1),
                "release_tip_height_mm_median": round(1000 * float(np.median(release_z)), 1),
                "release_above_grasp_mm_median": round(
                    1000 * float(np.median(release_z - grasp_z)), 1),
                "grasp_height_mm_per_100mm_reach": round(100 * float(slope_grasp), 1),
                "release_height_mm_per_100mm_reach": round(100 * float(slope_release), 1),
                "error_mm_median_abs": round(1000 * float(np.median(np.abs(error))), 1),
                "error_mm_rms": round(1000 * float(np.sqrt(np.mean(error ** 2))), 1),
            }

        even, odd = numbers % 2 == 0, numbers % 2 == 1
        everything = np.ones(len(numbers), dtype=bool)
        result = {"recordings": int(len(numbers))}
        for name, joints in SHIFTED.items():
            start = np.zeros(1 + len(joints))
            start[0] = float(np.median(heights(grasp_rows, (), np.zeros(0))[0]))
            fits = {}
            for label, chosen in (("even", even), ("all", everything)):
                solved = least_squares(
                    residuals, start, args=(joints, chosen), loss="soft_l1",
                    f_scale=0.005, diff_step=1e-3, x_scale=[0.005] + [1.0] * len(joints))
                fits[label] = solved.x
            row = {
                "fitted_on_all": {
                    "grasp_tip_height_relative_to_robot_base_mm": round(1000 * fits["all"][0], 1),
                    "zero_point_shift_deg": {
                        NAMES[j]: round(float(v), 2) for j, v in zip(joints, fits["all"][1:])},
                    **describe(fits["all"], joints, everything),
                },
                "fitted_on_even": {
                    "grasp_tip_height_relative_to_robot_base_mm": round(1000 * fits["even"][0], 1),
                    "zero_point_shift_deg": {
                        NAMES[j]: round(float(v), 2) for j, v in zip(joints, fits["even"][1:])},
                },
                "even_fit_checked_on_odd": describe(fits["even"], joints, odd),
            }
            result[name] = row
            print(name, json.dumps(row, indent=1), flush=True)
    finally:
        env.close()
    (ANALYSIS / "geometry-from-grasps.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
