"""Which explanation of the recording robot puts the red cube right in the wrist picture?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    wrist_cube_check.py

Independent check of calibrate_recording_robot.py: that fit used grasp
heights and the top pictures only. Here the simulated robot is posed at the
recorded readings shortly before the human closes the gripper, under each
explanation (its zero-point shifts, pan factor and table height), with the
red cube standing where that explanation says the grasp happened. The wrist
picture is rendered and the red cube is found in it with the same detector
that found it in the real wrist picture of the same moment
(analysis/wrist-frames-before-grasp.npz).

An explanation is good if the cube lands on the same pixels as in the real
picture. A constant difference over all pictures would be the camera's mount
on the hand; what an explanation cannot hide is scatter around that constant.

Writes analysis/wrist-cube-check.json and analysis/wrist-cube-check.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose

import so101_smolvla.sapien_stack_env as stack_env
from measure_objects import measure
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_env_worker import _configure_robot_appearance
from so101_smolvla.stack_recording_rig import RECORDING_RIG

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
JOINTS = {"lift": 1, "elbow": 2, "wrist_flex": 3}
SHEET_FRAMES_BEFORE_CLOSING = 20
SHEET_RECORDINGS = 5


def main() -> None:
    fitted = json.loads((ANALYSIS / "recording-robot-calibration.json").read_text())
    # name -> (shift added to the recorded reading, pan factor, table below robot base in m)
    explanations: dict[str, tuple[np.ndarray, float, float]] = {
        "rig scene today": (
            np.array([*np.rad2deg(RECORDING_RIG.arm_zero_points_rad), 0.0]),
            1.0,
            RECORDING_RIG.table_below_robot_base,
        )
    }
    for name in ("as recorded", "pan factor", "lift", "lift + elbow", "lift + elbow + wrist"):
        values = fitted[name]["fitted_on_all"]
        shift = np.zeros(6)
        for joint, degrees in values["zero_point_shift_deg"].items():
            shift[JOINTS[joint]] = degrees
        explanations[name] = (
            shift, float(values["pan_factor"]), values["table_below_robot_base_mm"] / 1000.0
        )

    real = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    places = np.load(ANALYSIS / "cube-placement.npz")
    close_of = dict(zip(places["training_episode"].tolist(), places["training_close_frame"].tolist()))
    release_of = dict(zip(places["training_episode"].tolist(), places["training_release_frame"].tolist()))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    states = {
        number: data["state"][data["episode"] == number].astype(np.float64)
        for number in set(real["episode"].tolist())
    }

    env = So101SimEnv(**environment_arguments("stack2-training"), episode_length=10)
    report: dict[str, dict] = {}
    sheet: dict[str, list[np.ndarray]] = {}
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        table_position = inner.table_pose.p.clone()

        def pose_robot(reading: np.ndarray, shift: np.ndarray, factor: float) -> np.ndarray:
            moved = reading + shift
            moved[0] *= factor
            qpos = stack_env._real_state_to_sim_qpos(moved)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)

        def stand(color: str, xy: np.ndarray, drop: float) -> None:
            half = inner.cube_sizes[color] / 2
            inner.cubes[color].set_pose(Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), half - drop]],
                dtype=torch.float32, device=inner.device)))

        for name, (shift, factor, drop) in explanations.items():
            lowered = table_position.clone()
            lowered[..., 2] -= drop
            inner.table_scene.table.set_pose(Pose.create_from_pq(p=lowered, q=inner.table_pose.q))
            difference, ratio, before, shown = [], [], [], []
            for number, frame, frames_before, centre, side in zip(
                real["episode"].tolist(), real["frame"].tolist(),
                real["frames_before_closing"].tolist(),
                real["red_centre_px"], real["red_side_px"],
            ):
                state = states[number]
                close, release = int(close_of[number]), int(release_of[number])
                stand("red", pose_robot(state[min(close + 10, release - 1)], shift, factor), drop)
                stand("blue", pose_robot(state[release], shift, factor), drop)
                pose_robot(state[frame], shift, factor)
                picture = env._format_raw_obs(inner.get_obs())["pixels"]["wrist"]
                if frames_before == SHEET_FRAMES_BEFORE_CLOSING and len(shown) < SHEET_RECORDINGS:
                    shown.append(picture)
                found = measure(picture, "red")
                if found is None or found["touches_edge"]:
                    continue
                difference.append(np.array(found["centre_px"]) - centre)
                ratio.append(found["side_px"] / side)
                before.append(frames_before)
            sheet[name] = shown
            difference, ratio, before = np.array(difference), np.array(ratio), np.array(before)

            def summary(chosen: np.ndarray) -> dict:
                offset = np.median(difference[chosen], axis=0)
                return {
                    "pictures_compared": int(chosen.sum()),
                    "simulated_minus_real_px_median": np.round(offset, 1).tolist(),
                    "distance_px_median": round(float(np.median(
                        np.linalg.norm(difference[chosen], axis=1))), 1),
                    "scatter_around_median_px_median": round(float(np.median(
                        np.linalg.norm(difference[chosen] - offset, axis=1))), 1),
                    "simulated_size_over_real_median": round(float(np.median(ratio[chosen])), 3),
                }

            report[name] = {
                "zero_point_shift_deg": {
                    "pan": float(shift[0]), **{j: float(shift[i]) for j, i in JOINTS.items()}},
                "pan_factor": factor,
                "table_below_robot_base_mm": round(1000 * drop, 1),
                "all pictures": summary(np.ones(len(before), dtype=bool)),
                **{f"{value} frames before closing": summary(before == value)
                   for value in sorted(set(before.tolist()), reverse=True)},
            }
            print(name, json.dumps(report[name], indent=1), flush=True)
    finally:
        env.close()

    (ANALYSIS / "wrist-cube-check.json").write_text(
        json.dumps(report, indent=1) + "\n", encoding="utf-8")
    chosen = np.flatnonzero(real["frames_before_closing"] == SHEET_FRAMES_BEFORE_CLOSING)[:SHEET_RECORDINGS]
    columns = [list(real["wrist"][chosen])] + [sheet[name] for name in explanations]
    rows = [
        np.concatenate([cv2.resize(column[index], (320, 240)) for column in columns], axis=1)
        for index in range(len(chosen))
    ]
    cv2.imwrite(
        str(ANALYSIS / "wrist-cube-check.png"),
        cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR),
    )
    print("sheet columns: real, " + ", ".join(explanations), flush=True)


if __name__ == "__main__":
    main()
