"""Does the recording robot's arm stand where its joint readings say, in every pose?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    arm_pose_from_outline.py

No model, nothing simulated forward. The stack2-recording-rig scene gives
every arm joint one fixed zero point, the same in every pose. With it the
drawn robot misses the real robot's outline where the arm is stretched out
(robot_outline_by_link.py), and moving the top camera does not remove the
miss (calibrate_top_camera_from_outline.py). This script asks whether the
miss is a matter of pose: for each of the 114 real top pictures (three
moments of 38 recordings) the five arm joints are turned, with the camera as
configured, until the drawn robot covers the dark robot of that picture
best. A small pull keeps a joint at its reading where the outline says
nothing about it.

Reported per moment: how well the outlines overlap before and after, and by
how much each joint had to be turned (median and the middle 80 %). If the
overlap rises and the turns are the same in all pictures of a moment but
differ between moments, the arm's zero points depend on the pose. If the
overlap does not rise, the two robots differ in shape.

Writes analysis/arm-pose-from-outline.json and .npz (per picture).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from scipy.optimize import least_squares

from fit_camera_from_robot import dark_robot
from release_aim_check import recorded, recording_rig_scene, tip_at
from so101_smolvla.sapien_stack_env import RECORDING_RIG_ABSENT_LINKS
from so101_smolvla.stack_recording_rig import ARM_JOINTS

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
MOMENTS = {0: "start", 1: "25 frames before the gripper closes", 2: "release"}
FAR_PX = 30.0
REAL_POINTS = 200
STEP_DEG = 0.5
# One degree away from the reading counts like this many pixels of miss at one point.
PULL_PER_DEG = 1.0
OUT_OF_VIEW = (-1.0, 1.5, 0.0)


def main() -> None:
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    rng = np.random.default_rng(0)
    env = recording_rig_scene()
    rows = []
    try:
        inner = env._env.unwrapped
        robot_ids = [
            int(link.per_scene_id[0]) for link in inner.agent.robot.links
            if link.name not in RECORDING_RIG_ABSENT_LINKS
        ]
        # No cube may hide a part of the robot from the camera.
        for cube in inner.cubes.values():
            cube.set_pose(Pose.create_from_pq(p=torch.tensor(
                [list(OUT_OF_VIEW)], dtype=torch.float32, device=inner.device)))

        for number, frame, moment, picture in zip(
            real["episode"].tolist(), real["frame"].tolist(), real["moment"].tolist(), real["top"]
        ):
            reading = recorded()["state"][recorded()["episode"] == number][frame].astype(np.float64)
            dark = dark_robot(picture)
            to_dark = np.minimum(
                cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX)
            dark_rows, dark_columns = np.nonzero(dark)
            chosen = rng.choice(len(dark_rows), size=REAL_POINTS, replace=False)
            points = (dark_rows[chosen], dark_columns[chosen])

            def drawn(turn: np.ndarray) -> np.ndarray:
                moved = reading.copy()
                moved[: len(ARM_JOINTS)] += turn
                tip_at(inner, moved)
                ids = inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()
                return np.isin(ids, robot_ids)

            def overlap(turn: np.ndarray) -> float:
                robot = drawn(turn)
                return float((dark & robot).sum() / (dark | robot).sum())

            def residuals(turn: np.ndarray) -> np.ndarray:
                robot = drawn(turn)
                to_robot = np.minimum(
                    cv2.distanceTransform((~robot).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX)
                return np.concatenate([
                    to_robot[points],
                    [np.sqrt(REAL_POINTS) * float(to_dark[robot].mean())],
                    PULL_PER_DEG * turn,
                ])

            def slopes(turn: np.ndarray) -> np.ndarray:
                base = residuals(turn)
                columns = []
                for joint in range(len(turn)):
                    moved = turn.copy()
                    moved[joint] += STEP_DEG
                    columns.append((residuals(moved) - base) / STEP_DEG)
                return np.column_stack(columns)

            start = np.zeros(len(ARM_JOINTS))
            found = least_squares(residuals, start, jac=slopes, loss="soft_l1", f_scale=5.0).x
            rows.append((number, moment, overlap(start), overlap(found), *found, *reading[: len(ARM_JOINTS)]))
            print(f"recording {number}, moment {moment}: overlap {rows[-1][2]:.3f} -> {rows[-1][3]:.3f}, "
                  f"turned by {np.round(found, 1).tolist()} degrees", flush=True)
    finally:
        env.close()

    rows = np.array(rows)
    joints = len(ARM_JOINTS)
    result = {}
    for moment, label in MOMENTS.items():
        chosen = rows[rows[:, 1] == moment]
        result[label] = {
            "pictures": len(chosen),
            "overlap_before_median": round(float(np.median(chosen[:, 2])), 3),
            "overlap_after_median": round(float(np.median(chosen[:, 3])), 3),
            "turned_by_deg_p10_median_p90": {
                name: np.round(np.percentile(chosen[:, 4 + joint], [10, 50, 90]), 1).tolist()
                for joint, name in enumerate(ARM_JOINTS)
            },
            "reading_deg_median": {
                name: round(float(np.median(chosen[:, 4 + joints + joint])), 1)
                for joint, name in enumerate(ARM_JOINTS)
            },
        }
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / "arm-pose-from-outline.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez(
        ANALYSIS / "arm-pose-from-outline.npz",
        episode=rows[:, 0].astype(int), moment=rows[:, 1].astype(int),
        overlap_before=rows[:, 2], overlap_after=rows[:, 3],
        turned_by_deg=rows[:, 4:4 + joints], reading_deg=rows[:, 4 + joints:],
    )


if __name__ == "__main__":
    main()
