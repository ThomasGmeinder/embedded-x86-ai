"""Where did the wrist camera of the recording robot sit on the hand, and with which lens?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    calibrate_wrist_camera.py

The simulator's wrist camera was measured on the simulator authors' robot
(/tmp/Xbotics-SO101-Sim/so101_sim/envs.py:58-60, 74). The real stacking
recordings come from another robot. wrist_cube_check.py shows that at the
recorded readings the simulated hand and the red cube both appear displaced
in the wrist picture, by the same amount whatever is assumed about the arm.

Facts used (analysis/wrist-frames-before-grasp.npz, 152 real wrist pictures
of 38 recordings shortly before the human closes the gripper):
  hand   the two black jaws are rigidly attached to the camera (one of them
         moves with the recorded gripper reading); the drawn hand must cover
         the dark pixels of the real picture and nothing else
  cube   the drawn red cube must have the centre and size of the real one

Unknowns: the camera's turn (roll about its viewing direction, pitch, yaw)
and shift (forward, left, up) against the simulator's wrist camera, and its
focal length. Fitted on the even-numbered recordings and checked on the odd
ones, then fitted on all. Nothing is simulated forward and no model is run.

Writes analysis/wrist-camera-calibration.json and
analysis/wrist-camera-calibration.png (real pictures with the drawn hand and
cube outlined: before in red, after in green).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose
from scipy.optimize import least_squares
from transforms3d.euler import euler2quat

import so101_smolvla.sapien_stack_env as stack_env
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
ARM = "lift + elbow"  # explanation of the arm used to pose it; the wrist picture hardly depends on it
JOINTS = {"lift": 1, "elbow": 2, "wrist_flex": 3}
DARK_BELOW = 70  # brightest colour channel of a black jaw in the real pictures
FAR = 60.0  # px; a miss counts at most this much
REAL_SAMPLES = 400
# degrees, degrees, degrees, mm, mm, mm, px
STEPS = np.array([0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 5.0])
SCALE = np.array([2.0, 2.0, 2.0, 5.0, 5.0, 5.0, 30.0])
NAMES = ("roll_deg", "pitch_deg", "yaw_deg", "forward_mm", "left_mm", "up_mm", "focal_length_px")
SHEET_PICTURES = 6


def slopes(function, x: np.ndarray, *arguments) -> np.ndarray:
    """How each residual changes per unit of each unknown, by fixed steps.

    Pictures are made of whole pixels, so the steps must be large enough to
    move the drawn hand by a good part of a pixel.
    """
    base = function(x, *arguments)
    columns = []
    for index, step in enumerate(STEPS):
        moved = x.copy()
        moved[index] += step
        columns.append((function(moved, *arguments) - base) / step)
    return np.column_stack(columns)


def main() -> None:
    arm = json.loads((ANALYSIS / "recording-robot-calibration.json").read_text())[ARM]["fitted_on_all"]
    shift = np.zeros(6)
    for joint, degrees in arm["zero_point_shift_deg"].items():
        shift[JOINTS[joint]] = degrees
    drop = arm["table_below_robot_base_mm"] / 1000.0

    real = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    places = np.load(ANALYSIS / "cube-placement.npz")
    close_of = dict(zip(places["training_episode"].tolist(), places["training_close_frame"].tolist()))
    release_of = dict(zip(places["training_episode"].tolist(), places["training_release_frame"].tolist()))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    numbers = real["episode"]
    count = len(numbers)
    rng = np.random.default_rng(0)
    dark, to_dark, dark_points = [], [], []
    for picture in real["wrist"]:
        mask = picture.max(axis=2) < DARK_BELOW
        dark.append(mask)
        to_dark.append(np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), FAR))
        ys, xs = np.nonzero(mask)
        chosen = rng.choice(len(ys), size=min(REAL_SAMPLES, len(ys)), replace=False)
        dark_points.append((ys[chosen], xs[chosen]))

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        robot_ids = np.array([int(link.per_scene_id[0]) for link in inner.agent.robot.links])
        red_id = int(inner.cubes["red"].per_scene_id[0])
        camera = inner._sensors["wrist"].camera
        mounted = camera.get_local_pose().sp
        focal_start = float(camera.get_intrinsic_matrix()[0, 1, 1])
        table = inner.table_pose.p.clone()
        table[..., 2] += -inner.TABLE_SURFACE_Z - drop
        inner.table_scene.table.set_pose(Pose.create_from_pq(p=table, q=inner.table_pose.q))

        def qpos_of(reading: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                stack_env._real_state_to_sim_qpos(reading + shift)[None],
                dtype=torch.float32, device=inner.device)

        def cube_pose(color: str, reading: np.ndarray) -> Pose:
            inner.agent.robot.set_qpos(qpos_of(reading))
            xy = np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)[:2]
            return Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), inner.cube_sizes[color] / 2 - drop]],
                dtype=torch.float32, device=inner.device))

        # Per picture: the arm pose and where the two cubes stood.
        scenes = []
        for number, frame in zip(numbers.tolist(), real["frame"].tolist()):
            state = data["state"][data["episode"] == number].astype(np.float64)
            close, release = int(close_of[number]), int(release_of[number])
            scenes.append((
                qpos_of(state[frame]),
                cube_pose("red", state[min(close + 10, release - 1)]),
                cube_pose("blue", state[release]),
            ))

        def draw(x: np.ndarray, index: int) -> tuple[np.ndarray, np.ndarray]:
            """Masks of the drawn hand and the drawn red cube for one picture."""
            qpos, red, blue = scenes[index]
            inner.agent.robot.set_qpos(qpos)
            inner.cubes["red"].set_pose(red)
            inner.cubes["blue"].set_pose(blue)
            turn = euler2quat(*np.deg2rad(x[:3]))
            camera.set_local_pose(mounted * sapien.Pose(p=(x[3:6] / 1000.0).tolist(), q=turn))
            camera.set_focal_lengths(float(x[6]), float(x[6]))
            ids = inner.get_obs()["sensor_data"]["wrist"]["segmentation"][0, ..., 0].cpu().numpy()
            return np.isin(ids, robot_ids), ids == red_id

        def residuals(x: np.ndarray, chosen: np.ndarray) -> np.ndarray:
            rows = []
            for index in np.flatnonzero(chosen):
                hand, cube = draw(x, index)
                if hand.any():
                    to_hand = np.minimum(
                        cv2.distanceTransform((~hand).astype(np.uint8), cv2.DIST_L2, 3), FAR)
                    drawn_off_real = float(to_dark[index][hand].mean())
                    real_off_drawn = float(to_hand[dark_points[index]].mean())
                else:
                    drawn_off_real = real_off_drawn = FAR
                if cube.sum() >= 50:
                    ys, xs = np.nonzero(cube)
                    centre = np.array([xs.mean(), ys.mean()]) - real["red_centre_px"][index]
                    side = float(np.sqrt(cube.sum())) - float(real["red_side_px"][index])
                else:
                    centre, side = np.array([FAR, FAR]), FAR
                rows.append([drawn_off_real, real_off_drawn, centre[0], centre[1], side])
            return np.array(rows).ravel()

        def describe(x: np.ndarray, chosen: np.ndarray) -> dict:
            rows = residuals(x, chosen).reshape(-1, 5)
            return {
                "pictures": int(chosen.sum()),
                "drawn_hand_off_real_hand_px_mean": round(float(rows[:, 0].mean()), 2),
                "real_hand_off_drawn_hand_px_mean": round(float(rows[:, 1].mean()), 2),
                "cube_centre_drawn_minus_real_px_median": np.round(np.median(rows[:, 2:4], axis=0), 1).tolist(),
                "cube_centre_distance_px_median": round(float(np.median(np.linalg.norm(rows[:, 2:4], axis=1))), 1),
                "cube_size_drawn_minus_real_px_median": round(float(np.median(rows[:, 4])), 1),
            }

        def fit(chosen: np.ndarray) -> np.ndarray:
            return least_squares(
                residuals, start, args=(chosen,),
                jac=lambda x, pictures: slopes(residuals, x, pictures),
                loss="soft_l1", f_scale=5.0, x_scale=SCALE).x

        start = np.array([0.0] * 6 + [focal_start])
        even, odd = numbers % 2 == 0, numbers % 2 == 1
        everything = np.ones(count, dtype=bool)
        from_even = fit(even)
        print("fitted on even:", dict(zip(NAMES, np.round(from_even, 2).tolist())), flush=True)
        from_all = fit(everything)
        print("fitted on all:", dict(zip(NAMES, np.round(from_all, 2).tolist())), flush=True)

        local = mounted * sapien.Pose(
            p=(from_all[3:6] / 1000.0).tolist(), q=euler2quat(*np.deg2rad(from_all[:3])))
        result = {
            "arm_explanation_used": ARM,
            "pictures": count,
            "simulator_camera": {
                "focal_length_px": focal_start,
                "checked_on_odd": describe(start, odd),
                "all": describe(start, everything),
            },
            "fitted_on_even": {
                **dict(zip(NAMES, np.round(from_even, 3).tolist())),
                "checked_on_odd": describe(from_even, odd),
            },
            "fitted_on_all": {
                **dict(zip(NAMES, np.round(from_all, 3).tolist())),
                "all": describe(from_all, everything),
                "vertical_field_of_view_deg": round(
                    float(np.degrees(2 * np.arctan(240.0 / from_all[6]))), 2),
                "pose_on_wrist_camera_optical_frame": {
                    "position_m": [float(v) for v in local.p],
                    "orientation_wxyz": [float(v) for v in local.q],
                },
            },
        }
        print(json.dumps(result, indent=1), flush=True)
        (ANALYSIS / "wrist-camera-calibration.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8")

        tiles = []
        for index in np.linspace(0, count - 1, SHEET_PICTURES).astype(int):
            tile = real["wrist"][index].copy()
            for x, colour in ((start, (255, 0, 0)), (from_all, (0, 255, 0))):
                for mask in draw(x, index):
                    outlines, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(tile, outlines, -1, colour, 2)
            tiles.append(cv2.resize(tile, (480, 360)))
        sheet = np.concatenate(
            [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
        cv2.imwrite(
            str(ANALYSIS / "wrist-camera-calibration.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    finally:
        env.close()


if __name__ == "__main__":
    main()
