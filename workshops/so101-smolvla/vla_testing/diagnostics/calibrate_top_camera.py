"""Top camera, joint zero points and table of the recording robot, from its pictures.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    calibrate_top_camera.py          fit both explanations (about 13 minutes)
    calibrate_top_camera.py check    only measure the scene as it is configured

The stack2-recording-rig scene was first measured on one recording (episode
33) with a factor of 1.202 on the pan number. The recorded arm numbers are
whole encoder counts (one step is 360/4096 degrees on every arm joint, pan
included), so a pan factor cannot be a property of the robot. This script
measures the rig on all 38 recordings whose pictures are on disk and compares
two explanations with the same evidence:

  zero points  pan factor 1; the zero of pan, lift, elbow and wrist flex of
               the recording robot differs from the simulated robot's
               (what configs/sapien_stack_recording_rig.json holds since
               this measurement)
  pan factor   a factor and a zero shift on the pan number, no other shift
               (what the configuration held before)

"scene today" is the top camera, the zero points and the lens of the
configuration as it is when the script runs.

Evidence (nothing is simulated forward, no model is run):
  robot    the drawn robot must cover the dark robot of the real top picture
           and nothing else, at three moments of each recording
           (analysis/top-frames-for-calibration.npz)
  cubes    in the first frame the drawn cubes must have the centre and size
           of the real ones; a cube stands where the gripper tip is when the
           human grasps it (red) or lets it go (blue)
  heights  the gripper tip is 10 mm above the table at the grasp and 40 mm
           higher at the release, in all 187 recordings

Unknowns: the top camera's turn and place against the camera of the scene
today, its focal length, and the joint numbers of the explanation. Each
explanation is fitted on the even-numbered recordings from two starting
lenses and checked on the odd ones.

Writes analysis/top-camera-calibration.json and
analysis/top-camera-calibration.png (real pictures with the drawn robot and
cubes outlined: scene today in red, fitted zero points in green).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose
from scipy.optimize import least_squares
from transforms3d.euler import euler2quat

import so101_smolvla.sapien_stack_env as stack_env
from fit_camera_from_robot import DARK_LINKS, dark_robot
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.stack_recording_rig import RECORDING_RIG

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
FAR = 60.0  # px; a miss counts at most this much
REAL_SAMPLES = 600
TIP_ABOVE_TABLE_AT_GRASP = 0.010
STACK_STEP = 0.040  # height of the larger cube
SIZE_WEIGHT = 2.0
HEIGHT_WEIGHT = 0.5  # per millimetre; there are many more heights than pictures
SECOND_START_FOCAL = 450.0
# camera: degrees x3, mm x3, px
CAMERA_STEPS = [0.2, 0.2, 0.2, 2.0, 2.0, 2.0, 4.0]
CAMERA_SCALE = [1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 50.0]
# The factor on the pan number that the configuration held before this measurement.
SUPERSEDED_PAN_FACTOR = 1.202
# name -> (names, start, steps, scale) of the joint numbers
EXPLANATIONS = {
    "zero points": (
        ("pan_zero_deg", "lift_zero_deg", "elbow_zero_deg", "wrist_flex_zero_deg"),
        [0.0, 0.0, 0.0, 0.0], [0.3] * 4, [2.0] * 4,
    ),
    "pan factor": (
        ("pan_factor", "pan_zero_deg"),
        [SUPERSEDED_PAN_FACTOR, 0.0], [0.005, 0.3], [0.05, 2.0],
    ),
}
SHEET_PICTURES = 6


def main() -> None:
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    numbers, moments = real["episode"], real["moment"]
    count = len(numbers)
    places = np.load(ANALYSIS / "cube-placement.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    rng = np.random.default_rng(0)
    to_dark, dark_points = [], []
    for picture in real["top"]:
        mask = dark_robot(picture)
        to_dark.append(np.minimum(cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3), FAR))
        ys, xs = np.nonzero(mask)
        chosen = rng.choice(len(ys), size=min(REAL_SAMPLES, len(ys)), replace=False)
        dark_points.append((ys[chosen], xs[chosen]))

    # Readings at the grasp and at the release of every recording, and at each picture.
    recordings = places["training_episode"].tolist()
    grasp_readings, release_readings = [], []
    for number, close, release in zip(
        recordings, places["training_close_frame"].tolist(), places["training_release_frame"].tolist()
    ):
        state = data["state"][data["episode"] == number].astype(np.float64)
        grasp_readings.append(state[min(int(close) + 10, int(release) - 1)])
        release_readings.append(state[int(release)])
    grasp_readings, release_readings = np.array(grasp_readings), np.array(release_readings)
    row_of = {number: index for index, number in enumerate(recordings)}
    picture_rows = np.array([row_of[number] for number in numbers.tolist()])
    picture_readings = np.array([
        data["state"][data["episode"] == number][frame].astype(np.float64)
        for number, frame in zip(numbers.tolist(), real["frame"].tolist())
    ])
    recording_even = np.array(recordings) % 2 == 0

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        links = inner.agent.robot.links_map
        robot_ids = np.array([int(links[name].per_scene_id[0]) for name in DARK_LINKS])
        cube_ids = {color: int(inner.cubes[color].per_scene_id[0]) for color in ("red", "blue")}
        camera = inner._sensors["top"].camera
        mount = links["top_camera_optical_frame"].pose
        today = camera.get_global_pose().sp
        focal_today = float(camera.get_intrinsic_matrix()[0, 1, 1])
        table_base = inner.table_pose.p.clone()
        table_base[..., 2] -= inner.TABLE_SURFACE_Z

        def simulated(readings: np.ndarray, name: str, joints: np.ndarray) -> np.ndarray:
            moved = readings.copy()
            if name == "zero points":
                moved[:, :4] += joints
            else:
                moved[:, 0] = joints[0] * moved[:, 0] + joints[1]
            return moved

        def pose_robot(reading: np.ndarray) -> np.ndarray:
            qpos = stack_env._real_state_to_sim_qpos(reading)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0).astype(np.float64)

        def stand(color: str, xy: np.ndarray, table: float) -> None:
            inner.cubes[color].set_pose(Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), table + inner.cube_sizes[color] / 2]],
                dtype=torch.float32, device=inner.device)))

        def scene(x: np.ndarray, name: str):
            """Place camera and table for x; return grasp tips, release tips, table height."""
            joints = x[7:]
            grasp = np.array([pose_robot(row) for row in simulated(grasp_readings, name, joints)])
            release = np.array([pose_robot(row) for row in simulated(release_readings, name, joints)])
            table = float(np.median(grasp[:, 2])) - TIP_ABOVE_TABLE_AT_GRASP
            lowered = table_base.clone()
            lowered[..., 2] += table
            inner.table_scene.table.set_pose(Pose.create_from_pq(p=lowered, q=inner.table_pose.q))
            world = sapien.Pose(
                p=(np.asarray(today.p) + x[3:6] / 1000.0).tolist(), q=today.q,
            ) * sapien.Pose(q=euler2quat(*np.deg2rad(x[:3])))
            camera.set_local_pose(
                (mount.inv() * Pose.create_from_pq(p=list(world.p), q=list(world.q))).sp)
            camera.set_focal_lengths(float(x[6]), float(x[6]))
            return grasp, release, table

        def draw(x: np.ndarray, name: str, index: int, grasp, release, table):
            """Masks of the drawn robot and the two drawn cubes for one picture."""
            row = picture_rows[index]
            stand("red", grasp[row, :2], table)
            stand("blue", release[row, :2], table)
            pose_robot(simulated(picture_readings[index][None], name, x[7:])[0])
            ids = inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()
            return np.isin(ids, robot_ids), ids == cube_ids["red"], ids == cube_ids["blue"]

        def residuals(x: np.ndarray, name: str, even: bool | None) -> np.ndarray:
            """All misses for the recordings chosen: even, odd (False) or all (None)."""
            grasp, release, table = scene(x, name)
            chosen = np.ones(len(recordings), dtype=bool) if even is None else recording_even == even
            heights = np.concatenate([
                grasp[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP,
                release[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP - STACK_STEP,
            ])
            rows = []
            for index in np.flatnonzero(chosen[picture_rows]):
                robot, red, blue = draw(x, name, index, grasp, release, table)
                if robot.any():
                    to_robot = np.minimum(
                        cv2.distanceTransform((~robot).astype(np.uint8), cv2.DIST_L2, 3), FAR)
                    row = [float(to_dark[index][robot].mean()), float(to_robot[dark_points[index]].mean())]
                else:
                    row = [FAR, FAR]
                for color, mask in (("red", red), ("blue", blue)):
                    centre, side = real[f"{color}_centre_px"][index], real[f"{color}_side_px"][index]
                    if not np.isfinite(side):
                        row += [0.0, 0.0, 0.0]  # this picture does not show the cube whole
                    elif mask.sum() < 30:
                        row += [FAR, FAR, FAR]
                    else:
                        ys, xs = np.nonzero(mask)
                        row += [xs.mean() - centre[0], ys.mean() - centre[1],
                                SIZE_WEIGHT * (np.sqrt(mask.sum()) - side)]
                rows.append(row)
            return np.concatenate([np.array(rows).ravel(), HEIGHT_WEIGHT * 1000.0 * heights])

        def describe(x: np.ndarray, name: str, even: bool | None) -> dict:
            grasp, release, table = scene(x, name)
            chosen = np.ones(len(recordings), dtype=bool) if even is None else recording_even == even
            pictures = int(chosen[picture_rows].sum())
            values = residuals(x, name, even)
            rows = values[: 8 * pictures].reshape(pictures, 8)
            heights = values[8 * pictures:] / HEIGHT_WEIGHT
            seen = {
                color: np.isfinite(real[f"{color}_side_px"][np.flatnonzero(chosen[picture_rows])])
                for color in ("red", "blue")
            }
            reach = np.linalg.norm(grasp[chosen, :2], axis=1)
            return {
                "pictures": pictures,
                "drawn_robot_off_real_robot_px_mean": round(float(rows[:, 0].mean()), 2),
                "real_robot_off_drawn_robot_px_mean": round(float(rows[:, 1].mean()), 2),
                "red_cube_centre_miss_px_median": round(float(np.median(
                    np.linalg.norm(rows[seen["red"], 2:4], axis=1))), 1),
                "blue_cube_centre_miss_px_median": round(float(np.median(
                    np.linalg.norm(rows[seen["blue"], 5:7], axis=1))), 1),
                "red_cube_size_drawn_minus_real_px_median": round(float(np.median(
                    rows[seen["red"], 4])) / SIZE_WEIGHT, 1),
                "blue_cube_size_drawn_minus_real_px_median": round(float(np.median(
                    rows[seen["blue"], 7])) / SIZE_WEIGHT, 1),
                "height_error_mm_median_abs": round(float(np.median(np.abs(heights))), 1),
                # reach in metres, height in millimetres: millimetres per metre, times 0.1
                "grasp_height_mm_per_100mm_reach": round(0.1 * float(np.polyfit(
                    reach, 1000 * (grasp[chosen, 2] - table), 1)[0]), 1),
                "table_below_robot_base_mm": round(-1000 * table, 1),
            }

        def slopes(x: np.ndarray, name: str, even: bool | None, steps: np.ndarray) -> np.ndarray:
            base = residuals(x, name, even)
            columns = []
            for index, step in enumerate(steps):
                moved = x.copy()
                moved[index] += step
                columns.append((residuals(moved, name, even) - base) / step)
            return np.column_stack(columns)

        def fit(start: np.ndarray, name: str, even: bool | None):
            _, _, steps, scale = EXPLANATIONS[name]
            return least_squares(
                residuals, start, args=(name, even),
                jac=lambda x, *_: slopes(x, name, even, np.array(CAMERA_STEPS + steps)),
                loss="soft_l1", f_scale=5.0, x_scale=np.array(CAMERA_SCALE + scale))

        def starts(name: str) -> list[np.ndarray]:
            """Scene camera of today, and the same view from nearer with a wider lens."""
            joints = EXPLANATIONS[name][1]
            forward = today.to_transformation_matrix()[:3, 0]
            # The camera looks along its x axis; keep the point it looks at on the table.
            distance = -today.p[2] / forward[2]
            nearer = (1.0 - SECOND_START_FOCAL / focal_today) * distance * forward
            return [
                np.array([0.0] * 6 + [focal_today] + joints),
                np.array([0.0, 0.0, 0.0, *(1000.0 * nearer), SECOND_START_FOCAL] + joints),
            ]

        def values(x: np.ndarray, name: str) -> dict:
            world = sapien.Pose(
                p=(np.asarray(today.p) + x[3:6] / 1000.0).tolist(), q=today.q,
            ) * sapien.Pose(q=euler2quat(*np.deg2rad(x[:3])))
            return {
                **dict(zip(EXPLANATIONS[name][0], np.round(x[7:], 3).tolist())),
                "top_camera_position_in_robot_base_m": [float(v) for v in world.p],
                "top_camera_orientation_in_robot_base_wxyz": [float(v) for v in world.q],
                "top_camera_vertical_field_of_view_deg": float(
                    np.degrees(2 * np.arctan(240.0 / x[6]))),
            }

        result: dict[str, dict] = {"recordings_with_pictures": int(len(set(numbers.tolist())))}
        today_x = np.array(
            [0.0] * 6 + [focal_today] + np.rad2deg(RECORDING_RIG.arm_zero_points_rad[:4]).tolist())
        result["scene today"] = {
            **values(today_x, "zero points"),
            "checked_on_odd": describe(today_x, "zero points", False),
            "all": describe(today_x, "zero points", None),
        }
        print("scene today", json.dumps(result["scene today"], indent=1), flush=True)
        if sys.argv[1:] == ["check"]:
            return
        fitted = {}
        for name in EXPLANATIONS:
            found = [fit(start, name, True) for start in starts(name)]
            for start, solution in zip(starts(name), found):
                print(f"{name}: start focal {start[6]:.0f} px -> cost {solution.cost:.0f}, "
                      f"focal {solution.x[6]:.0f} px", flush=True)
            best = min(found, key=lambda solution: solution.cost).x
            fitted[name] = best
            result[name] = {
                "fitted_on_even": values(best, name),
                "checked_on_odd": describe(best, name, False),
            }
            print(name, json.dumps(result[name], indent=1), flush=True)
        final = fit(fitted["zero points"], "zero points", None).x
        result["zero points"]["fitted_on_all"] = {
            **values(final, "zero points"), "all": describe(final, "zero points", None)}
        print("zero points, all", json.dumps(result["zero points"]["fitted_on_all"], indent=1), flush=True)
        (ANALYSIS / "top-camera-calibration.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8")

        tiles = []
        for index in np.linspace(0, count - 1, SHEET_PICTURES).astype(int):
            tile = real["top"][index].copy()
            for x, name, colour in (
                (today_x, "zero points", (255, 0, 0)), (final, "zero points", (0, 255, 0))
            ):
                for mask in draw(x, name, index, *scene(x, name)):
                    outlines, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(tile, outlines, -1, colour, 1)
            tiles.append(tile)
        sheet = np.concatenate(
            [np.concatenate(tiles[:3], axis=1), np.concatenate(tiles[3:], axis=1)], axis=0)
        cv2.imwrite(
            str(ANALYSIS / "top-camera-calibration.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    finally:
        env.close()


if __name__ == "__main__":
    main()
