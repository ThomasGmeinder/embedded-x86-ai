"""Top camera and joint zero points of the recording robot, from the robot's outline alone.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    calibrate_top_camera_from_outline.py          fit
    calibrate_top_camera_from_outline.py check    only measure the scene as it is configured

Why a second measurement. calibrate_top_camera.py fitted the camera to three
kinds of evidence at once: the robot's outline, the two cubes of the first
picture and the grasp and release heights. The outline entered as two
averaged numbers per picture and the cubes as six, so the cubes decided where
the camera stands, although their true places are not known: a cube is
assumed to stand where the gripper tip is when the human takes it or lets it
go. With that camera the drawn robot misses the real one where the arm stands
high above the table (robot_outline_by_link.py). The question this script
answers is whether a camera at another distance explains that miss.

Outcome (2026-11-20, analysis/top-camera-from-outline.json): it does not. The
fit moves the camera to 0.48 m above the robot base behind a 55.7 degree
lens, but on the recordings it did not see the overlap of drawn and real
robot is no better (0.706 / 0.767 / 0.773 at start, before the grasp and at
the release, against 0.776 / 0.781 / 0.774 for the scene as configured), the
drawn cubes come out 2.2 to 2.4 pixels too large and the red cube lands 11.2
pixels from the real one instead of 5.2. The configured camera stays; the
miss is a difference in the shape of the two robots
(arm_pose_from_outline.py gives the same answer for the joint zero points).

The robot is the one thing in the pictures whose shape and pose are known.
Here it is the only evidence for the camera: at every picture the drawn robot
must cover the dark robot of the real picture and nothing else. The grasp and
release heights of all recordings stay in, because they tie the lift, elbow
and wrist-flex zero points to the table. The cubes are not fitted; how well
the drawn cubes then land on the real ones is reported as a check, together
with the recordings the fit did not see (fitted on even, checked on odd).

Unknowns: the top camera's turn and place against the camera of the scene as
configured, its focal length, and the zero point of each of the five arm
joints. Nothing is simulated forward, no model is run.

Writes analysis/top-camera-from-outline.json and
analysis/top-camera-from-outline.png (odd recordings, real only red, drawn
only blue: scene as configured above, fitted below).
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
from fit_camera_from_robot import dark_robot
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_stack_env import RECORDING_RIG_ABSENT_LINKS
from so101_smolvla.stack_recording_rig import ARM_JOINTS, RECORDING_RIG

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
MOMENTS = {0: "start", 1: "25 frames before the gripper closes", 2: "release"}
FAR_PX = 30.0  # a miss counts at most this much
REAL_POINTS = 200  # real robot pixels looked at in every picture
TIP_ABOVE_TABLE_AT_GRASP = 0.010
STACK_STEP = 0.040  # height of the larger cube
HEIGHT_WEIGHT = 0.5  # per millimetre
SECOND_START_FOCAL = 440.0
OUT_OF_VIEW = (-1.0, 1.5)
# camera: degrees x3, millimetres x3, pixels; then degrees per joint
STEPS = np.array([0.2, 0.2, 0.2, 2.0, 2.0, 2.0, 4.0] + [0.3] * len(ARM_JOINTS))
SCALE = np.array([1.0, 1.0, 1.0, 20.0, 20.0, 20.0, 50.0] + [2.0] * len(ARM_JOINTS))
SHEET_PICTURES = 6


def main() -> None:
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    numbers, moments = real["episode"], real["moment"]
    places = np.load(ANALYSIS / "cube-placement.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    rng = np.random.default_rng(0)
    darks, to_dark, dark_points = [], [], []
    for picture in real["top"]:
        dark = dark_robot(picture)
        darks.append(dark)
        to_dark.append(np.minimum(
            cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX))
        rows, columns = np.nonzero(dark)
        chosen = rng.choice(len(rows), size=REAL_POINTS, replace=False)
        dark_points.append((rows[chosen], columns[chosen]))

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
        robot_ids = np.array([
            int(link.per_scene_id[0]) for link in inner.agent.robot.links
            if link.name not in RECORDING_RIG_ABSENT_LINKS
        ])
        cube_ids = {color: int(inner.cubes[color].per_scene_id[0]) for color in ("red", "blue")}
        camera = inner._sensors["top"].camera
        mount = links["top_camera_optical_frame"].pose
        today = camera.get_global_pose().sp
        focal_today = float(camera.get_intrinsic_matrix()[0, 1, 1])
        table_base = inner.table_pose.p.clone()
        table_base[..., 2] -= inner.TABLE_SURFACE_Z

        def pose_robot(reading: np.ndarray, zero_points: np.ndarray) -> np.ndarray:
            moved = reading.copy()
            moved[: len(ARM_JOINTS)] += zero_points
            qpos = stack_env._real_state_to_sim_qpos(moved)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0).astype(np.float64)

        def place_camera(x: np.ndarray) -> None:
            world = camera_pose(x)
            camera.set_local_pose(
                (mount.inv() * Pose.create_from_pq(p=list(world.p), q=list(world.q))).sp)
            camera.set_focal_lengths(float(x[6]), float(x[6]))

        def camera_pose(x: np.ndarray) -> sapien.Pose:
            return sapien.Pose(
                p=(np.asarray(today.p) + x[3:6] / 1000.0).tolist(), q=today.q,
            ) * sapien.Pose(q=euler2quat(*np.deg2rad(x[:3])))

        def stand(color: str, xy, table: float) -> None:
            inner.cubes[color].set_pose(Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), table + inner.cube_sizes[color] / 2]],
                dtype=torch.float32, device=inner.device)))

        def cubes_out_of_view() -> None:
            # No cube may hide a part of the robot from the camera.
            for color in inner.cubes:
                stand(color, OUT_OF_VIEW, 0.0)

        tips_of: dict[tuple, tuple] = {}

        def tips(x: np.ndarray):
            """Gripper tip at every grasp and release, and the table height they give."""
            key = tuple(np.round(x[7:], 9))
            if key not in tips_of:
                # Only the zero points move the tips; one entry is all a step of the fit needs.
                tips_of.clear()
                grasp = np.array([pose_robot(row, x[7:]) for row in grasp_readings])
                release = np.array([pose_robot(row, x[7:]) for row in release_readings])
                tips_of[key] = (
                    grasp, release, float(np.median(grasp[:, 2])) - TIP_ABOVE_TABLE_AT_GRASP)
            return tips_of[key]

        def drawn_ids(x: np.ndarray, index: int) -> np.ndarray:
            pose_robot(picture_readings[index], x[7:])
            return inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()

        def outline(x: np.ndarray, index: int) -> tuple[np.ndarray, float, float]:
            """Real points' distance to the drawn robot, drawn pixels' mean distance to the real one, overlap."""
            robot = np.isin(drawn_ids(x, index), robot_ids)
            if not robot.any():
                return np.full(REAL_POINTS, FAR_PX), FAR_PX, 0.0
            to_robot = np.minimum(
                cv2.distanceTransform((~robot).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX)
            dark = darks[index]
            return (
                to_robot[dark_points[index]], float(to_dark[index][robot].mean()),
                float((dark & robot).sum() / (dark | robot).sum()),
            )

        def residuals(x: np.ndarray, even: bool | None) -> np.ndarray:
            """All misses for the recordings chosen: even, odd (False) or all (None)."""
            place_camera(x)
            chosen = np.ones(len(recordings), dtype=bool) if even is None else recording_even == even
            grasp, release, table = tips(x)
            heights = np.concatenate([
                grasp[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP,
                release[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP - STACK_STEP,
            ])
            parts = []
            for index in np.flatnonzero(chosen[picture_rows]):
                real_to_drawn, drawn_to_real, _ = outline(x, index)
                # The drawn robot's miss counts like that of as many points.
                parts += [real_to_drawn, [np.sqrt(REAL_POINTS) * drawn_to_real]]
            return np.concatenate([*parts, HEIGHT_WEIGHT * 1000.0 * heights])

        def describe(x: np.ndarray, even: bool | None) -> dict:
            place_camera(x)
            chosen = np.ones(len(recordings), dtype=bool) if even is None else recording_even == even
            grasp, release, table = tips(x)
            lowered = table_base.clone()
            lowered[..., 2] += table
            inner.table_scene.table.set_pose(Pose.create_from_pq(p=lowered, q=inner.table_pose.q))
            heights = 1000.0 * np.concatenate([
                grasp[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP,
                release[chosen, 2] - table - TIP_ABOVE_TABLE_AT_GRASP - STACK_STEP,
            ])
            rows, cube_miss, cube_size = [], {"red": [], "blue": []}, {"red": [], "blue": []}
            for index in np.flatnonzero(chosen[picture_rows]):
                real_to_drawn, drawn_to_real, overlap = outline(x, index)
                rows.append((moments[index], overlap, real_to_drawn.mean(), drawn_to_real))
                if moments[index] != 0:
                    continue
                # Check only: the cubes of the first picture, standing where
                # the gripper tip takes the red one and lets it go on the blue one.
                row = picture_rows[index]
                stand("red", grasp[row], table)
                stand("blue", release[row], table)
                ids = drawn_ids(x, index)
                cubes_out_of_view()
                for color in ("red", "blue"):
                    side = real[f"{color}_side_px"][index]
                    mask = ids == cube_ids[color]
                    if not np.isfinite(side) or mask.sum() < 30:
                        continue  # the real or the drawn picture does not show the cube whole
                    cube_rows, cube_columns = np.nonzero(mask)
                    cube_miss[color].append(float(np.hypot(
                        cube_columns.mean() - real[f"{color}_centre_px"][index][0],
                        cube_rows.mean() - real[f"{color}_centre_px"][index][1])))
                    cube_size[color].append(float(np.sqrt(mask.sum()) - side))
            rows = np.array(rows)
            world = camera_pose(x)
            return {
                "pictures": len(rows),
                **{
                    label: {
                        "overlap_median": round(float(np.median(rows[rows[:, 0] == moment, 1])), 3),
                        "real_outside_drawn_px_mean": round(float(rows[rows[:, 0] == moment, 2].mean()), 2),
                        "drawn_outside_real_px_mean": round(float(rows[rows[:, 0] == moment, 3].mean()), 2),
                    }
                    for moment, label in MOMENTS.items()
                },
                "check_red_cube_centre_miss_px_median": round(float(np.median(cube_miss["red"])), 1),
                "check_blue_cube_centre_miss_px_median": round(float(np.median(cube_miss["blue"])), 1),
                "check_red_cube_size_drawn_minus_real_px_median": round(float(np.median(cube_size["red"])), 1),
                "check_blue_cube_size_drawn_minus_real_px_median": round(float(np.median(cube_size["blue"])), 1),
                "height_error_mm_median_abs": round(float(np.median(np.abs(heights))), 1),
                "table_below_robot_base_mm": round(-1000 * table, 1),
                "joint_zero_points_deg": dict(zip(ARM_JOINTS, np.round(x[7:], 3).tolist())),
                "top_camera_position_in_robot_base_m": [float(v) for v in world.p],
                "top_camera_orientation_in_robot_base_wxyz": [float(v) for v in world.q],
                "top_camera_vertical_field_of_view_deg": float(np.degrees(2 * np.arctan(240.0 / x[6]))),
            }

        def slopes(x: np.ndarray, even: bool | None) -> np.ndarray:
            base = residuals(x, even)
            columns = []
            for index, step in enumerate(STEPS):
                moved = x.copy()
                moved[index] += step
                columns.append((residuals(moved, even) - base) / step)
            return np.column_stack(columns)

        def fit(start: np.ndarray, even: bool | None):
            return least_squares(
                residuals, start, args=(even,), jac=slopes,
                loss="soft_l1", f_scale=5.0, x_scale=SCALE)

        cubes_out_of_view()
        today_x = np.array(
            [0.0] * 6 + [focal_today] + np.rad2deg(RECORDING_RIG.arm_zero_points_rad).tolist())
        result: dict[str, dict] = {
            "recordings_with_pictures": int(len(set(numbers.tolist()))),
            "scene as configured": {
                "checked_on_odd": describe(today_x, False), "all": describe(today_x, None)},
        }
        print("scene as configured", json.dumps(result["scene as configured"], indent=1), flush=True)
        if sys.argv[1:] == ["check"]:
            return

        # Second start: the same view of the table from nearer, through a wider lens.
        forward = today.to_transformation_matrix()[:3, 0]
        to_table = (inner.TABLE_SURFACE_Z - today.p[2]) / forward[2]
        nearer = (1.0 - SECOND_START_FOCAL / focal_today) * to_table * forward
        starts = [
            today_x,
            np.array([0.0, 0.0, 0.0, *(1000.0 * nearer), SECOND_START_FOCAL, *today_x[7:]]),
        ]
        found = [fit(start, True) for start in starts]
        for start, solution in zip(starts, found):
            print(f"start focal {start[6]:.0f} px -> cost {solution.cost:.0f}, "
                  f"focal {solution.x[6]:.0f} px", flush=True)
        best = min(found, key=lambda solution: solution.cost).x
        result["fitted on even"] = {"checked_on_odd": describe(best, False)}
        print("fitted on even", json.dumps(result["fitted on even"], indent=1), flush=True)
        final = fit(best, None).x
        result["fitted on all"] = {"all": describe(final, None)}
        print("fitted on all", json.dumps(result["fitted on all"], indent=1), flush=True)
        (ANALYSIS / "top-camera-from-outline.json").write_text(
            json.dumps(result, indent=1) + "\n", encoding="utf-8")

        odd = np.flatnonzero(~recording_even[picture_rows] & (moments != 0))
        shown = odd[np.linspace(0, len(odd) - 1, SHEET_PICTURES).astype(int)]
        strips = []
        for x in (today_x, best):
            place_camera(x)
            tiles = []
            for index in shown:
                robot = np.isin(drawn_ids(x, index), robot_ids)
                tile = np.full((*robot.shape, 3), 255, dtype=np.uint8)
                tile[darks[index] & robot] = (150, 150, 150)
                tile[darks[index] & ~robot] = (230, 40, 40)
                tile[robot & ~darks[index]] = (40, 40, 230)
                tiles.append(tile)
            strips.append(np.concatenate(tiles, axis=1))
        sheet = cv2.resize(np.concatenate(strips, axis=0), None, fx=0.5, fy=0.5,
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(ANALYSIS / "top-camera-from-outline.png"),
                    cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    finally:
        env.close()


if __name__ == "__main__":
    main()
