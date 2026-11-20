"""Joint zero points of the recording robot, its table and its top camera, together.

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    calibrate_recording_robot.py

No model, nothing simulated forward. The recorded arm readings are whole
encoder counts in degrees (one step is 360/4096 degrees on every arm joint),
so one recorded pan degree is one degree of turning; what an individual robot
does not share with the simulated one is where each joint's zero lies
(/tmp/Xbotics-SO101-Sim/so101_sim/robots/so101_base/so101.py:77-81). This
script looks for those zero points with the pan factor held at 1.

Facts used, all from the real stacking recordings:
  heights  the gripper tip is at one height when the human takes the 20 mm
           cube from the table and 40 mm higher when the human lets it go on
           the 40 mm cube (187 recordings, grip_events.grips)
  cubes    a camera must draw a 20 mm cube standing at the grasp place and a
           40 mm cube at the release place onto the two coloured blobs of the
           first top picture: same centre, same size (recordings on disk)

Explanations compared, each fitted on the even-numbered recordings and
checked on the odd ones:
  as recorded            no zero-point shift (camera and table only)
  lift                   lift zero point
  lift + elbow           lift and elbow zero points
  lift + elbow + wrist   lift, elbow and wrist-flex zero points
  pan factor             no zero-point shift, but a factor on the pan number
                         (the explanation now in sapien_stack_recording_rig.json)
Writes analysis/recording-robot-calibration.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

import so101_smolvla.sapien_stack_env as stack_env
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
OBJECTS = ANALYSIS / "objects"
PICTURES = "stacking training set (real recordings)"
CENTRE = np.array([320.0, 240.0])
SIZES = {"red": 0.020, "blue": 0.040}
YAWS = np.deg2rad([0.0, 30.0, 60.0])  # a cube looks the same every 90 degrees
SIZE_WEIGHT = 2.0
HEIGHT_WEIGHT = 1000.0  # one millimetre of height counts like one pixel
# Height of the gripper tip above the table while it holds the 20 mm cube.
TIP_ABOVE_TABLE_AT_GRASP = 0.010
FOCAL_STARTS = (600.0, 800.0)
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
JOINT_NAMES = {1: "lift", 2: "elbow", 3: "wrist_flex"}
# name -> (shifted joints, pan factor free)
EXPLANATIONS = {
    "as recorded": ((), False),
    "lift": ((1,), False),
    "lift + elbow": ((1, 2), False),
    "lift + elbow + wrist": ((1, 2, 3), False),
    "pan factor": ((), True),
}


def blob(centre: np.ndarray, size: float, camera: np.ndarray, focal: float) -> tuple[np.ndarray, float]:
    """Centre and side (square root of the area) of the drawn outline of a cube."""
    rotation, _ = cv2.Rodrigues(camera[:3])
    centres, areas = [], []
    for yaw in YAWS:
        turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        corners = centre + 0.5 * size * CORNERS @ turn.T
        seen = corners @ rotation.T + camera[3:6]
        pixels = focal * seen[:, :2] / np.maximum(seen[:, 2:3], 1e-3) + CENTRE
        # Outline of the eight corners, in full precision so that small
        # changes of the camera change the result smoothly.
        ring = pixels[ConvexHull(pixels).vertices]
        following = np.roll(ring, -1, axis=0)
        cross = ring[:, 0] * following[:, 1] - following[:, 0] * ring[:, 1]
        area = 0.5 * cross.sum()
        centres.append(((ring + following) * cross[:, None]).sum(axis=0) / (6.0 * area))
        areas.append(abs(area))
    return np.mean(centres, axis=0), float(np.sqrt(np.mean(areas)))


def slopes(function, x: np.ndarray, steps: np.ndarray, *arguments) -> np.ndarray:
    """How each residual changes per unit of each unknown, by stepping each unknown.

    The steps are fixed sizes large enough for the robot model's single
    precision: the default of the solver is relative to the value and far too
    small for a zero-point shift that starts at zero.
    """
    base = function(x, *arguments)
    columns = []
    for index, step in enumerate(steps):
        moved = x.copy()
        moved[index] += step
        columns.append((function(moved, *arguments) - base) / step)
    return np.column_stack(columns)


def main() -> None:
    per_picture = json.loads((OBJECTS / "object-measurements-per-picture.json").read_text())
    stored = np.load(OBJECTS / "real-first-frames.npz")["episode"].tolist()
    seen_red = dict(zip(stored, per_picture[PICTURES]["top camera, red cube"]))
    seen_blue = dict(zip(stored, per_picture[PICTURES]["top camera, blue cube"]))
    places = np.load(ANALYSIS / "cube-placement.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state = data["episode"], data["state"]
    rig = json.loads((ANALYSIS / "stack-lens-fit.json").read_text())["free lens + pan, outline + cubes"]
    camera_start = np.array(rig["world_to_camera_rotation_vector"] + rig["world_to_camera_translation"])

    # One row per grasp and per release: the reading, the recording, and the
    # blob of the cube that stood there if the first picture shows it whole.
    readings, numbers, is_release, blobs = [], [], [], []
    for number, close, release in zip(
        places["training_episode"].tolist(),
        places["training_close_frame"].tolist(),
        places["training_release_frame"].tolist(),
    ):
        chosen = np.flatnonzero(episode == number)
        for released, frame, found in (
            (False, min(int(close) + 10, int(release) - 1), seen_red.get(number)),
            (True, int(release), seen_blue.get(number)),
        ):
            readings.append(state[chosen[frame]].astype(np.float64))
            numbers.append(int(number))
            is_release.append(released)
            blobs.append(None if found is None or found["touches_edge"] else found)
    readings, numbers, is_release = np.array(readings), np.array(numbers), np.array(is_release)
    has_blob = np.array([found is not None for found in blobs])
    seen_centre = np.array([found["centre_px"] for found in blobs if found is not None])
    seen_side = np.array([found["side_px"] for found in blobs if found is not None])
    size = np.where(is_release, SIZES["blue"], SIZES["red"])
    wanted_height = np.where(is_release, SIZES["blue"], 0.0)
    pan_deg = readings[:, 0]

    env = So101SimEnv(**environment_arguments("stack2-training"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def tip(reading: np.ndarray) -> np.ndarray:
            qpos = stack_env._real_state_to_sim_qpos(reading)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0).astype(np.float64)

        # The pan joint turns the arm about a vertical line; find that line
        # and the turning direction from three poses of one reading.
        probe, at = readings[0].copy(), {}
        for angle in (0.0, 90.0, 180.0):
            probe[0] = angle
            at[angle] = tip(probe)[:2]
        axis = 0.5 * (at[0.0] + at[180.0])
        arm, quarter = at[0.0] - axis, at[90.0] - axis
        direction = float(np.sign(arm[0] * quarter[1] - arm[1] * quarter[0]))

        cache: dict[tuple, np.ndarray] = {}

        def unturned_tips(joints: tuple[int, ...], shift: np.ndarray) -> np.ndarray:
            key = (joints, tuple(np.round(shift, 9)))
            if key not in cache:
                if len(cache) > 64:
                    cache.clear()
                moved = readings.copy()
                moved[:, 0] = 0.0
                for joint, value in zip(joints, shift):
                    moved[:, joint] += value
                cache[key] = np.array([tip(row) for row in moved])
            return cache[key]

        def scene(x: np.ndarray, joints: tuple[int, ...], free_pan: bool):
            """Tip places of all grasps and releases, and the drawn cube blobs."""
            count = len(joints)
            shift, grasp_height = x[7:7 + count], x[7 + count]
            factor = float(x[8 + count]) if free_pan else 1.0
            flat = unturned_tips(joints, shift)
            angle = direction * np.deg2rad(factor * pan_deg)
            offset = flat[:, :2] - axis
            turned = np.column_stack([
                np.cos(angle) * offset[:, 0] - np.sin(angle) * offset[:, 1],
                np.sin(angle) * offset[:, 0] + np.cos(angle) * offset[:, 1],
            ]) + axis
            table = grasp_height - TIP_ABOVE_TABLE_AT_GRASP
            drawn = [
                blob(np.array([*place, table + 0.5 * side]), side, x[:6], float(x[6]))
                for place, side in zip(turned[has_blob], size[has_blob])
            ]
            return (
                flat[:, 2] - grasp_height - wanted_height,
                np.array([d[0] for d in drawn]) - seen_centre,
                np.array([d[1] for d in drawn]) - seen_side,
                turned,
            )

        def residuals(x, joints, free_pan, chosen):
            height, centre, side, _ = scene(x, joints, free_pan)
            with_blob = chosen[has_blob]
            return np.concatenate([
                HEIGHT_WEIGHT * height[chosen], centre[with_blob].ravel(),
                SIZE_WEIGHT * side[with_blob],
            ])

        def describe(x, joints, free_pan, chosen) -> dict:
            height, centre, side, turned = scene(x, joints, free_pan)
            with_blob = chosen[has_blob]
            miss = np.linalg.norm(centre, axis=1)[with_blob]
            grasp = chosen & ~is_release
            reach = np.linalg.norm(turned[:, :2] - axis, axis=1)
            return {
                "grasps_and_releases": int(chosen.sum()),
                "cubes_seen": int(with_blob.sum()),
                "height_error_mm_median_abs": round(1000 * float(np.median(np.abs(height[chosen]))), 1),
                "height_error_mm_rms": round(1000 * float(np.sqrt(np.mean(height[chosen] ** 2))), 1),
                "grasp_height_mm_per_100mm_reach": round(
                    100 * float(np.polyfit(reach[grasp], height[grasp], 1)[0]), 1),
                "cube_centre_miss_px_median": round(float(np.median(miss)), 1),
                "cube_centre_miss_px_p90": round(float(np.percentile(miss, 90)), 1),
                "drawn_cube_size_over_seen_median": round(
                    float(np.median((side[with_blob] + seen_side[with_blob]) / seen_side[with_blob])), 3),
                "tip_reach_from_pan_axis_at_grasp_mm_median": round(1000 * float(np.median(reach[grasp])), 1),
            }

        even, odd = numbers % 2 == 0, numbers % 2 == 1
        everything = np.ones(len(numbers), dtype=bool)
        result = {
            "grasps_and_releases": int(len(numbers)),
            "cubes_seen_in_first_top_pictures": int(has_blob.sum()),
            "pan_axis_in_robot_base_m": np.round(axis, 4).tolist(),
        }
        for name, (joints, free_pan) in EXPLANATIONS.items():
            fitted = {}
            for label, chosen in (("all", everything), ("even", even)):
                fits = []
                for focal in FOCAL_STARTS:
                    guess = camera_start.copy()
                    guess[3:6] *= focal / float(rig["focal_length_px"])
                    x0 = [*guess, focal, *([0.0] * len(joints)), -0.010] + ([1.0] if free_pan else [])
                    scale = (
                        [0.05] * 3 + [0.02] * 3 + [20.0] + [1.0] * len(joints) + [0.005]
                        + ([0.05] if free_pan else [])
                    )
                    # radians, metres, pixels, degrees, metres, pan factor
                    steps = np.array(
                        [1e-4] * 3 + [1e-4] * 3 + [0.1] + [0.05] * len(joints) + [1e-4]
                        + ([1e-3] if free_pan else [])
                    )
                    fits.append(least_squares(
                        residuals, np.array(x0), args=(joints, free_pan, chosen),
                        jac=lambda x, *arguments: slopes(residuals, x, steps, *arguments),
                        loss="soft_l1", f_scale=5.0, x_scale=scale))
                fitted[label] = min(fits, key=lambda found: found.cost).x
            x, count = fitted["all"], len(joints)
            rotation, _ = cv2.Rodrigues(x[:3])
            eye = -rotation.T @ x[3:6]

            def values(solution: np.ndarray) -> dict:
                return {
                    "zero_point_shift_deg": {
                        JOINT_NAMES[j]: round(float(v), 2) for j, v in zip(joints, solution[7:7 + count])},
                    "pan_factor": round(float(solution[8 + count]), 3) if free_pan else 1.0,
                    "table_below_robot_base_mm": round(
                        1000 * float(TIP_ABOVE_TABLE_AT_GRASP - solution[7 + count]), 1),
                    "vertical_field_of_view_deg": round(
                        float(np.degrees(2 * np.arctan(240 / solution[6]))), 1),
                }

            row = {
                "fitted_on_all": {**values(x), **describe(x, joints, free_pan, everything)},
                "fitted_on_even": values(fitted["even"]),
                "even_fit_checked_on_odd": describe(fitted["even"], joints, free_pan, odd),
                "camera_position_m": np.round(eye, 3).tolist(),
                "camera_tilt_from_straight_down_deg": round(
                    float(np.degrees(np.arccos(-rotation.T[2, 2]))), 1),
                "world_to_camera_rotation_vector": x[:3].tolist(),
                "world_to_camera_translation": x[3:6].tolist(),
                "focal_length_px": float(x[6]),
            }
            result[name] = row
            print(name, json.dumps(row, indent=1), flush=True)
    finally:
        env.close()
    (ANALYSIS / "recording-robot-calibration.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
