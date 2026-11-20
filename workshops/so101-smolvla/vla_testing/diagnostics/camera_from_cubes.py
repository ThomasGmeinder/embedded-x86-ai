"""Where was the top camera of the stacking recordings, judged by the cubes alone?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    camera_from_cubes.py

No model, nothing simulated forward, no robot outline. Each recording gives
two pairs of facts: where a cube is in the first top picture (centre and size
of its coloured blob), and where the gripper tip was when the human took the
red cube and when the human let it go on the blue cube (recorded joint
readings posed on the simulated robot model). A camera is right when it draws
a 20 mm and a 40 mm cube standing at those two places onto the blobs: same
centre, same size. The predicted blob is the outline of all eight cube
corners, so the visible side faces are counted as they are in the picture.

Several explanations are fitted, each with its own best camera, and each is
fitted on the even-numbered recordings and checked on the odd ones:
  pan factor fixed at 1 (the recorded pan number is the pan angle)
  pan factor fixed at 1.202 (the value in configs/sapien_stack_recording_rig.json)
  pan factor free
each with the lift reading as recorded and with the lift zero point shifted
as analysis/geometry-from-grasps.json found. Writes
analysis/camera-from-cubes.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import least_squares

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
FOCAL_STARTS = (450.0, 600.0, 710.0, 850.0)
CORNERS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
# name -> (pan factor or None when free, lift zero-point shift in degrees, table below robot base in m)
EXPLANATIONS = {
    "pan factor 1, lift as recorded": (1.0, 0.0, 0.023),
    "pan factor 1.202, lift as recorded": (1.202, 0.0, 0.023),
    "pan factor free, lift as recorded": (None, 0.0, 0.023),
    "pan factor 1, lift shifted": (1.0, None, None),
    "pan factor 1.202, lift shifted": (1.202, None, None),
    "pan factor free, lift shifted": (None, None, None),
}


def blob(centre: np.ndarray, size: float, camera: np.ndarray, focal: float) -> tuple[np.ndarray, float]:
    """Centre and side (square root of the area) of the drawn outline of a cube."""
    rotation, _ = cv2.Rodrigues(camera[:3])
    centres, areas = [], []
    for yaw in YAWS:
        turn = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        corners = centre + 0.5 * size * CORNERS @ turn.T
        seen = corners @ rotation.T + camera[3:6]
        pixels = (focal * seen[:, :2] / np.maximum(seen[:, 2:3], 1e-3) + CENTRE).astype(np.float32)
        hull = cv2.convexHull(pixels)
        moments = cv2.moments(hull)
        areas.append(moments["m00"])
        centres.append([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]])
    return np.mean(centres, axis=0), float(np.sqrt(np.mean(areas)))


def main() -> None:
    per_picture = json.loads((OBJECTS / "object-measurements-per-picture.json").read_text())
    stored = np.load(OBJECTS / "real-first-frames.npz")["episode"].tolist()
    places = np.load(ANALYSIS / "cube-placement.npz")
    close_of = dict(zip(places["training_episode"].tolist(), places["training_close_frame"].tolist()))
    release_of = dict(zip(places["training_episode"].tolist(), places["training_release_frame"].tolist()))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state = data["episode"], data["state"]
    heights = json.loads((ANALYSIS / "geometry-from-grasps.json").read_text())["table+lift"]["fitted_on_all"]
    lift_shift = float(heights["zero_point_shift_deg"]["lift"])
    rig = json.loads((ANALYSIS / "stack-lens-fit.json").read_text())["free lens + pan, outline + cubes"]
    camera_start = np.array(rig["world_to_camera_rotation_vector"] + rig["world_to_camera_translation"])

    cubes = []  # one row per visible cube
    for number, red, blue in zip(
        stored, per_picture[PICTURES]["top camera, red cube"], per_picture[PICTURES]["top camera, blue cube"]
    ):
        if number not in close_of:
            continue
        chosen = np.flatnonzero(episode == number)
        close, release = int(close_of[number]), int(release_of[number])
        for colour, found, frame in (
            ("red", red, min(close + 10, release - 1)), ("blue", blue, release)
        ):
            if found is None or found["touches_edge"]:
                continue
            cubes.append({
                "recording": int(number), "size": SIZES[colour],
                "reading": state[chosen[frame]].astype(np.float64),
                "centre_px": np.array(found["centre_px"]), "side_px": float(found["side_px"]),
            })
    recording = np.array([cube["recording"] for cube in cubes])
    pan_deg = np.array([cube["reading"][0] for cube in cubes])
    seen_centre = np.array([cube["centre_px"] for cube in cubes])
    seen_side = np.array([cube["side_px"] for cube in cubes])
    size = np.array([cube["size"] for cube in cubes])

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
        probe = cubes[0]["reading"].copy()
        at = {}
        for angle in (0.0, 90.0, 180.0):
            probe[0] = angle
            at[angle] = tip(probe)[:2]
        axis = 0.5 * (at[0.0] + at[180.0])
        arm = at[0.0] - axis
        quarter = at[90.0] - axis
        direction = float(np.sign(arm[0] * quarter[1] - arm[1] * quarter[0]))

        def tips_without_pan(shift: float) -> np.ndarray:
            rows = []
            for cube in cubes:
                reading = cube["reading"].copy()
                reading[0] = 0.0
                reading[1] += shift
                rows.append(tip(reading))
            return np.array(rows)

        unturned = {0.0: tips_without_pan(0.0), lift_shift: tips_without_pan(lift_shift)}
    finally:
        env.close()

    def cube_centres(shift: float, factor: float, table_drop: float) -> np.ndarray:
        angle = direction * np.deg2rad(factor * pan_deg)
        offset = unturned[shift][:, :2] - axis
        turned = np.column_stack([
            np.cos(angle) * offset[:, 0] - np.sin(angle) * offset[:, 1],
            np.sin(angle) * offset[:, 0] + np.cos(angle) * offset[:, 1],
        ]) + axis
        return np.column_stack([turned, -table_drop + 0.5 * size])

    def predicted(x: np.ndarray, shift: float, fixed: float | None, table_drop: float):
        factor = float(x[7]) if fixed is None else fixed
        centres = cube_centres(shift, factor, table_drop)
        drawn = [blob(centre, s, x[:6], float(x[6])) for centre, s in zip(centres, size)]
        return np.array([d[0] for d in drawn]), np.array([d[1] for d in drawn])

    def residuals(x, shift, fixed, table_drop, chosen):
        centre, side = predicted(x, shift, fixed, table_drop)
        return np.concatenate([
            (centre - seen_centre)[chosen].ravel(), SIZE_WEIGHT * (side - seen_side)[chosen]])

    def describe(x, shift, fixed, table_drop, chosen) -> dict:
        centre, side = predicted(x, shift, fixed, table_drop)
        miss_px = np.linalg.norm(centre - seen_centre, axis=1)[chosen]
        pixels_per_mm = (seen_side / (1000.0 * size))[chosen]
        return {
            "cubes": int(chosen.sum()),
            "centre_miss_px_median": round(float(np.median(miss_px)), 1),
            "centre_miss_px_p90": round(float(np.percentile(miss_px, 90)), 1),
            "centre_miss_mm_median_roughly": round(float(np.median(miss_px / pixels_per_mm)), 1),
            "drawn_size_over_seen_size_median": round(float(np.median((side / seen_side)[chosen])), 3),
        }

    even, odd = recording % 2 == 0, recording % 2 == 1
    everything = np.ones(len(cubes), dtype=bool)
    result = {
        "cubes_used": len(cubes), "recordings_used": int(len(set(recording.tolist()))),
        "pan_axis_in_robot_base_m": np.round(axis, 4).tolist(),
        "lift_zero_point_shift_deg": lift_shift,
    }
    for name, (fixed, shift, table_drop) in EXPLANATIONS.items():
        if shift is None:
            shift = lift_shift
            # Grasp tip height of that fit, less the 10 mm the tip is above the
            # table when it holds the 20 mm cube at half height.
            table_drop = 0.010 - heights["grasp_tip_height_relative_to_robot_base_mm"] / 1000.0
        fitted = {}
        for label, chosen in (("all", everything), ("even", even)):
            fits = []
            for focal in FOCAL_STARTS:
                guess = camera_start.copy()
                guess[3:6] *= focal / float(rig["focal_length_px"])
                x0 = [*guess, focal] + ([1.0] if fixed is None else [])
                scale = [0.05] * 3 + [0.02] * 3 + [20.0] + ([0.05] if fixed is None else [])
                fits.append(least_squares(
                    residuals, np.array(x0), args=(shift, fixed, table_drop, chosen),
                    loss="soft_l1", f_scale=5.0, x_scale=scale))
            fitted[label] = min(fits, key=lambda found: found.cost).x
        x = fitted["all"]
        rotation, _ = cv2.Rodrigues(x[:3])
        eye = -rotation.T @ x[3:6]
        row = {
            "table_below_robot_base_mm": round(1000 * table_drop, 1),
            "pan_factor": round(float(x[7]) if fixed is None else fixed, 3),
            "pan_factor_fitted_on_even": round(float(fitted["even"][7]) if fixed is None else fixed, 3),
            "vertical_field_of_view_deg": round(float(np.degrees(2 * np.arctan(240 / x[6]))), 1),
            "camera_position_m": np.round(eye, 3).tolist(),
            "camera_tilt_from_straight_down_deg": round(
                float(np.degrees(np.arccos(-rotation.T[2, 2]))), 1),
            "fitted_on_all": describe(x, shift, fixed, table_drop, everything),
            "even_fit_checked_on_odd": describe(fitted["even"], shift, fixed, table_drop, odd),
            "world_to_camera_rotation_vector": x[:3].tolist(),
            "world_to_camera_translation": x[3:6].tolist(),
            "focal_length_px": float(x[6]),
        }
        result[name] = row
        print(name, json.dumps(row, indent=1), flush=True)
    (ANALYSIS / "camera-from-cubes.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
