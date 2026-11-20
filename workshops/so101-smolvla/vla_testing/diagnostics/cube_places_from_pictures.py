"""Do the two cameras of the recording-rig scene agree on where the large cube stood?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict (their files
hold the real wrist pictures 25 frames before the human lets go):
    cube_places_from_pictures.py

Until now a cube's place was taken from the arm: the red cube where the
gripper tip is at the grasp reading, the large cube where the tip is at the
release reading. Here the places are read from the pictures instead, with the
cameras of the stack2-recording-rig scene, and nothing is rendered: a cube of
the scene's size is projected into a picture and moved over the table until
its outline covers the cube's pixels in the real picture.

  1. top picture, first frame of a recording (arm folded, both cubes free)
     -> place and turn of both cubes on the table
  2. the gripper tip at the grasp and release readings against those places
  3. real wrist picture 25 frames before the release, arm at the recorded
     readings: the large cube projected at the place from step 1, and the
     place at which the wrist picture itself would have it. If the scene's
     arm, table and wrist camera are those of the recording robot, the two
     places are the same.

Writes analysis/cube-places-from-pictures.json and, per recording,
analysis/cube-places-from-pictures.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
from grasp_aim_check import recordings
from measure_objects import colour_signal
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
DARKEST_TABLE_VALUE = 70
# (search reach in mm, step in mm, turn step in degrees), coarse then fine
SEARCHES = ((48.0, 6.0, 15.0), (6.0, 1.0, 5.0))
QUARTER_TURN_DEG = 90.0
SMALLEST_CUBE_PX = 150


def cube_corners(centre: np.ndarray, side: float, turn_deg: float) -> np.ndarray:
    half = side / 2
    c, s = np.cos(np.deg2rad(turn_deg)), np.sin(np.deg2rad(turn_deg))
    flat = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    flat = flat @ np.array([[c, s], [-s, c]]) + centre[:2]
    return np.array([[x, y, centre[2] + z] for z in (-half, half) for x, y in flat])


def outline(corners: np.ndarray, projection: np.ndarray, shape: tuple[int, int]) -> np.ndarray | None:
    """Pixels a cube with these corners covers, or None if it is not in front of the camera."""
    seen = projection @ np.column_stack([corners, np.ones(len(corners))]).T
    if (seen[2] <= 1e-3).any():
        return None
    points = np.round(seen[:2] / seen[2]).T.astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 1)
    return mask.astype(bool)


def coloured(image: np.ndarray, colour: str, free: bool) -> np.ndarray | None:
    """Pixels of the largest patch of this cube colour in a real picture.

    A cube that lies free is a compact patch; that rule keeps the bluish wall
    beyond the table out of the top picture. In the wrist picture the hand
    covers part of the cube, and nothing else there has its colour.
    """
    signal = colour_signal(image, colour)
    around = float(np.median(signal))
    candidate = (signal - around > (10.0 if colour == "red" else 6.0)).astype(np.uint8)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate)
    best, best_area = None, SMALLEST_CUBE_PX
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        if free and (area / (width * height) < 0.5 or not 0.4 < width / height < 2.5):
            continue
        if area > best_area:
            best, best_area = index, area
    if best is None:
        return None
    patch = labels == best
    peak = float(np.percentile(signal[patch], 90))
    return patch & (signal > around + 0.4 * (peak - around))


def agreement(real: np.ndarray, drawn: np.ndarray | None, counted: np.ndarray) -> float:
    """Shared pixels over all pixels of either, where the real picture shows the table or the cube."""
    if drawn is None:
        return 0.0
    either = ((real | drawn) & counted).sum()
    return float((real & drawn & counted).sum() / either) if either else 0.0


def place_cube(
    real: np.ndarray, counted: np.ndarray, projection: np.ndarray,
    start: np.ndarray, side: float, turns: np.ndarray | None,
) -> tuple[np.ndarray, float, float]:
    """Place (and turn) on the table at which the projected cube covers the real pixels best."""
    centre, turn = start.copy(), 0.0
    best = -1.0
    for reach, step, turn_step in SEARCHES:
        offsets = np.arange(-reach, reach + step / 2, step) / 1000.0
        candidates = turns if turns is not None else (
            np.arange(0.0, QUARTER_TURN_DEG, turn_step) if best < 0
            else turn + np.arange(-2 * turn_step, 2 * turn_step + 0.1, turn_step))
        found = (best, centre, turn)
        for dx in offsets:
            for dy in offsets:
                moved = centre + np.array([dx, dy, 0.0])
                for angle in candidates:
                    score = agreement(
                        real, outline(cube_corners(moved, side, angle), projection, real.shape),
                        counted)
                    if score > found[0]:
                        found = (score, moved, float(angle))
        best, centre, turn = found
    return centre, turn % QUARTER_TURN_DEG, best


def main() -> None:
    top_frames = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    wrist_frames = np.load(ANALYSIS / "release-aim-predictions.npz")
    release_scenes = np.load(ANALYSIS / "release-aim-pictures.npz")
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    frames_of = {number: (close, release) for number, close, release in recordings()}

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    rows: list[dict] = []
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        table = float(inner.TABLE_SURFACE_Z)
        sides = {colour: float(inner.cube_sizes[colour]) for colour in ("red", "blue")}

        def pose_robot(reading: np.ndarray) -> np.ndarray:
            qpos = inner._simulated_qpos(np.asarray(reading, dtype=np.float64))
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0).astype(np.float64)

        def projection(camera_name: str) -> np.ndarray:
            """Table coordinates to picture coordinates, for the arm pose set last."""
            inner.get_obs()
            camera = inner._sensors[camera_name].camera
            return (camera.get_intrinsic_matrix()[0].cpu().numpy().astype(np.float64)
                    @ camera.get_extrinsic_matrix()[0].cpu().numpy().astype(np.float64))

        # The projection used here must be the one the scene draws with.
        pose_robot(release_scenes["state"][0])
        drawn_ids = inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()
        blue_pose = inner.cubes["blue"].pose
        centre = blue_pose.p[0].cpu().numpy().astype(np.float64)
        w, x, y, z = blue_pose.q[0].cpu().numpy().astype(np.float64)
        turn = np.rad2deg(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        drawn = drawn_ids == int(inner.cubes["blue"].per_scene_id[0])
        visible = ~np.isin(drawn_ids, [int(link.per_scene_id[0]) for link in inner.agent.robot.links])
        check = agreement(
            drawn, outline(cube_corners(centre, sides["blue"], turn), projection("top"), drawn.shape),
            visible)
        print(f"projected cube against the cube the scene draws: {check:.3f} of the pixels shared",
              flush=True)
        if check < 0.9:
            raise RuntimeError("the projection is not the one the scene draws with")

        for index, number in enumerate(release_scenes["episode"].tolist()):
            close, release = frames_of[number]
            state = data["state"][data["episode"] == number].astype(np.float64)
            action = data["action"][data["episode"] == number].astype(np.float64)
            mine = (top_frames["episode"] == number) & (top_frames["moment"] == 0)
            first = top_frames["top"][mine][0]
            tips = {
                "red": pose_robot(state[min(close + 10, release - 1)]),
                "blue": pose_robot(state[release]),
            }
            commanded_release_tip = pose_robot(action[release])
            pose_robot(state[0])
            top = projection("top")
            row: dict = {"recording": int(number)}
            everywhere = np.ones(first.shape[:2], dtype=bool)
            places, turns = {}, {}
            for colour in ("red", "blue"):
                real = coloured(first, colour, free=True)
                if real is None:
                    break
                start = np.array([*tips[colour][:2], table + sides[colour] / 2])
                places[colour], turns[colour], score = place_cube(
                    real, everywhere, top, start, sides[colour], None)
                row[f"{colour}_top_picture_agreement"] = score
                row[f"{colour}_tip_minus_place_mm"] = (
                    1000.0 * (tips[colour][:2] - places[colour][:2])).tolist()
                row[f"{colour}_place_mm"] = (1000.0 * places[colour][:2]).tolist()
                row[f"{colour}_turn_deg"] = turns[colour]
            if len(places) < 2:
                print(f"recording {number}: a cube was not found in the first top picture", flush=True)
                continue
            row["commanded_release_tip_minus_large_place_mm"] = (
                1000.0 * (commanded_release_tip[:2] - places["blue"][:2])).tolist()
            row["release_tip_above_table_mm"] = 1000.0 * (tips["blue"][2] - table)

            # Wrist picture 25 frames before the release, arm at the recorded readings.
            picture = wrist_frames["real_wrist"][index]
            real = coloured(picture, "blue", free=False)
            if real is None:
                print(f"recording {number}: no large cube in the wrist picture", flush=True)
                continue
            hidden = cv2.dilate(
                ((picture.max(axis=2) < DARKEST_TABLE_VALUE)
                 | (colour_signal(picture, "red") > float(np.median(colour_signal(picture, "red"))) + 10.0)
                 ).astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            hand_tip = pose_robot(release_scenes["state"][index])
            wrist = projection("wrist")
            from_top = outline(
                cube_corners(places["blue"], sides["blue"], turns["blue"]), wrist, real.shape)
            row["wrist_agreement_at_top_picture_place"] = agreement(real, from_top, ~hidden)
            seen_place, _, score = place_cube(
                real, ~hidden, wrist, places["blue"], sides["blue"], np.array([turns["blue"]]))
            row["wrist_agreement_at_own_place"] = score
            row["wrist_place_minus_top_place_mm"] = (
                1000.0 * (seen_place[:2] - places["blue"][:2])).tolist()
            row["hand_tip_minus_top_place_mm"] = (1000.0 * (hand_tip[:2] - places["blue"][:2])).tolist()
            row["hand_tip_above_table_mm"] = 1000.0 * (hand_tip[2] - table)
            rows.append(row)
            print(json.dumps({key: (np.round(value, 2).tolist() if not isinstance(value, int) else value)
                              for key, value in row.items()}), flush=True)
    finally:
        env.close()

    def along(key: str, colour: str) -> dict:
        """Median of a table offset, split into away from the robot and to the robot's left."""
        places = np.array([row[f"{colour}_place_mm"] for row in rows])
        outward = places / np.linalg.norm(places, axis=1, keepdims=True)
        leftward = np.column_stack([-outward[:, 1], outward[:, 0]])
        offsets = np.array([row[key] for row in rows])
        return {
            "away_from_robot_mm_median": round(float(np.median((offsets * outward).sum(axis=1))), 1),
            "to_the_left_mm_median": round(float(np.median((offsets * leftward).sum(axis=1))), 1),
            "distance_mm_median": round(float(np.median(np.linalg.norm(offsets, axis=1))), 1),
            "distance_mm_p90": round(float(np.percentile(np.linalg.norm(offsets, axis=1), 90)), 1),
        }

    def middle(key: str) -> float:
        return round(float(np.median([row[key] for row in rows])), 3)

    summary = {
        "recordings": len(rows),
        "top picture: share of pixels shared by the real and the projected cube": {
            "red": middle("red_top_picture_agreement"), "large": middle("blue_top_picture_agreement")},
        "tip at the grasp reading minus the red cube's place in the top picture":
            along("red_tip_minus_place_mm", "red"),
        "tip at the release reading minus the large cube's place in the top picture":
            along("blue_tip_minus_place_mm", "blue"),
        "tip at the release command minus the large cube's place in the top picture":
            along("commanded_release_tip_minus_large_place_mm", "blue"),
        "wrist picture 25 frames before the release": {
            "share of pixels shared, large cube projected at its top-picture place":
                middle("wrist_agreement_at_top_picture_place"),
            "share of pixels shared, large cube at the place the wrist picture fits best":
                middle("wrist_agreement_at_own_place"),
            "place fitted in the wrist picture minus place in the top picture":
                along("wrist_place_minus_top_place_mm", "blue"),
            "hand tip minus the large cube's place in the top picture":
                along("hand_tip_minus_top_place_mm", "blue"),
            "hand tip above the table, mm": middle("hand_tip_above_table_mm"),
        },
    }
    print(json.dumps(summary, indent=1), flush=True)
    (ANALYSIS / "cube-places-from-pictures.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
    np.savez(
        ANALYSIS / "cube-places-from-pictures.npz",
        episode=np.array([row["recording"] for row in rows]),
        red_place_mm=np.array([row["red_place_mm"] for row in rows]),
        large_place_mm=np.array([row["blue_place_mm"] for row in rows]),
        red_turn_deg=np.array([row["red_turn_deg"] for row in rows]),
        large_turn_deg=np.array([row["blue_turn_deg"] for row in rows]),
    )


if __name__ == "__main__":
    main()
