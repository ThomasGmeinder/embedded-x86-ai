"""Does the red cube sit in the simulated hand where it sat in the recording robot's hand?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict (their file
holds the real wrist pictures 25 frames before the human lets go):
    red_cube_in_hand.py

The cube's seat is its centre measured from the gripper tip (the midpoint
between the two fingertips), in millimetres along three directions fixed to
the hand: across (from one fingertip to the other), along the fingers (from
the hand towards the fingertips) and off the fingers (square to both).

Three readings of the seat, 25 frames before the human lets go:
  scene      the stack2-recording-rig scene after the human's commands were
             replayed in it, also at the moment the replay has lifted the
             cube 5 mm, to see whether the cube slides in the hand afterwards
  picture    the seat at which a cube of the scene's size, projected with the
             scene's wrist camera, covers the red pixels of the real wrist
             picture best. The same search on the scene's own picture must
             give back the scene's seat; that error is reported.
  heights    from the recorded joint readings alone: the cube stands on the
             table when it is grasped and on the large cube when it is let
             go, so its centre height is known at both moments; the seat
             along the fingers is that height minus the tip's, divided by how
             steeply the fingers point down. All recordings with a known
             grasp and release frame.

Writes analysis/red-cube-in-hand.json and, per recording,
analysis/red-cube-in-hand.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transforms3d.axangles import axangle2mat
from transforms3d.quaternions import quat2mat

from cube_places_from_pictures import agreement, coloured, outline
from release_aim_check import recording_rig_scene, scenes_before_release

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
DARKEST_TABLE_VALUE = 70
HAND_MARGIN_PX = 2
LIFTED_MM = 5.0
# Fingers flatter than this say too little about the seat from heights alone.
STEEPEST_USABLE = 0.5
# (reach along and off the fingers in mm, step in mm, reach of the turn in
# degrees, step of the turn in degrees), coarse then fine
SEARCHES = ((30.0, 3.0, 45.0, 6.0), (3.0, 1.0, 6.0, 2.0))
WINDOW_MARGIN_PX = 120
CORNER_SIGNS = np.array(
    [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])


def hand_directions(inner) -> tuple[np.ndarray, np.ndarray]:
    """The gripper tip and the rows across, along and off the fingers, for the pose set last."""
    first, second = (
        tip.pose.p[0].cpu().numpy().astype(np.float64)
        for tip in (inner.agent.finger1_tip, inner.agent.finger2_tip))
    hand = inner.agent.robot.links_map["gripper_link"].pose.p[0].cpu().numpy().astype(np.float64)
    tip = (first + second) / 2
    across = (second - first) / np.linalg.norm(second - first)
    along = tip - hand
    along -= (along @ across) * across
    along /= np.linalg.norm(along)
    return tip, np.array([across, along, np.cross(across, along)])


def cube_pose(cube) -> tuple[np.ndarray, np.ndarray]:
    pose = cube.pose
    return (pose.p[0].cpu().numpy().astype(np.float64),
            quat2mat(pose.q[0].cpu().numpy().astype(np.float64)))


def turn_against_fingers(rotation: np.ndarray, directions: np.ndarray) -> float:
    """Degrees between the fingers and the cube edge nearest to them."""
    return float(np.rad2deg(np.arccos(min(1.0, np.abs(directions[1] @ rotation).max()))))


def seat_from_picture(
    red: np.ndarray, counted: np.ndarray, projection: np.ndarray,
    tip: np.ndarray, directions: np.ndarray, start: np.ndarray, rotation: np.ndarray, side: float,
) -> tuple[np.ndarray, float, float]:
    """Seat and turn about the across direction at which the projected cube covers the red pixels best.

    The cube stays between the fingers: its seat across is the start's.
    Returns the seat in metres, the cube's turn against the fingers in
    degrees and the share of pixels shared.
    """
    rows, columns = np.nonzero(red)
    top, left = max(rows.min() - WINDOW_MARGIN_PX, 0), max(columns.min() - WINDOW_MARGIN_PX, 0)
    window = (slice(top, rows.max() + WINDOW_MARGIN_PX), slice(left, columns.max() + WINDOW_MARGIN_PX))
    red, counted = red[window], counted[window]
    projection = np.array([[1.0, 0.0, -left], [0.0, 1.0, -top], [0.0, 0.0, 1.0]]) @ projection

    def score(seat: np.ndarray, turned: np.ndarray) -> float:
        corners = tip + seat @ directions + (side / 2) * CORNER_SIGNS @ turned.T
        return agreement(red, outline(corners, projection, red.shape), counted)

    seat, turn, best = start.copy(), 0.0, -1.0
    for reach, step, turn_reach, turn_step in SEARCHES:
        offsets = np.arange(-reach, reach + step / 2, step) / 1000.0
        found = (best, seat, turn)
        for angle in turn + np.arange(-turn_reach, turn_reach + turn_step / 2, turn_step):
            turned = axangle2mat(directions[0], np.deg2rad(angle)) @ rotation
            for along in offsets:
                for off in offsets:
                    moved = seat + np.array([0.0, along, off])
                    value = score(moved, turned)
                    if value > found[0]:
                        found = (value, moved, float(angle))
        best, seat, turn = found
    turned = axangle2mat(directions[0], np.deg2rad(turn)) @ rotation
    return seat, turn_against_fingers(turned, directions), best


def spread(values) -> list[float]:
    """10 %, median and 90 % of the values."""
    return [round(float(v), 1) for v in np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90])]


def main() -> None:
    stored = np.load(ANALYSIS / "release-aim-predictions.npz")
    real_wrist_of = dict(zip(stored["episode"].tolist(), stored["real_wrist"]))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    frames = np.load(ANALYSIS / "cube-placement.npz")

    env = recording_rig_scene()
    rows: list[dict] = []
    heights: list[dict] = []
    try:
        inner = env._env.unwrapped
        red = inner.cubes["red"]
        table = float(inner.TABLE_SURFACE_Z)
        sides = {colour: float(inner.cube_sizes[colour]) for colour in ("red", "blue")}
        robot_ids = [int(link.per_scene_id[0]) for link in inner.agent.robot.links]
        red_id = int(red.per_scene_id[0])

        def seat_now() -> tuple[np.ndarray, float]:
            tip, directions = hand_directions(inner)
            centre, rotation = cube_pose(red)
            return directions @ (centre - tip), turn_against_fingers(rotation, directions)

        # The seat while the human's commands are replayed.
        watched: dict[int, list] = {}

        def watch(number: int) -> None:
            seat, turn = seat_now()
            watched.setdefault(number, []).append((float(red.pose.p[0, 2]), seat, turn))

        for number, frame, _ in scenes_before_release(env, watch):
            seen = inner.get_obs()
            camera = inner._sensors["wrist"].camera
            projection = (camera.get_intrinsic_matrix()[0].cpu().numpy().astype(np.float64)
                          @ camera.get_extrinsic_matrix()[0].cpu().numpy().astype(np.float64))
            drawn_ids = seen["sensor_data"]["wrist"]["segmentation"][0, ..., 0].cpu().numpy()
            tip, directions = hand_directions(inner)
            centre, rotation = cube_pose(red)
            seat = directions @ (centre - tip)
            cube_height = np.array([height for height, _, _ in watched[number]])
            lifted = int(np.argmax(cube_height > cube_height[0] + LIFTED_MM / 1000.0))
            row: dict = {
                "recording": int(number),
                "scene_seat_mm": (1000.0 * seat).tolist(),
                "scene_turn_deg": turn_against_fingers(rotation, directions),
                "scene_seat_when_lifted_mm": (1000.0 * watched[number][lifted][1]).tolist(),
                "scene_turn_when_lifted_deg": watched[number][lifted][2],
            }

            # The search must give back the scene's own seat from the scene's own picture.
            drawn = drawn_ids == red_id
            if drawn.sum():
                own, own_turn, own_score = seat_from_picture(
                    drawn, ~np.isin(drawn_ids, robot_ids), projection,
                    tip, directions, seat, rotation, sides["red"])
                row["search_error_on_scene_picture_mm"] = (1000.0 * (own - seat)).tolist()
                row["search_turn_on_scene_picture_deg"] = own_turn
                row["search_agreement_on_scene_picture"] = own_score
                rows_px, columns_px = np.nonzero(drawn)
                row["scene_pixels"] = [float(columns_px.mean()), float(rows_px.mean()),
                                       float(np.sqrt(drawn.sum()))]

            real = real_wrist_of[number]
            real_red = coloured(real, "red", free=False)
            if real_red is not None:
                real_hand = cv2.dilate(
                    (real.max(axis=2) < DARKEST_TABLE_VALUE).astype(np.uint8),
                    np.ones((2 * HAND_MARGIN_PX + 1,) * 2, np.uint8)).astype(bool)
                fitted, fitted_turn, score = seat_from_picture(
                    real_red, ~real_hand, projection, tip, directions, seat, rotation, sides["red"])
                row["picture_seat_mm"] = (1000.0 * fitted).tolist()
                row["picture_turn_deg"] = fitted_turn
                row["picture_agreement"] = score
                rows_px, columns_px = np.nonzero(real_red)
                row["real_pixels"] = [float(columns_px.mean()), float(rows_px.mean()),
                                      float(np.sqrt(real_red.sum()))]
            rows.append(row)
            print(json.dumps({key: (np.round(value, 2).tolist() if not isinstance(value, int) else value)
                              for key, value in row.items()}), flush=True)

        # From the recorded joint readings alone.
        def tip_at(reading: np.ndarray) -> tuple[float, float]:
            qpos = inner._simulated_qpos(np.asarray(reading, dtype=np.float64))
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            tip, directions = hand_directions(inner)
            return float(tip[2]), float(directions[1, 2])

        cube_centre = {
            "grasp": table + sides["red"] / 2,
            "release": table + sides["blue"] + sides["red"] / 2,
        }
        for number, close, release in zip(
            frames["training_episode"].tolist(), frames["training_close_frame"].tolist(),
            frames["training_release_frame"].tolist(),
        ):
            state = data["state"][data["episode"] == number]
            moments = {"grasp": min(close + 10, release - 1), "release": release}
            row = {"recording": int(number)}
            for name, frame in moments.items():
                height, steepness = tip_at(state[frame])
                row[f"tip_above_table_at_{name}_mm"] = 1000.0 * (height - table)
                row[f"tip_above_cube_centre_at_{name}_mm"] = 1000.0 * (height - cube_centre[name])
                row[f"fingers_point_down_at_{name}"] = -steepness
                row[f"seat_along_fingers_at_{name}_mm"] = (
                    1000.0 * (cube_centre[name] - height) / steepness
                    if -steepness > STEEPEST_USABLE else None)
            heights.append(row)
    finally:
        env.close()

    def column(key: str, index: int | None = None, source: list[dict] = rows) -> np.ndarray:
        values = [row[key] for row in source if row.get(key) is not None]
        return np.array([value[index] for value in values] if index is not None else values)

    names = ("across", "along the fingers", "off the fingers")
    seat_difference = np.array([
        np.array(row["picture_seat_mm"]) - np.array(row["scene_seat_mm"])
        for row in rows if "picture_seat_mm" in row])
    slide = np.array([
        np.array(row["scene_seat_mm"]) - np.array(row["scene_seat_when_lifted_mm"]) for row in rows])
    summary = {
        "recordings with a wrist picture and a carried cube": len(rows),
        "scene: seat 25 frames before the release, mm (10 %, median, 90 %)": {
            name: spread(column("scene_seat_mm", index)) for index, name in enumerate(names)},
        "scene: turn of the cube against the fingers, degrees": spread(column("scene_turn_deg")),
        "scene: seat when the replay has lifted the cube 5 mm, mm": {
            name: spread(column("scene_seat_when_lifted_mm", index)) for index, name in enumerate(names)},
        "scene: turn when lifted, degrees": spread(column("scene_turn_when_lifted_deg")),
        "scene: how far the cube moves in the hand between the two moments, mm": {
            name: spread(slide[:, index]) for index, name in enumerate(names)},
        "search on the scene's own picture: error of the seat, mm": {
            name: spread(column("search_error_on_scene_picture_mm", index))
            for index, name in enumerate(names)},
        "search on the scene's own picture: share of pixels shared": spread(
            100.0 * column("search_agreement_on_scene_picture")),
        "real wrist picture: recordings in which the red cube was found": len(seat_difference),
        "real wrist picture: seat, mm": {
            name: spread(column("picture_seat_mm", index)) for index, name in enumerate(names)},
        "real wrist picture: turn of the cube against the fingers, degrees": spread(
            column("picture_turn_deg")),
        "real wrist picture: share of pixels shared, per cent": spread(
            100.0 * column("picture_agreement")),
        "real wrist picture minus scene: seat, mm": {
            name: spread(seat_difference[:, index]) for index, name in enumerate(names)},
        "red cube in the wrist picture, real (column, row, square root of its pixel count)": [
            spread(column("real_pixels", index)) for index in range(3)],
        "red cube in the wrist picture, scene (column, row, square root of its pixel count)": [
            spread(column("scene_pixels", index)) for index in range(3)],
        "from the recorded readings alone": {
            "recordings": len(heights),
            **{
                f"{what} at the {name}": spread(column(f"{key}_at_{name}{unit}", source=heights))
                for name in ("grasp", "release")
                for what, key, unit in (
                    ("tip above the table, mm", "tip_above_table", "_mm"),
                    ("tip above the red cube's centre, mm", "tip_above_cube_centre", "_mm"),
                    ("how steeply the fingers point down (1 = straight down)", "fingers_point_down", ""),
                    ("seat along the fingers, mm", "seat_along_fingers", "_mm"),
                )
            },
        },
    }
    print(json.dumps(summary, indent=1), flush=True)
    (ANALYSIS / "red-cube-in-hand.json").write_text(
        json.dumps({"summary": summary, "rows": rows, "heights": heights}, indent=1) + "\n",
        encoding="utf-8")
    np.savez(
        ANALYSIS / "red-cube-in-hand.npz",
        episode=np.array([row["recording"] for row in rows]),
        scene_seat_mm=column("scene_seat_mm"),
        scene_turn_deg=column("scene_turn_deg"),
    )


if __name__ == "__main__":
    main()
