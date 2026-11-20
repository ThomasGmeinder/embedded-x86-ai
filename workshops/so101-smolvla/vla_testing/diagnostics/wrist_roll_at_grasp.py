"""Does the training set turn the wrist roll to fit the cube? Numbers and pictures at the grasp.

Nothing is simulated and no model is loaded. Three steps, from the workshop folder:

  export   run with .venv (has pyarrow), PYTHONPATH = ~/.cache/so101-stack-training
           Writes analysis/states-sim-cylinder40.npz (the can episodes, read with
           export_states.load like the other states-*.npz files) and
           analysis/sim-episode-video-index.npz (where every simulated episode
           starts in its video files, and the folder of the data set).
  report   run with .venv-sapien, PYTHONPATH = /tmp/Xbotics-SO101-Sim
           Needs the output of export, analysis/states-sim-cube40.npz,
           states-sim-cube20.npz, states-real-stacking.npz, cube-placement.npz
           and wrist-frames-before-grasp.npz.
  turn     run with .venv-sapien, PYTHONPATH = ~/.cache/so101-stack-training
           Needs the output of export and of report. In the wrist picture
           5 frames before the gripper closes, how far is the red cube turned
           against the jaws? The camera sits on the gripper, so a cube that
           stands square between the jaws shows a level front edge whatever
           the wrist roll is. Measured: the tilt of the flattest long edge of
           the cube's outline (flattest_edge). Every stored real picture, and
           TURN_PICTURES simulated 20 mm red cube episodes as the reference.
           Writes analysis/wrist-roll-at-grasp-cube-turn.json and
           analysis/wrist-roll-at-grasp-cube-turn-<source>.png (cube outline
           green, measured edge yellow, sorted by tilt).

The grasp moment is not decided here:
  simulated episodes   recipe.close_frame, the data set author's own rule,
                       imported from the data set's source/0_dataset_gen
  real recordings      analysis/cube-placement.npz (cube_placement_distribution.py)

For the three simulated bin tasks and the real stacking recordings the report
gives, per episode at the grasp moment: the commanded and the read wrist roll,
the commanded pan, how far the commanded roll is from its value in the first
frame, and how many frames before the closing the commanded roll had reached
its grasp value (within SETTLED_DEG) for good.

Pictures, FRAMES_BEFORE_CLOSING frames before the gripper closes, of twelve
episodes spread evenly over the wrist-roll values: wrist and top view of the
simulated cube episodes, wrist view of the real recordings (5 frames before,
the frames wrist_frames_before_grasp.py stored).

Writes analysis/wrist-roll-at-grasp.json, analysis/wrist-roll-at-grasp.npz and
analysis/wrist-roll-at-grasp-<source>-<camera>.png.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
VIDEO_INDEX = ANALYSIS / "sim-episode-video-index.npz"
SIM_SCENES = ("cube40", "cube20", "cylinder40")
SCENES_WITH_PICTURES = ("cube40", "cube20")
FPS = 30
SETTLED_DEG = 1.0
FRAMES_BEFORE_CLOSING = 10
REAL_FRAMES_BEFORE_CLOSING = 5
PICTURES_PER_SHEET = 12
SHEET_COLUMNS = 4
TILE = (320, 240)
ROLL_BIN_EDGES_DEG = np.arange(-75, 121, 15)
TURN_PICTURES = 40
TURN_COLUMNS = 8
TURN_TILE_PX = 200
EDGE_MIN_SHARE_OF_SIDE = 0.4
UPRIGHT_EDGE_DEG = 65.0


def export() -> None:
    import pyarrow.parquet as pq

    from export_states import SIM_SNAPSHOT, load

    arrays = load([SIM_SNAPSHOT / "cylinder40/data/chunk-000/file-000.parquet"])
    np.savez_compressed(ANALYSIS / "states-sim-cylinder40.npz", **arrays)
    print(
        f"sim-cylinder40: {len(np.unique(arrays['episode']))} episodes, "
        f"{len(arrays['episode'])} frames",
        flush=True,
    )
    index: dict[str, np.ndarray] = {"snapshot": np.array(str(SIM_SNAPSHOT))}
    for scene in SCENES_WITH_PICTURES:
        cameras = ("wrist", "top")
        columns = ["episode_index"] + [
            f"videos/observation.images.{camera}/{field}"
            for camera in cameras
            for field in ("file_index", "from_timestamp")
        ]
        table = pq.read_table(
            SIM_SNAPSHOT / scene / "meta/episodes/chunk-000/file-000.parquet", columns=columns
        ).to_pydict()
        index[f"{scene}_episode"] = np.array(table["episode_index"], dtype=np.int64)
        for camera in cameras:
            key = f"videos/observation.images.{camera}"
            index[f"{scene}_{camera}_file"] = np.array(table[f"{key}/file_index"], dtype=np.int64)
            index[f"{scene}_{camera}_from_s"] = np.array(
                table[f"{key}/from_timestamp"], dtype=np.float64
            )
        print(f"{scene}: video places of {len(index[f'{scene}_episode'])} episodes", flush=True)
    np.savez_compressed(VIDEO_INDEX, **index)


def spread(values: np.ndarray) -> dict[str, float]:
    low5, low25, middle, high75, high95 = np.percentile(values, [5, 25, 50, 75, 95])
    return {
        "lowest": round(float(values.min()), 1),
        "5%": round(float(low5), 1),
        "25%": round(float(low25), 1),
        "median": round(float(middle), 1),
        "75%": round(float(high75), 1),
        "95%": round(float(high95), 1),
        "highest": round(float(values.max()), 1),
    }


def facts(state: np.ndarray, action: np.ndarray, close: int) -> dict[str, float]:
    """Wrist roll and pan of one episode at the frame the gripper closes."""
    roll = action[: close + 1, 4]
    away = np.flatnonzero(np.abs(roll - roll[close]) > SETTLED_DEG)
    settled = int(away[-1]) + 1 if len(away) else 0
    return {
        "close_frame": close,
        "roll_command": float(action[close, 4]),
        "roll_reading": float(state[close, 4]),
        "pan_command": float(action[close, 0]),
        "roll_command_at_start": float(action[0, 4]),
        "frames_roll_settled_before_closing": close - settled,
    }


def summary(rows: list[dict]) -> dict:
    def column(key: str) -> np.ndarray:
        return np.array([row[key] for row in rows], dtype=np.float64)

    roll, pan = column("roll_command"), column("pan_command")
    turned = roll - column("roll_command_at_start")
    counts, _ = np.histogram(roll, bins=ROLL_BIN_EDGES_DEG)
    slope, _ = np.polyfit(pan, roll, 1)
    return {
        "episodes": len(rows),
        "roll_command_at_grasp_deg": spread(roll),
        "roll_reading_at_grasp_deg": spread(column("roll_reading")),
        "roll_command_in_first_frame_deg": spread(column("roll_command_at_start")),
        "roll_at_grasp_minus_first_frame_deg": spread(turned),
        "episodes_turned_more_than_10_deg": int((np.abs(turned) > 10.0).sum()),
        "episodes_turned_2_deg_or_less": int((np.abs(turned) <= 2.0).sum()),
        "episodes_per_15_deg_of_roll_at_grasp": {
            f"{int(low):+d} to {int(high):+d}": int(count)
            for low, high, count in zip(ROLL_BIN_EDGES_DEG[:-1], ROLL_BIN_EDGES_DEG[1:], counts)
        },
        "episodes_outside_those_bins": int(len(roll) - counts.sum()),
        "pan_command_at_grasp_deg": spread(pan),
        "roll_against_pan": {
            "correlation": round(float(np.corrcoef(pan, roll)[0, 1]), 2),
            "roll_deg_per_pan_deg": round(float(slope), 2),
        },
        "frames_before_closing_at_which_roll_had_settled": spread(
            column("frames_roll_settled_before_closing")
        ),
    }


def video_frame(path: Path, seconds: float) -> np.ndarray:
    """One 640x480 RGB picture of a video file (the ffmpeg call of measure_objects)."""
    completed = subprocess.run(
        [
            "ffmpeg", "-v", "error",
            "-ss", f"{max(seconds - 0.5 / FPS, 0.0):.6f}", "-i", str(path),
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    )
    if len(completed.stdout) != 480 * 640 * 3:
        raise RuntimeError(f"{path} at {seconds}: no frame decoded")
    return np.frombuffer(completed.stdout, dtype=np.uint8).reshape(480, 640, 3).copy()


def spread_choice(values: np.ndarray, count: int) -> np.ndarray:
    """Positions of `count` entries spread evenly over the sorted values."""
    order = np.argsort(values, kind="stable")
    if len(order) <= count:
        return order
    return order[np.round(np.linspace(0, len(order) - 1, count)).astype(int)]


def sim_picture(
    index, snapshot: Path, scene: str, camera: str, number: int, frame: int
) -> np.ndarray:
    """Picture of one frame of a simulated episode, from the data set's video files."""
    at = int(np.flatnonzero(index[f"{scene}_episode"] == number)[0])
    path = (
        snapshot / scene / "videos" / f"observation.images.{camera}" / "chunk-000"
        / f"file-{int(index[f'{scene}_{camera}_file'][at]):03d}.mp4"
    )
    return video_frame(path, float(index[f"{scene}_{camera}_from_s"][at]) + frame / FPS)


def sheet(
    pictures: list[np.ndarray],
    labels: list[str],
    path: Path,
    columns: int = SHEET_COLUMNS,
    tile: tuple[int, int] = TILE,
) -> None:
    from PIL import Image, ImageDraw

    rows = -(-len(pictures) // columns)
    canvas = Image.new("RGB", (columns * tile[0], rows * tile[1]), "black")
    draw = ImageDraw.Draw(canvas)
    for place, (picture, label) in enumerate(zip(pictures, labels, strict=True)):
        left, top = (place % columns) * tile[0], (place // columns) * tile[1]
        canvas.paste(Image.fromarray(picture).resize(tile), (left, top))
        draw.rectangle((left, top, left + 8 + 6 * len(label), top + 14), fill="black")
        draw.text((left + 4, top + 2), label, fill="white")
    canvas.save(path)
    print(f"wrote {path}", flush=True)


def report() -> None:
    index = np.load(VIDEO_INDEX)
    snapshot = Path(str(index["snapshot"]))
    sys.path.insert(0, str(snapshot / "source" / "0_dataset_gen"))
    import recipe

    result: dict = {
        "settled_means_within_deg": SETTLED_DEG,
        "percentiles": "lowest, 5%, 25%, median, 75%, 95%, highest",
    }
    stored: dict[str, np.ndarray] = {}

    for scene in SIM_SCENES:
        data = np.load(ANALYSIS / f"states-sim-{scene}.npz")
        close_pct = recipe.CLOSE_PCT[recipe.SCENES[scene]["real_task"]]
        rows, numbers, no_grasp = [], [], 0
        for number in np.unique(data["episode"]):
            chosen = data["episode"] == number
            state, action = data["state"][chosen], data["action"][chosen]
            close = recipe.close_frame(action, close_pct)
            if close is None:
                no_grasp += 1
                continue
            rows.append(facts(state, action, close))
            numbers.append(int(number))
        name = f"sim-{scene}"
        result[name] = {
            "task": recipe.SCENES[scene]["task_text"],
            "episodes_without_a_grasp_moment": no_grasp,
            **summary(rows),
        }
        stored[f"{name}_episode"] = np.array(numbers)
        for key in rows[0]:
            stored[f"{name}_{key}"] = np.array([row[key] for row in rows])
        print(f"{name}: {len(rows)} episodes, {no_grasp} without a grasp moment", flush=True)

        if scene not in SCENES_WITH_PICTURES:
            continue
        roll = stored[f"{name}_roll_command"]
        picks = spread_choice(roll, PICTURES_PER_SHEET)
        for camera in ("wrist", "top"):
            pictures, labels = [], []
            for pick in picks:
                number, close = numbers[pick], rows[pick]["close_frame"]
                pictures.append(
                    sim_picture(
                        index, snapshot, scene, camera, number, close - FRAMES_BEFORE_CLOSING
                    )
                )
                labels.append(f"ep {number}  roll {roll[pick]:+.0f} deg")
            sheet(pictures, labels, ANALYSIS / f"wrist-roll-at-grasp-{name}-{camera}.png")

    data = np.load(ANALYSIS / "states-real-stacking.npz")
    places = np.load(ANALYSIS / "cube-placement.npz")
    rows, numbers = [], []
    for number, close in zip(
        places["training_episode"].tolist(), places["training_close_frame"].tolist(), strict=True
    ):
        chosen = data["episode"] == number
        rows.append(facts(data["state"][chosen], data["action"][chosen], int(close)))
        numbers.append(int(number))
    name = "real-stacking"
    result[name] = {
        "task": "Stack the smaller cube on the larger one",
        "recordings_in_the_data_set": int(len(np.unique(data["episode"]))),
        **summary(rows),
    }
    stored[f"{name}_episode"] = np.array(numbers)
    for key in rows[0]:
        stored[f"{name}_{key}"] = np.array([row[key] for row in rows])
    print(f"{name}: {len(rows)} recordings with a known grasp moment", flush=True)

    frames = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    near = np.flatnonzero(frames["frames_before_closing"] == REAL_FRAMES_BEFORE_CLOSING)
    picture_of = {int(frames["episode"][at]): at for at in near}
    known = [at for at, number in enumerate(numbers) if number in picture_of]
    roll = stored[f"{name}_roll_command"]
    picks = [known[at] for at in spread_choice(roll[known], PICTURES_PER_SHEET)]
    sheet(
        [frames["wrist"][picture_of[numbers[pick]]] for pick in picks],
        [f"ep {numbers[pick]}  roll {roll[pick]:+.0f} deg" for pick in picks],
        ANALYSIS / f"wrist-roll-at-grasp-{name}-wrist.png",
    )
    result[name]["recordings_with_a_stored_wrist_picture"] = len(known)

    np.savez_compressed(ANALYSIS / "wrist-roll-at-grasp.npz", **stored)
    text = json.dumps(result, indent=1)
    (ANALYSIS / "wrist-roll-at-grasp.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


def flattest_edge(picture: np.ndarray) -> dict | None:
    """Tilt of the flattest long edge of the red cube's outline in a wrist picture.

    The cube is found with measure_objects.measure. Its outline is everything
    in the found window that is redder than Otsu's threshold of
    measure_objects.colour_signal; the outline's convex hull is reduced to a
    polygon. Edges shorter than EDGE_MIN_SHARE_OF_SIDE of the cube's side and
    edges steeper than UPRIGHT_EDGE_DEG (the cube's upright edges) are left
    out. Tilt is in degrees from level, positive when the edge falls to the
    right. None when no whole red cube or no such edge is found.
    """
    import cv2

    from measure_objects import colour_signal, measure

    found = measure(picture, "red")
    if found is None or found["touches_edge"]:
        return None
    centre_x, centre_y = found["centre_px"]
    reach_x, reach_y = 0.75 * found["width_px"] + 12, 0.75 * found["height_px"] + 12
    x0, y0 = max(int(centre_x - reach_x), 0), max(int(centre_y - reach_y), 0)
    x1, y1 = min(int(centre_x + reach_x), 640), min(int(centre_y + reach_y), 480)
    signal = np.clip(colour_signal(picture, "red")[y0:y1, x0:x1], 0.0, None)
    if signal.max() <= 0.0:
        return None
    scaled = np.uint8(255.0 * signal / signal.max())
    _, mask = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = cv2.convexHull(max(contours, key=cv2.contourArea))
    polygon = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)[:, 0, :]
    polygon = polygon.astype(np.float64) + (x0, y0)
    best = None
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0), strict=True):
        along = end - start
        tilt = (float(np.degrees(np.arctan2(along[1], along[0]))) + 90.0) % 180.0 - 90.0
        if (
            float(np.hypot(*along)) < EDGE_MIN_SHARE_OF_SIDE * found["side_px"]
            or abs(tilt) > UPRIGHT_EDGE_DEG
        ):
            continue
        if best is None or abs(tilt) < abs(best["tilt_deg"]):
            best = {"tilt_deg": tilt, "edge": np.array([start, end])}
    if best is None:
        return None
    return {**best, "polygon": polygon, "centre_px": found["centre_px"]}


def turn() -> None:
    import cv2

    index = np.load(VIDEO_INDEX)
    snapshot = Path(str(index["snapshot"]))
    numbers = np.load(ANALYSIS / "wrist-roll-at-grasp.npz")

    def by_episode(source: str, key: str) -> dict[int, float]:
        return dict(
            zip(numbers[f"{source}_episode"].tolist(), numbers[f"{source}_{key}"].tolist(), strict=True)
        )

    sources: dict[str, list[tuple[int, np.ndarray]]] = {}
    frames = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    near = np.flatnonzero(frames["frames_before_closing"] == REAL_FRAMES_BEFORE_CLOSING)
    sources["real-stacking"] = [(int(frames["episode"][at]), frames["wrist"][at]) for at in near]
    close = by_episode("sim-cube20", "close_frame")
    picks = spread_choice(numbers["sim-cube20_roll_command"], TURN_PICTURES)
    sources["sim-cube20"] = [
        (
            number,
            sim_picture(
                index, snapshot, "cube20", "wrist", number,
                int(close[number]) - REAL_FRAMES_BEFORE_CLOSING,
            ),
        )
        for number in numbers["sim-cube20_episode"][picks].tolist()
    ]

    result: dict = {
        "frames_before_closing": REAL_FRAMES_BEFORE_CLOSING,
        "edge_at_least_this_share_of_the_cube_side": EDGE_MIN_SHARE_OF_SIDE,
        "edges_steeper_than_this_left_out_deg": UPRIGHT_EDGE_DEG,
        "percentiles": "lowest, 5%, 25%, median, 75%, 95%, highest",
    }
    half = TURN_TILE_PX // 2
    for name, pictures in sources.items():
        roll = by_episode(name, "roll_command")
        start = by_episode(name, "roll_command_at_start")
        rows = []
        for number, picture in pictures:
            found = flattest_edge(picture)
            drawn = picture.copy()
            if found is None:
                tile = cv2.resize(drawn, (TURN_TILE_PX, TURN_TILE_PX))
            else:
                cv2.polylines(drawn, [np.int32(np.round(found["polygon"]))], True, (0, 255, 0), 1)
                cv2.polylines(drawn, [np.int32(np.round(found["edge"]))], False, (255, 255, 0), 2)
                centre_x, centre_y = (int(round(value)) for value in found["centre_px"])
                x0 = min(max(centre_x - half, 0), 640 - TURN_TILE_PX)
                y0 = min(max(centre_y - half, 0), 480 - TURN_TILE_PX)
                tile = drawn[y0 : y0 + TURN_TILE_PX, x0 : x0 + TURN_TILE_PX]
            rows.append(
                {
                    "episode": number,
                    "roll_command_at_grasp_deg": round(roll[number], 1),
                    "roll_at_grasp_minus_first_frame_deg": round(roll[number] - start[number], 1),
                    "tilt_deg": None if found is None else round(found["tilt_deg"], 1),
                    "tile": tile,
                }
            )
        rows.sort(key=lambda row: 1e9 if row["tilt_deg"] is None else abs(row["tilt_deg"]))
        sheet(
            [row["tile"] for row in rows],
            [
                f"ep {row['episode']} "
                + ("not read" if row["tilt_deg"] is None else f"tilt {row['tilt_deg']:+.0f}")
                for row in rows
            ],
            ANALYSIS / f"wrist-roll-at-grasp-cube-turn-{name}.png",
            columns=TURN_COLUMNS,
            tile=(TURN_TILE_PX, TURN_TILE_PX),
        )
        read = [row for row in rows if row["tilt_deg"] is not None]
        tilt = np.abs([row["tilt_deg"] for row in read])
        wrist_turned = np.array(
            [abs(row["roll_at_grasp_minus_first_frame_deg"]) > 10.0 for row in read]
        )
        result[name] = {
            "pictures": len(rows),
            "pictures_read": len(read),
            "tilt_of_flattest_edge_deg": spread(tilt),
            "pictures_tilted_more_than_10_deg": int((tilt > 10.0).sum()),
            "pictures_tilted_more_than_20_deg": int((tilt > 20.0).sum()),
            "wrist_roll_turned_more_than_10_deg_before_grasp": {
                "pictures": int(wrist_turned.sum()),
                "of_which_tilted_more_than_10_deg": int((tilt[wrist_turned] > 10.0).sum()),
            },
            "wrist_roll_turned_10_deg_or_less_before_grasp": {
                "pictures": int((~wrist_turned).sum()),
                "of_which_tilted_more_than_10_deg": int((tilt[~wrist_turned] > 10.0).sum()),
            },
            "per_picture": [{key: row[key] for key in row if key != "tile"} for row in rows],
        }
    text = json.dumps(result, indent=1)
    (ANALYSIS / "wrist-roll-at-grasp-cube-turn.json").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)


if __name__ == "__main__":
    {"export": export, "report": report, "turn": turn}[sys.argv[1]]()
