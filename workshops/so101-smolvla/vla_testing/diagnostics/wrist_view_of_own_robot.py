"""Which parts of its own robot does the scene's wrist camera see, and did the real camera see them?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    wrist_view_of_own_robot.py

No model, nothing simulated forward. For every stored real wrist picture
(analysis/wrist-frames-before-grasp.npz: 5, 20, 40 and 60 frames before the
human closes the gripper on the red cube) the stack2-recording-rig scene is
posed at the recorded joint readings of that frame, with the cubes where the
recording had them, and the scene's wrist camera reports for every pixel
which part of the scene it shows.

The recording robot is black; nothing else in a wrist picture is that dark.
So where the scene shows a robot link, the real picture should be dark. Per
robot link that is not part of the hand: in how many pictures it shows, how
many pixels, and what share of those pixels is dark in the real picture of
the same moment. The hand links are listed too, as the check of the method.

Writes analysis/wrist-view-of-own-robot.json and
analysis/wrist-view-of-own-robot.png (real picture next to the scene's, for
the pictures in which the scene shows most of a link outside the hand).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from grasp_aim_check import recordings
from release_aim_check import recorded, recording_rig_scene, tip_at
from so101_smolvla.sapien_config import HAND_LINKS

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
# The real robot is black; the table and the cubes are never this dark.
DARKEST_TABLE_VALUE = 70
# A link counts as showing in a picture from this many pixels on.
SHOWS_FROM_PX = 50
SHEET_PICTURES = 8


def main() -> None:
    import torch
    from mani_skill.utils.structs.pose import Pose

    frames = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    episode_of, state = recorded()["episode"], recorded()["state"]
    events = {number: (close, release) for number, close, release in recordings()}
    env = recording_rig_scene()
    rows = []
    try:
        inner = env._env.unwrapped
        link_of_id = {
            int(identity): thing.name
            for identity, thing in inner.segmentation_id_map.items()
            if thing in inner.agent.robot.links
        }

        def stand(colour: str, xy: np.ndarray) -> None:
            height = inner.TABLE_SURFACE_Z + inner.cube_sizes[colour] / 2
            inner.cubes[colour].set_pose(Pose.create_from_pq(p=torch.tensor(
                [[float(xy[0]), float(xy[1]), height]],
                dtype=torch.float32, device=inner.device)))

        for index, (number, frame, before) in enumerate(zip(
                frames["episode"].tolist(), frames["frame"].tolist(),
                frames["frames_before_closing"].tolist())):
            if number not in events:
                continue
            close, release = events[number]
            readings = state[episode_of == number].astype(np.float64)
            stand("red", tip_at(inner, readings[min(close + 10, release - 1)]))
            stand("blue", tip_at(inner, readings[release]))
            tip = tip_at(inner, readings[frame])
            seen = inner.get_obs()["sensor_data"]["wrist"]
            shown = seen["segmentation"][0, ..., 0].cpu().numpy()
            real = frames["wrist"][index]
            dark = real.max(axis=2) < DARKEST_TABLE_VALUE
            links = {}
            for identity, name in link_of_id.items():
                here = shown == identity
                if here.sum() >= SHOWS_FROM_PX:
                    links[name] = {"px": int(here.sum()),
                                   "dark_in_real_share": round(float(dark[here].mean()), 3)}
            outside_hand = sum(v["px"] for name, v in links.items() if name not in HAND_LINKS)
            rows.append({
                "recording": number, "frames_before_closing": before,
                "wrist_flex_reading_deg": round(float(readings[frame][3]), 1),
                "tip_from_robot_mm": round(1000.0 * float(np.hypot(tip[0], tip[1])), 1),
                "links": links, "outside_hand_px": int(outside_hand),
                "picture": index, "scene_rgb": seen["rgb"][0].cpu().numpy(),
            })
    finally:
        env.close()

    names = sorted({name for row in rows for name in row["links"]})
    per_link = {}
    for name in names:
        with_it = [row for row in rows if name in row["links"]]
        per_link[name] = {
            "part_of_the_hand": name in HAND_LINKS,
            "pictures": len(with_it),
            "px_median": float(np.median([row["links"][name]["px"] for row in with_it])),
            "dark_in_real_share_median": round(float(np.median(
                [row["links"][name]["dark_in_real_share"] for row in with_it])), 3),
            "tip_from_robot_mm_median": round(float(np.median(
                [row["tip_from_robot_mm"] for row in with_it])), 1),
            "wrist_flex_reading_deg_median": round(float(np.median(
                [row["wrist_flex_reading_deg"] for row in with_it])), 1),
        }
    with_outside = [row for row in rows if row["outside_hand_px"] >= SHOWS_FROM_PX]
    without = [row for row in rows if row["outside_hand_px"] < SHOWS_FROM_PX]
    result = {
        "pictures": len(rows),
        "pictures_in_which_the_scene_shows_a_link_outside_the_hand": len(with_outside),
        "tip_from_robot_mm_median_in_those": round(float(np.median(
            [row["tip_from_robot_mm"] for row in with_outside])), 1) if with_outside else None,
        "tip_from_robot_mm_median_in_the_others": round(float(np.median(
            [row["tip_from_robot_mm"] for row in without])), 1) if without else None,
        "per_link": per_link,
        "per_picture": [
            {key: value for key, value in row.items() if key not in ("scene_rgb", "picture")}
            for row in rows
        ],
    }
    print(json.dumps({key: value for key, value in result.items() if key != "per_picture"},
                     indent=1), flush=True)
    (ANALYSIS / "wrist-view-of-own-robot.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")

    most = sorted(rows, key=lambda row: -row["outside_hand_px"])[:SHEET_PICTURES]
    tiles = []
    for row in most:
        pair = np.concatenate([frames["wrist"][row["picture"]], row["scene_rgb"]], axis=1)
        pair = np.ascontiguousarray(pair)
        cv2.putText(pair, f"recording {row['recording']}, {row['frames_before_closing']} frames "
                    f"before closing, {row['outside_hand_px']} px", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        tiles.append(pair)
    if tiles:
        tiles.extend([np.full_like(tiles[0], 255)] * (len(tiles) % 2))
        sheet = np.concatenate(
            [np.concatenate(tiles[i:i + 2], axis=1) for i in range(0, len(tiles), 2)], axis=0)
        cv2.imwrite(str(ANALYSIS / "wrist-view-of-own-robot.png"),
                    cv2.cvtColor(cv2.resize(sheet, None, fx=0.5, fy=0.5,
                                            interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
