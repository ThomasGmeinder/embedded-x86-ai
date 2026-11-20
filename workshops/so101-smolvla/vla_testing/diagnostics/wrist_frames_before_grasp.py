"""Real wrist pictures shortly before the human closes the gripper on the red cube.

Run with .venv (needs pandas and ffmpeg), PYTHONPATH = ~/.cache/so101-stack-training,
from the workshop folder:
    wrist_frames_before_grasp.py

For every real stacking recording whose pictures are on disk (video file 2)
and whose grasp moment is known (analysis/cube-placement.npz), this stores
the wrist picture a fixed number of frames before the gripper closes, and
where the red cube is in it (measure_objects.measure, the same detector used
for every other cube measurement).

Writes analysis/wrist-frames-before-grasp.npz for wrist_cube_check.py.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from measure_objects import FPS, VIDEO_FILE, measure, real_first_frame

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
FRAMES_BEFORE_CLOSING = (60, 40, 20, 5)


def main() -> None:
    episodes = pd.concat(
        pd.read_parquet(path)
        for path in sorted(glob.glob(str(ROOT / "meta" / "episodes" / "chunk-000" / "*.parquet")))
    ).set_index("episode_index")
    places = np.load(ANALYSIS / "cube-placement.npz")
    column = "videos/observation.images.wrist"
    rows, pictures, centres, sides = [], [], [], []
    for number, close in zip(
        places["training_episode"].tolist(), places["training_close_frame"].tolist()
    ):
        row = episodes.loc[number]
        if int(row[f"{column}/file_index"]) != VIDEO_FILE:
            continue
        for before in FRAMES_BEFORE_CLOSING:
            frame = int(close) - before
            if frame < 0:
                continue
            picture = real_first_frame(
                "wrist", float(row[f"{column}/from_timestamp"]) + frame / FPS
            )
            found = measure(picture, "red")
            if found is None or found["touches_edge"]:
                continue
            rows.append((int(number), frame, before))
            pictures.append(picture)
            centres.append(found["centre_px"])
            sides.append(found["side_px"])
    rows = np.array(rows)
    np.savez_compressed(
        ANALYSIS / "wrist-frames-before-grasp.npz",
        episode=rows[:, 0],
        frame=rows[:, 1],
        frames_before_closing=rows[:, 2],
        wrist=np.stack(pictures),
        red_centre_px=np.array(centres),
        red_side_px=np.array(sides),
    )
    for before in FRAMES_BEFORE_CLOSING:
        chosen = rows[:, 2] == before
        print(
            f"{before} frames before closing: red cube found whole in "
            f"{int(chosen.sum())} wrist pictures, median place "
            f"{np.round(np.median(np.array(centres)[chosen], axis=0), 0).tolist()} px, "
            f"median side {np.median(np.array(sides)[chosen]):.0f} px",
            flush=True,
        )
    print(f"recordings: {len(set(rows[:, 0].tolist()))}, pictures: {len(rows)}", flush=True)


if __name__ == "__main__":
    main()
