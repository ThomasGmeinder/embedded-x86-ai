"""Real top pictures of three moments of every stacking recording on disk.

Run with .venv (needs pandas and ffmpeg), PYTHONPATH = ~/.cache/so101-stack-training,
from the workshop folder:
    top_frames_for_calibration.py

Moments: the first frame (arm folded, both cubes free), 25 frames before the
human closes the gripper on the red cube (arm stretched out low), and the
frame the human lets the red cube go on the blue one (arm stretched out
higher). For the first frame this also stores where the two cubes are
(measure_objects.measure, the detector used for every earlier cube
measurement).

Writes analysis/top-frames-for-calibration.npz for calibrate_top_camera.py.
"""

from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from measure_objects import FPS, VIDEO_FILE, measure, real_first_frame

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
FRAMES_BEFORE_CLOSING = 25
MOMENTS = ("start", "before closing", "release")


def main() -> None:
    episodes = pd.concat(
        pd.read_parquet(path)
        for path in sorted(glob.glob(str(ROOT / "meta" / "episodes" / "chunk-000" / "*.parquet")))
    ).set_index("episode_index")
    places = np.load(ANALYSIS / "cube-placement.npz")
    column = "videos/observation.images.top"
    rows: dict[str, list] = {
        "episode": [], "frame": [], "moment": [], "top": [],
        "red_centre_px": [], "red_side_px": [], "blue_centre_px": [], "blue_side_px": [],
    }
    for number, close, release in zip(
        places["training_episode"].tolist(), places["training_close_frame"].tolist(),
        places["training_release_frame"].tolist(),
    ):
        row = episodes.loc[number]
        if int(row[f"{column}/file_index"]) != VIDEO_FILE or close <= FRAMES_BEFORE_CLOSING:
            continue
        for moment, frame in enumerate((0, int(close) - FRAMES_BEFORE_CLOSING, int(release))):
            picture = real_first_frame(
                "top", float(row[f"{column}/from_timestamp"]) + frame / FPS)
            rows["episode"].append(int(number))
            rows["frame"].append(frame)
            rows["moment"].append(moment)
            rows["top"].append(picture)
            for colour in ("red", "blue"):
                # A cube not found whole in the first frame is stored as "not a number".
                found = measure(picture, colour) if moment == 0 else None
                whole = found is not None and not found["touches_edge"]
                rows[f"{colour}_centre_px"].append(found["centre_px"] if whole else [np.nan, np.nan])
                rows[f"{colour}_side_px"].append(found["side_px"] if whole else np.nan)
    np.savez_compressed(
        ANALYSIS / "top-frames-for-calibration.npz",
        **{name: np.array(values) for name, values in rows.items()},
    )
    moment = np.array(rows["moment"])
    print(f"recordings: {len(set(rows['episode']))}, pictures: {len(moment)}", flush=True)
    for index, name in enumerate(MOMENTS):
        print(f"{name}: {int((moment == index).sum())} pictures", flush=True)
    for colour in ("red", "blue"):
        print(f"{colour} cube found whole in "
              f"{int(np.isfinite(np.array(rows[f'{colour}_side_px'])).sum())} first frames", flush=True)


if __name__ == "__main__":
    main()
