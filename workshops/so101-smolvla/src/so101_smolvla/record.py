"""Record one successful grasp per distinct cube position into a LeRobot dataset.

Each kept episode gets its own directory. The combined
scripted_testing/episodes/all-top.mp4 and all-wrist.mp4 are written once, after the
last episode.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from so101_smolvla.task import TASK

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "scripted_testing" / "dataset"
RAW = ROOT / "scripted_testing" / "raw"
EPISODES = ROOT / "scripted_testing" / "episodes"

NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def _features() -> dict:
    image = {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    }
    vector = {"dtype": "float32", "shape": (6,), "names": NAMES}
    return {
        "observation.state": vector,
        "action": dict(vector),
        "observation.images.top": image,
        "observation.images.wrist": image,
    }


def load_episode(path: Path) -> dict[str, np.ndarray]:
    """Copy each array out of the compressed episode once.

    Looking up ``packed["top"][frame]`` decompresses the whole camera on every
    frame. The returned arrays are independent of the file.
    """
    with np.load(path) as packed:
        return {
            "state": np.array(packed["state"]),
            "action": np.array(packed["action"]),
            "top": np.array(packed["top"]),
            "wrist": np.array(packed["wrist"]),
        }


def _capture(index: int, xy: tuple[float, float], dest: Path) -> bool:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["PYTHONPATH"] = "src"
    env.pop("LD_PRELOAD", None)
    env.pop("LD_LIBRARY_PATH", None)
    completed = subprocess.run(
        [sys.executable, "-m", "so101_smolvla.capture", str(index), str(xy[0]), str(xy[1]), "0", str(dest)],
        cwd=ROOT,
        env=env,
        check=False,
    )
    return completed.returncode == 0 and dest.exists()


def _write_clip(frames: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "640x480", "-r", "30", "-i", "-",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert proc.stdin is not None
    for frame in frames:
        proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def _accumulate() -> None:
    for camera in ("top", "wrist"):
        folders = sorted(path for path in EPISODES.glob("episode-*") if (path / f"{camera}.mp4").exists())
        listing = EPISODES / f"all-{camera}.txt"
        listing.write_text(
            "".join(f"file '{(folder / f'{camera}.mp4').resolve()}'\n" for folder in folders),
            encoding="utf-8",
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(listing), "-c", "copy", str(EPISODES / f"all-{camera}.mp4"),
            ],
            check=True,
        )


def record() -> Path:
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_smolvla.sim import HOLD_S

    sim_cfg = yaml.safe_load((ROOT / "configs" / "sim.yaml").read_text())
    positions = [tuple(item) for item in sim_cfg["distinct_cube_positions"]]
    if DATASET.exists():
        raise SystemExit(f"{DATASET} already exists")
    dataset = LeRobotDataset.create(
        repo_id="local/so101_box",
        fps=30,
        features=_features(),
        root=DATASET,
        robot_type="so101",
        use_videos=True,
        rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="ultrafast"),
    )
    log_path = ROOT / "scripted_testing" / "cube_positions.jsonl"
    kept = 0
    try:
        for position in positions:
            dest = RAW / f"ep_{kept:03d}.npz"
            if not _capture(kept, position, dest):
                raise SystemExit(f"STOP: grasp failed at {position}")
            episode = load_episode(dest)
            count = int(episode["state"].shape[0])
            for index in range(count):
                dataset.add_frame(
                    {
                        "observation.state": episode["state"][index],
                        "action": episode["action"][index],
                        "observation.images.top": episode["top"][index],
                        "observation.images.wrist": episode["wrist"][index],
                        "task": TASK,
                    }
                )
            folder = EPISODES / f"episode-{kept:03d}"
            _write_clip(episode["top"], folder / "top.mp4")
            _write_clip(episode["wrist"], folder / "wrist.mp4")
            dataset.save_episode(parallel_encoding=False)
            report = {
                "episode_index": kept,
                "task": TASK,
                "cube_xy": [position[0], position[1]],
                "fps": 30,
                "frames": count,
                "grasped": True,
                "held_for_s": HOLD_S,
                "placed_in_box": True,
                "outcome": "success",
                "videos": ["top.mp4", "wrist.mp4"],
            }
            (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            dest.unlink()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"episode": kept, "cube_xy": [position[0], position[1]]}) + "\n")
            kept += 1
            print(f"kept {kept}/{len(positions)} cube {position[0]:.2f} {position[1]:.2f} frames {count}", flush=True)
    finally:
        dataset.finalize()
        if kept > 0:
            _accumulate()
    print(f"dataset {DATASET} episodes {kept}")
    return DATASET


if __name__ == "__main__":
    record()
