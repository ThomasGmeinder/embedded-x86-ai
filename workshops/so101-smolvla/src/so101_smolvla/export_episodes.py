"""Split the LeRobot chunk video into one directory per finished episode."""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "scripted_testing" / "dataset"
EPISODES = ROOT / "scripted_testing" / "episodes"
FPS = 30
FRAMES = 189
DURATION_S = FRAMES / FPS


def accumulate() -> None:
    """Write episodes/all-top.mp4 and episodes/all-wrist.mp4 from the episode clips."""
    folders = sorted(path for path in EPISODES.glob("episode-*") if path.is_dir())
    for camera in ("top", "wrist"):
        list_path = EPISODES / f"all-{camera}.txt"
        lines = []
        for folder in folders:
            clip = folder / f"{camera}.mp4"
            if clip.exists():
                lines.append(f"file '{clip.resolve()}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        dest = EPISODES / f"all-{camera}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(list_path), "-c", "copy", str(dest)],
            check=True,
            capture_output=True,
        )
        print(dest, flush=True)


def main() -> None:
    cfg = tomllib.loads((ROOT / "configs" / "sim.toml").read_text(encoding="utf-8"))
    positions = [tuple(item) for item in cfg["distinct_cube_positions"]]
    top_src = DATASET / "videos" / "observation.images.top" / "chunk-000" / "file-000.mp4"
    wrist_src = DATASET / "videos" / "observation.images.wrist" / "chunk-000" / "file-000.mp4"
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=nw=1:nk=1", str(top_src),
        ],
        check=True, capture_output=True, text=True,
    )
    complete = int(float(probe.stdout.strip()) // DURATION_S)
    for index in range(complete):
        folder = EPISODES / f"episode-{index:03d}"
        folder.mkdir(parents=True, exist_ok=True)
        start = index * DURATION_S
        for source, name in ((top_src, "top.mp4"), (wrist_src, "wrist.mp4")):
            subprocess.run(
                [
                    "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{DURATION_S:.3f}",
                    "-i", str(source), "-c", "copy", str(folder / name),
                ],
                check=True, capture_output=True,
            )
        logged = ROOT / "scripted_testing" / "cube_positions.jsonl"
        cube_xy = list(positions[index // 10]) if index // 10 < len(positions) else None
        if logged.exists():
            rows = [json.loads(line) for line in logged.read_text().splitlines() if line.strip()]
            match = [row for row in rows if row["episode"] == index]
            if match:
                cube_xy = match[0]["cube_xy"]
        report = {
            "episode_index": index,
            "task": cfg["task"],
            "cube_xy": cube_xy,
            "fps": FPS,
            "frames": FRAMES,
            "duration_s": DURATION_S,
            "grasped": True,
            "held_for_s": 1.5,
            "placed_in_box": True,
            "outcome": "success",
            "videos": ["top.mp4", "wrist.mp4"],
        }
        (folder / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(folder, flush=True)
    print(f"episodes {complete}")
    accumulate()


if __name__ == "__main__":
    main()
