"""Read a written clip back, for the tests that check what a video shows."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def read_clip(path: Path) -> np.ndarray:
    """Every frame of the clip, as frames x height x width x 3."""
    size = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    width, height = (int(part) for part in size.strip().split(",")[:2])
    pixels = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-i", str(path),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ],
        check=True,
        capture_output=True,
    ).stdout
    return np.frombuffer(pixels, dtype=np.uint8).reshape(-1, height, width, 3)
