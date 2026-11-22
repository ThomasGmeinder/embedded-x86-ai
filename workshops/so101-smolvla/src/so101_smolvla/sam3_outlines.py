"""Cube outlines from SAM 3 for the pictures SmolVLA plans on.

SAM 3 (Segment Anything Model 3) runs in a process of its own
(sam3_worker.py) and follows what it finds for the word "cube" through the
pictures of one layout, with one tracking session per camera. It outlines
things and does not say which cube an outline is; that is read here from the
colour inside the outline. An outline with no cube's colour inside is kept
apart and the picture editor does not paint by it.
"""

from __future__ import annotations

import importlib.metadata
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from multiprocessing import Pipe
from pathlib import Path

import numpy as np

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.sapien_config import IMAGE_SHAPE, OUTLINE_CODES

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ("top", "wrist")
# What the worker's process is given on top of this process's environment.
# Without MIOPEN_FIND_MODE=FAST the first call of SAM 3 fails in the graphics
# processor's convolution library ("No suitable algorithm was found",
# 22 Nov 2026). The setting is the worker's alone, so SmolVLA runs in this
# process as it does without SAM 3. The model files are read from the local
# cache only.
WORKER_ENVIRONMENT = {"MIOPEN_FIND_MODE": "FAST", "HF_HUB_OFFLINE": "1"}
# Share of an outline's pixels that must have one cube's colour for the
# outline to be that cube.
COLOUR_SHARE = 0.5


def cube_colours(pixels: np.ndarray) -> dict[str, np.ndarray]:
    """Which of ``pixels`` (rows of red, green, blue) have which cube's colour.

    The scene's cubes are red, green and a pale blue; the table, the bin and
    the jaws are grey and the arm is orange. A colour is read by its hue. The
    pale blue cube has a saturation of 0.12 and a bluish-grey part of the
    scene one of 0.09, so blue asks for at least 0.105.
    """
    if pixels.ndim != 2 or pixels.shape[1] != 3 or pixels.dtype != np.uint8:
        raise ValueError(f"invalid pixels {pixels.dtype} of shape {pixels.shape}")
    levels = pixels.astype(np.float64) / 255.0
    high, low = levels.max(axis=1), levels.min(axis=1)
    spread = np.maximum(high - low, 1e-9)
    saturation = (high - low) / np.maximum(high, 1e-9)
    red, green, blue = levels[:, 0], levels[:, 1], levels[:, 2]
    hue = 60.0 * np.where(
        high == red,
        ((green - blue) / spread) % 6,
        np.where(high == green, (blue - red) / spread + 2, (red - green) / spread + 4),
    )
    coloured = saturation > 0.06
    colours = {
        "red": coloured & ((hue < 15) | (hue > 345)),
        "green": coloured & (hue >= 90) & (hue < 170),
        "blue": (saturation >= 0.105) & (hue >= 180) & (hue < 260),
    }
    return {color: colours[color] for color in BIN_CUBE_COLORS}


def cube_colour(picture: np.ndarray, mask: np.ndarray) -> str | None:
    """The cube whose colour at least COLOUR_SHARE of the outlined pixels have."""
    pixels = picture[mask]
    if len(pixels) == 0:
        return None
    shares = {
        color: float(inside.mean()) for color, inside in cube_colours(pixels).items()
    }
    best = max(shares, key=shares.get)
    return best if shares[best] >= COLOUR_SHARE else None


@dataclass(frozen=True)
class FoundOutline:
    """One outline of SAM 3: its number for the tracked thing, its size, its cube."""

    number: int
    pixels: int
    cube: str | None


@dataclass(frozen=True)
class CameraOutlines:
    """What SAM 3 outlined in one picture.

    ``cubes`` is the map the picture editor paints by
    (sapien_config.OUTLINE_CODES): a cube's code where an outline has that
    cube's colour inside, "other" everywhere else. ``unused`` marks the pixels
    of the outlines with no cube's colour inside. ``seconds`` is the time SAM 3
    took for the picture.
    """

    cubes: np.ndarray
    unused: np.ndarray
    found: tuple[FoundOutline, ...]
    seconds: float


def read_outlines(
    picture: np.ndarray, masks: np.ndarray, numbers: list[int], seconds: float
) -> CameraOutlines:
    """Name SAM 3's outlines of ``picture`` by the colour inside each."""
    if picture.ndim != 3 or picture.shape[2] != 3 or picture.dtype != np.uint8:
        raise ValueError(f"invalid picture {picture.dtype} of shape {picture.shape}")
    if masks.ndim != 3 or masks.shape[1:] != picture.shape[:2] or masks.dtype != bool:
        raise ValueError(
            f"outlines are {masks.dtype} of shape {masks.shape}, "
            f"the picture is of shape {picture.shape}"
        )
    if len(numbers) != len(masks):
        raise ValueError(f"{len(masks)} outlines with {len(numbers)} numbers")
    found = tuple(
        FoundOutline(
            number=int(number), pixels=int(mask.sum()), cube=cube_colour(picture, mask)
        )
        for mask, number in zip(masks, numbers)
    )
    cubes = np.full(picture.shape[:2], OUTLINE_CODES["other"], dtype=np.uint8)
    unused = np.zeros(picture.shape[:2], dtype=bool)
    for mask, outline in zip(masks, found):
        if outline.cube is None:
            unused |= mask
    # A cube keeps every pixel of its outline, also where another outline
    # that is on no cube lies over it.
    for mask, outline in zip(masks, found):
        if outline.cube is not None:
            cubes[mask] = OUTLINE_CODES[outline.cube]
    unused &= cubes == OUTLINE_CODES["other"]
    return CameraOutlines(cubes=cubes, unused=unused, found=found, seconds=float(seconds))


class Sam3Outlines:
    """SAM 3 in its worker process, asked for the cube outlines of both cameras."""

    def __init__(self) -> None:
        request_end, self._requests = Pipe(duplex=False)
        self._replies, reply_end = Pipe(duplex=False)
        env = os.environ.copy()
        env.update(WORKER_ENVIRONMENT)
        source = str(ROOT / "src")
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = source if not existing else source + os.pathsep + existing
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "so101_smolvla.sam3_worker",
                str(request_end.fileno()),
                str(reply_end.fileno()),
            ],
            cwd=ROOT,
            env=env,
            pass_fds=(request_end.fileno(), reply_end.fileno()),
        )
        # The worker holds these two ends now. With this process's copies
        # closed, a worker that ends closes the pipe, and a wait for its
        # reply ends with it.
        request_end.close()
        reply_end.close()
        self._closed = False
        try:
            hello = self._reply()
        except Exception:
            self._terminate()
            raise
        self.provenance = {
            "model": hello["model"],
            "phrase": hello["phrase"],
            "device": hello["device"],
            "transformers": importlib.metadata.version("transformers"),
            "worker_environment": dict(WORKER_ENVIRONMENT),
        }

    def _reply(self) -> dict:
        try:
            reply = self._replies.recv()
        except EOFError:
            code = self._process.wait()
            self._closed = True
            raise RuntimeError(f"SAM 3 worker exited with status {code}") from None
        if "error" in reply:
            raise RuntimeError(f"SAM 3 worker: {reply['error']}")
        return reply

    def _ask(self, request: dict) -> dict:
        if self._closed:
            raise RuntimeError("SAM 3 worker is closed")
        try:
            self._requests.send(request)
        except BrokenPipeError:
            # The worker is gone; the reply pipe says so with its status.
            pass
        return self._reply()

    def start_layout(self) -> None:
        """Begin a new tracking session for each camera."""
        self._ask({"operation": "start"})

    def outline(
        self, observation: PolicyObservation
    ) -> tuple[PolicyObservation, dict[str, CameraOutlines]]:
        """The next pictures of the layout: the observation with SAM 3's cube maps.

        The second value holds, per camera, everything SAM 3 outlined.
        """
        for camera in CAMERAS:
            picture = getattr(observation, camera)
            if picture.shape != IMAGE_SHAPE or picture.dtype != np.uint8:
                raise ValueError(
                    f"{camera} picture is {picture.dtype} of shape {picture.shape}, "
                    f"expected uint8 of shape {IMAGE_SHAPE}"
                )
        reply = self._ask(
            {"operation": "outline", "top": observation.top, "wrist": observation.wrist}
        )
        found = {
            camera: read_outlines(
                getattr(observation, camera),
                reply[camera]["masks"],
                reply[camera]["numbers"],
                reply[camera]["seconds"],
            )
            for camera in CAMERAS
        }
        outlined = replace(
            observation,
            top_outline=found["top"].cubes,
            wrist_outline=found["wrist"].cubes,
        )
        return outlined, found

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._ask({"operation": "close"})
            code = self._process.wait()
            if code != 0:
                raise RuntimeError(f"SAM 3 worker exited with status {code}")
        except Exception:
            self._terminate()
            raise
        finally:
            self._closed = True
            self._requests.close()
            self._replies.close()

    def _terminate(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
        self._process.wait()
        self._closed = True
