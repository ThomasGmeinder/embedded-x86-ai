"""How sharp, how noisy and how compressed are the recorded pictures against the scene's?

Run with .venv-sapien (needs the ffmpeg program), PYTHONPATH =
<workshop>/vla_testing/diagnostics, from the workshop folder:
    picture_quality.py measure

The model learned stacking from pictures that a real camera took and that
were then stored as H.264 video (High profile, colour at half resolution,
about 660 kbit/s for 640x480 at 30 pictures per second). The scene hands the
model pictures straight from the renderer. This script measures how the two
differ as pictures, apart from what they show.

Pictures: the moment 25 frames before the human lets the red cube go, 33
recordings, real and scene, top and wrist camera (release_aim_check.py).

Measures, each a median over the pictures of a set:
  strong edges     99th percentile of the brightness change per pixel
  flat-area noise  spread of the brightness around its neighbourhood where
                   the picture is flat (1.4826 x median absolute deviation)
  fine detail      share of the picture's brightness variation that lies at
                   patterns finer than four pixels per cycle
  colour edges     99th percentile of the change per pixel of the red-minus-
                   brightness colour channel

The scene's pictures are measured as they are and after a blur (Gaussian,
width in pixels) followed by an H.264 round trip at a constant rate factor
(x264's quality number: larger is coarser). The combination whose four
measures come closest to the real pictures' is reported for each camera.
Writes analysis/picture-quality.json.

Does it matter to the model? "variants" writes the scene's pictures of that
moment after some of those blurs and round trips, and the recorded pictures
blurred, to analysis/picture-quality-pictures.npz, with the pairs to ask
about. Then, with file names in analysis/:
    picture_quality.py variants
    release_aim_check.py predict picture-quality-pictures.npz picture-quality-predictions.npz   (.venv)
    release_aim_check.py report  picture-quality-pictures.npz picture-quality-predictions.npz picture-quality-release.json
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
BLURS = (0.0, 0.5, 0.75, 1.0, 1.5)
RATE_FACTORS = (None, 18, 23, 28, 33, 38)  # None: no video round trip
FLAT_CHANGE = 2.0  # brightness levels per pixel below which an area counts as flat
FINE_CYCLES_PER_PIXEL = 0.25
# "variants": (blur in pixels, rate factor or None) applied to both of the scene's pictures.
SCENE_AS_STORED = ((0.75, None), (0.0, 33), (0.5, 33), (0.75, 28), (1.5, 38))
ONE_CAMERA_AS_STORED = (0.5, 33)
REAL_BLUR = 1.0


def blurred(pictures: np.ndarray, width: float) -> np.ndarray:
    """Gaussian blur of every picture; width is the standard deviation in pixels."""
    if width == 0.0:
        return pictures
    return np.stack([cv2.GaussianBlur(picture, (0, 0), width) for picture in pictures])


def through_video(pictures: np.ndarray, rate_factor: int) -> np.ndarray:
    """Store the pictures as H.264 the way the recordings are stored, and read them back.

    Every picture is coded on its own (no picture is predicted from another),
    with colour at half resolution, at x264's constant rate factor.
    """
    count, height, width, _ = pictures.shape
    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / "pictures.h264"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{width}x{height}", "-r", "30", "-i", "-",
             "-c:v", "libx264", "-profile:v", "high", "-pix_fmt", "yuv420p",
             "-crf", str(rate_factor), "-x264-params", "keyint=1:scenecut=0",
             str(video)],
            input=np.ascontiguousarray(pictures).tobytes(), check=True)
        decoded = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video), "-f", "rawvideo",
             "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True).stdout
    back = np.frombuffer(decoded, dtype=np.uint8).reshape(-1, height, width, 3)
    if len(back) != count:
        raise RuntimeError(f"{count} pictures went into the video, {len(back)} came back")
    return back


def as_stored(pictures: np.ndarray, blur: float, rate_factor: int | None) -> np.ndarray:
    """Blur, then the video round trip."""
    pictures = blurred(pictures, blur)
    return pictures if rate_factor is None else through_video(pictures, rate_factor)


def measures(pictures: np.ndarray) -> dict[str, float]:
    """The four measures of the docstring, median over the pictures."""
    rows = []
    frequency = np.hypot(
        np.fft.fftfreq(pictures.shape[1])[:, None], np.fft.rfftfreq(pictures.shape[2])[None, :])
    for picture in pictures:
        luma_chroma = cv2.cvtColor(picture, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        brightness, red_minus_brightness = luma_chroma[..., 0], luma_chroma[..., 1]

        def change(channel: np.ndarray) -> np.ndarray:
            return np.hypot(
                cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3) / 8.0,
                cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3) / 8.0)

        smooth = cv2.GaussianBlur(brightness, (0, 0), 2.0)
        flat = change(smooth) < FLAT_CHANGE
        off = (brightness - smooth)[flat]
        power = np.abs(np.fft.rfft2(brightness - brightness.mean())) ** 2
        rows.append((
            np.percentile(change(brightness), 99),
            1.4826 * np.median(np.abs(off - np.median(off))),
            power[frequency > FINE_CYCLES_PER_PIXEL].sum() / power.sum(),
            np.percentile(change(red_minus_brightness), 99),
        ))
    strong, noise, fine, colour = np.median(np.array(rows), axis=0)
    return {
        "strong_edges_levels_per_px": round(float(strong), 1),
        "flat_area_noise_levels": round(float(noise), 2),
        "fine_detail_share_per_mille": round(1000.0 * float(fine), 3),
        "colour_edges_levels_per_px": round(float(colour), 1),
    }


def apart(one: dict[str, float], other: dict[str, float]) -> float:
    """How far two sets of measures are apart: sum of the ratios' logarithms, sizes only."""
    return float(sum(
        abs(np.log((one[name] + 0.05) / (other[name] + 0.05))) for name in one))


def measure() -> None:
    scene = np.load(ANALYSIS / "release-aim-pictures.npz")
    real = np.load(ANALYSIS / "release-aim-predictions.npz")
    result: dict[str, dict] = {"recordings": int(len(scene["episode"]))}
    for camera in ("top", "wrist"):
        recorded = measures(real[f"real_{camera}"])
        rows = {}
        for blur in BLURS:
            for rate_factor in RATE_FACTORS:
                label = f"blur {blur} px, " + (
                    "no video" if rate_factor is None else f"rate factor {rate_factor}")
                rows[label] = measures(as_stored(scene[camera], blur, rate_factor))
                rows[label]["apart_from_real"] = round(apart(recorded, {
                    name: rows[label][name] for name in recorded}), 2)
                print(camera, label, rows[label], flush=True)
        closest = min(rows, key=lambda label: rows[label]["apart_from_real"])
        result[camera] = {"real": recorded, "scene": rows, "closest_to_real": closest}
        print(camera, "real", recorded, "closest:", closest, flush=True)
    (ANALYSIS / "picture-quality.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


def variants() -> None:
    """Pictures of the release moment, stored as the recordings were, for release_aim_check.py."""
    scene = np.load(ANALYSIS / "release-aim-pictures.npz")
    real = np.load(ANALYSIS / "release-aim-predictions.npz")
    if not np.array_equal(scene["episode"], real["episode"]):
        raise RuntimeError("the scene's and the real pictures are not of the same recordings")
    pictures: dict[str, np.ndarray] = {}
    pairs: list[tuple[str, str, str]] = [
        ("real top + real wrist", "real_top", "real_wrist"),
        ("scene top + scene wrist", "top", "wrist"),
    ]
    for blur, rate_factor in SCENE_AS_STORED:
        how = f"blur {blur} px, " + (
            "no video" if rate_factor is None else f"rate factor {rate_factor}")
        for camera in ("top", "wrist"):
            pictures[f"{camera}, {how}"] = as_stored(scene[camera], blur, rate_factor)
        pairs.append((f"scene top + scene wrist, both {how}", f"top, {how}", f"wrist, {how}"))
    blur, rate_factor = ONE_CAMERA_AS_STORED
    how = f"blur {blur} px, rate factor {rate_factor}"
    pairs += [
        (f"scene top {how} + real wrist", f"top, {how}", "real_wrist"),
        (f"real top + scene wrist {how}", "real_top", f"wrist, {how}"),
    ]
    # How much the model minds a softer picture at all: the recorded pictures, blurred.
    for camera in ("top", "wrist"):
        pictures[f"recorded {camera}, blur {REAL_BLUR} px"] = blurred(
            real[f"real_{camera}"], REAL_BLUR)
    pairs.append((
        f"real top + real wrist, both blur {REAL_BLUR} px",
        f"recorded top, blur {REAL_BLUR} px", f"recorded wrist, blur {REAL_BLUR} px"))
    np.savez_compressed(
        ANALYSIS / "picture-quality-pictures.npz",
        **{name: scene[name] for name in scene.files},
        **pictures, pairs=np.array(pairs),
    )
    print(f"{len(pairs)} pairs for {len(scene['episode'])} recordings", flush=True)


if __name__ == "__main__":
    {"measure": measure, "variants": variants}[sys.argv[1]](*sys.argv[2:])
