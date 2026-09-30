# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Hand-landmark inference (MediaPipe BlazeHand, 21 points) via ONNX Runtime.

Stage 2 of the manipulation demo. YOLO26-pose gives a coarse wrist/elbow anchor;
this module crops a hand ROI from it (``roi_from_wrist`` / ``track_roi``) and runs
the dedicated 21-point hand model on that crop for fine finger geometry (pinch /
open-closed). Runs on the CPU by default, or on the NPU when a VitisAI session is
passed in (as the workshop teleop path does).

Model: models/hand_landmark.onnx
    input  input_1   [1, 224, 224, 3]  float32, RGB, /255, NHWC
    outputs (ORT run order, as used by detect()/spread())
      out[0]   [1, 63]   21 x (x, y, z) in crop space
      out[1]   [1, 1]    hand-presence score
      out[2]   [1, 1]    handedness (left/right)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = _REPO_ROOT / "models" / "hand_landmark.onnx"
_MODEL_URL = (
    "https://huggingface.co/unity/inference-engine-blaze-hand/resolve/main/"
    "models/hand_landmarks_detector.onnx?download=true"
)
# Expected size of the downloaded model. Used to reject a truncated/partial
# download instead of silently trusting (and later hanging on) a corrupt file.
_MODEL_BYTES = 10_903_207

# BlazePose screen-landmark indices for each hand's anchors.
_HAND_ANCHORS = {
    "left": (15, 17, 19, 21),  # wrist, pinky, index, thumb
    "right": (16, 18, 20, 22),
}
_FINGERTIPS = (4, 8, 12, 16, 20)
_INPUT = 224


def _clamp_roi(cx, cy, size, frame_w, frame_h):
    """Square ROI from a center+size, clamped to never exceed the frame (the
    hard guard against a runaway box blowing up memory)."""
    size = float(max(24.0, min(size, max(frame_w, frame_h))))
    cx = float(min(max(cx, 0.0), frame_w))
    cy = float(min(max(cy, 0.0), frame_h))
    return int(round(cx - size / 2)), int(round(cy - size / 2)), int(round(size))


def track_roi(
    lm_xy: np.ndarray,
    frame_w: int,
    frame_h: int,
    expand: float = 1.8,
    min_size: float = 60.0,
):
    """Square ROI around the previous frame's landmarks (clamped to the frame).

    This landmark-guided crop is what makes hand tracking stable (the same trick
    MediaPipe uses): once we have a hand, we look for it where it just was."""
    mn, mx = lm_xy.min(axis=0), lm_xy.max(axis=0)
    c = (mn + mx) / 2.0
    size = max(float((mx - mn).max()) * expand, min_size)
    return _clamp_roi(c[0], c[1], size, frame_w, frame_h)


def roi_from_wrist(
    wrist_xy,
    elbow_xy,
    frame_w: int,
    frame_h: int,
    scale: float = 1.9,
    min_size: float = 70.0,
):
    """Re-acquisition ROI from YOLO's wrist+elbow keypoints (hand sits beyond
    the wrist along the forearm direction). Used when tracking is lost."""
    wrist_xy = np.asarray(wrist_xy, float)
    elbow_xy = np.asarray(elbow_xy, float)
    fore = float(np.linalg.norm(wrist_xy - elbow_xy))
    size = max(fore * scale, min_size)
    d = wrist_xy - elbow_xy
    d = d / (np.linalg.norm(d) + 1e-6)
    c = wrist_xy + d * size * 0.30
    return _clamp_roi(c[0], c[1], size, frame_w, frame_h)


def ensure_hand_model(path: Path, *, timeout: float = 30.0, retries: int = 3) -> Path:
    """Make sure the hand-landmark ONNX exists at ``path``, downloading it if needed.

    Hardened for live-workshop use (the bare ``urlretrieve`` this replaces could
    hang forever on a stalled network and leave a corrupt half-file behind):

      * streams with a ``timeout`` so a dead connection raises instead of hanging;
      * downloads to a ``.part`` temp file and atomically moves it into place, so an
        interrupted run never leaves a broken final file;
      * verifies the finished size against ``_MODEL_BYTES`` (and re-downloads an
        existing file whose size is wrong -- the classic "worked once, now stuck");
      * prints progress so a slow download reads as working, not frozen;
      * retries a few times, then raises a clear, actionable error.
    """
    path = Path(path)

    # Offline mode (the workshop default): never touch the network. If the model
    # is already staged with the right size, return it; otherwise fail fast with a
    # clear, actionable message instead of attempting a download that would hang.
    if os.environ.get("WORKSHOP_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        if path.exists() and path.stat().st_size == _MODEL_BYTES:
            return path
        raise FileNotFoundError(
            f"Offline mode (WORKSHOP_OFFLINE): required model not found or wrong size "
            f"at {path}. Stage the {_MODEL_BYTES}-byte hand_landmark.onnx on this "
            f"machine (no download will be attempted)."
        )

    if path.exists() and path.stat().st_size == _MODEL_BYTES:
        return path
    if path.exists():
        print(
            f"[hand_landmarks] existing model at {path} has the wrong size "
            f"({path.stat().st_size} bytes, expected {_MODEL_BYTES}) - re-downloading."
        )

    import urllib.error
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[hand_landmarks] downloading hand-landmark model "
                f"(attempt {attempt}/{retries}) -> {path} ..."
            )
            with urllib.request.urlopen(_MODEL_URL, timeout=timeout) as resp:
                total = int(resp.headers.get("Content-Length") or _MODEL_BYTES)
                done = 0
                next_pct = 10
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        pct = int(done * 100 / total) if total else 0
                        if pct >= next_pct:
                            print(f"  {pct:3d}%  ({done // 1024} / {total // 1024} KB)")
                            next_pct += 10
            size = tmp.stat().st_size
            if size != _MODEL_BYTES:
                raise IOError(
                    f"downloaded {size} bytes, expected {_MODEL_BYTES} "
                    "(truncated/incomplete download)"
                )
            os.replace(tmp, path)  # atomic: no half-written final file
            print("[hand_landmarks] done.")
            return path
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            last_err = exc
            print(f"[hand_landmarks] attempt {attempt} failed: {exc}")
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass
            if attempt < retries:
                time.sleep(2.0)

    raise RuntimeError(
        f"Could not download the hand-landmark model after {retries} attempts "
        f"(last error: {last_err}).\n"
        f"Check your network and re-run, or download it manually to:\n"
        f"    {path}\n"
        f"from: {_MODEL_URL}"
    )


# Back-compat alias: older cells/imports call ``_ensure_model``.
_ensure_model = ensure_hand_model


def hand_roi(screen: np.ndarray, vis: np.ndarray, side: str, vis_thresh: float = 0.4):
    """Square hand ROI (x0, y0, side_px) from BlazePose anchors, or None."""
    idx = _HAND_ANCHORS[side]
    pts = screen[list(idx), :2]
    v = vis[list(idx)]
    if v[0] < vis_thresh or (v >= vis_thresh).sum() < 2:
        return None
    p = pts[v >= vis_thresh]
    cx, cy = float(p[:, 0].mean()), float(p[:, 1].mean())
    wrist = screen[idx[0], :2]
    reach = float(np.linalg.norm(pts - wrist, axis=1).max())
    side_px = max(reach * 3.2, 40.0)
    return (
        int(round(cx - side_px / 2)),
        int(round(cy - side_px / 2)),
        int(round(side_px)),
    )


class HandLandmarker:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        closed_thresh: float = 1.6,
        session: "onnxruntime.InferenceSession | None" = None,
    ):
        """21-point hand landmarker.

        Pass a prebuilt ``session`` (e.g. a VitisAI/NPU session from
        ``ep_factory.make_npu_session``) to run the model on the NPU; the graph
        is a clean MobileNet-style CNN that offloads 100% to the AIE. When no
        session is given we fall back to a CPU session on ``model_path``.
        """
        if session is not None:
            self.session = session
        else:
            model_path = Path(model_path)
            ensure_hand_model(model_path)
            self.session = onnxruntime.InferenceSession(
                str(model_path), providers=["CPUExecutionProvider"]
            )
        self._input = self.session.get_inputs()[0].name
        self.closed_thresh = closed_thresh

    def spread(self, frame_bgr: np.ndarray, roi) -> tuple[float, float]:
        """Return (fingertip_spread, score) for a hand ROI. spread < thresh => fist."""
        x0, y0, s = roi
        h, w = frame_bgr.shape[:2]
        crop = np.zeros((s, s, 3), dtype=np.uint8)
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x0 + s), min(h, y0 + s)
        if cx1 <= cx0 or cy1 <= cy0:
            return 99.0, 0.0
        crop[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = frame_bgr[cy0:cy1, cx0:cx1]
        inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (_INPUT, _INPUT))
        inp = (inp.astype(np.float32) / 255.0)[None]
        out = self.session.run(None, {self._input: inp})
        lm = out[0].reshape(21, 3)
        score = float(1.0 / (1.0 + np.exp(-np.clip(out[2].reshape(-1)[0], -30, 30))))
        palm = np.linalg.norm(lm[9] - lm[0]) + 1e-6
        spread = float(
            np.mean([np.linalg.norm(lm[t] - lm[0]) for t in _FINGERTIPS]) / palm
        )
        return spread, score

    def detect(self, frame_bgr: np.ndarray, roi, presence_thresh: float = 0.4):
        """Run the 21-landmark model on a hand ROI.

        Returns ``(lm_crop, lm_xy_frame, presence)`` where ``lm_crop`` is (21, 3)
        in crop space (for scale-invariant angle mapping) and ``lm_xy_frame`` is
        (21, 2) in full-frame pixels (for tracking + overlay), or ``None`` if no
        confident, geometrically plausible hand is found.

        Presence is ``out[1]`` (the blaze-hand flag); ``out[2]`` is handedness.
        Clipping it to [0, 1] makes the gate work whether the model emits a
        probability or a logit. The geometry check (landmarks inside the crop,
        sane palm size) is what prevents a garbage detection from growing the
        tracking box without bound.
        """
        x0, y0, s = roi
        h, w = frame_bgr.shape[:2]
        s = int(min(s, max(w, h)))  # hard cap (never allocate > frame)
        if s <= 8:
            return None
        crop = np.zeros((s, s, 3), dtype=np.uint8)
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(w, x0 + s), min(h, y0 + s)
        if cx1 <= cx0 or cy1 <= cy0:
            return None
        crop[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = frame_bgr[cy0:cy1, cx0:cx1]
        inp = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (_INPUT, _INPUT))
        inp = (inp.astype(np.float32) / 255.0)[None]
        out = self.session.run(None, {self._input: inp})
        lm = out[0].reshape(21, 3).astype(np.float32)
        presence = float(np.clip(out[1].reshape(-1)[0], 0.0, 1.0))
        if presence < presence_thresh:
            return None
        xy = lm[:, :2]
        inside = float(((xy > -0.3 * _INPUT) & (xy < 1.3 * _INPUT)).all(axis=1).mean())
        palm = float(np.linalg.norm(lm[9] - lm[0]))
        if inside < 0.8 or not (10.0 < palm < 1.5 * _INPUT):
            return None
        xy = np.clip(xy, 0.0, _INPUT)  # clamp before mapping -> no runaway
        lm_xy = np.empty_like(xy)
        lm_xy[:, 0] = x0 + xy[:, 0] / _INPUT * s
        lm_xy[:, 1] = y0 + xy[:, 1] / _INPUT * s
        return lm, lm_xy, presence

    def grip(self, frame_bgr, screen, vis, side, score_thresh: float = 0.4):
        """True (closed) / False (open) / None (no confident hand) for one side."""
        roi = hand_roi(screen, vis, side)
        if roi is None:
            return None
        spread, score = self.spread(frame_bgr, roi)
        if score < score_thresh:
            return None
        return spread < self.closed_thresh
