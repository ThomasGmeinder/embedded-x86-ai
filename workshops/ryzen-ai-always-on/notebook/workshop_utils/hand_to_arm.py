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

"""Map MediaPipe hand landmarks -> SO-101 tele-op command.

Two clean image-plane signals drive the arm by default, each smoothed downstream
by a per-axis One-Euro filter:

* **left / right** — palm-centre X drives the gripper laterally (robot Y).
* **up / down** — palm-centre Y drives the gripper height (robot Z).
* **pinch** — thumb-tip to index-tip distance (scale-normalised by palm size, so
  it is depth-invariant) closes the gripper to **grab** the cube.

These two axes are the reliable default: they read straight off the image plane,
so they never cross-couple or saturate. Monocular depth is the opposite — noisy,
it cross-talks with the other axes, and it buys nothing here (the cube and hole
sit at the same depth), so **reach is OFF by default** and the tool tip is pinned
at ``X_REACH``.

* **near / far** (opt-in, ``enable_depth=True``) — the apparent hand size
  (wrist->knuckle pixel span) drives reach (robot X): move your hand toward the
  camera and the span grows, so the arm reaches forward; pull it back and it
  retracts. Auto-calibrated to a neutral span captured at start. Experimental —
  it jitters and can cross-couple with left/right and up/down.
"""

from __future__ import annotations

import numpy as np

from .arm_teleop_sim import (
    X_REACH,
    X_NEAR,
    X_FAR,
    Y_LEFT,
    Y_RIGHT,
    Z_TOP,
    Z_BOTTOM,
    Z_CARRY,
)

# Active region of the frame (margins ignored so you needn't reach the edges).
# Kept comfortably inside the frame so the lateral endpoints (the cube at y=-0.13
# and hole at y=+0.13, which sit at the very ends of the band) are reachable with
# margin — you don't have to jam your hand to the frame edge to pick or place.
_X_LO, _X_HI = 0.28, 0.72  # horizontal band -> lateral (robot Y)
_Y_LO, _Y_HI = 0.22, 0.80  # vertical band   -> height  (robot Z)
# Depth (reach) sensitivity: the fraction the hand span must grow/shrink from its
# start neutral to sweep the full reach band (0.35 = +/-35% span = full sweep).
DEPTH_SENSITIVITY = 0.35
# Frames of valid detection averaged into the neutral span at start (then held).
_DEPTH_WARMUP = 6
# Pinch normalised distance: open hand ~0.7+, full pinch ~0.15.
# Two thresholds (hysteresis): grip CLOSES below GRAB, only OPENS above RELEASE.
# The gap between them stops landmark noise from flickering the grip open and
# dropping the carried cube.
PINCH_GRAB = 0.40  # below this thumb-index gap -> grip closes (grab)
PINCH_RELEASE = 0.55  # above this -> grip opens (release)
_PINCH_GRAB = PINCH_GRAB  # back-compat alias
_PALM = (0, 9)  # wrist -> middle-finger MCP (palm length)
_THUMB_TIP, _INDEX_TIP = 4, 8
_PALM_PTS = (0, 5, 9, 13, 17)


def _lerp(a, b, t):
    return a + (b - a) * float(np.clip(t, 0.0, 1.0))


def _remap(v, lo, hi):
    return (float(v) - lo) / (hi - lo)


def pinch_norm(lm_crop: np.ndarray) -> float:
    """Thumb-to-index distance over palm length (scale/depth-invariant)."""
    xy = np.asarray(lm_crop, float)[:, :2]
    palm = float(np.linalg.norm(xy[_PALM[1]] - xy[_PALM[0]])) + 1e-6
    return float(np.linalg.norm(xy[_THUMB_TIP] - xy[_INDEX_TIP]) / palm)


def hand_span(lm_xy: np.ndarray) -> float:
    """Apparent hand size in frame pixels (wrist -> middle-finger MCP).

    Grows as the hand nears the camera and shrinks as it recedes, so it is a
    monocular depth proxy. Uses the palm bone (indices 0 and 9), which barely
    moves when the fingers do — so pinching to grab does not disturb the reading.
    """
    xy = np.asarray(lm_xy, float)
    return float(np.linalg.norm(xy[_PALM[1]] - xy[_PALM[0]]))


class TeleopMapper:
    """Stateful hand -> SO-101 target mapper.

    Left/right (robot Y) and up/down (robot Z) are stateless per-frame remaps and
    are the reliable default. Reach/depth (robot X) is opt-in (``enable_depth``):
    when off, the tip is pinned at ``x_mid`` (``X_REACH``) so reach can never
    jitter or saturate; when on, it is *relative* to a neutral hand span captured
    over the first few frames, self-calibrating to the user's hand size/distance.
    Create one per demo, call :meth:`command` each frame, and call :meth:`reset`
    to recapture the neutral (e.g. when switching which hand is in control).
    """

    def __init__(
        self,
        x_near: float = X_NEAR,
        x_far: float = X_FAR,
        x_mid: float = X_REACH,
        z_top: float = Z_TOP,
        z_bottom: float = Z_BOTTOM,
        depth_sensitivity: float = DEPTH_SENSITIVITY,
        warmup: int = _DEPTH_WARMUP,
        enable_depth: bool = False,
    ):
        # Depth (reach) is opt-in; off by default it is pinned at x_mid so it can
        # never jitter, cross-couple, or saturate the reach band.
        self.enable_depth = bool(enable_depth)
        self.x_near = float(x_near)
        self.x_far = float(x_far)
        # Neutral-depth reach (tip sits here at rest); kept inside the band so a
        # relaxed hand hovers at the cube/pad depth, with a longer forward throw.
        self.x_mid = float(min(max(x_mid, min(x_near, x_far)), max(x_near, x_far)))
        self.z_top = float(z_top)
        self.z_bottom = float(z_bottom)
        self.sensitivity = max(float(depth_sensitivity), 1e-3)
        self.warmup = max(int(warmup), 1)
        self._span_ref = None  # neutral hand span (px); set on the first frame
        self._seen = 0

    def reset(self) -> None:
        """Forget the neutral span so the next frames re-calibrate the reach."""
        self._span_ref = None
        self._seen = 0

    def _neutral(self, span: float) -> float:
        """Running-mean the first ``warmup`` spans into the neutral, then hold."""
        if self._span_ref is None:
            self._span_ref = span
            self._seen = 1
        elif self._seen < self.warmup:
            self._seen += 1
            self._span_ref += (span - self._span_ref) / self._seen
        return self._span_ref

    def command(self, lm_crop: np.ndarray, lm_xy: np.ndarray, w: int, h: int):
        """Return ``(target_xyz, pinch, grip)`` for one frame of hand landmarks.

        ``target_xyz`` is the world tool-tip goal for IK: lateral (Y) from palm X
        and height (Z) from palm Y (the reliable default). Reach (X) is pinned at
        ``x_mid`` unless ``enable_depth`` is set, in which case it follows the
        hand span vs. the start neutral. ``grip`` is True when the pinch is closed
        past the grab threshold. All axes are One-Euro smoothed by the caller.
        """
        palm_px = np.asarray(lm_xy, float)[list(_PALM_PTS)].mean(axis=0)
        # left/right -> robot Y (lateral)
        nx = _remap(palm_px[0] / max(w, 1), _X_LO, _X_HI)
        ty = _lerp(Y_LEFT, Y_RIGHT, nx)
        # up/down -> robot Z (height)
        ny = _remap(palm_px[1] / max(h, 1), _Y_LO, _Y_HI)
        tz = _lerp(self.z_top, self.z_bottom, ny)
        # near/far -> robot X (reach). OFF by default: pin the tip at x_mid
        # (X_REACH, the cube/pad depth) and skip the hand-span/ratio math
        # entirely, so reach can never jitter, cross-couple, or saturate.
        if not self.enable_depth:
            tx = self.x_mid
        else:
            # relative to the start-neutral span and centred on x_mid: at rest
            # (depth 0.5) the tip sits at the cube/pad depth, pushing toward the
            # camera reaches out to x_far, pulling back retracts to x_near
            # (asymmetric so the forward throw is generous).
            span = hand_span(lm_xy)
            ref = self._neutral(span)
            ratio = span / (ref + 1e-6)
            depth = float(
                np.clip(
                    _remap(ratio, 1.0 - self.sensitivity, 1.0 + self.sensitivity),
                    0.0,
                    1.0,
                )
            )
            if depth >= 0.5:
                tx = self.x_mid + (depth - 0.5) * 2.0 * (self.x_far - self.x_mid)
            else:
                tx = self.x_mid - (0.5 - depth) * 2.0 * (self.x_mid - self.x_near)

        p = pinch_norm(lm_crop)
        grip = p < _PINCH_GRAB
        return np.array([tx, ty, tz]), p, grip


def hand_to_command(
    lm_crop: np.ndarray, lm_xy: np.ndarray, w: int, h: int, carry_z: float = Z_CARRY
):
    """Stateless single-frame mapper (back-compat wrapper around ``TeleopMapper``).

    Drives lateral (Y) and height (Z); reach (X) stays pinned at ``X_REACH``
    because depth is off by default. For the opt-in in/out reach axis build a
    persistent :class:`TeleopMapper` with ``enable_depth=True``. ``carry_z`` is
    accepted for signature compatibility but the height now follows the hand
    rather than being pinned.
    """
    return TeleopMapper().command(lm_crop, lm_xy, w, h)
