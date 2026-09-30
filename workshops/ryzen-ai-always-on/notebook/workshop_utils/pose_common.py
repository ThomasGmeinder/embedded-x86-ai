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

"""Shared helpers for the live demos: pose postprocess, skeleton draw, camera
source, One-Euro smoothing, and a drop-old queue put.

Kept dependency-free (only cv2 / numpy) so the live demos can reuse it without
pulling in any renderer or model code.
"""

from __future__ import annotations

import glob
import grp
import math
import os
from queue import Empty, Queue
from typing import Optional

import cv2
import numpy as np

# COCO-17 skeleton edges.
SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),  # torso
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),  # head
]
KPT_VIS_THRESHOLD = 0.3


class _OneEuro:
    """One-Euro filter: low jitter when still, low lag when moving fast.

    See Casiez et al. 2012. ``min_cutoff`` sets the smoothing floor (lower =
    steadier at rest), ``beta`` how aggressively it speeds up on fast motion.
    """

    def __init__(
        self, min_cutoff: float = 1.0, beta: float = 0.02, dcutoff: float = 1.0
    ):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, t: float, x: float) -> float:
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            return x
        dt = max(t - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.dcutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


def postprocess_pose(outputs, original_shape, info, conf_threshold=0.4):
    """Decode the NMS-free [1, 300, 57] pose head into boxes/scores/keypoints."""
    preds = outputs[0][0]  # [300, 57]
    boxes = preds[:, :4].copy()
    scores = preds[:, 4]
    kpts = preds[:, 6:].reshape(-1, 17, 3).copy()

    mask = scores > conf_threshold
    boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]
    if len(boxes) == 0:
        return boxes, scores, kpts

    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y
    boxes /= info.ratio
    kpts[..., 0] -= info.pad_x
    kpts[..., 1] -= info.pad_y
    kpts[..., :2] /= info.ratio

    h, w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes, scores, kpts


def draw_skeleton(frame, person_kpts, glow=True):
    """Draw a COCO-17 skeleton. With ``glow=True`` (default) a blurred bright
    underlay makes the pose the visual headline; ``glow=False`` skips it for a
    flatter look that matches the plain live-webcam feed (no brightening)."""
    h = frame.shape[0]
    bone_w = max(3, int(h / 120))  # scale line weight to the frame
    joint_r = max(4, int(h / 110))

    if glow:
        glow_img = np.zeros_like(frame)
        for a, b in SKELETON:
            xa, ya, va = person_kpts[a]
            xb, yb, vb = person_kpts[b]
            if va > KPT_VIS_THRESHOLD and vb > KPT_VIS_THRESHOLD:
                cv2.line(
                    glow_img,
                    (int(xa), int(ya)),
                    (int(xb), int(yb)),
                    (255, 220, 40),
                    bone_w * 3,
                    cv2.LINE_AA,
                )
        glow_img = cv2.GaussianBlur(glow_img, (0, 0), bone_w * 2)
        cv2.add(frame, glow_img, frame)

    for a, b in SKELETON:
        xa, ya, va = person_kpts[a]
        xb, yb, vb = person_kpts[b]
        if va > KPT_VIS_THRESHOLD and vb > KPT_VIS_THRESHOLD:
            cv2.line(
                frame,
                (int(xa), int(ya)),
                (int(xb), int(yb)),
                (255, 230, 90),
                bone_w,
                cv2.LINE_AA,
            )
    for x, y, v in person_kpts:
        if v > KPT_VIS_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), joint_r, (40, 60, 255), -1, cv2.LINE_AA)
            cv2.circle(
                frame, (int(x), int(y)), joint_r, (255, 255, 255), 1, cv2.LINE_AA
            )
    return frame


# Full ultralytics default COCO-17 skeleton (19 bones) + palette (BGR): blue arms,
# orange legs, magenta torso, green face. Matches the notebook's draw_poses exactly
# so the pose styling is identical throughout the workshop.
_POSE_BONES = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),  # arms
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),  # legs
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),  # shoulders, hips, torso sides
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 4),  # face
    (3, 5),
    (4, 6),  # ears to shoulders
]
_LIMB_COLORS = [
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (51, 153, 255),
    (51, 153, 255),
    (51, 153, 255),
    (51, 153, 255),
    (255, 128, 0),
    (255, 51, 255),
    (255, 51, 255),
    (255, 51, 255),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
    (0, 255, 0),
]
_KPT_COLORS = [(0, 255, 0)] * 5 + [(255, 128, 0)] * 6 + [(51, 153, 255)] * 6


def draw_pose_colored(frame, person_kpts):
    """Draw the full 19-bone COCO-17 skeleton in the ultralytics default palette
    (blue arms, orange legs, magenta torso, green face), flat (no glow). Matches
    the notebook's draw_poses so pose styling is identical throughout the workshop."""
    for (a, b), color in zip(_POSE_BONES, _LIMB_COLORS):
        xa, ya, va = person_kpts[a]
        xb, yb, vb = person_kpts[b]
        if va > KPT_VIS_THRESHOLD and vb > KPT_VIS_THRESHOLD:
            cv2.line(
                frame, (int(xa), int(ya)), (int(xb), int(yb)), color, 2, cv2.LINE_AA
            )
    for (x, y, v), color in zip(person_kpts, _KPT_COLORS):
        if v > KPT_VIS_THRESHOLD:
            cv2.circle(frame, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return frame


# Ultralytics default COCO-17 pose palette (BGR). Colors are per-bone / per-keypoint:
# blue arms, orange legs, magenta torso, green face — matches ultralytics' plot().
# Public aliases of the module palette so the notebook can share the exact colors.
LIMB_COLORS = _LIMB_COLORS
KPT_COLORS = _KPT_COLORS


def draw_poses(
    frame,
    boxes,
    scores,
    kpts,
    kpt_vis=KPT_VIS_THRESHOLD,
    show_boxes=True,
    show_skeleton=True,
    show_keypoints=True,
):
    """Annotate a frame with person boxes, the COCO-17 skeleton, and keypoints.

    Canonical pose overlay for the workshop (notebook Section 4 + the live demos):
    ultralytics default palette (blue arms, orange legs, magenta torso, green
    face) with a filled label chip, and per-part visibility gating at ``kpt_vis``.
    ``show_boxes`` / ``show_skeleton`` / ``show_keypoints`` toggle each layer so
    the live sliders can turn parts on/off without an NPU re-run.

    Self-contained: uses the module's own ``SKELETON`` + palette, so it has no
    notebook-global dependencies."""
    out = frame.copy()
    BOX_COLOR = (255, 42, 4)  # ultralytics default person-box color (BGR); dark blue
    for box, score, person_kpts in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = box.astype(int)
        if show_boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2, cv2.LINE_AA)
            # Filled label chip with white text (matches ultralytics box_label).
            label = f"person {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            outside = y1 - th - 3 >= 0
            y2c = y1 - th - 3 if outside else y1 + th + 3
            cv2.rectangle(out, (x1, y1), (x1 + tw, y2c), BOX_COLOR, -1, cv2.LINE_AA)
            ty = y1 - 2 if outside else y1 + th + 2
            cv2.putText(
                out,
                label,
                (x1, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        if show_skeleton:
            for (a, b), color in zip(SKELETON, LIMB_COLORS):
                xa, ya, va = person_kpts[a]
                xb, yb, vb = person_kpts[b]
                if va > kpt_vis and vb > kpt_vis:
                    cv2.line(
                        out,
                        (int(xa), int(ya)),
                        (int(xb), int(yb)),
                        color,
                        2,
                        cv2.LINE_AA,
                    )
        if show_keypoints:
            for (x, y, v), color in zip(person_kpts, KPT_COLORS):
                if v > kpt_vis:
                    cv2.circle(out, (int(x), int(y)), 4, color, -1, cv2.LINE_AA)
    return out


def _camera_diagnosis(index: int) -> str:
    """Build an actionable error message when a webcam won't open on Linux.

    The most common cause in a fresh workshop setup is that the user is not in
    the ``video`` group, so V4L2 sees ``/dev/video*`` but can't open it (the
    cryptic ``cap_v4l.cpp:914 can't open camera by index`` warning)."""
    devices = sorted(glob.glob("/dev/video*"))
    if not devices:
        return (
            f"could not open camera index {index}: no /dev/video* devices "
            f"found. Is a webcam plugged in?"
        )

    in_video_group = "video" in [grp.getgrgid(gid).gr_name for gid in os.getgroups()]
    readable = [d for d in devices if os.access(d, os.R_OK | os.W_OK)]

    lines = [
        f"could not open camera index {index}.",
        f"  detected devices: {', '.join(devices)}",
    ]
    if not readable:
        lines += [
            "  cause: no permission to open the camera device. On Linux, webcam",
            "         access is granted to the user logged in at the physical screen",
            "         (via systemd-logind seat ACLs).",
            "  fix:   log in at the machine's desktop, then run the demo from there.",
        ]
        if not in_video_group:
            lines += [
                "         (headless/SSH only) or add yourself to the 'video' group:",
                "         sudo usermod -aG video $USER   # then log out/in",
            ]
    else:
        lines += [
            "  the device is accessible but failed to open — it may be in use by",
            "  another process, or this index is a metadata node (try index 0).",
        ]
    return "\n".join(lines)


def open_camera(
    index: int = 0, width: int = 1280, height: int = 720
) -> cv2.VideoCapture:
    """Open the webcam at MJPG (30 fps) — without MJPG the C920 falls back to
    10 fps YUYV at 720p and looks like the NPU is the bottleneck."""
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def _camera_index_candidates(preferred: int) -> list[int]:
    """Try the requested index first, then every other /dev/videoN node.

    A single UVC webcam exposes two nodes (capture + metadata), and the capture
    node is not always index 0 (e.g. it enumerates as /dev/video2). Scanning the
    real device nodes lets the demo find the camera regardless of its index.

    Only indices that actually have a /dev/video* node are returned, so we don't
    probe (and emit scary V4L2 warnings for) nonexistent devices like video0."""
    present = []
    for dev in sorted(glob.glob("/dev/video*")):
        try:
            present.append(int(dev.replace("/dev/video", "")))
        except ValueError:
            continue
    candidates = [preferred] if preferred in present else []
    candidates += [idx for idx in present if idx not in candidates]
    return candidates


def _open_first_working_camera(preferred: int, width: int = 1280, height: int = 720):
    """Open the first camera index that both opens AND yields a frame, so we
    skip metadata-only nodes. Returns an opened VideoCapture or None.

    ``width``/``height`` set the capture resolution; a lower capture size cuts
    the per-frame MJPG decode (and downstream draw/encode) cost, which is the
    main lever for live fps once the NPU is in turbo (the model letterboxes to
    640 internally either way)."""
    for idx in _camera_index_candidates(preferred):
        cap = open_camera(idx, width, height)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    return None


def make_frame_source(source=0, width: int = 1280, height: int = 720):
    """Return (frame_source_callable, cap). ``source`` is a camera index (int)
    or a path to a video file (str/Path). Video files loop forever.

    For a camera index, the requested index is tried first; if it fails, the
    other /dev/video* nodes are scanned so the demo works even when the camera
    is not at index 0. ``width``/``height`` set the camera capture resolution
    (ignored for video files)."""
    is_file = not isinstance(source, int)
    if is_file:
        cap = cv2.VideoCapture(str(source))
    else:
        cap = _open_first_working_camera(source, width, height)

    if cap is None or not cap.isOpened():
        if is_file:
            raise RuntimeError(f"could not open video file: {source!r}")
        raise RuntimeError(_camera_diagnosis(int(source)))

    def frame_source() -> Optional[np.ndarray]:
        ok, frame = cap.read()
        if not ok:
            if is_file:  # loop the demo video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                return None
        return frame

    return frame_source, cap


def _put_dropping(q: Queue, item) -> None:
    """Keep only the latest item — replace rather than block/backlog."""
    try:
        q.put_nowait(item)
    except Exception:
        try:
            q.get_nowait()
        except Empty:
            pass
        try:
            q.put_nowait(item)
        except Exception:
            pass
