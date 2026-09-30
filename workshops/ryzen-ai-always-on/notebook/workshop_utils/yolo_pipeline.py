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

"""YOLO inference pipeline: letterbox preprocess, postprocess + draw.

Supports YOLO26n end-to-end output format:
  output shape: (1, 300, 6) — 300 pre-filtered detections,
  each row: [x1, y1, x2, y2, confidence, class_id] in letterboxed pixel space.

Also retains a standalone nms() helper for legacy anchor-based decoders (e.g. YOLOv12);
postprocess() itself only handles the YOLO26 end-to-end (1, 300, 6) format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


INPUT_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

COCO_CLASSES: list[str] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass
class LetterboxInfo:
    """How an image was letterboxed, needed to undo the transform later."""

    ratio: float
    pad_x: int  # pixels of padding on the left
    pad_y: int  # pixels of padding on the top


@dataclass
class Detection:
    """A single object detection in original-image coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int

    @property
    def class_name(self) -> str:
        if 0 <= self.class_id < len(COCO_CLASSES):
            return COCO_CLASSES[self.class_id]
        return f"cls_{self.class_id}"


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------


def letterbox(
    img: np.ndarray, new_size: int = INPUT_SIZE
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize `img` to fit in a `new_size`x`new_size` square, preserving aspect ratio."""
    h0, w0 = img.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_w, new_h = int(w0 * r), int(h0 * r)
    pad_w = new_size - new_w
    pad_h = new_size - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    resized = cv2.resize(img, (new_w, new_h))
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, LetterboxInfo(ratio=r, pad_x=left, pad_y=top)


def preprocess_frame(
    frame_bgr: np.ndarray, input_size: int = INPUT_SIZE
) -> tuple[np.ndarray, LetterboxInfo]:
    """BGR HxWx3 uint8 → NCHW float32 in [0, 1], plus letterbox info for undo."""
    padded, info = letterbox(frame_bgr, input_size)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    nchw = np.ascontiguousarray(
        np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...]
    )
    return nchw, info


# -----------------------------------------------------------------------------
# Postprocess
# -----------------------------------------------------------------------------


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    x, y, w, h = boxes.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Greedy NMS in NumPy. Returns indices to keep."""
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(iou <= iou_thres)[0] + 1]
    return keep


def postprocess(
    raw_outputs: Sequence[np.ndarray],
    original_shape: tuple[int, int],
    info: LetterboxInfo,
    conf_threshold: float = CONF_THRESHOLD,
    iou_threshold: float = IOU_THRESHOLD,
) -> list[Detection]:
    """ONNX outputs → list of Detection in original-image pixel coords.

    Handles YOLO26 end-to-end format: output shape (1, 300, 6)
    where each row is [x1, y1, x2, y2, confidence, class_id].
    """
    preds = raw_outputs[0][0]  # (300, 6)
    boxes = preds[:, :4]  # xyxy in letterboxed input space (640x640 by default)
    scores = preds[:, 4]
    class_ids = preds[:, 5].astype(int)

    mask = scores > conf_threshold
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]
    if len(boxes) == 0:
        return []

    # Undo letterbox: remove padding offset then scale to original image size
    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y
    boxes /= info.ratio

    h, w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)

    out: list[Detection] = []
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int).tolist()
        out.append(
            Detection(
                x1,
                y1,
                x2,
                y2,
                float(score),
                int(np.clip(cls, 0, len(COCO_CLASSES) - 1)),
            )
        )
    return out


# -----------------------------------------------------------------------------
# Rendering
# -----------------------------------------------------------------------------


def draw_detections(
    frame_bgr: np.ndarray,
    detections: list[Detection],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw boxes + labels onto a copy of `frame_bgr` and return it."""
    out = frame_bgr.copy()
    for d in detections:
        label = f"{d.class_name} {d.score:.2f}"
        cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, thickness)
        cv2.putText(
            out,
            label,
            (d.x1, max(d.y1 - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
    return out
