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

"""YOLO26-seg (instance segmentation) decode + background-blur compositing.

The "camera effects" section of the notebook. The NPU runs the seg backbone;
the mask assembly here (coefficients x prototypes -> sigmoid -> crop -> resize)
is light CPU work -- the same "a little bookkeeping stays on the CPU" story as
the pose model's decode head.

Reuses ``yolo_pipeline.preprocess_frame`` for the identical 640x640 letterbox,
so a seg session is fed exactly like the pose/detect sessions.

YOLO26-seg ONNX outputs (ultralytics end-to-end export, static batch-1) --
CONFIRMED against the exported model:
    out[0]  detections  (1, 300, 38)   300 pre-filtered rows (NMS-free, like the
            pose/detect heads). Each row: [x1, y1, x2, y2, score, class, 32 mask
            coeffs] -- box is XYXY in the 640 letterbox space.
    out[1]  prototypes  (1, 32, mask_h, mask_w)   e.g. (1, 32, 160, 160)

A per-instance mask = sigmoid(coeffs @ protos), reshaped to (mask_h, mask_w),
upsampled to the 640 letterbox, cropped to the instance box, then un-letterboxed
back to the original frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .yolo_pipeline import LetterboxInfo, INPUT_SIZE, COCO_CLASSES

PERSON_CLASS = 0


@dataclass
class SegInstance:
    """One segmented instance: box (original px), score, class, and full-frame mask."""

    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    class_id: int
    mask: np.ndarray  # uint8 {0,1}, HxW at ORIGINAL frame resolution

    @property
    def area(self) -> int:
        return int((self.x2 - self.x1) * (self.y2 - self.y1))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def decode_seg(
    raw_outputs,
    original_shape: tuple[int, int],
    info: LetterboxInfo,
    conf_threshold: float = 0.35,
    class_filter: int | None = PERSON_CLASS,
    input_size: int = INPUT_SIZE,
    max_instances: int = 10,
) -> list[SegInstance]:
    """ONNX seg outputs -> list of SegInstance with full-frame binary masks.

    Handles the end-to-end (NMS-free) seg head: out[0] = (1, 300, 38) rows of
    ``[x1, y1, x2, y2, score, class, 32 mask coeffs]`` (box XYXY in letterbox
    space), out[1] = (1, 32, mh, mw) prototypes.

    ``class_filter`` keeps only that class id (default: person). Pass ``None`` to
    keep all classes. Masks are returned at the original frame resolution.
    """
    dets = np.asarray(raw_outputs[0])
    protos = np.asarray(raw_outputs[1])
    if dets.ndim == 3:
        dets = dets[0]  # (300, 38)
    if protos.ndim == 4:
        protos = protos[0]  # (32, mh, mw)

    num_mask = protos.shape[0]  # 32
    boxes_lb = dets[:, :4].astype(np.float32)  # xyxy in letterbox space
    scores = dets[:, 4].astype(np.float32)
    class_ids = dets[:, 5].astype(int)
    coeffs = dets[:, 6 : 6 + num_mask].astype(np.float32)

    keep = scores > conf_threshold
    if class_filter is not None:
        keep &= class_ids == class_filter
    boxes_lb, scores, class_ids, coeffs = (
        boxes_lb[keep],
        scores[keep],
        class_ids[keep],
        coeffs[keep],
    )
    if len(scores) == 0:
        return []

    # Already NMS-filtered by the end-to-end head; just cap to the top-N by score.
    if len(scores) > max_instances:
        order = scores.argsort()[::-1][:max_instances]
        boxes_lb, scores, class_ids, coeffs = (
            boxes_lb[order],
            scores[order],
            class_ids[order],
            coeffs[order],
        )

    # Assemble masks in prototype space, then upsample to the letterbox canvas.
    mh, mw = protos.shape[1], protos.shape[2]
    protos_flat = protos.reshape(num_mask, -1)  # (32, mh*mw)
    masks_small = _sigmoid(coeffs @ protos_flat)  # (n, mh*mw)
    masks_small = masks_small.reshape(-1, mh, mw)  # (n, mh, mw)

    oh, ow = original_shape
    out: list[SegInstance] = []
    for i in range(len(scores)):
        # upsample this instance's mask to the full letterbox (input_size square)
        m = cv2.resize(
            masks_small[i], (input_size, input_size), interpolation=cv2.INTER_LINEAR
        )
        # crop to the instance box (in letterbox space) so masks don't bleed
        bx1, by1, bx2, by2 = boxes_lb[i]
        crop = np.zeros_like(m)
        cx1, cy1 = max(0, int(bx1)), max(0, int(by1))
        cx2, cy2 = min(input_size, int(bx2)), min(input_size, int(by2))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop[cy1:cy2, cx1:cx2] = m[cy1:cy2, cx1:cx2]
        # undo the letterbox: strip padding, resize back to the original frame
        x0, y0 = info.pad_x, info.pad_y
        unpadded_w = int(round(ow * info.ratio))
        unpadded_h = int(round(oh * info.ratio))
        crop = crop[y0 : y0 + unpadded_h, x0 : x0 + unpadded_w]
        if crop.size == 0:
            continue
        mask_full = cv2.resize(crop, (ow, oh), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_full > 0.5).astype(np.uint8)

        # box back to original coords too (for optional overlay / picking)
        rx1 = int(np.clip((bx1 - x0) / info.ratio, 0, ow))
        ry1 = int(np.clip((by1 - y0) / info.ratio, 0, oh))
        rx2 = int(np.clip((bx2 - x0) / info.ratio, 0, ow))
        ry2 = int(np.clip((by2 - y0) / info.ratio, 0, oh))
        out.append(
            SegInstance(
                rx1, ry1, rx2, ry2, float(scores[i]), int(class_ids[i]), mask_bin
            )
        )
    return out


def combined_mask(
    instances: list[SegInstance], shape: tuple[int, int], largest_only: bool = False
) -> np.ndarray:
    """Union of instance masks as a uint8 {0,1} map at ``shape`` (H, W).

    ``largest_only`` keeps just the biggest instance -- the "isolate the closest
    person" behavior used to reject background people.
    """
    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)
    if not instances:
        return out
    chosen = [max(instances, key=lambda s: s.area)] if largest_only else instances
    for inst in chosen:
        if inst.mask.shape == (h, w):
            out |= inst.mask
    return out


def blur_background(
    frame_bgr: np.ndarray, mask: np.ndarray, blur_strength: int = 35, feather: int = 9
) -> np.ndarray:
    """Keep the masked subject sharp; blur everything else (virtual-background).

    ``blur_strength`` is the Gaussian kernel size (odd; larger = blurrier).
    ``feather`` softens the mask edge so the composite isn't a hard cut-out.
    """
    k = max(3, int(blur_strength) | 1)  # force odd, >=3
    blurred = cv2.GaussianBlur(frame_bgr, (k, k), 0)
    if mask is None or mask.max() == 0:
        return blurred  # no subject -> all blurred
    alpha = mask.astype(np.float32)
    if feather and feather >= 3:
        f = int(feather) | 1
        alpha = cv2.GaussianBlur(alpha, (f, f), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]  # HxWx1
    out = frame_bgr.astype(np.float32) * alpha + blurred.astype(np.float32) * (
        1.0 - alpha
    )
    return out.astype(np.uint8)


def replace_background(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (32, 120, 40),
    feather: int = 9,
) -> np.ndarray:
    """Composite the masked subject over a flat color (green-screen style)."""
    bg = np.full_like(frame_bgr, color)
    if mask is None or mask.max() == 0:
        return bg
    alpha = mask.astype(np.float32)
    if feather and feather >= 3:
        f = int(feather) | 1
        alpha = cv2.GaussianBlur(alpha, (f, f), 0)
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    out = frame_bgr.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha)
    return out.astype(np.uint8)


def draw_mask_overlay(
    frame_bgr: np.ndarray,
    instances: list[SegInstance],
    color: tuple[int, int, int] = (60, 220, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Tint each instance mask onto the frame (a quick 'what did seg find' view)."""
    out = frame_bgr.copy()
    if not instances:
        return out
    overlay = out.copy()
    for inst in instances:
        overlay[inst.mask.astype(bool)] = color
    cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)
    return out
