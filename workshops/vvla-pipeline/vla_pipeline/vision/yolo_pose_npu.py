# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLOv26s-pose person/keypoint detection on the Ryzen AI NPU.

Wraps the FP32 ONNX export (``scripts/export_yolo26s.py pose``) in a
VitisAI EP session (BF16 on the NPU, CPU fallback) and exposes a simple
``PoseEstimator.infer(frame_bgr)`` API returning boxes, scores, and COCO-17
keypoints in original image coordinates. The letterbox, the decoder and the
overlay are plain functions here, shared with the detector.

Model heads. The export leaves ``nms`` unset, so the file is the raw
one-to-many head (pose ``[1, 56, 8400]`` = xywh, score, 17*(x, y, v)) and the
decoder runs NMS. ``nms=False`` exports the one-to-one head (``[1, 300, 57]``),
which skips NMS and scores far lower on the twin's objects; the decoder still
accepts it.

Independent test
----------------
    python -m vla_pipeline.vision.yolo_pose_npu                 # webcam overlay
    python -m vla_pipeline.vision.yolo_pose_npu --video clip.mp4
    python -m vla_pipeline.vision.yolo_pose_npu --device cpu    # no NPU stack
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from vla_pipeline.utils.config import load_config
from vla_pipeline.utils.npu_session import active_provider, build_npu_session, size_npu

logger = logging.getLogger(__name__)

# COCO-17 keypoint indices used throughout the pipeline.
KP = {
    "nose": 0,
    "l_eye": 1,
    "r_eye": 2,
    "l_ear": 3,
    "r_ear": 4,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}
_FACE_KPTS = (KP["nose"], KP["l_eye"], KP["r_eye"], KP["l_ear"], KP["r_ear"])

# BGR colors of the overlay
LEFT = (0, 255, 0)  # person's anatomical left
RIGHT = (0, 165, 255)  # person's anatomical right
CENTER = (255, 128, 0)  # spine / pelvis / shoulders
FACE = (255, 0, 200)

# (a, b, color) per bone
SKELETON = [
    (KP["l_ankle"], KP["l_knee"], LEFT),
    (KP["l_knee"], KP["l_hip"], LEFT),
    (KP["r_ankle"], KP["r_knee"], RIGHT),
    (KP["r_knee"], KP["r_hip"], RIGHT),
    (KP["l_hip"], KP["r_hip"], CENTER),
    (KP["l_shoulder"], KP["l_hip"], LEFT),
    (KP["r_shoulder"], KP["r_hip"], RIGHT),
    (KP["l_shoulder"], KP["r_shoulder"], CENTER),
    (KP["l_shoulder"], KP["l_elbow"], LEFT),
    (KP["l_elbow"], KP["l_wrist"], LEFT),
    (KP["r_shoulder"], KP["r_elbow"], RIGHT),
    (KP["r_elbow"], KP["r_wrist"], RIGHT),
    (KP["nose"], KP["l_eye"], FACE),
    (KP["nose"], KP["r_eye"], FACE),
    (KP["l_eye"], KP["l_ear"], FACE),
    (KP["r_eye"], KP["r_ear"], FACE),
    # head to torso - the link most broken skeletons drop
    (KP["l_ear"], KP["l_shoulder"], LEFT),
    (KP["r_ear"], KP["r_shoulder"], RIGHT),
]


@dataclass
class LetterboxInfo:
    """Scale ratio and padding applied by letterbox(), needed to map detections back to the original frame."""

    ratio: float
    pad_x: float
    pad_y: float


def letterbox(frame_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize+pad a BGR frame to (1,3,imgsz,imgsz) float32 NCHW in [0,1].

    Keep the returned info: the decoders need it to map boxes and keypoints
    back to original pixel coordinates.
    """
    h, w = frame_bgr.shape[:2]
    ratio = min(imgsz / w, imgsz / h)
    new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
    pad_x, pad_y = (imgsz - new_w) / 2, (imgsz - new_h) / 2

    resized = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    top, left = int(round(pad_y - 0.1)), int(round(pad_x - 0.1))
    canvas[top : top + new_h, left : left + new_w] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(nchw), LetterboxInfo(ratio, pad_x, pad_y)


def head_rows(output, end2end_width: int, raw_width: int) -> tuple[np.ndarray, bool]:
    """One row per candidate from a YOLO head, and whether it is the raw head.

    The end-to-end head is ``[1, 300, end2end_width]``; the raw head is
    ``[1, raw_width, 8400]`` and still needs NMS.
    """
    preds = np.asarray(output)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] in (raw_width, end2end_width) and preds.shape[-1] not in (raw_width, end2end_width):
        preds = preds.T
    if preds.shape[-1] not in (raw_width, end2end_width):
        raise ValueError(
            f"YOLO output shape {tuple(np.asarray(output).shape)} is neither "
            f"[1, 300, {end2end_width}] nor [1, {raw_width}, 8400]"
        )
    return preds, preds.shape[-1] == raw_width


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Center-format boxes ``[cx, cy, w, h]`` to corner format ``[x1, y1, x2, y2]``."""
    out = np.empty_like(boxes)
    out[:, :2] = boxes[:, :2] - boxes[:, 2:4] / 2
    out[:, 2:] = boxes[:, :2] + boxes[:, 2:4] / 2
    return out


def nms_keep(boxes: np.ndarray, scores: np.ndarray, iou: float = 0.45) -> np.ndarray:
    """Indices kept by greedy NMS over xyxy boxes.

    ``cv2.dnn.NMSBoxes`` takes top-left ``[x, y, width, height]``. Passing the
    corners would make each box as wide as its right edge, so two people or
    cups a few pixels apart would collapse into one detection.
    """
    if len(boxes) == 0:
        return np.empty(0, dtype=int)
    xywh = np.concatenate([boxes[:, :2], boxes[:, 2:] - boxes[:, :2]], axis=1)
    picked = cv2.dnn.NMSBoxes(xywh.tolist(), scores.astype(float).tolist(), 0.0, iou)
    return np.asarray(picked, dtype=int).reshape(-1)


def unmap_boxes(boxes: np.ndarray, info: LetterboxInfo, orig_shape) -> np.ndarray:
    """Letterbox-pixel xyxy boxes back to the original frame, clipped."""
    boxes = boxes.astype(np.float32, copy=True)
    boxes[:, [0, 2]] -= info.pad_x
    boxes[:, [1, 3]] -= info.pad_y
    boxes /= info.ratio
    h, w = orig_shape[:2]
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
    return boxes


def decode_pose(outputs, info: LetterboxInfo, orig_shape, conf: float):
    """Decode a YOLOv26s-pose output into original-pixel space.

    Returns ``boxes [N,4] xyxy``, ``scores [N]``, ``kpts [N,17,3] (x, y, vis)``.
    """
    preds, raw = head_rows(outputs[0], 57, 56)
    if raw:
        boxes = xywh_to_xyxy(preds[:, :4])
        kpts = preds[:, 5:].reshape(-1, 17, 3).copy()
    else:
        boxes = preds[:, :4].copy()
        kpts = preds[:, 6:].reshape(-1, 17, 3).copy()
    scores = preds[:, 4].copy()

    mask = scores > conf
    boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]
    if raw and len(boxes):
        keep = nms_keep(boxes, scores)
        boxes, scores, kpts = boxes[keep], scores[keep], kpts[keep]
    if len(boxes) == 0:
        return boxes, scores, kpts

    boxes = unmap_boxes(boxes, info, orig_shape)
    kpts[..., 0] -= info.pad_x
    kpts[..., 1] -= info.pad_y
    kpts[..., :2] /= info.ratio
    return boxes, scores, kpts


def pick_nearest(boxes: np.ndarray) -> int:
    """Index of the person CLOSEST to the camera - the largest-area box.

    With several people in frame, mimic follows exactly one: the nearest.
    Box area is the proximity proxy (a closer person subtends more pixels).
    ``boxes`` must not be empty.
    """
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def pick_wrist(kpts: np.ndarray, threshold: float):
    """Pixel ``(x, y)`` of the more-visible wrist, or None if neither is."""
    best, best_v = None, threshold
    for w in (KP["l_wrist"], KP["r_wrist"]):
        if kpts[w, 2] >= best_v:
            best, best_v = (float(kpts[w, 0]), float(kpts[w, 1])), float(kpts[w, 2])
    return best


def pick_arm(kpts: np.ndarray, threshold: float, side: str = "r"):
    """``(wrist_xy, elbow_xy)`` for a fixed side (default right).

    ``wrist_xy`` is None when that side's wrist doesn't clear ``threshold``
    (and then so is ``elbow_xy``); ``elbow_xy`` is None when only the elbow
    doesn't.
    """
    w, e = KP[f"{side}_wrist"], KP[f"{side}_elbow"]
    if float(kpts[w, 2]) < threshold:
        return None, None
    wrist = (float(kpts[w, 0]), float(kpts[w, 1]))
    elbow = (float(kpts[e, 0]), float(kpts[e, 1])) if float(kpts[e, 2]) >= threshold else None
    return wrist, elbow


def draw_skeleton(frame, boxes, scores, kpts, kpt_threshold: float = 0.3, highlight_right_hand: bool = True):
    """Draw person boxes + COCO-17 skeletons; returns a copy."""
    out = frame.copy()

    def valid(pt):
        x, y, v = pt
        return v > kpt_threshold and np.isfinite(x) and np.isfinite(y)

    for box, score, person in zip(boxes, scores, kpts):
        x1, y1, x2, y2 = box.astype(int)
        # stroke and joint size scale with body height, so it reads near and far
        t = max(1, round((y2 - y1) / 150))
        r = max(2, round((y2 - y1) / 90))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), t, cv2.LINE_AA)
        cv2.putText(out, f"person {score:.2f}", (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
        for a, b, color in SKELETON:
            pa, pb = person[a], person[b]
            if valid(pa) and valid(pb):
                cv2.line(out, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), color, t, cv2.LINE_AA)
        # joints on top, side-colored, with a thin dark outline so they stay crisp
        for i, (x, y, v) in enumerate(person):
            if not valid((x, y, v)):
                continue
            jc = FACE if i in _FACE_KPTS else LEFT if i % 2 == 1 else RIGHT  # odd COCO indices: the left side
            cv2.circle(out, (int(x), int(y)), r, jc, -1, cv2.LINE_AA)
            cv2.circle(out, (int(x), int(y)), r, (30, 30, 30), 1, cv2.LINE_AA)
        # anchor for the MediaPipe right-hand crop
        if highlight_right_hand and valid(person[KP["r_wrist"]]):
            wx, wy = int(person[KP["r_wrist"]][0]), int(person[KP["r_wrist"]][1])
            cv2.circle(out, (wx, wy), 8, (0, 255, 255), 2, cv2.LINE_AA)
    return out


class PoseEstimator:
    """YOLOv26s-pose ONNX inference (raw head [1, 56, 8400] + NMS; end-to-end head [1, 300, 57])."""

    def __init__(self, cfg: dict, device: str | None = None):
        y = cfg["yolo_pose"]
        self.imgsz: int = int(y["imgsz"])
        self.conf_threshold: float = float(y["conf_threshold"])
        self.kpt_threshold: float = float(y["kpt_threshold"])
        size_npu(cfg)
        self.session = build_npu_session(
            y["onnx"],
            device or y["device"],
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "cache"),
            cache_key=y["cache_key"],
        )
        self.provider = active_provider(self.session)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, frame_bgr: np.ndarray, conf: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run pose estimation on one BGR frame.

        Returns:
            boxes  [N, 4] xyxy in original pixels,
            scores [N],
            kpts   [N, 17, 3] (x, y, visibility) in original pixels.
        """
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        outputs = self.session.run(None, {self.input_name: nchw})
        return decode_pose(outputs, info, frame_bgr.shape, conf)

    def primary_person(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        """``(kpts[17,3], score)`` for the nearest (largest-box) person, or None."""
        boxes, scores, kpts = self.infer(frame_bgr)
        if len(boxes) == 0:
            return None
        i = pick_nearest(boxes)
        return kpts[i], float(scores[i])

    def draw(self, frame: np.ndarray, boxes, scores, kpts) -> np.ndarray:
        """Draw person boxes, skeleton, and keypoints onto a copy of ``frame``."""
        return draw_skeleton(frame, boxes, scores, kpts, self.kpt_threshold)


# =============================================================================
# Standalone component test (live overlay)
# =============================================================================


def main() -> None:
    """CLI entry point: live overlay of pose estimation on a webcam or video file."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="YOLOv26s-pose on Ryzen AI NPU - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--cam", type=int, default=None, help="camera index (default: config)"
    )
    parser.add_argument(
        "--video", default=None, help="run on a video file instead of webcam"
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    est = PoseEstimator(cfg, device=args.device)
    label = f"YOLO26s-pose [{est.provider.replace('ExecutionProvider', '')}]"
    print(f"Session ready - {label}")

    cap = (
        cv2.VideoCapture(args.video)
        if args.video
        else cv2.VideoCapture(
            args.cam if args.cam is not None else cfg["cameras"]["mount"]["device"]
        )
    )
    if not cap.isOpened():
        raise SystemExit("ERROR: could not open video source")

    fps_window: deque[float] = deque(maxlen=30)
    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        boxes, scores, kpts = est.infer(frame)
        fps_window.append(1.0 / max(time.perf_counter() - t0, 1e-6))

        out = est.draw(frame, boxes, scores, kpts)
        fps = sum(fps_window) / len(fps_window)
        cv2.putText(
            out,
            f"{label}  {fps:5.1f} FPS | {len(boxes)} people",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.imshow(label, out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
