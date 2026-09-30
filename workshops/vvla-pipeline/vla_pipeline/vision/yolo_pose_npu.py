# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLOv26s-pose person/keypoint detection on the Ryzen AI NPU.

Wraps the FP32 ONNX export (``scripts/export_yolo26s_pose.py``) in a
VitisAI EP session (BF16 on the NPU, CPU fallback) and exposes a simple
``PoseEstimator.infer(frame_bgr)`` API returning boxes, scores, and COCO-17
keypoints in original image coordinates.

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
from vla_pipeline.utils.npu_session import build_session

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

SKELETON = [
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
]


@dataclass
class LetterboxInfo:
    """Scale ratio and padding applied by letterbox(), needed to map detections back to the original frame."""

    ratio: float
    pad_x: float
    pad_y: float


def letterbox(frame_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize+pad a BGR frame to (1,3,imgsz,imgsz) float32 NCHW in [0,1]."""
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


class PoseEstimator:
    """YOLOv26s-pose ONNX inference (end-to-end head: [1, 300, 57], no NMS)."""

    def __init__(self, cfg: dict, device: str | None = None):
        y = cfg["yolo_pose"]
        self.imgsz: int = int(y["imgsz"])
        self.conf_threshold: float = float(y["conf_threshold"])
        self.kpt_threshold: float = float(y["kpt_threshold"])
        self.session = build_session(
            y["onnx"],
            device or y["device"],
            vitisai_config=y.get("vitisai_config", "config/vaiep_config.json"),
            cache_dir=cfg.get("system", {}).get("cache_dir", "cache"),
            cache_key=y["cache_key"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def infer(
        self, frame_bgr: np.ndarray, conf: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run pose estimation on one BGR frame.

        Returns:
            boxes  [N, 4] xyxy in original pixels,
            scores [N],
            kpts   [N, 17, 3] (x, y, visibility) in original pixels.
        """
        conf = self.conf_threshold if conf is None else conf
        nchw, info = letterbox(frame_bgr, self.imgsz)
        outputs = self.session.run(None, {self.input_name: nchw})

        preds = outputs[0][0]  # [300, 57]
        boxes = preds[:, :4].copy()
        scores = preds[:, 4]
        kpts = preds[:, 6:].reshape(-1, 17, 3).copy()

        mask = scores > conf
        boxes, scores, kpts = boxes[mask], scores[mask], kpts[mask]
        if len(boxes) == 0:
            return boxes, scores, kpts

        boxes[:, [0, 2]] -= info.pad_x
        boxes[:, [1, 3]] -= info.pad_y
        boxes /= info.ratio
        kpts[..., 0] -= info.pad_x
        kpts[..., 1] -= info.pad_y
        kpts[..., :2] /= info.ratio

        h, w = frame_bgr.shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, h)
        return boxes, scores, kpts

    def primary_person(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Return (kpts[17,3], score) for the largest-box detection, or None."""
        boxes, scores, kpts = self.infer(frame_bgr)
        if len(boxes) == 0:
            return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        return kpts[i], float(scores[i])

    def draw(self, frame: np.ndarray, boxes, scores, kpts) -> np.ndarray:
        """Draw person boxes, skeleton, and keypoints onto frame."""
        out = frame.copy()
        for box, score, person in zip(boxes, scores, kpts):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
            cv2.putText(
                out,
                f"person {score:.2f}",
                (x1, max(y1 - 5, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                1,
            )
            for a, b in SKELETON:
                xa, ya, va = person[a]
                xb, yb, vb = person[b]
                if va > self.kpt_threshold and vb > self.kpt_threshold:
                    cv2.line(
                        out, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2
                    )
            for x, y, v in person:
                if v > self.kpt_threshold:
                    cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
        return out


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
    label = f"YOLO26s-pose [{est.session.get_providers()[0].replace('ExecutionProvider', '')}]"
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
