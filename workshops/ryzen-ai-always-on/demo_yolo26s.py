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

"""
demo_yolo26s.py — Live YOLO26-pose (small) on Ryzen AI NPU.

Webcam or video → 17-keypoint skeleton overlay. Press 'q' to quit.

Usage:
    python demo_yolo26s.py                        # BF16, NPU, webcam
    python demo_yolo26s.py --cpu                  # run on CPU instead of NPU
    python demo_yolo26s.py --video assets/sample_video.mp4
"""

import json
import sys
import time
import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE / "notebook"))

from workshop_utils.yolo_pipeline import preprocess_frame

CACHE_DIR = HERE / "cache"

KPT_VIS_THRESHOLD = 0.3
CONF_THRESHOLD = 0.4

SKELETON = [
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (11, 13),
    (13, 15),  # left leg
    (12, 14),
    (14, 16),  # right leg
    (5, 6),
    (11, 12),  # shoulders, hips
    (5, 11),
    (6, 12),  # torso sides
    (0, 1),
    (0, 2),  # nose to eyes
    (1, 3),
    (2, 4),  # eyes to ears
]


def postprocess_pose(outputs, original_shape, info, conf_threshold=CONF_THRESHOLD):
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


def draw_poses(frame, boxes, scores, kpts):
    out = frame.copy()
    for box, score, person_kpts in zip(boxes, scores, kpts):
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
            xa, ya, va = person_kpts[a]
            xb, yb, vb = person_kpts[b]
            if va > KPT_VIS_THRESHOLD and vb > KPT_VIS_THRESHOLD:
                cv2.line(out, (int(xa), int(ya)), (int(xb), int(yb)), (255, 200, 0), 2)
        for x, y, v in person_kpts:
            if v > KPT_VIS_THRESHOLD:
                cv2.circle(out, (int(x), int(y)), 4, (0, 0, 255), -1)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--video", help="Use video file instead of webcam")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of NPU")
    parser.add_argument("--width", type=int, default=1920, help="Webcam capture width")
    parser.add_argument(
        "--height", type=int, default=1080, help="Webcam capture height"
    )
    args = parser.parse_args()

    onnx_path = HERE / "notebook" / "yolo26s-pose.onnx"
    cache_key = "yolo26s-pose_fp32"
    precision = "BF16"

    if not onnx_path.exists():
        print(f"ERROR: {onnx_path} not found.")
        print("Run steps 1-2 in the notebook to export the FP32 ONNX first.")
        sys.exit(1)

    if args.cpu:
        print(f"Building CPU session (small {precision})...")
        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        label = "CPU (YOLO26s-pose)"
    else:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        config = {
            "passes": [
                {"name": "init", "plugin": "vaip-pass_init"},
                {
                    "name": "vaiml_partition",
                    "plugin": "vaip-pass_vaiml_partition",
                    "vaiml_config": {
                        "enable_f32_to_bf16_conversion": True,
                        "keep_outputs": True,
                        "logging_level": "info",
                    },
                },
            ],
            "target": "VAIML",
            "targets": [{"name": "VAIML", "pass": ["init", "vaiml_partition"]}],
        }
        config_path = CACHE_DIR / "vitisai_config.json"
        config_path.write_text(json.dumps(config, indent=2))

        print(
            f"Building NPU BF16 session (small, cache hit → ~1s, first compile → ~30-60 min)..."
        )
        session = onnxruntime.InferenceSession(
            str(onnx_path),
            providers=["VitisAIExecutionProvider"],
            provider_options=[
                {
                    "config_file": str(config_path),
                    "cache_dir": str(CACHE_DIR),
                    "cache_key": cache_key,
                    "target": "VAIML",
                }
            ],
        )
        label = "NPU (YOLO26s-pose)"

    input_name = session.get_inputs()[0].name
    print(f"Session ready — {label}")

    if args.video:
        cap = cv2.VideoCapture(args.video)
        source = args.video
    else:
        # Request MJPG at the desired resolution; without MJPG most UVC webcams
        # fall back to a low-res 4:3 YUYV default (e.g. 640x480), which looks
        # cropped and nothing like the camera's native 1080p 16:9 field of view.
        cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        source = f"webcam {args.cam}"

    if not cap.isOpened():
        print(f"ERROR: could not open {source}")
        sys.exit(1)

    if not args.video:
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(
            f"Requested {args.width}x{args.height}, camera delivered {actual_w}x{actual_h}"
        )
    print(f"Opened {source}. Press 'q' to quit.")

    fps_window = deque(maxlen=30)
    window_name = f"YOLO26s-pose — {label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        t0 = time.perf_counter()
        nchw, info = preprocess_frame(frame)
        outputs = session.run(None, {input_name: nchw})
        boxes, scores, kpts = postprocess_pose(
            outputs, frame.shape[:2], info, args.conf
        )
        elapsed = time.perf_counter() - t0
        fps_window.append(1.0 / elapsed)

        annotated = draw_poses(frame, boxes, scores, kpts)
        fps = sum(fps_window) / len(fps_window)
        cv2.putText(
            annotated,
            f"{label}  {fps:5.1f} FPS  |  {len(boxes)} people",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
