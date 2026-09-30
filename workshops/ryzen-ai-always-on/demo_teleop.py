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
demo_teleop.py — standalone pinch-to-teleop, matching notebook Section 8.

The full "Physical AI on one chip" finale, outside Jupyter:

    NPU (VitisAI EP)  webcam -> YOLO26-pose (anchor) -> 21-pt hand-landmark model
                      -- BOTH neural nets run on the AIE, back-to-back
    CPU               21 landmarks -> palm position + pinch -> end-effector target
    iGPU (MuJoCo/EGL) SO-101 arm: IK solves to the target, a pinch grabs the cube

The same live NPU/CPU/iGPU/RAM resource dashboard from the notebook is rendered
and stacked directly under the feed, so all three compute units are visible at
once. Move your hand to fly the gripper; pinch to grab, open to drop into the hole.

Reuses the already-compiled VAIML caches (cache/yolo26s-pose_fp32 and
cache/hand_landmark_fp32) — NO recompile.

Controls:  q = quit   r = reset cube

Usage:
    python demo_teleop.py                         # webcam + dashboard
    python demo_teleop.py --cpu                   # both models on the CPU
    python demo_teleop.py --video assets/sample_video.mp4
    python demo_teleop.py --no-dashboard          # feed only
"""

import sys
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE / "notebook"))

from workshop_utils.ep_factory import npu_session_options, make_cpu_session
from workshop_utils.pose_common import make_frame_source
from workshop_utils.arm_teleop_mirror import ArmTeleopPipeline
from workshop_utils import pipeline_registry

CACHE_DIR = HERE / "cache"
CONFIG_PATH = CACHE_DIR / "vitisai_config.json"

POSE_ONNX = HERE / "notebook" / "yolo26s-pose.onnx"
POSE_CACHE_KEY = "yolo26s-pose_fp32"
HAND_ONNX = HERE / "models" / "hand_landmark.onnx"
HAND_CACHE_KEY = "hand_landmark_fp32"


def make_npu_session(onnx_path: Path, cache_key: str) -> onnxruntime.InferenceSession:
    """Build a VitisAI/NPU session that loads a pre-compiled VAIML cache.

    Identical provider_options to the notebook, so a matching cache_key is a
    ~1s cache hit rather than a 30-60 min recompile.
    """
    return onnxruntime.InferenceSession(
        str(onnx_path),
        sess_options=npu_session_options(),
        providers=["VitisAIExecutionProvider"],
        provider_options=[
            {
                "config_file": str(CONFIG_PATH),
                "cache_dir": str(CACHE_DIR),
                "cache_key": cache_key,
                "target": "VAIML",
            }
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--cam", type=int, default=0, help="Webcam index")
    parser.add_argument("--video", help="Use a video file instead of the webcam")
    parser.add_argument(
        "--conf", type=float, default=0.4, help="Pose confidence threshold"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run both models on the CPU instead of the NPU (no NPU/turbo needed)",
    )
    parser.add_argument(
        "--render-size",
        type=int,
        default=600,
        help="MuJoCo render square size (px); also sets the feed height",
    )
    parser.add_argument("--width", type=int, default=1920, help="Webcam capture width")
    parser.add_argument(
        "--height", type=int, default=1080, help="Webcam capture height"
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not horizontally flip the camera (mirror is on by default)",
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Hide the NPU/CPU/iGPU/RAM resource dashboard",
    )
    parser.add_argument(
        "--npu-infer-ms",
        type=float,
        default=20.0,
        help="Measured NPU latency used to scale the dashboard NPU %% (turbo ~20ms)",
    )
    args = parser.parse_args()

    for path in (POSE_ONNX, HAND_ONNX):
        if not path.exists():
            print(
                f"ERROR: {path} not found. Run the notebook Sections 1-3 (+ hand cell) first."
            )
            sys.exit(1)

    if args.cpu:
        print("Building CPU sessions (pose + hand on the CPU EP)...")
        pose_session = make_cpu_session(POSE_ONNX)
        hand_session = make_cpu_session(HAND_ONNX)
        print("Ready — pose + hand both on the CPU (expect much lower fps).")
    else:
        print("Building NPU sessions (cache hit -> ~1s each; no recompile)...")
        pose_session = make_npu_session(POSE_ONNX, POSE_CACHE_KEY)
        hand_session = make_npu_session(HAND_ONNX, HAND_CACHE_KEY)
        print("Ready — pose + hand both on the NPU.")

    source = args.video if args.video else args.cam
    frame_source, cap = make_frame_source(source, args.width, args.height)
    if not args.video:
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera delivered {aw}x{ah} (requested {args.width}x{args.height})")

    # Latest composite frame produced by the pipeline's iGPU/render thread.
    latest = {"frame": None}

    def on_frame(bgr):
        latest["frame"] = bgr

    pipeline_registry.stop_active_pipelines()
    pipe = ArmTeleopPipeline(
        pose_session,
        frame_source,
        on_frame,
        conf=args.conf,
        mirror=not args.no_mirror,
        render_size=args.render_size,
        hand_session=hand_session,
        feed_npu_gauge=not args.cpu,  # keep the NPU gauge idle on a CPU run
        # video file -> fixed target (canned motion must land); webcam -> moving
        randomize_target=not bool(args.video),
    )

    # Reuse the notebook's exact dashboard renderer (matplotlib Agg -> PNG). It
    # runs its own 1 Hz sampling thread; we just read the latest PNG and stack it
    # under the feed. ipywidgets is only used as the (unshown) PNG container here.
    dash = None
    if not args.no_dashboard:
        try:
            from workshop_utils.live_dashboard import LiveDashboard

            dash = LiveDashboard(
                window_seconds=60.0, sample_period=1.0, npu_infer_ms=args.npu_infer_ms
            )
        except Exception as exc:
            print(f"(dashboard disabled: {exc})")
            dash = None

    engine = "CPU" if args.cpu else "NPU"
    window = f"Pinch-to-teleop  —  {engine} pose+hand | CPU math | iGPU SO-101"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    print("Running. Show your hand.  [q] quit   [r] reset cube")

    pipe.start()
    if dash is not None:
        dash.start(auto_display=False)
    pipeline_registry.register(pipe, cap)

    last_dash_bytes = None
    dash_bgr = None
    try:
        while True:
            composite = latest["frame"]
            if composite is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue

            view = composite
            if dash is not None:
                val = dash.widget.value
                if val and val is not last_dash_bytes:  # re-decode only on update
                    arr = np.frombuffer(val, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        w = composite.shape[1]
                        h = int(img.shape[0] * w / img.shape[1])
                        dash_bgr = cv2.resize(img, (w, h))
                    last_dash_bytes = val
                if dash_bgr is not None:
                    view = cv2.vconcat([composite, dash_bgr])

            cv2.imshow(window, view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                pipe.reset_object()
            # Quit if the user closed the window with the [x] button.
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        if dash is not None:
            dash.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
