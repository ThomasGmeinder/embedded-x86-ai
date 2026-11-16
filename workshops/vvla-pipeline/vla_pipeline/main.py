# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pipeline orchestrator: microphone → Gemma 4 on the iGPU → behavior dispatch,
with an always-hot listener for the safety stop.

Architecture
------------
- A **listener thread** sends a hop of microphone audio to Gemma as soon as
  that hop contains speech (:mod:`vla_pipeline.llm.voice`). It does not wait
  for silence. The microphone stays hot *while behaviors run*, which is what
  makes a spoken "stop robot" able to interrupt a running behavior.
- ``STOP`` is matched on the raw transcript before any LLM round trip
  (:func:`~vla_pipeline.llm.intents.is_stop`: the phrase "stop robot", or a
  Gemma command JSON that says stop), then sets the shared stop event that
  every behavior's ``stop_check`` polls (including mid ``go_to``
  interpolation). The 'q' key in any preview window and Ctrl-C do the same.
- Every other transcript goes through the grammar-constrained intent parser
  and is dispatched via the registry (:func:`~vla_pipeline.user_robot_control.build_registry`) -
  adding a voice command never touches this file (see :mod:`vla_pipeline.llm.intents`).
- Behaviors run on the **main thread** (OpenCV preview windows want that);
  a new command interrupts the current behavior (graceful stop → switch).

Run
---
    python -m vla_pipeline.main                  # full pipeline
    python -m vla_pipeline.main --text-commands  # type instead of speaking
    python -m vla_pipeline.main --dry-run        # no robot hardware
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading

from vla_pipeline.user_robot_control import PipelineContext, build_registry
from vla_pipeline.utils.config import ensure_dirs, load_config

logger = logging.getLogger(__name__)


def apply_sim_overrides(cfg: dict) -> dict:
    """Switch ``cfg`` to the digital twin: sim arm + sim arm camera, real mount camera."""
    cfg.setdefault("robot", {})["backend"] = "sim"
    cams = cfg.setdefault("cameras", {})
    cams.setdefault("arm", {})["source"] = "sim"
    cams.setdefault("mount", {}).setdefault("source", "device")
    return cfg


def build_context(cfg: dict, args, *, sensors: bool = True) -> PipelineContext:
    """Build the :class:`PipelineContext`, wiring up the arm, cameras, vision models and the registry.

    Each vision component is optional: a failure to construct it is logged and
    left as None so the pipeline still runs with degraded capabilities.
    """
    from vla_pipeline.robot.arm_interface import make_arm
    from vla_pipeline.utils.gesture_map import mapper_from_config

    arm = make_arm(cfg)
    ctx = PipelineContext(cfg=cfg, arm=arm, headless=True, mapper=mapper_from_config(cfg),
                          registry=build_registry())
    if not sensors:
        return ctx

    from vla_pipeline.vision.camera import make_camera

    try:
        ctx.mount_camera = make_camera(cfg, "mount")
    except Exception as e:
        logger.warning("Mount camera unavailable: %s", e)
    try:
        ctx.arm_camera = make_camera(cfg, "arm")
    except Exception as e:
        logger.warning("Arm camera unavailable: %s", e)

    # The arm and camera clients above have already started DDS. A Vitis AI
    # session opened first leaves LARGE_DATA unable to discover peers.
    device = args.device
    try:
        from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

        ctx.pose = PoseEstimator(cfg, device=device)
    except Exception as e:
        logger.warning("Pose model unavailable (%s) - mimic/handoff degraded.", e)
    try:
        from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

        ctx.detector = ObjectDetector(cfg, device=device)
    except Exception as e:
        logger.warning("Detector unavailable (%s) - pick_place disabled.", e)
    try:
        from vla_pipeline.vision.mediapipe_hands import HandTracker

        ctx.hands = HandTracker(cfg)
    except Exception as e:
        logger.warning("Hand tracker unavailable: %s", e)
    return ctx


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    """The arguments ``./run_pipeline.sh`` and this module accept.

    ``run_pipeline.sh`` parses with this function before it sets anything up,
    so an unknown argument stops there instead of starting the arm.
    """
    parser = argparse.ArgumentParser(
        prog=prog, description="Strix VLA pipeline orchestrator"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--text-commands",
        action="store_true",
        help="type commands instead of speaking (skips the microphone and Gemma's audio)",
    )
    parser.add_argument("--dry-run", action="store_true", help="no robot hardware")
    parser.add_argument("--headless", action="store_true", help="no preview windows")
    parser.add_argument(
        "--device",
        choices=["npu", "cpu"],
        default=None,
        help="override the device for all ONNX models",
    )
    parser.add_argument(
        "--digital-twin",
        action="store_true",
        help="drive the MuJoCo digital twin instead of the real arm "
        "(robot.backend=sim, arm camera rendered on the gripper)",
    )
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="with --digital-twin: skip the twin console window",
    )
    return parser


def main() -> None:
    """Parse CLI args, build the pipeline context, and run the listen/dispatch loop until interrupted."""
    # Ctrl-C must kill this process even during an NPU run. The notebook kernel
    # keeps the default, which defers that signal until the run returns.
    from vla_pipeline.utils import npu_session

    npu_session.defer_sigint = False
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args()

    cfg = load_config(args.config)
    if args.digital_twin:
        apply_sim_overrides(cfg)
    ensure_dirs(cfg)
    # Cameras, YOLO, MediaPipe and the microphone run in the perception process.
    ctx = build_context(cfg, args, sensors=False)

    from vla_pipeline.llm.intents import keyword_parse
    from vla_pipeline.user_robot_control import UserRobotControl
    from vla_pipeline.wire import ControlLink

    # A command that arrived before the epoch is held by the voice outbox and
    # sent once this process publishes epoch 1. Nothing is dropped on a timer.
    robot_voice = UserRobotControl(ctx, keyword_parse)

    def _restore() -> bool:
        fn = getattr(ctx.arm, "reset", None)
        if not callable(fn):
            return False
        result = fn()
        return True if result is None else bool(result)

    link = ControlLink(ctx.arm.node, ctx, robot_voice, _restore)
    ctx.arm.on_cached_joints = link.set_joints
    shutdown = threading.Event()
    if args.text_commands:

        def _stdin_loop() -> None:
            """Read typed commands from stdin (--text-commands mode)."""
            while not shutdown.is_set():
                try:
                    line = input("you> ")
                except EOFError:
                    shutdown.set()
                    robot_voice.stop()
                    break
                if line.strip():
                    robot_voice.on_transcript(line.strip())

        threading.Thread(target=_stdin_loop, daemon=True, name="stdin").start()

    ctx.say(
        "Ready. Try: 'copy my movements', 'pick up the ball', 'wave', "
        "'dance', 'hold this', 'go home' - or 'stop robot' at any time. "
        "In the twin window: space stops, r resets the arm and the table."
    )

    def _on_signal(_signum, _frame) -> None:
        # rclpy replaces SIGINT with a handler that does not end this loop.
        # The loop is blocked on the transcript queue, so the sentinel is
        # what makes run() return.
        shutdown.set()
        ctx.stop_event.set()
        robot_voice.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)
    try:
        robot_voice.run(shutdown)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()
        ctx.stop_event.set()
        link.mailbox.close()
        logger.info("Shutting down - parking arm.")
        try:
            ctx.arm.close()
        except Exception:
            logger.exception("arm close")


if __name__ == "__main__":
    main()
