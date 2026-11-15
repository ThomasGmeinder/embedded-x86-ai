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


def build_context(cfg: dict, args) -> PipelineContext:
    """Build the :class:`PipelineContext`, wiring up the arm, cameras, vision models and the registry.

    Each vision component is optional: a failure to construct it is logged and
    left as None so the pipeline still runs with degraded capabilities.
    """
    from vla_pipeline.robot.arm_interface import make_arm
    from vla_pipeline.utils.gesture_map import mapper_from_config

    arm = make_arm(cfg)
    ctx = PipelineContext(cfg=cfg, arm=arm, headless=args.headless, mapper=mapper_from_config(cfg),
                          registry=build_registry())

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
    ctx = build_context(cfg, args)

    console = None
    if args.digital_twin and not (args.headless or args.no_console):
        from vla_pipeline.console import TwinConsole

        console = TwinConsole(ctx, voice="typed" if args.text_commands else "mic")

    from vla_pipeline.user_robot_control import UserRobotControl
    from vla_pipeline.llm.voice import open_voice

    llm = open_voice(cfg, device=args.device)
    # Ignore any transcript for a short grace window after launch - the mic goes
    # hot before the user is ready and the first frames are usually room noise
    # that the listener hears as words. Voice ('stop robot' included) only starts
    # being honored once the window passes. Tunable via audio.startup_grace_s.
    # Typed commands have no mic, so they get no window.
    grace_s = 0.0 if args.text_commands else float((cfg.get("audio", {}) or {}).get("startup_grace_s", 1.5))
    robot_voice = UserRobotControl(ctx, llm.parse, startup_grace_s=grace_s)
    _on_transcript = robot_voice.on_transcript
    shutdown = threading.Event()
    if console is not None:
        console.attach(robot_voice)
        # The twin process keeps the last physics state until something resets
        # it. The notebook's reset button does that before the console is used.
        if args.digital_twin:
            console.reset_scene()

    # ---------------- listener thread (always hot) ----------------
    if not args.text_commands:
        if llm.error:
            logger.warning("Voice is unavailable (%s). Use --text-commands.", llm.error)
        else:
            llm.start(
                _on_transcript,
                on_level=console.on_level if console is not None else None,
                on_busy=console.note_asr if console is not None else None,
                on_drop=lambda reason: ctx.emit("ignored", text="", reason=reason),
                stop_check=shutdown.is_set,
            )
    else:

        def _stdin_loop() -> None:
            """Read typed commands from stdin (--text-commands mode) and forward them to :func:`_on_transcript`."""
            while not shutdown.is_set():
                try:
                    line = input("you> ")
                except EOFError:
                    shutdown.set()
                    robot_voice.stop()
                    break
                if line.strip():
                    _on_transcript(line.strip())

        threading.Thread(target=_stdin_loop, daemon=True, name="stdin").start()

    # ---------------- dispatch loop (main thread, unless the console owns it) ----------------
    ctx.say(
        "Ready. Try: 'copy my movements', 'pick up the ball', 'wave', "
        "'dance', 'hold this', 'go home' - or 'stop robot' at any time. "
        "In the twin window: space stops, r resets the arm and the table."
    )
    try:
        if console is None:
            robot_voice.run(shutdown)
        else:
            worker = threading.Thread(
                target=robot_voice.run, args=(shutdown,), daemon=True, name="dispatch"
            )
            worker.start()
            console.run(shutdown, on_stop=lambda: _on_transcript("stop robot"))
            shutdown.set()
            ctx.stop_event.set()
            robot_voice.stop()
            worker.join(timeout=5.0)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown.set()
        ctx.stop_event.set()
        logger.info("Shutting down - parking arm.")
        for closer in (
            getattr(ctx.mount_camera, "close", None),
            getattr(ctx.arm_camera, "close", None),
            getattr(ctx.hands, "close", None),
        ):
            try:
                if closer:
                    closer()
            except Exception:
                pass
        try:
            ctx.arm.close()
        finally:
            llm.close()


if __name__ == "__main__":
    main()
