# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Pipeline orchestrator: microphone → Whisper (CPU by default, optional NPU) → Llama intent (iGPU) →
behavior dispatch, with an always-hot listener for the safety stop.

Architecture
------------
- A **listener thread** runs :class:`FastListener` continuously (VAD-based,
  ~0.35 s end-of-speech) and pushes transcripts onto a queue - the mic stays
  hot *while behaviors run*, which is what makes a spoken "stop" able to
  interrupt a running behavior.
- ``STOP`` is matched with plain keywords on the raw transcript
  (:data:`STOP_KEYWORDS`) before any LLM round trip, then sets the shared
  stop event that every behavior's ``stop_check`` polls (including mid
  ``go_to`` interpolation). The 'q' key in any preview window and Ctrl-C
  do the same.
- Every other transcript goes through the grammar-constrained intent parser
  and is dispatched via the :class:`CommandRegistry` - adding a voice command
  never touches this file (see :mod:`vla_pipeline.llm.intents`).
- Behaviors run on the **main thread** (OpenCV preview windows want that);
  a new command interrupts the current behavior (graceful stop → switch).

Run
---
    python -m vla_pipeline.main                  # full pipeline
    python -m vla_pipeline.main --text-commands  # type instead of speak
    python -m vla_pipeline.main --dry-run        # no robot hardware
"""

from __future__ import annotations

import argparse
import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field

from vla_pipeline.llm.intents import (
    STOP_KEYWORDS,
    CommandRegistry,
    Intent,
    ParsedCommand,
)
from vla_pipeline.utils.config import ensure_dirs, load_config

logger = logging.getLogger(__name__)

registry = CommandRegistry()


# =============================================================================
# Context shared with behavior handlers
# =============================================================================


@dataclass
class PipelineContext:
    """Shared state passed to every behavior handler: robot arm, vision models, camera clients, and stop/say hooks."""

    cfg: dict
    arm: object
    pose: object | None = None  # PoseEstimator (NPU)
    detector: object | None = None  # ObjectDetector (NPU)
    hands: object | None = None  # HandTracker (CPU)
    mount_camera: object | None = None  # CameraClient
    arm_camera: object | None = None  # CameraClient
    shared_state: dict = field(default_factory=dict)
    stop_event: threading.Event = field(default_factory=threading.Event)
    headless: bool = False

    def stop_check(self) -> bool:
        """Return True if a stop has been requested."""
        return self.stop_event.is_set()

    def say(self, message: str) -> None:
        """Print message to the console and log it."""
        print(f"{message}")
        logger.info("[say] %s", message)


# =============================================================================
# Behavior handlers (the registry IS the dispatch table)
# =============================================================================


@registry.register(Intent.GESTURE_MIMIC)
def _h_gesture_mimic(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_gesture_mimic`."""
    from vla_pipeline.behaviors.gesture_mimic import run_gesture_mimic

    if ctx.pose is None or ctx.mount_camera is None:
        ctx.say("Gesture mimic needs the mount camera and pose model.")
        return
    run_gesture_mimic(
        ctx.cfg,
        ctx.arm,
        ctx.pose,
        ctx.hands,
        ctx.mount_camera,
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        show_preview=not ctx.headless,
    )


@registry.register(Intent.FETCH_BLOCK)
def _h_fetch_block(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_fetch_block`."""
    from vla_pipeline.behaviors.fetch_block import run_fetch_block

    run_fetch_block(
        ctx.cfg, ctx.arm, ctx.pose, ctx.mount_camera, stop_check=ctx.stop_check
    )


@registry.register(Intent.PICK_PLACE)
def _h_pick_place(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_pick_place` for the requested object."""
    from vla_pipeline.behaviors.pick_place import run_pick_place

    if ctx.detector is None:
        ctx.say("Object detection isn't available - can't do pick and place.")
        return
    run_pick_place(
        ctx.cfg,
        ctx.arm,
        ctx.detector,
        ctx.pose,
        ctx.arm_camera,
        ctx.mount_camera,
        object_name=cmd.object_name or "",
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        feedback=ctx.say,
    )


@registry.register(Intent.DANCE)
def _h_dance(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_dance`."""
    from vla_pipeline.behaviors.dance import run_dance

    run_dance(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


@registry.register(Intent.GRIP)
def _h_grip(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_grip`."""
    from vla_pipeline.behaviors.grip import run_grip

    run_grip(
        ctx.cfg,
        ctx.arm,
        shared_state=ctx.shared_state,
        stop_check=ctx.stop_check,
        arm_camera=ctx.arm_camera,
        feedback=ctx.say,
    )


@registry.register(Intent.WAVE)
def _h_wave(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_wave`."""
    from vla_pipeline.behaviors.simple import run_wave

    run_wave(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


@registry.register(Intent.HOME)
def _h_home(ctx: PipelineContext, cmd: ParsedCommand) -> None:
    """Dispatch to :func:`run_home`."""
    from vla_pipeline.behaviors.simple import run_home

    run_home(ctx.cfg, ctx.arm, stop_check=ctx.stop_check)


# =============================================================================
# Stop fast path
# =============================================================================

_STOP_RE = re.compile(r"\b(" + "|".join(STOP_KEYWORDS) + r")\b", re.IGNORECASE)


def is_stop(transcript: str) -> bool:
    """Return True if transcript contains a stop keyword."""
    return bool(_STOP_RE.search(transcript or ""))


# =============================================================================
# Noise / hallucination gate
# =============================================================================
# Whisper, fed the first burst of room noise the moment the mic goes hot,
# reliably hallucinates a few stock phrases ("you", "thank you.", "Bye.",
# subtitle credits, etc.). With the grammar forcing every transcript into one
# valid command, such a phantom utterance used to dispatch a real behavior - the
# robot appearing to wave/dance on its own at startup. We drop these before they
# ever reach the LLM. The set is matched on the transcript stripped of
# whitespace/punctuation and lowercased; multi-word phrases are matched whole.
DEFAULT_NOISE_PHRASES = (
    "",
    "you",
    "thank you",
    "thanks",
    "thank you.",
    "bye",
    "bye.",
    "goodbye",
    "hello",
    "hi",
    "hmm",
    "uh",
    "um",
    "ah",
    "oh",
    "yeah",
    "okay",
    "ok",
    "the",
    "a",
    "so",
    "and",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by the amara.org community",
    ".",
    "..",
    "...",
)


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip surrounding whitespace and trailing punctuation/quotes."""
    return re.sub(r"[\s\.\,\!\?\"\'`]+$", "", (text or "").strip().lower()).strip()


def make_noise_gate(cfg: dict):
    """Build an ``is_noise(text) -> bool`` predicate from config.

    ``audio.noise_phrases`` (optional) extends/overrides the defaults; a very
    short transcript with no alphabetic content is always treated as noise.
    """
    a = cfg.get("audio", {}) or {}
    phrases = a.get("noise_phrases")
    phrase_set = {
        (_normalize_transcript(p))
        for p in (phrases if phrases is not None else DEFAULT_NOISE_PHRASES)
    }

    def is_noise(text: str) -> bool:
        """Return True if text looks like noise/hallucination rather than a real command."""
        norm = _normalize_transcript(text)
        if not norm:
            return True
        if not any(c.isalpha() for c in norm):
            return True  # pure punctuation / digits
        return norm in phrase_set

    return is_noise


# =============================================================================
# Orchestrator
# =============================================================================


def build_context(cfg: dict, args) -> PipelineContext:
    """Build the :class:`PipelineContext`, wiring up the arm, cameras, and vision models.

    Each vision component is optional: a failure to construct it is logged and
    left as None so the pipeline still runs with degraded capabilities.
    """
    from vla_pipeline.robot.arm_interface import DryRunArm, make_arm

    arm = DryRunArm(verbose=False) if args.dry_run else make_arm(cfg)
    ctx = PipelineContext(cfg=cfg, arm=arm, headless=args.headless)

    from vla_pipeline.vision.camera import make_camera

    try:
        ctx.mount_camera = make_camera(cfg, "mount")
    except Exception as e:
        logger.warning("Mount camera unavailable: %s", e)
    try:
        ctx.arm_camera = make_camera(cfg, "arm")
    except Exception as e:
        logger.warning("Arm camera unavailable: %s", e)

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


def main() -> None:
    """Parse CLI args, build the pipeline context, and run the listen/dispatch loop until interrupted."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Strix VLA pipeline orchestrator")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--text-commands",
        action="store_true",
        help="type commands instead of speaking (skips Whisper)",
    )
    parser.add_argument("--dry-run", action="store_true", help="no robot hardware")
    parser.add_argument("--headless", action="store_true", help="no preview windows")
    parser.add_argument(
        "--device",
        choices=["npu", "cpu"],
        default=None,
        help="override the device for all ONNX models",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    ctx = build_context(cfg, args)

    is_noise = make_noise_gate(cfg)
    # Ignore any transcript for a short grace window after launch - the mic goes
    # hot before the user is ready and the first frames are usually room noise
    # that Whisper hallucinates into words. Voice ('stop' included) only starts
    # being honored once the window passes. Tunable via audio.startup_grace_s.
    startup_grace_s = float((cfg.get("audio", {}) or {}).get("startup_grace_s", 1.5))
    start_time = time.monotonic()

    from vla_pipeline.llm.llama_intent import LlamaIntentParser

    llm = LlamaIntentParser(cfg)
    transcripts: queue.Queue[str] = queue.Queue()
    shutdown = threading.Event()

    # ---------------- listener thread (always hot) ----------------
    if not args.text_commands:
        from vla_pipeline.audio.vad_listener import FastListener
        from vla_pipeline.audio.whisper_npu import WhisperNPU

        whisper = WhisperNPU(cfg, device=args.device)
        listener = FastListener(cfg, whisper.transcribe)

        def _listen() -> None:
            """Run the always-hot listener loop, forwarding transcripts to :func:`_on_transcript`."""
            listener.run_forever(
                on_transcript=lambda t: _on_transcript(t),
                stop_check=shutdown.is_set,
            )

        threading.Thread(target=_listen, daemon=True, name="listener").start()
    else:

        def _stdin_loop() -> None:
            """Read typed commands from stdin (--text-commands mode) and forward them to :func:`_on_transcript`."""
            while not shutdown.is_set():
                try:
                    line = input("you> ")
                except EOFError:
                    shutdown.set()
                    break
                if line.strip():
                    _on_transcript(line.strip())

        threading.Thread(target=_stdin_loop, daemon=True, name="stdin").start()

    def _on_transcript(text: str) -> None:
        """Handle one transcript: apply the startup grace window, the stop fast-path, and the noise gate, then queue it for dispatch."""
        logger.info("Heard: %r", text)
        # Startup grace: swallow the mic's first noisy moments so a hallucinated
        # transcript can't dispatch a behavior before the user has spoken.
        if time.monotonic() - start_time < startup_grace_s:
            logger.info("Ignoring %r (startup grace period).", text)
            return
        if is_stop(text):
            # Safety stop: bypass the LLM entirely and interrupt NOW.
            ctx.say("Stopping.")
            ctx.stop_event.set()
            return
        # Drop known silence-hallucination phrases before the LLM/grammar can
        # coerce them into a real command (the spurious-startup-behavior fix).
        if is_noise(text):
            logger.info("Ignoring %r (noise/hallucination gate).", text)
            return
        transcripts.put(text)

    # ---------------- dispatch loop (main thread) ----------------
    ctx.say(
        "Ready. Try: 'copy my movements', 'pick up the ball', 'wave', "
        "'dance', 'hold this', 'go home' - or 'stop' at any time."
    )
    try:
        while not shutdown.is_set():
            try:
                text = transcripts.get(timeout=0.2)
            except queue.Empty:
                continue
            cmd = llm.parse(text)
            if cmd.intent in (Intent.UNKNOWN, Intent.STOP):
                if cmd.intent is Intent.UNKNOWN:
                    ctx.say("I didn't catch a command in that.")
                continue
            handler = registry.get(cmd.intent)
            if handler is None:
                ctx.say(f"No behavior registered for '{cmd.intent.value}'.")
                continue

            ctx.stop_event.clear()
            logger.info("=== %s %s ===", cmd.intent.value, cmd.params or "")
            try:
                handler(ctx, cmd)  # runs on the main thread (cv2 previews)
            except Exception:
                logger.exception("Behavior %s crashed - arm to rest.", cmd.intent.value)
                try:
                    ctx.arm.go_to_rest()
                except Exception:
                    pass
            ctx.stop_event.clear()

            # Seamless switching: drain queued commands except the newest, so
            # a backlog of utterances doesn't replay stale behaviors.
            newest = None
            while True:
                try:
                    newest = transcripts.get_nowait()
                except queue.Empty:
                    break
            if newest is not None:
                transcripts.put(newest)
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
