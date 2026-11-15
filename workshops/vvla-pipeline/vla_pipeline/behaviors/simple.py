# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Simple one-shot behaviors: wave and home.

Both are deliberately tiny: they are the canonical shape of a behavior
handler - ``(ctx, cmd)`` in, motion through ``ctx.arm``, ``ctx.stop_check``
polled so a spoken "stop robot" interrupts mid-move, park at the end - and the
template for adding new voice commands (see :mod:`vla_pipeline.llm.intents`).

Independent test
----------------
    python -m vla_pipeline.behaviors.simple --behavior wave --dry-run
    python -m vla_pipeline.behaviors.simple --behavior home
"""

from __future__ import annotations

import argparse
import logging

from vla_pipeline.user_robot_control import PipelineContext
from vla_pipeline.robot.arm_interface import make_arm, radians_from_calibrated
from vla_pipeline.utils.config import load_config

logger = logging.getLogger(__name__)

# Arm raised forward and up, open jaws turned to face the person like a palm.
# The hand sits in front of the pan axis, so the pan sweep moves it side to
# side; the wrist joints can't (flex bends toward/away, roll only twists).
WAVE_UP = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": -45.0,
    "wrist_flex": -30.0,
    "wrist_roll": 85.0,
    "gripper": 90.0,
}
WAVE_LEFT = {"shoulder_pan": -25.0}
WAVE_RIGHT = {"shoulder_pan": 25.0}



def _go_to(ctx, pose: dict, duration_s: float | None = None, velocity_scale: float | None = None) -> str | None:
    """Move to ``pose``.

    The speed is the global scale in ``joint_limits.yaml`` unless
    ``velocity_scale`` overrides that one move. ``duration_s`` plays the pose
    as one timed keyframe instead.
    """
    kwargs = {"stop_event": ctx.stop_event}
    if duration_s is not None:
        kwargs["duration_s"] = duration_s
    if velocity_scale is not None:
        kwargs["velocity_scale"] = velocity_scale
    how = ctx.arm.go_to(radians_from_calibrated(ctx.cfg, pose), **kwargs)
    logger.info("go_to %s", how)
    return how


def _rest(ctx) -> None:
    ctx.arm.go_to_rest()


def run_wave(ctx, cmd=None) -> None:
    """Raise the arm and wave ``wave.cycles`` times."""
    cycles = int((ctx.cfg.get("wave") or {}).get("cycles", 3))
    logger.info("[wave] waving %d times", cycles)
    # These times override the servo envelope: the raise is 2 s, each sweep 0.4 s.
    _go_to(ctx, WAVE_UP, 2.0)
    for _ in range(cycles):
        if ctx.stop_check():
            break
        _go_to(ctx, WAVE_LEFT, 0.4)
        _go_to(ctx, WAVE_RIGHT, 0.4)
    _rest(ctx)


def run_home(ctx, cmd=None) -> None:
    """Return to the rest pose. A locked grip stays locked: "release" opens
    it, and gesture-mimic hands the gripper to the pinch."""
    logger.info("[home] returning to rest pose")
    _rest(ctx)


def main() -> None:
    """CLI entry point: run the wave or home behavior standalone."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Simple behaviors - component test")
    parser.add_argument("--behavior", choices=["wave", "home"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    arm = make_arm(cfg)
    try:
        {"wave": run_wave, "home": run_home}[args.behavior](PipelineContext(cfg=cfg, arm=arm))
        print(f"\n{args.behavior} PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
