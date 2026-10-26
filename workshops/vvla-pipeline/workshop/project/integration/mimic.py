# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODO 3.3: one tick of gesture mimic - the workshop's flagship composition.

One tick is the whole pipeline in miniature::

    camera frame ──► YOLO-pose (NPU) ──► wrist x/y + forearm ─┐
                └──► MediaPipe (CPU) ──► size/roll/pinch ─────┴──► mapper ──► arm

Everything around the tick - the loop, the preview window, the HUD that
attributes each joint to the TODO that feeds it - is the pipeline's
:func:`~vla_pipeline.behaviors.gesture_mimic.run_mimic`; ``run_mimic`` below
runs it on YOUR tick.

Reference: ``vla_pipeline/behaviors/gesture_mimic.py`` (same name and parameters).
Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Self-test: ``python -m vvla_workshop.selftest_integration``
"""

from __future__ import annotations

from vla_pipeline.behaviors import gesture_mimic
from vla_pipeline.behaviors.gesture_mimic import MimicSignals
from vla_pipeline.vision.yolo_pose_npu import pick_arm, pick_nearest


def mimic_tick(ctx, frame) -> MimicSignals:
    """Turn ONE camera frame into ONE arm command.

    The composition, in order:

    1. ``ctx.pose.infer(frame)`` → boxes/scores/kpts; keep only the NEAREST
       person (largest box) via :func:`pick_nearest`, then pick a wrist AND
       its elbow with :func:`pick_arm` (threshold: ``ctx.pose.kpt_threshold``).
    2. ``ctx.hands.process(frame)`` → a HandObservation (or None) carrying
       hand_size / roll_ti / roll_ref / roll_ti_weight / roll / pinch.
    3. ``ctx.mapper.step(...)`` with everything from 1-2 (pass the elbow as
       ``elbow_xy=`` - wrist_flex measures your hand's direction against it
       to read the WRIST BEND; the hand's ``roll`` goes in as
       ``roll_fallback=``) → a clamped joint dict - then
       ``ctx.arm.send_joints(joints)``. If NEITHER a wrist NOR a hand was
       seen, send nothing (hold pose).

    Return a :class:`MimicSignals` with everything you used - the HUD shows
    your work. Keep the RAW results from ``ctx.hands.process`` too (the
    ``raw`` field) so the UI can draw the MediaPipe hand overlay.
    """
    # >>> TODO 3.3: mimic_tick - notebooks/03_integration/01_behavior.ipynb
    raise NotImplementedError(
        "TODO 3.3 (mimic_tick): implement me - see notebooks/03_integration/01_behavior.ipynb")
    # <<< TODO 3.3


def run_mimic(ctx, cmd=None):
    """PROVIDED: the pipeline's mimic loop, on your tick."""
    return gesture_mimic.run_mimic(ctx, cmd, tick=mimic_tick)
