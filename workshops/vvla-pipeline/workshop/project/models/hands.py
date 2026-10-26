# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODO 1.4: MediaPipe Hands on the CPU.

The pipeline's :class:`~vla_pipeline.vision.mediapipe_hands.HandTracker`
turns one detection into the mimic's control signals (pinch, hand size, roll);
it takes a prebuilt tracker, so ``HandTracker(cfg, hands=create_hand_tracker(cfg))``
runs it on yours.

Reference: ``vla_pipeline/vision/mediapipe_hands.py`` (same name and parameters).
Notebook: ``notebooks/01_ai_models/02_cpu_mediapipe.ipynb``
Self-test: ``python -m vvla_workshop.selftest_models``
"""

from __future__ import annotations


def create_hand_tracker(cfg: dict):
    """Build a configured ``mediapipe.solutions.hands.Hands`` (CPU).

    Read ``mediapipe:`` from the config: ``max_num_hands``,
    ``min_detection_confidence``, ``min_tracking_confidence``. Video mode
    (``static_image_mode=False``) so the tracker exploits temporal continuity
    between frames.
    """
    import mediapipe as mp  # heavy import, keep local

    m = (cfg or {}).get("mediapipe", {})
    # >>> TODO 1.4: create_hand_tracker - notebooks/01_ai_models/02_cpu_mediapipe.ipynb
    raise NotImplementedError(
        "TODO 1.4 (create_hand_tracker): implement me - see notebooks/01_ai_models/02_cpu_mediapipe.ipynb")
    # <<< TODO 1.4
