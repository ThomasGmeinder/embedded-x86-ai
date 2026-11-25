# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Captions on the twin console tiles.

The notebook console and ``run_pipeline.sh`` both draw these. A processor
name, when there is one, is added by :func:`with_processor`. The voice panel's
caption names the voice model the config selects (:func:`voice`).
"""

from __future__ import annotations

POSE = "mount: YOLOv26s-pose"
HANDS = "mount: MediaPipe hands"
DETECT = "arm: YOLOv26s-detect"
VIEW = "digital twin simulated in MuJoCo"
# The voice panel's caption before a voice model has been opened.
NO_VOICE = ("Voice: no model opened yet", "")


def voice(cfg: dict) -> tuple[str, str]:
    """The voice panel's two caption lines: the model ``voice_intent`` names in the pipeline's
    config (``config/pipeline.yaml``), by that model's own ``name``, and what it ``does``."""
    intent = str(cfg["voice_intent"])
    model = cfg[intent]
    return f"Voice: {model.get('name', intent)}", str(model.get("does", ""))


def with_processor(caption: str, processor: str) -> str:
    """``caption`` plus the processor it is running on, in brackets."""
    return f"{caption} [{processor}]"
