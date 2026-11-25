# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The console tile captions live in ``so101_sim.tiles`` and nowhere else."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CONSOLES = (
    REPO / "workshop/common/twin_console.py",
    REPO / "vla_pipeline/console.py",
)
CAPTIONS = (
    "mount: YOLOv26s-pose",
    "mount: MediaPipe hands",
    "arm: YOLOv26s-detect",
    "digital twin simulated in MuJoCo",
)


def test_tile_captions_are_only_in_the_shared_module():
    shared = (REPO / "so101_sim/src/so101_sim/tiles.py").read_text()
    for caption in CAPTIONS:
        assert caption in shared
        for path in CONSOLES:
            assert caption not in path.read_text(), path.name
            assert "so101_sim.tiles" in path.read_text()


def test_the_voice_panel_names_the_model_the_config_selects():
    """Both consoles take the voice panel's caption from ``so101_sim.tiles.voice``, which reads
    the model's name and what it does from the pipeline's config. Neither names a model itself."""
    from so101_sim.tiles import voice
    from vla_pipeline.llm.voice import VOICE_INTENTS
    from vla_pipeline.utils.config import load_config

    cfg = load_config()
    for intent in VOICE_INTENTS:
        name, does = voice({**cfg, "voice_intent": intent})
        assert name == f"Voice: {cfg[intent]['name']}" and does == cfg[intent]["does"], (intent, name, does)
        for line in (name, does):  # the panel is 400 px wide: about 47 characters of its font
            assert 0 < len(line) <= 47, line
    assert voice(cfg)[0] == f"Voice: {cfg[cfg['voice_intent']]['name']}"
    for path in CONSOLES:
        text = path.read_text()
        assert "voice_caption" in text, path.name
        for model in ("gemma", "Gemma", "d1-omni", "d1-Omni"):
            assert model not in text, f"{path.name} names a voice model itself: {model}"
