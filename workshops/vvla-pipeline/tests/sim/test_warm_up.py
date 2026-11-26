# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""What the warm-up request carries.

After a typed warm-up the first spoken command was slower than every later one
(its call after 0.34 s, later ones after 0.16 s, 2026-11-26): a typed request
does not run the part of the model that hears. The warm-up of a server that
hears audio is therefore a clip. The timing is measured on the real model by
``test_voice_live.py``; here, which request is sent, against the stand-in server.
"""

from __future__ import annotations

import io
import wave

import pytest

from vla_pipeline.llm.llama_intent import LlamaIntentParser
from vla_pipeline.llm.voice import _wav
from vla_pipeline.utils.config import load_config

pytestmark = pytest.mark.sim


def _parser(mock_llm, mmproj: bool) -> LlamaIntentParser:
    cfg = load_config()
    section = cfg["gemma4-e4b-it"]
    section["host"], section["port"] = mock_llm.host, mock_llm.port
    if not mmproj:
        section.pop("mmproj", None)
    return LlamaIntentParser(cfg, autostart=False)


def _user_content(payload: dict):
    return next(m["content"] for m in payload["messages"] if m["role"] == "user")


def test_a_server_that_hears_is_warmed_up_with_the_clip(mock_llm):
    parser = _parser(mock_llm, mmproj=True)
    before = len(mock_llm.requests)
    clip = _wav(bytes(16000), 16000)
    assert parser.warm_up(clip) >= 0.0
    sent = mock_llm.requests[before:]
    assert len(sent) == 1, "the warm-up is one request"
    kinds = [part["type"] for part in _user_content(sent[0])]
    assert "input_audio" in kinds
    with wave.open(io.BytesIO(clip)) as handle:
        assert handle.getnframes() / handle.getframerate() == 0.5


def test_without_a_clip_the_warm_up_is_a_typed_request(mock_llm):
    parser = _parser(mock_llm, mmproj=True)
    before = len(mock_llm.requests)
    parser.warm_up()
    sent = mock_llm.requests[before:]
    assert len(sent) == 1 and _user_content(sent[0]) == "hello"


def test_a_server_that_does_not_hear_gets_the_typed_request_even_with_a_clip(mock_llm):
    parser = _parser(mock_llm, mmproj=False)
    before = len(mock_llm.requests)
    parser.warm_up(_wav(bytes(16000), 16000))
    sent = mock_llm.requests[before:]
    assert len(sent) == 1 and _user_content(sent[0]) == "hello"
