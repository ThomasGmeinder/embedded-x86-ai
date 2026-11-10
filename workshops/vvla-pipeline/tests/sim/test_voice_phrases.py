# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Stop and home phrases, and the rolling window that feeds Gemma.

No microphone and no model download. "stop robot" is the only stop phrase.
"go home" is the only home phrase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vla_pipeline.audio.gemma_listen import FRAME_MS, CommandFilter, HopWindows
from vla_pipeline.llm.intents import command_from_json, keyword_parse
from vla_pipeline.main import is_stop

# llama.cpp's grammar lexer: a name is letters, digits, and '-'. '_' ends the
# name, and '"' ends a string even inside a character class. Both made
# llama-server reject intent.gbnf with "expecting newline or end".
_GRAMMAR_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-")

pytestmark = pytest.mark.sim


def test_stop_robot_is_the_only_stop_phrase():
    assert is_stop("stop robot")
    assert is_stop("Please STOP ROBOT now")
    for text in ("stop", "halt", "freeze", "stop it", "stop talking", "abort"):
        assert not is_stop(text), text


def test_stop_json_is_a_stop_without_a_second_parse():
    raw = '{"command": "stop", "object": ""}'
    assert is_stop(raw)
    cmd = command_from_json(raw)
    assert cmd is not None and cmd.intent.value == "stop"
    assert command_from_json("pick up the ball") is None


def test_heard_phrase_is_kept_with_the_command():
    raw = '{"command": "pick_place", "object": "football", "heard": "pick up the football"}'
    cmd = command_from_json(raw)
    assert cmd is not None
    assert cmd.intent.value == "pick_place"
    assert cmd.object_name == "football"
    assert cmd.transcript == "pick up the football"


def test_go_home_is_the_only_home_phrase():
    assert keyword_parse("go home").intent.value == "home"
    for text in ("home", "rest", "reset", "reset position"):
        assert keyword_parse(text).intent.value != "home", text


def test_keyword_pick_still_names_the_object():
    cmd = keyword_parse("pick up the ball")
    assert cmd.intent.value == "pick_place"
    assert cmd.object_name == "ball"


def test_hop_emits_only_when_the_hop_has_speech():
    rate = 16000
    windows = HopWindows(rate, window_s=0.06, hop_s=0.02, min_voice_ms=FRAME_MS)
    frame = b"\x00\x00" * (rate * FRAME_MS // 1000)
    assert windows.feed(frame, False) is None
    heard = windows.feed(frame, True)
    assert heard is not None and heard.endswith(frame)


def test_silence_does_not_emit_a_window():
    rate = 16000
    windows = HopWindows(rate, window_s=0.06, hop_s=0.04, min_voice_ms=FRAME_MS)
    frame = b"\x00\x00" * (rate * FRAME_MS // 1000)
    assert windows.feed(frame, False) is None
    assert windows.feed(frame, False) is None
    assert windows.feed(frame, False) is None


def test_repeated_command_inside_the_cooldown_is_dropped():
    gate = CommandFilter(cooldown_s=5)
    assert gate.accept("pick_place", 0.0, 1.0, now=1.0)
    assert not gate.accept("pick_place", 0.5, 1.5, now=1.5)
    assert gate.accept("wave", 0.5, 1.5, now=1.5)
    assert not gate.accept("unknown", 2.0, 3.0, now=3.0)


def _llama_grammar_fault(grammar: str) -> str | None:
    """The fault llama.cpp reports, or None when this file would parse.

    Names stop at '_' (``heard_char`` is read as the rule ``heard``). A
    double quote inside ``[^"]`` ends the string instead of the class.
    """
    i, n = 0, len(grammar)
    while i < n:
        ch = grammar[i]
        if ch in " \t\r\n":
            i += 1
            continue
        if ch == "#":
            while i < n and grammar[i] not in "\r\n":
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n and grammar[i] != '"':
                i += 2 if grammar[i] == "\\" and i + 1 < n else 1
            i += 1
            continue
        if ch == "[":
            i += 1
            if i < n and grammar[i] == "^":
                i += 1
            while i < n and grammar[i] != "]":
                if grammar[i] == '"':
                    return "quote inside a character class"
                i += 1
            i += 1
            continue
        if ch in _GRAMMAR_NAME:
            while i < n and grammar[i] in _GRAMMAR_NAME:
                i += 1
            if i < n and grammar[i] == "_":
                return "name stops before '_'"
            continue
        i += 1
    return None


def test_intent_grammar_is_one_llama_cpp_accepts():
    grammar = Path("config/intent.gbnf").read_text(encoding="utf-8")
    assert _llama_grammar_fault(grammar) is None
    assert '"heard"' in grammar
    # The two forms that made llama-server answer 400 for every spoken command.
    assert _llama_grammar_fault("heard_char ::= [a-z]\n") == "name stops before '_'"
    assert _llama_grammar_fault('hchar ::= [^"a]\n') == "quote inside a character class"
