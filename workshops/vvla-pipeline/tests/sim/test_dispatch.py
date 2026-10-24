# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline dispatcher's stop fast path and the LLM parse.

The stand-in LLM hears "stop" through ``Dispatcher.on_transcript`` while it
is still parsing, as the always-hot listener delivers it on the rig.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim


def _dispatcher(parse):
    from vla_pipeline.dispatch import Dispatcher
    from vla_pipeline.llm.intents import Intent

    ran, said, events = [], [], []
    ctx = SimpleNamespace(say=said.append, stop_event=threading.Event(), arm=None)
    registry = {Intent.WAVE: lambda _ctx, cmd: ran.append(cmd.intent)}
    d = Dispatcher(ctx, parse, registry, is_stop=lambda text: text == "stop")
    d.listeners.append(lambda event, **_data: events.append(event))
    return d, ctx, ran, said, events


def test_a_stop_said_while_the_llm_parses_cancels_the_command():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    held = {}

    def parse(text):
        held["dispatcher"].on_transcript("stop")
        return ParsedCommand(Intent.WAVE, transcript=text)

    d, ctx, ran, said, events = _dispatcher(parse)
    held["dispatcher"] = d
    d.dispatch("wave at me")

    assert ran == [], "the command the stop overlapped ran anyway"
    assert "cancelled" in events, events
    assert ctx.stop_event.is_set(), "the stop event was cleared"
    assert "Stopping." in said


def test_a_stop_said_before_the_command_does_not_cancel_it():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    d, ctx, ran, _said, events = _dispatcher(lambda text: ParsedCommand(Intent.WAVE, transcript=text))
    d.on_transcript("stop")
    d.dispatch("wave at me")

    assert ran == [Intent.WAVE]
    assert "cancelled" not in events
    assert not ctx.stop_event.is_set()
