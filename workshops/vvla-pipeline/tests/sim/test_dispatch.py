# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The stop fast path and the LLM parse: the pipeline's dispatcher and its
``handle_transcript`` (the answer key of the workshop's TODO 3.2).

The stand-in LLM hears "stop" while it is still parsing, as the always-hot
listener delivers it on the rig.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


def _context():
    """A context whose registry records each wave it runs, and what it says."""
    from vla_pipeline.dispatch import PipelineContext
    from vla_pipeline.llm.intents import CommandRegistry, Intent

    ran, said = [], []
    registry = CommandRegistry()
    registry.add(Intent.WAVE, lambda _ctx, cmd: ran.append(cmd.intent))
    ctx = PipelineContext(cfg={}, arm=None, registry=registry)
    ctx.say = said.append
    return ctx, ran, said


def _dispatcher(parse):
    from vla_pipeline.dispatch import Dispatcher

    ctx, ran, said = _context()
    events = []
    d = Dispatcher(ctx, parse)
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


def test_a_stop_discards_a_command_already_queued():
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    d, _ctx, ran, _said, events = _dispatcher(
        lambda text: ParsedCommand(Intent.WAVE, transcript=text)
    )
    d.on_transcript("wave at me")
    d.on_transcript("stop")

    generation, text = d.transcripts.get_nowait()
    d.dispatch(text, generation)

    assert ran == [], "a command queued before stop ran anyway"
    assert "cancelled" in events, events


def test_a_parse_failure_does_not_kill_dispatch():
    d, _ctx, ran, said, events = _dispatcher(
        lambda _text: (_ for _ in ()).throw(RuntimeError("llama unavailable"))
    )

    assert d.dispatch("wave at me") is None
    assert ran == []
    assert "parse_error" in events
    assert any("language service" in message for message in said)


def test_handle_transcript_drops_a_command_whose_parse_a_stop_overlapped():
    from vla_pipeline.dispatch import handle_transcript
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    ctx, ran, _said = _context()

    def parse(text):
        handle_transcript(ctx, "stop", parse)
        return ParsedCommand(Intent.WAVE, transcript=text)

    handle_transcript(ctx, "wave at me", parse)

    assert ran == [], "the command the stop overlapped ran anyway"
    assert ctx.stop_event.is_set(), "the stop event was cleared"


def test_a_stop_said_before_handle_transcript_does_not_cancel_the_command():
    from vla_pipeline.dispatch import handle_transcript
    from vla_pipeline.llm.intents import Intent, ParsedCommand

    ctx, ran, _said = _context()
    handle_transcript(ctx, "stop", None)
    handle_transcript(ctx, "wave at me", lambda text: ParsedCommand(Intent.WAVE, transcript=text))

    assert ran == [Intent.WAVE]
    assert not ctx.stop_event.is_set()
