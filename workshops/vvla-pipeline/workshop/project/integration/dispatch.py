# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODOs 3.1-3.2: how a sentence becomes a behavior.

The orchestrator never hardcodes behaviors. Handlers register on a
:class:`~vla_pipeline.llm.intents.CommandRegistry`, transcripts run a
gauntlet - stop fast path, noise gate, LLM - and whatever survives is
dispatched to whichever handler claimed that intent. Adding a voice command
to the pipeline never touches the dispatch loop; that property comes from
what you build here. The context every handler gets is the pipeline's
:class:`~vla_pipeline.user_robot_control.PipelineContext`.

Reference: ``vla_pipeline/user_robot_control.py`` (same names and parameters).
Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Self-test: ``python -m vvla_workshop.selftest_integration``
"""

from __future__ import annotations

import logging

from vla_pipeline.behaviors.manipulation import register_manipulation
from vla_pipeline.behaviors.simple import run_home, run_wave
from vla_pipeline.llm.intents import CommandRegistry, Intent, ParsedCommand, is_noise, is_stop
from vvla_workshop.project.integration.behaviors import run_dance
from vvla_workshop.project.integration.mimic import run_mimic

logger = logging.getLogger(__name__)


def build_registry() -> CommandRegistry:
    """Register every behavior on a fresh registry.

    Map :class:`Intent` members to the handlers imported above:
    GESTURE_MIMIC → ``run_mimic``, DANCE → ``run_dance``, WAVE →
    ``run_wave``, HOME → ``run_home``. (STOP is deliberately NOT a handler -
    it's an interrupt, handled before dispatch ever happens.) The provided
    manipulation behaviors (grip, release, place, pick_place, fetch_block) are
    added after yours.
    """
    registry = CommandRegistry()
    # >>> TODO 3.1: build_registry - notebooks/03_integration/01_behavior.ipynb
    raise NotImplementedError(
        "TODO 3.1 (build_registry): implement me - see notebooks/03_integration/01_behavior.ipynb")
    # <<< TODO 3.1
    register_manipulation(registry)
    return registry


def handle_transcript(
    ctx, text: str, parse, stops_before: int | None = None, command: ParsedCommand | None = None,
) -> ParsedCommand | None:
    """Route one raw transcript through the full gauntlet. Returns the parsed command.

    The order is a safety property - implement it exactly:

    1. **Stop fast path**: :func:`is_stop` on the RAW text →
       ``ctx.request_stop()``, ``ctx.say("Stopping.")``, return
       ``ParsedCommand(STOP)``. Never send a stop through the LLM.
    2. **Noise gate**: :func:`is_noise` → say nothing, do nothing, return
       ``ParsedCommand(UNKNOWN)``.
    3. **Parse**: ``heard`` is ``stops_before`` (the stop count when the
       transcript arrived) or, when that is None, ``ctx.stops_heard()``; then
       ``cmd = parse(text)``, unless ``command`` is given (already parsed
       upstream): then ``cmd = command``.
    4. **Dispatch**: UNKNOWN → tell the user you didn't catch a command.
       No handler registered (``ctx.registry.get``) → say that. Otherwise
       ``ctx.begin(heard)`` clears the stop event, so a stale stop never
       kills the *next* behavior - unless a stop came in since the
       transcript arrived: then it returns False, and the command must not
       run. Run ``handler(ctx, cmd)`` and clear the stop event again
       afterwards (in a ``finally``).
    """
    # >>> TODO 3.2: handle_transcript - notebooks/03_integration/01_behavior.ipynb
    raise NotImplementedError(
        "TODO 3.2 (handle_transcript): implement me - see notebooks/03_integration/01_behavior.ipynb")
    # <<< TODO 3.2
