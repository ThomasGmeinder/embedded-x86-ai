# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODO 3.4: dance, a behavior handler with a beat clock.

The pipeline's ``run_wave``/``run_home`` (:mod:`vla_pipeline.behaviors.simple`)
are the canonical shape of a behavior handler: ``(ctx, cmd)`` in, motion
through ``ctx.arm``, ``ctx.stop_check`` polled so a spoken "stop" interrupts
mid-move, park at the end. ``run_dance`` is the same shape with a beat clock;
the choreography (:data:`~vla_pipeline.behaviors.dance.DANCE_MOVES`) is
provided, the control flow is the exercise.

Reference: ``vla_pipeline/behaviors/dance.py`` (same name and parameters).
Notebook: ``notebooks/03_integration/01_behavior.ipynb``
Self-test: ``python -m vvla_workshop.selftest_integration``
"""

from __future__ import annotations

import itertools
import logging
import random
import time

from vla_pipeline.behaviors.dance import DANCE_MOVES

logger = logging.getLogger(__name__)


def run_dance(ctx, cmd=None, duration_s: float | None = None, seed: int | None = None) -> None:
    """Beat-paced keyframe choreography until the clock (or a stop) ends it.

    Read ``dance.bpm`` and ``dance.duration_s`` from ``ctx.cfg`` (honor the
    ``duration_s`` override). One beat is ``60/bpm`` seconds. Shuffle the
    move order (:data:`DANCE_MOVES`, with ``random.Random(seed)``), then cycle
    through it; play each move's keyframes x4 with ``ctx.arm.go_to(keyframe,
    duration_s=beat*0.9, stop_check=ctx.stop_check)``. Check the deadline
    AND ``ctx.stop_check()`` between keyframes - a dance that can't be
    stopped is a safety bug, not a feature. A stop holds the current pose;
    otherwise park with ``ctx.arm.go_to_rest()`` at the end.
    """
    d = ctx.cfg["dance"]
    # >>> TODO 3.4: run_dance - notebooks/03_integration/01_behavior.ipynb
    raise NotImplementedError(
        "TODO 3.4 (run_dance): implement me - see notebooks/03_integration/01_behavior.ipynb")
    # <<< TODO 3.4
