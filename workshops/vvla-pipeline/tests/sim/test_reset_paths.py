# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``run_pipeline.sh --digital-twin`` resets through ``MoveItArm.reset``.

The console has no in-process physics world, so a reset is that one call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim


def test_run_pipeline_reset_calls_arm_reset_not_the_local_world():
    """The console reset is ``MoveItArm.reset``."""
    from vla_pipeline.console import TwinConsole

    calls = []
    arm = SimpleNamespace(reset=lambda: calls.append("reset") or True)
    assert TwinConsole._restore_twin(SimpleNamespace(ctx=SimpleNamespace(arm=arm), _restore=None)) is True
    assert calls == ["reset"]
