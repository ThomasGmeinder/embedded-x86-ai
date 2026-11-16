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

import threading
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim

NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "workshop" / "notebooks" / "03_integration" / "02_digital_twin.ipynb"
)


def _console(restore):
    """A console with no cameras, whose reset is ``restore``."""
    from vla_pipeline.console import TwinConsole

    console = TwinConsole.__new__(TwinConsole)
    console._lock = threading.Lock()
    console._resetting = False
    console._restore = None
    console.ctx = SimpleNamespace(
        arm=SimpleNamespace(reset=restore, get_joints=lambda: {}),
        shared_state=None,
        stop_event=threading.Event(),
        say=lambda _message: None,
    )
    return console


def test_run_pipeline_reset_calls_arm_reset_not_the_local_world():
    """The console reset is ``MoveItArm.reset``."""
    from vla_pipeline.console import TwinConsole

    calls = []
    arm = SimpleNamespace(reset=lambda: calls.append("reset") or True)
    assert TwinConsole._restore_twin(SimpleNamespace(ctx=SimpleNamespace(arm=arm), _restore=None)) is True
    assert calls == ["reset"]


def test_a_second_reset_does_nothing_until_the_first_finishes():
    """The button may be clicked once. The next click runs only after the reset returns."""
    started = threading.Event()
    release = threading.Event()
    calls = []

    def restore():
        calls.append("reset")
        started.set()
        release.wait()
        return True

    console = _console(restore)
    paints = []
    worker = threading.Thread(
        target=lambda: console.reset_scene(on_busy=lambda: paints.append(console.resetting))
    )
    worker.start()
    assert started.wait(5)
    console.reset_scene(on_busy=lambda: paints.append("second"))
    assert calls == ["reset"]
    assert paints == [True]
    assert console.resetting
    release.set()
    worker.join(5)
    assert not worker.is_alive()
    assert not console.resetting
    console.reset_scene()
    assert calls == ["reset", "reset"]


def test_the_reset_button_turns_gray_while_a_reset_runs():
    """The OpenCV button is orange and says reset, then gray and says resetting."""
    from vla_pipeline.console import ORANGE, RESET_BUTTON, TILE_H, VIEW_W

    console = _console(lambda: True)
    console.voice = "typed"
    console.hint = "space: stop    q: quit"
    console._level = console._level_t = 0.0
    console._speaking = False
    console._asr_pending = False
    console._behavior = None
    console._behavior_t0 = 0.0
    console._stop_flash = 0.0
    console._log = deque()
    bx, by, bw, bh = RESET_BUTTON
    corner = (by - TILE_H + 2, bx - VIEW_W + 2)

    def swatch(panel):
        return tuple(int(v) for v in panel[corner])

    idle = console._panel()
    assert swatch(idle) == ORANGE
    console._resetting = True
    busy = console._panel()
    assert swatch(busy) != ORANGE
    assert max(swatch(busy)) - min(swatch(busy)) < 8


def test_the_notebook_reset_button_stays_disabled_until_the_redraw():
    """The notebook grays its button and ignores a click until the loop sees the reset finish."""
    source = NOTEBOOK.read_text(encoding="utf-8")
    assert "if reset.disabled:" in source
    assert "resetting..." in source
    assert "if reset.disabled and not console.resetting:" in source
