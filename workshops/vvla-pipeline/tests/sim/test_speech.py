# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The microphone front end: which sounds reach Whisper, and how the rest is reported.

Synthetic blocks through the energy detector (``audio.vad.backend: energy``):
a loud block is speech, a silent one is not. A fake ``sounddevice`` stream
feeds :class:`vla_pipeline.audio.vad_listener.FastListener` the same blocks,
so no microphone is needed.
"""

from __future__ import annotations

import sys
import threading
import types

import numpy as np
import pytest

pytestmark = pytest.mark.sim

CFG = {"audio": {"vad": {"backend": "energy", "hangover_s": 1.2, "min_speech_s": 0.25, "pre_roll_s": 0.3}}}


def _blocks():
    from vla_pipeline.audio.vad_listener import BLOCK_SAMPLES

    quiet = np.zeros(BLOCK_SAMPLES, np.float32)
    loud = (0.2 * np.sin(np.linspace(0, 60 * np.pi, BLOCK_SAMPLES))).astype(np.float32)
    return quiet, loud


def _utterance(ms: int) -> list:
    """``ms`` of speech, then enough silence to end it."""
    quiet, loud = _blocks()
    return [loud] * (ms // 30) + [quiet] * 45


def _segments(blocks) -> tuple[list, list]:
    """Utterances the segmenter keeps (in blocks) and the speech (s) of those it drops."""
    from vla_pipeline.audio.vad_listener import BLOCK_SAMPLES, SpeechSegmenter, VadConfig

    seg = SpeechSegmenter(VadConfig.from_cfg(CFG))
    kept, dropped = [], []
    for block in blocks:
        audio = seg.push(block)
        if audio is not None:
            kept.append(len(audio) // BLOCK_SAMPLES)
        if seg.dropped_s:
            dropped.append(seg.dropped_s)
    return kept, dropped


def test_short_command_after_silence_is_kept():
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(270))
    assert len(kept) == 1 and not dropped


def test_short_command_right_after_an_earlier_sound_is_kept():
    """The pre-roll ring restarts empty after an utterance; a command 90 ms
    later still counts all of its speech."""
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(360) + [quiet] * 3 + _utterance(270))
    assert len(kept) == 2, f"kept {kept}, dropped {dropped}"


def test_click_is_dropped_and_measured():
    quiet, _ = _blocks()
    kept, dropped = _segments([quiet] * 20 + _utterance(120))
    assert not kept
    assert dropped and 0.05 < dropped[0] < 0.25, dropped


class _FakeStream:
    """``sounddevice.InputStream``: plays ``script`` blocks into the callback back to back."""

    script: list = []

    def __init__(self, callback=None, **_kwargs):
        self._callback = callback
        self._thread = threading.Thread(target=self._play, daemon=True)

    def _play(self):
        for block, overflow in self.script:
            self._callback(block[:, None], len(block), None, types.SimpleNamespace(input_overflow=overflow))

    def start(self):
        self._thread.start()

    def stop(self):
        pass

    def close(self):
        pass


def test_the_listener_reports_what_never_becomes_text(monkeypatch):
    from vla_pipeline.audio import vad_listener

    quiet, _ = _blocks()
    script = [(b, False) for b in [quiet] * 20 + _utterance(120) + _utterance(450)]
    script[5] = (quiet, True)  # the device lost a block while the CPU was busy
    _FakeStream.script = script
    monkeypatch.setitem(sys.modules, "sounddevice", types.SimpleNamespace(InputStream=_FakeStream))
    monkeypatch.setattr(vad_listener.FastListener, "OVERFLOW_REPORT_S", 0.0)
    # The fake plays faster than a microphone: an unbounded queue keeps every block.
    monkeypatch.setattr(vad_listener, "_AUDIO_QUEUE_BLOCKS", 0)

    texts, drops = [], []
    three = threading.Event()

    def dropped(reason):
        drops.append(reason)
        if len(drops) >= 3:
            three.set()

    mic = vad_listener.FastListener(CFG, lambda audio: "").start(texts.append, on_drop=dropped)
    mic.opened.wait()
    assert mic.error is None, mic.error
    three.wait()  # the listener's own reports: too short, no words, audio lost
    mic.stop()
    assert not texts
    assert any(d.startswith("too short to hear") for d in drops), drops
    assert "Whisper heard no words" in drops, drops
    assert any(d.startswith("microphone audio lost") for d in drops), drops
