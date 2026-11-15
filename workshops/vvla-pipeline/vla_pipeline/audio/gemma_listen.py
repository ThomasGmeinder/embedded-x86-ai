# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Silero gate and rolling audio windows for Gemma.

Frames arrive from the microphone. A hop emits the latest window only when
that hop contains enough speech. Nothing here sleeps to discover speech.
"""

from __future__ import annotations

import math
from array import array
from collections import deque


FRAME_MS = 20


def dbfs(pcm: bytes) -> float:
    """Level of one int16 PCM frame, in decibels relative to full scale."""
    values = array("h", pcm)
    rms = math.sqrt(sum(x * x for x in values) / max(1, len(values)))
    return 20 * math.log10(max(rms / 32768, 1e-6))


class HopWindows:
    """Keep the latest window and emit it once per hop of qualified speech."""

    def __init__(self, sample_rate: int, window_s: float, hop_s: float, min_voice_ms: int):
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self.frames: deque[tuple[bytes, bool]] = deque(maxlen=round(window_s * 1000 / FRAME_MS))
        self.hop = round(hop_s * 1000 / FRAME_MS)
        self.min_voice_frames = max(1, round(min_voice_ms / FRAME_MS))
        self.count = 0

    def feed(self, pcm: bytes, qualified: bool) -> bytes | None:
        """Append one 20 ms frame. Return the window PCM when a hop is due."""
        if len(pcm) != self.frame_bytes:
            raise ValueError("Invalid PCM frame length")
        self.frames.append((pcm, qualified))
        self.count += 1
        if self.count % self.hop:
            return None
        recent = list(self.frames)[-self.hop :]
        if sum(1 for _, spoke in recent if spoke) < self.min_voice_frames:
            return None
        return b"".join(frame for frame, _ in self.frames)


class CommandFilter:
    """Drop unknown, stale, and repeated commands from overlapping windows."""

    def __init__(self, cooldown_s: float):
        self.cooldown_s = cooldown_s
        self.accepted: dict[str, tuple[float, float]] = {}

    def accept(self, command: str, window_start: float, window_end: float, now: float) -> bool:
        if not command or command == "unknown":
            return False
        last = self.accepted.get(command)
        if last and (window_start <= last[0] or now - last[1] < self.cooldown_s):
            return False
        self.accepted[command] = (window_end, now)
        return True
