"""A sentence in a synthetic voice (espeak-ng), inside a quiet clip, as the bytes of a WAV file.

The voice, speaking rate, loudness and lead-in are the ``synthetic_voice``
section of ``configs/voice.toml``; the clip's length and sample rate are those
of the live session's windows, so a clip reaches Gemma as a window does.
"""

from __future__ import annotations

import ctypes
import io
import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from so101_smolvla.voice.settings import Settings, SetupFailure, load_settings

ESPEAK_ONCE = Path(__file__).with_name("espeak_once.py")  # one sentence, in a process of its own


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    """16-bit mono PCM as the bytes of a WAV file."""
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


def clip(sound: np.ndarray, sample_rate: int, window_s: float, lead_s: float) -> bytes:
    """The sound, ``lead_s`` into a quiet clip of ``window_s`` seconds, as the bytes of a WAV file."""
    total, start = int(round(window_s * sample_rate)), int(round(lead_s * sample_rate))
    if start + len(sound) > total:
        raise SetupFailure(
            f"a sentence of {len(sound) / sample_rate:.1f} s does not fit a {window_s} s clip"
        )
    samples = np.zeros(total, dtype=np.int16)
    samples[start:start + len(sound)] = sound
    return wav_bytes(samples.tobytes(), sample_rate)


class SyntheticVoice:
    """espeak-ng through its C library, set to the voice and rate of the settings; each sentence is
    made in a process of its own (``espeak_once.py``), so it sounds the same every time."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        voice = self.settings.synthetic_voice
        self.library_name: str = voice["library"]
        self.voice: str = voice["voice"]
        self.words_per_minute = int(voice["words_per_minute"])
        self.peak = float(voice["peak"])
        self.lead_s = float(voice["lead_s"])
        try:
            ctypes.CDLL(self.library_name)  # only whether it is installed; no sentence is made in this process
        except OSError:
            raise SetupFailure(f"{self.library_name} is not installed (apt install espeak-ng)") from None
        self._spoken: dict[str, tuple[int, bytes]] = {}  # per sentence: espeak-ng's rate and samples

    def _espeak(self, text: str) -> tuple[int, bytes]:
        """espeak-ng's own samples of the sentence, made in a process of their own.

        espeak-ng keeps state between sentences: asked twice in one process, the same sentence
        comes out differently, and a spoken test's clip then depends on which tests spoke before
        it (found 2026-11-23: a sentence the NPU heard right in a fresh process and wrong after
        the 45 sentences of the tests before it). A process per sentence gives the same samples
        every time; they are kept, so a sentence is made once per voice.
        """
        if text not in self._spoken:
            # One child, awaited to its exit; the bound is for a child that hangs.
            ran = subprocess.run(
                [sys.executable, str(ESPEAK_ONCE), self.library_name, self.voice, str(self.words_per_minute), text],
                capture_output=True, timeout=60,
            )
            if ran.returncode != 0 or len(ran.stdout) <= 4:
                reason = ran.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or ["no sound"]
                raise SetupFailure(f"espeak-ng could not speak {text!r}: {reason[0]}")
            self._spoken[text] = (struct.unpack("<I", ran.stdout[:4])[0], ran.stdout[4:])
        return self._spoken[text]

    def say(self, text: str, out_rate: int) -> np.ndarray:
        """The sentence as int16 samples at ``out_rate``, without the silence before and after it.
        The same sentence gives the same samples, whatever was said before it."""
        rate, samples = self._espeak(text)
        sound = np.frombuffer(samples, dtype="<i2").astype(np.float64)
        loud = np.flatnonzero(np.abs(sound) > 0.01 * np.abs(sound).max())
        sound = sound[loud[0]:loud[-1] + 1]
        sound = resample_poly(sound, out_rate, rate)
        sound *= self.peak * 32767 / np.abs(sound).max()
        return np.round(sound).astype(np.int16)

    def clip(self, text: str) -> tuple[bytes, float]:
        """The sentence inside a quiet clip of the live session's window: (WAV bytes, spoken seconds)."""
        listening = self.settings.listening
        rate, window_s = int(listening["sample_rate"]), float(listening["window_s"])
        sound = self.say(text, rate)
        return clip(sound, rate, window_s, self.lead_s), len(sound) / rate

    def recorded(self) -> dict:
        """The voice's settings, as a result's provenance records them."""
        listening = self.settings.listening
        return {
            "library": self.library_name,
            "voice": self.voice,
            "words_per_minute": self.words_per_minute,
            "peak": self.peak,
            "lead_s": self.lead_s,
            "sample_rate": int(listening["sample_rate"]),
            "clip_s": float(listening["window_s"]),
        }
