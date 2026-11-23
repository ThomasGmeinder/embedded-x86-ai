"""A sentence in a synthetic voice (espeak-ng), inside a quiet clip, as the bytes of a WAV file.

The voice, speaking rate, loudness and lead-in are the ``synthetic_voice``
section of ``configs/voice.toml``; the clip's length and sample rate are those
of the live session's windows, so a clip reaches Gemma as a window does.
"""

from __future__ import annotations

import ctypes
import io
import wave

import numpy as np
from scipy.signal import resample_poly

from so101_smolvla.voice.settings import Settings, SetupFailure, load_settings


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
    """espeak-ng through its C library, set to the voice and rate of the settings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        voice = self.settings.synthetic_voice
        self.library_name: str = voice["library"]
        self.voice: str = voice["voice"]
        self.words_per_minute = int(voice["words_per_minute"])
        self.peak = float(voice["peak"])
        self.lead_s = float(voice["lead_s"])
        try:
            lib = ctypes.CDLL(self.library_name)
        except OSError:
            raise SetupFailure(f"{self.library_name} is not installed (apt install espeak-ng)") from None
        lib.espeak_Initialize.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
        lib.espeak_Initialize.restype = ctypes.c_int
        lib.espeak_SetVoiceByName.argtypes = [ctypes.c_char_p]
        lib.espeak_SetVoiceByName.restype = ctypes.c_int
        lib.espeak_SetParameter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.espeak_SetParameter.restype = ctypes.c_int
        lib.espeak_Synth.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
            ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p,
        ]
        lib.espeak_Synth.restype = ctypes.c_int
        lib.espeak_Synchronize.restype = ctypes.c_int
        rate = lib.espeak_Initialize(2, 0, None, 0)  # 2 = AUDIO_OUTPUT_SYNCHRONOUS: samples go to the callback
        if rate <= 0:
            raise SetupFailure(f"espeak_Initialize returned {rate}")
        if lib.espeak_SetVoiceByName(self.voice.encode()) != 0:
            raise SetupFailure(f"espeak-ng has no voice {self.voice!r}")
        if lib.espeak_SetParameter(1, self.words_per_minute, 0) != 0:  # 1 = espeakRATE
            raise SetupFailure("espeak-ng did not take the speaking rate")
        self._lib, self.rate = lib, rate
        self._callback_type = ctypes.CFUNCTYPE(
            ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p
        )

    def say(self, text: str, out_rate: int) -> np.ndarray:
        """The sentence as int16 samples at ``out_rate``, without the silence before and after it."""
        chunks = []

        def collect(samples, count, _events):
            if samples and count > 0:
                chunks.append(np.ctypeslib.as_array(samples, shape=(count,)).copy())
            return 0

        callback = self._callback_type(collect)
        self._lib.espeak_SetSynthCallback(callback)
        data = text.encode("utf-8")
        # position 0, counted in characters (1), no end position, text is UTF-8 (flag 1)
        if self._lib.espeak_Synth(data, len(data) + 1, 0, 1, 0, 1, None, None) != 0:
            raise SetupFailure(f"espeak-ng could not speak {text!r}")
        self._lib.espeak_Synchronize()
        if not chunks:
            raise SetupFailure(f"espeak-ng gave no sound for {text!r}")
        sound = np.concatenate(chunks).astype(np.float64)
        loud = np.flatnonzero(np.abs(sound) > 0.01 * np.abs(sound).max())
        sound = sound[loud[0]:loud[-1] + 1]
        sound = resample_poly(sound, out_rate, self.rate)
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
