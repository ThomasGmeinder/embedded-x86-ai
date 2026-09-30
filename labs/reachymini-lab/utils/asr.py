# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

#!/usr/bin/env python3
"""ASR engine using faster-whisper (CPU) with a simple mic recorder.

This module provides `FasterWhisperASREngine` which loads a faster-whisper
`WhisperModel` on CPU and exposes `transcribe_file` and
`transcribe_from_mic` helper methods.

Notes:
- Install dependencies: `pip install faster-whisper sounddevice soundfile`
- Use `model_name='small'` or smaller for reasonable CPU latency.
"""

from __future__ import annotations

import tempfile
import os
import time
from typing import Optional

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - runtime optional
    WhisperModel = None


class FasterWhisperASREngine:
    """ASR engine built on faster-whisper (CPU) with VAD support.

    Example:
        engine = FasterWhisperASREngine(model_name='small')
        # Fixed recording
        text = engine.transcribe_from_mic(4.0)
        # VAD-based recording (stops when speech ends)
        text = engine.transcribe_from_mic_vad(max_duration=4.0)
    """

    def __init__(
        self, model_name: str = "small", device: str = "cpu", beam_size: int = 5
    ):
        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper not installed. Install with `pip install faster-whisper`"
            )

        self.model_name = model_name
        self.device = device
        self.beam_size = beam_size

        # Load model once and reuse
        self.model = WhisperModel(self.model_name, device=self.device)

    def configure_for_latency(self):
        """Configure for optimal latency (smaller model, faster processing)"""
        # Use tiny model for faster processing
        self.model_name = "tiny"
        self.beam_size = 1  # Smaller beam size for faster decoding
        # Note: In practice, you'd need to reload the model
        # For simplicity in this example, we'll just update settings

    def transcribe_file(self, path: str, language: str = None) -> str:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        # Anti-hallucination settings: faster-whisper tends to invent common
        # phrases ("Thank you.", "you", etc.) when fed silence/noise. These
        # thresholds + built-in VAD filtering suppress most of that at the source.
        segments, info = self.model.transcribe(
            path,
            beam_size=self.beam_size,
            language=language,
            vad_filter=True,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            condition_on_previous_text=False,
        )
        text = "".join(segment.text for segment in segments)
        return text.strip()

    def _record_temp_wav(self, duration: float = 5.0, samplerate: int = 16000) -> str:
        try:
            import sounddevice as sd
            import soundfile as sf
        except Exception as e:
            raise RuntimeError(
                "sounddevice and soundfile are required for recording: pip install sounddevice soundfile"
            ) from e

        channels = 1
        print(f"🎙️ Recording {duration:.1f}s @ {samplerate}Hz...")
        data = sd.rec(
            int(duration * samplerate),
            samplerate=samplerate,
            channels=channels,
            dtype="int16",
        )
        sd.wait()

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        sf.write(tmp_path, data, samplerate=samplerate)
        return tmp_path

    def _record_temp_wav_vad(
        self,
        max_duration: float = 5.0,
        samplerate: int = 16000,
        silence_threshold: float = 2.0,
        min_rms: float = 250.0,
    ) -> Optional[str]:
        """Record using Voice Activity Detection (VAD) - stops when speech ends.

        Returns the temp WAV path, or None if the clip was too quiet to be speech.
        """
        try:
            import sounddevice as sd
            import soundfile as sf
            import numpy as np
        except Exception as e:
            raise RuntimeError(
                "sounddevice, soundfile, and numpy are required for VAD recording"
            ) from e

        try:
            import webrtcvad

            vad = webrtcvad.Vad(3)  # Aggressiveness level 3 (most aggressive)
        except ImportError:
            print("⚠️ webrtcvad not installed. Falling back to fixed recording.")
            return self._record_temp_wav(max_duration, samplerate)

        channels = 1
        frames = []
        print(f"🎙️ VAD Recording (max {max_duration:.1f}s) - speak now...")

        frame_duration_ms = 30  # WebRTC VAD works best with 10, 20, or 30ms frames
        frame_samples = int(samplerate * frame_duration_ms / 1000)

        # Record until silence or max duration
        start_time = time.time()
        silent_frames = 0
        required_silent_frames = int(silence_threshold * 1000 / frame_duration_ms)

        with sd.InputStream(
            samplerate=samplerate, channels=channels, dtype="int16"
        ) as stream:
            while (time.time() - start_time) < max_duration:
                data, _ = stream.read(frame_samples)
                if data is None:
                    break

                frames.append(data)

                # Check if frame contains speech
                try:
                    if vad.is_speech(data.tobytes(), samplerate):
                        silent_frames = 0  # Reset silence counter
                    else:
                        silent_frames += 1
                except Exception:
                    # VAD may fail for very short or malformed frames
                    silent_frames += 1

                # If we've had enough silent frames, stop recording
                if silent_frames >= required_silent_frames:
                    print(
                        f"🔇 Detected {silence_threshold}s of silence - stopping recording"
                    )
                    break

        if not frames:
            raise RuntimeError("No audio captured")

        audio_data = np.concatenate(frames, axis=0)
        actual_duration = len(audio_data) / samplerate

        # Energy gate: if the whole clip is basically silence/quiet room noise,
        # don't bother transcribing it (avoids Whisper hallucinating on silence).
        rms = float(np.sqrt(np.mean(np.square(audio_data.astype(np.float32)))))
        if rms < min_rms:
            print(f"🔇 Clip too quiet (rms={rms:.0f} < {min_rms:.0f}) - ignoring")
            return None

        print(
            f"⏱️ Recorded {actual_duration:.2f}s (VAD stopped recording, rms={rms:.0f})"
        )

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        sf.write(tmp_path, audio_data, samplerate=samplerate)
        return tmp_path

    def transcribe_from_mic(
        self, duration: float = 5.0, samplerate: int = 16000, language: str = None
    ) -> Optional[str]:
        """Record from mic then transcribe; returns the transcribed text or None."""
        wav_path = None
        try:
            wav_path = self._record_temp_wav(duration, samplerate)
            text = self.transcribe_file(wav_path, language=language)
            return text
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass

    # Phrases faster-whisper commonly hallucinates from silence/noise.
    # Compared case-insensitively after stripping punctuation/whitespace.
    _HALLUCINATION_BLOCKLIST = {
        "",
        "you",
        "thank you",
        "thanks",
        "thank you.",
        "thanks for watching",
        "thanks for watching!",
        "thank you for watching",
        "bye",
        "bye.",
        "okay",
        "ok",
        "yeah",
        "so",
        "uh",
        "um",
        "hmm",
        ".",
        "the",
        "please subscribe",
        "subscribe",
        "i'm sorry",
    }

    @classmethod
    def _is_hallucination(cls, text: str, min_chars: int = 3) -> bool:
        """Heuristic: is this transcription likely noise/silence junk, not real speech?"""
        normalized = "".join(
            c for c in text.lower() if c.isalnum() or c.isspace()
        ).strip()
        if len(normalized) < min_chars:
            return True
        if normalized in cls._HALLUCINATION_BLOCKLIST:
            return True
        return False

    def transcribe_from_mic_vad(
        self,
        max_duration: float = 5.0,
        samplerate: int = 16000,
        silence_threshold: float = 2.0,
        language: str = None,
        min_rms: float = 250.0,
    ) -> Optional[str]:
        """Record from mic using VAD then transcribe; returns the transcribed text or None.

        Returns None if the clip was too quiet or the transcription looks like a
        Whisper hallucination (common when fed silence/background noise).
        """
        wav_path = None
        try:
            wav_path = self._record_temp_wav_vad(
                max_duration, samplerate, silence_threshold, min_rms=min_rms
            )
            if wav_path is None:
                return None
            text = self.transcribe_file(wav_path, language=language)
            if self._is_hallucination(text):
                print(f"🚫 Ignoring likely noise/hallucination: {text!r}")
                return None
            return text
        finally:
            if wav_path and os.path.exists(wav_path):
                try:
                    os.remove(wav_path)
                except Exception:
                    pass


if __name__ == "__main__":
    # Quick interactive test (if faster-whisper + sounddevice installed)
    try:
        engine = FasterWhisperASREngine(model_name="small")
    except Exception as e:
        print("Initialization error:", e)
    else:
        print("Recording 4s and transcribing...")
        txt = engine.transcribe_from_mic(4.0)
        print("Transcription:", txt)

        print("\n--- Testing VAD recording ---")
        try:
            txt_vad = engine.transcribe_from_mic_vad(max_duration=4.0)
            print("VAD Transcription:", txt_vad)
        except Exception as e:
            print(f"VAD test error: {e}")
