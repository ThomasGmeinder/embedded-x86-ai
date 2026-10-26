# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Low-latency utterance capture with voice-activity detection.

Why
---
The original ``WhisperNPU.listen_once`` waited a fixed **1.2 s of silence**
before transcribing, so every command paid 1.2 s before ASR even started.
This module replaces that with a proper VAD front end:

- a **pre-roll ring buffer** (~0.3 s) so the first syllable isn't clipped
  while the detector is still deciding;
- **end-of-speech hangover of ~1.2 s** (config ``audio.vad.hangover_s``)
  so a short pause inside a command is not a new command. The clock is the
  detector's own silence decision. An energy floor here never elapsed on a
  real microphone, so phrases stayed open and were never transcribed;
- backend ``webrtcvad`` when installed (config ``audio.vad.backend: auto``),
  else an adaptive-noise-floor energy detector (no new hard dependency);
- minimum speech length to reject coughs/keyboard clacks, counted without
  the pre-roll the utterance actually started with (the ring restarts empty
  after each utterance).

Latency budget (whisper-base on the NPU, llama on iGPU): hangover 1.2 s +
ASR ~0.2-0.4 s + intent ~0.1 s from end-of-speech to behavior dispatch.
Transcription runs off the microphone thread, so capture keeps going while
Whisper decodes. :class:`FastListener` reports every sound that never
becomes text (``on_drop``): too short, no words, microphone audio lost.

The detection core (:class:`SpeechSegmenter`) is pure block-in/segment-out so
it unit-tests without a microphone:

    python -m vla_pipeline.audio.vad_listener --selftest
    python -m vla_pipeline.audio.vad_listener --mic        # live latency check
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
BLOCK_SAMPLES = 480  # 30 ms - webrtcvad's preferred frame size
_AUDIO_QUEUE_BLOCKS = 64
_DECODE_QUEUE_UTTERANCES = 2


@dataclass
class VadConfig:
    """Tunable VAD parameters, loaded from config/pipeline.yaml's audio and audio.vad sections."""

    backend: str = "auto"  # auto | webrtc | energy
    aggressiveness: int = 2  # webrtcvad mode 0..3
    silence_threshold: float = 0.015  # energy backend floor (RMS)
    hangover_s: float = 0.7  # end-of-speech silence
    pre_roll_s: float = 0.3
    min_speech_s: float = 0.25
    max_utterance_s: float = 12.0

    @classmethod
    def from_cfg(cls, cfg: dict) -> "VadConfig":
        """Build a VadConfig from the pipeline config dict."""
        a = cfg.get("audio", {})
        v = a.get("vad", {}) or {}
        return cls(
            backend=str(v.get("backend", "auto")),
            aggressiveness=int(v.get("aggressiveness", 2)),
            silence_threshold=float(a.get("silence_threshold", 0.015)),
            hangover_s=float(v.get("hangover_s", 0.7)),
            pre_roll_s=float(v.get("pre_roll_s", 0.3)),
            min_speech_s=float(v.get("min_speech_s", 0.25)),
            max_utterance_s=float(a.get("max_utterance_s", 12.0)),
        )


class _EnergyVad:
    """Adaptive energy detector: threshold rides ~3x above the noise floor."""

    def __init__(self, floor: float):
        self._abs_floor = floor
        self._noise = floor

    def is_speech(self, block: np.ndarray) -> bool:
        """Return True if block's RMS energy is above the adaptive noise floor."""
        rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2)))
        thr = max(self._abs_floor, self._noise * 3.0)
        speech = rms >= thr
        if not speech:  # track the noise floor only during silence
            self._noise = 0.95 * self._noise + 0.05 * rms
        return speech


class _WebRtcVad:
    """VAD backed by the optional ``webrtcvad`` package (more accurate than the energy detector)."""

    def __init__(self, aggressiveness: int):
        import webrtcvad  # optional dependency

        self._vad = webrtcvad.Vad(int(aggressiveness))

    def is_speech(self, block: np.ndarray) -> bool:
        """Return True if block is classified as speech by webrtcvad."""
        pcm = np.clip(block * 32767.0, -32768, 32767).astype(np.int16).tobytes()
        return self._vad.is_speech(pcm, SAMPLE_RATE)


def _make_vad(cfg: VadConfig):
    """Construct the configured VAD backend, falling back to the energy detector if webrtcvad is unavailable."""
    if cfg.backend in ("auto", "webrtc"):
        try:
            vad = _WebRtcVad(cfg.aggressiveness)
            logger.info("VAD backend: webrtcvad (mode %d)", cfg.aggressiveness)
            return vad
        except ImportError:
            if cfg.backend == "webrtc":
                raise
    logger.info("VAD backend: adaptive energy (floor %.3f)", cfg.silence_threshold)
    return _EnergyVad(cfg.silence_threshold)


class SpeechSegmenter:
    """Block-in / segment-out VAD state machine (pure, microphone-free).

    Feed fixed-size float32 mono blocks via :meth:`push`; it returns a full
    utterance (numpy array) the moment end-of-speech is confirmed, else None.
    """

    def __init__(self, cfg: VadConfig, block_samples: int = BLOCK_SAMPLES):
        self.cfg = cfg
        self.block_s = block_samples / SAMPLE_RATE
        self._vad = _make_vad(cfg)
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(int(cfg.pre_roll_s / self.block_s), 1)
        )
        self._speech: list[np.ndarray] = []
        self._silence_blocks = 0
        self._lead = 0  # pre-roll blocks the current utterance started with
        self.in_speech = False
        self.dropped_s = 0.0  # speech in a segment the last push() dropped as too short, else 0

    def reset(self) -> None:
        """Clear all segmenter state back to the not-in-speech starting point."""
        self._pre_roll.clear()
        self._speech.clear()
        self._silence_blocks = 0
        self._lead = 0
        self.in_speech = False

    def push(self, block: np.ndarray) -> np.ndarray | None:
        """Feed one audio block; return the completed utterance once end-of-speech is confirmed, else None."""
        c = self.cfg
        self.dropped_s = 0.0
        speechy = self._vad.is_speech(block)

        if not self.in_speech:
            self._pre_roll.append(block.copy())
            if speechy:
                self.in_speech = True
                self._speech = list(self._pre_roll)
                # The ring restarts empty after each utterance, so speech soon
                # after one starts with less pre-roll than its maximum. The
                # block that started it is speech.
                self._lead = len(self._pre_roll) - 1
                self._silence_blocks = 0
            return None

        self._speech.append(block.copy())
        if speechy:
            self._silence_blocks = 0
        else:
            self._silence_blocks += 1

        utt_len_s = len(self._speech) * self.block_s
        ended = (
            self._silence_blocks * self.block_s >= c.hangover_s
            or utt_len_s >= c.max_utterance_s
        )
        if not ended:
            return None

        audio = np.concatenate(self._speech)
        voiced_s = (len(self._speech) - self._silence_blocks - self._lead) * self.block_s
        self.reset()
        if voiced_s < c.min_speech_s:
            self.dropped_s = max(voiced_s, 0.0)  # too short - likely a click/cough
            return None
        return audio


class FastListener:
    """Microphone thread around :class:`SpeechSegmenter`: each utterance → ``on_transcript(text)``.

    ``transcriber`` is any callable ``np.ndarray → str`` - in the pipeline
    it's ``WhisperNPU.transcribe`` (NPU sessions stay warm between
    utterances). :meth:`start` opens the microphone on its own thread and
    keeps it hot until :meth:`stop`, so a spoken "stop" can interrupt a
    running behavior. ``on_level(rms, in_speech)`` gets every block (for a
    level meter), ``on_busy(True/False)`` brackets each decode, and
    ``on_drop(reason)`` hears about every utterance that never becomes text:
    speech too short to count, a transcript with no words, and microphone
    audio lost because the CPU was too busy to read it. ``opened`` is set
    once the microphone was tried; ``error`` then holds the reason if it
    could not be opened.
    """

    OVERFLOW_REPORT_S = 2.0  # at most one "audio lost" report per this many seconds

    def __init__(self, cfg: dict, transcriber, mic_device=None):
        self.vcfg = VadConfig.from_cfg(cfg)
        self.transcriber = transcriber
        self.mic_device = (
            mic_device
            if mic_device is not None
            else cfg.get("audio", {}).get("mic_device")
        )
        self.error: str | None = None
        self.opened = threading.Event()
        self.overflows = 0  # microphone blocks lost since start
        self._thread: threading.Thread | None = None

    def _dropped(self, reason: str) -> None:
        logger.info("utterance dropped: %s", reason)
        if self._on_drop is not None:
            try:
                self._on_drop(reason)
            except Exception:
                logger.exception("on_drop failed")

    def _audio_in(self, indata, frames, time_info, status) -> None:
        """sounddevice callback: queue the block, dropping the oldest when full; count lost blocks."""
        if status.input_overflow:
            self.overflows += 1
        block = indata[:, 0].copy()
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            self.overflows += 1
            try:
                self._blocks.get_nowait()
            except queue.Empty:
                pass
            try:
                self._blocks.put_nowait(block)
            except queue.Full:
                pass

    def start(self, on_transcript, on_level=None, on_busy=None, on_drop=None) -> "FastListener":
        """Open the microphone and listen on a thread until :meth:`stop`."""
        self._on_transcript, self._on_level, self._on_busy, self._on_drop = on_transcript, on_level, on_busy, on_drop
        self.error = None
        self.overflows = 0
        self.opened.clear()
        self._blocks: queue.Queue = queue.Queue(maxsize=_AUDIO_QUEUE_BLOCKS)  # None ends the listener
        self._jobs: queue.Queue = queue.Queue(maxsize=_DECODE_QUEUE_UTTERANCES)  # None ends the decoder
        self._thread = threading.Thread(target=self._run, daemon=True, name="mic")
        self._thread.start()
        return self

    def stop(self) -> None:
        """End the listener; the microphone closes and the last decode finishes."""
        if self._thread is None:
            return
        self._blocks.put(None)
        self._thread.join()
        self._thread = None

    def listen_once(self, timeout_s: float | None = None, on_drop=None) -> str | None:
        """One transcribed utterance, or None when ``timeout_s`` passes first or the microphone fails."""
        heard: queue.Queue[str] = queue.Queue()
        self.start(heard.put, on_drop=on_drop)
        try:
            self.opened.wait()
            if self.error is not None:
                return None
            return heard.get(timeout=timeout_s)
        except queue.Empty:
            return None
        finally:
            self.stop()

    def _decode(self) -> None:
        while (audio := self._jobs.get()) is not None:
            if self._on_busy is not None:
                self._on_busy(True)
            try:
                t0 = time.perf_counter()
                text = (self.transcriber(audio) or "").strip()
                logger.info(
                    "ASR %.1fs audio in %.0f ms: %r",
                    len(audio) / SAMPLE_RATE,
                    (time.perf_counter() - t0) * 1e3,
                    text,
                )
            except Exception as e:
                logger.exception("transcription failed")
                self._dropped(f"transcription failed ({e})")
                continue
            finally:
                if self._on_busy is not None:
                    self._on_busy(False)
            if not text:
                self._dropped("Whisper heard no words")
                continue
            try:
                self._on_transcript(text)
            except Exception:
                logger.exception("transcript callback failed")

    def _queue_job(self, audio: np.ndarray) -> None:
        """An utterance for Whisper; when it is behind, the oldest waiting one goes."""
        try:
            self._jobs.put_nowait(audio)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            self._jobs.put_nowait(audio)

    def _run(self) -> None:
        import sounddevice as sd

        seg = SpeechSegmenter(self.vcfg)
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SAMPLES,
                callback=self._audio_in,
                device=self.mic_device,
            )
            stream.start()
        except Exception as e:
            self.error = str(e)
            logger.error("Microphone unavailable: %s - voice control disabled.", e)
            return
        finally:
            self.opened.set()
        decoder = threading.Thread(target=self._decode, daemon=True, name="asr")
        decoder.start()
        reported, reported_at = 0, 0.0
        try:
            while (block := self._blocks.get()) is not None:
                audio = seg.push(block)
                if self._on_level is not None:
                    self._on_level(float(np.sqrt(np.mean(np.square(block)))), seg.in_speech)
                now = time.monotonic()
                if self.overflows > reported and now - reported_at >= self.OVERFLOW_REPORT_S:
                    self._dropped(f"microphone audio lost {self.overflows - reported} times (CPU busy)")
                    reported, reported_at = self.overflows, now
                if seg.dropped_s:
                    self._dropped(f"too short to hear ({seg.dropped_s:.2f} s of speech)")
                if audio is not None:
                    self._queue_job(audio)
        finally:
            stream.stop()
            stream.close()
            self._jobs.put(None)
            decoder.join()


# =============================================================================
# Standalone component tests
# =============================================================================


def _selftest() -> int:
    """Run synthetic-audio checks of the VAD segmenter and print PASS/FAIL for each."""
    failures = 0
    cfg = VadConfig(
        backend="energy",
        silence_threshold=0.02,
        hangover_s=0.35,
        pre_roll_s=0.3,
        min_speech_s=0.25,
    )

    def blocks(signal: np.ndarray):
        """Yield signal in fixed-size blocks, like a real audio stream would."""
        for i in range(0, len(signal) - BLOCK_SAMPLES + 1, BLOCK_SAMPLES):
            yield signal[i : i + BLOCK_SAMPLES]

    rng = np.random.default_rng(1)
    silence = lambda s: rng.normal(0, 0.002, int(SAMPLE_RATE * s)).astype(np.float32)  # noqa: E731
    speech = lambda s: (
        0.2 * np.sin(2 * np.pi * 220 * np.arange(int(SAMPLE_RATE * s)) / SAMPLE_RATE)  # noqa: E731
    ).astype(np.float32)

    # 1. One utterance is segmented, and end-of-speech fires ~hangover after it.
    seg = SpeechSegmenter(cfg)
    sig = np.concatenate([silence(1.0), speech(0.8), silence(1.0)])
    got, end_block = None, None
    for n, b in enumerate(blocks(sig)):
        out = seg.push(b)
        if out is not None:
            got, end_block = out, n
            break
    ok = got is not None
    print(
        f"  [{'PASS' if ok else 'FAIL'}] utterance segmented: {None if got is None else len(got)/SAMPLE_RATE:.2f}s"
    )
    failures += not ok
    if ok:
        end_t = (end_block + 1) * BLOCK_SAMPLES / SAMPLE_RATE
        latency = end_t - 1.8  # speech ends at t=1.8s
        ok = 0.3 <= latency <= 0.5
        print(
            f"  [{'PASS' if ok else 'FAIL'}] end-of-speech latency {latency*1000:.0f} ms (expect ≈350)"
        )
        failures += not ok
        # Pre-roll: utterance should include lead-in before detection point.
        ok = len(got) / SAMPLE_RATE >= 0.8
        print(
            f"  [{'PASS' if ok else 'FAIL'}] pre-roll preserved (utterance ≥ speech length)"
        )
        failures += not ok

    # 2. A 60 ms click is rejected by min_speech_s.
    seg = SpeechSegmenter(cfg)
    sig = np.concatenate([silence(0.5), speech(0.06), silence(1.0)])
    got = None
    for b in blocks(sig):
        out = seg.push(b)
        if out is not None:
            got = out
    ok = got is None
    print(f"  [{'PASS' if ok else 'FAIL'}] 60 ms click rejected")
    failures += not ok

    # 3. Two utterances in a row → two segments.
    seg = SpeechSegmenter(cfg)
    sig = np.concatenate(
        [silence(0.5), speech(0.5), silence(0.6), speech(0.5), silence(0.6)]
    )
    count = sum(1 for b in blocks(sig) if seg.push(b) is not None)
    ok = count == 2
    print(
        f"  [{'PASS' if ok else 'FAIL'}] back-to-back utterances → {count} segments (expect 2)"
    )
    failures += not ok

    # 4. A detector gap longer than the hangover ends the phrase. Holding the
    # phrase open on raw microphone energy never elapsed in a real room.
    class _Gap:
        def __init__(self):
            self.n = 0

        def is_speech(self, _block):
            self.n += 1
            return 14 <= self.n <= 23 or 41 <= self.n <= 53

    seg = SpeechSegmenter(cfg)
    seg._vad = _Gap()
    sig = np.concatenate([silence(0.4), speech(1.2), silence(1.0)])
    count = sum(1 for b in blocks(sig) if seg.push(b) is not None)
    ok = count == 2
    print(
        f"  [{'PASS' if ok else 'FAIL'}] detector gap longer than hangover → {count} segments (expect 2)"
    )
    failures += not ok

    print(
        f"\nvad_listener selftest: {'PASSED' if failures == 0 else f'{failures} FAILURES'}"
    )
    return failures


def main() -> None:
    """CLI entry point: run the selftest or a live microphone latency check."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="VAD listener - component test")
    parser.add_argument(
        "--selftest", action="store_true", help="synthetic audio, no mic"
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="live mic: prints transcripts + end-to-end latency",
    )
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(_selftest())
    if args.mic:
        from vla_pipeline.audio.whisper_npu import WhisperNPU
        from vla_pipeline.utils.config import load_config

        cfg = load_config(args.config)
        model = WhisperNPU(cfg)
        listener = FastListener(cfg, model.transcribe)
        listener.start(lambda text: print(f"→ {text!r}"), on_drop=lambda reason: print(f"(not heard: {reason})"))
        listener.opened.wait()
        if listener.error is not None:
            raise SystemExit(f"no microphone: {listener.error}")
        print("Speak - Ctrl-C to quit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
        finally:
            listener.stop()
        return
    parser.print_help()


if __name__ == "__main__":
    main()
