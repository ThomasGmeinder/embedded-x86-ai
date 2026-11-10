# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The one speech front end for the pipeline and the digital-twin notebook.

``voice_intent`` in ``config/pipeline.yaml`` chooses it. ``gemma4-e4b-it``
sends rolling microphone windows to Gemma 4 on llama-server. ``whisper_llama``
keeps WebRTC voice-activity detection, Whisper, and Llama 3.2.
"""

from __future__ import annotations

import io
import json
import logging
import queue
import threading
import time
import wave

from vla_pipeline.audio.gemma_listen import FRAME_MS, CommandFilter, HopWindows, dbfs
from vla_pipeline.llm.intents import keyword_parse
from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)

VOICE_INTENTS = ("gemma4-e4b-it", "whisper_llama")


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return output.getvalue()


class VoiceSession:
    """A parser plus the microphone thread that feeds it."""

    def __init__(self, kind: str, parse, llm=None, error: str | None = None):
        self.kind = kind
        self.parse = parse
        self.llm = llm
        self.error = error
        self.opened = threading.Event()
        self.transcribing = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_text, on_level=None, on_busy=None, on_drop=None, stop_check=None) -> "VoiceSession":
        """Begin listening. ``on_text`` receives a transcript or a command JSON string."""
        if self.error:
            self.opened.set()
            return self
        target = self._gemma if self.kind == "gemma4-e4b-it" else self._whisper
        self._thread = threading.Thread(
            target=target, args=(on_text, on_level, on_busy, on_drop, stop_check),
            daemon=True, name="voice",
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        """Ask the microphone thread to exit and wait for it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def close(self) -> None:
        """Stop the microphone and the language server this session started."""
        self.stop()
        if self.llm is not None:
            self.llm.close()

    def _whisper(self, on_text, on_level, on_busy, on_drop, stop_check) -> None:
        from vla_pipeline.audio.vad_listener import FastListener
        from vla_pipeline.audio.whisper_npu import WhisperNPU

        listener = FastListener(self._cfg, WhisperNPU(self._cfg, device=self._device).transcribe)
        self.opened.set()

        def stopped() -> bool:
            return self._stop.is_set() or (stop_check is not None and stop_check())

        listener.run_forever(on_text, stop_check=stopped, on_level=on_level, on_busy=on_busy)

    def _gemma(self, on_text, on_level, on_busy, on_drop, stop_check) -> None:
        import sounddevice as sd

        g = self._gemma_cfg
        rate = int((self._cfg.get("audio") or {}).get("sample_rate", 16000))
        samples = rate * FRAME_MS // 1000
        windows = HopWindows(rate, float(g["window_s"]), float(g["hop_s"]), int(g["min_voice_ms"]))
        commands = CommandFilter(float(g["cooldown_s"]))
        pending: queue.Queue = queue.Queue(maxsize=1)
        try:
            vad = _Silero(str(resolve(g["vad_model"])), float(g["vad_threshold"]))
        except Exception as exc:
            self.error = str(exc)
            self.opened.set()
            logger.error("Silero VAD unavailable: %s", exc)
            return
        threshold = float(g["min_dbfs"])
        max_age = float(g["max_age_s"])

        def worker() -> None:
            while not self._stop.is_set():
                try:
                    window, started, ended = pending.get(timeout=0.1)
                except queue.Empty:
                    continue
                now = time.monotonic()
                if now - ended > max_age:
                    continue
                if on_busy is not None:
                    on_busy(True)
                self.transcribing = True
                try:
                    cmd = self.llm.parse_audio(_wav(window, rate))
                except Exception:
                    logger.exception("Gemma audio request failed")
                    continue
                finally:
                    self.transcribing = False
                    if on_busy is not None:
                        on_busy(False)
                if not commands.accept(cmd.intent.value, started, ended, time.monotonic()):
                    if on_drop is not None and cmd.intent.value == "unknown":
                        on_drop("no command")
                    continue
                on_text(json.dumps({
                    "command": cmd.intent.value,
                    "object": cmd.object_name or "",
                    "heard": cmd.transcript,
                }))

        threading.Thread(target=worker, daemon=True, name="gemma-audio").start()
        try:
            stream = sd.RawInputStream(
                samplerate=rate, blocksize=samples, channels=1, dtype="int16",
                device=(self._cfg.get("audio") or {}).get("mic_device"),
            )
        except Exception as exc:
            self.error = str(exc)
            self.opened.set()
            logger.error("Microphone unavailable: %s", exc)
            return
        self.opened.set()
        with stream:
            while not self._stop.is_set() and not (stop_check is not None and stop_check()):
                buffer, overflow = stream.read(samples)
                now = time.monotonic()
                if overflow:
                    windows.frames.clear()
                    windows.count = 0
                    vad.reset()
                    continue
                pcm = bytes(buffer)
                level = dbfs(pcm)
                prob = vad.feed(pcm, rate)
                qualified = prob >= vad.threshold and level >= threshold
                if on_level is not None:
                    on_level(max(0.0, min(1.0, (level + 60) / 60)), qualified)
                window = windows.feed(pcm, qualified)
                if window is None:
                    continue
                item = (window, now - float(g["window_s"]), now)
                try:
                    pending.put_nowait(item)
                except queue.Full:
                    try:
                        pending.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        pending.put_nowait(item)
                    except queue.Full:
                        pass


class _Silero:
    """Streaming Silero voice-activity detection on the CPU."""

    def __init__(self, model_path: str, threshold: float):
        import os

        import numpy as np
        import onnxruntime

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Silero VAD model not found: {model_path}")
        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            model_path, providers=("CPUExecutionProvider",), sess_options=options,
        )
        self._outputs = tuple(item.name for item in self._session.get_outputs())
        self.threshold = threshold
        self._np = np
        self.reset()

    def reset(self) -> None:
        np = self._np
        self._buffer = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self.prob = 0.0

    def feed(self, pcm: bytes, sample_rate: int) -> float:
        np = self._np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        stride = max(1, sample_rate // 16000)
        if stride > 1:
            samples = samples[::stride]
        self._buffer = np.concatenate((self._buffer, samples))
        while self._buffer.size >= 512:
            chunk = self._buffer[:512]
            audio = np.concatenate((self._context, chunk))[np.newaxis]
            outputs = self._session.run(
                self._outputs, {"input": audio, "state": self._state, "sr": np.array(16000, np.int64)},
            )
            self._state, self._context, self.prob = outputs[1], chunk[-64:], float(outputs[0].reshape(-1)[0])
            self._buffer = self._buffer[512:]
        return self.prob


def open_voice(cfg: dict, *, device=None) -> VoiceSession:
    """Build the speech front end selected by ``cfg['voice_intent']``."""
    kind = cfg.get("voice_intent")
    if kind not in VOICE_INTENTS:
        raise ValueError(
            f"voice_intent must be gemma4-e4b-it or whisper_llama, not {kind!r}"
        )
    if kind == "whisper_llama":
        from vla_pipeline.llm.llama_intent import LlamaIntentParser

        llm = LlamaIntentParser(cfg)
        session = VoiceSession(kind, llm.parse, llm)
        session._cfg = cfg
        session._device = device
        return session
    try:
        from vla_pipeline.llm.llama_intent import LlamaIntentParser

        llm = LlamaIntentParser(cfg, section="gemma4-e4b-it")
    except FileNotFoundError as exc:
        logger.warning("Gemma is not ready (%s). Typed commands use the keyword match.", exc)
        session = VoiceSession(kind, keyword_parse, error=str(exc))
        return session
    session = VoiceSession(kind, llm.parse, llm)
    session._cfg = cfg
    session._gemma_cfg = cfg["gemma4-e4b-it"]
    return session
