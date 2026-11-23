"""The live session: the microphone, Silero speech detection, rolling windows, Gemma.

Frames of 20 ms arrive from the microphone. Each is gated by Silero and a
level threshold. Every ``hop_s`` that holds enough speech, the last ``window_s``
of sound goes to Gemma as one clip, and a command it finds is reported: its
call the moment Gemma has written it, the whole answer with the words heard
when that is complete. The same command is not reported again within
``cooldown_s``. The parameters
are the ``listening`` section of ``configs/voice.toml``. Nothing here sleeps to
discover speech: the microphone read blocks on sound, the request thread
blocks on its queue.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from array import array
from collections import deque
from typing import Callable

import numpy as np

from so101_smolvla.voice.client import GemmaClient
from so101_smolvla.voice.intents import ParsedCommand
from so101_smolvla.voice.settings import Settings, load_settings, silero_model
from so101_smolvla.voice.speech import wav_bytes

logger = logging.getLogger(__name__)

FRAME_MS = 20
_STOP = None  # the sentinel that ends the request thread


def dbfs(pcm: bytes) -> float:
    """Level of one int16 PCM frame, in decibels relative to full scale."""
    values = array("h", pcm)
    rms = math.sqrt(sum(x * x for x in values) / max(1, len(values)))
    return 20 * math.log10(max(rms / 32768, 1e-6))


class HopWindows:
    """Keep the latest window and emit it once per hop of qualified speech."""

    def __init__(self, sample_rate: int, window_s: float, hop_s: float, min_voice_ms: int) -> None:
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self.frames: deque[tuple[bytes, bool]] = deque(maxlen=round(window_s * 1000 / FRAME_MS))
        self.hop = round(hop_s * 1000 / FRAME_MS)
        self.min_voice_frames = max(1, round(min_voice_ms / FRAME_MS))
        self.count = 0

    def clear(self) -> None:
        self.frames.clear()
        self.count = 0

    def feed(self, pcm: bytes, qualified: bool) -> bytes | None:
        """Append one 20 ms frame. Return the window PCM when a hop is due and holds speech."""
        if len(pcm) != self.frame_bytes:
            raise ValueError("Invalid PCM frame length")
        self.frames.append((pcm, qualified))
        self.count += 1
        if self.count % self.hop:
            return None
        recent = list(self.frames)[-self.hop:]
        if sum(1 for _, spoke in recent if spoke) < self.min_voice_frames:
            return None
        return b"".join(frame for frame, _ in self.frames)


class CommandFilter:
    """Drop unknown, stale, and repeated commands from overlapping windows."""

    def __init__(self, cooldown_s: float) -> None:
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


class Silero:
    """Streaming Silero speech detection on the CPU, the model file of the silero-vad package."""

    def __init__(self, threshold: float) -> None:
        import onnxruntime

        options = onnxruntime.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(silero_model()), providers=("CPUExecutionProvider",), sess_options=options,
        )
        self._outputs = tuple(item.name for item in self._session.get_outputs())
        self.threshold = threshold
        self.reset()

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(64, dtype=np.float32)
        self.prob = 0.0

    def feed(self, pcm: bytes, sample_rate: int) -> float:
        """The probability of speech after this frame."""
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
            self._state, self._context = outputs[1], chunk[-64:]
            self.prob = float(outputs[0].reshape(-1)[0])
            self._buffer = self._buffer[512:]
        return self.prob


class Listener:
    """The microphone thread and the request thread that feed Gemma through ``client``."""

    def __init__(self, client: GemmaClient, *, device=None, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or load_settings()
        self.device = device if device is not None else self.settings.listening.get("mic_device")
        self.opened = threading.Event()  # set once the microphone streams, or once ``error`` is set
        self.error: str | None = None
        self.transcribing = False
        self._stop = threading.Event()
        self._pending: queue.Queue = queue.Queue(maxsize=1)
        self._threads: list[threading.Thread] = []

    def start(
        self, on_command: Callable[[ParsedCommand], None] | None = None, on_level=None, on_busy=None,
        on_call: Callable[[ParsedCommand, float], None] | None = None,
    ) -> "Listener":
        """Begin listening. A command Gemma finds is reported twice, each once: ``on_call`` receives
        the call (command, object, place) the moment Gemma has written it, with the seconds since the
        request began and without the words heard, which are still being written; ``on_command``
        receives the whole answer, words included, when it is complete. ``on_level`` gets the frame
        level in 0..1 and whether the frame counted as speech; ``on_busy`` whether a request is under
        way."""
        requester = threading.Thread(
            target=self._request, args=(on_command, on_busy, on_call), daemon=True, name="gemma-audio",
        )
        microphone = threading.Thread(
            target=self._listen, args=(on_level,), daemon=True, name="microphone",
        )
        self._threads = [requester, microphone]
        requester.start()
        microphone.start()
        return self

    def join(self, timeout: float | None = None) -> None:
        """Return when the microphone thread ends, by ``stop`` or by itself."""
        self._threads[1].join(timeout)

    def stop(self) -> None:
        """End both threads and wait for them."""
        self._stop.set()
        self._put(_STOP)
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _put(self, item) -> None:
        """Queue one item, dropping the one still waiting: Gemma gets the latest window."""
        try:
            self._pending.put_nowait(item)
        except queue.Full:
            try:
                self._pending.get_nowait()
            except queue.Empty:
                pass
            try:
                self._pending.put_nowait(item)
            except queue.Full:
                pass

    def _request(self, on_command, on_busy, on_call) -> None:
        listening = self.settings.listening
        rate = int(listening["sample_rate"])
        commands = CommandFilter(float(listening["cooldown_s"]))
        max_age = float(listening["max_age_s"])
        while True:
            item = self._pending.get()  # blocks on the next window or the stop sentinel
            if item is _STOP:
                return
            window, started, ended = item
            if time.monotonic() - ended > max_age:
                continue
            if on_busy is not None:
                on_busy(True)
            self.transcribing = True
            accepted = []  # the call of this window, once the filter has let it through

            def call_written(call: ParsedCommand, seconds: float) -> bool:
                # Runs inside the request, while Gemma still writes the words it heard. An early
                # window often has no command yet; the next hop can still carry the phrase.
                if not commands.accept(call.intent.value, started, ended, time.monotonic()):
                    return False  # no command, or a repeat: its words are not read, the next window is asked sooner
                accepted.append(call)
                if on_call is not None:
                    on_call(call, seconds)
                return True

            try:
                command = self.client.parse_audio(wav_bytes(window, rate), on_call=call_written)
            except Exception:
                logger.exception("Gemma audio request failed")
                continue
            finally:
                self.transcribing = False
                if on_busy is not None:
                    on_busy(False)
            if accepted and on_command is not None:
                on_command(command)

    def _listen(self, on_level) -> None:
        listening = self.settings.listening
        rate = int(listening["sample_rate"])
        samples = rate * FRAME_MS // 1000
        windows = HopWindows(
            rate, float(listening["window_s"]), float(listening["hop_s"]), int(listening["min_voice_ms"])
        )
        threshold = float(listening["min_dbfs"])
        window_s = float(listening["window_s"])
        try:
            import sounddevice

            vad = Silero(float(listening["vad_threshold"]))
            stream = sounddevice.RawInputStream(
                samplerate=rate, blocksize=samples, channels=1, dtype="int16", device=self.device,
            )
        except Exception as exc:
            self.error = str(exc)
            self.opened.set()
            logger.error("The microphone or the speech detection could not be opened: %s", exc)
            return
        with stream:
            self.opened.set()
            while not self._stop.is_set():
                buffer, overflow = stream.read(samples)  # blocks on sound
                now = time.monotonic()
                if overflow:
                    windows.clear()
                    vad.reset()
                    continue
                pcm = bytes(buffer)
                level = dbfs(pcm)
                qualified = vad.feed(pcm, rate) >= vad.threshold and level >= threshold
                if on_level is not None:
                    on_level(max(0.0, min(1.0, (level + 60) / 60)), qualified)
                window = windows.feed(pcm, qualified)
                if window is not None:
                    self._put((window, now - window_s, now))
