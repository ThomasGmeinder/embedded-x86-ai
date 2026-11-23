"""The live session: the microphone, Silero speech detection, utterances, Gemma.

Frames of 20 ms arrive from the microphone. Each is gated by Silero and a
level threshold. An utterance is the sound from the first frame of speech
(with a little before it) to the moment the speaker has been silent for
``end_silence_ms``; the frame that ends it sends it to Gemma as one clip, no
longer than ``window_s``. Gemma takes a whole clip, not sound as it comes, so
the end of the speech is the earliest moment a sentence can be sent. A command
Gemma finds in it is reported: its call the moment Gemma has written it, the
whole answer with the words heard when that is complete. The parameters are
the ``listening`` section of ``configs/voice.toml``. Nothing here sleeps to
discover speech: the microphone read blocks on sound, the end of an utterance
is counted in its own frames, the request thread blocks on its queue.

Until 2026-11-23 the session sent the last ``window_s`` of sound on a fixed
grid, every 2 s that held speech: up to 2 s passed after a sentence before it
was sent, half sentences were sent, and a filter dropped the repeats.
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
from scipy.signal import lfilter, lfilter_zi

from so101_smolvla.voice.client import GemmaClient
from so101_smolvla.voice.intents import Intent, ParsedCommand
from so101_smolvla.voice.settings import Settings, load_settings, silero_model
from so101_smolvla.voice.speech import wav_bytes

logger = logging.getLogger(__name__)

FRAME_MS = 20
WAITING_UTTERANCES = 4  # how many utterances may wait for Gemma; the oldest goes when one more arrives
_STOP = None  # the sentinel that ends the request thread


def dbfs(pcm: bytes) -> float:
    """Level of one int16 PCM frame, in decibels relative to full scale."""
    values = array("h", pcm)
    rms = math.sqrt(sum(x * x for x in values) / max(1, len(values)))
    return 20 * math.log10(max(rms / 32768, 1e-6))


class OffsetRemover:
    """Takes a constant offset out of the microphone's samples, frame after frame.

    This machine's built-in microphone delivers samples around +1028, not around 0 (measured
    2026-11-23: the mean alone reads -30.1 dBFS, the room without it -59.8 dBFS). With the
    offset in, the level of every frame reads about -30 dBFS whatever is said, and it is in
    the sound the detector and Gemma get. A first-order high-pass with its corner near 13 Hz
    at 16 kHz, far below speech; it starts settled on the first sample, so there is no thump.
    """

    def __init__(self, pole: float = 0.995) -> None:
        self._b, self._a = (1.0, -1.0), (1.0, -pole)
        self.reset()

    def reset(self) -> None:
        self._state = None

    def feed(self, pcm: bytes) -> bytes:
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
        if self._state is None:
            self._state = lfilter_zi(self._b, self._a) * samples[0]
        cleaned, self._state = lfilter(self._b, self._a, samples, zi=self._state)
        return np.clip(np.round(cleaned), -32768, 32767).astype("<i2").tobytes()


class Utterances:
    """Cut the microphone's frames into utterances and hand each one over the moment it has ended.

    An utterance begins at the first frame that counts as speech, with ``pre_roll_ms`` of the
    sound before it (the detector answers a little after a word begins). It has ended when
    ``end_silence_ms`` of frames in a row did not count as speech: that frame returns the
    utterance, so nothing waits for a clock. One that grows to ``max_s`` is handed over as it
    is and the next begins. One with less than ``min_voice_ms`` of speech in it (a click, a
    cough) is dropped.
    """

    def __init__(
        self, sample_rate: int, max_s: float, end_silence_ms: int, pre_roll_ms: int, min_voice_ms: int
    ) -> None:
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * 2
        self.max_frames = round(max_s * 1000 / FRAME_MS)
        self.end_silence_frames = max(1, round(end_silence_ms / FRAME_MS))
        self.min_voice_frames = max(1, round(min_voice_ms / FRAME_MS))
        self.before: deque[bytes] = deque(maxlen=max(1, round(pre_roll_ms / FRAME_MS)))
        self.clear()

    def clear(self) -> None:
        self.before.clear()
        self.frames: list[bytes] = []  # the utterance under way; empty between utterances
        self.voiced = 0  # its frames that counted as speech
        self.silent = 0  # its frames in a row, at its end, that did not

    @property
    def speaking(self) -> bool:
        """Whether an utterance is under way."""
        return bool(self.frames)

    def feed(self, pcm: bytes, qualified: bool) -> bytes | None:
        """Take one 20 ms frame. Return the utterance's PCM on the frame that ends it."""
        if len(pcm) != self.frame_bytes:
            raise ValueError("Invalid PCM frame length")
        if not self.frames:
            if not qualified:
                self.before.append(pcm)
                return None
            self.frames = [*self.before, pcm]
            self.before.clear()
            self.voiced, self.silent = 1, 0
            return None
        self.frames.append(pcm)
        if qualified:
            self.voiced += 1
            self.silent = 0
        else:
            self.silent += 1
        if self.silent < self.end_silence_frames and len(self.frames) < self.max_frames:
            return None
        ended, voiced = self.frames, self.voiced
        self.frames, self.voiced, self.silent = [], 0, 0
        # The end of this utterance is the sound before the next one. Gemma hears a sentence
        # badly that begins at a clip's first sample (measured 2026-11-23: two of four test
        # sentences lost their first words without sound before them).
        self.before.extend(ended[-self.before.maxlen:])
        return b"".join(ended) if voiced >= self.min_voice_frames else None


def at_level(pcm: bytes, peak: float, most_gain: float) -> bytes:
    """The sound scaled so that its loud samples reach ``peak`` of full scale, by ``most_gain`` at most.

    Gemma hears quiet sound worse (measured 2026-11-23: at a tenth of the synthetic test
    voice's level a "green" became "blue"), and this machine's microphone is quiet: speech
    frames at -43 dBFS in the median. "Loud" is the level only one sample in a thousand
    exceeds, so a click does not set the scale; what exceeds full scale after it is clipped.
    Silence stays silence: there is a most gain.
    """
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float64)
    loud = float(np.percentile(np.abs(samples), 99.9)) if samples.size else 0.0
    if loud <= 0.0:
        return pcm
    gain = min(most_gain, peak * 32767 / loud)
    return np.clip(np.round(samples * gain), -32768, 32767).astype("<i2").tobytes()


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
        self.overflows = 0  # reads at which the microphone's buffer had overflowed
        self.transcribing = False
        self._stop = threading.Event()
        # Utterances that wait while Gemma answers an earlier one. Each is a sentence of its own,
        # so they are kept; bounded, the oldest goes when a fifth arrives.
        self._pending: queue.Queue = queue.Queue(maxsize=WAITING_UTTERANCES)
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
        """Queue one item without blocking the microphone; when the queue is full the oldest goes."""
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
        max_age = float(listening["max_age_s"])
        while True:
            item = self._pending.get()  # blocks on the next utterance or the stop sentinel
            if item is _STOP:
                return
            utterance, ended = item
            if time.monotonic() - ended > max_age:
                continue
            if on_busy is not None:
                on_busy(True)
            self.transcribing = True
            accepted = []  # the call of this utterance, when it holds a command

            def call_written(call: ParsedCommand, seconds: float) -> bool:
                # Runs inside the request, while Gemma still writes the words it heard.
                if call.intent is Intent.UNKNOWN:
                    return False  # no command: its words are not read, the next utterance is asked sooner
                accepted.append(call)
                if on_call is not None:
                    on_call(call, seconds)
                return True

            try:
                command = self.client.parse_audio(wav_bytes(utterance, rate), on_call=call_written)
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
        utterances = Utterances(
            rate, float(listening["window_s"]), int(listening["end_silence_ms"]),
            int(listening["pre_roll_ms"]), int(listening["min_voice_ms"]),
        )
        threshold = float(listening["min_dbfs"])
        sent_peak, most_gain = float(listening["sent_peak"]), float(listening["most_gain"])
        offset = OffsetRemover()
        try:
            import sounddevice

            vad = Silero(float(listening["vad_threshold"]))
            # latency "low": the sound library's default for input is its "high" figure (34.8 ms
            # against 8.7 ms on this machine's default device, read from the library 2026-11-23).
            stream = sounddevice.RawInputStream(
                samplerate=rate, blocksize=samples, channels=1, dtype="int16", device=self.device,
                latency="low",
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
                    # Sound was lost between two reads: what was collected does not continue.
                    self.overflows += 1
                    logger.warning("the microphone's buffer overflowed (%d so far); the sentence under way is dropped", self.overflows)
                    utterances.clear()
                    vad.reset()
                    offset.reset()
                    continue
                # The offset comes out first: the level, the detector and Gemma get the sound itself.
                pcm = offset.feed(bytes(buffer))
                level = dbfs(pcm)
                qualified = vad.feed(pcm, rate) >= vad.threshold and level >= threshold
                if on_level is not None:
                    on_level(max(0.0, min(1.0, (level + 60) / 60)), qualified)
                # The frame that ends an utterance hands it over: it goes to Gemma now, at the
                # level Gemma was measured with.
                utterance = utterances.feed(pcm, qualified)
                if utterance is not None:
                    self._put((at_level(utterance, sent_peak, most_gain), now))
