"""The live session: the microphone, Silero speech detection, rolling windows, Gemma.

Made for a floor where someone is always talking: nothing waits for the
speaker to stop. Frames of 20 ms arrive from the microphone. A constant offset
is taken out of them, and Silero with a level threshold decides whether a
frame is speech; that only decides whether there is something to send. Gemma
answers one request at a time, and the moment it has finished one it gets the
newest ``window_s`` seconds of sound, as long as speech has arrived that no
earlier window held (``continuous.RollingAudio``). In a quiet room nothing is
sent; while people talk the requests follow each other without a gap. One
exception keeps a request from being wasted: when someone begins to speak
after a second of silence, the first window waits for the speaker's next short
pause, or for ``enough_speech_ms`` of talk.

A window is sent only up to the loud speaker's last short pause, so that it
does not end inside a sentence; from its first pause on when it begins inside
speech; with everything silenced that is much quieter than its loud speech
(``quiet_gate_db``: talk in the background); and raised to a fixed level.

Overlapping windows hold the same sentence several times.
``continuous.CommandFilter`` decides what is delivered: a command once per
spoken sentence, ``stop`` every time. A pick command without a place, from a
window after which the speaker went on ("pick up the small red cube", pause,
"and put it in the box"), waits: the next window is asked about with the sound
from that window's start, and its answer is delivered.

A delivered command is reported twice: its call the moment Gemma has written
it, the whole answer with the words heard when that is complete. An answer
that is not delivered is not read to its end. The parameters are the
``listening`` section of ``configs/voice.toml``. Nothing here sleeps: the
microphone read blocks on sound, the request thread blocks on the buffer's
condition.

This is the listening of ``~/GitHub/voice_intent`` (``gemma_robot_voice.py``
and ``continuous.py`` at its commit ``4161915``), taken over on 2026-11-27 on
the user's word. Until then the session sent one utterance per sentence, when
the speaker had been silent for 0.4 s.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from array import array
from typing import Callable

import numpy as np
from scipy.signal import lfilter, lfilter_zi

from so101_smolvla.voice.client import GemmaClient
from so101_smolvla.voice.continuous import FRAME_MS, AudioWindow, Call, CommandFilter, RollingAudio
from so101_smolvla.voice.intents import Intent, ParsedCommand
from so101_smolvla.voice.settings import Settings, load_settings, silero_model
from so101_smolvla.voice.speech import wav_bytes

logger = logging.getLogger(__name__)

PAUSE_DB = 16.0  # a pause is this much quieter than the loud speech: an unstressed syllable is not
PAUSE_MS = 200  # and lasts at least this long


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


# ---- a window on its way to Gemma ----------------------------------------------------------------

def _loud_frames(pcm: bytes, sample_rate: int, gate_db: float) -> tuple[np.ndarray, np.ndarray]:
    """(the samples as whole 20 ms frames, which of them are within gate_db of the loud speech)."""
    frame = sample_rate * FRAME_MS // 1000
    samples = np.frombuffer(pcm, dtype="<i2")
    whole = len(samples) // frame
    frames = samples[: whole * frame].astype(np.float64).reshape(whole, frame)
    level = 20 * np.log10(np.maximum(np.sqrt((frames ** 2).mean(axis=1)) / 32768, 1e-6))
    return frames, level >= np.percentile(level, 95) - gate_db


def up_to_last_pause(
    pcm: bytes, sample_rate: int, gate_db: float = PAUSE_DB, min_pause_ms: int = PAUSE_MS, least_ms: int = 500
) -> bytes:
    """The window up to the loud speaker's last pause, so that it does not end inside a sentence.

    A window is cut from the microphone at the moment Gemma is free, wherever the speaker is:
    "put the red cube" is then a command of its own, with the place missing (measured in
    voice_intent). A pause is ``min_pause_ms`` in a row that are quieter than the window's loud
    speech by more than ``gate_db`` (16 dB: at 8 dB the quiet syllables inside a sentence
    counted as pauses, and a window cut there gave "red apple" for "red cube"); what follows
    the last one is left for the next window, which holds it whole. This looks back at pauses
    that have happened; it waits for none. A window that ends in a pause, or holds none, or
    would keep less than ``least_ms``, is sent as it is.
    """
    frame = sample_rate * FRAME_MS // 1000
    if gate_db <= 0 or len(pcm) < 2 * frame:
        return pcm
    _, loud = _loud_frames(pcm, sample_rate, gate_db)
    need, quiet_run = max(1, round(min_pause_ms / FRAME_MS)), 0
    if not loud[-need:].any():
        return pcm  # it ends in a pause
    for index in range(len(loud) - 1, -1, -1):
        quiet_run = 0 if loud[index] else quiet_run + 1
        if quiet_run == need:
            keep = index + need  # up to the end of that pause
            return pcm[: keep * frame * 2] if keep * FRAME_MS >= least_ms else pcm
    return pcm


def from_first_pause(
    pcm: bytes, sample_rate: int, gate_db: float = PAUSE_DB, min_pause_ms: int = PAUSE_MS, least_ms: int = 500
) -> bytes:
    """The window from the loud speaker's first pause on, when it begins inside speech.

    A rolling window begins wherever the sound of three seconds ago was: often in the last
    word of the sentence before the command. Gemma follows what a window begins with (measured
    in voice_intent: the same command, whole in both windows, was answered with 0.18 s of the
    previous word before it and not with 0.16 s). A window that begins inside loud speech is
    sent from the end of the first pause in it; one that begins in a pause, holds none, or
    would keep less than ``least_ms``, is sent as it is.
    """
    frame = sample_rate * FRAME_MS // 1000
    if gate_db <= 0 or len(pcm) < 2 * frame:
        return pcm
    _, loud = _loud_frames(pcm, sample_rate, gate_db)
    need, quiet_run = max(1, round(min_pause_ms / FRAME_MS)), 0
    if not loud[:need].any():
        return pcm  # it begins in a pause
    for index, is_loud in enumerate(loud):
        quiet_run = 0 if is_loud else quiet_run + 1
        if quiet_run == need:
            rest = len(loud) - (index + 1)
            # Half of the pause stays in front: a sentence that starts at a clip's first sample is heard badly.
            start = index + 1 - need // 2
            return pcm[start * frame * 2:] if rest * FRAME_MS >= least_ms and loud[index + 1:].any() else pcm
    return pcm


def foreground(pcm: bytes, sample_rate: int, gate_db: float, hang_ms: int = 100) -> bytes:
    """Silence what is much quieter than the window's loud speech: talk in the background.

    Gemma follows the speech a clip begins with, also when a command spoken later in it is
    much louder (measured in voice_intent: a second voice 20 dB under the command, 5 of 10
    commands right; with this gate at 8 dB, 8 of 10). A 20 ms frame stays when its level is
    within ``gate_db`` of the level one frame in twenty exceeds, or when such a frame lies
    within ``hang_ms`` of it (the quiet sounds inside a word); every other frame becomes
    silence. A window that holds background talk alone keeps it: it is then the loudest speech
    there is, and Gemma answers "none" to it.
    """
    frame = sample_rate * FRAME_MS // 1000
    if gate_db <= 0 or len(pcm) < 2 * frame:
        return pcm
    frames, loud = _loud_frames(pcm, sample_rate, gate_db)
    hang = max(0, round(hang_ms / FRAME_MS))
    keep = np.convolve(loud, np.ones(2 * hang + 1), "same") > 0
    return (frames * keep[:, None]).astype("<i2").tobytes() + pcm[len(frames) * frame * 2:]


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


def window_wav(window: AudioWindow, quiet_gate_db: float, sent_peak: float, most_gain: float) -> bytes:
    """The window as Gemma gets it, the bytes of a WAV file: from its first pause on when it
    begins inside speech, background talk silenced, raised to the level Gemma was measured
    with. (The cut at the speaker's last pause is made where the window is taken.)"""
    pcm = from_first_pause(window.pcm, window.sample_rate)
    pcm = at_level(foreground(pcm, window.sample_rate, quiet_gate_db), sent_peak, most_gain)
    return wav_bytes(pcm, window.sample_rate)


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


def _parsed(call: Call) -> ParsedCommand:
    """A call of the filter as the answer it came from, without words heard."""
    command, thing, place = call
    params = {key: value for key, value in (("object", thing), ("place", place)) if value}
    return ParsedCommand(Intent.from_string(command), params)


class Listener:
    """The microphone thread and the request thread that feed Gemma through ``client``."""

    def __init__(self, client: GemmaClient, *, device=None, settings: Settings | None = None) -> None:
        self.client = client
        self.settings = settings or load_settings()
        listening = self.settings.listening
        self.device = device if device is not None else listening.get("mic_device")
        self.opened = threading.Event()  # set once the microphone streams, or once ``error`` is set
        self.error: str | None = None
        self.overflows = 0  # reads at which the microphone's buffer had overflowed
        self.transcribing = False
        self._stop = threading.Event()
        rate, window_s = int(listening["sample_rate"]), float(listening["window_s"])
        # The newest sound, and what of it no window held yet. It cuts each window at the
        # speaker's last pause and knows what it held back. Twice the window is kept: a sentence
        # that goes on after a window's cut is asked about whole.
        self.audio = RollingAudio(
            rate, window_s, int(listening["min_voice_ms"]), history_s=2 * window_s,
            end_at_pause=lambda pcm: up_to_last_pause(pcm, rate),
            pause_ms=PAUSE_MS, enough_speech_ms=int(listening["enough_speech_ms"]),
        )
        # Used by the request thread only: it judges every answer, in order. A pick command
        # without a place waits when the speaker went on after the window's cut.
        self.calls = CommandFilter(always=(Intent.STOP.value,), none=Intent.UNKNOWN.value,
                                   extend=(Intent.PICK_PLACE.value,))
        self._threads: list[threading.Thread] = []

    def start(
        self, on_command: Callable[[ParsedCommand], None] | None = None, on_level=None, on_busy=None,
        on_call: Callable[[ParsedCommand, float], None] | None = None,
    ) -> "Listener":
        """Begin listening. A command that is delivered is reported twice, each once: ``on_call``
        receives the call (command, object, place) the moment Gemma has written it, with the seconds
        since the request began and without the words heard, which are still being written;
        ``on_command`` receives the whole answer, words included, when it is complete. (A pick
        command that waited for the rest of its sentence in vain is reported by ``on_call`` alone.)
        ``on_level`` gets the frame level in 0..1 and whether the frame counted as speech;
        ``on_busy`` whether a request is under way."""
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
        self.audio.close()  # ends the request thread's wait for speech
        for thread in self._threads:
            thread.join(timeout=5.0)

    def _request(self, on_command, on_busy, on_call) -> None:
        listening = self.settings.listening
        max_age = float(listening["max_age_s"])
        quiet_gate_db = float(listening["quiet_gate_db"])
        sent_peak, most_gain = float(listening["sent_peak"]), float(listening["most_gain"])
        audio, calls = self.audio, self.calls
        while True:
            window = audio.next_window()  # blocks until there is speech no window held, or the stop
            if window is None or self._stop.is_set():
                return
            since = calls.waiting_since()
            if since is not None:  # a command waits for the rest of its sentence: ask about it whole
                window = audio.from_start(since, window)
            if on_busy is not None:
                on_busy(True)
            self.transcribing = True
            delivered = []  # the call of this window, when it is delivered

            def call_written(call: ParsedCommand, seconds: float, window: AudioWindow = window) -> bool:
                # Runs inside the request, while Gemma still writes the words it heard. False:
                # the words are not read, and Gemma is free for the next window sooner.
                age = time.monotonic() - window.end
                if not audio.current(window) or age > max_age:
                    logger.warning("an answer for sound %.1f s old, or from before a break, is not acted on", age)
                    return False
                released = calls.released_by(call.call)
                if released is not None and on_call is not None:
                    # The sentence did not go on as a pick command: the first half is delivered after all.
                    on_call(_parsed(released[0]), seconds)
                verdict = calls.judge(call.call, window)
                logger.info(
                    "window of %.2f s%s: %s -> %s, %.2f s after the request began, %.2f s after its last sample",
                    window.end - window.start, " (cut)" if window.cut else "", call.call, verdict, seconds, age,
                )
                if verdict != "command":
                    return False
                delivered.append(call)
                if on_call is not None:
                    on_call(call, seconds)
                return True

            try:
                command = self.client.parse_audio(
                    window_wav(window, quiet_gate_db, sent_peak, most_gain), on_call=call_written,
                )
            except Exception:
                if not self._stop.is_set():
                    logger.exception("Gemma audio request failed")
                continue
            finally:
                self.transcribing = False
                if on_busy is not None:
                    on_busy(False)
            if delivered and on_command is not None:
                on_command(command)

    def _listen(self, on_level) -> None:
        listening = self.settings.listening
        rate = int(listening["sample_rate"])
        samples = rate * FRAME_MS // 1000
        threshold = float(listening["min_dbfs"])
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
                    # Sound was lost between two reads: what was collected does not continue, and
                    # an answer about it is not acted on.
                    self.overflows += 1
                    logger.warning("the microphone's buffer overflowed (%d so far); the sound so far is dropped", self.overflows)
                    self.audio.reset(now)
                    vad.reset()
                    offset.reset()
                    continue
                # The offset comes out first: the level, the detector and Gemma get the sound itself.
                pcm = offset.feed(bytes(buffer))
                level = dbfs(pcm)
                qualified = vad.feed(pcm, rate) >= vad.threshold and level >= threshold
                if on_level is not None:
                    on_level(max(0.0, min(1.0, (level + 60) / 60)), qualified)
                self.audio.feed(pcm, qualified, now)  # wakes the request thread when it waits for speech
