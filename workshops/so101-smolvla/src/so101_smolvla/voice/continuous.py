"""Continuous listening without an end-of-speech event.

Gemma answers one request at a time. Whoever asks it calls
``RollingAudio.next_window()`` each time the previous request is finished and
gets the newest sound at once, as long as speech has arrived that no earlier
window held. Nothing waits for the speaker to stop, so it works where someone
is always talking. (Only the first window after a silence can be made to wait
for the speaker's next short pause: ``enough_speech_ms``.) Overlapping windows
hold the same sentence several times, and a window can end in the middle of a
sentence; ``CommandFilter`` decides what of Gemma's answers is delivered as a
command.

Taken over on 2026-11-27 from ``~/GitHub/voice_intent`` (``continuous.py`` at
its commit ``4161915``), on the user's word; the logic and its measurements
are that project's (its README, RESEARCH_LEDGER.md section 3). Left out: the
signal "someone is talking" for a proxy in front of the server, which this
workshop does not have, and the second-answer confirmation that project no
longer uses.
"""

from __future__ import annotations

import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

FRAME_MS = 20
Call = tuple[str, str, str]  # (command, object, place)


@dataclass(frozen=True)
class AudioWindow:
    id: str
    pcm: bytes
    sample_rate: int
    start: float  # monotonic time of its first sample
    end: float  # monotonic time at which its last frame was captured
    cut: bool = False  # it was cut at the speaker's last pause and the speaker went on after it


class RollingAudio:
    """The newest sound of the microphone, and whether it holds speech not yet handed out.

    ``feed`` is called by the capture loop for every 20 ms frame. ``next_window``
    is called by the one thread that asks Gemma: it blocks until at least
    ``min_voice_ms`` of speech has arrived that no handed-out window held (a
    click wakes nothing), and returns the newest ``window_s`` seconds, cut at
    the speaker's last pause when ``end_at_pause`` is given (``AudioWindow.cut``
    then says whether the speaker went on after it). ``from_start`` puts the
    sound from an earlier moment in front of a window, as far as ``history_s``
    holds it. ``close`` ends the wait.
    """

    def __init__(
        self,
        sample_rate: int,
        window_s: float,
        min_voice_ms: int = 100,
        history_s: float | None = None,
        end_at_pause: Callable[[bytes], bytes] | None = None,
        pause_ms: int = 200,
        enough_speech_ms: int | None = None,
        silence_ms: int = 1000,
    ) -> None:
        """``end_at_pause(pcm) -> pcm``: returns the beginning of a window up to the speaker's last
        pause, so that it does not end inside a sentence (``microphone.up_to_last_pause``). The
        speech it holds back is not lost: it counts as not handed out, and the wait of
        ``next_window`` also ends when ``pause_ms`` without speech have followed it, so that the
        sentence is handed out whole even when nobody says anything more.

        ``enough_speech_ms``: when someone begins to speak in a quiet room (after ``silence_ms``
        without any speech), new speech does not end the wait before ``pause_ms`` without speech
        have followed it, or the talk has gone on for this long. Without it a window taken while
        Gemma is idle holds the first syllable of a sentence and nothing else: a request that can
        give no command and makes the sentence wait for its end (measured in voice_intent: 1.2 s).
        It must be longer than a short command (1 to 2 s), or the window is sent in the middle of
        it. Where someone always talks it changes nothing: every bit of new speech ends the wait
        as before. None: the wait ends on ``min_voice_ms`` of new speech at once."""
        self.sample_rate = sample_rate
        self._end_at_pause = end_at_pause
        self._pause_frames = max(1, round(pause_ms / FRAME_MS))
        self._enough_frames = None if enough_speech_ms is None else max(1, round(enough_speech_ms / FRAME_MS))
        self._silence_frames = max(self._pause_frames, round(silence_ms / FRAME_MS))
        self._held_back = False  # the last window was cut and speech followed the cut
        self._quiet = 0  # frames without speech in a row, up to the newest
        self._talk = 0  # frames since someone began to speak after a silence; 0 in a silence
        self._handed_out_until = float("-inf")  # capture time of the last frame of the last window
        self._discontinuity = float("-inf")  # the moment of the last break in the sound (``reset``)
        self._window_frames = round(window_s * 1000 / FRAME_MS)
        self._frames: deque = deque(
            maxlen=max(self._window_frames, round((history_s or window_s) * 1000 / FRAME_MS))
        )  # (pcm, captured_at, is_speech)
        self._min_voice_frames = max(1, round(min_voice_ms / FRAME_MS))
        self._unsent = 0  # speech frames that no handed-out window held
        self._closed = False
        self._changed = threading.Condition()

    def feed(self, pcm: bytes, is_speech: bool, captured_at: float) -> None:
        with self._changed:
            self._frames.append((pcm, captured_at, is_speech))
            if is_speech:
                self._unsent += 1
                self._quiet = 0
            else:
                self._quiet += 1
            # The talk goes on through pauses and ends with a silence.
            self._talk = 0 if self._quiet >= self._silence_frames else (
                self._talk + 1 if is_speech or self._talk else 0
            )
            if self._something_to_hand_out():
                self._changed.notify()

    def _something_to_hand_out(self) -> bool:
        """New speech has arrived (not the mere beginning of talk in a quiet room), or the pause
        after speech that was held back has."""
        paused = self._quiet >= self._pause_frames
        new = self._unsent >= self._min_voice_frames and (
            self._enough_frames is None or paused or self._talk >= self._enough_frames
        )
        return new or (self._held_back and paused)

    def next_window(self) -> AudioWindow | None:
        """The newest window, once it holds speech no earlier one held; None after ``close``."""
        with self._changed:
            while True:
                self._changed.wait_for(lambda: self._closed or self._something_to_hand_out())
                if self._closed:
                    return None
                # Woken by the pause after held-back speech: that speech goes out now, uncut, whatever
                # the cut would make of the window (otherwise the same cut could hold it back for ever).
                after_pause = self._held_back and self._quiet >= self._pause_frames
                self._unsent, self._held_back = 0, False
                frames = list(self._frames)[-self._window_frames:]
                if self._end_at_pause is not None and not after_pause:
                    whole = b"".join(frame[0] for frame in frames)
                    kept = len(self._end_at_pause(whole)) // (len(whole) // len(frames))
                    # Speech after the cut is in no window yet: it is handed out when more speech
                    # or the pause after it has arrived.
                    self._held_back = any(frame[2] for frame in frames[kept:])
                    frames = frames[:kept]
                    if self._held_back and frames[-1][1] <= self._handed_out_until:
                        continue  # nothing new before the cut: asking again would ask the same; wait on
                self._handed_out_until = frames[-1][1]
                return self._window(frames, cut=self._held_back)

    def from_start(self, start: float, window: AudioWindow) -> AudioWindow:
        """``window`` with the sound from ``start`` on in front of it, as far as the history holds
        it: for asking about a sentence whole whose beginning an earlier window held. Changes
        nothing of what is handed out next."""
        with self._changed:
            frames = [frame for frame in self._frames if start < frame[1] <= window.end]
        if not frames or frames[0][1] - FRAME_MS / 1000 >= window.start:
            return window
        return AudioWindow(
            window.id, b"".join(frame[0] for frame in frames), self.sample_rate,
            frames[0][1] - FRAME_MS / 1000, window.end, window.cut,
        )

    def _window(self, frames, cut: bool = False) -> AudioWindow:
        return AudioWindow(
            uuid.uuid4().hex, b"".join(frame[0] for frame in frames), self.sample_rate,
            frames[0][1] - FRAME_MS / 1000, frames[-1][1], cut,
        )

    def reset(self, at: float) -> None:
        """Forget the sound, after a discontinuity at the moment ``at`` such as a device overflow.
        A window handed out before it is no longer current."""
        with self._changed:
            self._frames.clear()
            self._unsent, self._held_back, self._quiet, self._talk = 0, False, 0, 0
            self._discontinuity = at

    def current(self, window: AudioWindow) -> bool:
        """Whether the sound has gone on without a break since the window began: an answer about
        a window from before a break is not acted on."""
        with self._changed:
            return window.start >= self._discontinuity

    def close(self) -> None:
        with self._changed:
            self._closed = True
            self._changed.notify_all()


class CommandFilter:
    """What of Gemma's answers, one per window, is delivered as a command.

    ``judge`` is called with every answer, in the order the windows were asked,
    and returns one of:

    ``'command'``  deliver it.
    ``'repeat'``   the same command was delivered from a window this one overlaps: such a
                   window may still hold the same spoken sentence, or what is left of it
                   (a window that has lost the sentence's first words gives other arguments).
                   Held back until a window begins after the end of the delivering one. No
                   time constant. Consequence: the same command said twice within one window
                   length is delivered once.
    ``'waiting'``  a command of ``extend`` without a place, from a window after which the speaker
                   went on (``window.cut``): "pick up the small red cube" may go on with "and put
                   it in the box". It is not delivered yet. Whoever asks Gemma asks about the
                   next window with the sound from this window's start in front
                   (``waiting_since``). If that answer is again such a command, it is delivered
                   in its place; if it is another answer, ``released_by`` hands the waiting
                   command out to be delivered after all. One answer is waited for, not more.
    ``'none'``     no command (``unknown``).

    Commands in ``always`` (the emergency stop) are delivered every time: stopping twice does
    no harm, and holding a stop back does.
    """

    def __init__(
        self, always: tuple[str, ...] = ("stop",), none: str = "unknown", extend: tuple[str, ...] = ()
    ) -> None:
        self._always, self._none, self._extend = frozenset(always), none, frozenset(extend)
        self._delivered_until: dict[str, float] = {}  # command -> the end of the window that delivered it
        self._waiting: tuple[Call, AudioWindow] | None = None  # a command without a place that may go on

    def waiting_since(self) -> float | None:
        """The start of the window whose command waits for the rest of its sentence; None: none waits."""
        return None if self._waiting is None else self._waiting[1].start

    def released_by(self, call: Call) -> tuple[Call, AudioWindow] | None:
        """To be called with every answer before ``judge``: (call, window) of the waiting command
        when this answer is no command of ``extend``, so that it is delivered now; else None."""
        waiting = self._waiting
        if waiting is None or call[0] in self._extend:
            return None
        self._waiting = None
        self._delivered_until[waiting[0][0]] = waiting[1].end
        return waiting

    def judge(self, call: Call, window: AudioWindow) -> str:
        command = call[0]
        if command == self._none:
            return "none"
        if command in self._always:
            return "command"
        until = self._delivered_until.get(command)
        if until is not None and window.start < until:
            return "repeat"
        if command in self._extend and not call[2] and window.cut and self._waiting is None:
            self._waiting = (call, window)
            return "waiting"
        self._waiting = None
        self._delivered_until[command] = window.end
        return "command"
