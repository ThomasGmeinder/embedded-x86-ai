"""Listening without an end-of-speech event: RollingAudio, CommandFilter and a window's way to Gemma.

No server and no microphone. The tests are those of ~/GitHub/voice_intent (tests/test_continuous.py and
tests/test_units.py at its commit 4161915), for the code taken over from there on 2026-11-27.
"""

from __future__ import annotations

import io
import threading
import wave

import numpy as np
import pytest

from so101_smolvla.voice.continuous import AudioWindow, CommandFilter, RollingAudio
from so101_smolvla.voice.microphone import foreground, from_first_pause, up_to_last_pause, window_wav

RATE = 16000
FRAME = b"\x01\x00" * (RATE * 20 // 1000)
LOUD = np.round(3000 * np.sin(2 * np.pi * 300 * np.arange(RATE * 20 // 1000) / RATE)).astype("<i2").tobytes()
QUIET = bytes(len(LOUD))


def fed(audio, speech_flags, start=0.0):
    for index, is_speech in enumerate(speech_flags):
        audio.feed(bytes([index % 256]) * len(FRAME), is_speech, start + index * 0.02)


class Asking:
    """One call of ``next_window`` on a thread of its own; ``done`` is set when it has returned."""

    def __init__(self, audio: RollingAudio) -> None:
        self.got, self.done = [], threading.Event()
        threading.Thread(target=self._ask, args=(audio,), daemon=True).start()

    def _ask(self, audio) -> None:
        self.got.append(audio.next_window())
        self.done.set()


class Speaker:
    """Feeds a tone as speech and silence as a pause, one 20 ms frame after the other."""

    def __init__(self, audio: RollingAudio) -> None:
        self.audio, self.frames = audio, 0

    def say(self, pcm: bytes, frames: int, is_speech: bool) -> None:
        for _ in range(frames):
            self.audio.feed(pcm, is_speech, self.frames * 0.02)
            self.frames += 1


# ---- RollingAudio: the newest sound, and what of it no window held --------------------------------

def test_the_newest_window_is_handed_out_once_speech_has_arrived():
    audio = RollingAudio(RATE, window_s=0.2, min_voice_ms=40)  # ten frames; two speech frames wake
    fed(audio, [False] * 12 + [True, True])
    window = audio.next_window()  # returns at once: the speech is there
    assert len(window.pcm) == 10 * len(FRAME)  # the last ten frames, not more
    assert window.pcm[-len(FRAME):] == bytes([13]) * len(FRAME)  # ending with the newest one
    assert abs(window.end - 13 * 0.02) < 1e-9 and abs(window.start - (4 * 0.02 - 0.02)) < 1e-9
    assert window.sample_rate == RATE and not window.cut


def test_the_wait_ends_on_new_speech_only_and_not_on_silence_or_a_click():
    audio = RollingAudio(RATE, window_s=0.2, min_voice_ms=40)
    fed(audio, [True, True])
    assert audio.next_window() is not None
    asking = Asking(audio)
    fed(audio, [False] * 20 + [True], start=1.0)  # the old speech, silence and one speech frame (a click)
    assert not asking.done.is_set()
    fed(audio, [True], start=2.0)  # the second speech frame: the wait ends on this feed
    assert asking.done.wait(5) and abs(asking.got[0].end - 2.0) < 1e-9


def test_close_ends_the_wait_and_a_break_forgets_the_sound_and_what_was_handed_out_before_it():
    audio = RollingAudio(RATE, window_s=0.2, min_voice_ms=40)
    fed(audio, [True, True])
    before = audio.next_window()
    assert audio.current(before)
    fed(audio, [True, True], start=0.04)
    audio.reset(0.1)  # a break in the microphone's frames at 0.1 s
    assert not audio.current(before)  # an answer about that window is not acted on
    asking = Asking(audio)
    assert not asking.done.is_set()  # the speech before the break wakes nothing
    fed(audio, [True, True], start=0.2)
    assert asking.done.wait(5) and audio.current(asking.got[0])  # sound from after it is current
    assert len(asking.got[0].pcm) == 2 * len(FRAME)  # and holds nothing from before
    asking = Asking(audio)
    audio.close()
    assert asking.done.wait(5) and asking.got == [None]


def test_a_sentence_held_back_by_the_cut_is_handed_out_whole_after_its_pause_without_more_speech():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=40, end_at_pause=lambda pcm: up_to_last_pause(pcm, RATE))
    speaker = Speaker(audio)
    # A sentence (0.6 s), a pause (0.3 s), a second sentence (0.6 s), and 0.1 s after its last word
    # Gemma is free: the window is cut at the pause, the second sentence is not in it.
    speaker.say(LOUD, 30, True), speaker.say(QUIET, 15, False), speaker.say(LOUD, 30, True), speaker.say(QUIET, 5, False)
    first = audio.next_window()
    assert len(first.pcm) == 45 * len(LOUD) and abs(first.end - 44 * 0.02) < 1e-9  # the first sentence and its pause
    assert first.cut  # the speaker went on after it
    asking = Asking(audio)
    speaker.say(QUIET, 4, False)  # 0.18 s after the last word: not yet a pause
    assert not asking.done.is_set()
    speaker.say(QUIET, 1, False)  # 0.2 s: the pause after the held-back sentence, and nobody has said more
    assert asking.done.wait(5)
    assert len(asking.got[0].pcm) == 85 * len(LOUD) and not asking.got[0].cut  # both sentences, whole
    assert asking.got[0].pcm[45 * len(LOUD):75 * len(LOUD)] == LOUD * 30
    # Nothing is held back now: silence alone wakes nothing.
    asking = Asking(audio)
    speaker.say(QUIET, 30, False)
    assert not asking.done.is_set()
    audio.close()
    assert asking.done.wait(5)


def test_a_window_that_holds_nothing_new_before_its_cut_is_not_handed_out_again():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=40, end_at_pause=lambda pcm: up_to_last_pause(pcm, RATE))
    speaker = Speaker(audio)
    speaker.say(LOUD, 30, True), speaker.say(QUIET, 15, False), speaker.say(LOUD, 10, True)  # a sentence, a pause, the next one begins
    first = audio.next_window()
    assert len(first.pcm) == 45 * len(LOUD)  # cut at the pause
    asking = Asking(audio)
    speaker.say(LOUD, 40, True)  # the speaker goes on, without a pause: a window now would end at the same cut
    assert not asking.done.is_set()
    speaker.say(QUIET, 15, False), speaker.say(LOUD, 5, True)  # a new pause, and speech after it: now there is something new
    assert asking.done.wait(5)
    assert len(asking.got[0].pcm) == 110 * len(LOUD) and asking.got[0].end > first.end  # up to the new pause
    audio.close()


def test_the_first_syllable_of_a_sentence_wakes_nothing_its_pause_or_enough_speech_does():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=100, pause_ms=200, enough_speech_ms=1000)
    asking = Asking(audio)
    fed(audio, [False] * 20 + [True] * 30)  # 0.6 s of a sentence: Gemma is idle, and still nothing is sent
    assert not asking.done.is_set()
    fed(audio, [False] * 9, start=50 * 0.02)  # 0.18 s after its last word: not yet a pause
    assert not asking.done.is_set()
    fed(audio, [False], start=59 * 0.02)  # 0.2 s: the sentence and its pause are handed out
    assert asking.done.wait(5) and abs(asking.got[0].end - 59 * 0.02) < 1e-9
    assert len(asking.got[0].pcm) == 60 * len(FRAME)  # everything so far, the whole sentence in it

    # After a silence, someone who talks on without a pause: enough talk ends the wait too.
    asking = Asking(audio)
    fed(audio, [False] * 50 + [True] * 49, start=2.0)  # a second of silence, then 0.98 s of speech
    assert not asking.done.is_set()
    fed(audio, [True], start=2.0 + 99 * 0.02)  # 1 s of talk
    assert asking.done.wait(5) and abs(asking.got[0].end - (2.0 + 99 * 0.02)) < 1e-9
    audio.close()


def test_a_sentence_soon_after_another_is_not_held_back():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=100, pause_ms=200, enough_speech_ms=1000)
    fed(audio, [True] * 30 + [False] * 10)  # a sentence and its pause, in a quiet room
    assert audio.next_window() is not None
    asking = Asking(audio)
    fed(audio, [False] * 20 + [True] * 4, start=40 * 0.02)  # 0.6 s later the next one begins: no silence between
    assert not asking.done.is_set()
    fed(audio, [True], start=64 * 0.02)  # 100 ms of it: sent at once, as without the setting
    assert asking.done.wait(5)
    audio.close()


def test_where_the_talk_never_stops_every_bit_of_new_speech_ends_the_wait_as_without_the_setting():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=100, pause_ms=200, enough_speech_ms=1000)
    # Talk that the detector takes for speech only now and then (a quiet second voice): three
    # frames of speech, five without, on and on. No gap is a pause, so the talk goes on.
    talk = ([True] * 3 + [False] * 5) * 10  # 1.6 s, of which 0.6 s count as speech
    fed(audio, talk)
    assert audio.next_window() is not None  # at once: the talk has gone on for more than a second
    asking = Asking(audio)
    fed(audio, [True] * 3 + [False] * 5 + [True], start=1.6)  # 80 ms of new speech: not yet
    assert not asking.done.is_set()
    fed(audio, [True], start=1.6 + 9 * 0.02)  # 100 ms: sent, though far less than a second is new
    assert asking.done.wait(5)
    audio.close()


def test_a_click_followed_by_silence_still_wakes_nothing_with_enough_speech_set():
    audio = RollingAudio(RATE, window_s=3.0, min_voice_ms=100, pause_ms=200, enough_speech_ms=1000)
    asking = Asking(audio)
    fed(audio, [True] * 4 + [False] * 50)  # 80 ms of sound, then a second of silence
    assert not asking.done.is_set()
    audio.close()
    assert asking.done.wait(5) and asking.got == [None]


def test_the_sound_from_an_earlier_start_is_put_in_front_of_a_window_without_changing_what_comes_next():
    audio = RollingAudio(RATE, window_s=0.2, min_voice_ms=40, history_s=0.6)  # windows of 10 frames, 30 kept
    fed(audio, [True] * 20)
    first = audio.next_window()  # frames 10 to 19
    fed(audio, [True] * 10, start=20 * 0.02)
    second = audio.next_window()  # frames 20 to 29
    whole = audio.from_start(first.start, second)
    assert len(whole.pcm) == 20 * len(FRAME) and whole.pcm == first.pcm + second.pcm
    assert abs(whole.start - first.start) < 1e-9 and whole.end == second.end and whole.id == second.id
    assert audio.from_start(second.start, second) is second  # nothing earlier asked for: the window itself
    assert len(audio.from_start(-1.0, second).pcm) == 30 * len(FRAME)  # no more than the history holds


# ---- CommandFilter: which answer is delivered ---------------------------------------------------------

def window(start, end, cut=False):
    return AudioWindow("w", b"", RATE, start, end, cut)


def test_a_command_is_delivered_once_while_windows_overlap_and_again_after():
    calls, learn = CommandFilter(), ("learn_stack", "", "")
    assert calls.judge(learn, window(0, 3)) == "command"
    assert calls.judge(learn, window(1, 4)) == "repeat"  # begins before the delivering window ended
    assert calls.judge(("unknown", "", ""), window(2, 5)) == "none"
    assert calls.judge(learn, window(2.9, 5.9)) == "repeat"  # still overlaps the delivering window
    assert calls.judge(learn, window(3, 6)) == "command"  # no longer holds that sound: said again
    assert calls.judge(("rebuild_stack", "", ""), window(3.5, 6.5)) == "command"  # another command
    # What is left of a sentence in later windows gives other arguments: held back by the command's name.
    assert calls.judge(("pick_place", "red cube", "box"), window(7, 10)) == "command"
    assert calls.judge(("pick_place", "cube", "box"), window(8, 11)) == "repeat"


def test_stop_is_delivered_every_time():
    calls, stop = CommandFilter(), ("stop", "", "")
    assert calls.judge(stop, window(0, 3)) == calls.judge(stop, window(1, 4)) == "command"


def test_a_pick_command_without_a_place_waits_when_the_speaker_went_on_and_the_whole_sentence_takes_its_place():
    calls = CommandFilter(extend=("pick_place",))
    half, whole = ("pick_place", "small red cube", ""), ("pick_place", "small red cube", "box")
    assert calls.judge(half, window(0.0, 1.8, cut=True)) == "waiting" and calls.waiting_since() == 0.0
    # The next window is asked about from the first one's start; its answer is the whole sentence.
    assert calls.released_by(whole) is None
    assert calls.judge(whole, window(0.0, 3.6)) == "command" and calls.waiting_since() is None
    assert calls.judge(whole, window(1.0, 4.0)) == "repeat"  # and it is delivered once


def test_a_waiting_pick_command_is_delivered_after_all_when_the_sentence_went_on_as_something_else():
    calls = CommandFilter(extend=("pick_place",))
    half, cut = ("pick_place", "ball", ""), window(0.0, 1.5, cut=True)
    assert calls.judge(half, cut) == "waiting"
    released = calls.released_by(("unknown", "", ""))  # what followed was other talk
    assert released == (half, cut) and calls.waiting_since() is None
    assert calls.judge(("unknown", "", ""), window(0.0, 3.5)) == "none"
    assert calls.judge(half, window(1.0, 4.0)) == "repeat"  # it counts as delivered from its own window


def test_a_pick_command_without_a_place_followed_by_a_pause_is_delivered_at_once_and_only_one_answer_is_waited_for():
    calls = CommandFilter(extend=("pick_place",))
    half = ("pick_place", "ball", "")
    assert calls.judge(half, window(0, 3)) == "command"  # the window was not cut: nobody went on
    assert calls.judge(half, window(5.0, 6.5, cut=True)) == "waiting"
    assert calls.released_by(half) is None
    assert calls.judge(half, window(5.0, 8.0, cut=True)) == "command"  # the second answer is delivered, cut or not


# ---- a window on its way to Gemma ---------------------------------------------------------------------

TONE = np.sin(2 * np.pi * 300 * np.arange(2 * RATE) / RATE)
SPEECH, TALK = np.round(3000 * TONE).astype("<i2"), np.round(300 * TONE).astype("<i2")  # 20 dB apart
SOFT = np.round(950 * TONE).astype("<i2")  # 10 dB under the loud speech: an unstressed syllable


def _joined(*parts) -> bytes:
    """(samples, seconds) one after the other."""
    return np.concatenate([samples[: int(seconds * RATE)] for samples, seconds in parts]).tobytes()


def test_what_is_much_quieter_than_the_loud_speech_of_a_window_is_silenced():
    sound = _joined((TALK, 1.0), (SPEECH, 1.0), (TALK, 1.0))
    out = np.frombuffer(foreground(sound, RATE, 8), "<i2").reshape(-1, 320)
    kept = np.abs(out).max(axis=1) > 0
    assert kept[50:100].all()  # the command stays, whole
    assert not kept[:44].any() and not kept[106:].any()  # the talk is gone, but for 100 ms beside the command
    assert kept[45:50].all() and kept[100:105].all()
    assert np.array_equal(out[50:100].reshape(-1), SPEECH[:RATE])  # and it is not changed
    # Talk alone is the loudest there is: it stays. A gate of 0 changes nothing.
    alone = TALK[:RATE].tobytes()
    assert foreground(alone, RATE, 8) == alone and foreground(sound, RATE, 0) == sound
    assert foreground(b"", RATE, 8) == b""


def test_a_window_is_sent_up_to_the_loud_speakers_last_pause():
    def cut(*parts) -> float:  # the seconds of the window that are sent
        sound = _joined(*parts)
        sent = up_to_last_pause(sound, RATE)
        assert sound.startswith(sent)
        return len(sent) / 2 / RATE

    # A sentence, a pause of 0.24 s (background talk goes on in it), the beginning of the next
    # sentence: sent up to the end of the pause.
    assert cut((SPEECH, 1.0), (TALK, 0.24), (SPEECH, 0.6)) == pytest.approx(1.24)
    # The last pause counts, not an earlier one.
    assert cut((SPEECH, 0.6), (TALK, 0.24), (SPEECH, 0.6), (TALK, 0.24), (SPEECH, 0.4)) == pytest.approx(1.68)
    # A dip shorter than 200 ms is inside a word, and a quiet syllable is no pause: no cut. No pause: no cut.
    assert cut((SPEECH, 1.0), (TALK, 0.1), (SPEECH, 0.6)) == pytest.approx(1.7)
    assert cut((SPEECH, 1.0), (SOFT, 0.4), (SPEECH, 0.6)) == pytest.approx(2.0)
    assert cut((SPEECH, 1.0),) == pytest.approx(1.0)
    # The window ends in a pause (the speaker has stopped): sent whole.
    assert cut((SPEECH, 1.0), (TALK, 0.5)) == pytest.approx(1.5)
    # Less than half a second would be left: sent whole.
    assert cut((SPEECH, 0.2), (TALK, 0.24), (SPEECH, 1.0)) == pytest.approx(1.44)
    sound = _joined((SPEECH, 1.0), (TALK, 0.25), (SPEECH, 0.5))
    assert up_to_last_pause(sound, RATE, 0) == sound and up_to_last_pause(b"", RATE) == b""


def test_a_window_that_begins_inside_speech_is_sent_from_its_first_pause():
    def sent_from(*parts) -> float:  # the second from which the window is sent
        sound = _joined(*parts)
        sent = from_first_pause(sound, RATE)
        assert sound.endswith(sent)
        return (len(sound) - len(sent)) / 2 / RATE

    # The tail of a sentence (0.18 s), a pause of 0.24 s, the command: sent from the middle of the pause's first 0.2 s.
    assert sent_from((SPEECH, 0.18), (TALK, 0.24), (SPEECH, 1.6)) == pytest.approx(0.18 + 0.1)
    # It begins in a pause, or holds no pause: sent whole.
    assert sent_from((TALK, 0.3), (SPEECH, 1.0), (TALK, 0.3), (SPEECH, 1.0)) == 0
    assert sent_from((SPEECH, 2.0),) == 0
    # Only a pause at its end, or less than half a second after the pause: sent whole.
    assert sent_from((SPEECH, 1.5), (TALK, 0.6)) == 0
    assert sent_from((SPEECH, 1.0), (TALK, 0.24), (SPEECH, 0.3)) == 0
    assert from_first_pause(b"", RATE) == b""


def test_a_window_reaches_gemma_from_its_first_pause_without_the_background_and_at_the_measured_level():
    # The tail of the sentence before, a pause with background talk in it, then a quiet command.
    quiet_command = np.round(1000 * TONE).astype("<i2")
    background = np.round(100 * TONE).astype("<i2")  # 20 dB under the command
    sound = _joined((quiet_command, 0.18), (background, 0.24), (quiet_command, 1.0), (background, 0.5))
    wav = window_wav(AudioWindow("w", sound, RATE, 0.0, 1.92), quiet_gate_db=8, sent_peak=0.5, most_gain=20)
    with wave.open(io.BytesIO(wav)) as handle:
        assert (handle.getframerate(), handle.getnchannels(), handle.getsampwidth()) == (RATE, 1, 2)
        sent = np.frombuffer(handle.readframes(handle.getnframes()), "<i2").astype(float)
    assert len(sent) == len(sound) // 2 - int(0.28 * RATE)  # from the middle of the first pause on
    frames = np.abs(sent[: len(sent) // 320 * 320].reshape(-1, 320)).max(axis=1)
    assert abs(frames.max() - 0.5 * 32767) < 300  # the command at half of full scale (raised 16 times, under the most gain)
    assert (frames[:2] == 0).all() and (frames[-15:] == 0).all()  # the background talk is silence
