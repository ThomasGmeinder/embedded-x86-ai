# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The speech front end against each real voice model: sentences in, commands out, as a user hears it.

Once for every value of ``voice_intent``: Gemma 4 and the decision model
d1-Omni-2B. A stand-in microphone plays sentences in a synthetic voice
(espeak-ng), in real time, into the real ``open_voice`` session: the offset
removal, Silero, the utterance cut, the level raise and the request to a
``llama-server`` this test starts on a free port, on the integrated graphics.
The sound is made as a quiet built-in microphone delivers it: samples around
+1028, speech at a tenth of full scale.

Needs the model's files, the Silero model and the built llama-server
(``scripts/ensure_voice.sh``, once with each ``voice_intent``), and
``libespeak-ng1``. A missing piece fails the test; nothing is skipped.

What is asserted is what must hold: one request per sentence, each with its
command; the command delivered when the model has decided it, before the words
it heard (Gemma writes them; the decision model writes none); a question
delivers nothing. The seconds from the end of a sentence to its command are
printed, not asserted.
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import resample_poly

from vla_pipeline.llm.intents import Intent
from vla_pipeline.llm.voice import VOICE_INTENTS
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_detect_npu import resolve_class

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(180)]

ESPEAK_ONCE = Path(__file__).with_name("espeak_once.py")
RATE = 16000
OFFSET = 1028       # the constant a quiet built-in microphone was measured to add to its samples
SPEECH_PEAK = 0.1   # of full scale: quiet speech
LEAD_S = 1.5        # silence before the first sentence
GAP_S = 2.5         # silence after each sentence: room for the model's answer
# (sentence, the command it must give, its object). The last is a question: no command.
SENTENCES = [
    ("pick up the ball", Intent.PICK_PLACE, "ball"),
    ("fetch the cube", Intent.FETCH_BLOCK, ""),
    ("go home", Intent.HOME, ""),
    ("stop robot", Intent.STOP, ""),
    ("open your hand", Intent.RELEASE, ""),
    ("what is the weather like today", Intent.UNKNOWN, ""),
]


def spoken(text: str) -> np.ndarray:
    """The sentence in espeak-ng's voice at 16 kHz, trimmed, at SPEECH_PEAK. A process per sentence:
    espeak-ng keeps state between sentences, and the same sentence must be the same sound every time."""
    ran = subprocess.run(
        [sys.executable, str(ESPEAK_ONCE), "libespeak-ng.so.1", "en-us", "150", text],
        capture_output=True, timeout=60,
    )
    assert ran.returncode == 0 and len(ran.stdout) > 4, (
        "espeak-ng gave no sound (sudo apt install libespeak-ng1): " + ran.stderr.decode("utf-8", "replace")
    )
    rate = struct.unpack("<I", ran.stdout[:4])[0]
    sound = np.frombuffer(ran.stdout[4:], "<i2").astype(np.float64)
    loud = np.flatnonzero(np.abs(sound) > 0.01 * np.abs(sound).max())
    sound = resample_poly(sound[loud[0]:loud[-1] + 1], RATE, rate)
    return sound * (SPEECH_PEAK * 32767 / np.abs(sound).max())


@pytest.mark.parametrize("voice_intent", VOICE_INTENTS)
def test_spoken_sentences_give_their_commands_one_request_each_the_call_before_the_words(
    voice_intent, monkeypatch, tmp_path,
):
    parts, begins, ends = [np.zeros(int(LEAD_S * RATE))], [], []
    for sentence, _, _ in SENTENCES:
        begins.append(sum(len(part) for part in parts) / RATE)
        parts.append(spoken(sentence))
        ends.append(sum(len(part) for part in parts) / RATE)
        parts.append(np.zeros(int(GAP_S * RATE)))
    sound = np.round(np.concatenate(parts) + OFFSET).astype("<i2").tobytes()
    played_s = len(sound) / 2 / RATE
    answered = threading.Event()  # set when the parser has answered one request per sentence

    class Microphone:
        """Stands in for sounddevice.RawInputStream: the sound above, a 20 ms frame every 20 ms of
        real time, as a device delivers it; then the offset alone. It fails the run when the sound
        has been over for 30 s and the parser still has not answered a request for every sentence."""

        began: float | None = None

        def __init__(self, **settings):
            assert settings["samplerate"] == RATE and settings["latency"] == "low"
            self.position = 0

        def __enter__(self):
            Microphone.began = time.monotonic()
            return self

        def __exit__(self, *_):
            return False

        def read(self, samples):
            self.position += samples
            if self.position / RATE > played_s + 30:
                raise RuntimeError("no answer for every sentence 30 s after the sound ended")
            rest = Microphone.began + self.position / RATE - time.monotonic()
            if rest > 0:
                time.sleep(rest)  # the device's own clock
            chunk = sound[(self.position - samples) * 2:self.position * 2]
            return chunk + np.full(samples - len(chunk) // 2, OFFSET, "<i2").tobytes(), False

    import sounddevice

    from vla_pipeline.llm.voice import open_voice

    monkeypatch.setattr(sounddevice, "RawInputStream", Microphone)
    cfg = load_config()
    cfg["voice_intent"] = voice_intent
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        cfg[voice_intent]["port"] = sock.getsockname()[1]  # a server of this test's own
    cfg.setdefault("system", {})["log_dir"] = str(tmp_path)

    session = open_voice(cfg)
    try:
        assert not session.error, (
            f"{voice_intent} is not ready ({session.error}); set voice_intent: {voice_intent} in "
            f"config/pipeline.yaml and run scripts/ensure_voice.sh"
        )
        events: list[tuple[str, float, object]] = []
        requests: list[tuple[float, object, float]] = []  # when it ended, the model's whole answer, seconds to the call

        def at() -> float:
            return time.monotonic() - Microphone.began

        ask = session.llm.parse_audio

        def recorded(wav, on_call=None):
            """The parser's own request, with its answer kept: what the model made of each utterance."""
            answer = ask(wav, on_call=on_call)
            requests.append((at(), answer, session.llm.last_call_s))
            if len(requests) == len(SENTENCES):
                answered.set()
            return answer

        monkeypatch.setattr(session.llm, "parse_audio", recorded)
        session.start(
            lambda text: events.append(("command", at(), json.loads(text))),
            stop_check=answered.is_set,
            on_heard=lambda words: events.append(("heard", at(), words)),
        )
        assert session.opened.wait(30) and not session.error, session.error
        session._thread.join(played_s + 60)  # the microphone thread ends when ``answered`` is set
        assert not session._thread.is_alive(), "the microphone thread did not end"
    finally:
        session.close()

    def thing(name: str) -> str:
        """An object as the detector knows it: Gemma says "ball", the decision model "sports ball"."""
        return (resolve_class(name) or name) if name else ""

    print()
    commands = [(when, data) for kind, when, data in events if kind == "command"]
    delivered = iter(commands)
    seconds = []
    for (sentence, intent, obj), end, (ended_at, answer, call_s) in zip(SENTENCES, ends, requests):
        line = (f"{voice_intent} {sentence!r:34} -> {answer.intent.value}({answer.object_name or ''}) "
                f"heard {answer.transcript!r}")
        if answer.intent is not Intent.UNKNOWN:  # the session delivered a command for this answer
            when, _ = next(delivered, (None, None))
            if when is not None:
                seconds.append(when - end)
                line += (f"; command {when - end:.2f} s after the last word "
                         f"(request {call_s:.2f} s to the call, answer complete {ended_at - end:.2f} s after it)")
        if (answer.intent, thing(answer.object_name or "")) != (intent, thing(obj)):
            line += f"  WRONG: should be {intent.value}({obj})"
        print(line)
    seconds.sort()
    if seconds:
        print(f"{voice_intent}: seconds from the last word to the command, min / median / max: "
              f"{seconds[0]:.2f} / {seconds[len(seconds) // 2]:.2f} / {seconds[-1]:.2f}; "
              f"server log: {tmp_path / 'llama-server.log'}")

    # One request per sentence, in the order spoken, each with its command and object: no half
    # sentence was sent, nothing was sent twice.
    assert len(requests) == len(SENTENCES), [answer for _, answer, _ in requests]
    assert [(answer.intent, thing(answer.object_name or "")) for _, answer, _ in requests] == [
        (intent, thing(obj)) for _, intent, obj in SENTENCES]
    # A command is delivered for each sentence that is one, once, in order; the question's answer
    # is `unknown` (asserted above from the model's own answer) and delivers nothing.
    expected = [(intent.value, thing(obj)) for _, intent, obj in SENTENCES if intent is not Intent.UNKNOWN]
    assert [(data["command"], thing(data["object"])) for _, data in commands] == expected
    assert all(data["heard"] == "" for _, data in commands)  # delivered at the call: the words follow
    # After each command its words, once: the order of the events is command, heard, command, heard...
    assert [kind for kind, _, _ in events] == ["command", "heard"] * len(expected), events
    # Each command came after its own sentence had ended and before the next one began, and
    # before its request's answer was complete.
    command_sentences = [index for index, (_, intent, _) in enumerate(SENTENCES) if intent is not Intent.UNKNOWN]
    for index, (when, _) in zip(command_sentences, commands):
        following = begins[index + 1] if index + 1 < len(begins) else played_s
        assert ends[index] < when < following, (SENTENCES[index][0], when)
        assert when < requests[index][0]
    # Gemma wrote down the words of every command; the decision model writes none.
    words_written = [bool(words) for kind, _, words in events if kind == "heard"]
    assert words_written == [voice_intent == "gemma4-e4b-it"] * len(expected), events
