"""The whole listening session against the real Gemma, on each backend: one speaker who does not stop talking.

A stand-in microphone plays, in real time and in the synthetic voice, a speaker who says other things
before, between and after three commands, with 0.25 s between sentences: less than any end-of-speech
silence. The sound goes through the real Listener: offset removal, Silero, the rolling windows, the cut
at a pause, the quiet gate, the level raise and the requests to a Gemma server this test starts on the
runner's port of configs/voice.toml. From ~/GitHub/voice_intent (tests/test_live.py at its commit
4161915), where it runs against FastFlowLM.

Once per backend that is set up; a backend that is not set up is skipped with what is missing
(scripts/setup_voice.sh), anything else that goes wrong fails. The server needs its processor to
itself: no other Gemma server beside it.
"""

from __future__ import annotations

import logging
import threading
import time

import pytest

from so101_smolvla.voice.client import GemmaClient
from so101_smolvla.voice.commands import voice_ready
from so101_smolvla.voice.microphone import Listener
from so101_smolvla.voice.server import GemmaServer
from so101_smolvla.voice.settings import BACKENDS, load_settings
from so101_smolvla.voice.speech import SyntheticVoice
from spoken_track import monologue, stand_in_microphone

READY = {backend: voice_ready(backend) for backend in BACKENDS}
pytestmark = pytest.mark.skipif(
    not any(ready for ready, _ in READY.values()),
    reason="voice: " + "; ".join(f"{backend}: {reason}" for backend, (_, reason) in READY.items()),
)

# (sentence, the call it must give or None). No sentence is among the examples of the instructions
# except "learn the stack".
SPEECH = [
    ("so this is our new demo for the show", None),
    ("put the red cube in the box", ("pick_place", "red cube", "box")),
    ("and as you can see it works rather well", None),
    ("learn the stack", ("learn_stack", "", "")),
    ("we trained it for a couple of weeks", None),
    ("stop robot", ("stop", "", "")),
    ("that is all for today thank you", None),
]


@pytest.fixture(scope="module", params=BACKENDS)
def gemma(request, tmp_path_factory):
    """A Gemma server of this test's own and its client, warmed up; ended when the module is done."""
    backend = request.param
    ready, reason = READY[backend]
    if not ready:
        pytest.skip(f"{backend}: {reason}")
    settings = load_settings()
    server = GemmaServer(
        tmp_path_factory.mktemp(f"gemma-{backend}"), port=settings.runner_port, settings=settings, backend=backend
    ).start()
    client = GemmaClient(server.base, settings, backend=backend)
    try:
        print(f"{backend}: warmed up with one request, {client.warm_up():.2f} s; server log {server.log_path}")
        yield client
    finally:
        client.close()
        server.close()


def test_commands_are_delivered_while_the_speaker_keeps_talking(gemma, monkeypatch, caplog):
    import sounddevice

    settings = load_settings()
    rate = int(settings.listening["sample_rate"])
    sound, ends = monologue(SyntheticVoice(settings), [sentence for sentence, _ in SPEECH], rate)
    ended = threading.Event()
    Microphone = stand_in_microphone(sound, rate, ended.set)  # the track's last frame ends the run
    monkeypatch.setattr(sounddevice, "RawInputStream", Microphone)
    caplog.set_level(logging.INFO, logger="so101_smolvla.voice.microphone")

    calls, wholes = [], []
    listener = Listener(gemma, settings=settings).start(
        on_call=lambda call, seconds: calls.append((time.monotonic() - Microphone.began, call.call, seconds)),
        on_command=lambda whole: wholes.append((time.monotonic() - Microphone.began, whole.call, whole.transcript)),
    )
    try:
        assert listener.opened.wait(30) and listener.error is None, listener.error
        # One bounded wait for the microphone's own event: the track's last frame was handed out.
        assert ended.wait(len(sound) / 2 / rate + 30), "the track did not play to its end"
    finally:
        listener.stop()

    expected = [(end, call) for (_, call), end in zip(SPEECH, ends) if call]
    print()
    for record in caplog.records:
        print("  ", record.getMessage())
    for at, call, seconds in calls:
        print(f"{gemma.backend} delivered at {at:6.2f} s: {call}  (its request began {seconds:.2f} s before)")
    for at, call, heard in wholes:
        print(f"{gemma.backend} words at     {at:6.2f} s: {call}  heard {heard!r}")
    for end, call in expected:
        print(f"{gemma.backend} spoken until {end:6.2f} s: {call}")
    delivered = [call for _, call, _ in calls]
    # Each command is delivered, in the order spoken, with its object and place; nothing else is.
    # "stop" may come more than once: it is delivered every time Gemma returns it.
    once = [call for index, call in enumerate(delivered) if call[0] != "stop" or call not in delivered[:index]]
    assert once == [call for _, call in expected], delivered
    # None is delivered before its own last word, except by less than a request's first token (half
    # a sentence may already be the command), and none later than 8 s after it.
    first = {}
    for at, call, _ in calls:
        first.setdefault(call, at)
    for end, call in expected:
        assert end - 1.0 < first[call] < end + 8.0, (call, end, first[call])
    latency = sorted(first[call] - end for end, call in expected)
    print(f"{gemma.backend} seconds from a command's last word to its call: " + ", ".join(f"{s:.2f}" for s in latency))
    # Every delivered command's whole answer followed its call, and no microphone frame was lost.
    assert [call for _, call, _ in wholes] == delivered and listener.overflows == 0
