"""so101_smolvla.voice without a Gemma server: settings, readiness, the answer's words with their pattern and
grammar, the synthetic voice, the fields, the client's reading, the listener's reports. (The rolling windows
and a window's way to Gemma are in test_voice_continuous.py.)"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import re
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from so101_smolvla.voice import intents
from so101_smolvla.voice import settings as voice_settings
from so101_smolvla.voice.client import GemmaClient, answer_pieces
from so101_smolvla.voice.commands import SPOKEN, TYPED, CommandFields, voice_ready
from so101_smolvla.voice.intents import (
    INTENT_SYSTEM_PROMPT,
    PATTERN,
    Intent,
    ParsedCommand,
    call_from_partial,
    command_from_words,
    grammar_text,
    heard_from_partial,
    response_format,
)
from so101_smolvla.voice.microphone import Listener, OffsetRemover, at_level, dbfs
from so101_smolvla.voice.server import RUNTIMES, GemmaServer
from so101_smolvla.voice.settings import BACKENDS, SetupFailure, check_ready, load_settings
from so101_smolvla.voice.speech import SyntheticVoice


def test_the_settings_file_is_the_one_source_and_names_no_model_inside_the_workshop():
    settings = load_settings()
    assert settings.backend in BACKENDS
    assert settings.server_bin.is_relative_to(voice_settings.ROOT / "third_party")
    assert not any("models/" in str(value) for value in settings.build.values())
    assert settings.igpu["repo"].count("/") == 1  # a Hugging Face repository, resolved in its cache
    assert "/" not in settings.igpu["model"] and "/" not in settings.igpu["mmproj"]  # file names of it
    assert settings.npu["model"].count(":") == 1  # a FastFlowLM tag, resolved in FastFlowLM's store
    assert set(settings.backends) == set(BACKENDS)
    # Each backend's section names the same things the same way.
    for section in settings.backends.values():
        assert {"model", "context", "ready_timeout_s"} <= set(section)
    # What TOML cannot write as null and the file leaves out reads as None.
    assert settings.npu["context"] is None and settings.listening["mic_device"] is None


def test_the_backend_and_its_sections_come_from_the_models_file_and_a_wrong_one_is_refused(tmp_path):
    text = voice_settings.MODELS_FILE.read_text(encoding="utf-8")
    settings = load_settings()
    # The voice path's own file names no backend and no model: one place for that, the models file.
    own = voice_settings.SETTINGS_FILE.read_text(encoding="utf-8")
    assert not re.search(r"(?m)^backend\s*=|^\[backends|^model\s*=", own)
    assert f'backend = "{settings.backend}"' in text
    other = "igpu" if settings.backend == "npu" else "npu"
    switched = tmp_path / "switched.toml"
    switched.write_text(re.sub(r'(?m)^backend = "\w+"', f'backend = "{other}"', text), encoding="utf-8")
    assert load_settings(models_path=switched).backend == other
    unknown = tmp_path / "unknown.toml"
    unknown.write_text(re.sub(r'(?m)^backend = "\w+"', 'backend = "cpu"', text), encoding="utf-8")
    with pytest.raises(SetupFailure, match="names the backend 'cpu'"):
        load_settings(models_path=unknown)
    lacking = tmp_path / "lacking.toml"
    lacking.write_text(text.replace("[command.npu]", "[elsewhere]"), encoding="utf-8")
    with pytest.raises(SetupFailure, match="no section command.npu"):
        load_settings(models_path=lacking)


def test_the_runners_server_is_not_on_the_default_port():
    settings = load_settings()
    assert settings.runner_port != int(settings.server["port"])


def test_readiness_names_each_missing_piece_and_the_setup_script():
    settings = load_settings()
    elsewhere = dataclasses.replace(
        settings,
        backends={
            "igpu": {
                **settings.igpu,
                "repo": "nobody/no-such-model",
                "build": {**settings.build, "server_bin": "third_party/nowhere/llama-server"},
            },
            "npu": {**settings.npu, "binary": "no-such-flm"},
        },
    )
    missing = check_ready(elsewhere, backend="igpu")
    assert any("no llama-server" in line and "scripts/setup_voice.sh" in line for line in missing)
    assert sum("no-such-model" in line and "Hugging Face cache" in line for line in missing) == 2
    assert not any("flm" in line for line in missing)  # the other backend's pieces are not asked for
    missing = check_ready(elsewhere, backend="npu")
    assert any("no no-such-flm on the PATH" in line and "FastFlowLM" in line for line in missing)
    assert not any("llama-server" in line for line in missing)
    assert check_ready(settings, backend="cpu") == ["unknown backend 'cpu'; one of igpu, npu"]
    ready, reason = voice_ready()
    assert isinstance(ready, bool) and reason
    if not ready:
        assert "setup_voice.sh" in reason or "espeak" in reason or "FastFlowLM" in reason


def _picks() -> list[str]:
    """Every pick answer the lists allow."""
    return [
        f"pick{size}{colour} {thing}{place}"
        for size in ("", *(f" {word}" for word in intents.SIZES))
        for colour in ("", *(f" {word}" for word in intents.COLOURS))
        for thing in intents.OBJECTS
        for place in ("", *(f" {word}" for word in intents.PLACES))
    ]


def test_every_command_has_its_word_and_the_pattern_the_grammar_and_the_instructions_name_each():
    # Every command but pick_place and unknown is one word of its own; those two are "pick" and "none".
    assert set(intents.SIMPLE.values()) == set(Intent) - {Intent.PICK_PLACE, Intent.UNKNOWN}
    assert len(set(intents.SIMPLE.values())) == len(intents.SIMPLE)
    text = grammar_text()
    for word in (*intents.SIMPLE, intents.PICK, intents.NONE):
        assert f'"{word}"' in text and f'"{word}"' in INTENT_SYSTEM_PROMPT, word
    # Every word an object or a place can be made of is in the pattern, the grammar and the instructions.
    for word in (*intents.SIZES, *intents.COLOURS, *intents.OBJECTS, *intents.PLACES):
        assert f'"{word}"' in text and word in PATTERN and re.search(rf"\b{word}\b", INTENT_SYSTEM_PROMPT), word
    assert 'root   ::= "\\"" words ": " said "\\""' in text
    # The instructions say the shape: the command's words, a colon, the words heard.
    assert "answer ONLY\nwith the command's words in lower case" in INTENT_SYSTEM_PROMPT
    assert "After the command's words write a colon and then the words the person said" in INTENT_SYSTEM_PROMPT
    assert response_format() == {
        "type": "json_schema", "json_schema": {"name": "command", "schema": {"type": "string", "pattern": PATTERN}},
    }


def test_every_allowed_answer_decodes_to_a_call_and_nothing_else_does():
    allowed = re.compile(PATTERN)
    assert command_from_words("stop") == ParsedCommand(Intent.STOP)
    assert command_from_words("learn") == ParsedCommand(Intent.LEARN_STACK)
    assert command_from_words("build") == ParsedCommand(Intent.REBUILD_STACK)
    assert command_from_words("copy") == ParsedCommand(Intent.GESTURE_MIMIC)
    assert command_from_words("none") == ParsedCommand(Intent.UNKNOWN)
    assert command_from_words("pick ball") == ParsedCommand(Intent.PICK_PLACE, {"object": "ball"})
    assert command_from_words("pick red cube box").call == ("pick_place", "red cube", "box")
    assert command_from_words("pick small green bottle bin").call == ("pick_place", "small green bottle", "bin")
    assert command_from_words("pick large cylinder").call == ("pick_place", "large cylinder", "")
    # The pattern Gemma is held to and the decoder agree, on every combination and on what is not one.
    picks = _picks()
    for words in [*intents.SIMPLE, intents.NONE, *picks]:
        assert allowed.match(f"{words}: ") and command_from_words(words) is not None, words
    assert len({command_from_words(words).call for words in picks}) == len(picks) == 3 * 5 * 5 * 3
    # Two colours can be written, so that an order naming two cubes is not read as a pick of the first.
    for words, call in (("pick red green cube box", ("pick_place", "red green cube", "box")),
                        ("pick small blue red ball", ("pick_place", "small blue red ball", ""))):
        assert allowed.match(f"{words}: ") and command_from_words(words).call == call, words
    for words in ("", "pick", "pick box", "pick red", "pick red small cube", "pick cube table", "pick two cubes",
                  "stop now", "learn_stack", "pick_place", "Stop", "pick red cube box bin", "pick red green blue cube"):
        assert not allowed.match(f"{words}: ") and command_from_words(words) is None, words
    # The words heard: letters, digits and plain punctuation; no quote, no colon, nothing else.
    for heard in ("", "Put the red cube in the box", "I'd like it, please. OK? Yes! Re-do 2"):
        assert allowed.match(f"stop: {heard}"), heard
    for heard in ('say "this"', "ümlaut", "tab\tin", "colon: no"):
        assert not allowed.match(f"stop: {heard}"), heard
    assert not allowed.match("stop") and not allowed.match("stop:")  # the colon and its space are part of the shape


def test_the_synthetic_voice_gives_a_clip_of_its_own_length_at_the_listening_rate():
    settings = load_settings()
    try:
        voice = SyntheticVoice(settings)
    except SetupFailure as exc:
        pytest.fail(str(exc))  # espeak-ng is a requirement of this machine, not a reason to skip
    clip, spoken_s = voice.clip("put the red cube in the bin")
    with wave.open(io.BytesIO(clip)) as handle:
        rate, frames = handle.getframerate(), handle.getnframes()
    assert rate == int(settings.listening["sample_rate"])
    assert frames == int(settings.synthetic_voice["clip_s"]) * rate
    assert 0.5 < spoken_s < settings.synthetic_voice["clip_s"] - voice.lead_s
    assert voice.recorded()["clip_s"] == settings.synthetic_voice["clip_s"]
    # The clip holds one sentence whole, so it is longer than a window of the live session.
    assert settings.synthetic_voice["clip_s"] > settings.listening["window_s"]


def test_a_sentence_sounds_the_same_whatever_the_voice_said_before_it():
    settings = load_settings()
    sentence = "robot, please place the green cube into the container"
    voice = SyntheticVoice(settings)
    first, spoken_s = voice.clip(sentence)
    # espeak-ng keeps state between sentences in one process; each is made in a process of its own.
    for other in ("stop robot", "put the red cube in the bin", "grab the blue one and drop it in the box"):
        voice.clip(other)
    assert voice.clip(sentence) == (first, spoken_s)
    assert SyntheticVoice(settings).clip(sentence) == (first, spoken_s)  # another voice object, the same sound
    assert voice.clip("stop robot")[0] != first
    nobody = dataclasses.replace(settings, synthetic_voice={**settings.synthetic_voice, "voice": "no-such-voice"})
    with pytest.raises(SetupFailure, match="espeak-ng could not speak 'stop robot': espeak-ng has no voice 'no-such-voice'"):
        SyntheticVoice(nobody).clip("stop robot")


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_server_refuses_a_port_something_already_answers_on(tmp_path, backend):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):  # answers 200 at any path: /health and /v1/models alike
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    other = HTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(SetupFailure, match="already answers"):
            GemmaServer(tmp_path, port=other.server_address[1], backend=backend).start()
    finally:
        other.shutdown()
        thread.join()
    assert not (tmp_path / RUNTIMES[backend].log_name).exists()


def test_each_backends_server_command_and_recorded_settings(tmp_path):
    settings = load_settings()
    igpu = GemmaServer(tmp_path, port=1, settings=settings, backend="igpu")
    assert igpu.recorded_settings["backend"] == "igpu" and igpu.recorded_settings["port"] == 1
    assert igpu.recorded_settings == {
        "backend": "igpu", "port": 1, "temperature": settings.request["temperature"],
        "context": settings.igpu["context"], "n_gpu_layers": settings.igpu["n_gpu_layers"],
        "parallel": settings.igpu["parallel"], "no_warmup": settings.igpu["no_warmup"],
        "hsa_override_gfx_version": settings.build["hsa_override_gfx_version"],
    }
    assert igpu.environment()["HSA_OVERRIDE_GFX_VERSION"] == str(settings.build["hsa_override_gfx_version"])
    assert igpu.log_path.name == "llama-server.log" and igpu.runtime.health_path == "/health"
    npu = GemmaServer(tmp_path, port=2, settings=settings, backend="npu")
    assert npu.recorded_settings == {
        "backend": "npu", "port": 2, "temperature": settings.request["temperature"],
        "model": settings.npu["model"], "pmode": settings.npu["pmode"], "context": settings.npu["context"],
    }
    # No graphics override for FastFlowLM: its environment is the caller's, as it is.
    assert npu.environment().get("HSA_OVERRIDE_GFX_VERSION") == os.environ.get("HSA_OVERRIDE_GFX_VERSION")
    assert npu.log_path.name == "flm-server.log" and npu.runtime.health_path == "/v1/models"
    # Each server reads its own section, the other's never: the readiness time is the backend's.
    assert igpu.options is settings.igpu and npu.options is settings.npu
    assert settings.igpu["ready_timeout_s"] != settings.npu["ready_timeout_s"]
    # Without a port given, the server is on the default one of the settings.
    assert GemmaServer(tmp_path, settings=settings, backend="npu").port == int(settings.server["port"])
    ready, reason = voice_ready("npu")
    if not ready:
        pytest.skip(f"the NPU backend is not set up: {reason}")
    command = npu.command()
    assert command[1:3] == ["serve", settings.npu["model"]]
    assert command[3:9] == ["--host", "127.0.0.1", "--port", "2", "--pmode", settings.npu["pmode"]]
    assert ("-c" in command) == (settings.npu["context"] is not None)


def test_the_fields_record_as_the_dictionary_a_result_stores():
    fields = CommandFields("put the red cube in the bin", TYPED, "pick_place", "red cube", "bin",
                           "put the red cube in the bin", 0.5)
    assert fields.as_record() == {
        "sentence": "put the red cube in the bin", "how": "typed", "command": "pick_place",
        "object": "red cube", "place": "bin", "heard": "put the red cube in the bin",
        "seconds": 0.5, "spoken_s": None, "call_seconds": None,
    }
    assert SPOKEN == "synthetic voice"


# ---- the answer read as it is written: the call before the words heard ---------------------------

ANSWER = '"pick green cube bin: Put the green cube in the bin"'
CALL_ENDS = ANSWER.index(":") + 1  # the colon after the command's words settles a pick command


def test_the_call_is_read_from_the_text_so_far_once_it_is_settled():
    for length in range(CALL_ENDS):
        assert call_from_partial(ANSWER[:length]) is None, ANSWER[:length]
    for length in range(CALL_ENDS, len(ANSWER) + 1):
        call = call_from_partial(ANSWER[:length])
        assert call.intent is Intent.PICK_PLACE and call.params == {"object": "green cube", "place": "bin"}
        assert call.transcript == ""  # the words are not part of the call
    assert call.call == ("pick_place", "green cube", "bin")
    # The words so far grow to all of them.
    assert heard_from_partial(ANSWER[:CALL_ENDS]) == ""
    assert heard_from_partial(ANSWER[: ANSWER.index("Put the gre") + len("Put the gre")]) == "Put the gre"
    assert heard_from_partial(ANSWER) == "Put the green cube in the bin"
    # A command without parameters, and "none", is settled by its word: before the colon.
    assert call_from_partial('"') is None and call_from_partial('"sto') is None
    stop = call_from_partial('"stop')
    assert stop.intent is Intent.STOP and stop.params == {} and stop.call == ("stop", "", "")
    assert call_from_partial('"stop: st').intent is Intent.STOP and heard_from_partial('"stop: st') == "st"
    assert call_from_partial('"none').intent is Intent.UNKNOWN
    assert call_from_partial('"learn').intent is Intent.LEARN_STACK
    # A pick command is not: a place can still follow. Without words heard the closing quote settles it.
    assert call_from_partial('"pick') is None and call_from_partial('"pick red cube box') is None
    assert call_from_partial('"pick red cube"').call == ("pick_place", "red cube", "")
    # Complete words that are no allowed answer are no command; text without the opening quote is no answer yet.
    assert call_from_partial('"fly: off').intent is Intent.UNKNOWN
    assert call_from_partial('"pick red: the red one').intent is Intent.UNKNOWN
    assert call_from_partial("") is None and call_from_partial("stop") is None
    assert heard_from_partial("") == "" and heard_from_partial('"stop') == ""
    # No answer is the beginning of another: the first word settles a command without parameters.
    answers = [*intents.SIMPLE, intents.NONE]
    assert not [(a, b) for a in answers for b in [*answers, intents.PICK] if a != b and b.startswith(a)]


def _events(*pieces: str, usage: dict | None = None) -> list[bytes]:
    lines = [b"data: " + json.dumps({"choices": [{"delta": {"content": piece}}]}).encode() for piece in pieces]
    if usage is not None:
        lines.append(b"data: " + json.dumps({"choices": [], "usage": usage}).encode())
    return [*lines, b"", b"data: [DONE]", b"data: " + json.dumps({"choices": [{"delta": {"content": "late"}}]}).encode()]


def test_the_servers_events_give_the_answers_pieces_and_its_usage_up_to_done():
    got = list(answer_pieces(_events("{", '"command"', ": ", usage={"completion_tokens": 25})))
    assert got == [("{", None), ('"command"', None), (": ", None), ("", {"completion_tokens": 25})]
    with pytest.raises(RuntimeError, match="the Gemma server reported: no model"):
        list(answer_pieces([b'data: {"error": "no model"}']))


class StreamingStandIn:
    """A stand-in for the Gemma server's stream: it writes the call, then waits for ``go`` before it
    writes the words heard. Plumbing only; the real servers are asked in the Gemma module."""

    def __init__(self) -> None:
        self.go = threading.Event()
        self.released: list[bool] = []
        self.requests: list[dict] = []
        stand_in = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                stand_in.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    self._send(ANSWER[:CALL_ENDS])
                    # Held until the client's on_call says it has the call (or the test's bound).
                    stand_in.released.append(stand_in.go.wait(10))
                    self._send(ANSWER[CALL_ENDS:])
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    stand_in.released.append(False)  # the client closed: it wanted the call only

            def _send(self, text):
                for character in text:
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": character}}]}).encode() + b"\n\n")
                self.wfile.flush()

            def log_message(self, *_):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = "http://127.0.0.1:%d" % self.server.server_address[1]

    def close(self) -> None:
        self.go.set()
        self.server.shutdown()
        self.thread.join()


def test_the_client_hands_over_the_call_while_the_words_are_still_unwritten():
    stand_in = StreamingStandIn()
    client = GemmaClient(stand_in.base, backend="npu")
    try:
        calls = []

        def on_call(call, seconds):
            calls.append((call, seconds))
            stand_in.go.set()  # only now may the server write the words heard

        command = client.parse("put the green cube in the bin", on_call=on_call)
        # The server was released by on_call, not by its bound: the call came before the words.
        assert stand_in.released == [True]
        assert len(calls) == 1 and calls[0][0].transcript == ""
        assert (calls[0][0].intent, calls[0][0].params) == (command.intent, command.params)
        assert command.intent is Intent.PICK_PLACE and command.transcript == "Put the green cube in the bin"
        assert calls[0][1] == client.last_call_s and 0 < client.last_call_s <= client.last_total_s
        assert stand_in.requests[0]["stream"] is True

        # A caller that has the sentence already gets the call and does not read the words.
        stand_in.go.clear()
        short = client.parse("put the green cube in the bin", words=False)
        assert short.intent is Intent.PICK_PLACE and short.params == {"object": "green cube", "place": "bin"}
        assert short.transcript == "put the green cube in the bin"  # the sentence stands in
    finally:
        client.close()
        stand_in.close()


LOUD = np.round(3000 * np.sin(2 * np.pi * 300 * np.arange(320) / 16000)).astype("<i2").tobytes()  # 20 ms of a tone
QUIET = bytes(640)


class ScriptedGemma:
    """A stand-in for the client: one scripted answer per request, the call handed to ``on_call`` first.
    Each request is held at its end until the test has fed the sound of the next one (``go``), so what
    the listener asks about next does not depend on how fast its thread runs. Plumbing only; the real
    Gemma hears the real listener in test_gemma_listening_integration.py."""

    def __init__(self, answers: list[ParsedCommand]) -> None:
        self.answers, self.wanted, self.asked = answers, [], threading.Semaphore(0)
        self.go = threading.Semaphore(0)
        self.before_call = lambda: None  # what happens while Gemma works on the request

    def parse_audio(self, wav, on_call=None):
        whole = self.answers[len(self.wanted)]
        self.before_call()
        self.wanted.append(on_call(ParsedCommand(whole.intent, dict(whole.params)), 1.5))  # False: no use for the words
        self.asked.release()
        assert self.go.acquire(timeout=10)
        return whole


class FedListener:
    """A Listener without a microphone: the test feeds its buffer the frames a microphone would."""

    def __init__(self, answers: list[ParsedCommand]) -> None:
        self.gemma, self.order, self.frames = ScriptedGemma(answers), [], 0
        self.listener = Listener(self.gemma, settings=load_settings())
        self.thread = threading.Thread(
            target=self.listener._request,
            args=(lambda command: self.order.append(("whole", command.call, command.transcript)), None,
                  lambda call, seconds: self.order.append(("call", call.call, call.transcript, seconds))),
        )
        self.began = None

    def feed(self, pattern: str) -> None:
        """One 20 ms frame per character: 'v' a loud tone that counts as speech, '.' silence."""
        self.began = self.began or time.monotonic()
        for character in pattern:
            self.listener.audio.feed(LOUD if character == "v" else QUIET, character == "v",
                                     self.began + self.frames * 0.02)
            self.frames += 1

    def run(self, *sounds: str) -> list:
        """Each sound is asked about once, in order; the reports of all of them."""
        self.feed(sounds[0])
        self.thread.start()
        for following in (*sounds[1:], None):
            assert self.gemma.asked.acquire(timeout=10)  # the request about the sound fed so far is at its end
            if following is not None:
                self.feed(following)
            else:
                self.listener.audio.close()  # nothing more: the listener's wait for speech ends
            self.gemma.go.release()
        self.thread.join(10)
        assert not self.thread.is_alive() and len(self.gemma.wanted) == len(sounds)
        return self.order


# A sentence of 0.6 s and its pause. Said one after the other, each one's window (the newest 3 s) still
# holds the ones before it.
SENTENCE = "v" * 30 + "." * 10
# The same after three seconds of silence: its window begins after the end of every window before it.
MUCH_LATER = "." * 150 + "v" * 30 + "." * 10
# Someone who has been talking for 2 s says "pick up the small red cube", pauses for 0.3 s and goes on:
# the window is cut at the pause.
GOES_ON = "v" * 100 + "." * 15 + "v" * 15


def test_the_listener_reports_a_command_once_per_sentence_the_call_first_then_the_whole_answer():
    pick = ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}, "grab the green cube")
    stop = ParsedCommand(Intent.STOP, {}, "stop robot")
    fed = FedListener([pick, ParsedCommand(Intent.UNKNOWN, {}, "nice weather"), pick, stop, stop, pick])
    order = fed.run(SENTENCE, SENTENCE, SENTENCE, SENTENCE, SENTENCE, MUCH_LATER)
    call, whole = ("pick_place", "green cube", ""), ("stop", "", "")
    assert order == [
        ("call", call, "", 1.5), ("whole", call, "grab the green cube"),  # the call, then the answer with its words
        # no command: nothing. The same command from a window that overlaps the delivering one: nothing.
        ("call", whole, "", 1.5), ("whole", whole, "stop robot"),
        ("call", whole, "", 1.5), ("whole", whole, "stop robot"),          # stop is reported every time
        ("call", call, "", 1.5), ("whole", call, "grab the green cube"),  # said again, after the first one's window
    ]
    # The words are read only of an answer that is reported.
    assert fed.gemma.wanted == [True, False, False, True, True, True]


def test_a_pick_command_without_a_place_waits_while_the_speaker_goes_on_and_the_whole_sentence_is_reported():
    half = ParsedCommand(Intent.PICK_PLACE, {"object": "small red cube"}, "pick up the small red cube")
    whole = ParsedCommand(Intent.PICK_PLACE, {"object": "small red cube", "place": "box"},
                          "pick up the small red cube and put it in the box")
    # The first window ends at the pause; the second is asked about with the first one's sound in front.
    fed = FedListener([half, whole])
    order = fed.run(GOES_ON, SENTENCE)
    assert order == [("call", whole.call, "", 1.5), ("whole", whole.call, whole.transcript)]
    assert fed.gemma.wanted == [False, True]  # the half is not reported, and its words are not read

    # The talk that followed was no pick command: the half is reported after all, without words.
    fed = FedListener([half, ParsedCommand(Intent.UNKNOWN, {}, "and that is all")])
    order = fed.run(GOES_ON, SENTENCE)
    assert order == [("call", half.call, "", 1.5)] and fed.gemma.wanted == [False, False]

    # Followed by a pause, it is reported at once.
    fed = FedListener([half])
    assert fed.run(SENTENCE) == [("call", half.call, "", 1.5), ("whole", half.call, half.transcript)]


def test_an_answer_about_sound_from_before_a_break_or_too_old_is_not_acted_on():
    pick = ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}, "grab the green cube")
    # The microphone's buffer overflows while Gemma answers: the answer is about sound that did not go on.
    broken = FedListener([pick])
    broken.gemma.before_call = lambda: broken.listener.audio.reset(time.monotonic())
    assert broken.run(SENTENCE) == [] and broken.gemma.wanted == [False]
    # A window that is older than max_age_s when its call is written.
    old = FedListener([pick])
    old.began = time.monotonic() - 2 * float(load_settings().listening["max_age_s"])
    assert old.run(SENTENCE) == [] and old.gemma.wanted == [False]


def test_the_microphones_constant_offset_is_taken_out_and_the_speech_is_left():
    rate = 16000
    time_s = np.arange(rate) / rate
    tone = 2000 * np.sin(2 * np.pi * 300 * time_s)  # in the range of speech
    samples = np.round(tone + 1028).astype("<i2")  # this machine's microphone: samples around +1028
    frames = [samples[i:i + 320].tobytes() for i in range(0, rate, 320)]
    assert -31 < dbfs(frames[10]) < -22  # with the offset in, the level is the offset's and the tone's together

    remover = OffsetRemover()
    cleaned = np.concatenate([np.frombuffer(remover.feed(frame), "<i2") for frame in frames]).astype(float)
    assert abs(cleaned[rate // 2:].mean()) < 2  # the offset is gone
    assert abs(np.abs(cleaned[rate // 2:]).max() - 2000) < 30  # the tone is as loud as it was
    assert np.abs(cleaned[:320]).max() < 2100  # no thump at the start: it begins settled on the first sample

    # Silence with the offset reads as silence, not as -30 dBFS.
    quiet = OffsetRemover()
    offset_only = np.full(320, 1028, "<i2").tobytes()
    assert dbfs(offset_only) > -31 and dbfs(quiet.feed(offset_only)) < -90
    # After a break in the frames it settles anew on the next first sample.
    quiet.reset()
    assert dbfs(quiet.feed(np.full(320, -500, "<i2").tobytes())) < -90


def test_an_utterance_is_sent_at_the_test_voices_level_and_silence_is_not_blown_up():
    time_s = np.arange(16000) / 16000
    speech = np.round(300 * np.sin(2 * np.pi * 300 * time_s)).astype("<i2")  # quiet: about -40 dBFS
    loud = np.frombuffer(at_level(speech.tobytes(), peak=0.5, most_gain=100), "<i2")
    assert abs(int(np.abs(loud).max()) - 16384) < 200  # half of full scale
    # The gain has a ceiling: quiet speech is raised by most_gain and no further.
    capped = np.frombuffer(at_level(speech.tobytes(), peak=0.5, most_gain=20), "<i2")
    assert abs(int(np.abs(capped).max()) - 6000) < 20
    # One click does not set the scale: the speech still comes up, the click is clipped.
    clicked = speech.copy()
    clicked[100] = 32000
    out = np.frombuffer(at_level(clicked.tobytes(), peak=0.5, most_gain=100), "<i2")
    assert abs(int(np.abs(out[200:]).max()) - 16384) < 400 and out[100] == 32767
    # Sound that is already loud is brought down to the same level; nothing stays nothing.
    assert abs(int(np.abs(np.frombuffer(at_level((speech * 100).astype("<i2").tobytes(), 0.5, 20), "<i2")).max()) - 16384) < 200
    assert at_level(bytes(640), 0.5, 20) == bytes(640) and at_level(b"", 0.5, 20) == b""
    settings = load_settings()
    assert settings.listening["sent_peak"] == settings.synthetic_voice["peak"]  # the level Gemma was measured with
