"""so101_smolvla.voice without a Gemma server: settings, readiness, grammar and schema, the synthetic voice, the fields."""

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

from so101_smolvla.voice import settings as voice_settings
from so101_smolvla.voice.client import GemmaClient, answer_pieces
from so101_smolvla.voice.commands import SPOKEN, TYPED, CommandFields, voice_ready
from so101_smolvla.voice.intents import (
    FIELDS,
    HEARD_PATTERN,
    INTENT_SYSTEM_PROMPT,
    OBJECT_PATTERN,
    PLACE_PATTERN,
    Intent,
    ParsedCommand,
    call_from_partial,
    command_from_json,
    heard_from_partial,
    response_format,
    response_schema,
    schema_text,
)
from so101_smolvla.voice.microphone import _STOP, Listener, OffsetRemover, Utterances, at_level, dbfs
from so101_smolvla.voice.server import RUNTIMES, GemmaServer
from so101_smolvla.voice.settings import BACKENDS, SetupFailure, check_ready, grammar, load_settings
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


def test_the_grammar_and_the_schema_name_every_command_and_the_instructions_describe_each():
    text = grammar()
    schema = response_schema()
    # One answer is a list of four strings, in the grammar's order: command, object, place, heard.
    assert FIELDS == ("command", "object", "place", "heard")
    assert schema["type"] == "array" and schema["items"] is False
    assert schema["minItems"] == schema["maxItems"] == len(schema["prefixItems"]) == len(FIELDS)
    assert schema["prefixItems"][0]["enum"] == [intent.value for intent in Intent]
    assert [item.get("pattern") for item in schema["prefixItems"][1:]] == [OBJECT_PATTERN, PLACE_PATTERN, HEARD_PATTERN]
    for intent in Intent:
        assert f'"\\"{intent.value}\\""' in text, intent
        assert f'"{intent.value}"' in INTENT_SYSTEM_PROMPT, intent
    assert 'root    ::= "[" ws cmd ws "," ws obj ws "," ws place ws "," ws said ws "]"' in text
    # The instructions show the same shape and name the three fields after the command in its order.
    assert 'with a JSON list ["<name>", "<object or empty string>", "<place or empty string>", "<words>"]' in INTENT_SYSTEM_PROMPT
    assert 'the command, then "object", then "place", then "heard".' in INTENT_SYSTEM_PROMPT
    assert response_format() == {"type": "json_schema", "json_schema": {"name": "intent", "schema": schema}}
    assert schema_text() == schema_text() and schema_text().startswith('{"items":false')


@pytest.mark.parametrize(
    "pattern, accepted, rejected",
    [
        # obj ::= "\"\"" | "\"" word (" " word)? (" " word)? "\""   word ::= [a-z] [a-z]{0,15}
        (OBJECT_PATTERN, ["", "ball", "red cube", "small green bottle", "a" * 16],
         ["Red cube", "a b c d", "bottle!", "red  cube", " ball", "a" * 17]),
        # place ::= "\"\"" | "\"" word (" " word)? "\""
        (PLACE_PATTERN, ["", "box", "big box"], ["the big box", "Box", "box1"]),
        # said ::= "\"" hchar* "\""   hchar ::= [a-zA-Z0-9] | "'" | "," | "." | "?" | "!" | "-" | " "
        (HEARD_PATTERN, ["", "Put the red cube in the box", "I'd like it, please. OK? Yes! Re-do 2"],
         ['say "this"', "ümlaut", "tab\tin", "colon: no"]),
    ],
)
def test_the_schemas_patterns_accept_and_reject_what_the_grammar_does(pattern, accepted, rejected):
    for text in accepted:
        assert re.fullmatch(pattern, text), text
    for text in rejected:
        assert not re.fullmatch(pattern, text), text


def test_an_answer_is_read_from_gemmas_json_and_anything_else_is_none():
    command = command_from_json('["pick_place", "red cube", "bin", "put it"]')
    assert command.intent is Intent.PICK_PLACE and command.object_name == "red cube"
    assert command.place == "bin" and command.transcript == "put it"
    stop = command_from_json('["stop", "", "", "stop robot"]')
    assert stop.intent is Intent.STOP and stop.params == {} and stop.transcript == "stop robot"
    assert command_from_json('["fly"]').intent is Intent.UNKNOWN
    assert command_from_json("not json") is None and command_from_json("") is None
    assert command_from_json("[]") is None and command_from_json('["stop", 3]') is None
    # The shape of before 2026-11-23, an object with named keys, is not an answer any more.
    assert command_from_json('{"command": "stop", "object": "", "place": "", "heard": ""}') is None


def test_the_synthetic_voice_gives_a_clip_of_the_windows_length_at_the_listening_rate():
    settings = load_settings()
    try:
        voice = SyntheticVoice(settings)
    except SetupFailure as exc:
        pytest.fail(str(exc))  # espeak-ng is a requirement of this machine, not a reason to skip
    clip, spoken_s = voice.clip("put the red cube in the bin")
    with wave.open(io.BytesIO(clip)) as handle:
        rate, frames = handle.getframerate(), handle.getnframes()
    assert rate == int(settings.listening["sample_rate"])
    assert frames == int(settings.listening["window_s"]) * rate
    assert 0.5 < spoken_s < settings.listening["window_s"] - voice.lead_s
    assert voice.recorded()["clip_s"] == settings.listening["window_s"]


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

ANSWER = '["pick_place", "green cube", "bin", "Put the green cube in the bin"]'
CALL_ENDS = ANSWER.index(', "Put') + 1  # the comma after the place completes the call


def test_the_call_is_read_from_the_text_so_far_once_its_last_character_is_in():
    for length in range(CALL_ENDS):
        assert call_from_partial(ANSWER[:length]) is None, ANSWER[:length]
    for length in range(CALL_ENDS, len(ANSWER) + 1):
        call = call_from_partial(ANSWER[:length])
        assert call.intent is Intent.PICK_PLACE and call.params == {"object": "green cube", "place": "bin"}
        assert call.transcript == ""  # the words are not part of the call
    # The whole answer read both ways gives the same call; the words so far grow to all of them.
    whole = command_from_json(ANSWER)
    assert (whole.intent, whole.params) == (call.intent, call.params)
    assert heard_from_partial(ANSWER[:CALL_ENDS]) == ""
    assert heard_from_partial(ANSWER[: ANSWER.index("Put the gre") + len("Put the gre")]) == "Put the gre"
    assert heard_from_partial(ANSWER) == whole.transcript == "Put the green cube in the bin"
    # Empty fields stay out of the parameters; a command the model may not name is unknown.
    stop = call_from_partial('["stop", "", "", "st')
    assert stop.intent is Intent.STOP and stop.params == {}
    assert heard_from_partial('["stop", "", "", "st') == "st"
    assert call_from_partial('["fly", "", "",').intent is Intent.UNKNOWN
    assert call_from_partial("") is None and call_from_partial("not json") is None
    # Written without spaces, as a server may write it, it reads the same.
    tight = call_from_partial('["pick_place","green cube","bin","Put')
    assert tight.params == {"object": "green cube", "place": "bin"} and heard_from_partial('["stop","","","go') == "go"


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


def test_the_listener_reports_the_call_first_then_the_whole_answer_and_nothing_for_no_command():
    answers = [
        (ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}),
         ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}, "grab the green cube")),
        (ParsedCommand(Intent.UNKNOWN), ParsedCommand(Intent.UNKNOWN, {}, "nice weather")),
        # The same command again is a sentence of its own: utterances do not overlap, nothing is dropped.
        (ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}),
         ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}, "grab the green cube")),
    ]
    order, wanted = [], []

    class Client:
        def parse_audio(self, wav, on_call=None):
            call, whole = answers[len(wanted)]
            wanted.append(on_call(call, 1.5))  # False: the listener has no use for the words
            order.append("words written")
            return whole

    listener = Listener(Client(), settings=load_settings())
    now = time.monotonic()
    reported = threading.Thread(
        target=listener._request,
        args=(lambda command: order.append(("whole", command.transcript)), None,
              lambda found, seconds: order.append(("call", found.transcript, seconds))),
    )
    for _ in answers:  # utterances wait in the queue while Gemma answers an earlier one
        listener._put((bytes(2 * 16000), now))
    listener._put(_STOP)
    reported.start()
    reported.join(10)
    assert not reported.is_alive()
    assert order == [
        ("call", "", 1.5), "words written", ("whole", "grab the green cube"),
        "words written",
        ("call", "", 1.5), "words written", ("whole", "grab the green cube"),
    ]
    assert wanted == [True, False, True]


def test_an_utterance_is_handed_over_on_the_frame_that_ends_it():
    frame = bytes(640)  # 20 ms at 16 kHz
    cut = Utterances(16000, max_s=1.0, end_silence_ms=100, pre_roll_ms=40, min_voice_ms=40)
    assert cut.end_silence_frames == 5 and cut.max_frames == 50 and cut.before.maxlen == 2

    def feed(pattern: str) -> list[int | None]:
        """One frame per character, 'v' speech and '.' not; the length in frames of what each returned."""
        return [None if got is None else len(got) // 640 for got in (cut.feed(frame, c == "v") for c in pattern)]

    # Silence, then speech, then silence: handed over on the fifth silent frame, not a frame later,
    # with the two frames from before the speech: 2 + 4 + 5 frames.
    assert feed("....vvvv....") == [None] * 12 and cut.speaking
    assert feed(".") == [11] and not cut.speaking
    # A pause shorter than the end silence stays inside the utterance. It begins right after the
    # one before it, whose last two frames are the sound in front of it: Gemma hears a sentence
    # badly that starts at a clip's first sample.
    assert feed("vv...vv....") == [None] * 11
    assert feed(".") == [2 + 2 + 3 + 2 + 5]
    # Less speech than min_voice_ms (a click) is dropped.
    cut.clear()
    assert feed("v.....") == [None] * 6 and not cut.speaking
    # One that reaches the longest length is handed over as it is; the speech after it is the
    # next, again with the two frames before it.
    cut.clear()
    assert feed("v" * 49) == [None] * 49
    assert feed("v") == [50] and not cut.speaking
    assert feed("vv.....") == [None] * 6 + [2 + 2 + 5]
    # After a break in the microphone's frames nothing of the old sound is kept.
    feed("vvv")
    cut.clear()
    assert not cut.speaking and not cut.before and feed(".....") == [None] * 5
    with pytest.raises(ValueError, match="Invalid PCM frame length"):
        cut.feed(bytes(10), True)


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
