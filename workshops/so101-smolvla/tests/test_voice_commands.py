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

import pytest

from so101_smolvla.voice import settings as voice_settings
from so101_smolvla.voice.client import GemmaClient, answer_pieces
from so101_smolvla.voice.commands import SPOKEN, TYPED, CommandFields, voice_ready
from so101_smolvla.voice.intents import (
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
from so101_smolvla.voice.microphone import _STOP, Listener
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
    assert schema["properties"]["command"]["enum"] == [intent.value for intent in Intent]
    for intent in Intent:
        assert f'"\\"{intent.value}\\""' in text, intent
        assert f'"{intent.value}"' in INTENT_SYSTEM_PROMPT, intent
    assert schema["required"] == ["command", "object", "place", "heard"]  # the grammar's order
    assert schema["additionalProperties"] is False
    assert response_format() == {"type": "json_schema", "json_schema": {"name": "intent", "schema": schema}}
    assert schema_text() == schema_text() and schema_text().startswith('{"additionalProperties":false')


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
    command = command_from_json('{"command": "pick_place", "object": "red cube", "place": "bin", "heard": "put it"}')
    assert command.intent is Intent.PICK_PLACE and command.object_name == "red cube"
    assert command.place == "bin" and command.transcript == "put it"
    assert command_from_json('{"command": "fly"}').intent is Intent.UNKNOWN
    assert command_from_json("not json") is None and command_from_json("") is None


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

ANSWER = '{"command": "pick_place", "object": "green cube", "place": "bin", "heard": "Put the green cube in the bin"}'
CALL_ENDS = ANSWER.index(', "heard"') + 1  # the comma after the place completes the call


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
    stop = call_from_partial('{"command": "stop", "object": "", "place": "", "heard": "st')
    assert stop.intent is Intent.STOP and stop.params == {}
    assert call_from_partial('{"command": "fly", "object": "", "place": "",').intent is Intent.UNKNOWN
    assert call_from_partial("") is None and call_from_partial("not json") is None


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


def test_the_listener_reports_the_call_first_then_the_whole_answer_and_a_repeat_neither():
    call = ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"})
    whole = ParsedCommand(Intent.PICK_PLACE, {"object": "green cube"}, "grab the green cube")
    order = []
    asked = threading.Semaphore(0)  # released each time the listener gives the client a window

    class Client:
        def parse_audio(self, wav, on_call=None):
            asked.release()
            on_call(call, 1.5)
            order.append("words written")
            return whole

    listener = Listener(Client(), settings=load_settings())
    now = time.monotonic()
    window = bytes(2 * 16000 * 6)
    reported = threading.Thread(
        target=listener._request,
        args=(lambda command: order.append(("whole", command.transcript)), None,
              lambda found, seconds: order.append(("call", found.transcript, seconds))),
    )
    reported.start()
    listener._pending.put((window, now - 6, now))
    # The queue holds one window: the next goes in once the client was given this one.
    assert asked.acquire(timeout=10)
    listener._pending.put((window, now - 4, now + 2))  # the next hop holds the same sentence: within the cooldown
    assert asked.acquire(timeout=10)
    listener._pending.put(_STOP)
    reported.join(10)
    assert not reported.is_alive()
    assert order == [("call", "", 1.5), "words written", ("whole", "grab the green cube"), "words written"]
