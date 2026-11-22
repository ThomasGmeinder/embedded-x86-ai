"""so101_smolvla.voice without a Gemma server: settings, readiness, grammar, the synthetic voice, the fields."""

from __future__ import annotations

import dataclasses
import io
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from so101_smolvla.voice import settings as voice_settings
from so101_smolvla.voice.commands import SPOKEN, TYPED, CommandFields, voice_ready
from so101_smolvla.voice.intents import INTENT_SYSTEM_PROMPT, Intent, command_from_json
from so101_smolvla.voice.server import GemmaServer
from so101_smolvla.voice.settings import SetupFailure, check_ready, grammar, load_settings
from so101_smolvla.voice.speech import SyntheticVoice


def test_the_settings_file_is_the_one_source_and_names_no_model_inside_the_workshop():
    settings = load_settings()
    assert settings.server_bin.is_relative_to(voice_settings.ROOT / "third_party")
    assert not any("models/" in str(value) for value in settings.llama_cpp.values())
    assert settings.gemma["repo"].count("/") == 1  # a Hugging Face repository, resolved in its cache
    assert not any(key.startswith("_") for section in dataclasses.astuple(settings) for key in section)


def test_the_runners_server_is_not_on_the_default_port():
    settings = load_settings()
    assert settings.runner_port != int(settings.server["port"])


def test_readiness_names_each_missing_piece_and_the_setup_script():
    settings = load_settings()
    elsewhere = dataclasses.replace(
        settings,
        llama_cpp={**settings.llama_cpp, "server_bin": "third_party/nowhere/llama-server"},
        gemma={**settings.gemma, "repo": "nobody/no-such-model"},
    )
    missing = check_ready(elsewhere)
    assert any("no llama-server" in line and "scripts/setup_voice.sh" in line for line in missing)
    assert sum("no-such-model" in line and "Hugging Face cache" in line for line in missing) == 2
    ready, reason = voice_ready()
    assert isinstance(ready, bool) and reason
    if not ready:
        assert "setup_voice.sh" in reason or "espeak" in reason


def test_the_grammar_names_every_command_and_the_instructions_describe_each():
    text = grammar()
    for intent in Intent:
        assert f'"\\"{intent.value}\\""' in text, intent
        assert f'"{intent.value}"' in INTENT_SYSTEM_PROMPT, intent


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


def test_the_server_refuses_a_port_something_already_answers_on(tmp_path):
    class Health(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    other = HTTPServer(("127.0.0.1", 0), Health)
    thread = threading.Thread(target=other.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(SetupFailure, match="already answers"):
            GemmaServer(tmp_path, port=other.server_address[1]).start()
    finally:
        other.shutdown()
        thread.join()
    assert not (tmp_path / "llama-server.log").exists()


def test_the_fields_record_as_the_dictionary_a_result_stores():
    fields = CommandFields("put the red cube in the bin", TYPED, "pick_place", "red cube", "bin",
                           "put the red cube in the bin", 0.5)
    assert fields.as_record() == {
        "sentence": "put the red cube in the bin", "how": "typed", "command": "pick_place",
        "object": "red cube", "place": "bin", "heard": "put the red cube in the bin",
        "seconds": 0.5, "spoken_s": None,
    }
    assert SPOKEN == "synthetic voice"
