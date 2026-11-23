"""The real Gemma 4 on the three-cube test's sentences, typed and in the synthetic voice, on each backend.

VoiceCommands starts its own server on the runner's port of configs/voice.toml
and stops it at the end; once per backend that is set up (llama-server on the
integrated graphics, FastFlowLM on the NPU). A backend that is not set up is
skipped with what is missing (scripts/setup_voice.sh); anything else that goes
wrong fails. The module is skipped only when neither backend is set up.
"""

from __future__ import annotations

import os
import re

import pytest

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.cube_commands import (
    REFUSED_SENTENCES,
    call_text,
    choose_cube,
    pick_sentences,
)
from so101_smolvla.voice.commands import VoiceCommands, voice_ready
from so101_smolvla.voice.examples import (
    NOT_STOP_ORDERS,
    SPOKEN_NOT_STOP_ORDERS,
    SPOKEN_STOP_ORDERS,
    STOP_ORDERS,
)
from so101_smolvla.voice.settings import BACKENDS, load_settings

READY = {backend: voice_ready(backend) for backend in BACKENDS}
pytestmark = pytest.mark.skipif(
    not any(ready for ready, _ in READY.values()),
    reason="voice: " + "; ".join(f"{backend}: {reason}" for backend, (_, reason) in READY.items()),
)


ENVIRONMENT_OVERRIDE_BEFORE = os.environ.get("HSA_OVERRIDE_GFX_VERSION")


@pytest.fixture(scope="module", params=BACKENDS)
def voice(request, tmp_path_factory):
    backend = request.param
    ready, reason = READY[backend]
    if not ready:
        pytest.skip(f"{backend}: {reason}")
    commands = VoiceCommands(log_dir=tmp_path_factory.mktemp(f"gemma-{backend}"), backend=backend)
    try:
        yield commands
    finally:
        commands.close()


def _ask_all(voice, spoken: bool) -> list[tuple]:
    rows = []
    for sentence, cube in pick_sentences():
        fields = voice.ask(sentence, spoken=spoken)
        rows.append((sentence, cube, fields, choose_cube(fields.command, fields.object)))
    for sentence, reason in REFUSED_SENTENCES:
        fields = voice.ask(sentence, spoken=spoken)
        rows.append((sentence, None, fields, choose_cube(fields.command, fields.object)))
    for sentence, cube, fields, choice in rows:
        print(
            f"{voice.backend:4} {fields.how:15} {sentence!r:62} heard {fields.heard!r:62} "
            f"{call_text(fields.command, fields.object, fields.place):48} "
            f"-> {choice.cube or 'refused: ' + choice.refusal:45} should be {cube or 'refused':8} "
            f"{fields.seconds:.2f} s"
        )
    seconds = sorted(fields.seconds for _, _, fields, _ in rows)
    right = sum(choice.cube == cube for _, cube, _, choice in rows)
    print(
        f"{voice.backend} {rows[0][2].how}: {right} of {len(rows)} as they should be; seconds per request "
        f"min {seconds[0]:.2f}, median {seconds[len(seconds) // 2]:.2f}, max {seconds[-1]:.2f}"
    )
    # The same requests, up to the moment the call was complete (the words heard still being written).
    to_call = sorted(fields.call_seconds for _, _, fields, _ in rows if fields.call_seconds is not None)
    assert len(to_call) == len(rows)  # every answer of the test's sentences holds a call
    print(
        f"{voice.backend} {rows[0][2].how}: seconds to the complete call "
        f"min {to_call[0]:.2f}, median {to_call[len(to_call) // 2]:.2f}, max {to_call[-1]:.2f}"
    )
    if voice.client.last_usage:
        usage = voice.client.last_usage
        print(f"{voice.backend} last answer's usage: " + ", ".join(
            f"{key} {usage[key]}" for key in ("prompt_tokens", "completion_tokens", "prefill_speed_tps", "decoding_speed_tps")
            if key in usage
        ))
    return rows


def test_the_voice_reports_the_server_it_started_and_the_client_it_is(voice):
    settings = load_settings()
    provenance = voice.provenance
    assert provenance["route"].startswith("gemma4-e4b-it voice client")
    assert provenance["backend"] == voice.backend
    assert provenance["base"].endswith(f":{settings.runner_port}") and settings.runner_port == 8083
    assert provenance["settings"]["backend"] == voice.backend
    assert len(provenance["instructions_sha256"]) == 64
    assert provenance["synthetic_voice"]["sample_rate"] == 16000
    assert provenance["synthetic_voice"]["clip_s"] == 6
    if voice.backend == "igpu":
        assert provenance["settings"]["parallel"] == 2 and provenance["settings"]["context"] == 4096
        assert provenance["model_gguf"].endswith(".gguf") and provenance["mmproj"].endswith(".gguf")
        # The server names the commit it was built from; it is the pinned one of the settings.
        built = re.search(r"commit ([0-9a-f]{7,})", provenance["server_version"])
        assert built and settings.build["commit"].startswith(built.group(1)), provenance["server_version"]
        assert len(provenance["grammar_sha256"]) == 64
        # The graphics override is the server process's alone: recorded as its setting, never put into
        # this process, where SmolVLA's PyTorch would see it.
        assert provenance["settings"]["hsa_override_gfx_version"] == settings.build["hsa_override_gfx_version"]
    else:
        assert provenance["settings"]["model"] == provenance["model_tag"] == settings.npu["model"]
        assert provenance["settings"]["pmode"] == settings.npu["pmode"]
        version = re.fullmatch(r"FastFlowLM (\d+)\.(\d+)\.(\d+)", provenance["server_version"])
        assert version, provenance["server_version"]
        minimum = tuple(int(part) for part in settings.npu["min_version"].split("."))
        assert tuple(int(part) for part in version.groups()) >= minimum
        assert provenance["model_name"].endswith("-NPU2") and provenance["quantization"].startswith("Q")
        assert provenance["context_length"] >= 1024
        assert len(provenance["schema_sha256"]) == 64
    assert os.environ.get("HSA_OVERRIDE_GFX_VERSION") == ENVIRONMENT_OVERRIDE_BEFORE
    print({key: provenance[key] for key in ("route", "settings", "server_version", "server_log")})


def test_the_call_is_handed_over_while_gemma_still_writes_the_words(voice):
    """The real server's stream: the call arrives before the answer ends, and is the answer's call."""
    sentence = "put the green cube in the bin"
    client = voice.client
    handed = []
    command = client.parse(sentence, on_call=lambda call, seconds: handed.append((call, seconds, client.last_total_s)))
    assert len(handed) == 1
    call, call_s, total_at_call = handed[0]
    assert total_at_call is None  # the request had not ended when the call was handed over
    assert call.transcript == "" and (call.intent, call.params) == (command.intent, command.params)
    assert command.intent.value == "pick_place" and command.params["object"] == "green cube"
    assert command.transcript  # the words followed the call
    assert call_s == client.last_call_s and 0 < call_s < client.last_total_s
    print(
        f"{voice.backend} typed {sentence!r}: call after {call_s:.2f} s, the whole answer after "
        f"{client.last_total_s:.2f} s; heard {command.transcript!r}"
    )
    # A caller that has the sentence stops at the call; the server answers the next request as before.
    short = client.parse(sentence, words=False)
    assert (short.intent, short.params) == (command.intent, command.params) and short.transcript == sentence
    print(f"{voice.backend} typed, the call only: {client.last_call_s:.2f} s to the call, {client.last_total_s:.2f} s in all")
    again = client.parse("grab the red one and drop it in the box")
    assert again.intent.value == "pick_place" and again.params["object"].startswith("red") and again.transcript
    print(f"{voice.backend} the request after it: call after {client.last_call_s:.2f} s, in all {client.last_total_s:.2f} s")
    # The warm-up is an ordinary request that stops at its call; the server answers after it.
    warm = client.warm_up()
    assert warm == client.last_total_s >= client.last_call_s > 0  # it ended at its call, a clock reading later
    assert client.parse("stop robot").intent.value == "stop"
    print(f"{voice.backend} a warm-up request: {warm:.2f} s")


def test_every_typed_sentence_gives_its_cube_and_every_other_one_is_refused(voice):
    rows = _ask_all(voice, spoken=False)
    wrong = [
        (sentence, choice.cube or choice.refusal, cube)
        for sentence, cube, _, choice in rows
        if choice.cube != cube
    ]
    assert not wrong, wrong
    assert {cube for _, cube, _, _ in rows} == {None, *BIN_CUBE_COLORS}
    # Gemma's words for what it heard are the sentence, as the grammar asks.
    assert all(fields.heard for _, _, fields, _ in rows)


def test_no_spoken_sentence_moves_a_cube_it_does_not_name(voice):
    """The synthetic voice: a sentence may be refused (not heard well), never read as another cube."""
    rows = _ask_all(voice, spoken=True)
    wrong_cube = [
        (sentence, choice.cube, cube)
        for sentence, cube, _, choice in rows
        if choice.cube is not None and choice.cube != cube
    ]
    assert not wrong_cube, wrong_cube
    assert all(fields.spoken_s and fields.spoken_s < 6 for _, _, fields, _ in rows)
    # The count is reported, not asserted: it is the figure of the testing ledger.
    heard = sum(choice.cube == cube for _, cube, _, choice in rows if cube is not None)
    print(f"{voice.backend} spoken sentences that gave their cube: {heard} of {len(pick_sentences())}")
    assert heard > 0


def _stop_rows(voice, sentences, wanted: str, spoken: bool) -> list[tuple]:
    rows = [(sentence, voice.ask(sentence, spoken=spoken)) for sentence in sentences]
    for sentence, fields in rows:
        print(
            f"{voice.backend:4} {fields.how:15} {sentence!r:30} heard {fields.heard!r:30} -> {fields.command:8} "
            f"should be {wanted:8} {fields.seconds:.2f} s"
        )
    return [(sentence, fields.command, fields.heard) for sentence, fields in rows if fields.command != wanted]


def test_a_typed_order_to_stop_is_stop_and_talk_about_stopping_is_not(voice):
    """ "stop" and "stop robot" in their typed forms; the instructions' stop entry of 2026-11-23."""
    wrong = _stop_rows(voice, STOP_ORDERS, "stop", spoken=False)
    wrong += _stop_rows(voice, NOT_STOP_ORDERS, "unknown", spoken=False)
    assert not wrong, wrong


def test_a_spoken_order_to_stop_is_stop_and_talk_about_stopping_is_not(voice):
    wrong = _stop_rows(voice, SPOKEN_STOP_ORDERS, "stop", spoken=True)
    wrong += _stop_rows(voice, SPOKEN_NOT_STOP_ORDERS, "unknown", spoken=True)
    assert not wrong, wrong
