"""The real Gemma 4 on the three-cube test's sentences, typed and in the synthetic voice.

The worker starts its own server on voice_commands.PORT in the voice
workshop's environment and stops it at the end. Skipped while that workshop
(its Python, llama-server, the model files) is not set up; anything else that
goes wrong fails.
"""

from __future__ import annotations

import pytest

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.cube_commands import (
    REFUSED_SENTENCES,
    call_text,
    choose_cube,
    pick_sentences,
)
from so101_smolvla.voice_commands import VoiceCommands, voice_workshop_ready

READY, REASON = voice_workshop_ready()
pytestmark = pytest.mark.skipif(not READY, reason=f"voice workshop: {REASON}")


@pytest.fixture(scope="module")
def voice(tmp_path_factory):
    worker = VoiceCommands(log_dir=tmp_path_factory.mktemp("gemma"))
    try:
        yield worker
    finally:
        worker.close()


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
            f"{fields.how:15} {sentence!r:62} heard {fields.heard!r:62} "
            f"{call_text(fields.command, fields.object, fields.place):48} "
            f"-> {choice.cube or 'refused: ' + choice.refusal:45} should be {cube or 'refused':8} "
            f"{fields.seconds:.2f} s"
        )
    seconds = sorted(fields.seconds for _, _, fields, _ in rows)
    right = sum(choice.cube == cube for _, cube, _, choice in rows)
    print(
        f"{rows[0][2].how}: {right} of {len(rows)} as they should be; seconds per request "
        f"min {seconds[0]:.2f}, median {seconds[len(seconds) // 2]:.2f}, max {seconds[-1]:.2f}"
    )
    return rows


def test_the_worker_reports_the_server_it_started_and_the_client_it_is(voice):
    provenance = voice.provenance
    assert provenance["route"].startswith("gemma4-e4b-it voice client")
    assert provenance["base"].endswith(":8083")
    assert provenance["settings"]["parallel"] == 2 and provenance["settings"]["ctx_size"] == 4096
    assert provenance["model_gguf"].endswith(".gguf") and provenance["mmproj"].endswith(".gguf")
    assert provenance["branch_commit"] == "1d231e1"
    assert len(provenance["grammar_sha256"]) == 64 and len(provenance["instructions_sha256"]) == 64
    assert provenance["synthetic_voice"]["sample_rate"] == 16000
    assert provenance["synthetic_voice"]["clip_s"] == 6
    # The environment launcher gave the server the voice workshop's graphics override.
    assert provenance["gfx_override_in_environment"] is not None
    print({key: provenance[key] for key in ("settings", "gfx_override_in_environment", "server_log")})


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
    print(f"spoken sentences that gave their cube: {heard} of {len(pick_sentences())}")
    assert heard > 0
