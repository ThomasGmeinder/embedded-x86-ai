"""configs/models.toml and its one Python reader, so101_smolvla.models: which model does what, and where."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from so101_smolvla import models, track
from so101_smolvla.models import COMMAND_BACKENDS, MODELS_FILE, SetupFailure, load_models, motion_checkpoint
from so101_smolvla.voice import settings as voice_settings

WORKSHOP = Path(__file__).resolve().parents[1]


def test_the_file_names_the_three_models_and_where_each_runs():
    loaded = load_models()
    assert set(loaded) == {"command", "outlines", "motion"}
    command = loaded["command"]
    assert command["backend"] in COMMAND_BACKENDS
    # Each backend's section names the same things the same way; the NPU's context may be left out.
    for name in COMMAND_BACKENDS:
        assert {"model", "context", "ready_timeout_s"} <= set(command[name])
    assert command["npu"]["model"].count(":") == 1  # a FastFlowLM tag
    assert command["igpu"]["repo"].count("/") == 1  # a Hugging Face repository
    assert loaded["outlines"]["device"] == loaded["motion"]["device"] == "igpu"
    assert set(loaded["outlines"]) == {"device", "checkout", "commit", "model_dir", "compiled_dir"}
    assert loaded["motion"]["checkpoint"].count("/") == 1  # a Hugging Face repository


def test_the_checkpoint_the_runs_load_is_the_one_of_the_file():
    assert track.CHECKPOINT == motion_checkpoint() == load_models()["motion"]["checkpoint"]


def test_the_voice_settings_and_the_setup_scripts_read_this_file():
    loaded = load_models()
    assert voice_settings.SetupFailure is SetupFailure and voice_settings.BACKENDS is COMMAND_BACKENDS
    settings = voice_settings.load_settings()
    assert settings.backend == loaded["command"]["backend"]
    assert settings.npu == loaded["command"]["npu"] and settings.igpu == loaded["command"]["igpu"]
    # The shell readers: the backend the voice setup would use, and SAM 3's commit.
    said = subprocess.run(
        ["bash", "scripts/setup_voice.sh", "--check"], cwd=WORKSHOP, capture_output=True, text=True, timeout=120
    )
    assert said.stdout.splitlines()[0] == f"backend: {loaded['command']['backend']}", said.stdout + said.stderr
    printed = subprocess.run(
        ["bash", "scripts/run_sam3_reference.sh", "--print", "commit"],
        cwd=WORKSHOP, capture_output=True, text=True, timeout=60,
    )
    assert printed.stdout.strip() == loaded["outlines"]["commit"], printed.stderr


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda text: re.sub(r'(?m)^backend = "\w+"', 'backend = "cpu"', text), "names the backend 'cpu'"),
        # Both headers of the section: [command.igpu.build] alone would still make a command.igpu.
        (lambda text: text.replace("[command.igpu", "[elsewhere"), "no section command.igpu"),
        (lambda text: text.replace("[motion]", "[elsewhere]"), "no section motion"),
        # SAM 3 and SmolVLA run on the integrated graphics only: another device is refused, not ignored.
        (lambda text: re.sub(r'(\[outlines\][^\[]*?device = )"igpu"', r'\1"npu"', text),
         "gives outlines.device as 'npu'"),
        (lambda text: re.sub(r'(\[motion\][^\[]*?device = )"igpu"', r'\1"cpu"', text),
         "gives motion.device as 'cpu'"),
    ],
)
def test_a_file_that_is_wrong_is_refused_with_what_is_wrong(tmp_path, change, message):
    text = MODELS_FILE.read_text(encoding="utf-8")
    changed = change(text)
    assert changed != text  # the change found its place in the file
    path = tmp_path / "models.toml"
    path.write_text(changed, encoding="utf-8")
    with pytest.raises(SetupFailure, match=message):
        load_models(path)
    assert models.GRAPHICS == "igpu"
