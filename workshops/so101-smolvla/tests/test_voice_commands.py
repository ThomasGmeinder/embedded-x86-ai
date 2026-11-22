"""voice_commands: the Gemma worker's launcher, port and fields, without a server."""

from __future__ import annotations

import os
import subprocess

import yaml

from so101_smolvla.voice_commands import LAUNCHER, PORT, ROOT, WORKER, CommandFields


def test_the_worker_runs_through_the_voice_environment_launcher_and_imports_nothing_of_the_package():
    assert LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)
    assert WORKER.is_file()
    # The worker runs in the voice workshop's Python, where this package is
    # not installed: no import of it after the module's description.
    assert "so101_smolvla" not in WORKER.read_text(encoding="utf-8").split('"""', 2)[2]
    # The notebooks' kernel starts through the same launcher.
    kernel_script = (ROOT / "notebooks" / "run_notebooks.sh").read_text(encoding="utf-8")
    assert "run_voice_env.sh" in kernel_script
    assert "ryzen_ai_env.sh" not in kernel_script.split("# --- The kernel", 1)[1]


def test_the_workers_server_is_not_on_the_clients_own_port():
    section = yaml.safe_load(
        (ROOT / "voice" / "config" / "pipeline.yaml").read_text(encoding="utf-8")
    )["gemma4-e4b-it"]
    assert PORT != int(section["port"])


def test_the_launcher_refuses_an_empty_command():
    ran = subprocess.run([str(LAUNCHER)], capture_output=True, text=True)
    assert ran.returncode == 1 and "no command given" in ran.stderr


def test_the_fields_record_as_the_dictionary_a_result_stores():
    fields = CommandFields("put the red cube in the bin", "typed", "pick_place", "red cube", "bin",
                           "put the red cube in the bin", 0.5)
    assert fields.as_record() == {
        "sentence": "put the red cube in the bin", "how": "typed", "command": "pick_place",
        "object": "red cube", "place": "bin", "heard": "put the red cube in the bin",
        "seconds": 0.5, "spoken_s": None,
    }
