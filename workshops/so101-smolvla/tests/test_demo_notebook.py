"""notebooks/pick_place_demo.ipynb, run the way a person runs it: its kernel, the real Gemma, SAM 3 and SAPIEN.

Every cell runs (none is tagged to be left out: without a microphone the live cell says so and goes on). The
typed command of section 4 must come back as a pick of the red cube; whether the cube ended in the bin is
SmolVLA's result and is printed, not required. Section 6 must leave no server on Gemma's default port and no
utilization table running. Skipped while Gemma, the SAM 3 environment or the SAPIEN twin is not set up;
anything else that goes wrong fails. Needs a display for the utilization window and nothing answering on the
Gemma port. Jupyter's data and settings go to a temporary directory, as in the launcher's tests.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest

from so101_smolvla.sapien_config import SAPIEN_PYTHON, SAPIEN_SOURCE
from so101_smolvla.voice.commands import voice_ready
from test_notebooks_launcher import KERNEL, LAUNCHER, NOTEBOOKS, VENV, WORKSHOP, _environment
from test_voice_notebook import _code_after, _errors, _gemma_answers, _printed

NOTEBOOK = NOTEBOOKS / "pick_place_demo.ipynb"
RUN_LIMIT_S = 900  # the stack loads (about 1 min), one command (about 30 s), a new layout, the closes

SETTINGS = json.loads((WORKSHOP / "configs" / "sam3_reference.json").read_text())
SAM3_READY = (WORKSHOP / ".venv-sam3" / "bin" / "python").is_file() and (
    Path(os.path.expanduser(SETTINGS["compiled_dir"])) / "ARTIFACT_MANIFEST.json"
).is_file()
VOICE_READY, VOICE_REASON = voice_ready()

pytestmark = [
    pytest.mark.skipif(
        not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
        reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
    ),
    pytest.mark.skipif(not SAM3_READY, reason="run scripts/setup_sam3_env.sh and build the reference's model files"),
    pytest.mark.skipif(not VOICE_READY, reason=f"voice: {VOICE_REASON}"),
]


def test_the_demo_notebook_runs_a_typed_command_through_gemma_sam3_and_the_scene_and_stops_everything(tmp_path):
    assert os.environ.get("DISPLAY"), "the utilization window needs a display"
    assert not _gemma_answers(), "something already answers on the Gemma port; stop it before this test"
    env = _environment(tmp_path)
    subprocess.run([str(LAUNCHER), "--kernel-only"], env=env, check=True, timeout=120)
    out = tmp_path / "executed.ipynb"
    ran = subprocess.run(
        [str(VENV / "bin" / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
         f"--ExecutePreprocessor.kernel_name={KERNEL}", f"--ExecutePreprocessor.timeout={RUN_LIMIT_S}",
         "--output", out.stem, "--output-dir", str(out.parent), str(NOTEBOOK)],
        env=env, cwd=NOTEBOOKS, capture_output=True, text=True, timeout=RUN_LIMIT_S + 60,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not _gemma_answers(), "the notebook left its Gemma server running"
    executed = json.loads(out.read_text(encoding="utf-8"))
    cells = executed["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            assert not _errors(cell), "".join(cell["source"])

    started = _printed(_code_after(cells, "## 1."))
    for part in ("Gemma 4 server at", "SAM 3 worker ready", "SmolVLA", "SAPIEN scene, layout seed 0", "utilization and power table"):
        assert part in started, started
    live = _printed(_code_after(cells, "## 3."))
    assert "microphone" in live, live
    typed = _printed(_code_after(cells, "## 4."))
    assert re.search(r"^cube: red \| in the bin at the end: (True|False) \| the rule first held at step: (\d+|None)$", typed, flags=re.M), typed
    print(typed.strip())
    layout = _printed(_code_after(cells, "## 5."))
    assert layout.startswith("layout seed 1"), layout
    stopped = _printed(_code_after(cells, "## 6."))
    assert "table closed" in stopped and "stopped" in stopped, stopped
