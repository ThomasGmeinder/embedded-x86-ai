"""notebooks/pick_place_demo.ipynb, run the way a person runs it: its kernel, the real Gemma, SAM 3 and SAPIEN.

Every cell runs in a row, as "Run All" runs them, and after them three cells of this test. Section 1 has to
bring up every part. Section 3 has to open the microphone: this machine has one, so a microphone that does not
open fails the test. Section 4 shows its stop button and must stop nothing: on 2026-11-23 it stopped
everything, "Run All" ended the demo a second after the live cell, and the picture stood on its first frame,
"no microphone". The test's first cell then checks the microphone in the running demo: frames keep reaching
the session, the picture keeps being drawn on them, the panel says it is listening, no read overflowed, and
Gemma's server still answers. Its second cell types the session's first command and checks that the warm-up
of the start worked: the command's first outline and first plan each took less than half of what the
warm-up's run of that model took. Its third cell presses the stop button, which must leave no server on
Gemma's default port and no utilization table running. (Until 2026-11-23 the notebook also ran one typed command and
a new layout from cells, its sections 4 and 5; the user had them removed, so a command now goes through the
live cell's text box or the microphone only.) Skipped while Gemma, the SAM 3 environment or the SAPIEN twin is
not set up; anything else that goes wrong fails. Needs a display for the utilization window, a microphone and
nothing answering on the Gemma port. Jupyter's data and settings go to a temporary directory, as in the
launcher's tests.
"""
from __future__ import annotations

import copy
import json
import os
import subprocess
import uuid
from pathlib import Path

import pytest

from so101_smolvla.models import load_models
from so101_smolvla.sapien_config import SAPIEN_PYTHON, SAPIEN_SOURCE
from so101_smolvla.voice.commands import voice_ready
from test_notebooks_launcher import KERNEL, LAUNCHER, NOTEBOOKS, VENV, WORKSHOP, _environment
from test_voice_notebook import _code_after, _errors, _gemma_answers, _printed

NOTEBOOK = NOTEBOOKS / "pick_place_demo.ipynb"
RUN_LIMIT_S = 900  # the stack loads (about 1 min), the live picture, the closes

SETTINGS = load_models()["outlines"]  # the SAM 3 section of configs/models.toml
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


# The test's first cell, run after every cell of the notebook: the microphone in the running demo. Each wait
# is one bounded wait for the view's own event that it has drawn; at rest the microphone's frames are the
# only thing that makes it draw.
MICROPHONE_CHECK = '''
import json, time
from so101_smolvla.demo_console import voice_state
_listener, _session = stack.listener, stack.session
_frame_before, _drawn_before, _draws = _session.voice.time, view.frames, 0
for _ in range(10):
    view.drawn.clear()
    if not view.drawn.wait(5):
        break
    _draws += 1
_voice = _session.voice
print("MICROPHONE " + json.dumps({
    "opened": _listener.opened.is_set(), "error": _listener.error, "overflows": _listener.overflows,
    "threads_alive": [thread.is_alive() for thread in _listener._threads],
    "available": _voice.available, "panel": voice_state(_voice, time.monotonic())[0],
    "draws_awaited": _draws, "frames_drawn": view.frames - _drawn_before,
    "a_later_frame_arrived": _voice.time > _frame_before,
    "gemma_answers": stack.server.answers(), "stop_pressed": stop.done,
}))
'''
# The test's second cell: the first command of the session, typed as into the text box, run to its end
# (one bounded wait for the session's own event that it is idle). What the warm-up of the start took and
# what the command's first plan took are the stack's and the pick loop's own figures.
FIRST_COMMAND = '''
import json
stack.session.submit("put the red cube in the bin")
_ended = stack.session.wait_idle(300)
_record = stack.scene.record
print("WARMUP " + json.dumps({
    "command_ended": _ended,
    "cube": stack.session.results[-1]["cube"] if stack.session.results else None,
    "warm_up_outline_s": stack.warmed["sam3"], "warm_up_plan_s": stack.warmed["smolvla"],
    "first_command_outline_s": sum(camera["seconds"] for camera in _record.sam3_found[0].values()),
    "first_command_plan_s": _record.plan_seconds[0],
    "plans_of_the_command": len(_record.plan_seconds),
}))
'''
# The test's third cell: the stop button pressed, as a person presses it.
PRESS_STOP = "stop.button.click()\nprint(stop.said)\n"


def test_run_all_leaves_the_demo_running_with_a_live_microphone_and_the_button_stops_everything(tmp_path):
    assert os.environ.get("DISPLAY"), "the utilization window needs a display"
    assert not _gemma_answers(), "something already answers on the Gemma port; stop it before this test"
    env = _environment(tmp_path)
    # The kernel as a person's kernel runs: without pytest's variable. With it in its environment the
    # kernel does not show what its child processes write (ipykernel/iostream.py: "Pytest set its own
    # capture. Don't redirect from within pytest"), and the SAM 3 worker, its launcher and the simulator
    # worker are child processes: their lines, and anything they write to standard error, were missing
    # from the cell this test reads (2026-11-24).
    env.pop("PYTEST_CURRENT_TEST", None)
    subprocess.run([str(LAUNCHER), "--kernel-only"], env=env, check=True, timeout=120)

    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    template = next(c for c in notebook["cells"] if c["cell_type"] == "code")
    for source in (MICROPHONE_CHECK, FIRST_COMMAND, PRESS_STOP):
        cell = copy.deepcopy(template)
        cell.update(source=source, metadata={}, outputs=[], execution_count=None)
        if "id" in cell:
            cell["id"] = uuid.uuid4().hex[:8]
        notebook["cells"].append(cell)

    out = tmp_path / "executed.ipynb"
    ran = subprocess.run(
        [str(VENV / "bin" / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
         f"--ExecutePreprocessor.kernel_name={KERNEL}", f"--ExecutePreprocessor.timeout={RUN_LIMIT_S}",
         "--output", out.stem, "--output-dir", str(out.parent), "--stdin"],
        env=env, cwd=NOTEBOOKS, input=json.dumps(notebook), capture_output=True, text=True,
        timeout=RUN_LIMIT_S + 60,
    )
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not _gemma_answers(), "the notebook left its Gemma server running"
    executed = json.loads(out.read_text(encoding="utf-8"))
    cells = executed["cells"]
    for cell in cells:
        if cell["cell_type"] == "code":
            assert not _errors(cell), "".join(cell["source"])

    start_cell = _code_after(cells, "## 1.")
    started = _printed(start_cell)
    for part in ("Gemma 4 server at", "Gemma 4 warmed up", "SAM 3 worker ready", "SmolVLA", "SAPIEN scene, layout seed 0",
                 "SAM 3 warmed up with one outline of both cameras", "SmolVLA with one plan", "utilization and power table"):
        assert part in started, started
    # The start says nothing on standard error: a notebook shows that as if something had gone wrong.
    # Until 2026-11-23 it held uv's line about its environment, two deprecation notices of libraries, the
    # simulator's warning about a CUDA device it was never going to get (twice) and progress bars.
    on_standard_error = "".join(
        "".join(output["text"]) for output in start_cell["outputs"]
        if output["output_type"] == "stream" and output["name"] == "stderr"
    )
    assert on_standard_error == "", on_standard_error
    # The weights' loading is said in plain lines: SmolVLA's language model, and SAM 3 once per camera.
    assert started.count("Loading weights: ") == 3, started
    live = _printed(_code_after(cells, "## 3."))
    assert "listening to the microphone" in live, live  # this machine has a microphone: it has to open
    # Section 4 run in the row of all cells: its button is shown, nothing is stopped.
    shown = _printed(_code_after(cells, "## 4."))
    assert "the demo keeps running" in shown and "stopped" not in shown, shown

    checked, commanded, pressed = (_printed(cell) for cell in cells[-3:])
    print(checked.strip())
    print(commanded.strip())
    # The warm-up worked: the slow first run of SAM 3 and of SmolVLA was the start's, not the first
    # command's. Each model's first run in a command took less than half of what its warm-up run took
    # (measured 2026-11-23: 0.4 s against 6.8 s for SAM 3's two pictures, 0.2 s against 1.3 s for the plan).
    warm = json.loads(commanded.split("WARMUP ", 1)[1])
    assert warm["command_ended"] and warm["cube"] == "red", warm
    assert warm["plans_of_the_command"] >= 1, warm
    assert 0.0 < warm["first_command_outline_s"] < 0.5 * warm["warm_up_outline_s"], warm
    assert 0.0 < warm["first_command_plan_s"] < 0.5 * warm["warm_up_plan_s"], warm
    microphone = json.loads(checked.split("MICROPHONE ", 1)[1])
    assert microphone["opened"] and microphone["error"] is None, microphone
    assert microphone["threads_alive"] == [True, True], microphone
    # Frames keep coming and reach the picture: the panel is live, not a first frame left standing.
    assert microphone["available"] and microphone["a_later_frame_arrived"], microphone
    assert microphone["draws_awaited"] == 10 and microphone["frames_drawn"] >= 10, microphone
    assert microphone["panel"] in ("listening", "hearing you", "asking Gemma..."), microphone
    assert microphone["overflows"] == 0, microphone
    assert microphone["gemma_answers"] and not microphone["stop_pressed"], microphone
    assert "table closed" in pressed and "Gemma, SAM 3 and the scene stopped." in pressed, pressed
