"""notebooks/voice_control_test.ipynb, run the way a person runs it: its registered kernel and the real Gemma server.

Two runs, both with `jupyter nbconvert --execute` of the voice workshop's Python:

1. The notebook as it lies in git. nbconvert leaves out the cells tagged "skip-execution", the two that need a
   person at the microphone. Every other cell has to run without an error, and section 5 has to report every
   typed sentence right. The count for the synthetic voice is printed (pytest -rP shows it) and not required:
   the user's decision of 2026-11-21, because the synthetic voice gave two different sounds for one sentence.
2. A short notebook made of the notebook's own cells: setup, start of Gemma, a second start of the server (has to
   be refused), a cell that sets MIC to a microphone that does not exist, the live-session cell (has to stop
   with its error), stop of Gemma.

Needed on the machine: the set-up voice workshop (workshops/vvla-pipeline: Python environment, model files,
llama-server), its graphics processor, espeak-ng, and nothing answering on the Gemma port. A missing piece fails
the tests; nothing is skipped. Jupyter's data and settings go to a temporary directory, as in the launcher's
tests, so the user's own kernels are not touched. The server's log is notebooks/logs/llama-server.log.
"""
import copy
import json
import re
import subprocess
import urllib.error
import urllib.request
import uuid

import pytest
import yaml

from test_notebooks_launcher import KERNEL, LAUNCHER, NOTEBOOKS, VOICE_WORKSHOP, WORKSHOP, _environment

NOTEBOOK = NOTEBOOKS / "voice_control_test.ipynb"
CLIENT_CONFIG = WORKSHOP / "voice" / "config" / "pipeline.yaml"
SKIP_TAG = "skip-execution"
# Ends a run that hangs. The whole notebook took 66 s when it was first run (2026-11-21, 60 requests to Gemma).
RUN_LIMIT_S = 600


def _gemma_answers() -> bool:
    """Whether a server answers on the port the client's own settings name."""
    section = yaml.safe_load(CLIENT_CONFIG.read_text(encoding="utf-8"))["gemma4-e4b-it"]
    try:
        urllib.request.urlopen(f"http://{section['host']}:{section['port']}/health", timeout=5).close()
    except urllib.error.HTTPError:
        return True  # it answers, with an error status
    except OSError:
        return False
    return True


@pytest.fixture(scope="module")
def jupyter(tmp_path_factory) -> dict:
    """The environment of the runs, with the notebooks' kernel registered by the launcher itself."""
    env = _environment(tmp_path_factory.mktemp("jupyter"))
    subprocess.run([str(LAUNCHER), "--kernel-only"], env=env, check=True, timeout=120)
    return env


def _execute(env: dict, out, source=None, allow_errors: bool = False) -> dict:
    """Run a notebook in the notebooks directory and return it as executed.

    source: the notebook's JSON text, handed to nbconvert on its standard input; without it, the notebook file."""
    assert not _gemma_answers(), "something already answers on the Gemma port; stop it before this test"
    command = [str(VOICE_WORKSHOP / ".venv" / "bin" / "jupyter"), "nbconvert", "--to", "notebook", "--execute",
               f"--ExecutePreprocessor.kernel_name={KERNEL}", "--output", out.stem, "--output-dir", str(out.parent)]
    command += ["--allow-errors"] if allow_errors else []
    command += ["--stdin"] if source is not None else [str(NOTEBOOK)]
    ran = subprocess.run(command, env=env, cwd=NOTEBOOKS, input=source, capture_output=True, text=True,
                         timeout=RUN_LIMIT_S)
    assert ran.returncode == 0, ran.stdout + ran.stderr
    assert not _gemma_answers(), "the notebook left its Gemma server running"
    return json.loads(out.read_text(encoding="utf-8"))


def _code_after(cells: list, heading: str) -> dict:
    """The first code cell after the markdown cell whose text starts with the heading."""
    for n, cell in enumerate(cells):
        if cell["cell_type"] == "markdown" and "".join(cell["source"]).startswith(heading):
            return next(c for c in cells[n + 1:] if c["cell_type"] == "code")
    raise AssertionError(f"no section {heading!r} in the notebook")


def _printed(cell: dict) -> str:
    return "".join("".join(o["text"]) for o in cell["outputs"] if o["output_type"] == "stream")


def _errors(cell: dict) -> list:
    return [(o["ename"], o["evalue"]) for o in cell["outputs"] if o["output_type"] == "error"]


def test_notebook_runs_and_every_typed_sentence_gives_its_call(jupyter, tmp_path):
    executed = _execute(jupyter, tmp_path / "executed.ipynb")
    code = [c for c in executed["cells"] if c["cell_type"] == "code"]
    left_out = [c for c in code if SKIP_TAG in c["metadata"].get("tags", [])]
    assert len(left_out) == 2  # sections 6 and 7, which need a person
    for cell in code:
        assert not _errors(cell), "".join(cell["source"])
        if cell in left_out:
            assert cell["execution_count"] is None and not cell["outputs"]
        else:
            assert cell["execution_count"] is not None

    table = _printed(_code_after(executed["cells"], "## 5."))
    typed = re.search(r"^typed: (\d+) of (\d+) right", table, flags=re.M)
    spoken = re.search(r"^spoken, synthetic voice: (\d+) of (\d+) right.*$", table, flags=re.M)
    assert typed and spoken, table
    print(typed.group(0))
    print(spoken.group(0))
    assert int(typed.group(2)) >= 30  # the table of section 1
    assert typed.group(1) == typed.group(2), table
    assert _printed(_code_after(executed["cells"], "## 8.")) == "Gemma stopped.\n"


def test_live_cell_stops_with_an_error_without_a_microphone_and_a_second_server_is_refused(jupyter, tmp_path):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    setup = next(c for c in cells if c["cell_type"] == "code")
    start, live, stop = (_code_after(cells, heading) for heading in ("## 2.", "## 7.", "## 8."))

    def added(source: str) -> dict:
        cell = copy.deepcopy(stop)
        cell.update(source=source, metadata={}, outputs=[], execution_count=None)
        if "id" in cell:
            cell["id"] = uuid.uuid4().hex[:8]
        return cell

    live = copy.deepcopy(live)
    live["metadata"] = {}  # without the tag that leaves it out
    second_start = added("t.start_server(LOG_DIR, {}, [])")
    no_microphone = added("MIC = 'no-such-microphone'")  # what a user does to choose another input
    notebook["cells"] = [setup, start, second_start, no_microphone, live, stop]

    executed = _execute(jupyter, tmp_path / "executed.ipynb", source=json.dumps(notebook), allow_errors=True)
    ran_setup, ran_start, ran_second, ran_mic, ran_live, ran_stop = executed["cells"]
    for cell in (ran_setup, ran_start, ran_mic, ran_stop):
        assert not _errors(cell), "".join(cell["source"])
    (name, message), = _errors(ran_second)
    assert name == "RuntimeError" and message.startswith("SETUP FAILURE: something already answers")
    (name, message), = _errors(ran_live)
    assert name == "RuntimeError"
    assert message.startswith("The microphone or the speech detection could not be opened"), message
    assert "Listening" not in _printed(ran_live)
    assert _printed(ran_stop) == "Gemma stopped.\n"
