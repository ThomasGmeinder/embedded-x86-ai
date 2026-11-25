# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The voice server is started on every core, whatever its starter is pinned to.

``cores`` in ``config/pipeline.yaml`` pins the perception process to one core,
and perception starts ``llama-server``. A child inherits its parent's allowed
cores: the server was seen with 38 of its 40 threads allowed on that one core
only (2026-11-25), sharing it with the cameras and the vision models, while the
config said it is not pinned.

A stand-in server reports the cores it may use; the parser that starts it runs
in a process pinned to one core, as perception is. No model and no GPU.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.timeout(60)]

# Stands in for llama-server: answers /health, says which cores it may use, then the line the
# parser waits for. It serves until it is terminated.
STAND_IN = '''#!{python}
import os, sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *_):
        pass


server = HTTPServer(("127.0.0.1", int(sys.argv[sys.argv.index("--port") + 1])), Handler)
print("allowed cores:", len(os.sched_getaffinity(0)), flush=True)
print("listening on http://127.0.0.1", flush=True)
server.serve_forever()
'''

# Runs pinned to one core: starts the parser (and with it the stand-in server), then closes it.
STARTER = '''
import os, sys
from vla_pipeline.llm.llama_intent import LlamaIntentParser
from vla_pipeline.utils.config import load_config

server_bin, model, port, log_dir = sys.argv[1:5]
assert len(os.sched_getaffinity(0)) == 1, os.sched_getaffinity(0)
cfg = load_config()
cfg["gemma4-e4b-it"].update(server_bin=server_bin, model_gguf=model, mmproj=None, host="127.0.0.1", port=int(port))
cfg.setdefault("system", {{}})["log_dir"] = log_dir
LlamaIntentParser(cfg).close()
'''


@pytest.fixture
def program_dir():
    """A folder a program can be started from: pytest's temp folder may be mounted noexec."""
    import shutil

    from vla_pipeline.utils.config import REPO_ROOT

    folder = REPO_ROOT / "_artifacts" / f"stand-in-server-{os.getpid()}"
    folder.mkdir(parents=True, exist_ok=True)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


@pytest.mark.skipif(len(os.sched_getaffinity(0)) < 2, reason="this machine gives the test one core")
def test_a_pinned_starter_gets_a_server_that_may_use_every_core(tmp_path, program_dir):
    stand_in = program_dir / "llama-server"
    stand_in.write_text(STAND_IN.format(python=sys.executable))
    stand_in.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    one_core = str(min(os.sched_getaffinity(0)))
    run = subprocess.run(
        ["taskset", "-c", one_core, sys.executable, "-c", STARTER.format(), str(stand_in), str(model), str(port),
         str(tmp_path)],
        capture_output=True, text=True, timeout=40,
    )
    assert run.returncode == 0, run.stdout + run.stderr
    log = (tmp_path / "llama-server.log").read_text()
    assert f"allowed cores: {os.cpu_count()}" in log, log


def test_a_server_that_cannot_start_is_reported_with_its_exit_and_its_log(tmp_path, program_dir):
    """A server that ends before it is ready: the error names its exit code and shows its output.
    (Its output ending is not readiness; the start waits for the exit, not for a second.)"""
    from vla_pipeline.llm.llama_intent import LlamaIntentParser
    from vla_pipeline.utils.config import load_config

    broken = program_dir / "llama-server"
    broken.write_text(f"#!{sys.executable}\nimport sys\nprint('no such model', flush=True)\nsys.exit(3)\n")
    broken.chmod(0o755)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    cfg = load_config()
    cfg["gemma4-e4b-it"].update(server_bin=str(broken), model_gguf=str(model), mmproj=None, host="127.0.0.1", port=port)
    cfg.setdefault("system", {})["log_dir"] = str(tmp_path)
    with pytest.raises(RuntimeError, match=r"exited with code 3 during startup(.|\n)*no such model"):
        LlamaIntentParser(cfg)
