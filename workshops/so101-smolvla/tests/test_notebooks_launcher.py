"""notebooks/run_notebooks.sh: the kernel it registers and the Jupyter server it starts.

The tests run the launcher itself, with JUPYTER_DATA_DIR in a temporary directory, so the user's own kernels are
not touched. They need the voice workshop's Python (workshops/vvla-pipeline/.venv), as the launcher does.
"""
import json
import os
import re
import socket
import subprocess
import threading
import urllib.request
from pathlib import Path

WORKSHOP = Path(__file__).resolve().parents[1]
NOTEBOOKS = WORKSHOP / "notebooks"
LAUNCHER = NOTEBOOKS / "run_notebooks.sh"
VOICE_WORKSHOP = WORKSHOP.parent / "vvla-pipeline"
KERNEL = "so101-smolvla-notebooks"
# What a kernel is asked: which Python it is, one variable of scripts/ryzen_ai_env.sh, and where it started.
PROBE = ("import json, os, sys\n"
         "print(json.dumps({'prefix': sys.prefix, 'gfx': os.environ.get('HSA_OVERRIDE_GFX_VERSION'), "
         "'cwd': os.getcwd()}))\n")


def _environment(tmp_path) -> dict:
    """The caller's environment without the workshop's variable, and with Jupyter's data and settings in tmp_path."""
    env = dict(os.environ, JUPYTER_DATA_DIR=str(tmp_path / "jupyter"), JUPYTER_CONFIG_DIR=str(tmp_path / "config"))
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)  # the kernel has to set it itself
    return env


def test_kernel_starts_in_the_voice_workshop_environment(tmp_path):
    env = _environment(tmp_path)
    subprocess.run([str(LAUNCHER), "--kernel-only"], env=env, check=True, timeout=120)
    spec = json.loads((tmp_path / "jupyter" / "kernels" / KERNEL / "kernel.json").read_text(encoding="utf-8"))
    assert spec["language"] == "python"

    # The value the workshop's own scripts give, not a number typed here.
    expected = subprocess.run(
        ["bash", "-c", 'source "$1/.venv/bin/activate" && source "$1/scripts/ryzen_ai_env.sh" >&2 && '
                       'printf %s "$HSA_OVERRIDE_GFX_VERSION"', "probe", str(VOICE_WORKSHOP)],
        env=env, check=True, capture_output=True, text=True, timeout=120).stdout
    assert expected

    probe = tmp_path / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    # "jupyter run" starts the kernel from its registered start command, as a notebook front end does.
    ran = subprocess.run([str(VOICE_WORKSHOP / ".venv" / "bin" / "jupyter"), "run", "--kernel", KERNEL, str(probe)],
                         env=env, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=120)
    got = json.loads(ran.stdout.strip().splitlines()[-1])
    assert Path(got["prefix"]).resolve() == (VOICE_WORKSHOP / ".venv").resolve()
    assert got["gfx"] == expected
    assert Path(got["cwd"]).resolve() == tmp_path.resolve()  # the start command does not change directory


def test_every_notebook_names_the_kernel():
    notebooks = sorted(NOTEBOOKS.glob("*.ipynb"))
    assert notebooks
    for path in notebooks:
        metadata = json.loads(path.read_text(encoding="utf-8"))["metadata"]
        assert metadata["kernelspec"]["name"] == KERNEL, path.name


def test_launcher_serves_the_notebooks_directory(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = subprocess.Popen([str(LAUNCHER), "--no-browser", "--port", str(port)], env=_environment(tmp_path),
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The address line is awaited on the server's own output. If it does not come within 60 s the server is
    # ended, its output ends, and the test fails on the missing address.
    limit = threading.Timer(60, server.terminate)
    limit.start()
    address, lines = None, []
    try:
        for line in server.stdout:
            lines.append(line)
            address = re.search(r"(http://127\.0\.0\.1:%d)/lab\?token=(\w+)" % port, line)
            if address:
                break
        assert address, "no address line from JupyterLab:\n" + "".join(lines)
        base, token = address.groups()

        def get(path):
            with urllib.request.urlopen(f"{base}{path}?token={token}", timeout=30) as response:
                return json.load(response)

        served = {item["name"] for item in get("/api/contents")["content"]}
        assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} <= served
        kernels = get("/api/kernelspecs")
        assert KERNEL in kernels["kernelspecs"]
        assert kernels["default"] == KERNEL

        # JupyterLab saves its layout (open tabs, the file browser's folder) the way the browser does here. It has
        # to land in this launcher's own directory: the default one is shared by every JupyterLab of the user, and
        # a layout of another project names notebooks that do not exist in this directory.
        save = urllib.request.Request(
            f"{base}/lab/api/workspaces/default", method="PUT",
            data=json.dumps({"data": {}, "metadata": {"id": "default"}}).encode("utf-8"),
            headers={"Authorization": f"token {token}", "Content-Type": "application/json"})
        with urllib.request.urlopen(save, timeout=30) as response:
            assert response.status == 204
        lab = tmp_path / "config" / "lab"
        assert list((lab / f"workspaces-{KERNEL}").glob("*.jupyterlab-workspace"))
        assert not list((lab / "workspaces").glob("*.jupyterlab-workspace"))
    finally:
        limit.cancel()
        server.terminate()
        server.wait(timeout=30)
