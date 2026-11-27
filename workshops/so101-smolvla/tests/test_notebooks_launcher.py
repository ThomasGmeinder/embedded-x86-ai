"""notebooks/run_notebooks.sh: the kernel it registers and the Jupyter server it starts.

The tests run the launcher itself, with JUPYTER_DATA_DIR in a temporary directory, so the user's own kernels are
not touched. They need ipykernel and JupyterLab in the workshop's .venv (scripts/setup_voice.sh), as the launcher
does.
"""
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import urllib.request
from pathlib import Path

from so101_smolvla.notebook_server import LISTENING
from so101_smolvla.sapien_config import SAPIEN_COMMIT, SAPIEN_PYTHON, SAPIEN_SOURCE

WORKSHOP =Path(__file__).resolve().parents[1]
NOTEBOOKS = WORKSHOP / "notebooks"
LAUNCHER = NOTEBOOKS / "run_notebooks.sh"
VENV = WORKSHOP / ".venv"
KERNEL = "so101-smolvla-notebooks"
# What a kernel is asked: which Python it is, whether it carries the server's graphics override (it must not:
# that variable is the llama-server process's alone), whether the package imports, and where it started.
PROBE = ("import json, os, sys\n"
         "import so101_smolvla.voice.settings\n"
         "print(json.dumps({'prefix': sys.prefix, 'gfx': os.environ.get('HSA_OVERRIDE_GFX_VERSION'), "
         "'cwd': os.getcwd()}))\n")


def _environment(tmp_path) -> dict:
    """The caller's environment without the server's variable, and with Jupyter's data and settings in tmp_path."""
    env = dict(os.environ, JUPYTER_DATA_DIR=str(tmp_path / "jupyter"), JUPYTER_CONFIG_DIR=str(tmp_path / "config"))
    env.pop("HSA_OVERRIDE_GFX_VERSION", None)
    return env


def test_kernel_starts_in_the_workshops_environment_with_its_package(tmp_path):
    env = _environment(tmp_path)
    subprocess.run([str(LAUNCHER), "--kernel-only"], env=env, check=True, timeout=120)
    spec = json.loads((tmp_path / "jupyter" / "kernels" / KERNEL / "kernel.json").read_text(encoding="utf-8"))
    assert spec["language"] == "python"
    assert Path(spec["argv"][0]).resolve() == (VENV / "bin" / "python").resolve()
    assert Path(spec["env"]["PYTHONPATH"]).resolve() == (WORKSHOP / "src").resolve()

    probe = tmp_path / "probe.py"
    probe.write_text(PROBE, encoding="utf-8")
    # "jupyter run" starts the kernel from its registered start command, as a notebook front end does.
    ran = subprocess.run([str(VENV / "bin" / "jupyter"), "run", "--kernel", KERNEL, str(probe)],
                         env=env, cwd=tmp_path, check=True, capture_output=True, text=True, timeout=120)
    got = json.loads(ran.stdout.strip().splitlines()[-1])
    assert Path(got["prefix"]).resolve() == VENV.resolve()
    assert got["gfx"] is None
    assert Path(got["cwd"]).resolve() == tmp_path.resolve()  # the start command does not change directory


def test_launcher_restores_the_sapien_twin_a_restart_removed(tmp_path):
    """The simulator's checkout lies under /tmp, which a restart empties; the launcher clones and installs it again.

    The checkout is renamed for the test, which is what a restart leaves behind (no directory, .venv-sapien pointing
    at it), and put back afterwards. The launcher's clone needs the network, as the setup script does.
    """
    assert (SAPIEN_SOURCE / ".git").is_dir(), "the SAPIEN twin is not set up (scripts/setup_sapien_env.sh)"
    aside = SAPIEN_SOURCE.with_name(SAPIEN_SOURCE.name + ".test-aside")
    assert not aside.exists(), f"{aside} is left from an earlier run; move it back to {SAPIEN_SOURCE}"
    SAPIEN_SOURCE.rename(aside)
    try:
        subprocess.run([str(LAUNCHER), "--kernel-only"], env=_environment(tmp_path), check=True, timeout=300)
        head = subprocess.run(["git", "-C", str(SAPIEN_SOURCE), "rev-parse", "HEAD"],
                              check=True, capture_output=True, text=True).stdout.strip()
        assert head == SAPIEN_COMMIT
        imported = subprocess.run([str(SAPIEN_PYTHON), "-c", "import so101_sim; print(so101_sim.__file__)"],
                                  check=True, capture_output=True, text=True, timeout=120).stdout.strip()
        assert Path(imported).is_relative_to(SAPIEN_SOURCE)
        assert (tmp_path / "jupyter" / "kernels" / KERNEL / "kernel.json").is_file()  # and it went on to the kernel
    finally:
        # The checkout that was there before the test comes back, whatever the launcher did.
        if SAPIEN_SOURCE.exists():
            shutil.rmtree(SAPIEN_SOURCE)
        aside.rename(SAPIEN_SOURCE)


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
    # Two lines are awaited on the server's own output: its address, and then the line of the workshop's server
    # extension that says it listens. JupyterLab prints the address before it listens, so the address alone is
    # not the moment to connect. If the lines do not come within 60 s the server is ended, its output ends, and
    # the test fails on the missing line.
    limit = threading.Timer(60, server.terminate)
    limit.start()
    address, listens, lines = None, False, []
    try:
        for line in server.stdout:
            lines.append(line)
            address = address or re.search(r"(http://127\.0\.0\.1:%d)/lab\?token=(\w+)" % port, line)
            if f"{LISTENING} {port}" in line:
                listens = True
                break
        assert address, "no address line from JupyterLab:\n" + "".join(lines)
        assert listens, f"JupyterLab did not say {LISTENING!r}:\n" + "".join(lines)
        base, token = address.groups()
        # Keep reading the server's output, so a server that ends after its address line explains itself.
        threading.Thread(target=lambda: lines.extend(server.stdout), daemon=True).start()

        def get(path):
            try:
                with urllib.request.urlopen(f"{base}{path}?token={token}", timeout=30) as response:
                    return json.load(response)
            except OSError as exc:
                raise AssertionError(
                    f"{path}: {exc}; JupyterLab's output so far (exit status {server.poll()}):\n" + "".join(lines)
                ) from exc

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
