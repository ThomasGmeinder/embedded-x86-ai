# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Ctrl-C in run_notebooks.sh releases the webcam.

workshop/notebook_session.sh is the stop. This drives that file the way the
terminal drives the script: SIGINT to the script's process group while it is
waiting on the notebook server. The stand-in server ignores SIGINT and exits
on SIGTERM, which is Jupyter's shutdown signal. The stand-in camera leads its
own session, as so101_sim.stack does, and exits on SIGINT.
"""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SESSION = ROOT / "workshop" / "notebook_session.sh"
LAUNCHER = ROOT / "workshop" / "run_notebooks.sh"

pytestmark = [pytest.mark.sim, pytest.mark.timeout(30)]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _reap(path: Path) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    pid = int(text)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_the_launcher_uses_the_session_stop():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'source "${WORKSHOP_DIR}/notebook_session.sh"' in text
    assert "trap stop_session EXIT INT TERM" in text
    assert 'run_notebook_server "$PY" "$LAUNCHER" "$JUPYTER_PORT"' in text
    # The pipeline starts after the --test exit, the same place the gemma4
    # branch started the twin: the docs tests own domain 93.
    test_exit = text.index("exit 0\nfi\n\n# --- 3.")
    pipeline = text.index('setsid "${REPO_ROOT}/run_pipeline.sh" --digital-twin')
    assert test_exit < pipeline
    session = SESSION.read_text(encoding="utf-8")
    assert 'kill -TERM "$PIPELINE_PID"' in session
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    subprocess.run(["bash", "-n", str(SESSION)], check=True)


def test_an_attached_session_does_not_own_a_launch():
    from vla_pipeline.single_instance import running_pipeline
    from vvla_workshop.twin_ros import TwinSession

    session = TwinSession.__new__(TwinSession)
    session._proc = None
    assert session.pid is None
    assert session._exited() is None
    if running_pipeline():
        return
    with pytest.raises(RuntimeError, match="run_pipeline.sh --digital-twin"):
        TwinSession.attach({})


def test_ctrl_c_stops_the_server_and_the_camera_launch(tmp_path):
    """SIGINT to the waiting shell SIGTERMs the server and SIGINTs the camera."""
    ready = tmp_path / "ready"
    os.mkfifo(ready)
    server_log = tmp_path / "server.log"
    camera_log = tmp_path / "camera.log"
    server_pid = tmp_path / "server.pid"
    camera_pid = tmp_path / "camera.pid"
    # /tmp is mounted noexec, so the stand-in Jupyter executable cannot live
    # in tmp_path. run_notebook_server execs it.
    stub_dir = Path.home() / ".cache" / "vvla-notebook-session-test"
    stub_dir.mkdir(parents=True, exist_ok=True)
    server_py = stub_dir / f"server-{os.getpid()}.py"
    server_bin = stub_dir / f"server-{os.getpid()}.sh"
    server_py.write_text(
        "import os, signal, sys\n"
        "log, ready = sys.argv[1], sys.argv[2]\n"
        "def term(*_):\n"
        "    open(log, 'a').write('term\\n')\n"
        "    raise SystemExit(0)\n"
        "signal.signal(signal.SIGTERM, term)\n"
        "os.write(os.open(ready, os.O_WRONLY), b'server\\n')\n"
        "signal.pause()\n",
        encoding="utf-8",
    )
    server_bin.write_text(
        f"#!/bin/bash\nexec python3 {server_py} {server_log} {ready}\n",
        encoding="utf-8",
    )
    server_bin.chmod(0o755)
    (tmp_path / "camera.py").write_text(
        f"""import ctypes, os, signal, sys
log, ready = sys.argv[1], sys.argv[2]
# A background job inherits SIGINT ignored. The real stack clears that
# before it becomes the launch, or the group's SIGINT would do nothing.
ctypes.CDLL(None).signal(signal.SIGINT, 0)
def stop(*_):
    open(log, "a").write("int\\n")
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
os.write(os.open(ready, os.O_WRONLY), b"camera\\n")
signal.pause()
""",
        encoding="utf-8",
    )
    (tmp_path / "arm.py").write_text(
        """import os, signal, sys
ready, leader = sys.argv[1], int(sys.argv[2])
fd = os.open(ready, os.O_RDONLY)
buf = b""
while buf.count(b"\\n") < 2:
    chunk = os.read(fd, 64)
    if not chunk:
        raise SystemExit("the server and the camera did not both become ready")
    buf += chunk
os.close(fd)
os.killpg(leader, signal.SIGINT)
""",
        encoding="utf-8",
    )
    script = tmp_path / "drive.sh"
    script.write_text(
        f"""#!/bin/bash
set -euo pipefail
source {SESSION}
JUPYTER_PID=""
LAUNCH_PIDS=()
MONITOR_PID=""
trap stop_session EXIT INT TERM
# The arm sits in its own session so the SIGINT it sends does not hit it.
setsid python3 {tmp_path / "arm.py"} {ready} $$ &
setsid python3 {tmp_path / "camera.py"} {camera_log} {ready} &
LAUNCH_PIDS+=("$!")
printf '%s\\n' "${{LAUNCH_PIDS[0]}}" > {camera_pid}
run_notebook_server {server_bin} lab 9
printf '%s\\n' "$JUPYTER_PID" > {server_pid}
echo done
""",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        ["bash", str(script)],
        cwd=tmp_path,
        start_new_session=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=15)
        assert proc.returncode == 0, stderr
        assert stdout.strip() == "done"
        assert server_log.read_text(encoding="utf-8") == "term\n"
        assert camera_log.read_text(encoding="utf-8") == "int\n"
        assert not _alive(int(server_pid.read_text(encoding="utf-8")))
        assert not _alive(int(camera_pid.read_text(encoding="utf-8")))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        raise
    finally:
        _reap(camera_pid)
        _reap(server_pid)
        server_bin.unlink(missing_ok=True)
        server_py.unlink(missing_ok=True)
