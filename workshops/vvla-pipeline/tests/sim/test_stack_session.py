# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The stack session stops its processes on the events that end a run.

A stand-in console exiting is the OpenCV window closing. A byte on the
session's pipe is Ctrl-C. Each launch is signaled on its own process, and a
child in that launch's group does not get the signal.
"""

from __future__ import annotations

import os
import select
import signal
import sys
import threading
from pathlib import Path

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.timeout(30)]

PAUSE = """
import os, signal, sys
log, role, ready = sys.argv[1], sys.argv[2], int(sys.argv[3])
child = None
if role == "launch":
    if os.getpgrp() != os.getpid():
        os.setsid()
    child = os.fork()
    if child == 0:
        def hit(signum, _frame):
            with open(log, "a") as fh:
                fh.write(f"child {signum}\\n")
            raise SystemExit(0)
        signal.signal(signal.SIGINT, hit)
        signal.signal(signal.SIGTERM, hit)
        signal.pause()
def stop(signum, _frame):
    with open(log, "a") as fh:
        fh.write(f"{role} {signum}\\n")
    raise SystemExit(0)
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)
if child:
    with open(log + ".child", "w") as fh:
        fh.write(str(child))
os.write(ready, b"\\0")
signal.pause()
"""


def _argv(log: Path, role: str, ready: int) -> list[str]:
    return [sys.executable, "-c", PAUSE, str(log), role, str(ready)]


def _start(spawn, log: Path, role: str):
    """``spawn(argv, pass_fds=)`` and block until that process is waiting for a signal."""
    read, write = os.pipe()
    os.set_inheritable(write, True)
    try:
        proc = spawn(_argv(log, role, write), pass_fds=(write,))
    finally:
        os.close(write)
    select.select([read], [], [])
    os.close(read)
    return proc


def _reap_child(log: Path) -> None:
    path = Path(str(log) + ".child")
    if not path.exists():
        return
    pid = int(path.read_text())
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def test_a_console_exit_stops_control_then_the_launch(tmp_path):
    """The OpenCV window closing is the console process exiting."""
    from so101_sim.stack import Session

    log = tmp_path / "events.log"
    session = Session()
    _start(session.add_control, log, "control")
    _start(session.add_launch, log, "launch")
    session.console = session.add_side([sys.executable, "-c", "raise SystemExit(0)"])
    try:
        session.wait()
        session.stop()
        text = log.read_text()
        assert text.index(f"control {signal.SIGTERM}") < text.index(f"launch {signal.SIGINT}")
        assert "child" not in text
    finally:
        session.stop()
        _reap_child(log)


def test_a_signal_stops_the_side_before_the_launch(tmp_path):
    """Ctrl-C wakes the session. Perception stops before the launch."""
    from so101_sim.stack import Session

    log = tmp_path / "events.log"
    session = Session()
    _start(session.add_side, log, "perception")
    _start(session.add_control, log, "control")
    _start(session.add_launch, log, "launch")
    try:
        box: dict = {}

        def run() -> None:
            session.wait()
            box["woke"] = True

        thread = threading.Thread(target=run)
        thread.start()
        session.notify()
        thread.join(5)
        assert box.get("woke") is True
        session.stop()
        text = log.read_text()
        assert text.index(f"perception {signal.SIGTERM}") < text.index(f"control {signal.SIGTERM}")
        assert text.index(f"control {signal.SIGTERM}") < text.index(f"launch {signal.SIGINT}")
        assert "child" not in text
    finally:
        session.stop()
        _reap_child(log)


def test_child_env_drops_the_localhost_profile(monkeypatch):
    monkeypatch.setenv("FASTRTPS_DEFAULT_PROFILES_FILE", "/tmp/twin_localhost.xml")
    from so101_sim.stack import child_env

    assert "FASTRTPS_DEFAULT_PROFILES_FILE" not in child_env()
