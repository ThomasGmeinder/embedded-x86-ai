"""open_meters() and close_meters() through the real window program on the real display of this machine."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time

import pytest

from so101_smolvla.utilization_meters import (
    LAUNCHER,
    LOCK_NAME,
    close_meters,
    open_meters,
    running_facts,
    runtime_dir,
)
from utilization_machines import WORKSHOP, this_machine

MACHINE = this_machine()
pytestmark = pytest.mark.skipif(
    MACHINE is None, reason="this machine is not in tests/utilization_machines.py"
)


def _lock_is_free() -> bool:
    fd = os.open(runtime_dir() / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    return True
    # noqa: the descriptor closes with the process


def _xwininfo(window: int) -> dict:
    out = subprocess.run(["xwininfo", "-id", str(window)], capture_output=True, text=True, check=True).stdout
    return dict(line.strip().split(":", 1) for line in out.splitlines() if ":" in line and not line.startswith("xwininfo"))


@pytest.fixture(autouse=True)
def nothing_running():
    assert MACHINE["has_display"] and os.environ.get("DISPLAY"), "this machine has a display; DISPLAY is not set"
    assert running_facts() is None, "a table is running before the test; close it first"
    yield
    close_meters()
    assert _lock_is_free()


def test_open_meters_shows_a_window_outside_the_window_manager_and_a_second_call_returns_it():
    focus_before = subprocess.run(["xdotool", "getwindowfocus"], capture_output=True, text=True).stdout.strip()
    facts = open_meters()
    assert facts["host"] == os.uname().nodename and facts["display"] == os.environ["DISPLAY"]
    assert facts["renderer"] in ("OpenGL", "Software", "Vulkan")
    info = _xwininfo(facts["window"])
    assert info["Map State"].strip() == "IsViewable"
    assert info["Override Redirect State"].strip() == "yes"
    assert info["Depth"].strip() == "32"
    focus_after = subprocess.run(["xdotool", "getwindowfocus"], capture_output=True, text=True).stdout.strip()
    assert focus_after == focus_before, "the table took the keyboard focus"
    assert not _lock_is_free()
    print(facts, info["Absolute upper-left X"], info["Absolute upper-left Y"])

    again = open_meters()
    assert again == facts, "a second open_meters() started another program"
    assert running_facts()["pid"] == facts["pid"]

    assert close_meters() is True
    assert running_facts() is None and _lock_is_free()
    assert close_meters() is False
    assert subprocess.run(["xwininfo", "-id", str(facts["window"])], capture_output=True).returncode != 0


def test_close_meters_from_another_process_ends_the_window():
    facts = open_meters()
    ran = subprocess.run(
        [str(WORKSHOP / ".venv" / "bin" / "python"), "-c",
         "from so101_smolvla.utilization_meters import close_meters; print(close_meters())"],
        cwd=WORKSHOP, env={**os.environ, "PYTHONPATH": str(WORKSHOP / "src")},
        capture_output=True, text=True, timeout=60,
    )
    assert ran.returncode == 0 and ran.stdout.strip() == "True", ran.stderr
    assert running_facts() is None and _lock_is_free()
    assert subprocess.run(["xwininfo", "-id", str(facts["window"])], capture_output=True).returncode != 0


def test_without_a_display_open_meters_raises_and_leaves_nothing_behind(monkeypatch):
    monkeypatch.delenv("DISPLAY")
    with pytest.raises(RuntimeError, match="no DISPLAY"):
        open_meters()
    assert running_facts() is None and _lock_is_free()


def test_the_launcher_runs_the_window_program_in_the_workshops_python():
    assert LAUNCHER.is_file() and os.access(LAUNCHER, os.X_OK)
    text = LAUNCHER.read_text(encoding="utf-8")
    assert ".venv/bin/python -m so101_smolvla.utilization_window" in text
    assert runtime_dir().is_dir()
