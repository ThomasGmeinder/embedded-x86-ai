# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Cells run while a ``common.ui_poll.ui_events`` cell loops, in a real kernel.

A loop cell like the twin console's (``poll(10)`` at 15 Hz until
``finished``) runs in a ``python3`` kernel. The test queues other cells the
way JupyterLab does while it loops, presses Jupyter's stop button (SIGINT
to the kernel's process group, as the server sends it), and reads each
cell's reply and output.
"""

from __future__ import annotations

import threading
import time

import pytest

from notebook_harness import PIPE_ROOT

pytestmark = pytest.mark.sim
pytest.importorskip("jupyter_ui_poll")
pytest.importorskip("jupyter_client")

SETUP = f"""
import sys, threading, time
sys.path.insert(0, {str(PIPE_ROOT / "workshop" / "solution")!r})
from common.ui_poll import ui_events
"""

LOOP = """
finished = threading.Event()
{end}
print("loop started")
try:
    with ui_events(on_rerun=finished.set) as poll:
        while not finished.is_set():
            poll(10)
            time.sleep(1 / 15)
finally:
    finished.set()
    print("loop ended")
"""
UNTIL_STOPPED = LOOP.format(end="")
FOR_2S = LOOP.format(end="threading.Timer(2.0, finished.set).start()")


class Kernel:
    def __init__(self):
        from jupyter_client.manager import KernelManager

        self.km = KernelManager(kernel_name="python3")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        self.kc.wait_for_ready(timeout=60)
        self._names: dict[str, str] = {}
        self._lock = threading.Lock()
        self.replies: dict[str, dict] = {}
        self.started: set[str] = set()
        self.stdout: dict[str, str] = {}
        self._running = True
        self._threads = [threading.Thread(target=f, daemon=True) for f in (self._iopub, self._shell)]
        for t in self._threads:
            t.start()
        self.run("setup", SETUP)
        self.wait_reply("setup")

    def _name(self, msg) -> str:
        return self._names.get(msg["parent_header"].get("msg_id"), "?")

    def _iopub(self):
        while self._running:
            try:
                msg = self.kc.get_iopub_msg(timeout=0.2)
            except Exception:
                continue
            with self._lock:
                name = self._name(msg)
                if msg["msg_type"] == "execute_input":
                    self.started.add(name)
                elif msg["msg_type"] == "stream":
                    self.stdout[name] = self.stdout.get(name, "") + msg["content"]["text"]

    def _shell(self):
        while self._running:
            try:
                msg = self.kc.get_shell_msg(timeout=0.2)
            except Exception:
                continue
            with self._lock:
                self.replies[self._name(msg)] = msg["content"]

    def run(self, name: str, code: str) -> None:
        with self._lock:
            self._names[self.kc.execute(code)] = name

    def until(self, predicate, timeout: float, what: str):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if predicate():
                    return
            time.sleep(0.05)
        raise AssertionError(f"timed out after {timeout:.0f} s waiting for {what}")

    def wait_reply(self, name: str, timeout: float = 15.0) -> dict:
        self.until(lambda: name in self.replies, timeout, f"the reply to {name}")
        return self.replies[name]

    def status(self, name: str, timeout: float = 15.0) -> str:
        return self.wait_reply(name, timeout)["status"]

    def stop_button(self) -> None:
        self.km.interrupt_kernel()

    def close(self) -> None:
        self._running = False
        for t in self._threads:
            t.join(timeout=1.0)
        self.kc.stop_channels()
        self.km.shutdown_kernel(now=True)


@pytest.fixture
def kernel():
    k = Kernel()
    yield k
    k.close()


def _start_loop(kernel, code):
    kernel.run("loop", code)
    kernel.until(lambda: "loop started" in kernel.stdout.get("loop", ""), 10.0, "the loop cell")
    time.sleep(0.5)


def test_held_back_cell_runs_after_the_loop_ends(kernel):
    _start_loop(kernel, FOR_2S)
    kernel.run("after", 'print("after ran")')
    assert kernel.status("loop") == "ok"
    assert kernel.status("after") == "ok"
    assert kernel.stdout.get("after") == "after ran\n"


def test_stop_button_cancels_held_back_cells(kernel):
    _start_loop(kernel, UNTIL_STOPPED)
    kernel.run("after", 'print("after ran")')
    time.sleep(0.5)
    kernel.stop_button()
    loop = kernel.wait_reply("loop")
    assert loop["status"] == "error" and loop["ename"] == "KeyboardInterrupt"
    assert "loop ended" in kernel.stdout["loop"]
    assert kernel.status("after") == "aborted"
    kernel.run("next", 'print("next ran")')
    assert kernel.status("next") == "ok"


def test_stop_button_interrupts_a_held_back_cell(kernel):
    _start_loop(kernel, FOR_2S)
    kernel.run("slow", "time.sleep(60)")
    kernel.until(lambda: "slow" in kernel.started, 15.0, "the held-back cell to start")
    time.sleep(0.5)
    kernel.stop_button()
    slow = kernel.wait_reply("slow", timeout=5.0)
    assert slow["status"] == "error" and slow["ename"] == "KeyboardInterrupt", slow
    kernel.run("next", 'print("next ran")')
    assert kernel.status("next") == "ok"


@pytest.mark.parametrize("order", [("rerun", "after"), ("after", "rerun")], ids=["rerun_first", "after_first"])
def test_running_the_loop_cell_again_ends_it(kernel, order):
    _start_loop(kernel, UNTIL_STOPPED)
    for name in order:
        kernel.run(name, UNTIL_STOPPED if name == "rerun" else 'print("after ran")')
    assert kernel.status("loop", timeout=5.0) == "ok"
    assert kernel.status("rerun") == "aborted"
    assert kernel.status("after") == "ok"
    assert "rerun" not in kernel.started
