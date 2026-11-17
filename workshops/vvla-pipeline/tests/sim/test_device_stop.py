# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The webcam guard forwards one SIGINT and ignores the signals that follow.

A stand-in camera records every signal. SIGTERM and a second SIGINT to the
guard must not be forwarded: that second signal is what wedges the AMD ISP
driver inside its stop call.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.timeout(30)]

CAMERA = """
import signal, sys
log = sys.argv[1]
def hit(signum, _frame):
    with open(log, "a") as fh:
        fh.write(f"{signum}\\n")
    print("signaled", flush=True)
signal.signal(signal.SIGINT, hit)
signal.signal(signal.SIGTERM, hit)
print("ready", flush=True)
while True:
    try:
        if sys.stdin.readline():
            break
    except InterruptedError:
        continue
"""


def test_a_follow_up_signal_is_not_forwarded_to_the_camera(tmp_path):
    log = tmp_path / "signals.log"
    guard = subprocess.Popen(
        [sys.executable, "-m", "so101_sim.device_stop", "--",
         sys.executable, "-c", CAMERA, str(log)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    try:
        assert guard.stdout.readline().strip() == "ready"
        os.kill(guard.pid, signal.SIGINT)
        assert guard.stdout.readline().strip() == "signaled"
        os.kill(guard.pid, signal.SIGTERM)
        os.kill(guard.pid, signal.SIGINT)
        guard.stdin.write("\n")
        guard.stdin.flush()
        assert guard.wait(timeout=5) == 0
        assert log.read_text().split() == [str(signal.SIGINT)]
    finally:
        if guard.poll() is None:
            os.kill(guard.pid, signal.SIGKILL)
            guard.wait()


def test_the_prefix_pins_the_guard_when_a_core_is_set():
    from so101_sim.device_stop import command_prefix

    assert command_prefix("") == f"{sys.executable} -m so101_sim.device_stop --"
    assert command_prefix("4") == f"taskset -c 4 {sys.executable} -m so101_sim.device_stop --"
