# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline's entry points, driven like a person typing.

``run_pipeline.sh --digital-twin`` starts the twin and its controllers as it
starts the arm on the rig; each typed command waits for the line its behavior
ends with, UserRobotControl's ``=== <intent> ended (<how>) ===``. The robot
notebook's "On the rig" commands start the arm server and its controllers,
then the client's round trip.

Tier 2 (ROS 2): ``./run_sim_tests.sh``. The pipeline's run needs llama-server
and the webcam, so it runs with the docs tests: ``cd workshop && ./run_notebooks.sh --test``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from notebook_harness import PIPE_ROOT
from twin_ros_rig import assert_domain_free, join_test_domain, launch_env

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(600)]
pexpect = pytest.importorskip("pexpect")
popen_spawn = pytest.importorskip("pexpect.popen_spawn")
pytest.importorskip("rclpy")

# The stacks were ready 3.9-5.7 s after their start; a cold llama-server load
# adds 10-30 s (run_notebooks.sh). The longest behavior, a dance, took 22.4 s.
START_S = 120.0
BEHAVIOR_S = 60.0
PIPELINE_ENDED = r"=== (\w+) ended \((ok|stopped|failed)\) ==="
# The spawner's line for the last controller the launch activates.
CONTROLLERS_UP = r"Configured and activated gripper_controller"


@pytest.fixture
def twin_domain():
    join_test_domain()
    assert_domain_free("a twin already runs on the test domain")
    yield
    assert_domain_free("a twin outlived its entry point", wait=True)


def _spawn(command: str, cwd, log_path):
    """``command`` on pipes, not a pty (a pty fork in this multi-threaded process
    may deadlock), in a session of its own, so :func:`_close` can stop all of it."""
    child = popen_spawn.PopenSpawn(
        ["setsid", "bash", "-c", command], cwd=str(cwd), env=launch_env(),
        encoding="utf-8", codec_errors="replace", timeout=None,
    )
    child.logfile_read = open(log_path, "w")
    return child


def _close(child) -> int | None:
    """Let ``child`` finish and return its exit status. One that still runs gets
    SIGINT on its process group, as Ctrl-C sends it; one that outlasts that is
    killed, and fails the test."""
    if child.proc.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
        try:
            child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
        except pexpect.TIMEOUT:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise AssertionError(f"{child.proc.args} did not exit within {BEHAVIOR_S:.0f} s of SIGINT") from None
    else:
        child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    return child.wait()


@pytest.mark.docs
def test_run_pipeline_drives_the_twin(twin_domain, tmp_path):
    """Every object picked up and dropped, the scripted behaviors, a stop, and
    a clean shutdown: the launch reports no node that died."""
    log = tmp_path / "run_pipeline.log"
    child = _spawn("./run_pipeline.sh --digital-twin --text-commands --headless", PIPE_ROOT, log)
    try:
        child.expect(r"\[say\] Ready\.", timeout=START_S)
        # A behavior that gives up ends as failed: "hold this" with nothing between the jaws.
        for text, spoken, how in (
            ("pick up the ball", "Got the sports ball", "ok"),
            ("drop it", "Released", "ok"),
            ("pick up the bottle", "Got the bottle", "ok"),
            ("drop it", "Released", "ok"),
            ("pick up the toothbrush", "Got the toothbrush", "ok"),
            ("drop it", "Released", "ok"),
            ("wave at me", None, "ok"),
            ("hold this", "No object detected", "failed"),
            ("go home", None, "ok"),
            ("dance", None, "ok"),
        ):
            child.sendline(text)
            child.expect(PIPELINE_ENDED, timeout=BEHAVIOR_S)
            said = [line.strip() for line in (child.before or "").splitlines() if "[say]" in line]
            assert child.match.group(2) == how, (
                f"{text!r}: {child.match.group(1)} ended {child.match.group(2)}; said {said}"
            )
            assert spoken is None or f"[say] {spoken}" in child.before, f"{text!r} never said {spoken!r}"
        child.sendline("dance")
        child.expect(r"=== dance  ===", timeout=BEHAVIOR_S)
        # The trajectory controller's own report that a keyframe finished.
        # The behavior loggers do not show up on this process's stream.
        child.expect(r"Goal reached, success!", timeout=BEHAVIOR_S)
        child.sendline("stop robot")
        child.expect(PIPELINE_ENDED, timeout=BEHAVIOR_S)
        assert child.match.group(2) == "stopped", f"'stop' left the dance {child.match.group(2)}"
        child.sendeof()
        child.expect(r"Shutting down - parking arm", timeout=BEHAVIOR_S)
        child.expect(pexpect.EOF, timeout=BEHAVIOR_S)
    finally:
        status = _close(child)
        child.logfile_read.close()
    assert status == 0, f"run_pipeline.sh exited {status}"
    # The apt move_group segfaults when the launch sends SIGINT (the TEM plugin
    # unloads). A death of any other node, or any other exit code, still fails.
    died = [
        line for line in log.read_text().splitlines()
        if "process has died" in line and not ("move_group" in line and "exit code -11" in line)
    ]
    assert not died, "a node of the launch died on shutdown:\n" + "\n".join(died)


def test_the_robot_notebooks_rig_commands_round_trip(twin_domain, tmp_path):
    """The robot notebook's "On the rig": mock hardware and its controllers, then a MoveIt round trip."""
    log = tmp_path / "launch.log"
    launch = _spawn("exec .venv/bin/python $(which ros2) launch launch/strix_vla.launch.py cameras:=false dry_run:=true",
                    PIPE_ROOT, log)
    try:
        launch.expect(CONTROLLERS_UP, timeout=START_S)
        client = subprocess.run([sys.executable, "-m", "vla_pipeline.robot.robot_node", "--client-test"],
                                cwd=PIPE_ROOT, env=os.environ.copy(), capture_output=True, text=True,
                                timeout=BEHAVIOR_S)
        assert "ROS 2 round-trip PASSED" in client.stdout, client.stdout[-3000:] + client.stderr[-3000:]
    finally:
        status = _close(launch)
    text = log.read_text()
    # _close sends SIGINT to the whole launch, which is exit code -2.
    # move_group also exits -11 when its plugin unload hits that signal.
    # Any other death still fails.
    died = [
        line for line in text.splitlines()
        if "process has died" in line
        and "exit code -2" not in line
        and not ("move_group" in line and "exit code -11" in line)
    ]
    assert not died, "a node of the launch died:\n" + "\n".join(died)
    shutdown = "move_group" in text and ("exit code -11" in text or "exit code -2" in text)
    assert status == 0 or shutdown, f"the launch exited {status}"
