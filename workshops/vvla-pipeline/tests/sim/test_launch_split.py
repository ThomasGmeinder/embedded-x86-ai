# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""run_pipeline.sh starts the cameras and the arm as two ``ros2 launch`` runs.

One launch ends everything in it when one of its parts raises. A missing xacro
in the arm's part therefore took the camera nodes down with it. The arm is
made to fail here with a stand-in ``xacro`` package that raises on import; the
real ``strix_vla.launch.py`` runs unchanged.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys

import pytest

from notebook_harness import PIPE_ROOT
from twin_ros_rig import TEST_DOMAIN_ID

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.timeout(300)]
pexpect = pytest.importorskip("pexpect")
popen_spawn = pytest.importorskip("pexpect.popen_spawn")

# Cold start of a camera node and of a launch: a few seconds on this machine.
# These bound the wait for an event that is expected to happen.
CAMERA_START_S = 60.0
LAUNCH_EXIT_S = 120.0
STOP_S = 30.0
BROKEN_XACRO = 'raise ImportError("simulated missing xacro (test_launch_split)")\n'


@pytest.fixture
def broken_xacro(tmp_path):
    """A ``xacro`` package on PYTHONPATH whose import fails, as a missing one does."""
    pkg = tmp_path / "broken_xacro" / "xacro"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(BROKEN_XACRO)
    return tmp_path / "broken_xacro"


def _env(broken_path=None):
    env = {**os.environ, "ROS_DOMAIN_ID": str(TEST_DOMAIN_ID), "PYTHONUNBUFFERED": "1"}
    if broken_path is not None:
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(broken_path), os.environ.get("PYTHONPATH")]))
    return env


def _launch(arm: bool, cameras: bool, log_path, broken_path=None):
    """One ``ros2 launch`` of strix_vla.launch.py, as run_pipeline.sh starts it: its own session."""
    ros2 = shutil.which("ros2")
    assert ros2, "ROS 2 is not sourced - run this through ./run_sim_tests.sh"
    # setsid makes the launch a process-group leader, so _stop can SIGINT all of it.
    cmd = ["setsid", sys.executable, ros2, "launch", "launch/strix_vla.launch.py",
           f"arm:={str(arm).lower()}", f"cameras:={str(cameras).lower()}", "dry_run:=true"]
    child = popen_spawn.PopenSpawn(
        cmd, cwd=str(PIPE_ROOT), env=_env(broken_path), encoding="utf-8",
        codec_errors="replace", timeout=None,
    )
    child.logfile_read = open(log_path, "w")
    return child


def _stop(child) -> int:
    """SIGINT the launch's process group, as Ctrl-C does, and return its exit status."""
    if child.proc.poll() is None:
        os.killpg(child.pid, signal.SIGINT)
    try:
        return child.proc.wait(timeout=STOP_S)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.proc.wait()
        pytest.fail(f"launch did not stop within {STOP_S:.0f} s of SIGINT (log: {child.logfile_read.name})")


def test_an_arm_launch_with_a_missing_xacro_exits_with_an_error(tmp_path, broken_xacro):
    log = tmp_path / "arm.log"
    arm = _launch(arm=True, cameras=False, log_path=log, broken_path=broken_xacro)
    try:
        arm.expect(pexpect.EOF, timeout=LAUNCH_EXIT_S)
    finally:
        rc = _stop(arm)
    assert rc != 0, f"the arm launch should fail, not exit {rc} (log: {log})"
    assert "simulated missing xacro" in log.read_text(), f"the failure is not in the log: {log}"


def test_a_failing_arm_launch_does_not_end_the_camera_launch(tmp_path, broken_xacro):
    cam_log, arm_log = tmp_path / "cameras.log", tmp_path / "arm.log"
    cam = _launch(arm=False, cameras=True, log_path=cam_log)
    try:
        # The event: the mount camera node has started inside the camera launch.
        cam.expect(r"\[camera_mount-\d+\]: process started", timeout=CAMERA_START_S)
        arm = _launch(arm=True, cameras=False, log_path=arm_log, broken_path=broken_xacro)
        arm.expect(pexpect.EOF, timeout=LAUNCH_EXIT_S)
        arm_rc = _stop(arm)
        assert arm_rc != 0, f"the arm launch should fail (log: {arm_log})"
        # The arm launch has exited. Its failure must not have ended the camera launch.
        assert cam.proc.poll() is None, f"the camera launch ended with the arm launch (log: {cam_log})"
    finally:
        _stop(cam)
