# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ros2_control for the SO-101: the bringup launch and the names its clients use.

``launch/so101_control.launch.py`` starts upstream ``so_arm101`` on
``mujoco``, ``mock_components``, or the Feetech bus, plus ``move_group``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

from vla_pipeline.utils.config import REPO_ROOT  # noqa: E402
LAUNCH_FILE = REPO_ROOT / "launch" / "so101_control.launch.py"


def sensor_qos():
    """Best-effort, keep-last, depth 1: the profile camera and joint subscribers use."""
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1
    )


def start_controllers(config_path, gripper: str = "forward", log=None, hardware_type: str = "mujoco") -> subprocess.Popen:
    """``launch/so101_control.launch.py`` with this Python, in a process group of
    its own, shut down (SIGINT) when this process exits however it exits.

    ``hardware_type`` is ``mujoco`` (the digital twin), ``mock_components`` or
    ``real`` (the Feetech bus). ``gripper`` is accepted so older callers still
    import, and the ros2_so_arm bringup chooses the gripper controller itself.
    """
    import shutil

    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 is not on PATH: source /opt/ros/jazzy/setup.bash first")
    return subprocess.Popen(
        [sys.executable, "-m", "so101_sim.parent_death", sys.executable, ros2, "launch", str(LAUNCH_FILE),
         f"config:={config_path}", f"hardware_type:={hardware_type}"],
        cwd=str(REPO_ROOT), stdout=log, stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True,
    )


def stop_controllers(proc: subprocess.Popen, budget_s: float = 10.0) -> None:
    """SIGINT the launch's process group; kill it when it has not exited within ``budget_s``."""
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=budget_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise RuntimeError(f"the controllers' launch did not exit within {budget_s:.0f} s of SIGINT") from None
