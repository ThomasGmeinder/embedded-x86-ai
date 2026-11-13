# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The ROS graph ``run_pipeline.sh`` starts. The notebook starts this same launch.

``python -m so101_sim.stack`` resets the ignored SIGINT a background shell job
inherits, then replaces itself with ``ros2 launch launch/strix_vla.launch.py``.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LAUNCH = REPO / "launch" / "strix_vla.launch.py"


def pipeline_twin_arm() -> dict:
    """The arm launch ``run_pipeline.sh --digital-twin`` starts. The notebook starts this too."""
    return {"arm": True, "cameras": False, "digital_twin": True, "dry_run": False}


def launch_argv(
    *,
    arm: bool,
    cameras: bool,
    digital_twin: bool,
    dry_run: bool,
    config: str | None = None,
) -> list[str]:
    """The ``ros2 launch`` command."""
    ros2 = shutil.which("ros2")
    if ros2 is None:
        raise RuntimeError("ros2 is not on PATH: source /opt/ros/jazzy/setup.bash first")
    argv = [
        sys.executable, ros2, "launch", str(LAUNCH),
        f"arm:={str(arm).lower()}",
        f"cameras:={str(cameras).lower()}",
        f"digital_twin:={str(digital_twin).lower()}",
        f"dry_run:={str(dry_run).lower()}",
    ]
    if config:
        argv.append(f"config:={config}")
    return argv


def accept_stop_signals() -> None:
    """Leave a background shell job able to stop on SIGINT.

    ``run_pipeline.sh`` starts this module with ``&``. A non-interactive bash
    does not give that job its own process group, and it inherits SIGINT
    ignored. ``signal.signal`` cannot clear an inherited ignore. The cleanup
    in the script signals the group (``kill -INT -- -$pid``), so this process
    has to lead the group and accept the signal before it becomes the launch.
    """
    if os.getpgrp() != os.getpid():
        os.setsid()
    libc = ctypes.CDLL(None)
    libc.signal(signal.SIGINT, 0)
    libc.signal(signal.SIGQUIT, 0)


def exec_launch(**kwargs) -> None:
    """Replace this process with the launch. The process dies with its parent."""
    from so101_sim.parent_death import die_with_parent

    die_with_parent()
    argv = launch_argv(**kwargs)
    accept_stop_signals()
    os.execvp(argv[0], argv)


def popen_launch(*, log=None, env=None, **kwargs) -> subprocess.Popen:
    """The launch in its own session. The shell uses :func:`exec_launch` instead."""
    return subprocess.Popen(
        launch_argv(**kwargs),
        cwd=str(REPO),
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True,
        env=env,
    )


def popen_pipeline_twin_arm(*, log=None, env=None) -> subprocess.Popen:
    """``python -m so101_sim.stack --pipeline-twin-arm``, the notebook's arm launch."""
    return subprocess.Popen(
        [sys.executable, "-m", "so101_sim.stack", "--pipeline-twin-arm"],
        cwd=str(REPO),
        stdout=log,
        stderr=subprocess.STDOUT if log is not None else None,
        start_new_session=True,
        env=env,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start launch/strix_vla.launch.py")
    parser.add_argument("--pipeline-twin-arm", action="store_true",
                        help="the arm launch of run_pipeline.sh --digital-twin")
    parser.add_argument("--arm", choices=("true", "false"))
    parser.add_argument("--cameras", choices=("true", "false"))
    parser.add_argument("--digital-twin", default="false", choices=("true", "false"))
    parser.add_argument("--dry-run", default="false", choices=("true", "false"))
    parser.add_argument("--config", default=None)
    flag = {"true": True, "false": False}
    args = parser.parse_args()
    if args.pipeline_twin_arm:
        exec_launch(**pipeline_twin_arm(), config=args.config)
        return
    if args.arm is None or args.cameras is None:
        parser.error("--arm and --cameras are required")
    exec_launch(
        arm=flag[args.arm],
        cameras=flag[args.cameras],
        digital_twin=flag[args.digital_twin],
        dry_run=flag[args.dry_run],
        config=args.config,
    )


if __name__ == "__main__":
    main()
