# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""One stop signal for a webcam, then wait until that process has exited.

``launch/strix_vla.launch.py`` starts ``v4l2_camera`` behind this process, for
``./run_pipeline.sh`` and for the digital-twin notebook. ros2 launch signals
this process. A second signal while the AMD ISP driver is in
``isp4sd_stop_stream`` leaves ``/dev/video0`` stuck. The camera runs in its
own session, so a signal to the launch's process group does not reach it.
The first SIGINT or SIGTERM here is forwarded as one SIGINT. Anything after
that is ignored until the camera exits.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys


def command_prefix(core: str = "") -> str:
    """The launch prefix in front of ``v4l2_camera_node``, pinned to ``core`` when set."""
    guard = f"{sys.executable} -m so101_sim.device_stop --"
    core = str(core or "").strip()
    if not core:
        return guard
    return f"taskset -c {core} {guard}"


def main() -> None:
    """Run ``argv`` after ``--`` and stop it with one SIGINT."""
    if "--" not in sys.argv:
        raise SystemExit("usage: python -m so101_sim.device_stop -- CMD [ARGS...]")
    cmd = sys.argv[sys.argv.index("--") + 1:]
    if not cmd:
        raise SystemExit("usage: python -m so101_sim.device_stop -- CMD [ARGS...]")
    if os.getpgrp() != os.getpid():
        try:
            os.setsid()
        except OSError:
            pass
    # The camera must not inherit a blocked SIGINT: that mask is copied at
    # fork, and a blocked stop signal never reaches the driver.
    camera_holder: list[subprocess.Popen | None] = [None]
    sent = False
    requested = False

    def _send() -> None:
        nonlocal sent
        camera = camera_holder[0]
        if sent or camera is None:
            return
        sent = True
        try:
            os.kill(camera.pid, signal.SIGINT)
        except ProcessLookupError:
            pass

    def _forward(_signum, _frame) -> None:
        nonlocal requested
        requested = True
        _send()

    signal.signal(signal.SIGINT, _forward)
    signal.signal(signal.SIGTERM, _forward)
    camera_holder[0] = subprocess.Popen(cmd, start_new_session=True)
    if requested:
        _send()
    code = camera_holder[0].wait()
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
