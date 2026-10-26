# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop README's Instructor's corner, run as written.

The commands are read from ``workshop/README.md``, so a README edit that
breaks one fails here.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from notebook_harness import PIPE_ROOT
from twin_ros_rig import join_test_domain

pytestmark = [pytest.mark.sim, pytest.mark.docs, pytest.mark.timeout(600)]

WORKSHOP = PIPE_ROOT / "workshop"
README = WORKSHOP / "README.md"


def _corner() -> list[str]:
    """The Instructor's corner commands, one per line (continuations joined)."""
    m = re.search(r"## Instructor's corner\s+```bash\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert m, "workshop/README.md has no Instructor's corner bash block"
    return [line.strip() for line in m.group(1).replace("\\\n", " ").splitlines() if line.strip()]


def _env() -> dict:
    """This setup's environment, with the venv's python first on PATH (README: activate .venv)."""
    env = os.environ.copy()
    env["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}"
    return env


def test_instructor_commands_run():
    join_test_domain()
    commands = _corner()
    assert commands, "the Instructor's corner lists no command"
    for command in commands:
        done = subprocess.run(command, shell=True, cwd=WORKSHOP, env=_env(), capture_output=True,
                              text=True, executable="/bin/bash")
        assert done.returncode == 0, (
            f"README command failed ({done.returncode}): {command}\n"
            f"--- stdout\n{done.stdout[-3000:]}\n--- stderr\n{done.stderr[-3000:]}"
        )
