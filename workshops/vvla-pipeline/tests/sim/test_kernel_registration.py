# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``scripts/register_kernel.sh``: the notebook kernel always matches this clone.

The ``vvla-workshop`` kernel.json lives in the user's Jupyter directory, which
every clone on the machine shares, so an older clone's launcher can overwrite
it. ``run_notebooks.sh`` registers the kernel on every launch. These tests use a
throwaway Jupyter data directory (``JUPYTER_DATA_DIR``), so the user's own
kernel is not touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from notebook_harness import KERNEL, PIPE_ROOT

pytestmark = [pytest.mark.sim, pytest.mark.timeout(180)]

REGISTER = PIPE_ROOT / "scripts" / "register_kernel.sh"

# The kernel's own answer, through the same Jupyter lookup by name that
# JupyterLab and the notebook tests use.
START_KERNEL = f"""
import os, sys
from jupyter_client.manager import start_new_kernel
km, kc = start_new_kernel(kernel_name={KERNEL!r}, startup_timeout=90)
try:
    reply = kc.execute_interactive(
        "import os, vla_pipeline\\nassert os.environ['AMENT_PREFIX_PATH'], 'no ROS 2 in the kernel'",
        timeout=60,
    )
    sys.exit(0 if reply["content"]["status"] == "ok" else 1)
finally:
    kc.stop_channels()
    km.shutdown_kernel(now=True)
"""


@pytest.fixture
def jupyter_data(tmp_path):
    data = tmp_path / "jupyter-data"
    spec_dir = data / "kernels" / KERNEL
    spec_dir.mkdir(parents=True)
    return data, spec_dir / "kernel.json"


def _register(data) -> subprocess.CompletedProcess:
    env = {**os.environ, "JUPYTER_DATA_DIR": str(data)}
    return subprocess.run([str(REGISTER)], env=env, capture_output=True, text=True, timeout=120)


def test_a_kernel_left_by_an_older_clone_is_replaced_and_starts(jupyter_data):
    """The failure that killed JupyterLab's kernel: a launcher module this tree no longer has."""
    data, spec = jupyter_data
    spec.write_text(json.dumps({
        "argv": [sys.executable, "-m", "so101_sim.bringup", "kernel", "-f", "{connection_file}"],
        "display_name": "Python (VVLA workshop)",
        "language": "python",
        "env": {},
    }))

    done = _register(data)
    assert done.returncode == 0, done.stdout + done.stderr

    written = json.loads(spec.read_text())
    assert "so101_sim.bringup" not in written["argv"]
    assert "ipykernel_launcher" in written["argv"]
    assert str(PIPE_ROOT / ".venv" / "bin" / "python") == written["argv"][0]
    assert written["env"].get("AMENT_PREFIX_PATH") and written["env"].get("LD_LIBRARY_PATH")

    started = subprocess.run(
        [str(PIPE_ROOT / ".venv" / "bin" / "python"), "-c", START_KERNEL],
        env={**os.environ, "JUPYTER_DATA_DIR": str(data)}, capture_output=True, text=True, timeout=150,
    )
    assert started.returncode == 0, "the registered kernel did not start:\n" + started.stdout + started.stderr


def test_registering_twice_leaves_one_working_kernel(jupyter_data):
    data, spec = jupyter_data
    assert _register(data).returncode == 0
    first = spec.read_text()
    assert _register(data).returncode == 0
    assert spec.read_text() == first
