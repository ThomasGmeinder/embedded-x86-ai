#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Registers the notebooks' Jupyter kernel "vvla-workshop" for THIS clone.
#
# The kernel.json lives in the user's Jupyter directory, so every clone on this
# machine shares it and the last writer wins. ./bootstrap.sh and
# workshop/run_notebooks.sh both call this script, and run_notebooks.sh does it
# on every launch, so the kernel always matches the tree you start it from.
#
# VS Code and JupyterLab start a kernel from its kernel.json alone and source
# nothing, so the NPU runtime (scripts/ryzen_ai_env.sh), ROS 2 and the ros_ws
# overlay are written into it.
#
#   scripts/register_kernel.sh            # uses ../.venv
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
PY="${VENV_DIR}/bin/python"
KERNEL_NAME="vvla-workshop"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
ROS_WS_SETUP="${REPO_ROOT}/ros_ws/install/setup.bash"

[[ -x "$PY" ]] || { echo "ERROR: no venv python at ${PY}" >&2; exit 1; }
[[ -f "$ROS_SETUP" ]] || { echo "ERROR: ROS 2 is not at ${ROS_SETUP}" >&2; exit 1; }
[[ -f "$ROS_WS_SETUP" ]] || { echo "ERROR: no ros_ws overlay at ${ROS_WS_SETUP}" >&2; exit 1; }

"$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python (VVLA workshop)" >/dev/null

KERNEL_ENV_JSON="$(
  set +eu
  # The same environment as an activated venv.
  export VIRTUAL_ENV="$VENV_DIR"
  export PATH="${VENV_DIR}/bin:${PATH}"
  # shellcheck disable=SC1090
  source "$ROS_SETUP"
  # shellcheck disable=SC1090
  source "$ROS_WS_SETUP"
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/scripts/ryzen_ai_env.sh"
  "$PY" - <<'PY'
import json, os
keys = [
    "PATH", "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PYTHONPATH",
    "XLNX_VART_FIRMWARE", "XILINX_XRT", "HSA_OVERRIDE_GFX_VERSION", "FASTDDS_BUILTIN_TRANSPORTS",
    "AMENT_PREFIX_PATH", "AMENT_CURRENT_PREFIX", "COLCON_PREFIX_PATH", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH",
    "ROS_DISTRO", "ROS_VERSION", "ROS_PYTHON_VERSION", "ROS_LOCALHOST_ONLY", "RMW_IMPLEMENTATION",
]
print(json.dumps({k: os.environ[k] for k in keys if os.environ.get(k)}))
PY
)" || { echo "ERROR: could not read the runtime environment for the kernel." >&2; exit 1; }

"$PY" - "$KERNEL_NAME" "$KERNEL_ENV_JSON" <<'PY'
import json, sys
from pathlib import Path
from jupyter_client.kernelspec import KernelSpecManager
path = Path(KernelSpecManager().get_kernel_spec(sys.argv[1]).resource_dir) / "kernel.json"
spec = json.loads(path.read_text())
spec["env"] = json.loads(sys.argv[2])
path.write_text(json.dumps(spec, indent=1) + "\n")
print(f"  kernel '{sys.argv[1]}' registered for this clone: {path}")
PY
