#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Single entry point for the Strix VLA pipeline.
#
#   ./run_pipeline.sh                     # voice control, full hardware
#   ./run_pipeline.sh --text-commands     # type commands instead of speaking
#   ./run_pipeline.sh --dry-run           # no robot hardware
#   ./run_pipeline.sh --digital-twin      # no robot hardware: MuJoCo twin of the arm,
#                                         #   shown in the twin console window
#   ./run_pipeline.sh --headless          # no preview windows
#
# Execs ``python -m so101_sim.stack serve``. That process starts the webcam
# launch and the arm launch (they stay apart: ros2 launch ends a whole launch
# when one part raises), then perception, the OpenCV console, and control, on
# the cores in config/pipeline.yaml. It stops them when the console window is
# closed, when control exits, or on Ctrl-C.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# ROS 2 underlay, then the ros_ws overlay (so_arm101, the twin plugins, MoveIt).
[[ -f /opt/ros/jazzy/setup.bash ]] && source /opt/ros/jazzy/setup.bash
[[ -f "${REPO_ROOT}/ros_ws/install/setup.bash" ]] && source "${REPO_ROOT}/ros_ws/install/setup.bash"

[[ -f .venv/bin/activate ]] || { echo "No .venv - run ./bootstrap.sh first." >&2; exit 1; }

# build_parser() in vla_pipeline.main is the list of accepted arguments.
# Check it before any setup: an unknown flag must not start a model download
# or the arm. --help prints and stops here too.
status=0
parsed=$(.venv/bin/python - "$@" <<'PY'
import sys
from vla_pipeline.main import build_parser
build_parser(prog="run_pipeline.sh").parse_args(sys.argv[1:])
print("ARGS_OK")
PY
) || status=$?
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi
if [[ "$parsed" != *ARGS_OK* ]]; then
  printf '%s\n' "$parsed"
  exit 0
fi

# One publisher of /control/state. A leftover control process from an earlier
# run would interleave its last sentence with this run's.
busy=$(.venv/bin/python -c 'from vla_pipeline.single_instance import format_running; print(format_running())')
if [[ -n "$busy" ]]; then
  echo "A pipeline process is already running. Quit it before starting another:" >&2
  printf '%s\n' "$busy" >&2
  exit 1
fi

VENV_DIR="${REPO_ROOT}/.venv" "${REPO_ROOT}/scripts/ensure_voice.sh"
source .venv/bin/activate

# Ryzen AI NPU runtime libs (LD_LIBRARY_PATH for the VitisAI EP).
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/ryzen_ai_env.sh"

export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1   # SmolVLA/torch attention on ROCm
export HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.5.1}"  # gfx1151 ROCm dispatch fix

NPU_CONFIGURED="$(python - <<'PY'
from vla_pipeline.utils.config import load_config
cfg = load_config()
sections = ("yolo_pose", "yolo_detect")
print("yes" if any(str((cfg.get(s) or {}).get("device", "cpu")).lower() == "npu"
                   for s in sections) else "no")
PY
)"
if [[ "$NPU_CONFIGURED" == "yes" ]]; then
  python scripts/verify_npu_stack.py --preflight || {
    echo "WARNING: NPU preflight failed; models configured for the NPU will run on CPU." >&2
  }
fi

# so101_sim.stack serve is the process that starts the stack and stops it.
# exec so Ctrl-C reaches that process, and typed commands stay on its stdin.
exec python -m so101_sim.stack serve "$@"
