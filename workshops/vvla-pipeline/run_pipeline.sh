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
# Starts launch/strix_vla.launch.py in the background, as two independent
# launches: the webcam nodes when a camera's use_ros2 is set, and the arm
# bringup (MuJoCo twin, mock hardware, or the Feetech bus) when robot.use_ros2
# is set. ros2 launch ends everything in a launch when one of its parts raises
# (a missing xacro, an absent arm), so the cameras never share a launch with
# the arm. Then it runs the orchestrator, whose MoveIt arm waits for the
# controllers.
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

USE_ROS2="$(python - <<'PY'
from vla_pipeline.utils.config import load_config
cfg = load_config()
robot = str(cfg["robot"].get("use_ros2", False)).lower()
cams = cfg.get("cameras") or {}
cam_ros2 = any(str((cams.get(r) or {}).get("use_ros2", False)).lower() == "true"
               for r in ("mount", "arm"))
print(f"{robot} {str(cam_ros2).lower()}")
PY
)"
ROBOT_ROS2="${USE_ROS2%% *}"
CAMERA_ROS2="${USE_ROS2##* }"

LAUNCH_PIDS=()
cleanup() {
  # Each launch is a background job, so it and the servo wrapper inherit SIGINT
  # ignored. Bash will not install the wrapper's stop trap for a signal it
  # inherited ignored, and a non-interactive wait does not return on SIGINT.
  # SIGTERM is the signal the wrapper traps: it stops the servo user unit, and
  # the launch process, which does not ignore SIGTERM, exits. A further Ctrl-C
  # must not cut this short. Both launches get the signal first, then each is
  # waited for.
  trap '' INT
  local pid
  for pid in "${LAUNCH_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || true
    fi
  done
  for pid in "${LAUNCH_PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT

DRY=""
SIM=""
for a in "$@"; do
  [[ "$a" == "--dry-run" ]] && DRY="--dry-run"
  [[ "$a" == "--digital-twin" ]] && SIM="--digital-twin"
done

if [[ "$ROBOT_ROS2" == "true" || "$CAMERA_ROS2" == "true" ]]; then
  command -v ros2 >/dev/null || { echo "ROS 2 not found - install ROS 2 Jazzy (/opt/ros/jazzy)." >&2; exit 1; }
  # On the twin the wrist and scene images come from the MuJoCo plugins.
  # The camera launch still starts the real mount webcam.
  TWIN=false
  [[ -n "$SIM" && -z "$DRY" && "$ROBOT_ROS2" == "true" ]] && TWIN=true
  DRY_ARM=false
  [[ -n "$DRY" ]] && DRY_ARM=true
  # start_launch ARM CAMERAS: one strix_vla.launch.py in a session of its own.
  start_launch() {
    setsid python "$(command -v ros2)" launch launch/strix_vla.launch.py \
      arm:="$1" cameras:="$2" digital_twin:="$TWIN" dry_run:="$DRY_ARM" &
    LAUNCH_PIDS+=("$!")
  }
  if [[ "$CAMERA_ROS2" == "true" ]]; then
    echo "Starting the camera nodes..."
    start_launch false true
  fi
  if [[ "$ROBOT_ROS2" == "true" ]]; then
    echo "Starting the arm nodes (digital twin: $TWIN)..."
    start_launch true false
  fi
fi

python -m vla_pipeline.main "$@"
