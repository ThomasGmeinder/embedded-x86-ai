#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# =============================================================================
# run_notebooks.sh - one command to open the workshop notebooks.
#
# ./bootstrap.sh installs everything this needs: the repo venv (../.venv) with
# the pipeline, the workshop helpers and the twin. This script, in order:
#   1. Checks the venv, registers the "vvla-workshop" Jupyter kernel for this
#      clone (scripts/register_kernel.sh: the NPU runtime and ROS 2 go in its
#      kernel.json, so VS Code runs the notebooks the same way; every launch
#      does it because that file is shared by all clones), and sources the Ryzen AI runtime and
#      ROS 2 for the llama server and the tests started here. Installs this
#      clone's vla_pipeline into the venv, and fetches the Gemma 4 voice
#      models (scripts/ensure_voice.sh) when they are missing.
#   2. Starts the Gemma 4 llama-server on the iGPU through the pipeline
#      (python -m vla_pipeline.llm.llama_intent --serve, from config/pipeline.yaml)
#      unless one already answers. It logs to workshop/logs/llama-serve.log and
#      stops when you quit Jupyter.
#   3. Opens the always-on-top resource HUD (CPU% / GPU% / NPU inf/s) so you
#      can watch the devices light up while a cell runs. It closes with Jupyter.
#   4. Launches Jupyter Lab in workshop/ (browse the notebooks/ folder).
#
# Usage:
#   ./run_notebooks.sh                 # launch Jupyter Lab (+ llama server + HUD)
#   ./run_notebooks.sh --no-launch     # set everything up, don't start Jupyter
#   ./run_notebooks.sh --classic       # use `jupyter notebook` instead of lab
#   ./run_notebooks.sh --no-llm        # don't start the llama server
#   ./run_notebooks.sh --no-monitor    # don't open the resource HUD
#   ./run_notebooks.sh --test [args]   # steps 1-2, then the notebooks instead of Jupyter
#                                      # (args go to those tests). The twin and the NPU
#                                      # tests are ../run_sim_tests.sh
#
# Env overrides:
#   JUPYTER_PORT=8888             port to serve on
# =============================================================================
set -euo pipefail

say()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

say "run_notebooks.sh${*:+ $*}"

WORKSHOP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${WORKSHOP_DIR}/.." && pwd)"
KERNEL_NAME="vvla-workshop"  # the kernel the notebooks ask for; scripts/register_kernel.sh registers it
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

NO_LAUNCH=0; LAUNCHER="lab"; WITH_LLM=1; WITH_MONITOR=1; TEST_MODE=0
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-launch)    NO_LAUNCH=1 ;;
    --classic)      LAUNCHER="notebook" ;;
    --no-llm)       WITH_LLM=0 ;;
    --no-monitor)   WITH_MONITOR=0 ;;
    --test)         TEST_MODE=1; NO_LAUNCH=1; WITH_MONITOR=0; shift; PYTEST_ARGS=("$@"); break ;;
    -h|--help)      sed -n '/^# run_notebooks.sh/,/^# =====/p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
if [[ "$TEST_MODE" -eq 1 && "$WITH_LLM" -eq 0 ]]; then
  echo "--test needs the llama server: the intent notebook runs against it (drop --no-llm)" >&2
  exit 2
fi

# --- 1. The venv, the kernel, and the runtime environment --------------------
VENV_DIR="${REPO_ROOT}/.venv"
PY="${VENV_DIR}/bin/python"
[[ -x "$PY" ]] || die "no venv at ${VENV_DIR} - run ./bootstrap.sh from the repo root first."
say "Python: $("$PY" -c 'import sys; print(sys.version.split()[0], "@", sys.executable)')"

# The kernel.json is per user and shared by every clone, so register it from
# this clone on every launch.
"${REPO_ROOT}/scripts/register_kernel.sh" || die "could not register the '${KERNEL_NAME}' kernel - run ./bootstrap.sh from the repo root."
say "Kernel '${KERNEL_NAME}' registered for this clone (NPU runtime and ROS 2 in its kernel.json)"

say "Sourcing scripts/ryzen_ai_env.sh (NPU runtime, llama.cpp GPU target)"
# shellcheck disable=SC1091
VIRTUAL_ENV="$VENV_DIR" source "${REPO_ROOT}/scripts/ryzen_ai_env.sh" || die "ryzen_ai_env.sh failed"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
[[ -f "$ROS_SETUP" ]] || die "ROS 2 is required and is not at $ROS_SETUP - run ./bootstrap.sh from the repo root"
say "Sourcing ROS 2 ($ROS_SETUP)"
# ROS's setup.bash dereferences unbound vars (AMENT_TRACE_SETUP_FILES, ...) and
# can return nonzero on benign conditions. Relax nounset/errexit across the
# source, then restore our strict mode.
set +u +e
# shellcheck disable=SC1090
source "$ROS_SETUP" || warn "ROS 2 source returned nonzero - continuing"
ROS_WS_SETUP="${REPO_ROOT}/ros_ws/install/setup.bash"
[[ -f "$ROS_WS_SETUP" ]] || die "no ros_ws overlay at $ROS_WS_SETUP - run ./bootstrap.sh from the repo root"
say "Sourcing the ros_ws overlay ($ROS_WS_SETUP)"
# shellcheck disable=SC1090
source "$ROS_WS_SETUP" || die "ros_ws setup failed"
set -e -u

# A pull does not re-run bootstrap. Without this, the kernel cannot import vla_pipeline.
say "Installing vla_pipeline from this clone"
"$PY" -m pip install --quiet --no-deps --upgrade --force-reinstall -e "${REPO_ROOT}"
# The digital-twin notebook's microphone needs the voice models and a llama-server with Gemma 4 audio.
VENV_DIR="$VENV_DIR" "${REPO_ROOT}/scripts/ensure_voice.sh"

# --- 2. llama-server on the iGPU (component 1's brain, torn down on exit) ----
LLAMA_PID=""
MONITOR_PID=""
cleanup() {
  [[ -n "$LLAMA_PID" ]] && kill -INT "$LLAMA_PID" 2>/dev/null || true
  [[ -n "$MONITOR_PID" ]] && kill "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$WITH_LLM" -eq 1 ]]; then
  mkdir -p "${WORKSHOP_DIR}/logs"
  LLAMA_LOG="${WORKSHOP_DIR}/logs/llama-serve.log"
  say "Starting llama-server on the iGPU: python -m vla_pipeline.llm.llama_intent --serve"
  say "  log: ${LLAMA_LOG} (first load takes 10-30 s while the model streams into iGPU memory)"
  "$PY" -m vla_pipeline.llm.llama_intent --serve >"$LLAMA_LOG" 2>&1 &
  LLAMA_PID=$!
fi

# --test: the notebooks, in this environment. Twin and NPU tests are ../run_sim_tests.sh.
if [[ "$TEST_MODE" -eq 1 ]]; then
  # --serve prints its ready line once llama-server answers, or exits when it
  # cannot start; tail follows the log until then and ends with the process.
  say "Waiting for llama-server to be ready (${LLAMA_LOG})"
  exec {llama_fd}< <(tail -n +1 -F --pid="$LLAMA_PID" "$LLAMA_LOG")
  LLAMA_TAIL_PID=$!
  LLAMA_READY=0
  grep -m1 -q "llama-server ready at" <&"$llama_fd" || LLAMA_READY=$?
  kill "$LLAMA_TAIL_PID" 2>/dev/null || true
  exec {llama_fd}<&-
  [[ "$LLAMA_READY" -eq 0 ]] || die "llama-server did not start - see ${LLAMA_LOG}"
  cd "$REPO_ROOT"
  say "Notebooks"
  "$PY" -m pytest tests/sim -m docs "${PYTEST_ARGS[@]}" || exit $?
  exit 0
fi

# --- 3. Pop the always-on-top resource HUD (CPU/GPU/NPU), torn down on exit --
# Only when we're really launching Jupyter and a display is available. It reuses
# launch_monitor.sh, logs to workshop/logs/monitor.log, and the cleanup trap
# kills it (alongside the llama server) when Jupyter exits.
if [[ "$WITH_MONITOR" -eq 1 && "$NO_LAUNCH" -eq 0 ]]; then
  if [[ -n "${DISPLAY:-}" || -n "${WAYLAND_DISPLAY:-}" ]]; then
    mkdir -p "${WORKSHOP_DIR}/logs"
    say "Opening the resource HUD (CPU/GPU/NPU; --no-monitor to skip)"
    "${WORKSHOP_DIR}/launch_monitor.sh" >>"${WORKSHOP_DIR}/logs/monitor.log" 2>&1 &
    MONITOR_PID=$!
    say "  resource HUD PID ${MONITOR_PID} (closes when you quit Jupyter)"
  else
    warn "No display (DISPLAY/WAYLAND_DISPLAY unset) - skipping the resource HUD."
  fi
fi

# --- 4. Launch Jupyter (or stop here with --no-launch) ----------------------
if [[ "$NO_LAUNCH" -eq 1 ]]; then
  say "Setup complete (--no-launch). Start it yourself with:"
  echo "    cd ${WORKSHOP_DIR} && ${PY} -m jupyter ${LAUNCHER} notebooks"
  exit 0
fi

# A live camera cell holds /dev/video0 in its kernel. Jupyter leaves that
# kernel running when the notebook tab is closed, so the next notebook cannot
# open the AMD ISP camera until this whole script is restarted. Shut the
# kernel down with the notebook instead.
LAB_OVERRIDES="${VENV_DIR}/share/jupyter/lab/settings/overrides.json"
mkdir -p "$(dirname "$LAB_OVERRIDES")"
"$PY" - "$LAB_OVERRIDES" <<'PY'
import json, sys
path = sys.argv[1]
try:
    with open(path) as fh:
        data = json.load(fh)
except FileNotFoundError:
    data = {}
data.setdefault("@jupyterlab/notebook-extension:tracker", {})["kernelShutdown"] = True
with open(path, "w") as fh:
    json.dump(data, fh, indent=2)
    fh.write("\n")
PY

say "Launching Jupyter ${LAUNCHER} - the workshop kernel is auto-selected"
cd "$WORKSHOP_DIR"
# Not exec'd: keep this shell alive so the trap tears the llama server down
# after Jupyter exits.
"$PY" -m jupyter "$LAUNCHER" notebooks --port "$JUPYTER_PORT"
