# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Sourced by run_notebooks.sh. The notebook server and the camera launches.
#
# A foreground jupyter catches the first Ctrl-C and asks whether to shut
# down. Bash does not run the caller's trap until that process exits, so
# whatever holds /dev/video0 keeps it. The server is therefore a background
# job this shell waits on, and the trap below is what Ctrl-C runs.
#
# The server ignores SIGINT, and its stdin is not a terminal. Python keeps
# an ignored SIGINT ignored, so Ctrl-C does not become KeyboardInterrupt.
# Jupyter installs its shutdown prompt only when stdin is a terminal;
# closing stdin keeps that prompt from replacing the ignored signal.
# SIGTERM is the signal Jupyter handles by shutting its kernels down. A
# notebook that opened the webcam holds it in the kernel, and the kernel's
# exit closes the device.
#
# Each camera launch leads its own session. SIGINT is the signal
# so101_sim.stack armed that session with. One SIGINT releases the device.
# A second signal while the node is releasing it drops the AMD ISP driver
# into an uninterruptible stop and /dev/video0 stays open, so this stop
# ignores a further Ctrl-C until the launches and the server have exited.

# The caller sets JUPYTER_PID="" and LAUNCH_PIDS=() before the trap.
# LLAMA_PID and MONITOR_PID are optional.

release_webcam() {
  local pid
  local -a launches=()
  if declare -p LAUNCH_PIDS >/dev/null 2>&1; then
    launches=("${LAUNCH_PIDS[@]}")
  fi
  for pid in "${launches[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      # The pid leads the launch session, so the group id is the pid.
      kill -INT -- "-$pid" 2>/dev/null || true
    fi
  done
  if [[ -n "${JUPYTER_PID:-}" ]] && kill -0 "$JUPYTER_PID" 2>/dev/null; then
    kill -TERM "$JUPYTER_PID" 2>/dev/null || true
  fi
  for pid in "${launches[@]}"; do
    wait "$pid" 2>/dev/null || true
  done
  if [[ -n "${JUPYTER_PID:-}" ]]; then
    wait "$JUPYTER_PID" 2>/dev/null || true
  fi
}

stop_session() {
  trap '' INT TERM EXIT
  # run_pipeline.sh is its own session, and a background job ignores SIGINT.
  # SIGTERM runs that script's EXIT cleanup: perception and the console, then
  # control, then the launches. LAUNCH_PIDS stay the camera-style SIGINT path.
  if [[ -n "${PIPELINE_PID:-}" ]] && kill -0 "$PIPELINE_PID" 2>/dev/null; then
    kill -TERM "$PIPELINE_PID" 2>/dev/null || true
    wait "$PIPELINE_PID" 2>/dev/null || true
  fi
  release_webcam
  if [[ -n "${LLAMA_PID:-}" ]]; then
    kill -INT "$LLAMA_PID" 2>/dev/null || true
  fi
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "$MONITOR_PID" 2>/dev/null || true
  fi
}

# Blocks until the server exits. The caller's trap is stop_session.
run_notebook_server() {
  local py=$1 launcher=$2 port=$3
  (trap '' INT; exec "$py" -m jupyter "$launcher" notebooks --port "$port") </dev/null &
  JUPYTER_PID=$!
  wait "$JUPYTER_PID" || true
}
