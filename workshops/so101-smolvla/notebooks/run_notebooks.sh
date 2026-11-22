#!/usr/bin/env bash
# =============================================================================
# run_notebooks.sh - open the notebooks of this directory.
#
# Every notebook here runs in one Jupyter kernel:
#     name     so101-smolvla-notebooks
#     shown as "SO-101 SmolVLA notebooks (voice workshop environment)"
# The kernel is the voice workshop's Python (workshops/vvla-pipeline/.venv),
# started through that workshop's own environment scripts (.venv/bin/activate
# and scripts/ryzen_ai_env.sh). The scripts run at every kernel start, so
# JupyterLab, VS Code and Cursor get the same environment and no notebook sets
# it up itself. The notebooks name this kernel, so it is preselected.
#
# Usage:
#   ./run_notebooks.sh                register the kernel, start JupyterLab in
#                                     this directory and open the browser
#   ./run_notebooks.sh --no-browser   the same without a browser; JupyterLab
#                                     prints its address with the access token
#   ./run_notebooks.sh --kernel-only  register the kernel and stop
#   ./run_notebooks.sh --port N       serve on port N (default: JUPYTER_PORT or 8888)
#
# VS Code / Cursor, either way:
#   a) ./run_notebooks.sh --kernel-only   (once), open the notebook, then
#      "Select Kernel" -> "Jupyter Kernel..." ->
#      "SO-101 SmolVLA notebooks (voice workshop environment)".
#      Not "Python Environments...": that starts the bare Python of a .venv,
#      without the environment scripts, and the notebook's first cell stops.
#   b) ./run_notebooks.sh --no-browser, then "Select Kernel" ->
#      "Existing Jupyter Server..." and paste the printed address. The same
#      kernel is offered there.
# =============================================================================
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_NAME="so101-smolvla-notebooks"
KERNEL_DISPLAY="SO-101 SmolVLA notebooks (voice workshop environment)"
PORT="${JUPYTER_PORT:-8888}"
KERNEL_ONLY=0
LAB_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kernel-only) KERNEL_ONLY=1 ;;
    --no-browser)  LAB_ARGS+=(--no-browser) ;;
    --port)        [[ $# -ge 2 ]] || die "--port needs a number"; PORT="$2"; shift ;;
    -h|--help)     sed -n '/^# run_notebooks.sh/,/^# =====/p' "$0"; exit 0 ;;
    *)             die "unknown option: $1 (see --help)" ;;
  esac
  shift
done

# --- Readiness: the voice workshop's Python and its two environment scripts ---
# The kernel starts through scripts/run_voice_env.sh, which sources the two
# scripts; the same launcher runs the Gemma worker of the three-cube test.
ENV_LAUNCHER="$(cd "$HERE/../scripts" && pwd)/run_voice_env.sh"
[[ -x "$ENV_LAUNCHER" ]] || die "missing $ENV_LAUNCHER"
[[ -d "$HERE/../../vvla-pipeline" ]] || die "the voice workshop is not at $HERE/../../vvla-pipeline"
VOICE_WORKSHOP="$(cd "$HERE/../../vvla-pipeline" && pwd)"
VENV="$VOICE_WORKSHOP/.venv"
[[ -x "$VENV/bin/python" ]] || die "no Python at $VENV/bin/python - run ./bootstrap.sh in $VOICE_WORKSHOP first"
[[ -f "$VENV/bin/activate" ]] || die "missing $VENV/bin/activate"
[[ -f "$VOICE_WORKSHOP/scripts/ryzen_ai_env.sh" ]] || die "missing $VOICE_WORKSHOP/scripts/ryzen_ai_env.sh"
"$VENV/bin/python" -c 'import ipykernel, jupyter_client' \
  || die "ipykernel / jupyter_client are not installed in $VENV"

# --- The kernel: the workshop's two scripts, then the kernel itself ----------
# The start command is the environment launcher with the kernel's own command.
# It does not change directory, so a notebook starts where it lies, and it
# "exec"s the kernel, so interrupting a cell reaches the process Jupyter
# started.
SPEC_DIR="$(mktemp -d)"
trap 'rm -rf "$SPEC_DIR"' EXIT
"$VENV/bin/python" - "$SPEC_DIR/kernel.json" "$KERNEL_NAME" "$KERNEL_DISPLAY" "$ENV_LAUNCHER" <<'PYEOF'
import json
import sys

path, name, display, launcher = sys.argv[1:]
spec = {
    "argv": [launcher, "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": display,
    "language": "python",
    "metadata": {"debugger": True},
}
with open(path, "w", encoding="utf-8") as fh:
    json.dump(spec, fh, indent=1)
    fh.write("\n")
PYEOF
"$VENV/bin/jupyter" kernelspec install --user --replace --name "$KERNEL_NAME" "$SPEC_DIR"
rm -rf "$SPEC_DIR"
trap - EXIT

if [[ "$KERNEL_ONLY" -eq 1 ]]; then
  echo "Kernel registered. In VS Code / Cursor: \"Select Kernel\" -> \"Jupyter Kernel...\" -> \"$KERNEL_DISPLAY\"."
  exit 0
fi

# --- JupyterLab, serving this directory --------------------------------------
"$VENV/bin/python" -c 'import jupyterlab' || die "jupyterlab is not installed in $VENV"
# JupyterLab saves its layout (open tabs, the file browser's folder) per user,
# by default in one directory that every JupyterLab server of the user shares.
# This launcher keeps its own. With the shared one the browser reopens the tabs
# of another project's notebooks, which do not exist here (the server then logs
# "404 ... No such file or directory"), and this directory's tabs replace theirs.
WORKSPACES="$("$VENV/bin/jupyter" --config-dir)/lab/workspaces-$KERNEL_NAME"
echo "Starting JupyterLab in $HERE (kernel: $KERNEL_DISPLAY)"
exec "$VENV/bin/jupyter" lab --ServerApp.root_dir="$HERE" --port "$PORT" \
  --MultiKernelManager.default_kernel_name="$KERNEL_NAME" \
  --LabApp.workspaces_dir="$WORKSPACES" "${LAB_ARGS[@]}"
