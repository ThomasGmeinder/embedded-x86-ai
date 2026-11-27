#!/usr/bin/env bash
# =============================================================================
# run_notebooks.sh - open the notebooks of this directory.
#
# Every notebook here runs in one Jupyter kernel:
#     name     so101-smolvla-notebooks
#     shown as "SO-101 SmolVLA notebooks"
# The kernel is the workshop's own Python (../.venv) with the workshop's
# package on its path (PYTHONPATH=../src), the way the workshop's scripts run
# it. It needs ipykernel and JupyterLab in that environment:
# ../scripts/setup_voice.sh installs them. The notebooks name this kernel, so
# it is preselected.
#
# Before anything else it puts the SAPIEN twin back when a restart of the
# machine removed it (../scripts/setup_sapien_env.sh), in every mode below.
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
#      "Select Kernel" -> "Jupyter Kernel..." -> "SO-101 SmolVLA notebooks".
#   b) ./run_notebooks.sh --no-browser, then "Select Kernel" ->
#      "Existing Jupyter Server..." and paste the printed address. The same
#      kernel is offered there.
# =============================================================================
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSHOP="$(cd "$HERE/.." && pwd)"
KERNEL_NAME="so101-smolvla-notebooks"
KERNEL_DISPLAY="SO-101 SmolVLA notebooks"
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

# --- Readiness: the workshop's Python with the kernel in it -------------------
VENV="$WORKSHOP/.venv"
[[ -x "$VENV/bin/python" ]] || die "no Python at $VENV/bin/python - create .venv as README.md, Environment, says"
"$VENV/bin/python" -c 'import ipykernel, jupyter_client' \
  || die "ipykernel / jupyter_client are not installed in $VENV - run $WORKSHOP/scripts/setup_voice.sh"

# --- The SAPIEN twin: restored when a restart removed it ---------------------
# The simulator's source is checked out under /tmp, which a restart of the
# machine empties, and its environment (.venv-sapien) holds only a pointer to
# that checkout. The two paths are the ones the notebooks check
# (so101_smolvla.sapien_config); when either is missing the setup script puts
# both back (it needs the network for the clone).
SAPIEN_PATHS="$(PYTHONPATH="$WORKSHOP/src" "$VENV/bin/python" -c \
  'from so101_smolvla.sapien_config import SAPIEN_PYTHON, SAPIEN_SOURCE; print(SAPIEN_SOURCE); print(SAPIEN_PYTHON)')" \
  || die "could not read the SAPIEN twin's paths from so101_smolvla.sapien_config"
{ read -r SAPIEN_SOURCE_DIR; read -r SAPIEN_PYTHON_BIN; } <<<"$SAPIEN_PATHS"
if [[ ! -d "$SAPIEN_SOURCE_DIR/.git" || ! -x "$SAPIEN_PYTHON_BIN" ]]; then
  echo "The SAPIEN twin is missing ($SAPIEN_SOURCE_DIR or $SAPIEN_PYTHON_BIN); running scripts/setup_sapien_env.sh"
  SAPIEN_SOURCE="$SAPIEN_SOURCE_DIR" "$WORKSHOP/scripts/setup_sapien_env.sh" \
    || die "scripts/setup_sapien_env.sh failed - the pick-and-place notebooks need the SAPIEN twin"
fi

# --- The kernel: the workshop's Python, its package on the path --------------
# The start command is the interpreter itself, so interrupting a cell reaches
# the process Jupyter started; it does not change directory, so a notebook
# starts where it lies.
SPEC_DIR="$(mktemp -d)"
trap 'rm -rf "$SPEC_DIR"' EXIT
"$VENV/bin/python" - "$SPEC_DIR/kernel.json" "$KERNEL_NAME" "$KERNEL_DISPLAY" "$VENV/bin/python" "$WORKSHOP/src" <<'PYEOF'
import json
import sys

path, name, display, python, source = sys.argv[1:]
spec = {
    "argv": [python, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": display,
    "language": "python",
    "env": {"PYTHONPATH": source},
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
"$VENV/bin/python" -c 'import jupyterlab' || die "jupyterlab is not installed in $VENV - run $WORKSHOP/scripts/setup_voice.sh"
# JupyterLab saves its layout (open tabs, the file browser's folder) per user,
# by default in one directory that every JupyterLab server of the user shares.
# This launcher keeps its own. With the shared one the browser reopens the tabs
# of another project's notebooks, which do not exist here (the server then logs
# "404 ... No such file or directory"), and this directory's tabs replace theirs.
WORKSPACES="$("$VENV/bin/jupyter" --config-dir)/lab/workspaces-$KERNEL_NAME"
echo "Starting JupyterLab in $HERE (kernel: $KERNEL_DISPLAY)"
# JupyterLab prints its address before it listens. The workshop's own server
# extension (so101_smolvla.notebook_server) logs one line once it does:
# "The notebooks' server listens on port N". A program that connects waits
# for that line. The extension is found on the workshop's package path.
export PYTHONPATH="$WORKSHOP/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$VENV/bin/jupyter" lab --ServerApp.root_dir="$HERE" --port "$PORT" \
  --MultiKernelManager.default_kernel_name="$KERNEL_NAME" \
  --LabApp.workspaces_dir="$WORKSPACES" \
  --ServerApp.jpserver_extensions="{'so101_smolvla.notebook_server': True}" "${LAB_ARGS[@]}"
