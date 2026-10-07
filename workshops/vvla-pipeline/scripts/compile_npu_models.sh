#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RYZEN_AI_VENV="${RYZEN_AI_VENV:-${RYZEN_AI_COMPILE_VENV:-}}"
if [[ -z "$RYZEN_AI_VENV" && -n "${RYZEN_AI_WHEELS:-}" ]]; then
  RYZEN_AI_VENV="${RYZEN_AI_WHEELS}/venv"
fi

if [[ -z "$RYZEN_AI_VENV" || ! -f "$RYZEN_AI_VENV/bin/activate" ]]; then
  echo "Set RYZEN_AI_WHEELS to the Ryzen AI 1.7.1 install that contains venv/." >&2
  echo "Example: export RYZEN_AI_WHEELS=\$HOME/ryzen_ai-1.7.1" >&2
  exit 2
fi

# AMD's activate script references optional unset variables.
set +u
# shellcheck disable=SC1090
source "$RYZEN_AI_VENV/bin/activate"
if [[ -z "${XILINX_XRT:-}" && -f /opt/xilinx/xrt/setup.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/xilinx/xrt/setup.sh
fi
set -u

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/compile_npu_models.py "$@"
