#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RYZEN_AI_VENV="${RYZEN_AI_VENV:-${RYZEN_AI_COMPILE_VENV:-}}"

if [[ -z "$RYZEN_AI_VENV" || ! -f "$RYZEN_AI_VENV/bin/activate" ]]; then
  echo "Set RYZEN_AI_VENV to the external full Ryzen AI SDK venv." >&2
  echo "Example: export RYZEN_AI_VENV=\$HOME/ryzen_ai-1.8.0/venv" >&2
  exit 2
fi

# AMD's activate script references optional unset variables.
set +u
# shellcheck disable=SC1090
source "$RYZEN_AI_VENV/bin/activate"
_SDK_FIX="$(cd "$RYZEN_AI_VENV/.." && pwd)/fix_activate.sh"
if [[ ! -f "$_SDK_FIX" ]]; then
  echo "Missing ${_SDK_FIX}" >&2
  echo "Ryzen AI's venv/bin/activate puts voe/lib ahead of system XRT and omits Peano." >&2
  echo "That correction is specific to the SDK install and is not part of this repo." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$_SDK_FIX"
set -u

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec python scripts/compile_npu_models.py "$@"
