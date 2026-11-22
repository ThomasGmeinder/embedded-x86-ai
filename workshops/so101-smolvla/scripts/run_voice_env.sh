#!/usr/bin/env bash
# Runs a command in the voice workshop's environment (workshops/vvla-pipeline):
# its Python (.venv/bin/activate) and its environment script
# (scripts/ryzen_ai_env.sh), the way the notebooks' kernel starts. The
# command runs where this script was called; it does not change directory.
#
#   scripts/run_voice_env.sh python src/so101_smolvla/voice_worker.py <fd> <fd>
#   scripts/run_voice_env.sh python -m ipykernel_launcher -f <file>
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[[ -d "$here/../../vvla-pipeline" ]] || die "the voice workshop is not at $here/../../vvla-pipeline"
voice_workshop="$(cd "$here/../../vvla-pipeline" && pwd)"
[[ -x "$voice_workshop/.venv/bin/python" ]] || die "no Python at $voice_workshop/.venv/bin/python - run ./bootstrap.sh in $voice_workshop first"
[[ -f "$voice_workshop/.venv/bin/activate" ]] || die "missing $voice_workshop/.venv/bin/activate"
[[ -f "$voice_workshop/scripts/ryzen_ai_env.sh" ]] || die "missing $voice_workshop/scripts/ryzen_ai_env.sh"
[[ $# -ge 1 ]] || die "no command given (see the comment at the top of $0)"

# shellcheck disable=SC1091
source "$voice_workshop/.venv/bin/activate"
# shellcheck disable=SC1091
source "$voice_workshop/scripts/ryzen_ai_env.sh"
exec "$@"
