#!/usr/bin/env bash
# One hundred seeded pickups. Writes a timestamped directory with REPORT.md
# and the accumulated top and wrist videos.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export MUJOCO_GL=egl
exec .venv/bin/python -u -m so101_smolvla.vla_test random "$@"
