#!/usr/bin/env bash
# Seeded SAPIEN stacking episodes with task-specific reports and videos.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec .venv/bin/python -u -m so101_smolvla.sapien_rollout "$@"
