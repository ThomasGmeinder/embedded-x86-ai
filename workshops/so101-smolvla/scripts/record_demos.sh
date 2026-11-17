#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
export MUJOCO_GL=egl
exec .venv/bin/python -m so101_smolvla.record
