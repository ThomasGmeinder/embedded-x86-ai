#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

exec .venv/bin/python -u -m so101_smolvla.sapien_rollout "$@"
