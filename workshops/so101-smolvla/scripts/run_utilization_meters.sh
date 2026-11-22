#!/usr/bin/env bash
# The utilization and power table (so101_smolvla.utilization_window) in the workshop's Python, which has Qt.
# Started by open_meters(), from a shell, or on a remote machine through
#   ssh -o ForwardX11=yes -o ForwardX11Trusted=yes <machine> <checkout>/workshops/so101-smolvla/scripts/run_utilization_meters.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
exec .venv/bin/python -m so101_smolvla.utilization_window "$@"
