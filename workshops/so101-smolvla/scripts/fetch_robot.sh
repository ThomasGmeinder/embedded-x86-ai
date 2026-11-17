#!/usr/bin/env bash
# Assets are already vendored. This script records the pins.
set -euo pipefail
cd "$(dirname "$0")/.."
cat assets/PINS.txt
