#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Python environment for the AMD Physical AI Workshop.
# Installs PyTorch (ROCm wheel) FIRST so the GPU build is in place, then Genesis
# and JupyterLab. Genesis reuses the already-installed GPU torch (pip will not
# downgrade it to a CPU build because the torch lines are filtered out below).
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR="${BASE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_DIR="${ENV_DIR:-$BASE_DIR/rocm-env}"

# PyTorch ROCm wheel index — must match the machine's ROCm major.minor.
# Override with:  TORCH_ROCM_INDEX=rocm6.3 bash scripts/setup_env.sh
TORCH_ROCM_INDEX="${TORCH_ROCM_INDEX:-rocm7.2}"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

echo "===== Python Environment Setup ====="
echo "ENV_DIR=$ENV_DIR"
echo "Torch wheel index = https://download.pytorch.org/whl/$TORCH_ROCM_INDEX"

# Auto-install Python 3.10 if missing. The ROCm torch wheels are cp310 builds, so
# 3.10 is required — rather than fail, provision it (deadsnakes PPA) the same way
# install.sh provisions lld/ffmpeg. Skips if PYTHON_BIN was overridden to another
# interpreter that already exists.
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "  ! $PYTHON_BIN not found — installing it automatically..."
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.10 python3.10-venv python3.10-dev
fi
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "ERROR: $PYTHON_BIN still not found after auto-install."
    echo "  Install it manually, then re-run install.sh:"
    echo "    sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update"
    echo "    sudo apt install -y python3.10 python3.10-venv python3.10-dev"
    exit 1
fi
echo "Using Python: $PYTHON_BIN"

# ── Create venv ──────────────────────────────────────────────────────────────
"$PYTHON_BIN" -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# ── PyTorch (ROCm / GPU build) ───────────────────────────────────────────────
echo "Installing PyTorch ROCm build (GPU)..."
pip install torch torchvision torchaudio \
  --index-url "https://download.pytorch.org/whl/$TORCH_ROCM_INDEX"

# ── JupyterLab + Genesis + remaining deps ────────────────────────────────────
echo "Installing JupyterLab, Genesis and workshop requirements..."
# Drop torch + triton lines: the ROCm GPU torch wheel above already pulls in the
# matching pytorch-triton-rocm. The public PyPI has no standalone triton-rocm, and
# reinstalling torch from PyPI would overwrite the GPU build with a CPU wheel.
grep -viE '^(torch|torchvision|torchaudio|triton|triton-rocm|pytorch-triton-rocm)([=<>! ]|$)' "$BASE_DIR/requirements.txt" > /tmp/requirements_no_torch.txt
pip install --no-cache-dir -r /tmp/requirements_no_torch.txt

echo "===== Environment setup complete ====="
echo "Verify GPU + Genesis with:"
echo "  $ENV_DIR/bin/python -c \"import torch; print('GPU:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')\""
