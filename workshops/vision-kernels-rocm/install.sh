#!/bin/bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Vision Kernels (ROCm) — standalone installer
#
# Assumes ROCm is ALREADY installed (run scripts/setup_rocm.sh first if not).
#
# It only:
#   1. Builds a Python venv + installs PyTorch (ROCm wheel) + Genesis
#   2. Symlinks a ROCm-matched ld.lld + installs ffmpeg (Genesis GPU JIT needs
#      these — without the version-matched linker, gs.init() fails with
#      "ld.lld: error: unknown abi version")
#
# Launch the notebook yourself with:
#   source .venv/bin/activate && jupyter lab ROCm_Physical_AI_Agent_Workshop.ipynb
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BASE_DIR
export ENV_DIR="${ENV_DIR:-$BASE_DIR/.venv}"

echo "===== Vision Kernels (ROCm) — install ====="
echo "BASE_DIR=$BASE_DIR"

# ── Detect ROCm ──────────────────────────────────────────────────────────────
ROCM_PATH="${ROCM_PATH:-$(ls -d /opt/rocm* 2>/dev/null | sort -V | tail -1)}"
if [ -z "$ROCM_PATH" ] || [ ! -d "$ROCM_PATH" ]; then
    echo "  ! No ROCm found under /opt/rocm*. Install it first:"
    echo "      bash scripts/setup_rocm.sh   # (Ubuntu 24.04; installs ROCm 7.2, reboots)"
    echo "    Continuing — Genesis will fall back to CPU without a GPU ROCm stack."
    ROCM_PATH="/opt/rocm"
else
    echo "  OK ROCm at: $ROCM_PATH"
fi
export ROCM_PATH

# ── STEP 1: Python env (venv + torch-rocm + Genesis) ─────────────────────────
echo "===== STEP 1: Python environment ====="
bash "$BASE_DIR/scripts/setup_env.sh"

# ── STEP 2: ld.lld (ROCm-matched) + ffmpeg for Genesis GPU JIT ───────────────
# Genesis JITs its physics kernels to AMDGPU code objects and links them with
# ld.lld. That linker MUST match the ROCm LLVM that produced the objects. apt's
# lld-18 is too OLD for ROCm 7.2.x objects and fails at link time with
#   ld.lld: error: unknown abi version:
# which surfaces in the notebook as a RuntimeError from gs.init(). So symlink
# ROCm's OWN bundled ld.lld (version-matched) into /usr/local/bin (first on PATH).
echo "===== STEP 2: ld.lld + ffmpeg (Genesis GPU JIT) ====="
sudo apt-get install -y ffmpeg > /dev/null 2>&1 || true
ROCM_LLD=""
for d in $(ls -d /opt/rocm* 2>/dev/null | sort -Vr); do
    if [ -x "$d/lib/llvm/bin/ld.lld" ]; then ROCM_LLD="$d/lib/llvm/bin/ld.lld"; break; fi
done
if [ -n "$ROCM_LLD" ]; then
    sudo ln -sf "$ROCM_LLD" /usr/local/bin/ld.lld
    echo "  OK ld.lld -> $ROCM_LLD ($("$ROCM_LLD" --version 2>/dev/null | head -1))"
else
    sudo apt-get install -y lld-18 > /dev/null 2>&1 || true
    if [ -f /usr/lib/llvm-18/bin/ld.lld ]; then
        sudo ln -sf /usr/lib/llvm-18/bin/ld.lld /usr/local/bin/ld.lld
        echo "  ! WARNING: no ROCm-bundled ld.lld found; fell back to apt lld-18 —"
        echo "    Genesis GPU JIT may fail on ROCm 7.2.x objects (unknown abi version)."
    else
        echo "  X WARNING: ld.lld not found; Genesis GPU JIT may fail."
    fi
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "===== Install complete ====="
echo "Strix Halo (gfx1151) needs an HSA gfx override. Launch the lab with:"
echo ""
echo "  source $ENV_DIR/bin/activate"
echo "  HSA_OVERRIDE_GFX_VERSION=11.0.0 jupyter lab ROCm_Physical_AI_Agent_Workshop.ipynb"
echo ""
