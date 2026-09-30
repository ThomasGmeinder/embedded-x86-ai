#!/usr/bin/env bash

# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

# Ryzen AI NPU runtime environment (venv-only).
#
# The VitisAI ONNX Runtime EP needs its native libraries (libxcompiler-core-*,
# libonnxruntime_vitisai_ep.so, etc.) on LD_LIBRARY_PATH. The `voe` and
# `onnxruntime-vitisai` wheels install them into the venv but do NOT add their
# directories to the loader path, so the EP fails with
#   "libxcompiler-core-without-symbol.so: cannot open shared object file"
# and onnxruntime silently falls back to the CPU EP.
#
# Source this AFTER activating the venv:
#     source scripts/ryzen_ai_env.sh
#
# Safe to source repeatedly; a no-op if the libs aren't found.

# Resolve the active venv (VIRTUAL_ENV is set by `source .venv/bin/activate`).
_RAI_VENV="${VIRTUAL_ENV:-}"
if [[ -z "$_RAI_VENV" ]]; then
  _RAI_SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  _RAI_VENV="${_RAI_SELF}/.venv"
fi

_RAI_PYVER="$("${_RAI_VENV}/bin/python" -c 'import sys;print(f"python{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo python3.12)"
_RAI_SP="${_RAI_VENV}/lib/${_RAI_PYVER}/site-packages"

# Directories inside the venv that hold the EP's native .so files.
_RAI_CANDIDATES=(
  "${_RAI_SP}/voe/lib"             # libxcompiler-core-*, flexml-runner, etc.
  "${_RAI_SP}/flexmlrt/lib"        # libflexmlrt.so - needed to RUN compiled VAIML models
  "${_RAI_SP}/onnxruntime/capi"    # libonnxruntime_vitisai_ep.so + providers
)

for _d in "${_RAI_CANDIDATES[@]}"; do
  if [[ -d "$_d" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":$_d:"*) ;;                                  # already present
      *) export LD_LIBRARY_PATH="${_d}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
  fi
done

# Strix (STX) NPU firmware xclbin ships inside the flexml wheel. XLNX_VART_FIRMWARE
# must point at a specific .xclbin FILE (not the directory), else the EP errors
# with "xclbin is set to a directory".
_RAI_XCLBIN_DIR="${_RAI_SP}/flexml/flexml_extras/data/ryzen-ai/stx"
if [[ -z "${XLNX_VART_FIRMWARE:-}" && -d "$_RAI_XCLBIN_DIR" ]]; then
  if [[ -n "${RAI_XCLBIN:-}" && -f "$RAI_XCLBIN" ]]; then
    export XLNX_VART_FIRMWARE="$RAI_XCLBIN"
  else
    # Strix Halo uses the 2x4x4 overlay; fall back to 4x4 then anything.
    for _x in unified-2x4x4.xclbin unified-4x4.xclbin; do
      if [[ -f "${_RAI_XCLBIN_DIR}/${_x}" ]]; then
        export XLNX_VART_FIRMWARE="${_RAI_XCLBIN_DIR}/${_x}"
        break
      fi
    done
  fi
  if [[ -z "${XLNX_VART_FIRMWARE:-}" ]]; then
    _x=$(ls "${_RAI_XCLBIN_DIR}"/*.xclbin 2>/dev/null | head -1 || true)
    [[ -n "$_x" ]] && export XLNX_VART_FIRMWARE="$_x"
  fi
fi

if [[ ! -e "${_RAI_SP}/voe/lib/libxcompiler-core-without-symbol.so" ]]; then
  echo "[ryzen_ai_env] warning: VitisAI native libs not found in venv voe/lib;" >&2
  echo "[ryzen_ai_env] the NPU EP will fall back to CPU. Ensure the 'voe' wheel installed." >&2
fi

unset _RAI_VENV _RAI_SELF _RAI_PYVER _RAI_SP _RAI_CANDIDATES _d _RAI_XCLBIN_DIR _x

# llama.cpp is built for gfx1100 on Strix (gfx1150/gfx1151); present the
# iGPU as gfx1100 so its HIP kernels load instead of segfaulting. Affects
# only the ROCm/HIP GPU runtime - the XDNA2 NPU (VitisAI) is unaffected.
export HSA_OVERRIDE_GFX_VERSION=11.0.0
