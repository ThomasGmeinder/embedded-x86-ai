# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODOs 1.1-1.3: the NPU session factory - the centerpiece of the workshop.

Every ONNX model the pipeline runs on the XDNA2 NPU goes through ONE
function, ``build_npu_session``. The two classic Ryzen AI failure modes both
start here:

1. **Silent CPU fallback.** Requesting the NPU is not the same as getting
   it. Always verify which execution provider a session *actually* got
   (``active_provider``) - a model that quietly landed on CPU still works,
   just 10-30x slower.
2. **Load order.** The models on the NPU share buffers within a process,
   created by its first NPU session and freed when the last one closes. Open
   YOLO-pose first and keep it open, so YOLO-detect can load after it.

Compiled NPU artifacts are keyed by ``cache_dir + cache_key``: the first
compile can take tens of minutes, then every load is ~1 s from the cache.
Each model MUST use a unique ``cache_key``.

Reference: ``vla_pipeline/utils/npu_session.py`` (same names and parameters).
Notebook: ``notebooks/01_ai_models/01_npu_yolo.ipynb``
Self-test: ``python -m vvla_workshop.selftest_models``
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)


def npu_available() -> bool:
    """True if this onnxruntime build exposes the VitisAI execution provider.

    The stock ``pip install onnxruntime`` does NOT include it - only the
    Ryzen AI build (installed from the Ryzen AI wheel directory) does.
    """
    # >>> TODO 1.1: npu_available - notebooks/01_ai_models/01_npu_yolo.ipynb
    raise NotImplementedError(
        "TODO 1.1 (npu_available): implement me - see notebooks/01_ai_models/01_npu_yolo.ipynb")
    # <<< TODO 1.1


def active_provider(session: ort.InferenceSession) -> str:
    """The execution provider this session ACTUALLY got (first = active).

    Requesting the NPU is a wish; this is the receipt. Check it after every
    session build - a silent CPU fallback looks identical until you profile.
    """
    # >>> TODO 1.3: active_provider - notebooks/01_ai_models/01_npu_yolo.ipynb
    raise NotImplementedError(
        "TODO 1.3 (active_provider): implement me - see notebooks/01_ai_models/01_npu_yolo.ipynb")
    # <<< TODO 1.3


def build_npu_session(
    onnx_path: str | Path,
    device: str = "npu",
    *,
    vitisai_config: str | Path = "config/vaiep_config.json",
    cache_dir: str | Path = "cache",
    cache_key: str = "model",
) -> ort.InferenceSession:
    """Create an ``InferenceSession`` on the NPU (VitisAI EP) or the CPU.

    Args:
        onnx_path: Path to the FP32 ONNX model (resolved against the repo root).
        device: ``"npu"`` or ``"cpu"``. ``"npu"`` must fall back to CPU (with
            a warning) when the VitisAI EP is not present, so every component
            still runs on machines without the Ryzen AI stack.
        vitisai_config: VitisAI compiler pass configuration (JSON) - selects
            the VAIML partitioning pass and BF16 conversion.
        cache_dir: NPU compile cache directory (shared with the pipeline so
            nothing recompiles).
        cache_key: Unique cache key for THIS model. Two models sharing a key
            corrupt each other's compiled artifacts.
    """
    onnx_path = resolve(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found - run ./bootstrap.sh (model export step) first."
        )
    # >>> TODO 1.2: build_npu_session - notebooks/01_ai_models/01_npu_yolo.ipynb
    raise NotImplementedError(
        "TODO 1.2 (build_npu_session): implement me - see notebooks/01_ai_models/01_npu_yolo.ipynb")
    # <<< TODO 1.2
