# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ONNX Runtime session factory for the Ryzen AI NPU (VitisAIExecutionProvider).

All three NPU models (Whisper-base, YOLOv26s-pose, and YOLOv26s object
detection) build their sessions through :func:`build_session` so the VitisAI
compiler config, cache directory, and CPU fallback behavior live in exactly one
place.

Notes for Ryzen AI SW 1.7.1 on Linux (Strix Halo / XDNA2):
- ``onnxruntime`` must be the Ryzen AI build that ships
  ``VitisAIExecutionProvider`` (installed by ``bootstrap.sh`` from the
  Ryzen AI wheel directory, see ``RYZEN_AI_WHEELS``).
- The first compile of a model for the NPU can take tens of minutes; the
  result is cached under ``cache_dir/cache_key`` and reloads in ~1 s.
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnxruntime as ort

from vla_pipeline.utils.config import resolve

logger = logging.getLogger(__name__)


def npu_available() -> bool:
    """True if this onnxruntime build exposes the VitisAI EP."""
    return "VitisAIExecutionProvider" in ort.get_available_providers()


def build_session(
    onnx_path: str | Path,
    device: str = "npu",
    *,
    vitisai_config: str | Path = "config/vaiep_config.json",
    cache_dir: str | Path = "cache",
    cache_key: str = "model",
) -> ort.InferenceSession:
    """Create an InferenceSession on the NPU (VitisAI EP) or CPU.

    Args:
        onnx_path: Path to the FP32 ONNX model.
        device: ``"npu"`` or ``"cpu"``. ``"npu"`` silently falls back to CPU
            (with a warning) when the VitisAI EP is not present, so every
            component still runs on machines without the Ryzen AI stack.
        vitisai_config: VitisAI compiler pass configuration (JSON).
        cache_dir: NPU compile cache directory.
        cache_key: Unique cache key per model.
    """
    onnx_path = resolve(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found - run ./bootstrap.sh (model export step) first."
        )

    if device == "npu" and not npu_available():
        logger.warning(
            "VitisAIExecutionProvider not available in this onnxruntime build - "
            "falling back to CPU. Install the Ryzen AI 1.7.1 onnxruntime wheel "
            "(see bootstrap.sh / RYZEN_AI_WHEELS) for NPU execution."
        )
        device = "cpu"

    if device == "cpu":
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    cache = resolve(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building NPU session for %s (cache hit ~1s, first compile can take 30-60 min)",
        onnx_path.name,
    )
    return ort.InferenceSession(
        str(onnx_path),
        providers=["VitisAIExecutionProvider"],
        provider_options=[
            {
                "config_file": str(resolve(vitisai_config)),
                "cache_dir": str(cache),
                "cache_key": cache_key,
                "target": "VAIML",
            }
        ],
    )
