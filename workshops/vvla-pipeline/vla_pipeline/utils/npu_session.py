# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ONNX Runtime session factory for the Ryzen AI NPU (VitisAIExecutionProvider).

All three NPU models (Whisper-base, YOLOv26s-pose, and YOLOv26s object
detection) build their sessions through :func:`build_session` so the VitisAI
compiler config, cache directory, and provider verification live in exactly one
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


def active_provider(session: ort.InferenceSession) -> str:
    """Return the highest-priority provider active on a session."""
    providers = session.get_providers()
    return providers[0] if providers else "none"


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
        device: ``"npu"`` or ``"cpu"``. NPU requests fail closed when VitisAI
            is unavailable or fails to load. Select ``"cpu"`` explicitly for
            machines without the Ryzen AI stack.
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
        raise RuntimeError(
            "VitisAIExecutionProvider not available in this onnxruntime build - "
            "install the Ryzen AI onnxruntime-vitisai wheel (see "
            "bootstrap.sh / RYZEN_AI_WHEELS), or select device: cpu."
        )

    if device == "cpu":
        return ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    cache = resolve(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    logger.info(
        "Building NPU session for %s (cache hit ~1s, first compile can take 30-60 min)",
        onnx_path.name,
    )
    session = ort.InferenceSession(
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
    if "VitisAIExecutionProvider" not in session.get_providers():
        raise RuntimeError(
            f"VitisAI session for {onnx_path.name} fell back to "
            f"{session.get_providers()}; check scripts/verify_npu_stack.py."
        )
    logger.info("%s active provider: %s", onnx_path.name, active_provider(session))
    return session
