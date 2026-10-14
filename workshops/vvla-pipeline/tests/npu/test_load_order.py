# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""YOLO-pose and YOLO-detect land on the NPU when the detector is built first.

    source scripts/ryzen_ai_env.sh && pytest tests/npu

The NPU buffers a process's models share live as long as it keeps an NPU
session open, so each case runs in a fresh process. Needs the Ryzen AI runtime and
the compiled models in ``cache/``; skipped without them.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKSHOP = ROOT / "workshop" / "solution"


def _npu_ready() -> bool:
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    return "VitisAIExecutionProvider" in ort.get_available_providers() and all(
        (ROOT / "cache" / key).is_dir()
        for key in ("yolo26s_pose_fp32", "yolo26s_detect_fp32", "whisper_base_encoder", "whisper_base_decoder")
    )


pytestmark = pytest.mark.skipif(not _npu_ready(), reason="no Ryzen AI runtime or compiled NPU models")


def _providers(code: str, cwd: Path, path: Path) -> dict:
    """Run ``code`` in a fresh interpreter; it prints a JSON dict of providers."""
    out = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(path)!r})\n{code}"],
        cwd=cwd, capture_output=True, text=True, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_pipeline_detector_before_pose():
    got = _providers(
        "import json\n"
        "from vla_pipeline.utils.config import load_config\n"
        "from vla_pipeline.vision.yolo_detect_npu import ObjectDetector\n"
        "from vla_pipeline.vision.yolo_pose_npu import PoseEstimator\n"
        "cfg = load_config()\n"
        "det = ObjectDetector(cfg, device='npu')\n"
        "pose = PoseEstimator(cfg, device='npu')\n"
        "print(json.dumps({'detect': det.session.get_providers()[0], 'pose': pose.session.get_providers()[0]}))",
        ROOT, ROOT,
    )
    assert got == {"detect": "VitisAIExecutionProvider", "pose": "VitisAIExecutionProvider"}


def test_pipeline_whisper_after_both_yolos():
    got = _providers(
        "import json\n"
        "from vla_pipeline.utils.config import load_config\n"
        "from vla_pipeline.vision.yolo_pose_npu import PoseEstimator\n"
        "from vla_pipeline.vision.yolo_detect_npu import ObjectDetector\n"
        "from vla_pipeline.audio.whisper_npu import WhisperNPU\n"
        "cfg = load_config()\n"
        "cfg['whisper']['device'] = 'npu'\n"
        "pose, det = PoseEstimator(cfg), ObjectDetector(cfg)\n"  # main.py's order
        "w = WhisperNPU(cfg)\n"
        "print(json.dumps({'pose': pose.session.get_providers()[0], 'detect': det.session.get_providers()[0],\n"
        "                  'encoder': w.model.encoder.get_providers()[0], 'decoder': w.model.decoder.get_providers()[0]}))",
        ROOT, ROOT,
    )
    assert set(got.values()) == {"VitisAIExecutionProvider"}, got


def test_workshop_detector_before_pose():
    got = _providers(
        "import json\n"
        "from common.config import load_config\n"
        "from models.vision import ObjectDetector, PoseEstimator\n"
        "cfg = load_config()\n"
        "det = ObjectDetector(cfg, device='npu')\n"
        "pose = PoseEstimator(cfg, device='npu')\n"
        "print(json.dumps({'detect': det.provider, 'pose': pose.provider}))",
        WORKSHOP, WORKSHOP,
    )
    assert got == {"detect": "VitisAIExecutionProvider", "pose": "VitisAIExecutionProvider"}
