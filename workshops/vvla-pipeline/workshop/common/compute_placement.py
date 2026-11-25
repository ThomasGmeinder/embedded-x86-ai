# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The compute placement table (PROVIDED), built from the settings that decide it.

Which model does a job and which processor it runs on are settings, not text:

* the voice model is ``voice_intent`` in the pipeline's ``config/pipeline.yaml``,
  with that model's own ``name``, ``does`` and ``n_gpu_layers``;
* the two YOLO models run where ``yolo_pose.device`` and ``yolo_detect.device``
  of the workshop's config say.

``placement(cfg)`` reads them and returns a table a notebook cell shows by
ending with it: Jupyter asks the object for Markdown, so the table follows the
notebook's theme. Change a setting, run the cell again, and the table says so.
"""

from __future__ import annotations

from dataclasses import dataclass

COLUMNS = ("Task", "Model or algorithm", "What it does", "Runs on", "Runtime")
# What a `device:` setting means: the processor, and the ONNX Runtime provider behind it.
DEVICES = {"npu": ("NPU", "ONNX Runtime, VitisAI EP"), "cpu": ("CPU", "ONNX Runtime, CPU EP")}


@dataclass(frozen=True)
class Row:
    task: str
    model: str
    does: str
    runs_on: str
    runtime: str


class Placement:
    """Rows of the table. A notebook shows it as Markdown; ``print`` shows the same text."""

    def __init__(self, rows: list[Row]):
        self.rows = rows

    def _repr_markdown_(self) -> str:
        lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
        previous = None
        for row in self.rows:
            task = f"**{row.task}**" if row.task != previous else ""  # a task is named once
            previous = row.task
            lines.append("| " + " | ".join((task, row.model, row.does, row.runs_on, row.runtime)) + " |")
        return "\n".join(lines)

    __str__ = _repr_markdown_


def _onnx(section: dict) -> tuple[str, str]:
    """``(processor, runtime)`` of a model whose section has ``device: npu | cpu``."""
    return DEVICES[str(section["device"]).lower()]


def placement(cfg: dict, pipeline_cfg: dict | None = None) -> Placement:
    """The table for the workshop's config ``cfg``; the voice model comes from the pipeline's
    config (``pipeline_cfg``, by default the file ``./run_pipeline.sh`` reads)."""
    if pipeline_cfg is None:
        from vla_pipeline.utils.config import load_config

        pipeline_cfg = load_config()
    voice = pipeline_cfg[pipeline_cfg["voice_intent"]]
    voice_on = "iGPU" if int(voice.get("n_gpu_layers", 0)) > 0 else "CPU"
    pose_on, pose_runtime = _onnx(cfg["yolo_pose"])
    detect_on, detect_runtime = _onnx(cfg["yolo_detect"])
    return Placement([
        Row("Voice control", "Silero voice-activity detection", "finds speech in the microphone audio",
            "CPU", "ONNX Runtime"),
        Row("Voice control", str(voice["name"]), str(voice["does"]), voice_on, "`llama.cpp` `llama-server` (HIP)"),
        Row("Gestures", "YOLOv26s-pose", "finds your body and wrist", pose_on, pose_runtime),
        Row("Gestures", "MediaPipe Hands", "detects the hand gesture (pinch, roll)", "CPU", "MediaPipe Tasks"),
        Row("Object localization", "YOLOv26s-detect", "finds the named object in the arm camera",
            detect_on, detect_runtime),
        Row("Digital twin", "MuJoCo 3D simulation", "simulates the arm and the table scene",
            "CPU (rendering on the iGPU)", "MuJoCo, EGL"),
    ])
