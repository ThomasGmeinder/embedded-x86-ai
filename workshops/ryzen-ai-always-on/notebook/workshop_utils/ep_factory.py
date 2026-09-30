# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""ONNX Runtime InferenceSession factories for CPU and Ryzen AI NPU (VitisAI EP).

Single entry point per backend so the notebook reads cleanly. Provides the
standard VitisAI provider_options shape.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
from pathlib import Path

import onnxruntime as ort


# Default VAIML config for FP32->BF16 NPU offload, derived from the upstream
# VitisAI YOLOv12 reference config (which reached ~100% AIE offload). Key flag:
# enable_f32_to_bf16_conversion lets VAIML accept an FP32 ONNX directly and
# convert to BF16 at compile time, without a separate quantization step.
DEFAULT_VAIML_CONFIG: dict = {
    "ai_analyzer_visualization": "true",
    "ai_analyzer_profiling": "true",
    "passes": [
        {"name": "init", "plugin": "vaip-pass_init"},
        {
            "name": "vaiml_partition",
            "plugin": "vaip-pass_vaiml_partition",
            "vaiml_config": {
                "keep_outputs": True,
                "optimize_level": 2,
                "aie_single_core_compiler": "peano",
                "enable_f32_to_bf16_conversion": True,
                "logging_level": "info",
                "fe_experiment": "small-tensor-threshold-unwrapping=0",
            },
        },
    ],
    "target": "VAIML",
    "targets": [
        {"name": "VAIML", "pass": ["init", "vaiml_partition"]},
    ],
}


def npu_session_options() -> ort.SessionOptions:
    """ORT options for an NPU session: 1 CPU thread, no spin-wait.

    The NPU does the compute, so ORT's CPU thread pool would otherwise spin one
    thread per core busy-waiting for it and peg the CPU for no useful work. This
    keeps results identical while cutting CPU dramatically.
    """
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    # Errors-only. Silences the harmless, expected VitisAI warning that ORT keeps
    # shape/decode ops on CPU ("VerifyEachNodeIsAssignedToAnEp ... assigns shape
    # related ops to CPU to improve perf"). It's not a problem, but a red warning
    # in a live workshop reads as "broken". Real errors still surface at level 3.
    so.log_severity_level = 3
    return so


@contextlib.contextmanager
def redirect_native_output(log_path: str | Path):
    """Redirect FD-level stdout/stderr to a log file for the duration of the block.

    The VAIML compile spawns a native aiecompiler subprocess that floods FDs 1/2
    with thousands of INFO lines. In a terminal those go to the console and are
    harmless. In a Jupyter kernel, FDs 1/2 are the ZMQ/IOPub pipe, and the flood
    back-pressures and deadlocks the blocking native call — the cold compile hangs.
    Python-level log settings (ort.set_default_logger_severity, log_severity_level)
    do not reach the native subprocess, so we redirect at the file-descriptor level
    with os.dup2. Output goes to a file (not devnull) so the full compile log is
    kept for debugging. Restores the original FDs on exit even if the block raises.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(log_fd)
        os.close(saved_out)
        os.close(saved_err)


def make_cpu_session(
    model_path: str | Path, intra_op_num_threads: int | None = None
) -> ort.InferenceSession:
    """Build an ORT session pinned to the CPU EP."""
    so = ort.SessionOptions()
    if intra_op_num_threads is not None:
        so.intra_op_num_threads = intra_op_num_threads
    return ort.InferenceSession(
        str(model_path),
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )


def make_npu_session(
    model_path: str | Path,
    cache_dir: str | Path,
    cache_key: str | None = None,
    config_path: str | Path | None = None,
    enable_profiling: bool = False,
    log_output: bool = True,
) -> ort.InferenceSession:
    """Build an ORT session on the Ryzen AI NPU via the VitisAI EP.

    First call for a given (model, cache_key) triggers VAIML compilation —
    this can take a minute. Subsequent calls hit the cache and are fast.

    Args:
        model_path: ONNX file (typically FP32; VAIML converts to BF16 at compile
            time via enable_f32_to_bf16_conversion).
        cache_dir: Root directory for the VAIML compile cache. Per-model artifacts
            live under `cache_dir/cache_key/`.
        cache_key: Cache subdir name. Defaults to the model file stem.
        config_path: Optional path to a VAIML config JSON. If None, writes the
            default config to `cache_dir/vitisai_config.json` and uses that.
        enable_profiling: Additionally enable ORT session profiling and pass the
            ai_analyzer_* hooks via provider_options. Note: the default
            vitisai_config.json already enables AI Analyzer profiling/visualization.
        log_output: If True (default), redirect the native VAIML compile flood to
            `cache_dir/cache_key/compile.log` so it can't deadlock a Jupyter kernel.
            Set False from a terminal to let the logs stream live to the console.
    """
    model_path = Path(model_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_key = cache_key or model_path.stem

    if config_path is None:
        config_path = cache_dir / "vitisai_config.json"
        with open(config_path, "w") as f:
            json.dump(DEFAULT_VAIML_CONFIG, f, indent=2)

    provider_options: dict = {
        "config_file": str(config_path),
        "cache_dir": str(cache_dir),
        "cache_key": resolved_key,
        "target": "VAIML",
    }
    if enable_profiling:
        provider_options["ai_analyzer_profiling"] = True
        provider_options["ai_analyzer_visualization"] = True

    so = npu_session_options()
    if enable_profiling:
        so.enable_profiling = True

    ctx = (
        redirect_native_output(cache_dir / resolved_key / "compile.log")
        if log_output
        else contextlib.nullcontext()
    )
    with ctx:
        return ort.InferenceSession(
            str(model_path),
            sess_options=so,
            providers=["VitisAIExecutionProvider"],
            provider_options=[provider_options],
        )


def list_providers() -> list[str]:
    """Convenience for the notebook setup cell."""
    return ort.get_available_providers()
