#!/usr/bin/env python3
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
"""Compile a model for the Ryzen AI NPU from the terminal.

Triggers the VAIML FP32->BF16 compile and pre-warms the VitisAI cache outside
of the Jupyter notebook, so you can watch the compile logs stream live and so
the notebook's NPU section becomes an instant cache hit afterwards.

Reuses the project's proven compile path (`workshop_utils.ep_factory
.make_npu_session`) rather than reinventing the VitisAI provider_options.

Usage:
    # Compile an existing ONNX (cache_key defaults to <stem>_fp32, matching the
    # notebook's CACHE_KEY, so the notebook then hits this exact cache):
    python compile.py notebook/yolo26s-pose.onnx
    python compile.py --model notebook/yolo26m-pose.onnx --cache-key yolo26m-pose_fp32

    # Export from an ultralytics checkpoint first, then compile it:
    python compile.py --export yolo26m           # detection
    python compile.py --export yolo26s-pose      # pose

    # Bring your own VAIML config JSON (otherwise the default one is written):
    python compile.py notebook/yolo26s-pose.onnx --config my_vaiml_config.json

    # Tee the compile logs to a file as well:
    python compile.py --export yolo26m 2>&1 | tee cache/compile.log

Prereqs (same env the notebook needs): the Ryzen AI venv activated and XRT on
the environment, otherwise the VitisAI EP silently falls back to CPU. Example:

    export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:$HOME/ryzenai/venv/onnxruntime/lib/:${LD_LIBRARY_PATH:-}"
    source /opt/xilinx/xrt/setup.sh
    source $HOME/ryzenai/venv/bin/activate
    sudo xrt-smi configure --pmode turbo   # for an honest NPU number

Note: the first compile per (model, cache_key) can take minutes (up to ~60 min
for larger models per cache/README.md). Subsequent runs hit the VAIML cache and
build in ~1s.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "false")

import numpy as np
import onnxruntime

HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(HERE / "notebook"))

from workshop_utils.ep_factory import make_npu_session  # noqa: E402

CACHE_DIR = HERE / "cache"
ONNX_DIR = HERE / "notebook"


def export_onnx(model_name: str, imgsz: int = 640) -> Path:
    """Download + export an ultralytics YOLO26 checkpoint to ONNX.

    Uses the same export params as the notebook (opset 17, 640x640, NMS-free,
    static batch=1). Skips the export if the target ONNX already exists.
    """
    onnx_path = ONNX_DIR / f"{model_name}.onnx"
    if onnx_path.exists():
        print(f"ONNX exists: {onnx_path} (skipping export)")
        return onnx_path

    from ultralytics import YOLO

    print(f"Exporting {model_name}.pt -> ONNX (imgsz={imgsz}, opset 17)...")
    exported = YOLO(f"{model_name}.pt").export(
        format="onnx",
        imgsz=imgsz,
        opset=17,
        simplify=True,
        dynamic=False,
        batch=1,
        nms=False,  # NMS-free one-to-one head -> [1, 300, 57]
    )
    exported = Path(exported)
    if exported.resolve() != onnx_path.resolve():
        exported.replace(onnx_path)
    print(f"Exported -> {onnx_path}\n")
    return onnx_path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "model",
        nargs="?",
        help="Path to an existing .onnx file to compile.",
    )
    ap.add_argument(
        "--model",
        dest="model_opt",
        help="Path to an existing .onnx file to compile (same as the positional arg).",
    )
    ap.add_argument(
        "--export",
        help="Ultralytics checkpoint to export first, e.g. 'yolo26m' or "
        "'yolo26s-pose'. Exports to notebook/<name>.onnx then compiles it.",
    )
    ap.add_argument(
        "--cache-key",
        help="VAIML cache subdir name. Defaults to the ONNX file stem.",
    )
    ap.add_argument(
        "--config",
        help="Path to a VAIML config JSON. If omitted, make_npu_session writes "
        "the default FP32->BF16 config to cache/vitisai_config.json.",
    )
    ap.add_argument(
        "--imgsz", type=int, default=640, help="Export image size (with --export)."
    )
    args = ap.parse_args()

    providers = onnxruntime.get_available_providers()
    print(f"== Providers: {providers}")
    if "VitisAIExecutionProvider" not in providers:
        print("\n!! VitisAIExecutionProvider NOT available — you'd compile/run on CPU.")
        print("!! Activate the Ryzen AI venv and source the XRT env, then retry.")
        sys.exit(1)

    # Resolve the ONNX to compile: either export from a checkpoint or use a path.
    model_arg = args.model_opt or args.model
    if args.export:
        onnx_path = export_onnx(args.export, args.imgsz)
    elif model_arg:
        onnx_path = Path(model_arg).resolve()
        if not onnx_path.exists():
            print(f"ERROR: ONNX file not found: {onnx_path}")
            sys.exit(1)
        if onnx_path.suffix.lower() != ".onnx":
            print(f"ERROR: expected a .onnx file, got: {onnx_path}")
            sys.exit(1)
    else:
        print(
            "ERROR: provide a model to compile — either a path to an .onnx "
            "file or --export <checkpoint>."
        )
        print("       e.g.  python compile.py notebook/yolo26s-pose.onnx")
        print("             python compile.py --export yolo26m")
        sys.exit(1)

    # Default to the notebook's cache_key convention (<stem>_fp32) so a terminal
    # pre-warm lands in the exact folder the notebook reads. The notebook uses
    # CACHE_KEY = f"{MODEL_NAME}_fp32"; matching it here avoids a silent cache miss.
    cache_key = args.cache_key or f"{onnx_path.stem}_fp32"
    config_path = Path(args.config).resolve() if args.config else None

    print(f"== Model    : {onnx_path}")
    print(f"== Cache    : {CACHE_DIR / cache_key}")
    print(
        f"== Config   : {config_path if config_path else '(default, written by make_npu_session)'}"
    )
    print()
    print(
        "Building NPU session (first compile can take minutes — up to ~60 min "
        "for larger models; subsequent runs hit the cache in ~1s)...\n"
    )

    t0 = time.perf_counter()
    session = make_npu_session(
        onnx_path,
        CACHE_DIR,
        cache_key=cache_key,
        config_path=config_path,
        log_output=False,  # terminal: stream the VAIML logs live to the console
    )
    compile_s = time.perf_counter() - t0
    print(
        f"\nCompile/session build done in {compile_s:.1f}s ({compile_s / 60.0:.1f} min)."
    )
    print(f"Cache ready at: {CACHE_DIR / cache_key}")

    # One warmup run with a correctly-shaped random input to fully warm the cache.
    inp = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in inp.shape]
    dummy = np.random.rand(*shape).astype(np.float32)
    session.run(None, {inp.name: dummy})
    print("cache warmed")
    print("\nThe notebook's NPU compile section will now be an instant cache hit.")


if __name__ == "__main__":
    main()
