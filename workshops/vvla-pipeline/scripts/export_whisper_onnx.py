# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Export Whisper-base to ONNX (encoder + decoder) for the Ryzen AI NPU.

Run by bootstrap.sh; safe to re-run (skips if outputs exist).

    python scripts/export_whisper_onnx.py [--out models/whisper-base] [--force]

Uses Hugging Face Optimum to export ``openai/whisper-base`` into
``encoder_model.onnx`` / ``decoder_model.onnx`` plus the tokenizer and
feature-extractor files that ``vla_pipeline.audio.whisper_npu`` loads.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "openai/whisper-base"


def main() -> None:
    """Export Whisper to ONNX (encoder + decoder) and validate the resulting graphs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models/whisper-base")
    parser.add_argument(
        "--force", action="store_true", help="re-export even if files exist"
    )
    args = parser.parse_args()

    out = Path(args.out)
    encoder = out / "encoder_model.onnx"
    decoder = out / "decoder_model.onnx"
    if encoder.exists() and decoder.exists() and not args.force:
        print(
            f"Whisper ONNX already present in {out} - skipping export (use --force to redo)."
        )
        return

    out.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} → {out} (this downloads ~290 MB on first run)...")
    # Use the optimum-cli entry point: stable across optimum 1.x and 2.x
    # (in 2.x the exporters moved to the separate 'optimum-onnx' package and
    # 'python -m optimum.exporters.onnx' no longer resolves).
    optimum_cli = Path(sys.executable).parent / "optimum-cli"
    if optimum_cli.exists():
        cmd = [str(optimum_cli), "export", "onnx"]
    else:  # fall back to the legacy module path (optimum 1.x)
        cmd = [sys.executable, "-m", "optimum.exporters.onnx"]
    cmd += [
        "--model",
        MODEL_ID,
        "--task",
        "automatic-speech-recognition",
        "--opset",
        "17",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit(
            "Whisper ONNX export failed. Make sure the exporter package is "
            "installed in this venv:\n    uv pip install optimum optimum-onnx\n"
            "then re-run this script."
        )

    # Optimum may name the decoder 'decoder_model.onnx' already; normalize if not.
    if not decoder.exists():
        for cand in ("decoder_model_merged.onnx", "decoder_with_past_model.onnx"):
            p = out / cand
            if p.exists():
                shutil.copy(p, decoder)
                break
    if not (encoder.exists() and decoder.exists()):
        raise SystemExit(
            f"Export finished but {encoder.name}/{decoder.name} not found in {out}"
        )

    print("Validating ONNX graphs...")
    import onnx

    for p in (encoder, decoder):
        onnx.checker.check_model(onnx.load(str(p)))
        print(f"  {p.name}: VALID")
    print("Done. Run: python -m vla_pipeline.audio.whisper_npu --input mic")


if __name__ == "__main__":
    main()
