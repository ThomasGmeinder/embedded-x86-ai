#!/usr/bin/env python3
"""Verify that the Ryzen AI deployment stack can load and use VitisAI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, errors: list[str]) -> None:
    print(f"  [FAIL] {message}")
    errors.append(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true", help="return nonzero on any required check")
    args = parser.parse_args()
    errors: list[str] = []

    if Path("/dev/accel/accel0").exists():
        print("  [ok] XDNA device: /dev/accel/accel0")
    else:
        fail("XDNA device /dev/accel/accel0 is missing", errors)

    xrt = os.environ.get("XILINX_XRT")
    if xrt and Path(xrt, "lib").is_dir():
        print(f"  [ok] XILINX_XRT={xrt}")
    else:
        fail("XILINX_XRT is not configured", errors)

    ld_dirs = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    xrt_lib = str(Path(xrt, "lib")) if xrt else ""
    voe_index = next((i for i, p in enumerate(ld_dirs) if p.endswith("/voe/lib")), None)
    if xrt_lib in ld_dirs and (voe_index is None or ld_dirs.index(xrt_lib) < voe_index):
        print("  [ok] XRT libraries precede voe/lib")
    else:
        fail("XRT libraries must precede voe/lib on LD_LIBRARY_PATH", errors)

    firmware = os.environ.get("XLNX_VART_FIRMWARE", "")
    if firmware and Path(firmware).is_file():
        print(f"  [ok] firmware: {firmware}")
    else:
        fail("XLNX_VART_FIRMWARE does not name an existing xclbin", errors)

    try:
        import onnxruntime as ort
    except Exception as exc:
        fail(f"onnxruntime import failed: {exc}", errors)
        return 1

    available = ort.get_available_providers()
    print(f"  providers: {available}")
    if "VitisAIExecutionProvider" not in available:
        fail("VitisAIExecutionProvider is not registered", errors)
    else:
        # Use the already-compiled Whisper cache as a real load probe when it
        # exists. This must not invoke compilation in the deployment venv.
        import yaml

        cfg = yaml.safe_load((ROOT / "config/pipeline.yaml").read_text())
        whisper = cfg["whisper"]
        model = ROOT / whisper["encoder_onnx"]
        cache_dir = ROOT / whisper.get("cache_dir", "cache")
        cache_key = whisper.get("cache_key_encoder", "whisper_base_encoder")
        artifact = cache_dir / cache_key
        if model.is_file() and artifact.is_dir() and any(artifact.iterdir()):
            try:
                session = ort.InferenceSession(
                    str(model),
                    providers=[
                        (
                            "VitisAIExecutionProvider",
                            {
                                "config_file": str(ROOT / whisper["encoder_vitisai_config"]),
                                "cache_dir": str(cache_dir),
                                "cache_key": cache_key,
                            },
                        )
                    ],
                )
                active = session.get_providers()
                if "VitisAIExecutionProvider" in active:
                    print("  [ok] cached Whisper session uses VitisAIExecutionProvider")
                else:
                    fail(f"cached Whisper session fell back to {active}", errors)
            except Exception as exc:
                fail(f"cached Whisper session failed to load: {exc}", errors)
        else:
            print("  [warn] no compiled Whisper cache available for a real session probe")

    if errors:
        print(f"NPU preflight failed ({len(errors)} problem(s)).")
        return 1 if args.preflight else 0
    print("NPU preflight passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
