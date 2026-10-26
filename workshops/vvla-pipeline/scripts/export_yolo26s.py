# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Export YOLOv26s-pose or YOLOv26s (detection) to FP32 ONNX for the Ryzen AI NPU.

Manual setup step. bootstrap.sh does not run this: Ultralytics is AGPL-3.0.
Safe to re-run (skips if output exists).

    python scripts/export_yolo26s.py pose     [--out models/yolo26s-pose]
    python scripts/export_yolo26s.py detect   [--out models/yolo26s]

Downloads the checkpoint via ultralytics if needed, exports a static batch-1
640×640 ONNX (opset 17, simplified), and validates the graph. ``nms`` is left
unset, so the file is the one-to-many head (pose ``[1, 56, 8400]``, detect
``[1, 84, 8400]``) and this process runs NMS. ``nms=False`` is the
one-to-one head (``[1, 300, 57]``, ``[1, 300, 6]``), whose class scores fall
under the 0.2 cutoff. The VitisAI EP converts FP32→BF16 at NPU compile time (see
config/vaiep_config.json); the first compile is slow and cached under
``cache/`` (config ``yolo_pose.cache_key`` / ``yolo_detect.cache_key``).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Prevent auto-install of onnxruntime triggered by ultralytics.
os.environ["YOLO_AUTOINSTALL"] = "false"

import onnx

IMG_SIZE = 640
# model: (checkpoint stem, what to run next)
MODELS = {
    "pose": ("yolo26s-pose", "python -m vla_pipeline.vision.yolo_pose_npu"),
    "detect": ("yolo26s", "python -m vla_pipeline.vision.yolo_detect_npu --find ball"),
}


def export_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    """Export the ultralytics checkpoint at pt_path to a static FP32 ONNX at onnx_path.

    Leave nms unset. Ultralytics sets end2end = (nms is False), so False
    exports the one-to-one head whose class scores fall under the 0.2 cutoff;
    unset exports the one-to-many head, and nms_keep runs NMS here.
    """
    from ultralytics import YOLO

    print(f"Exporting {pt_path} to ONNX FP32...")
    model = YOLO(str(pt_path))
    model.export(
        format="onnx",
        imgsz=IMG_SIZE,
        opset=17,
        simplify=True,
        dynamic=False,
        batch=1,
    )
    default_name = Path(str(pt_path).replace(".pt", ".onnx"))
    if default_name.exists():
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        default_name.replace(onnx_path)
    # ultralytics writes <stem>.pt/.onnx into the CWD; tuck the weights
    # next to the ONNX and remove any stray copy left in the repo root.
    for stray in (Path(pt_path.name), Path(pt_path.name.replace(".pt", ".onnx"))):
        if stray.exists() and stray.resolve() != onnx_path.resolve():
            dest = onnx_path.parent / stray.name
            try:
                stray.replace(dest) if stray.suffix == ".pt" else stray.unlink()
            except OSError:
                pass
    print("Saved:", onnx_path)


def validate(onnx_path: Path) -> None:
    """Load the ONNX model, run the checker, and print its input/output shapes."""
    print("\nValidating ONNX model...")
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input:  {inp.name}  {shape}")
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output: {out.name}  {shape}")
    print("ONNX model is VALID")


def main() -> None:
    """Export the chosen YOLOv26s checkpoint to ONNX and validate it."""
    parser = argparse.ArgumentParser(description="Export YOLOv26s-pose or YOLOv26s detection to ONNX")
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--out", default=None, help="output folder (default: models/<checkpoint>)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stem, then = MODELS[args.model]
    out = Path(args.out or f"models/{stem}")
    onnx_out = out / f"{stem}.onnx"
    if onnx_out.exists() and not args.force:
        print(f"{onnx_out} already present - skipping export (use --force to redo).")
        return

    out.mkdir(parents=True, exist_ok=True)
    pt = out / f"{stem}.pt"
    export_to_onnx(pt if pt.exists() else Path(f"{stem}.pt"), onnx_out)
    validate(onnx_out)
    print(f"\nDone - run: {then}")


if __name__ == "__main__":
    main()
