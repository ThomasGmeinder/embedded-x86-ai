"""SAM 3 (the reference sam3-tracker-rocm) in a process of its own: it outlines what it finds for one word.

usage: scripts/run_sam3_reference.sh src/so101_smolvla/sam3_worker.py <descriptor to read requests from> <descriptor to write replies to>

Started by so101_smolvla.sam3_outlines.Sam3Outlines through the reference's
launcher, which puts this process into the reference's Python environment
(.venv-sam3) and its checkout and names the model folder (SAM3_MODEL_DIR) and
the compiled model files (SAM3_ONNX_DIR). This file imports nothing of the
workshop's package, which is not installed in that environment; the one module
it takes from the workshop is ``loading_messages``, the file beside it, which
needs the standard library only. The requests:

  start    begin a new tracking session for each camera named in ``cameras``,
           for the word ``phrase``
  outline  the next picture of each camera; the reply holds per camera the
           outlines, the numbers of the tracked things they belong to and the
           seconds taken
  close    end the process

Every call runs the full detection, not only the tracker: the pictures of one
layout are 50 steps apart. No cap on the number of tracked things.

A request that fails is answered with ``{"error": ...}``.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import sys
import time
import traceback
import warnings
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np

ROUTE = "sam3-tracker-rocm"
IMAGE_SIZE = 504


def _versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "migraphx", "onnxruntime", "onnxruntime-ep-migraphx", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


def serve(requests: Connection, replies: Connection) -> None:
    try:
        # The reference compiles parts of the model with torch.jit.script, which this torch
        # announces as deprecated: the reference's to act on, not shown at every start.
        warnings.filterwarnings("ignore", message=r"`torch\.jit\.script` is deprecated", category=FutureWarning)
        import torch
        from tracker.live_inference import SAM3Live

        # One line per model when its weights are loaded, instead of a progress bar.
        from loading_messages import loading_weights_as_a_line

        loading_weights_as_a_line()
        model_dir = Path(os.environ["SAM3_MODEL_DIR"])
        onnx_dir = Path(os.environ["SAM3_ONNX_DIR"])
        manifest = onnx_dir / "ARTIFACT_MANIFEST.json"
        if not (model_dir / "model.safetensors").is_file():
            raise RuntimeError(f"the SAM 3 weights are missing: {model_dir / 'model.safetensors'}")
        if not manifest.is_file():
            raise RuntimeError(
                f"the compiled SAM 3 model files are not built: {manifest} is missing "
                "(export/build_text_prompt_mig.py through scripts/run_sam3_reference.sh)"
            )
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        manifest_source = json.loads(manifest.read_text()).get("source", {})
    except Exception:
        replies.send({"error": traceback.format_exc()})
        raise

    live: dict[str, SAM3Live] = {}
    phrase: str | None = None

    def start(cameras: list[str], word: str) -> None:
        nonlocal phrase
        if phrase == word and set(cameras) == set(live):
            # The same word: the tracked things are dropped, the word stays.
            for instance in live.values():
                instance.reset_tracking()
            return
        for instance in live.values():
            instance.close()
        live.clear()
        for camera in cameras:
            live[camera] = SAM3Live(
                checkpoint=model_dir,
                prompts=[word],
                onnx_dir=onnx_dir,
                imgsz=IMAGE_SIZE,
                max_objects_per_prompt=None,
            )
        phrase = word

    def outline(camera: str, picture: np.ndarray) -> dict:
        if camera not in live:
            raise RuntimeError("no layout was started")
        if picture.ndim != 3 or picture.shape[2] != 3 or picture.dtype != np.uint8:
            raise ValueError(f"invalid picture {picture.dtype} of shape {picture.shape}")
        began = time.perf_counter()
        # The reference takes pictures in OpenCV's order of colours.
        result = live[camera].infer(
            np.ascontiguousarray(picture[..., ::-1]), full_detection=True
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began
        numbers = [int(number) for number in result["object_ids"]]
        masks = [np.asarray(result["masks"][number], dtype=bool) for number in numbers]
        for mask in masks:
            if mask.shape != picture.shape[:2]:
                raise RuntimeError(
                    f"an outline of shape {mask.shape} for a picture of shape {picture.shape[:2]}"
                )
        return {
            "masks": (
                np.stack(masks)
                if masks
                else np.zeros((0, *picture.shape[:2]), dtype=bool)
            ),
            "numbers": numbers,
            "scores": [float(result["scores"][number]) for number in numbers],
            "seconds": seconds,
        }

    replies.send(
        {
            "ready": True,
            "route": ROUTE,
            "model_dir": str(model_dir),
            "compiled_dir": str(onnx_dir),
            "manifest_sha256": manifest_sha256,
            "source": manifest_source,
            "source_commit": os.environ.get("SAM3_SOURCE_COMMIT"),
            "source_dirty": os.environ.get("SAM3_SOURCE_DIRTY"),
            "image_id": os.environ.get("SAM3_DOCKER_IMAGE_ID"),
            "versions": _versions(),
            "device": torch.cuda.get_device_name(0),
            "image_size": IMAGE_SIZE,
            "dtype": "float16",
            "full_detection_every_call": True,
            "max_objects_per_prompt": None,
            "miopen_find_mode": os.environ.get("MIOPEN_FIND_MODE"),
        }
    )
    while True:
        try:
            request = requests.recv()
        except EOFError:
            # The process that asks is gone.
            break
        try:
            operation = request["operation"]
            if operation == "close":
                replies.send({"closed": True})
                break
            if operation == "start":
                start(list(request["cameras"]), str(request["phrase"]))
                reply = {"started": True}
            elif operation == "outline":
                if not live:
                    raise RuntimeError("no layout was started")
                # One reply per camera the layout was started for; a request
                # without a camera's picture is a fault of the client.
                reply = {
                    camera: outline(camera, request[camera])
                    for camera in live
                }
            else:
                raise ValueError(f"unknown operation {operation!r}")
        except Exception:
            reply = {"error": traceback.format_exc()}
        replies.send(reply)
    for instance in live.values():
        instance.close()


def main(argv: list[str] | None = None) -> None:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        raise SystemExit(__doc__)
    serve(
        Connection(int(arguments[0]), writable=False),
        Connection(int(arguments[1]), readable=False),
    )


if __name__ == "__main__":
    main()
