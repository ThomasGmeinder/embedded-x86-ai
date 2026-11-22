"""SAM 3 in a process of its own: it outlines what it finds for one word.

usage: python -m so101_smolvla.sam3_worker <descriptor to read requests from> <descriptor to write replies to>

Started by so101_smolvla.sam3_outlines.Sam3Outlines, which also sets this
process's environment. A request is a dictionary with an ``operation``:

  start    begin a new tracking session for each camera
  outline  the next picture of each camera (``top``, ``wrist``); the reply
           holds per camera the outlines, the numbers of the tracked things
           they belong to and the seconds taken
  close    end the process

A request that fails is answered with ``{"error": ...}``.
"""

from __future__ import annotations

import sys
import time
import traceback
from multiprocessing.connection import Connection

from so101_smolvla.sam3_outlines import CAMERAS

MODEL = "facebook/sam3"
# The one word SAM 3 is asked for. It follows every thing it finds for it
# from picture to picture and keeps the thing's number.
PHRASE = "cube"


def serve(requests: Connection, replies: Connection) -> None:
    try:
        import torch
        from PIL import Image
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        processor = Sam3VideoProcessor.from_pretrained(MODEL, local_files_only=True)
        model = (
            Sam3VideoModel.from_pretrained(MODEL, local_files_only=True)
            .to("cuda")
            .eval()
        )
    except Exception:
        replies.send({"error": traceback.format_exc()})
        raise
    sessions: dict[str, object] = {}

    def new_session():
        session = processor.init_video_session(
            inference_device="cuda",
            processing_device="cpu",
            video_storage_device="cpu",
            dtype=torch.float32,
        )
        return processor.add_text_prompt(session, PHRASE)

    def outline(camera: str, picture) -> dict:
        if camera not in sessions:
            raise RuntimeError("no layout was started")
        began = time.perf_counter()
        inputs = processor(images=Image.fromarray(picture), return_tensors="pt")
        with torch.no_grad():
            outputs = model(
                inference_session=sessions[camera], frame=inputs.pixel_values[0]
            )
        found = processor.postprocess_outputs(
            sessions[camera], outputs, original_sizes=inputs.original_sizes
        )
        masks = found["masks"].cpu().numpy().astype(bool)
        numbers = [int(number) for number in found["object_ids"].tolist()]
        torch.cuda.synchronize()
        return {
            "masks": masks.reshape(-1, *picture.shape[:2]),
            "numbers": numbers,
            "seconds": time.perf_counter() - began,
        }

    replies.send(
        {
            "ready": True,
            "model": MODEL,
            "phrase": PHRASE,
            "device": torch.cuda.get_device_name(0),
        }
    )
    while True:
        try:
            request = requests.recv()
        except EOFError:
            # The process that asks is gone.
            return
        try:
            operation = request["operation"]
            if operation == "close":
                replies.send({"closed": True})
                return
            if operation == "start":
                sessions.clear()
                sessions.update({camera: new_session() for camera in CAMERAS})
                reply = {"started": True}
            elif operation == "outline":
                reply = {camera: outline(camera, request[camera]) for camera in CAMERAS}
            else:
                raise ValueError(f"unknown operation {operation!r}")
        except Exception:
            reply = {"error": traceback.format_exc()}
        replies.send(reply)


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
