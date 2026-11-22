"""An NPU load for the utilization tests: YOLO26s detection on the NPU until standard input closes.

Runs in the voice workshop's environment (its activate_venv.sh gives the NPU runtime). Prints "running"
after the first inference and "stopped <count>" at the end.
"""

from __future__ import annotations

import select
import sys

import numpy as np

from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_detect_npu import ObjectDetector


def main() -> None:
    detector = ObjectDetector(load_config(None), device="npu")
    providers = detector.session.get_providers()
    if providers[0] != "VitisAIExecutionProvider":
        raise SystemExit(f"SETUP FAILURE: the detector runs on {providers}, not on the NPU")
    frame = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)
    detector.infer(frame)
    print("running", flush=True)
    count = 0
    while True:
        # The load is the inference; the pipe check between two inferences costs nothing.
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if readable and sys.stdin.readline() == "":
            break
        detector.infer(frame)
        count += 1
    print(f"stopped {count}", flush=True)


if __name__ == "__main__":
    main()
