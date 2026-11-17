#!/usr/bin/env python3
"""Stage 1 environment check. Reads the venv, not the system ROCm install."""

from __future__ import annotations

import importlib.metadata
import os
import sys
import time

MIN_CORES = 4


def main() -> int:
    cores = os.cpu_count() or 0
    print(f"python: {sys.version.split()[0]}")
    print(f"cpu_count: {cores}")
    if cores < MIN_CORES:
        print(f"STOP: {cores} CPU cores, {MIN_CORES} are required.")
        return 2
    try:
        import torch
    except ImportError:
        print("STOP: torch is not installed in this virtual environment.")
        return 2
    print(f"torch: {torch.__version__}")
    print(f"torch_hip: {getattr(torch.version, 'hip', None)}")
    try:
        print(f"rocm: {importlib.metadata.version('rocm')}")
    except importlib.metadata.PackageNotFoundError:
        print("rocm: missing")
    if "rocm10.1" not in torch.__version__:
        print("STOP: this environment's torch is not the ROCm 10.1 build.")
        return 2
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
        device = "cuda"
    else:
        print("STOP: the ROCm device is not visible. A CPU run is not a pass.")
        return 2

    import mujoco

    print(f"mujoco: {mujoco.__version__}")
    try:
        import lerobot

        print(f"lerobot: {getattr(lerobot, '__version__', 'installed')}")
    except ImportError:
        print("STOP: lerobot is not installed.")
        return 2

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy = policy.to(device).eval()
    # One forward through select_action is recorded by scripts/write_contract.py.
    print(f"checkpoint_loaded_s: {time.perf_counter() - started:.2f}")
    print(f"chunk_size: {policy.config.chunk_size}")
    print(f"n_action_steps_loaded: {policy.config.n_action_steps}")
    print(f"device: {device}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
