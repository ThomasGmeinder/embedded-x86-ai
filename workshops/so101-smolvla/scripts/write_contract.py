#!/usr/bin/env python3
"""Write configs/contract.yaml from the loaded checkpoint and time one action."""

from __future__ import annotations

import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("STOP: ROCm device is not visible. A CPU run is not a pass.")
    started = time.perf_counter()
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    load_s = time.perf_counter() - started
    cfg = policy.config
    if tuple(cfg.input_features["observation.state"].shape) != (6,):
        raise SystemExit("STOP: state width is not 6")
    if tuple(cfg.output_features["action"].shape) != (6,):
        raise SystemExit("STOP: action width is not 6")
    if cfg.use_delta_joint_actions_aloha:
        raise SystemExit("STOP: checkpoint actions are deltas")
    cameras = set(cfg.input_features)
    for key in ("observation.images.camera1", "observation.images.camera2", "observation.images.camera3"):
        if key not in cameras:
            raise SystemExit(f"STOP: missing {key}")
    if cfg.resize_imgs_with_padding is None:
        raise SystemExit("STOP: processor does not resize")

    from so101_smolvla.policy import select_action

    from lerobot.policies.smolvla import make_smolvla_pre_post_processors

    loaded_steps = int(cfg.n_action_steps)
    policy.config.n_action_steps = 1
    pre, post = make_smolvla_pre_post_processors(policy.config)
    policy.reset()
    import numpy as np

    state = np.zeros(6, dtype=np.float32)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    _, first = select_action(policy, pre, post, state, blank, blank, device)
    _, second = select_action(policy, pre, post, state, blank, blank, device)
    contract = {
        "checkpoint": {
            "path": "lerobot/smolvla_base",
            "state_shape": [6],
            "action_shape": [6],
            "absolute": True,
            "camera_keys": ["observation.images.camera1", "observation.images.camera2", "observation.images.camera3"],
            "resize": list(cfg.resize_imgs_with_padding),
            "normalization": {k: str(v) for k, v in cfg.normalization_mapping.items()},
            "chunk_size": int(cfg.chunk_size),
            "loaded_n_action_steps": loaded_steps,
        },
        "project": {
            "n_action_steps": 1,
            "empty_cameras": 1,
            "rename_map": {
                "observation.images.top": "observation.images.camera1",
                "observation.images.wrist": "observation.images.camera2",
            },
            "arm_unit": "degrees",
            "gripper_unit": "0_closed_100_open",
            "inference_budget_s": round(second, 4),
            "first_forward_s": round(first, 4),
            "load_s": round(load_s, 2),
            "device": device,
        },
    }
    path = ROOT / "configs" / "contract.yaml"
    path.write_text(yaml.safe_dump(contract, sort_keys=False), encoding="utf-8")
    print(path.read_text())


if __name__ == "__main__":
    main()
