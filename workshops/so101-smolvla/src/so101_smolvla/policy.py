"""Load SmolVLA and run one action."""

from __future__ import annotations

import time

import numpy as np
import torch

from pathlib import Path

from so101_smolvla.action_scale import dataset_stats
from so101_smolvla.task import TASK

_STATS = (
    "models--lerobot--smolvla_base/snapshots/"
    "c83c3163b8ca9b7e67c509fffd9121e66cb96205/"
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
)


def _checkpoint_stats() -> dict:
    from safetensors.torch import load_file

    path = Path.home() / ".cache/huggingface/hub" / _STATS
    if not path.is_file():
        raise FileNotFoundError(path)
    return dataset_stats(load_file(path))


def load_policy(device: str = "cuda"):
    from lerobot.policies.smolvla import make_smolvla_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to(device).eval()
    policy.config.device = device
    policy.config.n_action_steps = 1
    policy.config.empty_cameras = 1
    pre, post = make_smolvla_pre_post_processors(policy.config, dataset_stats=_checkpoint_stats())
    policy.reset()
    return policy, pre, post, device


def select_action(
    policy,
    pre,
    post,
    state: np.ndarray,
    top: np.ndarray,
    wrist: np.ndarray,
    device: str,
    task: str | None = None,
) -> tuple[np.ndarray, float]:
    def image(frame: np.ndarray) -> torch.Tensor:
        # The policy pads this frame to 512x512. Squashing it to a square first
        # changes where the cube appears.
        return torch.from_numpy(np.ascontiguousarray(frame)).permute(2, 0, 1).float() / 255.0

    batch = {
        "observation.state": torch.from_numpy(state.astype(np.float32)),
        "observation.images.camera1": image(top),
        "observation.images.camera2": image(wrist),
        "task": TASK if task is None else task,
    }
    started = time.perf_counter()
    with torch.inference_mode():
        action = post(policy.select_action(pre(batch)))
    elapsed = time.perf_counter() - started
    return action.detach().float().cpu().numpy().reshape(-1)[:6], elapsed
