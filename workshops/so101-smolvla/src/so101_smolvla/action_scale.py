"""Joint scale stored with lerobot/smolvla_base.

The checkpoint records one SO-100 joint scale, under ``so100.buffer.action``.
State and action are that same six-vector, so both use it. Without that,
the open gripper is a raw 100 and the network answers with its average pose.
"""

from __future__ import annotations

SO100_ACTION = "so100.buffer.action"


def dataset_stats(tensors: dict) -> dict:
    mean = tensors[f"{SO100_ACTION}.mean"]
    std = tensors[f"{SO100_ACTION}.std"]
    scale = {"mean": mean, "std": std}
    return {"action": scale, "observation.state": scale}
