"""CPU SAPIEN worker. The ROCm parent communicates only through framed IPC."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback

import numpy as np

from so101_smolvla.sapien_config import (
    ACTION_SHAPE,
    ENVIRONMENT_ARGUMENTS,
    PROTOCOL_VERSION,
    SAPIEN_COMMIT,
    SAPIEN_SOURCE,
)
from so101_smolvla.sapien_protocol import (
    MAX_REQUEST_BYTES,
    RESPONSE_HEADER,
    encode_payload,
)


def _plain(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _observation(obs: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    state = np.asarray(obs["agent_pos"], dtype=np.float32)
    if state.shape != ACTION_SHAPE or not np.isfinite(state).all():
        raise ValueError(f"invalid SAPIEN state {state}")
    metadata = {"state": state.tolist()}
    return metadata, obs["pixels"]["top"], obs["pixels"]["wrist"]


def _scene_facts(env) -> dict:
    inner = env._env.unwrapped
    evaluated = inner.evaluate()
    keys = (
        "success",
        "item_lifted",
        "is_item_grasped",
        "is_item_above_bin",
        "robot_touching_item",
    )
    facts = {key: _plain(evaluated[key])[0] for key in keys if key in evaluated}
    facts["item_position"] = _plain(inner.item.pose.p)[0]
    facts["bin_position"] = _plain(inner.bin.pose.p)[0]
    return facts


def serve() -> None:
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    from so101_sim.lerobot_env import So101SimEnv

    actual_commit = subprocess.check_output(
        ["git", "-C", str(SAPIEN_SOURCE), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    env = None

    def reply(request_id: int, status: int, payload: bytes) -> None:
        protocol.write(RESPONSE_HEADER.pack(status, request_id, len(payload)))
        protocol.write(payload)

    try:
        for line in sys.stdin.buffer:
            request_id = 0
            try:
                if len(line) > MAX_REQUEST_BYTES:
                    raise ValueError(f"request exceeds {MAX_REQUEST_BYTES} bytes")
                request = json.loads(line)
                request_id = int(request["request_id"])
                operation = request["operation"]

                if operation == "hello":
                    if int(request["protocol_version"]) != PROTOCOL_VERSION:
                        raise ValueError(
                            f"protocol {request['protocol_version']} != {PROTOCOL_VERSION}"
                        )
                    if actual_commit != SAPIEN_COMMIT:
                        raise RuntimeError(
                            f"SAPIEN source is {actual_commit}, expected {SAPIEN_COMMIT}"
                        )
                    horizon = int(request["horizon"])
                    if horizon < 1:
                        raise ValueError(f"invalid horizon {horizon}")
                    env = So101SimEnv(
                        **ENVIRONMENT_ARGUMENTS,
                        episode_length=horizon + 10,
                    )
                    payload = encode_payload(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "simulator_commit": actual_commit,
                            "action_shape": list(ACTION_SHAPE),
                            "image_shape": [480, 640, 3],
                            "environment_arguments": {
                                **ENVIRONMENT_ARGUMENTS,
                                "episode_length": horizon + 10,
                            },
                        }
                    )
                    reply(request_id, 1, payload)
                    continue

                if env is None:
                    raise RuntimeError("hello must initialize the environment first")
                if operation == "reset":
                    obs, _ = env.reset(seed=int(request["environment_seed"]))
                    metadata, top, wrist = _observation(obs)
                    metadata["facts"] = _scene_facts(env)
                    reply(request_id, 1, encode_payload(metadata, top, wrist))
                elif operation == "step":
                    action = np.asarray(request["action"], dtype=np.float32)
                    if action.shape != ACTION_SHAPE or not np.isfinite(action).all():
                        raise ValueError(f"invalid action {action}")
                    obs, reward, terminated, truncated, info = env.step(action)
                    metadata, top, wrist = _observation(obs)
                    metadata.update(
                        {
                            "commanded": action.tolist(),
                            "reward": float(reward),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            "success": bool(info["is_success"]),
                            "facts": _scene_facts(env),
                        }
                    )
                    reply(request_id, 1, encode_payload(metadata, top, wrist))
                elif operation == "close":
                    reply(request_id, 1, encode_payload({"closed": True}))
                    return
                else:
                    raise ValueError(f"unknown operation {operation!r}")
            except Exception:
                reply(request_id, 0, traceback.format_exc().encode())
    finally:
        if env is not None:
            env.close()
        protocol.close()


if __name__ == "__main__":
    if sys.argv[1:] != ["--serve"]:
        raise SystemExit("use --serve")
    serve()
