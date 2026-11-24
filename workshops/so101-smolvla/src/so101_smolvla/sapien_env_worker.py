"""CPU SAPIEN worker. The ROCm parent communicates only through framed IPC."""

from __future__ import annotations

import functools
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

from so101_smolvla.loading_messages import WithoutNotice
from so101_smolvla.sapien_config import (
    ACTION_SHAPE,
    HAND_LINKS,
    MOTOR_BLACK,
    ORANGE_ROBOT_LINKS,
    PROTOCOL_VERSION,
    ROBOT_ORANGE,
    SAPIEN_COMMIT,
    SAPIEN_SOURCE,
    environment_arguments,
    task_spec,
)
from so101_smolvla.sapien_grip import grip_facts
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


def _robot_part_color(link_name: str, mesh_filename: str):
    if link_name not in ORANGE_ROBOT_LINKS:
        return None
    if Path(mesh_filename).stem.startswith("sts3215_"):
        return MOTOR_BLACK
    return ROBOT_ORANGE


def _configure_robot_appearance(env) -> None:
    from sapien.render import RenderBodyComponent

    inner = env._env.unwrapped
    inner.domain_randomization_config.robot_color = None
    for link in inner.agent.robot.links:
        for obj in link._objs:
            render_body = obj.entity.find_component_by_type(RenderBodyComponent)
            if render_body is None:
                continue
            for render_shape in render_body.render_shapes:
                color = _robot_part_color(link.name, render_shape.filename)
                if color is None:
                    continue
                for part in render_shape.parts:
                    part.material.set_base_color(list(color))


def _new_environment(arguments: dict, episode_length: int):
    """The complete environment, painted, that has run nothing yet."""
    from so101_sim.lerobot_env import So101SimEnv

    env = So101SimEnv(**arguments, episode_length=episode_length)
    _configure_robot_appearance(env)
    return env


def _scene_facts(env) -> dict:
    inner = env._env.unwrapped
    if hasattr(inner, "scene_facts"):
        return inner.scene_facts()
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
    facts["grip"] = grip_facts(inner.agent, inner.item)
    return facts


def _outline_maps(env) -> tuple[np.ndarray | None, np.ndarray | None]:
    """The outline maps of the last observation, from a scene that hands them out."""
    inner = env._env.unwrapped
    if not hasattr(inner, "outline_maps"):
        return None, None
    return inner.outline_maps()


def serve() -> None:
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "wb", buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

    import so101_smolvla.sapien_bin_env  # noqa: F401
    import so101_smolvla.sapien_stack_env  # noqa: F401
    from mani_skill.utils.logging_utils import logger as simulator_log
    from so101_sim.envs import ROBOT_COLOR as TRAINING_ROBOT_COLOR

    # ManiSkill asks for a CUDA device to render with, finds none on this machine, says so and
    # renders on the processor: at every environment it makes, so here at every reset. The
    # simulator's environment class hands no render device on (so101_sim/lerobot_env.py passes
    # ``sim_backend`` only and drops what it does not know), so the request cannot be changed
    # from here. That the twin renders on the processor is in README.md, "Compute placement".
    simulator_log.addFilter(WithoutNotice('Requested to use render device "sapien_cuda"'))

    actual_commit = subprocess.check_output(
        ["git", "-C", str(SAPIEN_SOURCE), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    make_environment = None
    env = None
    # Whether the environment has already started a test.
    started = False

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
                    task_mode = str(request["task_mode"])
                    spec = task_spec(task_mode)
                    arguments = environment_arguments(task_mode)
                    make_environment = functools.partial(
                        _new_environment, arguments, horizon + 10
                    )
                    env = make_environment()
                    started = False
                    payload = encode_payload(
                        {
                            "protocol_version": PROTOCOL_VERSION,
                            "simulator_commit": actual_commit,
                            "task_mode": task_mode,
                            "task_text": spec.instruction,
                            "action_shape": list(ACTION_SHAPE),
                            "image_shape": [480, 640, 3],
                            "robot_appearance": {
                                "printed_parts_rgba": list(ROBOT_ORANGE),
                                "motors_rgba": list(MOTOR_BLACK),
                                "hand_links": sorted(HAND_LINKS),
                                "hand_rgba": [*TRAINING_ROBOT_COLOR, 1.0],
                            },
                            "environment_arguments": {
                                **arguments,
                                "episode_length": horizon + 10,
                            },
                        }
                    )
                    reply(request_id, 1, payload)
                    continue

                if env is None:
                    raise RuntimeError("hello must initialize the environment first")
                if operation == "reset":
                    # The simulator's reset moves robot and objects back but
                    # keeps its physics scene, and what that scene keeps from
                    # earlier tests changes how the next one runs. Every test
                    # therefore starts in an environment that has run nothing.
                    if started:
                        env.close()
                        env = None
                        env = make_environment()
                    started = True
                    if "stack_replay" in request:
                        raw_obs, _ = env._env.reset(
                            seed=int(request["environment_seed"]),
                            options={"stack_replay": request["stack_replay"]},
                        )
                        obs = env._format_raw_obs(raw_obs)
                    else:
                        obs, _ = env.reset(
                            seed=int(request["environment_seed"])
                        )
                    metadata, top, wrist = _observation(obs)
                    metadata["facts"] = _scene_facts(env)
                    reply(
                        request_id,
                        1,
                        encode_payload(metadata, top, wrist, *_outline_maps(env)),
                    )
                elif operation == "set_target":
                    inner = env._env.unwrapped
                    if not hasattr(inner, "set_target"):
                        raise ValueError("selected task has no cube to name")
                    inner.set_target(str(request["cube"]))
                    reply(
                        request_id,
                        1,
                        encode_payload({"facts": _scene_facts(env)}),
                    )
                elif operation == "scene_without_cubes":
                    inner = env._env.unwrapped
                    if not hasattr(inner, "pictures_without_cubes_and_robot"):
                        raise ValueError(
                            "selected task cannot show its scene without cubes"
                        )
                    top, wrist = inner.pictures_without_cubes_and_robot()
                    reply(request_id, 1, encode_payload({}, top, wrist))
                elif operation == "set_command":
                    inner = env._env.unwrapped
                    if not hasattr(inner, "set_stack_goal"):
                        raise ValueError("selected task does not accept stack commands")
                    inner.set_stack_goal(
                        str(request["source"]), str(request["target"])
                    )
                    reply(
                        request_id,
                        1,
                        encode_payload({"facts": _scene_facts(env)}),
                    )
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
                    reply(
                        request_id,
                        1,
                        encode_payload(metadata, top, wrist, *_outline_maps(env)),
                    )
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
