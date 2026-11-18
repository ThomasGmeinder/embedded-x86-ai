"""Run the unchanged SmolVLA checkpoint in the complete SAPIEN environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from so101_smolvla.policy_env import (
    PolicyObservation,
    create_policy_environment,
    random_output_directory,
)
from so101_smolvla.record import _write_clip, accumulate
from so101_smolvla.sapien_config import TASK_TEXT
from so101_smolvla.track import CHECKPOINT, JOINT_NAMES, trajectory_metrics

ROOT = Path(__file__).resolve().parents[2]
VLA_TESTING = ROOT / "vla_testing"
DEFAULT_STEPS = 300


def _load_policy():
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    checkpoint = Path(snapshot_download(repo_id=CHECKPOINT, local_files_only=True))
    config = PreTrainedConfig.from_pretrained(checkpoint)
    policy_class = get_policy_class(config.type)
    policy = policy_class.from_pretrained(checkpoint).to("cuda").eval()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {
                "rename_map": {
                    "observation.images.top": "observation.images.camera1",
                    "observation.images.wrist": "observation.images.camera2",
                }
            },
        },
    )
    return checkpoint, policy, pre, post


def _action(policy, pre, post, observation: PolicyObservation) -> np.ndarray:
    from lerobot.envs.utils import preprocess_observation

    batched = {
        "agent_pos": observation.state[None],
        "pixels": {
            "top": observation.top[None],
            "wrist": observation.wrist[None],
        },
    }
    feed = preprocess_observation(batched)
    feed["task"] = [TASK_TEXT]
    with torch.inference_mode():
        action = post(policy.select_action(pre(feed)))
    target = action.squeeze(0).detach().cpu().numpy().astype(np.float64)
    if target.shape != (6,) or not np.isfinite(target).all():
        raise RuntimeError(f"checkpoint returned invalid action {target}")
    return target


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in ("lerobot", "torch", "numpy", "huggingface-hub"):
        versions[distribution] = importlib.metadata.version(distribution)
    return versions


def _atomic_episode(
    destination: Path,
    *,
    result: dict,
    arrays: dict[str, np.ndarray],
    policy_observations: dict[str, np.ndarray],
    top: np.ndarray,
    wrist: np.ndarray,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed episode {destination}")
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        np.savez_compressed(temporary / "trajectory.npz", **arrays)
        np.savez_compressed(
            temporary / "policy-observations.npz", **policy_observations
        )
        _write_clip(top, temporary / "top.mp4")
        _write_clip(wrist, temporary / "wrist.mp4")
        _write_clip(
            np.concatenate((top, wrist), axis=2),
            temporary / "rollout.mp4",
        )
        artifact_names = (
            "result.json",
            "trajectory.npz",
            "policy-observations.npz",
            "top.mp4",
            "wrist.mp4",
            "rollout.mp4",
        )
        completion = {
            "complete": True,
            "artifacts": {
                name: {"sha256": _hash(temporary / name)}
                for name in artifact_names
            },
        }
        (temporary / "COMPLETE.json").write_text(
            json.dumps(completion, indent=2) + "\n", encoding="utf-8"
        )
        os.rename(temporary, destination)
    except Exception:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise


def run_episode(
    env,
    policy,
    pre,
    post,
    checkpoint: Path,
    destination: Path,
    *,
    environment_seed: int,
    policy_seed: int,
    steps: int,
) -> dict:
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    policy.reset()
    observation = env.reset(environment_seed)

    targets = []
    commanded = []
    achieved = []
    rewards = []
    successes = []
    terminated = []
    truncated = []
    chunk_index = []
    item_positions = []
    bin_positions = []
    top_frames = []
    wrist_frames = []
    policy_top = []
    policy_wrist = []
    first_success = None
    picked_up = False
    held = False

    action_steps = int(policy.config.n_action_steps)
    for step in range(steps):
        if step % action_steps == 0:
            policy_top.append(observation.top.copy())
            policy_wrist.append(observation.wrist.copy())
        target = _action(policy, pre, post, observation)
        outcome = env.step(target)
        observation = outcome.observation
        targets.append(target)
        commanded.append(outcome.commanded)
        achieved.append(outcome.achieved)
        rewards.append(outcome.reward)
        successes.append(outcome.success)
        terminated.append(outcome.terminated)
        truncated.append(outcome.truncated)
        chunk_index.append(step // action_steps)
        top_frames.append(observation.top)
        wrist_frames.append(observation.wrist)
        facts = outcome.facts
        item_positions.append(facts["item_position"])
        bin_positions.append(facts["bin_position"])
        picked_up |= bool(facts.get("item_lifted", False))
        held |= bool(facts.get("is_item_grasped", False))
        if outcome.success and first_success is None:
            first_success = step
        if (step + 1) % 25 == 0:
            print(
                f"{destination.name} {step + 1}/{steps}; "
                f"success={outcome.success}",
                flush=True,
            )

    target_rows = np.stack(targets)
    achieved_rows = np.stack(achieved)
    commanded_rows = np.stack(commanded)
    error = np.abs(target_rows - achieved_rows)
    result = {
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": checkpoint.name,
        "backend": env.backend,
        "provenance": env.provenance,
        "package_versions": _package_versions(),
        "task": TASK_TEXT,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "steps": steps,
        "first_success_step": first_success,
        "success": first_success is not None,
        "picked_up": picked_up,
        "held": held,
        "raw_model_targets": trajectory_metrics(target_rows),
        "achieved_joints": trajectory_metrics(achieved_rows),
        "tracking_error": {
            name: {
                "median": float(np.median(error[:, index])),
                "p95": float(np.percentile(error[:, index], 95)),
                "max": float(np.max(error[:, index])),
            }
            for index, name in enumerate(JOINT_NAMES)
        },
    }
    _atomic_episode(
        destination,
        result=result,
        arrays={
            "target": target_rows,
            "commanded": commanded_rows,
            "achieved": achieved_rows,
            "reward": np.asarray(rewards, dtype=np.float32),
            "success": np.asarray(successes, dtype=bool),
            "terminated": np.asarray(terminated, dtype=bool),
            "truncated": np.asarray(truncated, dtype=bool),
            "chunk_index": np.asarray(chunk_index, dtype=np.int64),
            "item_position": np.asarray(item_positions, dtype=np.float32),
            "bin_position": np.asarray(bin_positions, dtype=np.float32),
        },
        policy_observations={
            "top": np.stack(policy_top),
            "wrist": np.stack(policy_wrist),
        },
        top=np.stack(top_frames),
        wrist=np.stack(wrist_frames),
    )
    return result


def _write_report(rows: list[dict], destination: Path) -> Path:
    lines = [
        f"# SAPIEN random regression {destination.name}",
        "",
        "Accumulated video: [top](overall-top.mp4), [wrist](overall-wrist.mp4).",
        "",
        "| Environment seed | Policy seed | Success | Picked up | Held | Lift target reversals | Lift achieved reversals | Elbow target reversals | Elbow achieved reversals |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["raw_model_targets"]
        achieved = row["achieved_joints"]
        lines.append(
            "| {environment_seed} | {policy_seed} | {success} | {picked_up} | {held} | "
            "{raw_lift:.1%} | {achieved_lift:.1%} | {raw_elbow:.1%} | {achieved_elbow:.1%} |".format(
                environment_seed=row["environment_seed"],
                policy_seed=row["policy_seed"],
                success="pass" if row["success"] else "fail",
                picked_up="pass" if row["picked_up"] else "fail",
                held="pass" if row["held"] else "fail",
                raw_lift=raw["shoulder_lift"]["reversal_rate"],
                achieved_lift=achieved["shoulder_lift"]["reversal_rate"],
                raw_elbow=raw["elbow_flex"]["reversal_rate"],
                achieved_elbow=achieved["elbow_flex"]["reversal_rate"],
            )
        )
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_batch(
    count: int,
    *,
    destination: Path,
    steps: int = DEFAULT_STEPS,
    environment_seed_base: int = 0,
    policy_seed_base: int = 0,
) -> list[dict]:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite batch {destination}")
    destination.mkdir(parents=True)
    checkpoint, policy, pre, post = _load_policy()
    env = create_policy_environment("sapien", horizon=steps)
    rows = []
    try:
        for index in range(count):
            rows.append(
                run_episode(
                    env,
                    policy,
                    pre,
                    post,
                    checkpoint,
                    destination / f"seed-{environment_seed_base + index:05d}",
                    environment_seed=environment_seed_base + index,
                    policy_seed=policy_seed_base + index,
                    steps=steps,
                )
            )
    finally:
        env.close()
    (destination / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    accumulate(
        [destination / f"seed-{environment_seed_base + index:05d}" for index in range(count)],
        destination,
        "overall",
    )
    _write_report(rows, destination)
    (destination / "COMPLETE.json").write_text(
        json.dumps({"complete": True, "episodes": count}, indent=2) + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--environment-seed-base", type=int, default=0)
    parser.add_argument("--policy-seed-base", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    destination = args.output or random_output_directory(
        VLA_TESTING,
        "sapien",
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    rows = run_batch(
        args.count,
        destination=destination,
        steps=args.steps,
        environment_seed_base=args.environment_seed_base,
        policy_seed_base=args.policy_seed_base,
    )
    print(
        f"succeeded {sum(1 for row in rows if row['success'])} of {len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
