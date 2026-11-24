"""Run the unchanged SmolVLA checkpoint in the complete SAPIEN environment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

from so101_smolvla.grasp_geometry import (
    FINGER_ANGLE_NOTE,
    GripMoment,
    GripSample,
    finger_angle_summary,
)
from so101_smolvla.policy_env import (
    PolicyObservation,
    create_policy_environment,
    random_stacking_output_directory,
)
from so101_smolvla.record import _write_clip, accumulate
from so101_smolvla.sapien_config import (
    PICK_PLACE_TASKS,
    TASK_SPECS,
    TASK_TEXT,
    THREE_CUBE_TASK,
    task_spec,
)
from so101_smolvla.sapien_stack_session import CommandTrace, StackCommandSession
from so101_smolvla.track import CHECKPOINT, JOINT_NAMES, trajectory_metrics

ROOT = Path(__file__).resolve().parents[2]
VLA_TESTING = ROOT / "vla_testing"
DEFAULT_STEPS = 300
# Steps for one stacking command. One step is one frame of the recordings. In
# the 20 random tests of seeds 0 to 19 the latest stack is reported at step
# 604, and limits of 900 and 1,500 steps bring no further stack. The longest
# of the 200 real stacking recordings is 898 frames, so a test that needs as
# long as that is cut off. A command still ends at the first step at which
# the scene reports the stack.
DEFAULT_STACK_STEPS = 650


def _load_policy():
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    import lerobot.policies.smolvla.modeling_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors

    from so101_smolvla.loading_messages import drop_transformers_notice, loading_weights_as_a_line

    # One line when the weights are loaded instead of a progress bar, and not the notice about
    # ``torch_dtype``: LeRobot passes it for SmolVLA's language model (smolvlm_with_expert.py).
    loading_weights_as_a_line()
    drop_transformers_notice("`torch_dtype` is deprecated")
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


def _action(
    policy,
    pre,
    post,
    observation: PolicyObservation,
    task_text: str = TASK_TEXT,
) -> np.ndarray:
    from lerobot.envs.utils import preprocess_observation

    batched = {
        "agent_pos": observation.state[None],
        "pixels": {
            "top": observation.top[None],
            "wrist": observation.wrist[None],
        },
    }
    feed = preprocess_observation(batched)
    feed["task"] = [task_text]
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
    feeds: Iterable[np.ndarray] | None = None,
) -> None:
    """Write one episode's folder, complete or not at all.

    ``feeds``, where a test has them, are the frames of a further clip
    ``feeds.mp4`` (so101_smolvla.feeds_video).
    """
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
        artifact_names = [
            "result.json",
            "trajectory.npz",
            "policy-observations.npz",
            "top.mp4",
            "wrist.mp4",
            "rollout.mp4",
        ]
        if feeds is not None:
            _write_clip(feeds, temporary / "feeds.mp4")
            artifact_names.append("feeds.mp4")
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
    grip_moment = GripMoment()

    action_steps = int(policy.config.n_action_steps)
    for step in range(steps):
        if step % action_steps == 0:
            policy_top.append(observation.top.copy())
            policy_wrist.append(observation.wrist.copy())
        target = _action(policy, pre, post, observation, env.task_text)
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
        grip = facts["grip"]
        grip_moment.observe(
            GripSample(
                step=step,
                distance=float(grip["distance"]),
                finger_face_deg=float(grip["finger_face_deg"]),
                roll_deg=float(outcome.achieved[JOINT_NAMES.index("wrist_roll")]),
                contact=bool(grip["contact"]),
            )
        )
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
        "task_mode": env.task_mode,
        "task": env.task_text,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "steps": steps,
        "first_success_step": first_success,
        "success": first_success is not None,
        "picked_up": picked_up,
        "held": held,
        **grip_moment.chosen().as_result(),
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


STACK_COMMANDS = {
    "stack2-training": ("put red cube on blue cube",),
    "stack2-sim-view": ("put red cube on blue cube",),
    "stack2-recording-rig": ("put red cube on blue cube",),
    "stack3-color": (
        "put green cube on red cube",
        "put blue cube on green cube",
    ),
}


def _trace_result(trace: CommandTrace) -> dict[str, object]:
    return {
        "original_command": trace.command.original,
        "canonical_command": trace.command.canonical,
        "source": trace.command.source,
        "target": trace.command.target,
        "success": trace.success,
        "success_step": trace.success_step,
        "failure_stage": trace.failure_stage,
        "selected": trace.selected,
        "lifted": trace.lifted,
        "released": trace.released,
        **trace.grip.as_result(),
        "final_facts": trace.facts,
    }


def run_stack_episode(
    env,
    policy,
    pre,
    post,
    checkpoint: Path,
    destination: Path,
    *,
    task_mode: str,
    environment_seed: int,
    policy_seed: int,
    steps_per_command: int,
    stack_replay: dict[str, object] | None = None,
) -> dict:
    """Run one persistent stacking scene and atomically record every stage."""
    try:
        commands = STACK_COMMANDS[task_mode]
    except KeyError as exc:
        raise ValueError(f"unsupported stacking task {task_mode!r}") from exc
    spec = task_spec(task_mode)
    torch.manual_seed(policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(policy_seed)
    observation = env.reset(environment_seed, stack_replay=stack_replay)
    initial_layout = dict(env.facts["initial_layout"])

    def select_action(frame: PolicyObservation, command: str) -> np.ndarray:
        # A task with a trained sentence uses it; the color task has none and
        # is prompted with the user's command.
        return _action(policy, pre, post, frame, spec.instruction or command)

    session = StackCommandSession(
        environment=env,
        policy=policy,
        observation=observation,
        action=select_action,
        max_steps_per_command=steps_per_command,
    )
    traces: list[CommandTrace] = []
    for index, command in enumerate(commands):
        trace = session.execute(command)
        traces.append(trace)
        print(
            f"{destination.name} stage {index + 1}/{len(commands)} "
            f"success={trace.success} failure={trace.failure_stage}",
            flush=True,
        )
        if not trace.success:
            break

    if len(commands) > 1 and traces and not traces[0].success:
        failure_stage = "first stack failed"
    elif traces and not traces[-1].success:
        failure_stage = traces[-1].failure_stage
    elif len(traces) != len(commands):
        failure_stage = "first stack failed"
    else:
        failure_stage = None
    success = len(traces) == len(commands) and all(
        trace.success for trace in traces
    )
    target_rows = np.concatenate([trace.target for trace in traces])
    commanded_rows = np.concatenate([trace.commanded for trace in traces])
    achieved_rows = np.concatenate([trace.achieved for trace in traces])
    top = np.concatenate([trace.top for trace in traces])
    wrist = np.concatenate([trace.wrist for trace in traces])
    cube_colors = sorted(traces[0].cube_positions)
    arrays = {
        "target": target_rows,
        "commanded": commanded_rows,
        "achieved": achieved_rows,
    }
    for color in cube_colors:
        arrays[f"cube_position_{color}"] = np.concatenate(
            [trace.cube_positions[color] for trace in traces]
        )
    stage_lengths = np.asarray(
        [len(trace.target) for trace in traces], dtype=np.int64
    )
    arrays["stage_lengths"] = stage_lengths
    result = {
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": checkpoint.name,
        "backend": env.backend,
        "provenance": env.provenance,
        "package_versions": _package_versions(),
        "task_mode": task_mode,
        "task": spec.instruction,
        "environment_seed": environment_seed,
        "policy_seed": policy_seed,
        "steps_per_command": steps_per_command,
        "initial_layout": initial_layout,
        "commands": [_trace_result(trace) for trace in traces],
        "success": success,
        "failure_stage": failure_stage,
        "raw_model_targets": trajectory_metrics(target_rows),
        "achieved_joints": trajectory_metrics(achieved_rows),
    }
    _atomic_episode(
        destination,
        result=result,
        arrays=arrays,
        policy_observations={"top": top, "wrist": wrist},
        top=top,
        wrist=wrist,
    )
    return result


def _write_report(rows: list[dict], destination: Path) -> Path:
    lines = [
        f"# SAPIEN random regression {destination.name}",
        "",
        "Accumulated video: [top](overall-top.mp4), [wrist](overall-wrist.mp4).",
        "",
        *FINGER_ANGLE_NOTE,
        "",
        finger_angle_summary(row["finger_face_deg"] for row in rows),
        "",
        "| Environment seed | Policy seed | Success | Picked up | Held | Jaw contact | Finger angle ° | Wrist roll at grip ° | Lift target reversals | Lift achieved reversals | Elbow target reversals | Elbow achieved reversals |",
        "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        raw = row["raw_model_targets"]
        achieved = row["achieved_joints"]
        lines.append(
            "| {environment_seed} | {policy_seed} | {success} | {picked_up} | {held} | "
            "{grip} | {angle:.1f} | {roll:.1f} | "
            "{raw_lift:.1%} | {achieved_lift:.1%} | {raw_elbow:.1%} | {achieved_elbow:.1%} |".format(
                environment_seed=row["environment_seed"],
                policy_seed=row["policy_seed"],
                success="pass" if row["success"] else "fail",
                picked_up="pass" if row["picked_up"] else "fail",
                held="pass" if row["held"] else "fail",
                grip="pass" if row["grip"] else "fail",
                angle=row["finger_face_deg"],
                roll=row["grip_roll_deg"],
                raw_lift=raw["shoulder_lift"]["reversal_rate"],
                achieved_lift=achieved["shoulder_lift"]["reversal_rate"],
                raw_elbow=raw["elbow_flex"]["reversal_rate"],
                achieved_elbow=achieved["elbow_flex"]["reversal_rate"],
            )
        )
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_stack_report(rows: list[dict], destination: Path) -> Path:
    task_mode = rows[0]["task_mode"]
    complete = sum(bool(row["success"]) for row in rows)
    command_count = max(len(row["commands"]) for row in rows)
    command_success = [
        sum(
            len(row["commands"]) > index
            and bool(row["commands"][index]["success"])
            for row in rows
        )
        for index in range(command_count)
    ]
    lines = [
        f"# SAPIEN stacking regression {destination.name}",
        "",
        f"Task: `{task_mode}`",
        f"Complete stacks: {complete}/{len(rows)}",
    ]
    for index, passed in enumerate(command_success):
        lines.append(f"Command {index + 1} success: {passed}/{len(rows)}")
    lines.extend(
        [
            "",
            "Accumulated video: [top](overall-top.mp4), [wrist](overall-wrist.mp4).",
            "",
            *FINGER_ANGLE_NOTE,
            "A test with several commands lists one value per command.",
            "",
            finger_angle_summary(
                stage["finger_face_deg"] for row in rows for stage in row["commands"]
            ),
            "",
            "| Environment seed | Policy seed | Complete | Command results | Failure stage | Jaw contact | Finger angle ° | Wrist roll at grip ° |",
            "|---:|---:|---|---|---|---|---:|---:|",
        ]
    )
    for row in rows:
        command_results = ", ".join(
            f"{stage['canonical_command']}: "
            f"{'pass' if stage['success'] else 'fail'}"
            for stage in row["commands"]
        )
        contact = ", ".join(
            "pass" if stage["grip"] else "fail" for stage in row["commands"]
        )
        angles = ", ".join(
            f"{stage['finger_face_deg']:.1f}" for stage in row["commands"]
        )
        rolls = ", ".join(f"{stage['grip_roll_deg']:.1f}" for stage in row["commands"])
        lines.append(
            f"| {row['environment_seed']} | {row['policy_seed']} | "
            f"{'pass' if row['success'] else 'fail'} | {command_results} | "
            f"{row['failure_stage'] or '-'} | {contact} | {angles} | {rolls} |"
        )
    path = destination / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_stack_batch(
    count: int,
    *,
    task_mode: str,
    destination: Path,
    steps_per_command: int = DEFAULT_STACK_STEPS,
    environment_seed_base: int = 0,
    policy_seed_base: int = 0,
) -> list[dict]:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if task_mode not in STACK_COMMANDS:
        raise ValueError(f"unsupported stacking task {task_mode!r}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite batch {destination}")
    destination.mkdir(parents=True)
    checkpoint, policy, pre, post = _load_policy()
    command_count = len(STACK_COMMANDS[task_mode])
    env = create_policy_environment(
        "sapien",
        horizon=steps_per_command * command_count,
        task_mode=task_mode,
    )
    rows = []
    try:
        for index in range(count):
            seed = environment_seed_base + index
            rows.append(
                run_stack_episode(
                    env,
                    policy,
                    pre,
                    post,
                    checkpoint,
                    destination / f"seed-{seed:05d}",
                    task_mode=task_mode,
                    environment_seed=seed,
                    policy_seed=policy_seed_base + index,
                    steps_per_command=steps_per_command,
                )
            )
    finally:
        env.close()
    (destination / "summary.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    accumulate(
        [
            destination / f"seed-{environment_seed_base + index:05d}"
            for index in range(count)
        ],
        destination,
        "overall",
    )
    _write_stack_report(rows, destination)
    (destination / "COMPLETE.json").write_text(
        json.dumps(
            {
                "complete": True,
                "episodes": count,
                "task_mode": task_mode,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def run_batch(
    count: int,
    *,
    destination: Path,
    task_mode: str = "pick-place",
    steps: int = DEFAULT_STEPS,
    environment_seed_base: int = 0,
    policy_seed_base: int = 0,
) -> list[dict]:
    if count < 1:
        raise ValueError(f"count must be positive, got {count}")
    if task_mode not in PICK_PLACE_TASKS:
        raise ValueError(f"unsupported pick-and-place task {task_mode!r}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite batch {destination}")
    destination.mkdir(parents=True)
    checkpoint, policy, pre, post = _load_policy()
    env = create_policy_environment("sapien", horizon=steps, task_mode=task_mode)
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
        json.dumps(
            {"complete": True, "episodes": count, "task_mode": task_mode},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    # The three-cube runner builds on this module, so it is loaded from here.
    from so101_smolvla import cube_commands, three_cube_rollout

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument(
        "--steps",
        type=int,
        help=(
            f"steps per bin episode (default {DEFAULT_STEPS}), per stacking "
            f"command (default {DEFAULT_STACK_STEPS}) or per three-cube command "
            f"(default {three_cube_rollout.DEFAULT_STEPS_PER_PICK})"
        ),
    )
    parser.add_argument("--environment-seed-base", type=int, default=0)
    parser.add_argument("--policy-seed-base", type=int, default=0)
    parser.add_argument("--backend", choices=("sapien",), default="sapien")
    parser.add_argument("--task", choices=tuple(TASK_SPECS), default="pick-place")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--pictures",
        choices=tuple(three_cube_rollout.PICTURE_MODES),
        help=f"what SmolVLA is shown in the {THREE_CUBE_TASK} task",
    )
    parser.add_argument(
        "--order",
        help=(
            f"the order of the cubes in the {THREE_CUBE_TASK} task, such as "
            "green,red,blue; without it each layout draws its own"
        ),
    )
    parser.add_argument(
        "--outlines",
        choices=tuple(three_cube_rollout.OUTLINE_SOURCES),
        help=(
            f"where the {THREE_CUBE_TASK} task takes the outlines from that the "
            "pictures are edited by: the simulator (the default) or SAM 3, "
            "asked for every picture SmolVLA plans on"
        ),
    )
    parser.add_argument(
        "--sentences",
        help=(
            f"the commands of the {THREE_CUBE_TASK} task as sentences through "
            "Gemma 4 instead of colours: 'built-in' for the test's own ten "
            "phrasings per colour, or a JSON file with a list of sentences per "
            "layout (so101_smolvla.cube_commands)"
        ),
    )
    parser.add_argument(
        "--spoken",
        action="store_true",
        help="say the sentences in the synthetic voice instead of typing them",
    )
    args = parser.parse_args(argv)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.task != THREE_CUBE_TASK and (
        args.pictures or args.order or args.outlines or args.sentences or args.spoken
    ):
        parser.error(
            f"--pictures, --order, --outlines, --sentences and --spoken belong to "
            f"--task {THREE_CUBE_TASK}"
        )
    if args.task == THREE_CUBE_TASK:
        if args.pictures is None:
            parser.error(f"--task {THREE_CUBE_TASK} needs --pictures")
        if args.order and args.sentences:
            parser.error("--order and --sentences: the commands are colours or sentences")
        if args.spoken and not args.sentences:
            parser.error("--spoken needs --sentences")
        try:
            order = (
                None
                if args.order is None
                else three_cube_rollout.parse_order(args.order)
            )
            if args.sentences is None:
                sentences = None
            elif args.sentences == "built-in":
                sentences = cube_commands.layout_sentences(
                    args.count, args.environment_seed_base
                )
            else:
                sentences = cube_commands.read_sentences(Path(args.sentences))
        except ValueError as exc:
            parser.error(str(exc))
        outlines = args.outlines or "simulator"
        # A run with the simulator's outlines keeps the folder name it had.
        source = "" if outlines == "simulator" else f"-{outlines}"
        if sentences is not None:
            source += "-gemma-spoken" if args.spoken else "-gemma-typed"
        destination = args.output or (
            VLA_TESTING
            / f"{task_spec(args.task).output_label}-{args.pictures}{source}-{stamp}"
        )
        rows = three_cube_rollout.run_three_cube_batch(
            args.count,
            destination=destination,
            pictures=args.pictures,
            steps_per_pick=(
                three_cube_rollout.DEFAULT_STEPS_PER_PICK
                if args.steps is None
                else args.steps
            ),
            environment_seed_base=args.environment_seed_base,
            policy_seed_base=args.policy_seed_base,
            order=order,
            outlines=outlines,
            sentences=sentences,
            spoken=args.spoken,
        )
    elif args.task in PICK_PLACE_TASKS:
        destination = args.output or (
            VLA_TESTING / f"{task_spec(args.task).output_label}-{stamp}"
        )
        rows = run_batch(
            args.count,
            destination=destination,
            task_mode=args.task,
            steps=DEFAULT_STEPS if args.steps is None else args.steps,
            environment_seed_base=args.environment_seed_base,
            policy_seed_base=args.policy_seed_base,
        )
    else:
        destination = args.output or random_stacking_output_directory(
            VLA_TESTING, "sapien", args.task, stamp
        )
        rows = run_stack_batch(
            args.count,
            task_mode=args.task,
            destination=destination,
            steps_per_command=(
                DEFAULT_STACK_STEPS if args.steps is None else args.steps
            ),
            environment_seed_base=args.environment_seed_base,
            policy_seed_base=args.policy_seed_base,
        )
    print(
        f"succeeded {sum(1 for row in rows if row['success'])} of {len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
