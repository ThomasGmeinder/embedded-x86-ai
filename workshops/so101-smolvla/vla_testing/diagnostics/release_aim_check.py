"""Where does the model let the red cube go when shown the stacking scene instead of the real pictures?

Run from the workshop folder, in this order:
    release_aim_check.py render    (.venv-sapien) pictures of the recording-rig scene
    release_aim_check.py predict   (.venv) the model's answers
    release_aim_check.py report    (.venv-sapien) where those answers let the cube go
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

Moment looked at: 25 frames before the human lets the red cube go above the
large cube, in the real stacking recordings whose pictures are on disk. The
model's answer to that moment is its next 50 joint targets, so it contains
its own letting go.

"render" brings the scene to that moment the way the recording got there: the
human's recorded commands are played in the stack2-recording-rig scene from
the recording's start, so the red cube sits in the gripper as the simulated
grasp left it. The arm is then set to the recorded readings of the moment,
with the red cube kept where it sits in the hand, and both cameras take their
picture. Recordings in which the replayed commands do not carry the red cube
to that moment are left out and counted.

"predict" gives the unchanged model the recorded joint state, the stacking
sentence and one pair of pictures: real or scene top picture with real or
scene wrist picture.

"report" finds in every answer the first target that opens the gripper, poses
that target on the simulated robot and measures the gripper tip against the
large cube and against the table. The human's own next 50 commands are
measured the same way. Nothing is simulated forward in "report".

Writes analysis/release-aim-pictures.npz (render),
analysis/release-aim-predictions.npz (predict), analysis/release-aim-check.json
and, per recording, analysis/release-aim-check-per-recording.npz (report).

Other pictures of the same moment (release_picture_parts.py) are asked about
and reported with file names in analysis/ as arguments:
    release_aim_check.py predict <pictures .npz> <answers .npz>
    release_aim_check.py report <pictures .npz> <answers .npz> <result .json>
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
PICTURES = ANALYSIS / "release-aim-pictures.npz"
PREDICTIONS = ANALYSIS / "release-aim-predictions.npz"
FRAMES_BEFORE_RELEASE = 25
ANSWER_LENGTH = 50
CARRIED_MM = 10.0
# The gripper counts as opening once its target is this far above the
# narrowest target of the answer so far (per cent of the gripper's range).
OPENING_PCT = 10.0
# label -> (top picture, wrist picture)
PAIRS = {
    "real top + real wrist": ("real_top", "real_wrist"),
    "scene top + scene wrist": ("top", "wrist"),
    "scene top + real wrist": ("top", "real_wrist"),
    "real top + scene wrist": ("real_top", "wrist"),
}


def recording_rig_scene():
    """The stack2-recording-rig scene, long enough to replay any recording up to its release."""
    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from grasp_aim_check import recordings
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    longest = max(release for _, _, release in recordings())
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=longest + 10)
    _configure_robot_appearance(env)
    env._env.reset(seed=0)
    return env


@lru_cache(maxsize=1)
def recorded() -> dict[str, np.ndarray]:
    """Recording number, joint readings and joint commands of every frame of the real recordings."""
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    return {name: data[name] for name in ("episode", "state", "action")}


def tip_at(inner, numbers: np.ndarray) -> np.ndarray:
    """Pose the simulated robot at recorded joint numbers; return its gripper tip in metres."""
    import torch

    qpos = inner._simulated_qpos(np.asarray(numbers, dtype=np.float64))
    inner.agent.robot.set_qpos(
        torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
    return np.mean(
        [finger.pose.p[0].cpu().numpy()
         for finger in (inner.agent.finger1_tip, inner.agent.finger2_tip)], axis=0)


def replay_commands(env, number: int, close: int, release: int):
    """Play the human's commands of one recording in the scene, physics only.

    The cubes stand where the gripper tip is at the recorded grasp and at the
    recorded release. Yields the number of commands played so far: 0 once the
    cubes stand and the arm is at the recording's start, then after every
    command. After n commands the arm reads as in frame n of the recording.
    """
    import torch

    inner = env._env.unwrapped
    chosen = np.flatnonzero(recorded()["episode"] == number)
    state, action = recorded()["state"], recorded()["action"]
    grasp_tip = tip_at(inner, state[chosen[min(close + 10, release - 1)]])
    release_tip = tip_at(inner, state[chosen[release]])
    env._env.reset(
        seed=int(number),
        options={"stack_replay": {
            "state_index": int(number),
            "layout": {"red": grasp_tip[:2].tolist(), "blue": release_tip[:2].tolist()},
        }},
    )
    yield 0
    for played, index in enumerate(chosen, start=1):
        inner._step_action(torch.as_tensor(
            inner._simulated_qpos(action[index])[None],
            dtype=torch.float32, device=inner.device))
        yield played


def scenes_before_release(env, watch=None):
    """Bring the scene to the moment of each recording in turn.

    Yields (recording, frame, recorded reading of that frame) with the arm at
    the recorded reading, the red cube where the replayed grasp left it in
    the hand and the large cube where the replay left it. Recordings in which
    the replayed commands do not carry the red cube are left out and named.
    "watch", if given, is called with the recording's number after every
    replayed command, while the scene stands as the physics left it.
    """
    from grasp_aim_check import recordings

    inner = env._env.unwrapped
    hand = inner.agent.robot.links_map["gripper_link"]
    red = inner.cubes["red"]
    not_carried: list[int] = []

    for number, close, release in recordings():
        frame = release - FRAMES_BEFORE_RELEASE
        for played in replay_commands(env, number, close, release):
            if played == 0:
                red_start = float(red.pose.p[0, 2])
            elif watch is not None:
                watch(number)
            if played == frame:
                break
        lift = 1000.0 * (float(red.pose.p[0, 2]) - red_start)
        if lift <= CARRIED_MM:
            not_carried.append(number)
            continue
        reading = recorded()["state"][recorded()["episode"] == number][frame]
        in_hand = hand.pose.inv() * red.pose
        tip_at(inner, reading)
        red.set_pose(hand.pose * in_hand)
        yield number, frame, reading
    print(f"the replayed commands did not carry the red cube in recordings {not_carried}",
          flush=True)


def scene_facts(env) -> dict[str, np.ndarray]:
    """Where the cubes and the table are in the scene as it stands, in metres."""
    inner = env._env.unwrapped
    return {
        "large_cube": inner.cubes["blue"].pose.p[0].cpu().numpy().copy(),
        "red_cube": inner.cubes["red"].pose.p[0].cpu().numpy().copy(),
        "table_height": np.float64(inner.TABLE_SURFACE_Z),
    }


def render() -> None:
    env = recording_rig_scene()
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "top": [], "wrist": [],
        "large_cube": [], "red_cube": [], "table_height": [],
    }
    try:
        inner = env._env.unwrapped
        for number, frame, reading in scenes_before_release(env):
            seen = env._format_raw_obs(inner.get_obs())["pixels"]
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["top"].append(seen["top"].copy())
            rows["wrist"].append(seen["wrist"].copy())
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            print(f"recording {number}: picture taken", flush=True)
    finally:
        env.close()
    np.savez_compressed(PICTURES, **{name: np.array(values) for name, values in rows.items()})
    print(f"pictures of {len(rows['episode'])} recordings, {FRAMES_BEFORE_RELEASE} frames before "
          f"the human lets go", flush=True)


def pairs_in(stored) -> dict[str, tuple[str, str]]:
    """Picture pairs to ask about, for one pictures file.

    A file may list its pairs itself, in "pairs": rows of (label, top picture,
    wrist picture), where "real_top" and "real_wrist" are the recorded
    pictures. Otherwise a file with extra pictures names them "wrist: <what>"
    or "top: <what>" and each is paired with the real picture of the other
    camera. A file with neither gets the four standard pairs.
    """
    if "pairs" in stored.files:
        return {label: (top, wrist) for label, top, wrist in stored["pairs"].tolist()}
    names = stored.files
    extra = {
        **{f"real top + {name[7:]}": ("real_top", name) for name in names if name.startswith("wrist: ")},
        **{f"{name[5:]} + real wrist": (name, "real_wrist") for name in names if name.startswith("top: ")},
    }
    return extra or PAIRS


def predict(pictures: str = PICTURES.name, predictions: str = PREDICTIONS.name) -> None:
    import glob
    import time

    import pandas as pd

    from compare_real_vs_sim_images import predict as answer
    from measure_objects import FPS, real_first_frame
    from so101_smolvla.policy_env import PolicyObservation
    from so101_smolvla.sapien_config import task_spec
    from so101_smolvla.sapien_rollout import _load_policy

    stored = np.load(ANALYSIS / pictures)
    pairs = pairs_in(stored)
    shown = {name: stored[name] for name in stored.files if stored[name].ndim == 4}
    episodes = pd.concat(
        pd.read_parquet(path)
        for path in sorted(glob.glob(str(ROOT / "meta" / "episodes" / "chunk-000" / "*.parquet")))
    ).set_index("episode_index")
    for camera in ("top", "wrist"):
        column = f"videos/observation.images.{camera}/from_timestamp"
        shown[f"real_{camera}"] = np.stack([
            real_first_frame(camera, float(episodes.loc[number][column]) + frame / FPS)
            for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist())
        ])
    text = task_spec("stack2-recording-rig").instruction
    print(f"task sentence: {text!r}", flush=True)
    checkpoint, policy, pre, post = _load_policy()
    print(f"checkpoint revision {checkpoint.name}", flush=True)

    started = time.monotonic()
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in pairs}
    for index, state in enumerate(stored["state"]):
        for name, (top, wrist) in pairs.items():
            observation = PolicyObservation(
                state=state.astype(np.float32),
                top=shown[top][index], wrist=shown[wrist][index],
            )
            chunks[name].append(answer(policy, pre, post, observation, text, 0))
        print(f"recording {int(stored['episode'][index])} answered", flush=True)
    count = len(stored["state"]) * len(pairs)
    print(f"{count} answers, {(time.monotonic() - started) / count:.2f} s per answer", flush=True)
    np.savez_compressed(
        ANALYSIS / predictions,
        episode=stored["episode"], frame=stored["frame"],
        real_top=shown["real_top"], real_wrist=shown["real_wrist"],
        **{name: np.stack(values) for name, values in chunks.items()},
    )


def opening_index(targets: np.ndarray) -> int | None:
    """First target of an answer that opens the gripper, or None if none does."""
    gripper = targets[:, 5]
    narrowest = np.minimum.accumulate(gripper)
    opening = np.flatnonzero(gripper > narrowest + OPENING_PCT)
    return int(opening[0]) if len(opening) else None


def report(
    pictures: str = PICTURES.name, predictions: str = PREDICTIONS.name,
    output: str = "release-aim-check.json",
) -> None:
    import torch

    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments

    answers = np.load(ANALYSIS / predictions)
    stored = np.load(ANALYSIS / pictures)
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    large = 1000.0 * stored["large_cube"]
    table = 1000.0 * stored["table_height"]
    outward = large[:, :2] / np.linalg.norm(large[:, :2], axis=1, keepdims=True)
    leftward = np.column_stack([-outward[:, 1], outward[:, 0]])

    human_answers = []
    for number, frame in zip(stored["episode"].tolist(), stored["frame"].tolist()):
        commands = data["action"][data["episode"] == number][frame:frame + ANSWER_LENGTH]
        if len(commands) < ANSWER_LENGTH:
            # The recording ends inside the answer's reach: the last command stays.
            commands = np.concatenate(
                [commands, np.repeat(commands[-1:], ANSWER_LENGTH - len(commands), axis=0)])
        human_answers.append(commands)
    sets = {"human": np.array(human_answers)}
    sets.update({name: answers[name] for name in pairs_in(stored)})

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    result: dict[str, dict] = {"recordings": len(large)}
    per_recording: dict[str, np.ndarray] = {}
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def tip(numbers: np.ndarray) -> np.ndarray:
            qpos = inner._simulated_qpos(np.asarray(numbers, dtype=np.float64))
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return 1000.0 * np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)

        for name, chunks in sets.items():
            when = [opening_index(chunk) for chunk in chunks]
            opened = np.array([index is not None for index in when])
            # Where the gripper opens; an answer that never opens is measured at its last target.
            tips = np.array([
                tip(chunk[index if index is not None else ANSWER_LENGTH - 1])
                for chunk, index in zip(chunks, when)
            ])
            off = tips[:, :2] - large[:, :2]
            distance = np.linalg.norm(off, axis=1)
            height = tips[:, 2] - table
            per_recording[f"{name}: tip from large cube and above table, mm"] = np.column_stack(
                [(off * outward).sum(axis=1), (off * leftward).sum(axis=1), height])
            per_recording[f"{name}: opening target"] = np.array(
                [index if index is not None else -1 for index in when])
            if not opened.any():
                result[name] = {"answers_that_open_the_gripper": 0}
                continue
            result[name] = {
                "answers_that_open_the_gripper": int(opened.sum()),
                "opening_target_number_median": float(np.median(
                    [index for index in when if index is not None])),
                "tip_from_large_cube_centre_mm_median": round(float(np.median(distance[opened])), 1),
                "tip_from_large_cube_centre_mm_p90": round(
                    float(np.percentile(distance[opened], 90)), 1),
                "tip_within_15_mm_of_large_cube_centre": int((distance[opened] <= 15.0).sum()),
                "tip_past_the_large_cube_mm_median": round(
                    float(np.median((off * outward).sum(axis=1)[opened])), 1),
                "tip_left_of_the_large_cube_mm_median": round(
                    float(np.median((off * leftward).sum(axis=1)[opened])), 1),
                "tip_above_table_mm_median": round(float(np.median(height[opened])), 1),
                "tip_above_table_mm_p90": round(float(np.percentile(height[opened], 90)), 1),
            }
    finally:
        env.close()
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    np.savez(ANALYSIS / f"{Path(output).stem}-per-recording.npz",
             episode=stored["episode"], **per_recording)


if __name__ == "__main__":
    {"render": render, "predict": predict, "report": report}[sys.argv[1]](*sys.argv[2:])
