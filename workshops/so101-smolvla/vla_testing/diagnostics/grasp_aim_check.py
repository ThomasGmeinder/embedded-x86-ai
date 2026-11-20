"""Does the model aim at the red cube when shown the stacking scene instead of the real pictures?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder, in this order:
    grasp_aim_check.py render      pictures of the recording-rig scene
    grasp_aim_predict.py           (with .venv) the model's answers
    grasp_aim_check.py report      where those answers put the gripper
"report" also takes other answers of the same moment and recordings:
    grasp_aim_check.py report <answers .npz in analysis/> <result .json in analysis/>

Moment looked at: 25 frames before the human closes the gripper on the red
cube, in each of the 38 real stacking recordings whose pictures are on disk.
On the real pictures of that moment the unchanged model puts the gripper as
close to the red cube as the human did (real_picture_aim.py). "render" draws
the same moment in the stack2-recording-rig scene: arm at the recorded
readings, cubes where the scene's robot grasped and released them. The wrist
picture is drawn twice, by the scene's wrist camera (the one measured on the
recording robot) and by the pinned simulator's own wrist camera.

"report" poses every answer's target for the human's grasp frame on the
simulated robot, as recorded numbers, and measures the gripper tip against
the red cube and against the human's own target of that frame. Nothing is
simulated forward.

Writes analysis/grasp-aim-pictures.npz (render), analysis/grasp-aim-check.json
and, per recording, analysis/grasp-aim-check.npz (report).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sapien
import torch
from mani_skill.utils.structs.pose import Pose

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
from so101_sim.envs import WRIST_CAMERA_FOV, _OPTICAL_CONV
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_env_worker import _configure_robot_appearance

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
FRAMES_BEFORE_CLOSING = 25
PICTURE_HEIGHT = 480


def recordings() -> list[tuple[int, int, int]]:
    """(recording, closing frame, release frame) of every recording with pictures on disk."""
    on_disk = set(np.load(ANALYSIS / "wrist-frames-before-grasp.npz")["episode"].tolist())
    places = np.load(ANALYSIS / "cube-placement.npz")
    return [
        (int(number), int(close), int(release))
        for number, close, release in zip(
            places["training_episode"], places["training_close_frame"],
            places["training_release_frame"],
        )
        if int(number) in on_disk and int(close) > FRAMES_BEFORE_CLOSING
    ]


def scenes_before_closing(env, before: int = FRAMES_BEFORE_CLOSING):
    """Bring the scene to the moment of each recording in turn.

    The moment is "before" frames before the human closes the gripper.
    Yields (recording, frame, recorded reading of that frame) with the arm at
    the recorded reading and the cubes standing where the scene's robot
    grasped and released them. A recording that closes earlier than "before"
    frames after its start is left out.
    """
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    inner = env._env.unwrapped
    fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

    def pose_robot(reading: np.ndarray) -> np.ndarray:
        qpos = inner._simulated_qpos(reading)
        inner.agent.robot.set_qpos(
            torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
        return np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)

    def stand(color: str, xy: np.ndarray) -> None:
        height = inner.TABLE_SURFACE_Z + inner.cube_sizes[color] / 2
        inner.cubes[color].set_pose(Pose.create_from_pq(p=torch.tensor(
            [[float(xy[0]), float(xy[1]), height]],
            dtype=torch.float32, device=inner.device)))

    for number, close, release in recordings():
        if close < before:
            continue
        state = data["state"][data["episode"] == number].astype(np.float64)
        frame = close - before
        stand("red", pose_robot(state[min(close + 10, release - 1)]))
        stand("blue", pose_robot(state[release]))
        pose_robot(state[frame])
        yield number, frame, state[frame]


def render() -> None:
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    rows: dict[str, list] = {
        "episode": [], "frame": [], "state": [], "top": [],
        "wrist_measured": [], "wrist_simulator": [],
    }
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        camera = inner._sensors["wrist"].camera
        measured_pose = camera.get_local_pose().sp
        measured_focal = float(camera.get_intrinsic_matrix()[0, 1, 1])
        simulator_focal = PICTURE_HEIGHT / 2 / np.tan(WRIST_CAMERA_FOV / 2)

        def pictures() -> dict[str, np.ndarray]:
            return env._format_raw_obs(inner.get_obs())["pixels"]

        for number, frame, reading in scenes_before_closing(env):
            seen = pictures()
            camera.set_local_pose(sapien.Pose(q=_OPTICAL_CONV))
            camera.set_focal_lengths(simulator_focal, simulator_focal)
            simulator_wrist = pictures()["wrist"]
            camera.set_local_pose(measured_pose)
            camera.set_focal_lengths(measured_focal, measured_focal)
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["top"].append(seen["top"])
            rows["wrist_measured"].append(seen["wrist"])
            rows["wrist_simulator"].append(simulator_wrist)
    finally:
        env.close()
    np.savez_compressed(
        ANALYSIS / "grasp-aim-pictures.npz",
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"rendered {len(rows['episode'])} recordings, "
          f"{FRAMES_BEFORE_CLOSING} frames before the gripper closes", flush=True)


def report(
    predictions: str = "grasp-aim-predictions.npz", output: str = "grasp-aim-check.json"
) -> None:
    answers = np.load(ANALYSIS / predictions)
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    close_of = {number: (close, release) for number, close, release in recordings()}
    # An answer is 50 targets of 6 numbers per recording; a file may also hold pictures.
    labels = [name for name in answers.files if answers[name].ndim == 3]

    # The recording-rig scene poses the arm as the recording robot stood (joint zero points).
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    result: dict[str, dict] = {}
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def tip(numbers: np.ndarray) -> np.ndarray:
            qpos = inner._simulated_qpos(np.asarray(numbers, dtype=np.float64))
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return 1000.0 * np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)

        human_rows, red_rows = [], []
        for number in answers["episode"].tolist():
            chosen = data["episode"] == number
            close, release = close_of[number]
            human_rows.append(tip(data["action"][chosen][close]))
            # The red cube stands where the gripper tip is while it holds the cube (render).
            red_rows.append(tip(data["state"][chosen][min(close + 10, release - 1)])[:2])
        human = np.array(human_rows)
        red = np.array(red_rows)
        outward = red / np.linalg.norm(red, axis=1, keepdims=True)
        leftward = np.column_stack([-outward[:, 1], outward[:, 0]])

        def describe(tips: np.ndarray, gripper: np.ndarray | None) -> dict:
            off = tips[:, :2] - red
            row = {
                "from_red_cube_mm_median": round(float(np.median(np.linalg.norm(off, axis=1))), 1),
                "from_red_cube_mm_p90": round(float(np.percentile(np.linalg.norm(off, axis=1), 90)), 1),
                "past_the_red_cube_mm_median": round(float(np.median((off * outward).sum(axis=1))), 1),
                "left_of_the_red_cube_mm_median": round(float(np.median((off * leftward).sum(axis=1))), 1),
                "within_10_mm_of_the_red_cube": int((np.linalg.norm(off, axis=1) <= 10.0).sum()),
                "tip_height_minus_human_mm_median": round(float(np.median(tips[:, 2] - human[:, 2])), 1),
                "tip_height_minus_human_mm_p90": round(
                    float(np.percentile(tips[:, 2] - human[:, 2], 90)), 1),
            }
            if gripper is not None:
                row["gripper_target_pct_median"] = round(float(np.median(gripper)), 1)
            return row

        result["recordings"] = len(red)
        result["human"] = describe(human, None)
        tips: dict[str, np.ndarray] = {}
        for label in labels:
            chunk = answers[label]
            targets = chunk[:, FRAMES_BEFORE_CLOSING]
            tips[label] = np.array([tip(row) for row in targets])
            result[label] = describe(tips[label], targets[:, 5])
    finally:
        env.close()
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / output).write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    # Per recording, in millimetres, for comparisons between picture sets.
    np.savez(
        (ANALYSIS / output).with_suffix(".npz"),
        episode=answers["episode"], red_cube=red, human_tip=human, **tips,
    )


if __name__ == "__main__":
    {"render": render, "report": report}[sys.argv[1]](*sys.argv[2:])
