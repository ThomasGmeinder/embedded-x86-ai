"""Do the humans' recorded commands stack the cubes in the scene of the recording rig?

Run from the workshop folder with .venv-sapien and
PYTHONPATH = src : /tmp/Xbotics-SO101-Sim : vla_testing/diagnostics :
    replay_recordings_fresh_scene.py <label> [<first recording> <last recording>]
        [--servo-state "<state>"] [--places-from-pictures]

No model. For every real stacking recording whose grasp and release frame are
known (analysis/cube-placement.npz) the scene ``stack2-recording-rig`` is
built new, as the product's worker builds it for every test, and reset to the
recording's own start pose. The red cube is put where the scene's robot has
its gripper tip ten frames after the human starts to close the gripper, the
large cube where it has the tip when the human lets go, and the recorded
commands are played one per step. If the scene agrees with the recording
rig, the red cube ends on the large cube.

What this cannot know: how the cubes were turned on the real table. The scene
turns each cube by a random angle, so the angle between the jaws and the
cube's faces is stored for every step and can be held against the outcome.

With ``--places-from-pictures`` only the recordings are played whose first
top picture is on this machine, and both cubes stand where and turned as
that picture shows them (analysis/cube-places-from-pictures.npz, written by
cube_places_from_pictures.py). Then nothing about the cubes is assumed, and
on the real table every one of these recordings ended in a stack.

With ``--servo-state`` the arm's joints are given the drives and the
stop-short of one state measured by ``servo_stop_short.py`` (a name under
``states`` in analysis/servo-stop-short-fit-40.json): each joint's target
stays where it is while the command is within the joint's stop-short of it
(``servo_hold.held_targets``). Without it the scene is played as the product builds it.

This takes over the rule of ``~/.cache/so101-stack-training/replay_recordings.py``
(cube places, facts per recording), which kept one scene for all recordings.

Writes analysis/replay-fresh-scene-<label>.json (one row per recording and a
summary) and analysis/replay-fresh-scene-<label>-courses.npz (every step of
every recording). The label ``trial`` writes nothing. An existing result file
is not overwritten.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from mani_skill.utils.structs.pose import Pose
from transforms3d.euler import euler2quat

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the scenes)
from servo_hold import MEASURED, held_targets
from so101_sim.robots.so101_base.so101 import ARM_FORCE_LIMIT
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_env_worker import _new_environment
from so101_smolvla.sapien_grip import grip_facts
from so101_smolvla.stack_recording_rig import ARM_JOINTS

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
PLACES_FROM_PICTURES = ANALYSIS / "cube-places-from-pictures.npz"
ARM = len(ARM_JOINTS)
TASK_MODE = "stack2-recording-rig"
SOURCE, TARGET = "red", "blue"
# The names of the two cubes in PLACES_FROM_PICTURES.
NAME_IN_PICTURES = {SOURCE: "red", TARGET: "large"}
FRAMES_AFTER_CLOSING = 10
LIFT_MM = 10.0
THROWN_MM = 150.0
PUSHED_MM = 20.0
START_POSE_TOLERANCE = 0.01

STACK = "stack"
NOT_LIFTED = "red cube not lifted"
LOST = "lifted, lost before the human lets go"
THROWN = "in the hand at the release, then thrown"
OFF_CENTRE = "rests on the large cube, outside the stack check"
PUSHED = "in the hand at the release, large cube pushed before it"
BESIDE = "in the hand at the release, ends beside the large cube"


def outcome(row: dict) -> str:
    """The one place that says what became of a replayed recording."""
    if row["stacked"]:
        return STACK
    if row["red_lift_mm"] <= LIFT_MM:
        return NOT_LIFTED
    if row["red_lift_at_release_mm"] <= LIFT_MM:
        return LOST
    if row["red_lift_mm"] > THROWN_MM:
        return THROWN
    if row["red_height_error_on_large_at_end_mm"] <= 1.0:
        return OFF_CENTRE
    if row["large_pushed_before_release_mm"] > PUSHED_MM:
        return PUSHED
    return BESIDE


def servo_state(name: str) -> dict[str, np.ndarray]:
    """Stiffness, damping and stop-short of every arm joint in one measured state."""
    drives = json.loads(MEASURED.read_text(encoding="utf-8"))["states"][name]["drives"]
    return {
        key: np.array([drives[joint][key] for joint in ARM_JOINTS])
        for key in ("stiffness_Nm_per_rad", "damping_Nms_per_rad", "stop_short_deg")
    }


def places_from_pictures() -> dict[int, dict[str, tuple[list[float], float]]]:
    """Per recording and cube: place on the table (m) and turn (degrees) in the first top picture."""
    seen = np.load(PLACES_FROM_PICTURES)
    return {
        int(number): {
            color: (
                (seen[f"{name}_place_mm"][row] / 1000.0).tolist(),
                float(seen[f"{name}_turn_deg"][row]),
            )
            for color, name in NAME_IN_PICTURES.items()
        }
        for row, number in enumerate(seen["episode"])
    }


def turn_cube(cube, turn_deg: float) -> None:
    """Turn a cube about the vertical axis where it stands.

    The scene's replay start takes a place for each cube and draws its turn
    at random; this sets the turn the way that start sets it
    (sapien_stack_env.py, ``_initialize_episode``).
    """
    quaternion = torch.tensor(
        np.asarray([euler2quat(0, 0, np.deg2rad(turn_deg))]),
        dtype=torch.float32, device=cube.pose.p.device,
    )
    cube.set_pose(Pose.create_from_pq(cube.pose.p, quaternion))


def fingertips(inner) -> np.ndarray:
    return np.stack([
        finger.pose.p[0].cpu().numpy()
        for finger in (inner.agent.finger1_tip, inner.agent.finger2_tip)
    ])


def tip_at(inner, reading: np.ndarray) -> np.ndarray:
    """The gripper tip of the scene's robot posed at one recorded reading."""
    qpos = inner._simulated_qpos(np.asarray(reading, dtype=np.float64))
    inner.agent.robot.set_qpos(
        torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device)
    )
    return fingertips(inner).mean(axis=0)


def replay(number: int, state: np.ndarray, action: np.ndarray, close: int,
           release: int, grasp_tip: np.ndarray, release_tip: np.ndarray,
           servo: dict[str, np.ndarray] | None,
           seen: dict[str, tuple[list[float], float]] | None) -> tuple[dict, dict]:
    """Play one recording in a scene that has run nothing; its facts and its course.

    ``servo`` is a measured state of the arm's joints (``servo_state``) or
    None for the scene as the product builds it. ``seen`` is each cube's
    place and turn in the recording's first top picture
    (``places_from_pictures``) or None for the cubes at the gripper tip of
    the grasp and of the release, turned at random.
    """
    started = time.monotonic()
    grasp_frame = min(close + FRAMES_AFTER_CLOSING, release - 1)
    layout = {SOURCE: grasp_tip[:2].tolist(), TARGET: release_tip[:2].tolist()}
    if seen is not None:
        layout = {color: seen[color][0] for color in layout}
    env = _new_environment(environment_arguments(TASK_MODE), len(action) + 10)
    try:
        env._env.reset(
            seed=number,
            options={"stack_replay": {"state_index": number, "layout": layout}},
        )
        inner = env._env.unwrapped
        if seen is not None:
            for color, (_, turn_deg) in seen.items():
                turn_cube(inner.cubes[color], turn_deg)
        start_pose_error = float(np.abs(np.asarray(inner.initial_state_real) - state[0]).max())
        if start_pose_error > START_POSE_TOLERANCE:
            raise RuntimeError(
                f"recording {number}: the scene's start pose is {start_pose_error:.3f} "
                "from the recording's first reading"
            )
        inner.set_stack_goal(SOURCE, TARGET)
        if servo is not None:
            joints = inner.agent.robot.active_joints[:ARM]
            for joint, stiffness, damping in zip(
                joints, servo["stiffness_Nm_per_rad"], servo["damping_Nms_per_rad"]
            ):
                joint.set_drive_properties(
                    stiffness, damping, force_limit=ARM_FORCE_LIMIT, mode="force")
            action = action.copy()
            action[:, :ARM] = held_targets(
                action[:, :ARM], state[0, :ARM], servo["stop_short_deg"])
        table = float(inner.TABLE_SURFACE_Z)
        cubes = {color: inner.cubes[color] for color in (SOURCE, TARGET)}

        def place(color: str) -> np.ndarray:
            return cubes[color].pose.p[0].cpu().numpy().copy()

        start = {color: place(color) for color in cubes}
        course = {key: [] for key in (
            "red", "blue", "tip", "finger_gap", "reached", "held", "touching", "finger_face_deg")}
        for command in action:
            obs, _, _, _, _ = env.step(np.asarray(command, dtype=np.float32))
            both = fingertips(inner)
            grip = grip_facts(inner.agent, cubes[SOURCE])
            course["red"].append(place(SOURCE))
            course["blue"].append(place(TARGET))
            course["tip"].append(both.mean(axis=0))
            course["finger_gap"].append(float(np.linalg.norm(both[0] - both[1])))
            course["reached"].append(np.asarray(obs["agent_pos"], dtype=np.float64))
            course["held"].append(bool(inner.agent.is_grasping(cubes[SOURCE])[0]))
            course["touching"].append(grip["contact"])
            course["finger_face_deg"].append(grip["finger_face_deg"])
        course = {key: np.asarray(values) for key, values in course.items()}
        facts = inner.scene_facts()
    finally:
        env.close()

    red, blue = course["red"], course["blue"]
    interface = facts["interfaces"][f"{SOURCE}_on_{TARGET}"]
    lift = 1000.0 * (red[:, 2] - start[SOURCE][2])

    def mm(value: float) -> float:
        return round(1000.0 * float(value), 1)

    row = {
        "recording": number,
        "frames": len(action),
        "grasp_frame": close,
        "release_frame": release,
        "cubes_apart_mm": mm(np.linalg.norm(grasp_tip[:2] - release_tip[:2])),
        "red_cube_from_base_mm": mm(np.linalg.norm(grasp_tip[:2])),
        "grasp_tip_from_red_cube_start_mm": mm(
            np.linalg.norm(grasp_tip[:2] - start[SOURCE][:2])),
        "release_tip_from_large_cube_start_mm": mm(
            np.linalg.norm(release_tip[:2] - start[TARGET][:2])),
        "human_tip_above_table_at_grasp_mm": mm(grasp_tip[2] - table),
        "human_tip_above_table_at_release_mm": mm(release_tip[2] - table),
        "replayed_tip_above_table_at_grasp_mm": mm(course["tip"][grasp_frame, 2] - table),
        "replayed_finger_gap_at_grasp_mm": mm(course["finger_gap"][grasp_frame]),
        "finger_face_deg_when_closing_starts": round(float(course["finger_face_deg"][close]), 1),
        "red_pushed_before_grasp_mm": mm(np.linalg.norm(red[close, :2] - start[SOURCE][:2])),
        "large_pushed_before_release_mm": mm(
            np.linalg.norm(blue[: release + 1, :2] - start[TARGET][:2], axis=1).max()),
        "red_lift_mm": round(float(lift.max()), 1),
        "red_lift_at_release_mm": round(float(lift[release]), 1),
        "red_held_steps": int(course["held"].sum()),
        "red_to_large_at_release_mm": mm(np.linalg.norm(red[release, :2] - blue[release, :2])),
        "red_to_large_at_end_mm": mm(interface["xy_distance"]),
        "red_height_error_on_large_at_end_mm": mm(interface["z_error"]),
        "stacked": bool(interface["supported"]),
        "arm_follows_recording_deg_median": [
            round(float(v), 2) for v in np.median(np.abs(course["reached"] - state), axis=0)
        ],
        "seconds": round(time.monotonic() - started, 1),
    }
    row["outcome"] = outcome(row)
    return row, course


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label")
    parser.add_argument("recordings", nargs="*", type=int, help="first and last recording")
    parser.add_argument("--servo-state")
    parser.add_argument("--places-from-pictures", action="store_true")
    arguments = parser.parse_args()
    if len(arguments.recordings) not in (0, 2):
        parser.error("give no recording or the first and the last one")
    label = arguments.label
    servo = None if arguments.servo_state is None else servo_state(arguments.servo_state)
    seen = places_from_pictures() if arguments.places_from_pictures else None
    result = ANALYSIS / f"replay-fresh-scene-{label}.json"
    courses = ANALYSIS / f"replay-fresh-scene-{label}-courses.npz"
    store = label != "trial"
    if store:
        for path in (result, courses):
            if path.exists():
                raise SystemExit(f"SETUP FAILURE: {path} exists and is not overwritten")

    places = np.load(ANALYSIS / "cube-placement.npz")
    wanted = [int(n) for n in places["training_episode"]]
    close_of = dict(zip(wanted, (int(v) for v in places["training_close_frame"])))
    release_of = dict(zip(wanted, (int(v) for v in places["training_release_frame"])))
    if arguments.recordings:
        first, last = arguments.recordings
        wanted = [n for n in wanted if first <= n <= last]
    if seen is not None:
        wanted = [n for n in wanted if n in seen]
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state, action = data["episode"], data["state"], data["action"]

    # One scene that is never stepped tells where the scene's robot has its
    # gripper tip at a recorded reading.
    started = time.monotonic()
    tips = {}
    measuring = _new_environment(environment_arguments(TASK_MODE), 10)
    try:
        measuring._env.reset(seed=0)
        inner = measuring._env.unwrapped
        for number in wanted:
            frames = np.flatnonzero(episode == number)
            close, release = close_of[number], release_of[number]
            tips[number] = (
                tip_at(inner, state[frames[min(close + FRAMES_AFTER_CLOSING, release - 1)]]),
                tip_at(inner, state[frames[release]]),
            )
    finally:
        measuring.close()

    rows, stored = [], {}
    for number in wanted:
        frames = np.flatnonzero(episode == number)
        row, course = replay(
            number, state[frames], action[frames],
            close_of[number], release_of[number], *tips[number], servo,
            None if seen is None else seen[number],
        )
        rows.append(row)
        for key, values in course.items():
            stored[f"{number}: {key}"] = values
        print(json.dumps(row), flush=True)

    outcomes = Counter(row["outcome"] for row in rows)
    summary = {
        "task_mode": TASK_MODE,
        "a new scene for every recording": True,
        "cubes": (
            "place and turn from the first top picture" if seen is not None
            else "at the gripper tip of the grasp and of the release, turned at random"),
        "servo_state": arguments.servo_state,
        "arm_joints": None if servo is None else {
            key: dict(zip(ARM_JOINTS, values.tolist())) for key, values in servo.items()
        },
        "recordings": len(rows),
        "red cube lifted more than 10 mm": sum(row["red_lift_mm"] > LIFT_MM for row in rows),
        "red cube still up when the human lets go": sum(
            row["red_lift_at_release_mm"] > LIFT_MM for row in rows),
        "red cube ends on the large cube": sum(row["stacked"] for row in rows),
        "outcomes": dict(outcomes.most_common()),
        "medians": {
            key: round(float(np.median([row[key] for row in rows])), 1)
            for key in (
                "human_tip_above_table_at_grasp_mm", "replayed_tip_above_table_at_grasp_mm",
                "human_tip_above_table_at_release_mm", "finger_face_deg_when_closing_starts",
                "red_pushed_before_grasp_mm", "large_pushed_before_release_mm",
                "red_to_large_at_release_mm", "grasp_tip_from_red_cube_start_mm",
                "release_tip_from_large_cube_start_mm",
            )
        },
        "arm_follows_recording_deg_median": np.median(
            [row["arm_follows_recording_deg_median"] for row in rows], axis=0).round(2).tolist(),
        "seconds": round(time.monotonic() - started),
    }
    if sum(outcomes.values()) != len(rows):
        raise RuntimeError("the outcomes do not add up to the recordings")
    print(json.dumps(summary, indent=1), flush=True)
    if store:
        np.savez_compressed(courses, **stored)
        result.write_text(
            json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {result} and {courses}", flush=True)


if __name__ == "__main__":
    main()
