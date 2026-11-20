"""Does the scene's arm end where the recording robot's arm ended, for the same commands?

Run from the workshop folder with .venv-sapien and
PYTHONPATH = src : /tmp/Xbotics-SO101-Sim : vla_testing/diagnostics :
    servo_stop_short.py <label>

The recordings hold, for every frame, the command sent to each joint and the
angle the joint then read. On the recording robot a joint stops short of its
command, on the side it came from: its motor does not move the joint while
the command is closer than a fixed angle. The scene's joints are springs with
the arm's weight on them; they have no such angle.

This script
  1. reads that angle for every arm joint from all recordings
     (``stop_short_in_recordings``),
  2. plays the recorded commands in the scene ``stack2-recording-rig``,
     physics only, no pictures, no model, three ways: as the scene is; with
     each joint's target held until the command is more than the joint's
     angle away (``servo_hold.held_targets``); and the same after a small fit of the
     angle and the drives on 40 recordings,
  3. reports, for all recordings and for those not used in the fit, how far
     the scene's gripper tip is from the tip of the recorded reading when the
     human starts to close the gripper and five frames before the human lets
     go, and the scene's own stop-short measured as in step 1.

One scene is kept for all replays. What the simulator keeps from one replay
to the next changes contacts with the cubes; the arm's readings before the
grasp do not depend on it.

Writes analysis/servo-stop-short-<label>.json and
analysis/servo-stop-short-<label>-tip-miss.npz. The label ``trial`` plays
eight recordings, fits nothing and writes nothing. An existing result file is
not overwritten.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the scenes)
from replay_recordings_fresh_scene import FRAMES_AFTER_CLOSING, TASK_MODE, tip_at
from servo_hold import held_targets
from so101_sim.robots.so101_base.so101 import ARM_FORCE_LIMIT
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.sapien_env_worker import _new_environment
from so101_smolvla.stack_recording_rig import ARM_JOINTS, RECORDING_RIG

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
ARM = len(ARM_JOINTS)
# A joint counts as having moved once its reading is this far from where it
# last turned; as standing still when reading and command each stay within
# STILL_DEG over STILL_FRAMES frames.
MOVED_DEG = 0.5
STILL_DEG = 0.06
STILL_FRAMES = 6
FIT_RECORDINGS = 40
ANGLE_FACTORS = (0.0, 0.5, 0.75, 1.0, 1.25)
FOLLOW_TIME_FACTORS = (0.6, 0.8, 1.0)
STIFFNESS_FACTORS = (0.85, 1.18)
MOMENTS = (("when the closing starts", "close", 0), ("5 frames before the release", "release", -5))
MISS_COLUMNS = ("away from the base", "to the left", "up", "size on the table", "size with the height")


def stop_short_in_recordings(courses: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """How each arm joint stands to its command when both stand still.

    ``courses`` holds, per recording, the readings and the commands (frames x
    arm joints, degrees). Per joint: the median of command minus reading
    after the reading last rose and after it last fell, the middle of the two
    (what the arm's weight does) and half their distance (the stop-short).
    """
    result = {}
    for joint, name in enumerate(ARM_JOINTS):
        after = {1: [], -1: []}
        for reading, command in courses:
            angle, wanted = reading[:, joint], command[:, joint]
            direction, turned_at = 0, angle[0]
            for frame in range(len(angle)):
                if abs(angle[frame] - turned_at) > MOVED_DEG:
                    direction = 1 if angle[frame] > turned_at else -1
                    turned_at = angle[frame]
                window = slice(frame - STILL_FRAMES, frame + 1)
                if (
                    direction != 0
                    and frame >= STILL_FRAMES
                    and np.ptp(angle[window]) < STILL_DEG
                    and np.ptp(wanted[window]) < STILL_DEG
                ):
                    after[direction].append(wanted[frame] - angle[frame])
        rose, fell = (float(np.median(after[side])) if after[side] else float("nan") for side in (1, -1))
        result[name] = {
            "still_frames_after_rising": len(after[1]),
            "still_frames_after_falling": len(after[-1]),
            "command_minus_reading_after_rising_deg": round(rose, 2),
            "command_minus_reading_after_falling_deg": round(fell, 2),
            "middle_deg": round((rose + fell) / 2, 2),
            "stop_short_deg": round(max(0.0, (rose - fell) / 2), 2),
        }
    return result


class Scene:
    """The scene of the recording rig, stepped without pictures."""

    def __init__(self, episode, state, action, close_of, release_of):
        self.episode, self.state, self.action = episode, state, action
        self.close_of, self.release_of = close_of, release_of
        self.env = _new_environment(environment_arguments(TASK_MODE), 10)
        self.env._env.reset(seed=0)
        self.inner = self.env._env.unwrapped
        self.joints = self.inner.agent.robot.active_joints[:ARM]
        self.zero_deg = np.rad2deg(RECORDING_RIG.arm_zero_points_rad)
        self.layouts = {}

    def close(self) -> None:
        self.env.close()

    def frames(self, number: int) -> np.ndarray:
        return np.flatnonzero(self.episode == number)

    def layout(self, number: int) -> dict:
        """Cubes where the gripper tip is at the grasp and at the release."""
        if number not in self.layouts:
            frames = self.frames(number)
            close, release = self.close_of[number], self.release_of[number]
            grasp = tip_at(self.inner, self.state[frames[min(close + FRAMES_AFTER_CLOSING, release - 1)]])
            let_go = tip_at(self.inner, self.state[frames[release]])
            self.layouts[number] = {"red": grasp[:2].tolist(), "blue": let_go[:2].tolist()}
        return self.layouts[number]

    def replay(self, drives: np.ndarray, stop_short: np.ndarray, group: list[int]) -> dict:
        """The scene's arm readings (degrees, recording numbers' terms) per recording."""
        layouts = {number: self.layout(number) for number in group}
        for joint, (stiffness, follow_time) in zip(self.joints, drives):
            joint.set_drive_properties(
                stiffness, stiffness * follow_time, force_limit=ARM_FORCE_LIMIT, mode="force")
        readings = {}
        for number in group:
            frames = self.frames(number)
            self.env._env.reset(seed=number, options={"stack_replay": {
                "state_index": number, "layout": layouts[number]}})
            if not np.isclose(float(self.joints[0].stiffness[0]), drives[0][0]):
                raise RuntimeError("the reset replaced the joint drives")
            commands = self.action[frames].copy()
            commands[:, :ARM] = held_targets(
                commands[:, :ARM], self.state[frames[0], :ARM], stop_short)
            course = []
            for command in commands:
                pose = self.inner._simulated_qpos(command)
                self.inner._step_action(
                    torch.as_tensor(pose[None], dtype=torch.float32, device=self.inner.device))
                course.append(
                    np.rad2deg(self.inner.agent.robot.get_qpos()[0, :ARM].cpu().numpy()) - self.zero_deg)
            readings[number] = np.array(course)
        return readings

    def tip_miss(self, readings: dict) -> dict:
        """Tip at the scene's reading minus tip at the recorded reading, mm, per moment."""
        misses = {name: [] for name, _, _ in MOMENTS}
        for number, course in readings.items():
            frames = self.frames(number)
            at = {"close": self.close_of[number], "release": self.release_of[number]}
            for name, key, shift in MOMENTS:
                frame = at[key] + shift
                recorded = self.state[frames[frame]].copy()
                simulated = recorded.copy()
                simulated[:ARM] = course[frame]
                real_tip, scene_tip = tip_at(self.inner, recorded), tip_at(self.inner, simulated)
                away = real_tip[:2] / np.linalg.norm(real_tip[:2])
                left = np.array([-away[1], away[0]])
                delta = 1000.0 * (scene_tip - real_tip)
                misses[name].append([
                    delta[:2] @ away, delta[:2] @ left, delta[2],
                    np.linalg.norm(delta[:2]), np.linalg.norm(delta),
                ])
        return {name: np.array(values) for name, values in misses.items()}

    def joint_error(self, readings: dict) -> np.ndarray:
        """Mean size of scene's reading minus recorded reading per arm joint, all frames."""
        return np.concatenate([
            np.abs(course - self.state[self.frames(number), :ARM])
            for number, course in readings.items()
        ]).mean(axis=0)

    def stop_short(self, readings: dict) -> dict:
        return stop_short_in_recordings([
            (course, self.action[self.frames(number), :ARM]) for number, course in readings.items()
        ])


def described(misses: dict, chosen: np.ndarray) -> dict:
    """Quarter, median, three quarters and counts of the tip miss over some recordings."""
    result = {}
    for name, values in misses.items():
        part = values[chosen]
        result[name] = {
            "recordings": int(len(part)),
            **{
                column: [round(float(np.percentile(part[:, index], share)), 1) for share in (25, 50, 75)]
                for index, column in enumerate(MISS_COLUMNS)
            },
            "size on the table, 90 %": round(float(np.percentile(part[:, 3], 90)), 1),
            "more than 5 mm off on the table": int((part[:, 3] > 5.0).sum()),
            "more than 10 mm off on the table": int((part[:, 3] > 10.0).sum()),
        }
    return result


def main() -> None:
    label = sys.argv[1]
    trial = label == "trial"
    result_path = ANALYSIS / f"servo-stop-short-{label}.json"
    miss_path = ANALYSIS / f"servo-stop-short-{label}-tip-miss.npz"
    if not trial:
        for path in (result_path, miss_path):
            if path.exists():
                raise SystemExit(f"SETUP FAILURE: {path} exists and is not overwritten")

    places = np.load(ANALYSIS / "cube-placement.npz")
    known = [int(n) for n in places["training_episode"]]
    close_of = dict(zip(known, (int(v) for v in places["training_close_frame"])))
    release_of = dict(zip(known, (int(v) for v in places["training_release_frame"])))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode = data["episode"]
    state, action = data["state"].astype(np.float64), data["action"].astype(np.float64)

    recorded = stop_short_in_recordings([
        (state[episode == number, :ARM], action[episode == number, :ARM])
        for number in np.unique(episode)
    ])
    print("recording robot", json.dumps(recorded), flush=True)
    angle = np.array([recorded[name]["stop_short_deg"] for name in ARM_JOINTS])
    drives = np.array([
        (stiffness, damping / stiffness)
        for stiffness, damping in zip(RECORDING_RIG.arm_drive_stiffness, RECORDING_RIG.arm_drive_damping)
    ])

    if trial:
        known = known[:: len(known) // 8][:8]
    fit_group = [] if trial else sorted({
        known[index] for index in np.linspace(0, len(known) - 1, FIT_RECORDINGS).astype(int)
    })
    unseen = np.array([number not in fit_group for number in known])

    started = time.monotonic()
    scene = Scene(episode, state, action, close_of, release_of)
    try:
        states = {"scene as it is": (drives, np.zeros(ARM)),
                  "stop-short as read from the recordings": (drives, angle)}
        if not trial:
            # Every joint gets the same factors; each keeps what brings its
            # own readings closest to the recorded ones.
            best_drives, best_angle = drives.copy(), angle.copy()
            best_error = np.full(ARM, np.inf)
            for angle_factor in ANGLE_FACTORS:
                for time_factor in FOLLOW_TIME_FACTORS:
                    tried_drives = drives * np.array([1.0, time_factor])
                    error = scene.joint_error(scene.replay(tried_drives, angle * angle_factor, fit_group))
                    print(f"angle x {angle_factor:.2f}, follow time x {time_factor:.1f}: "
                          f"{np.round(error, 2).tolist()}", flush=True)
                    better = error < best_error
                    best_error[better] = error[better]
                    best_angle[better] = (angle * angle_factor)[better]
                    best_drives[better] = tried_drives[better]
            for stiffness_factor in STIFFNESS_FACTORS:
                tried_drives = best_drives * np.array([stiffness_factor, 1.0])
                error = scene.joint_error(scene.replay(tried_drives, best_angle, fit_group))
                print(f"stiffness x {stiffness_factor:.2f}: {np.round(error, 2).tolist()}", flush=True)
                better = error < best_error
                best_error[better] = error[better]
                best_drives[better] = tried_drives[better]
            states["stop-short and drives fitted on 40 recordings"] = (best_drives, best_angle)
            print(f"fit: {time.monotonic() - started:.0f} s", flush=True)

        result = {
            "recording_robot": recorded,
            "fitted_on_recordings": [] if trial else fit_group,
            "recordings": len(known),
            "recordings_not_used_in_the_fit": int(unseen.sum()),
            "states": {},
        }
        stored = {"recording": np.array(known), "not_used_in_the_fit": unseen}
        for name, (state_drives, state_angle) in states.items():
            readings = scene.replay(state_drives, state_angle, known)
            misses = scene.tip_miss(readings)
            result["states"][name] = {
                "drives": {
                    joint: {"stiffness_Nm_per_rad": round(float(stiffness), 2),
                            "damping_Nms_per_rad": round(float(stiffness * follow_time), 3),
                            "stop_short_deg": round(float(short), 2)}
                    for joint, (stiffness, follow_time), short in zip(ARM_JOINTS, state_drives, state_angle)
                },
                "reading_error_deg_mean_size": dict(zip(
                    ARM_JOINTS, np.round(scene.joint_error(readings), 2).tolist())),
                "scene_stop_short": scene.stop_short(readings),
                "tip_miss_mm_all_recordings": described(misses, np.ones(len(known), dtype=bool)),
                "tip_miss_mm_recordings_not_used_in_the_fit": described(misses, unseen),
            }
            for moment, values in misses.items():
                stored[f"{name}: {moment}"] = values
            print(name, json.dumps(result["states"][name], indent=1), flush=True)
        result["seconds"] = round(time.monotonic() - started)
    finally:
        scene.close()
    print(f"seconds: {result['seconds']}", flush=True)
    if not trial:
        np.savez(miss_path, columns=np.array(MISS_COLUMNS), **stored)
        result_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {result_path} and {miss_path}", flush=True)


if __name__ == "__main__":
    main()
