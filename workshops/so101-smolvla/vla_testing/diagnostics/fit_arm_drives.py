"""Which joint drives make the simulated arm follow commands as the recording robot did?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    fit_arm_drives.py

The recordings hold, for every frame, the command sent to each joint and the
angle the joint then read. On the recording robot the reading stays behind
the command: its motors hold the arm's weight with a position error (elbow
about 2 degrees) and lag while moving.

The simulated arm has no weight: the simulator library switches gravity off
for every link of a robot (mani_skill/agents/base_agent.py, "balance passive
force"), and its drives (stiffness 1000 N*m/rad, damping 100 N*m*s/rad on
every arm joint) therefore stand exactly on the command. Without weight the
stiffness of a drive changes nothing at rest; the first run of this script
showed that (logs/arm-drives-1795132050.log).

This script gives the arm its weight, plays the recorded commands of some
recordings in the stack2-recording-rig scene (physics only, no pictures, no
model) and compares the simulated readings, frame by frame, with the
recorded ones. A drive pulls its joint towards the command with
    torque = stiffness * (command - angle) - damping * speed,
so there are two numbers per joint: the stiffness, and the time
damping / stiffness the joint takes to follow.

  1. every arm joint gets the same pair from a grid; each joint keeps the
     pair under which its own readings came closest to the recorded ones
  2. with the other joints at their choice, each joint's pair is varied
     around it, twice over all joints
  3. the result is checked on recordings that were not used for the fit

Writes analysis/arm-drives.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
from command_minus_reading import still_frames
from so101_sim.lerobot_env import So101SimEnv
from so101_sim.robots.so101_base.so101 import ARM_FORCE_LIMIT
from so101_smolvla.sapien_config import environment_arguments
from so101_smolvla.stack_recording_rig import ARM_JOINTS, RECORDING_RIG

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
SIMULATOR_DRIVE = (1000.0, 0.1)  # stiffness in N*m/rad, damping / stiffness in s
STIFFNESSES = (5.0, 7.0, 10.0, 14.0, 20.0, 28.0, 40.0, 60.0, 100.0, 200.0, 1000.0)
FOLLOW_TIMES = (0.05, 0.08, 0.12, 0.18)
VARIATIONS = ((0.85, 1.0), (1.18, 1.0), (1.0, 0.8), (1.0, 1.25))
VARIATION_ROUNDS = 2
RECORDINGS_PER_GROUP = 20


def main() -> None:
    places = np.load(ANALYSIS / "cube-placement.npz")
    known = places["training_episode"].tolist()
    close_of = dict(zip(known, places["training_close_frame"].tolist()))
    release_of = dict(zip(known, places["training_release_frame"].tolist()))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode = data["episode"]
    state, action = data["state"].astype(np.float64), data["action"].astype(np.float64)
    still = still_frames(episode, action) & still_frames(episode, state)
    # Two groups spread over all recordings: one to fit on, one to check on.
    spread = np.linspace(0, len(known) - 1, 2 * RECORDINGS_PER_GROUP).astype(int)
    fit_group = [known[index] for index in spread[0::2]]
    check_group = [known[index] for index in spread[1::2]]
    zero_deg = np.rad2deg(RECORDING_RIG.arm_zero_points_rad)
    arm = len(ARM_JOINTS)

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        joints = inner.agent.robot.active_joints[:arm]
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)
        table = float(inner.TABLE_SURFACE_Z)

        def pose(numbers: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(
                inner._simulated_qpos(numbers)[None], dtype=torch.float32, device=inner.device)

        def tip() -> np.ndarray:
            return np.mean([finger.pose.p[0].cpu().numpy() for finger in fingers], axis=0)

        def posed_tip(numbers: np.ndarray) -> np.ndarray:
            inner.agent.robot.set_qpos(pose(numbers))
            return tip()

        # Cubes stand where the gripper tip is when the human grasps and lets go.
        layouts, human_grasp_height = {}, {}
        for number in fit_group + check_group:
            chosen = np.flatnonzero(episode == number)
            grasp = chosen[min(int(close_of[number]) + 10, int(release_of[number]) - 1)]
            grasp_tip, release_tip = posed_tip(state[grasp]), posed_tip(state[chosen[int(release_of[number])]])
            layouts[number] = {"red": grasp_tip[:2].tolist(), "blue": release_tip[:2].tolist()}
            human_grasp_height[number] = 1000.0 * (grasp_tip[2] - table)

        def set_weight(has_weight: bool) -> None:
            for link in inner.agent.robot.links:
                link.disable_gravity = not has_weight

        def set_drives(drives: list[tuple[float, float]]) -> None:
            for joint, (stiffness, follow_time) in zip(joints, drives):
                joint.set_drive_properties(
                    stiffness, stiffness * follow_time, force_limit=ARM_FORCE_LIMIT, mode="force")

        def replay(drives: list[tuple[float, float]], group: list[int]) -> dict:
            """Play the commands of a group; return how the simulated readings differ."""
            set_drives(drives)
            difference, at_rest, grasp_height = [], [], []
            for number in group:
                chosen = np.flatnonzero(episode == number)
                env._env.reset(seed=int(number), options={"stack_replay": {
                    "state_index": int(number), "layout": layouts[number]}})
                if not np.isclose(float(joints[0].stiffness[0]), drives[0][0]):
                    raise RuntimeError("the reset replaced the joint drives")
                grasp = min(int(close_of[number]) + 10, int(release_of[number]) - 1)
                readings = []
                for frame, index in enumerate(chosen):
                    inner._step_action(pose(action[index]))
                    readings.append(
                        np.rad2deg(inner.agent.robot.get_qpos()[0, :arm].cpu().numpy()) - zero_deg)
                    if frame == grasp:
                        grasp_height.append(1000.0 * (tip()[2] - table) - human_grasp_height[number])
                readings = np.array(readings)
                difference.append(readings - state[chosen, :arm])
                at_rest.append((readings - state[chosen, :arm])[still[chosen]])
            difference, at_rest = np.concatenate(difference), np.concatenate(at_rest)
            return {
                "mean_abs": np.abs(difference).mean(axis=0),
                "median_abs": np.median(np.abs(difference), axis=0),
                "at_rest_median": np.median(at_rest, axis=0),
                "tip_at_grasp_minus_human_mm_median": float(np.median(grasp_height)),
            }

        def rounded(measured: dict) -> dict:
            return {
                "simulated_minus_recorded_reading_deg_mean_abs": dict(
                    zip(ARM_JOINTS, np.round(measured["mean_abs"], 2).tolist())),
                "simulated_minus_recorded_reading_deg_median_abs": dict(
                    zip(ARM_JOINTS, np.round(measured["median_abs"], 2).tolist())),
                "simulated_minus_recorded_reading_at_rest_deg_median": dict(
                    zip(ARM_JOINTS, np.round(measured["at_rest_median"], 2).tolist())),
                "tip_at_grasp_minus_human_mm_median": round(
                    measured["tip_at_grasp_minus_human_mm_median"], 1),
            }

        started = time.monotonic()
        result: dict[str, object] = {
            "fitted_on_recordings": fit_group, "checked_on_recordings": check_group,
            "simulator_drive": {"stiffness_Nm_per_rad": SIMULATOR_DRIVE[0],
                                "damping_Nms_per_rad": SIMULATOR_DRIVE[0] * SIMULATOR_DRIVE[1]},
        }
        set_weight(False)
        simulator = {
            "fit": replay([SIMULATOR_DRIVE] * arm, fit_group),
            "check": replay([SIMULATOR_DRIVE] * arm, check_group),
        }
        print(f"two replays of {len(fit_group)} recordings: {time.monotonic() - started:.0f} s", flush=True)
        print("simulator as it is (no weight)", json.dumps(rounded(simulator["fit"])), flush=True)
        set_weight(True)

        # 1. the same pair on every joint; each joint keeps its best pair
        best: list[tuple[float, float]] = [SIMULATOR_DRIVE] * arm
        best_error = np.full(arm, np.inf)
        for stiffness in STIFFNESSES:
            for follow_time in FOLLOW_TIMES:
                measured = replay([(stiffness, follow_time)] * arm, fit_group)
                print(f"stiffness {stiffness:6.1f} follow time {follow_time:.2f}: mean abs "
                      f"{np.round(measured['mean_abs'], 2).tolist()} at rest "
                      f"{np.round(measured['at_rest_median'], 2).tolist()}", flush=True)
                for joint in range(arm):
                    if measured["mean_abs"][joint] < best_error[joint]:
                        best[joint], best_error[joint] = (stiffness, follow_time), measured["mean_abs"][joint]
        print("each joint's best pair", best, flush=True)

        # 2. one joint at a time around its pair, the others at theirs
        together = replay(best, fit_group)
        for _ in range(VARIATION_ROUNDS):
            for joint in range(arm):
                for stiffness_factor, time_factor in VARIATIONS:
                    trial = list(best)
                    trial[joint] = (best[joint][0] * stiffness_factor, best[joint][1] * time_factor)
                    measured = replay(trial, fit_group)
                    if measured["mean_abs"][joint] < together["mean_abs"][joint]:
                        best, together = trial, measured
                print(f"{ARM_JOINTS[joint]}: {best[joint]}", flush=True)

        result["fitted_drives"] = {
            name: {"stiffness_Nm_per_rad": round(stiffness, 2),
                   "damping_Nms_per_rad": round(stiffness * follow_time, 3)}
            for name, (stiffness, follow_time) in zip(ARM_JOINTS, best)
        }
        result["on_fitted_recordings"] = {
            "simulator_as_it_is_no_weight": rounded(simulator["fit"]),
            "weight_and_fitted_drives": rounded(together),
        }
        result["on_checked_recordings"] = {
            "simulator_as_it_is_no_weight": rounded(simulator["check"]),
            "weight_and_simulator_drives": rounded(replay([SIMULATOR_DRIVE] * arm, check_group)),
            "weight_and_fitted_drives": rounded(replay(best, check_group)),
        }
        result["seconds"] = round(time.monotonic() - started)
    finally:
        env.close()
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / "arm-drives.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
