"""How stiffly did the recording robot's motors hold the arm against its own weight?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    arm_stiffness_from_recordings.py

On the real robot the reading of a joint stays behind its command while the
arm stands still (command_minus_reading.py): the motor holds the weight of
the arm with a position error. The simulated arm has none; it stands exactly
where it is commanded. A motor that pulls towards its command like a spring
leaves the error  (command - reading) = holding torque / stiffness.

This script takes the still frames of the real stacking recordings, poses the
simulated robot at the readings (with the joint zero points of
analysis/top-camera-calibration.json), asks the simulator for the torque each
joint needs to hold the arm there against gravity, and fits one stiffness per
joint. Nothing is simulated forward.

Writes analysis/arm-stiffness.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import so101_smolvla.sapien_stack_env as stack_env
from command_minus_reading import STILL, still_frames
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
JOINTS = {"lift": 1, "elbow": 2, "wrist_flex": 3}
ZERO_NAMES = ("pan_zero_deg", "lift_zero_deg", "elbow_zero_deg", "wrist_flex_zero_deg")
FRAMES = 4000
# Below this lift reading the folded arm lies on its own base, not on the motors.
LIFT_OFF_THE_REST = -90.0


def main() -> None:
    fitted = json.loads((ANALYSIS / "top-camera-calibration.json").read_text())
    zero = np.zeros(6)
    zero[:4] = [fitted["zero points"]["fitted_on_all"][name] for name in ZERO_NAMES]
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state, action = data["episode"], data["state"].astype(np.float64), data["action"].astype(np.float64)
    # Still: neither the command nor the reading moved over the last five frames.
    still = still_frames(episode, action) & still_frames(episode, state)
    still &= state[:, 1] > LIFT_OFF_THE_REST
    chosen = np.flatnonzero(still)
    rng = np.random.default_rng(0)
    chosen = np.sort(rng.choice(chosen, size=min(FRAMES, len(chosen)), replace=False))
    gap = np.deg2rad(action[chosen] - state[chosen])

    env = So101SimEnv(**environment_arguments("stack2-sim-view"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        articulation = inner.agent.robot._objs[0]
        torques = []
        for reading in state[chosen]:
            qpos = stack_env._real_state_to_sim_qpos(reading + zero)
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            torques.append(np.asarray(articulation.compute_passive_force(
                gravity=True, coriolis_and_centrifugal=False), dtype=np.float64))
        torques = np.array(torques)
        names = [joint.name for joint in inner.agent.robot.active_joints]
    finally:
        env.close()

    result = {
        "still_frames_used": int(len(chosen)),
        "still_means": f"command and reading each moved less than {STILL} per frame over five frames",
        "joint_zero_points_deg": dict(zip(ZERO_NAMES, zero[:4].tolist())),
        "simulated_joint_order": names,
    }
    for name, index in JOINTS.items():
        torque, error = torques[:, index], gap[:, index]
        # error = torque / stiffness, a line through zero; and the same with a free offset.
        through_zero = float(torque @ error / (torque @ torque))
        slope, offset = np.polyfit(torque, error, 1)
        left = error - through_zero * torque
        result[name] = {
            "holding_torque_Nm_p10_p50_p90": np.round(np.percentile(torque, [10, 50, 90]), 3).tolist(),
            "command_minus_reading_deg_p10_p50_p90": np.round(
                np.percentile(np.rad2deg(error), [10, 50, 90]), 2).tolist(),
            "stiffness_Nm_per_rad": round(1.0 / through_zero, 2),
            "error_left_after_fit_deg_median_abs": round(float(np.median(np.abs(np.rad2deg(left)))), 2),
            "error_before_fit_deg_median_abs": round(float(np.median(np.abs(np.rad2deg(error)))), 2),
            "correlation_of_error_with_torque": round(float(np.corrcoef(torque, error)[0, 1]), 3),
            "with_free_offset": {
                "stiffness_Nm_per_rad": round(1.0 / float(slope), 2),
                "offset_deg": round(float(np.rad2deg(offset)), 2),
            },
        }
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / "arm-stiffness.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
