"""How far is the joint reading from the joint command, real robot against simulator?

Run with any Python that has numpy:
    command_minus_reading.py

Reads ~/.cache/so101-stack-training/analysis/states-real-stacking.npz (200
human recordings on the real robot) and states-sim-cube20.npz (simulated
demonstrations of the upstream simulator). For both it reports, per joint, the
command minus the reading:
  * while the arm is nearly still (the command changed less than 0.2 units per
    frame over five frames) -- the standing offset the motors leave;
  * at the grasp of the real recordings (ten frames after the gripper closes),
    because that offset decides how high the gripper is at the cube.
Writes analysis/command-minus-reading.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
JOINTS = ("pan", "lift", "elbow", "wrist_flex", "wrist_roll", "gripper")
STILL = 0.2


def percentiles(values: np.ndarray) -> list[float]:
    return [round(float(v), 2) for v in np.percentile(values, [10, 50, 90])]


def still_frames(episode: np.ndarray, action: np.ndarray) -> np.ndarray:
    """Frames whose command moved less than STILL per frame over the last five frames."""
    still = np.zeros(len(action), dtype=bool)
    for number in np.unique(episode):
        chosen = np.flatnonzero(episode == number)
        if len(chosen) < 6:
            continue
        moved = np.abs(action[chosen][5:, :5] - action[chosen][:-5, :5]).max(axis=1) / 5.0
        still[chosen[5:]] = moved < STILL
    return still


def describe(name: str) -> dict:
    data = np.load(ANALYSIS / f"states-{name}.npz")
    episode, state, action = data["episode"], data["state"], data["action"]
    gap = action - state
    still = still_frames(episode, action)
    return {
        "frames": int(len(gap)),
        "still_frames": int(still.sum()),
        "command_minus_reading_all_frames": {
            joint: percentiles(gap[:, index]) for index, joint in enumerate(JOINTS)},
        "command_minus_reading_still_frames": {
            joint: percentiles(gap[still, index]) for index, joint in enumerate(JOINTS)},
    }


def at_grasp() -> dict:
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state, action = data["episode"], data["state"], data["action"]
    places = np.load(ANALYSIS / "cube-placement.npz")
    rows, lifts, elbows = [], [], []
    for number, close, release in zip(
        places["training_episode"].tolist(),
        places["training_close_frame"].tolist(),
        places["training_release_frame"].tolist(),
    ):
        chosen = np.flatnonzero(episode == number)
        index = chosen[min(int(close) + 10, int(release) - 1)]
        rows.append(action[index] - state[index])
        lifts.append(state[index, 1])
        elbows.append(state[index, 2])
    rows = np.array(rows)
    return {
        "recordings": int(len(rows)),
        "command_minus_reading": {
            joint: percentiles(rows[:, index]) for index, joint in enumerate(JOINTS)},
        "reading_lift_deg": percentiles(np.array(lifts)),
        "reading_elbow_deg": percentiles(np.array(elbows)),
    }


def main() -> None:
    result = {
        "real stacking recordings": describe("real-stacking"),
        "simulated demonstrations, 20 mm cube": describe("sim-cube20"),
        "real stacking recordings at the grasp": at_grasp(),
    }
    print(json.dumps(result, indent=1))
    (ANALYSIS / "command-minus-reading.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
