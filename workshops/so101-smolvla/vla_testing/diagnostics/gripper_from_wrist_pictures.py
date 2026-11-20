"""How far open is the recording robot's gripper in its wrist pictures, against its gripper reading?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after wrist_frames_before_grasp.py and release_aim_check.py
render and predict (their files hold the real wrist pictures):
    gripper_from_wrist_pictures.py

The wrist camera sits on the hand, so its picture shows the two black jaws at
the same place whatever the arm does; only the moving jaw changes with the
gripper. The stack2-recording-rig scene draws its hand at every gripper value
from -10 to 60 per cent. For a real wrist picture, the gripper value it shows
is the one whose drawn hand agrees best with the black pixels of the picture,
counted over the pixels the moving jaw passes through.

Two sets of real pictures:
  empty     5 to 60 frames before the gripper closes on the red cube
  carrying  25 frames before the human lets the red cube go
If the recording robot's gripper numbers mean the same opening as the
simulated robot's, the value shown equals the recorded reading in both sets.

Writes analysis/gripper-from-wrist-pictures.json and, per picture,
analysis/gripper-from-wrist-pictures.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from release_aim_check import recording_rig_scene

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
# The real hand is black; nothing else in a wrist picture is this dark.
DARKEST_TABLE_VALUE = 70
GRIPPER_VALUES = np.arange(-10.0, 60.5, 1.0)


def spread(values) -> list[float]:
    """10 %, median and 90 % of the values."""
    return [round(float(v), 1) for v in np.percentile(np.asarray(values, dtype=np.float64), [10, 50, 90])]


def main() -> None:
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    empty = np.load(ANALYSIS / "wrist-frames-before-grasp.npz")
    carrying = np.load(ANALYSIS / "release-aim-predictions.npz")
    carrying_states = np.load(ANALYSIS / "release-aim-pictures.npz")
    if carrying["episode"].tolist() != carrying_states["episode"].tolist():
        raise RuntimeError("the two release files do not hold the same recordings")
    sets = {
        "empty": (
            empty["wrist"],
            np.array([data["state"][data["episode"] == number][frame, 5]
                      for number, frame in zip(empty["episode"].tolist(), empty["frame"].tolist())]),
            empty["episode"],
        ),
        "carrying": (carrying["real_wrist"], carrying_states["state"][:, 5], carrying["episode"]),
    }

    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        robot_ids = [int(link.per_scene_id[0]) for link in inner.agent.robot.links]
        arm = carrying_states["state"][0, :5].astype(np.float64)
        drawn = []
        for value in GRIPPER_VALUES:
            qpos = inner._simulated_qpos(np.array([*arm, value]))
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            ids = inner.get_obs()["sensor_data"]["wrist"]["segmentation"][0, ..., 0].cpu().numpy()
            drawn.append(np.isin(ids, robot_ids))
    finally:
        env.close()
    drawn = np.array(drawn)
    # Pixels that are hand at some gripper values and not at others.
    passed = drawn.any(axis=0) & ~drawn.all(axis=0)
    drawn_passed = drawn[:, passed]
    print(f"the moving jaw passes through {int(passed.sum())} pixels of the wrist picture", flush=True)

    def shown(picture: np.ndarray) -> tuple[float, np.ndarray]:
        black = (picture.max(axis=2) < DARKEST_TABLE_VALUE)[passed]
        agreement = (drawn_passed == black).mean(axis=1)
        return float(GRIPPER_VALUES[int(agreement.argmax())]), agreement

    summary: dict[str, dict] = {}
    stored: dict[str, np.ndarray] = {}
    for name, (pictures, readings, episodes) in sets.items():
        values, best, at_reading = [], [], []
        for picture, reading in zip(pictures, readings):
            value, agreement = shown(picture)
            values.append(value)
            best.append(agreement.max())
            at_reading.append(agreement[int(np.abs(GRIPPER_VALUES - reading).argmin())])
        values = np.array(values)
        difference = values - readings
        summary[name] = {
            "pictures": len(values),
            "recorded gripper reading, per cent (10 %, median, 90 %)": spread(readings),
            "gripper value the picture shows, per cent": spread(values),
            "value shown minus reading, per cent": spread(difference),
            "share of the passed pixels that agree at the value shown, per cent": spread(
                100.0 * np.array(best)),
            "share of the passed pixels that agree at the recorded reading, per cent": spread(
                100.0 * np.array(at_reading)),
        }
        if name == "empty":
            # A wrong zero shifts every reading alike; a wrong scale grows with the reading.
            slope, offset = np.polyfit(readings, values, 1)
            summary[name]["value shown = slope x reading + offset"] = {
                "slope": round(float(slope), 3), "offset_pct": round(float(offset), 1)}
            for low, high in ((0, 15), (15, 25), (25, 40)):
                chosen = (readings >= low) & (readings < high)
                summary[name][f"value shown minus reading, readings {low} to {high} per cent"] = {
                    "pictures": int(chosen.sum()),
                    "per cent": spread(difference[chosen]) if chosen.any() else None,
                }
        stored[f"{name}_episode"] = episodes
        stored[f"{name}_reading"] = readings
        stored[f"{name}_shown"] = values
    print(json.dumps(summary, indent=1), flush=True)
    (ANALYSIS / "gripper-from-wrist-pictures.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    np.savez(ANALYSIS / "gripper-from-wrist-pictures.npz", **stored)


if __name__ == "__main__":
    main()
