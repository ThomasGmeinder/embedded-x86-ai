"""Is the recorded pan number the pan angle, or 1.2 times smaller?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training, from the workshop folder:
    pan_scale_from_cubes.py

A check that needs no camera model. In the first top picture of a recording
both cubes are visible, and the cubes themselves are the ruler: the red one
is 20 mm wide, the blue one 40 mm. That gives the distance between the two
cubes in millimetres straight from the picture. The same distance follows
from the recorded joints (gripper tip at the grasp and at the release), and it
depends on how many degrees the arm really turned per recorded pan degree.
The script reports, for several pan factors, how well the two distances
agree. Writes analysis/pan-scale-from-cubes.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import so101_smolvla.sapien_stack_env as stack_env
from so101_sim.lerobot_env import So101SimEnv
from so101_smolvla.sapien_config import environment_arguments

ROOT = Path.home() / ".cache" / "so101-stack-training"
ANALYSIS = ROOT / "analysis"
OBJECTS = ANALYSIS / "objects"
PICTURES = "stacking training set (real recordings)"
FACTORS = (0.9, 1.0, 1.1, 1.2, 1.3)
# The blob of a cube in the picture is its top face plus a little of its side
# faces, and the top face is nearer to the camera than the table. Both make
# the cube look larger than a flat 20 or 40 mm square on the table would.
# The two limits bracket that: no correction, and 8 % too large.
RULER_CORRECTIONS = {"cube blob as measured": 1.0, "cube blob 8 % too large": 1.08}


def main() -> None:
    per_picture = json.loads((OBJECTS / "object-measurements-per-picture.json").read_text())
    stored = np.load(OBJECTS / "real-first-frames.npz")["episode"].tolist()
    reds = per_picture[PICTURES]["top camera, red cube"]
    blues = per_picture[PICTURES]["top camera, blue cube"]
    places = np.load(ANALYSIS / "cube-placement.npz")
    close_of = dict(zip(places["training_episode"].tolist(), places["training_close_frame"].tolist()))
    release_of = dict(zip(places["training_episode"].tolist(), places["training_release_frame"].tolist()))
    data = np.load(ANALYSIS / "states-real-stacking.npz")
    episode, state = data["episode"], data["state"]

    rows = []
    for number, red, blue in zip(stored, reds, blues):
        if red is None or blue is None or blue["touches_edge"] or red["touches_edge"]:
            continue
        if number not in close_of:
            continue
        chosen = np.flatnonzero(episode == number)
        close, release = int(close_of[number]), int(release_of[number])
        rows.append({
            "recording": int(number),
            "pixels_apart": float(np.hypot(*(np.subtract(red["centre_px"], blue["centre_px"])))),
            "pixels_per_mm": 0.5 * (red["side_px"] / 20.0 + blue["side_px"] / 40.0),
            "grasp": state[chosen[min(close + 10, release - 1)]].astype(np.float64),
            "release": state[chosen[release]].astype(np.float64),
        })

    env = So101SimEnv(**environment_arguments("stack2-training"), episode_length=10)
    try:
        env._env.reset(seed=0)
        inner = env._env.unwrapped
        fingers = (inner.agent.finger1_tip, inner.agent.finger2_tip)

        def tip(reading: np.ndarray, factor: float) -> np.ndarray:
            qpos = stack_env._real_state_to_sim_qpos(reading)
            qpos[0] *= factor
            inner.agent.robot.set_qpos(
                torch.as_tensor(qpos[None], dtype=torch.float32, device=inner.device))
            return 1000.0 * np.mean([f.pose.p[0].cpu().numpy() for f in fingers], axis=0)[:2]

        joints_apart = {
            factor: np.array([
                np.linalg.norm(tip(row["grasp"], factor) - tip(row["release"], factor))
                for row in rows])
            for factor in FACTORS
        }
    finally:
        env.close()

    pan_between = np.array([abs(row["grasp"][0] - row["release"][0]) for row in rows])
    result = {
        "recordings": len(rows),
        "pixels_per_mm_from_the_cubes_median": round(
            float(np.median([row["pixels_per_mm"] for row in rows])), 3),
        "recorded_pan_between_grasp_and_release_deg_median": round(float(np.median(pan_between)), 1),
    }
    # Recordings where the arm turned a lot between the cubes tell the most.
    wide = pan_between >= np.median(pan_between)
    for label, correction in RULER_CORRECTIONS.items():
        picture_apart = np.array([
            row["pixels_apart"] / (row["pixels_per_mm"] / correction) for row in rows])
        table = {}
        for factor in FACTORS:
            ratio = joints_apart[factor] / picture_apart
            table[f"pan factor {factor}"] = {
                "joints_distance_over_picture_distance_median": round(float(np.median(ratio)), 3),
                "same_for_the_half_with_the_larger_turn": round(float(np.median(ratio[wide])), 3),
                "difference_mm_median_abs": round(
                    float(np.median(np.abs(joints_apart[factor] - picture_apart))), 1),
            }
        result[label] = {
            "picture_distance_mm_median": round(float(np.median(picture_apart)), 1),
            **table,
        }
    print(json.dumps(result, indent=1))
    (ANALYSIS / "pan-scale-from-cubes.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
