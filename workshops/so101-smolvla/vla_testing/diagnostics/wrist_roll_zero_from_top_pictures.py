"""Where is the zero of the recording robot's wrist roll, read from its top pictures?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    wrist_roll_zero_from_top_pictures.py <result .json in analysis/>

configs/sapien_stack_recording_rig.json holds a zero point for pan, lift,
elbow and wrist flex, measured on the top pictures, and 0.0 for the wrist
roll, which was not measured. A zero point is what has to be added to a
recorded joint number to get the angle of the simulated robot.

No model, nothing simulated forward. For each of the 114 real top pictures
the top camera and the other zero points were measured on (three moments of
38 recordings), the stack2-recording-rig scene draws its robot at the
recorded joint readings with a trial value added to the wrist roll. Turning
the wrist roll moves only the hand, so the drawn robot is compared with the
dark robot of the real picture inside a window around the drawn hand:

  overlap   shared area over joint area of the two outlines in the window

Reported: the mean overlap at every trial value (all pictures, each moment,
and the even- and odd-numbered recordings apart), the trial value that is
best for each picture, and how much the overlap changes over the scan. If the
overlap hardly changes, the outline cannot tell the zero point and the
result says so; nothing is concluded from the place of a flat maximum.

Writes analysis/<result>.json and refuses to overwrite it.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from fit_camera_from_robot import dark_robot
from random_run_grasp_and_release import ANALYSIS
from release_aim_check import recorded, recording_rig_scene, tip_at
from so101_smolvla.sapien_config import HAND_LINKS
from so101_smolvla.sapien_stack_env import RECORDING_RIG_ABSENT_LINKS
from so101_smolvla.stack_recording_rig import ARM_JOINTS

MOMENTS = {0: "start", 1: "25 frames before the gripper closes", 2: "release"}
TRIAL_DEG = tuple(float(value) for value in np.arange(-30.0, 30.1, 2.5))
# The hand with jaws and camera mount fits inside this distance from its centre.
WINDOW_PX = 70.0
WRIST_ROLL = ARM_JOINTS.index("wrist_roll")


def main(output: str) -> None:
    result_file = ANALYSIS / output
    if result_file.exists():
        raise SystemExit(f"{result_file} exists; choose another name")
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    state, episode_of = recorded()["state"], recorded()["episode"]
    env = recording_rig_scene()
    rows = []
    try:
        inner = env._env.unwrapped
        robot_ids = [int(link.per_scene_id[0]) for link in inner.agent.robot.links
                     if link.name not in RECORDING_RIG_ABSENT_LINKS]
        hand_ids = [int(link.per_scene_id[0]) for link in inner.agent.robot.links
                    if link.name in HAND_LINKS]

        def drawn_ids(reading: np.ndarray, trial: float) -> np.ndarray:
            turned = np.array(reading, dtype=np.float64)
            turned[WRIST_ROLL] += trial
            tip_at(inner, turned)
            return inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()

        pixel_rows, pixel_columns = np.indices(real["top"].shape[1:3])
        for number, frame, moment, picture in zip(
            real["episode"].tolist(), real["frame"].tolist(), real["moment"].tolist(), real["top"]
        ):
            reading = state[episode_of == number][frame]
            dark = dark_robot(picture)
            hand = np.isin(drawn_ids(reading, 0.0), hand_ids)
            if hand.sum() < 50:
                print(f"recording {number}, {MOMENTS[moment]}: the hand is not in the picture", flush=True)
                continue
            hand_rows, hand_columns = np.nonzero(hand)
            window = np.hypot(pixel_columns - hand_columns.mean(),
                              pixel_rows - hand_rows.mean()) < WINDOW_PX
            overlaps = []
            for trial in TRIAL_DEG:
                drawn = np.isin(drawn_ids(reading, trial), robot_ids)
                joint = ((dark | drawn) & window).sum()
                overlaps.append(float((dark & drawn & window).sum() / joint) if joint else 0.0)
            row = {
                "recording": number, "moment": moment,
                "wrist_roll_reading_deg": round(float(reading[WRIST_ROLL]), 1),
                "overlap": [round(value, 4) for value in overlaps],
                "best_trial_deg": TRIAL_DEG[int(np.argmax(overlaps))],
                "overlap_gain_of_best_over_zero": round(max(overlaps) - overlaps[TRIAL_DEG.index(0.0)], 4),
            }
            rows.append(row)
            print(json.dumps({key: row[key] for key in row if key != "overlap"}), flush=True)
    finally:
        env.close()

    def curve(chosen: list[dict]) -> dict:
        mean = np.mean([row["overlap"] for row in chosen], axis=0)
        best = [row["best_trial_deg"] for row in chosen]
        low, median, high = np.percentile(best, [25, 50, 75])
        return {
            "pictures": len(chosen),
            "mean_overlap_by_trial": {f"{trial:+.1f}": round(float(value), 4)
                                      for trial, value in zip(TRIAL_DEG, mean)},
            "best_trial_of_the_mean_deg": TRIAL_DEG[int(np.argmax(mean))],
            "mean_overlap_at_zero": round(float(mean[TRIAL_DEG.index(0.0)]), 4),
            "mean_overlap_at_its_best": round(float(mean.max()), 4),
            "mean_overlap_at_its_worst": round(float(mean.min()), 4),
            "best_trial_per_picture_deg": {"quarter": float(low), "median": float(median),
                                           "three_quarters": float(high)},
        }

    groups = {
        "all pictures": rows,
        **{label: [row for row in rows if row["moment"] == moment]
           for moment, label in MOMENTS.items()},
        "even-numbered recordings": [row for row in rows if row["recording"] % 2 == 0],
        "odd-numbered recordings": [row for row in rows if row["recording"] % 2 == 1],
    }
    summary = {name: curve(chosen) for name, chosen in groups.items() if chosen}
    for name, values in summary.items():
        print(name, json.dumps(values), flush=True)
    result_file.write_text(
        json.dumps({"trial_deg": list(TRIAL_DEG), "window_px": WINDOW_PX,
                    "summary": summary, "pictures": rows}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
