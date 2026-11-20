"""Where does the outline of the scene's robot differ from the recording robot's in the top picture?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    robot_outline_by_link.py

No model, nothing simulated forward. For each of the 114 real top pictures
that the top camera and the joint zero points were measured on (three moments
of 38 recordings: the start, 25 frames before the gripper closes, the
release), the stack2-recording-rig scene draws its robot at the recorded
joint readings. The drawn robot is compared with the dark robot of the real
picture:

  overlap        shared area over joint area of the two outlines
  real only      real robot pixels the drawn robot does not cover, counted by
                 the drawn link they lie nearest to, and how far outside that
                 link they lie
  drawn only     drawn pixels of each link where the real picture is not dark
  side           for each link, on which side of it the real robot sticks
                 out: along the picture's right (+) or left (-) and down (+)
                 or up (-), as the mean offset of the real-only pixels from
                 the link's centre minus that of the link's own pixels

Writes analysis/robot-outline-by-link.json and
analysis/robot-outline-by-link.png (real only red, drawn only blue).
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from fit_camera_from_robot import dark_robot
from release_aim_check import recorded, recording_rig_scene, tip_at
from so101_smolvla.sapien_stack_env import RECORDING_RIG_ABSENT_LINKS

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
MOMENTS = {0: "start", 1: "25 frames before the gripper closes", 2: "release"}
# A real-only pixel farther than this from every drawn link is not robot
# outline (a cable on the table, the far table edge).
NEAR_PX = 25.0
SHEET_PER_MOMENT = 4


def main() -> None:
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        links = {
            int(link.per_scene_id[0]): link.name
            for link in inner.agent.robot.links
            if link.name not in RECORDING_RIG_ABSENT_LINKS
        }
        rows = []
        tiles: dict[int, list[np.ndarray]] = {moment: [] for moment in MOMENTS}
        for number, frame, moment, picture in zip(
            real["episode"].tolist(), real["frame"].tolist(), real["moment"].tolist(), real["top"]
        ):
            reading = recorded()["state"][recorded()["episode"] == number][frame]
            tip_at(inner, reading)
            ids = inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()
            drawn = np.isin(ids, list(links))
            dark = dark_robot(picture)
            row = {
                "recording": number, "moment": moment,
                "overlap": float((dark & drawn).sum() / (dark | drawn).sum()),
                "real_px": int(dark.sum()), "drawn_px": int(drawn.sum()),
                "links": {},
            }
            # Nearest drawn link of every pixel, and the distance to it.
            distance, labels = cv2.distanceTransformWithLabels(
                (~drawn).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            drawn_rows, drawn_columns = np.nonzero(drawn)
            # Label n belongs to the n-th drawn pixel in row order.
            nearest_link = np.zeros(labels.max() + 1, dtype=np.int64)
            nearest_link[labels[drawn_rows, drawn_columns]] = ids[drawn_rows, drawn_columns]
            link_of = nearest_link[labels]
            real_only = dark & ~drawn & (distance < NEAR_PX)
            for link_id, name in links.items():
                own = ids == link_id
                if own.sum() < 50:
                    continue
                outside = real_only & (link_of == link_id)
                own_rows, own_columns = np.nonzero(own)
                out_rows, out_columns = np.nonzero(outside)
                row["links"][name] = {
                    "drawn_px": int(own.sum()),
                    "drawn_only_px": int((own & ~dark).sum()),
                    "real_only_px": int(outside.sum()),
                    "real_only_distance_px_mean": float(distance[outside].mean()) if outside.any() else 0.0,
                    "real_only_right_px": float(out_columns.mean() - own_columns.mean()) if outside.any() else 0.0,
                    "real_only_down_px": float(out_rows.mean() - own_rows.mean()) if outside.any() else 0.0,
                }
            rows.append(row)
            if len(tiles[moment]) < SHEET_PER_MOMENT:
                tile = np.full((*dark.shape, 3), 255, dtype=np.uint8)
                tile[dark & drawn] = (150, 150, 150)
                tile[dark & ~drawn] = (230, 40, 40)
                tile[drawn & ~dark] = (40, 40, 230)
                tiles[moment].append(tile)
    finally:
        env.close()

    result: dict[str, dict] = {}
    for moment, label in MOMENTS.items():
        chosen = [row for row in rows if row["moment"] == moment]
        names = sorted({name for row in chosen for name in row["links"]})
        per_link = {}
        for name in names:
            found = [row["links"][name] for row in chosen if name in row["links"]]
            per_link[name] = {
                "pictures": len(found),
                **{
                    key: round(float(np.median([link[key] for link in found])), 1)
                    for key in found[0]
                },
            }
        result[label] = {
            "pictures": len(chosen),
            "overlap_median": round(float(np.median([row["overlap"] for row in chosen])), 3),
            "real_robot_px_median": float(np.median([row["real_px"] for row in chosen])),
            "drawn_robot_px_median": float(np.median([row["drawn_px"] for row in chosen])),
            "links": per_link,
        }
    print(json.dumps(result, indent=1), flush=True)
    (ANALYSIS / "robot-outline-by-link.json").write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8")
    sheet = np.concatenate(
        [np.concatenate(tiles[moment], axis=1) for moment in MOMENTS], axis=0)
    cv2.imwrite(
        str(ANALYSIS / "robot-outline-by-link.png"),
        cv2.cvtColor(cv2.resize(sheet, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA),
                     cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
