"""Does a nearer top camera with a wider lens draw the robot more like the real pictures?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    top_camera_height_scan.py [<result .json in analysis/> <vertical lens in degrees> ...]

No model, nothing simulated forward, nothing fitted. A camera that looks at a
flat table from farther away through a narrower lens shows the table almost
as a nearer camera with a wider lens does; what tells the two apart is how
far the parts of the robot that stand high above the table are drawn away
from the point under the camera. The scene's top camera is moved along its
line of sight towards the table, with the lens widened so that the table
point it looks at keeps its size in the picture, and at each place the drawn
robot is compared with the dark robot of the 114 real top pictures the camera
was measured on (three moments of 38 recordings). Joint zero points stay as
configured.

Per lens and moment: the overlap of the two outlines (shared area over joint
area) and how far, on average, real robot pixels lie outside the drawn robot
and drawn pixels outside the real one. The overlap is also given for the
even- and the odd-numbered recordings apart.

Writes analysis/top-camera-height-scan.json, or the named result, and
refuses to overwrite it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from mani_skill.utils.structs.pose import Pose

from fit_camera_from_robot import dark_robot
from release_aim_check import recorded, recording_rig_scene, tip_at
from so101_smolvla.sapien_stack_env import RECORDING_RIG_ABSENT_LINKS

ANALYSIS = Path.home() / ".cache" / "so101-stack-training" / "analysis"
MOMENTS = {0: "start", 1: "25 frames before the gripper closes", 2: "release"}
PICTURE_HEIGHT = 480
VERTICAL_LENSES_DEG = (51.3, 55.0, 59.17, 63.0, 67.0, 72.0)
FAR_PX = 30.0


class TopCameraOnItsLineOfSight:
    """The scene's top camera, to be moved along its line of sight with the lens widened to match."""

    def __init__(self, inner) -> None:
        self._camera = inner._sensors["top"].camera
        self._mount = inner.agent.robot.links_map["top_camera_optical_frame"].pose
        self._today = self._camera.get_global_pose().sp
        self._focal_today = float(self._camera.get_intrinsic_matrix()[0, 1, 1])
        # The camera looks along its x axis.
        self._forward = self._today.to_transformation_matrix()[:3, 0]
        self._to_table = (inner.TABLE_SURFACE_Z - self._today.p[2]) / self._forward[2]

    def use(self, vertical_lens_deg: float) -> float:
        """Give the camera this lens, at the distance at which the table point it looks at keeps its size.

        Returns the camera's height above the robot base in metres.
        """
        focal = 0.5 * PICTURE_HEIGHT / np.tan(np.deg2rad(vertical_lens_deg) / 2)
        place = (np.asarray(self._today.p)
                 + (1.0 - focal / self._focal_today) * self._to_table * self._forward)
        self._camera.set_local_pose(
            (self._mount.inv() * Pose.create_from_pq(p=place.tolist(), q=list(self._today.q))).sp)
        self._camera.set_focal_lengths(float(focal), float(focal))
        return float(place[2])


def main(output: str = "top-camera-height-scan.json", *lenses: str) -> None:
    result_file = ANALYSIS / output
    if result_file.exists():
        raise SystemExit(f"{result_file} exists; choose another name")
    chosen = tuple(float(lens) for lens in lenses) or VERTICAL_LENSES_DEG
    real = np.load(ANALYSIS / "top-frames-for-calibration.npz")
    darks = [dark_robot(picture) for picture in real["top"]]
    to_dark = [
        np.minimum(cv2.distanceTransform((~dark).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX)
        for dark in darks
    ]
    readings = [
        recorded()["state"][recorded()["episode"] == number][frame]
        for number, frame in zip(real["episode"].tolist(), real["frame"].tolist())
    ]
    moments = real["moment"]
    even = real["episode"] % 2 == 0
    env = recording_rig_scene()
    result: dict[str, dict] = {}
    try:
        inner = env._env.unwrapped
        robot_ids = [
            int(link.per_scene_id[0]) for link in inner.agent.robot.links
            if link.name not in RECORDING_RIG_ABSENT_LINKS
        ]
        camera = TopCameraOnItsLineOfSight(inner)

        for lens in chosen:
            height = camera.use(lens)
            rows = []
            for reading, dark, distance in zip(readings, darks, to_dark):
                tip_at(inner, reading)
                ids = inner.get_obs()["sensor_data"]["top"]["segmentation"][0, ..., 0].cpu().numpy()
                drawn = np.isin(ids, robot_ids)
                to_drawn = np.minimum(
                    cv2.distanceTransform((~drawn).astype(np.uint8), cv2.DIST_L2, 3), FAR_PX)
                rows.append((
                    (dark & drawn).sum() / (dark | drawn).sum(),
                    to_drawn[dark].mean(), distance[drawn].mean(),
                ))
            rows = np.array(rows)
            result[f"{lens} degree lens"] = {
                "camera_above_robot_base_m": round(height, 3),
                **{
                    label: {
                        "overlap_median": round(float(np.median(rows[moments == moment, 0])), 3),
                        "overlap_median_even_recordings": round(
                            float(np.median(rows[(moments == moment) & even, 0])), 3),
                        "overlap_median_odd_recordings": round(
                            float(np.median(rows[(moments == moment) & ~even, 0])), 3),
                        "real_outside_drawn_px_mean": round(float(rows[moments == moment, 1].mean()), 2),
                        "drawn_outside_real_px_mean": round(float(rows[moments == moment, 2].mean()), 2),
                    }
                    for moment, label in MOMENTS.items()
                },
            }
            print(lens, json.dumps(result[f"{lens} degree lens"]), flush=True)
    finally:
        env.close()
    result_file.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(*sys.argv[1:])
