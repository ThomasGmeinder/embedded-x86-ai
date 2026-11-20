"""The scene's top picture of the moment before the grasp, drawn by a nearer top camera.

Run from the workshop folder, in this order:
    grasp_top_picture_at_lens.py <pictures .npz in analysis/> <vertical lens in degrees> ...
                                   (.venv-sapien) the scene's top picture per lens
    release_aim_check.py predict <pictures .npz> <answers .npz>
                                   (.venv) the model's answers, each with the real wrist picture
    grasp_aim_check.py report <answers .npz> <result .json>
                                   (.venv-sapien) where those answers put the gripper
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

Moment: 25 frames before the human closes the gripper on the red cube, the
moment of grasp_aim_check.py, in the same recordings. The real robot's outline
in the top pictures is covered best by a camera that stands nearer than the
configured one, with a wider lens (top_camera_height_scan.py). This asks
whether the unchanged model aims differently at the red cube when the scene
is drawn by such a camera. For every lens given the camera is moved along its
line of sight so that the table point it looks at keeps its size; nothing
else in the scene changes. The configured lens, given as one of the lenses,
must give the answers of the scene's own top picture
(analysis/top-picture-parts.json, "scene table, scene cubes, scene robot").
"""

from __future__ import annotations

import sys

import numpy as np

from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS, recording_rig_scene
from top_camera_height_scan import TopCameraOnItsLineOfSight


def main(output: str, *lenses: str) -> None:
    if not lenses:
        raise SystemExit(__doc__)
    if (ANALYSIS / output).exists():
        raise SystemExit(f"{ANALYSIS / output} exists; give another name")
    names = {lens: f"top: scene top picture, {float(lens):g} degree lens" for lens in lenses}
    if len(set(names.values())) != len(lenses):
        raise SystemExit(f"a lens is given twice: {lenses}")
    rows: dict[str, list] = {"episode": [], "frame": [], "state": []}
    rows.update({name: [] for name in names.values()})
    heights: dict[str, float] = {}
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        camera = TopCameraOnItsLineOfSight(inner)
        for number, frame, reading in scenes_before_closing(env):
            for lens, name in names.items():
                heights[lens] = camera.use(float(lens))
                rows[name].append(env._format_raw_obs(inner.get_obs())["pixels"]["top"].copy())
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
    finally:
        env.close()
    for lens, height in heights.items():
        print(f"{float(lens):g} degree lens: top camera {1000 * height:.0f} mm above the robot base",
              flush=True)
    np.savez_compressed(
        ANALYSIS / output, **{name: np.array(values) for name, values in rows.items()})
    print(f"{len(rows['episode'])} recordings, {len(lenses)} top pictures each", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
