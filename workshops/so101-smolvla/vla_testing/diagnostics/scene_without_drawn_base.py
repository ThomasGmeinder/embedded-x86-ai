"""What does the model answer when the scene does not draw the robot's base?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, for one moment:
    scene_without_drawn_base.py <moment>
<moment> is a number of frames before the human closes the gripper on the red
cube (the reach and the grasp: 105, 55, 25) or the word "release" (25 frames
before the human lets the red cube go). Then, with file names in analysis/:
    release_aim_check.py predict base-not-drawn-<moment>-pictures.npz base-not-drawn-<moment>-predictions.npz   (.venv)
and for a number of frames
    early_reach_check.py report <moment> base-not-drawn-<moment>
or for the release
    release_aim_check.py report base-not-drawn-release-pictures.npz base-not-drawn-release-predictions.npz base-not-drawn-release.json

Why: the table picture that the stack2-recording-rig scene shows behind robot
and cubes is a picture of the recording table, and it holds the recording
robot's own base, which never moves. The scene draws its robot's base on top
of it. Taken apart (top_picture_parts.py), the drawn base is the one part of
the drawn robot that moves the model's reach in nearly every recording, and
the base in the table picture serves the model as the real base does. Those
were real pictures with single parts changed. Here the scene's own two
pictures are taken, as the scene is built and with the base link not drawn,
so that the table picture shows at its place in both cameras.

Writes analysis/base-not-drawn-<moment>-pictures.npz and -<n>.png, and
refuses to overwrite the pictures file.

A trial that writes nothing: a number of recordings as second argument takes
the pictures of only that many.
    scene_without_drawn_base.py <moment> <recordings>
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS, recording_rig_scene, scene_facts, scenes_before_release
from wrist_picture_parts import render_bodies, show

BASE_LINK = "base_link"
RELEASE = "release"
CAMERAS = ("top", "wrist")
# (label, top picture, wrist picture); "real_top" and "real_wrist" are the recorded pictures.
PAIRS = (
    ("real top + real wrist", "real_top", "real_wrist"),
    ("scene top + real wrist", "top", "real_wrist"),
    ("scene top, base not drawn + real wrist", "top_without_base", "real_wrist"),
    ("scene top + scene wrist", "top", "wrist"),
    ("scene top + scene wrist, base not drawn in either", "top_without_base", "wrist_without_base"),
)
SHEET_RECORDINGS = (3, 20)


def main(moment: str, trial: str | None = None) -> None:
    stem = f"base-not-drawn-{moment if moment == RELEASE else int(moment)}"
    pictures_file = ANALYSIS / f"{stem}-pictures.npz"
    if trial is None and pictures_file.exists():
        raise SystemExit(f"{pictures_file} exists; it is not overwritten")

    rows: dict[str, list] = {name: [] for name in (
        "episode", "frame", "state", "large_cube", "red_cube", "table_height",
        *CAMERAS, *(f"{camera}_without_base" for camera in CAMERAS))}
    changed: dict[str, list[int]] = {camera: [] for camera in CAMERAS}
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        base = render_bodies(inner.agent.robot.links_map[BASE_LINK])
        if not base:
            raise RuntimeError(f"the scene draws nothing of {BASE_LINK}")

        def pictures() -> dict[str, np.ndarray]:
            seen = env._format_raw_obs(inner.get_obs())["pixels"]
            return {camera: seen[camera].copy() for camera in CAMERAS}

        scenes = (scenes_before_release(env) if moment == RELEASE
                  else scenes_before_closing(env, int(moment)))
        for number, frame, reading in scenes:
            as_built = pictures()
            show(base, False)
            without = pictures()
            show(base, True)
            for camera in CAMERAS:
                rows[camera].append(as_built[camera])
                rows[f"{camera}_without_base"].append(without[camera])
                changed[camera].append(
                    int(np.any(as_built[camera] != without[camera], axis=2).sum()))
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            print(f"recording {number}: pictures taken", flush=True)
            if trial is not None and len(rows["episode"]) == int(trial):
                break
        if not all(np.array_equal(again, rows[camera][-1])
                   for camera, again in pictures().items()):
            raise RuntimeError("the base was not put back into the scene's pictures")
    finally:
        env.close()

    for camera, counts in changed.items():
        print(f"{camera} picture: pixels that change when the base is not drawn, smallest / "
              f"median / largest: {min(counts)} / {int(np.median(counts))} / {max(counts)}; "
              f"pictures that change: {sum(count > 0 for count in counts)} of {len(counts)}",
              flush=True)
    if trial is not None:
        print(f"trial of {len(rows['episode'])} recordings, nothing written", flush=True)
        return
    np.savez_compressed(
        pictures_file, pairs=np.array(PAIRS),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"pictures of {len(rows['episode'])} recordings, {len(PAIRS)} pairs, moment {moment}",
          flush=True)

    # Both cameras of two recordings, as built and without the drawn base, to look at.
    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = [rows[name][index] for camera in CAMERAS
                 for name in (camera, f"{camera}_without_base")]
        cv2.imwrite(str(ANALYSIS / f"{stem}-{sheet}.png"),
                    cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
