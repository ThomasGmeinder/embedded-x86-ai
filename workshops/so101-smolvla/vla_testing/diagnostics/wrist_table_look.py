"""What about the table in the wrist picture makes the model open the gripper early?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict:
    wrist_table_look.py
then, with file names in analysis/:
    release_aim_check.py predict wrist-table-look-pictures.npz wrist-table-look-predictions.npz
    release_aim_check.py report  wrist-table-look-pictures.npz wrist-table-look-predictions.npz wrist-table-look-release.json
    release_answer_paths.py      wrist-table-look-pictures.npz wrist-table-look-predictions.npz wrist-table-look-paths.json

Found before (release_picture_parts.py): with the real table behind the
scene's cubes and hand the model opens the gripper at the same target as on
real pictures; with the scene's table it opens earlier, and so higher. The
scene's table is one flat light grey. The recorded table is darker, gets
darker towards one side, and carries the shadows of the cubes and the hand.

Moment and recordings are those of release_aim_check.py: 25 frames before the
human lets the red cube go. Cubes and hand are always the scene's. Only the
table pixels of the scene's wrist picture change:
  as it is        the scene's table
  colour          the scene's table multiplied so that its colour is the
                  recorded table's (one colour for all recordings, the median
                  of the recordings' table pixels) - what a different table
                  material in the scene would give
  shading         the real table of the same moment, blurred until grain and
                  shadow edges are gone: its colour and its smooth change of
                  brightness over the picture
  real            the real table of the same moment, continued under the real
                  cubes and hand
Each is asked about with the real top picture and with the scene's.

Writes analysis/wrist-table-look-pictures.npz, analysis/wrist-table-look.json
(the table colours) and analysis/wrist-table-look.png.
"""

from __future__ import annotations

import json

import cv2
import numpy as np

from cube_places_from_pictures import coloured
from release_aim_check import ANALYSIS, recording_rig_scene, scene_facts, scenes_before_release
from wrist_picture_parts import (
    CUBE_MARGIN_PX, DARKEST_TABLE_VALUE, HAND_MARGIN_PX, continued, grow, render_bodies, show,
)

AS_IT_IS = "scene wrist"
COLOUR = "scene wrist, table in the recorded table's colour"
SHADING = "scene wrist, table with the real table's colour and smooth shading"
REAL = "scene wrist, on the real table"
WRISTS = (AS_IT_IS, COLOUR, SHADING, REAL)
TOPS = {"real top": "real_top", "scene top": "scene_top"}
# Wide enough to wipe out the table's grain and the edges of shadows. Not measured.
SHADING_BLUR_PX = 25.0
SHEET_RECORDINGS = (2, 20)


def main() -> None:
    real_pictures = np.load(ANALYSIS / "release-aim-predictions.npz")
    real_wrist_of = dict(zip(real_pictures["episode"].tolist(), real_pictures["real_wrist"]))
    scene_pictures = np.load(ANALYSIS / "release-aim-pictures.npz")
    scene_top_of = dict(zip(scene_pictures["episode"].tolist(), scene_pictures["top"]))

    rows: dict[str, list] = {
        name: [] for name in
        ("episode", "frame", "state", "large_cube", "red_cube", "table_height", "scene_top")}
    taken: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    recorded_colours, scene_colours = [], []
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        robot = [body for link in inner.agent.robot.links for body in render_bodies(link)]
        cubes = [body for cube in inner.cubes.values() for body in render_bodies(cube)]

        def wrist() -> np.ndarray:
            return env._format_raw_obs(inner.get_obs())["pixels"]["wrist"].copy()

        for number, frame, reading in scenes_before_release(env):
            real = real_wrist_of[number]
            scene = wrist()
            show(robot, False), show(cubes, False)
            scene_table = wrist()
            show(robot, True), show(cubes, True)
            table_shown = ~np.any(scene != scene_table, axis=2)

            found = [coloured(real, colour, free=False) for colour in ("red", "blue")]
            real_things = grow(real.max(axis=2) < DARKEST_TABLE_VALUE, HAND_MARGIN_PX)
            for mask in found:
                if mask is not None:
                    real_things |= grow(mask, CUBE_MARGIN_PX)
            recorded_colours.append(np.median(real[~real_things], axis=0))
            scene_colours.append(np.median(scene_table.reshape(-1, 3), axis=0))
            taken.append((scene, scene_table, table_shown, continued(real, real_things)))

            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["scene_top"].append(scene_top_of[number])
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            print(f"recording {number}: pictures taken", flush=True)
    finally:
        env.close()

    recorded_colour = np.median(recorded_colours, axis=0)
    scene_colour = np.median(scene_colours, axis=0)
    colours = {
        "recorded table in the wrist picture, median colour": recorded_colour.round(1).tolist(),
        "recorded table, darkest and lightest recording (median colour of each)": [
            np.min(recorded_colours, axis=0).round(1).tolist(),
            np.max(recorded_colours, axis=0).round(1).tolist()],
        "scene's table in the wrist picture, median colour": scene_colour.round(1).tolist(),
    }
    print(json.dumps(colours, indent=1), flush=True)
    (ANALYSIS / "wrist-table-look.json").write_text(
        json.dumps(colours, indent=1) + "\n", encoding="utf-8")

    wrists: dict[str, list] = {name: [] for name in WRISTS}
    for scene, scene_table, table_shown, real_table in taken:
        tables = {
            AS_IT_IS: scene_table,
            COLOUR: np.clip(
                scene_table.astype(np.float64) * recorded_colour / scene_colour, 0, 255
            ).round().astype(np.uint8),
            SHADING: cv2.GaussianBlur(real_table, (0, 0), SHADING_BLUR_PX),
            REAL: real_table,
        }
        for name, table in tables.items():
            picture = scene.copy()
            picture[table_shown] = table[table_shown]
            wrists[name].append(picture)
        if not np.array_equal(wrists[AS_IT_IS][-1], scene):
            raise RuntimeError("the scene's table put back is not the scene's picture")

    pairs = np.array([
        (f"{top} + {name}", top_pictures, name)
        for top, top_pictures in TOPS.items() for name in WRISTS])
    np.savez_compressed(
        ANALYSIS / "wrist-table-look-pictures.npz",
        pairs=pairs,
        **{name: np.array(values) for name, values in {**rows, **wrists}.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures", flush=True)

    tiles = []
    for index in SHEET_RECORDINGS:
        for name in WRISTS:
            tile = cv2.resize(wrists[name][index], (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, name[13:] or "as it is", (4, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.append(cv2.resize(
            real_wrist_of[rows["episode"][index]], (320, 240), interpolation=cv2.INTER_AREA))
    per_row = len(WRISTS) + 1
    cv2.imwrite(
        str(ANALYSIS / "wrist-table-look.png"),
        cv2.cvtColor(np.concatenate(
            [np.concatenate(tiles[i:i + per_row], axis=1) for i in range(0, len(tiles), per_row)],
            axis=0), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
