"""Is it the shadows on the real table that the scene's wrist picture lacks at the release?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict:
    wrist_table_shadows_at_the_release.py
then, with file names in analysis/:
    release_aim_check.py predict release-table-shadows-pictures.npz release-table-shadows-predictions.npz   (.venv)
    release_aim_check.py report  release-table-shadows-pictures.npz release-table-shadows-predictions.npz release-table-shadows.json

Found at the reach (wrist_table_shadows_at_the_reach.py): the scene's table
in the wrist picture does what the real table does once the real table is
kept in a strip around the cubes and the hand, where the real table carries
their shadows. At the release the model lets the red cube go too high, and
the scene's wrist picture is part of that.

Moment and recordings are those of release_aim_check.py: 25 frames before the
human lets the red cube go. The wrist pictures are those of the reach script
(its function tables): the real wrist picture with only its table changed,
cubes and hand the real ones. The scene's own wrist picture is stored and
asked about as well. Each is asked about with the real top picture. Nothing
is simulated beyond the replay that brings the scene to the moment.

Writes analysis/release-table-shadows-pictures.npz and -<n>.png, and refuses
to overwrite the pictures file.

A trial that writes nothing: a number of recordings as argument takes the
pictures of only that many.
    wrist_table_shadows_at_the_release.py <recordings>
"""

from __future__ import annotations

import sys
from dataclasses import replace

import cv2
import numpy as np

from release_aim_check import ANALYSIS, recording_rig_scene, scene_facts, scenes_before_release
from wrist_picture_parts import SceneInParts
from wrist_table_shadows_at_the_reach import ONLY_TABLE, REAL, tables

STEM = "release-table-shadows"
SCENE_WRIST = "scene wrist"
SHEET_RECORDINGS = (2, 20)


def main(trial: str | None = None) -> None:
    pictures_file = ANALYSIS / f"{STEM}-pictures.npz"
    if trial is None and pictures_file.exists():
        raise SystemExit(f"{pictures_file} exists; it is not overwritten")
    recorded = np.load(ANALYSIS / "release-aim-predictions.npz")
    real_wrist = dict(zip(recorded["episode"].tolist(), recorded["real_wrist"]))

    rows: dict[str, list] = {
        name: [] for name in
        ("episode", "frame", "state", "large_cube", "red_cube", "table_height", SCENE_WRIST)}
    names: list[str] = []
    env = recording_rig_scene()
    try:
        scene = SceneInParts(env)
        for number, frame, reading in scenes_before_release(env):
            parts = scene.taken_apart(real_wrist[number])
            tried = tables(parts)
            names = list(tried)
            for name, table in tried.items():
                rows.setdefault(name, []).append(
                    replace(parts, scene_table=table).mixed(ONLY_TABLE))
            if not np.array_equal(rows[REAL][-1], parts.real):
                raise RuntimeError(
                    f"recording {number}: the picture with the real table is not the real picture")
            rows[SCENE_WRIST].append(parts.scene)
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            for name, value in scene_facts(env).items():
                rows[name].append(value)
            print(f"recording {number}: pictures taken", flush=True)
            if trial is not None and len(rows["episode"]) == int(trial):
                break
    finally:
        env.close()
    if trial is not None:
        print(f"trial of {len(rows['episode'])} recordings, nothing written; "
              f"{len(names)} tables: {names}", flush=True)
        return
    if (rows["episode"] != recorded["episode"].tolist()
            or rows["frame"] != recorded["frame"].tolist()):
        raise RuntimeError("the real pictures are not of these recordings and frames")

    pairs = [(f"real top + real cubes and hand on the {name}", "real_top", name) for name in names]
    pairs.append((f"real top + {SCENE_WRIST}", "real_top", SCENE_WRIST))
    np.savez_compressed(
        pictures_file, pairs=np.array(pairs),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures", flush=True)

    shown = [*names, SCENE_WRIST]
    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = []
        for name in shown:
            tile = cv2.resize(rows[name][index], (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(tile)
        tiles.extend([np.full_like(tiles[0], 255)] * (-len(tiles) % 5))
        cv2.imwrite(
            str(ANALYSIS / f"{STEM}-{sheet}.png"),
            cv2.cvtColor(np.concatenate(
                [np.concatenate(tiles[i:i + 5], axis=1) for i in range(0, len(tiles), 5)], axis=0),
                cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
