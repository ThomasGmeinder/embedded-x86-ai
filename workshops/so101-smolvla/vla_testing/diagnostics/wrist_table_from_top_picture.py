"""Does the model let go lower when the wrist camera sees the recorded table?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder, after release_aim_check.py render and predict:
    wrist_table_from_top_picture.py release
    wrist_table_from_top_picture.py grasp
then, with file names in analysis/ (<m> is release or grasp):
    release_aim_check.py predict wrist-table-from-top-<m>-pictures.npz wrist-table-from-top-<m>-predictions.npz
    release_aim_check.py report  wrist-table-from-top-release-pictures.npz wrist-table-from-top-release-predictions.npz wrist-table-from-top-release-aim.json
    release_answer_paths.py      wrist-table-from-top-release-pictures.npz wrist-table-from-top-release-predictions.npz wrist-table-from-top-release-paths.json
    grasp_aim_check.py report    wrist-table-from-top-grasp-predictions.npz wrist-table-from-top-grasp-aim.json

Found before (wrist_table_look.py): the scene's table is one flat light grey
in the wrist picture, the recorded table is darker and shaded, and with the
recorded table's colour and shading the model opens the gripper later and
lower. The stack2-recording-rig scene owns a picture of the recorded table,
taken by the top camera, and shows it to the top camera only.

Tried here: the same table picture lies on the table plane, and the wrist
camera sees it from where it stands. Every table pixel of the scene's wrist
picture is followed to the table plane and from there into the top camera's
table picture. Where the top camera did not see the table, the picture's
nearest edge is continued. Pixels that look past the table's plane keep what
the scene draws.

"release" is the moment of release_aim_check.py (25 frames before the human
lets the red cube go), "grasp" that of grasp_aim_check.py (25 frames before
the human closes the gripper). Each wrist picture is asked about with the
real top picture and with the scene's.

At the release moment the table painted this way is also compared with the
table in the real wrist picture of the same moment, pixel by pixel.

Writes analysis/wrist-table-from-top-<m>-pictures.npz,
analysis/wrist-table-from-top-<m>.json and analysis/wrist-table-from-top-<m>.png.

Outcome (job wrist-table-from-top-1795143496): the table painted this way is
9 of 255 levels from the real wrist picture's table (the scene's flat table:
43). With the scene's top picture the model opens the gripper 71.1 mm above
the table instead of 80.3 mm and 15.9 mm instead of 18.8 mm from the large
cube's centre; at the grasp its tip is 16.1 mm above the human's instead of
20.0 mm. The stack2-recording-rig scene has painted its wrist picture this
way itself since then (so101_smolvla.table_view, RecordingRigMixin), so a new
run of this script gets two equal wrist pictures.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from cube_places_from_pictures import coloured
from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS, recording_rig_scene, scene_facts, scenes_before_release
from wrist_picture_parts import CUBE_MARGIN_PX, DARKEST_TABLE_VALUE, HAND_MARGIN_PX, grow
from so101_smolvla.stack_recording_rig import RECORDING_RIG
from so101_smolvla.table_view import table_seen_from

MOMENTS = {"release": scenes_before_release, "grasp": scenes_before_closing}
AS_IT_IS = "scene wrist"
PAINTED = "scene wrist, table from the top camera's table picture"
WRISTS = (AS_IT_IS, PAINTED)
TOPS = {"real top": "real_top", "scene top": "scene_top"}
SHEET_RECORDINGS = (2, 20)


def main(moment: str) -> None:
    table_picture = cv2.cvtColor(cv2.imread(str(RECORDING_RIG.table_picture)), cv2.COLOR_BGR2RGB)
    real_wrist_of: dict[int, np.ndarray] = {}
    if moment == "release":
        real_pictures = np.load(ANALYSIS / "release-aim-predictions.npz")
        real_wrist_of = dict(zip(real_pictures["episode"].tolist(), real_pictures["real_wrist"]))

    rows: dict[str, list] = {
        name: [] for name in ("episode", "frame", "state", "scene_top", *WRISTS)}
    facts: dict[str, list] = {}
    difference: dict[str, list] = {AS_IT_IS: [], PAINTED: []}
    painted_share = []
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        kept = inner._segmentation_ids_to_keep.cpu().numpy()

        def projection(name: str) -> np.ndarray:
            camera = inner._sensors[name].camera
            return (camera.get_intrinsic_matrix()[0].cpu().numpy().astype(np.float64)
                    @ camera.get_extrinsic_matrix()[0].cpu().numpy().astype(np.float64))

        for number, frame, reading in MOMENTS[moment](env):
            seen = inner.get_obs()
            pixels = env._format_raw_obs(seen)["pixels"]
            scene = pixels["wrist"].copy()
            if table_picture.shape != pixels["top"].shape:
                raise RuntimeError(
                    f"the table picture is {table_picture.shape}, the top camera's pictures are "
                    f"{pixels['top'].shape}")
            drawn_ids = seen["sensor_data"]["wrist"]["segmentation"][0, ..., 0].cpu().numpy()
            table, in_front = table_seen_from(
                table_picture, projection("top"), projection("wrist"), float(inner.TABLE_SURFACE_Z))
            table_shown = ~np.isin(drawn_ids, kept) & in_front
            picture = scene.copy()
            picture[table_shown] = table[table_shown]
            painted_share.append(float(table_shown.mean()))

            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["scene_top"].append(pixels["top"].copy())
            rows[AS_IT_IS].append(scene)
            rows[PAINTED].append(picture)
            if moment == "release":
                for name, value in scene_facts(env).items():
                    facts.setdefault(name, []).append(value)

            real = real_wrist_of.get(number)
            if real is not None:
                real_things = grow(real.max(axis=2) < DARKEST_TABLE_VALUE, HAND_MARGIN_PX)
                for colour in ("red", "blue"):
                    mask = coloured(real, colour, free=False)
                    if mask is not None:
                        real_things |= grow(mask, CUBE_MARGIN_PX)
                both_table = table_shown & ~real_things
                for name, shown in ((AS_IT_IS, scene), (PAINTED, picture)):
                    difference[name].append(np.median(
                        shown[both_table].astype(np.float64) - real[both_table], axis=0))
            print(f"recording {number}: pictures taken", flush=True)
    finally:
        env.close()

    summary: dict[str, object] = {
        "recordings": len(rows["episode"]),
        "share of the wrist picture painted from the table picture (10 %, median, 90 %)": [
            round(float(v), 3) for v in np.percentile(painted_share, [10, 50, 90])],
    }
    if real_wrist_of:
        for name, values in difference.items():
            values = np.array(values)
            summary[f"{name}: table minus the real wrist picture's table, levels of 255"] = {
                "median over recordings (red, green, blue)": np.median(values, axis=0).round(1).tolist(),
                "median size over recordings": round(float(np.median(np.abs(values).mean(axis=1))), 1),
                "largest size": round(float(np.abs(values).mean(axis=1).max()), 1),
            }
    print(json.dumps(summary, indent=1), flush=True)
    stem = f"wrist-table-from-top-{moment}"
    (ANALYSIS / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    pairs = np.array([
        (f"{top} + {name}", top_pictures, name)
        for top, top_pictures in TOPS.items() for name in WRISTS])
    np.savez_compressed(
        ANALYSIS / f"{stem}-pictures.npz",
        pairs=pairs,
        **{name: np.array(values) for name, values in {**rows, **facts}.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures", flush=True)

    tiles = []
    for index in SHEET_RECORDINGS:
        shown = [rows[name][index] for name in WRISTS]
        real = real_wrist_of.get(rows["episode"][index])
        if real is not None:
            shown.append(real)
        tiles.append(np.concatenate(
            [cv2.resize(picture, (320, 240), interpolation=cv2.INTER_AREA) for picture in shown],
            axis=1))
    cv2.imwrite(
        str(ANALYSIS / f"{stem}.png"), cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(sys.argv[1])
