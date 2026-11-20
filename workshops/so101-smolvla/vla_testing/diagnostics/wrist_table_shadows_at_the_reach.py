"""Is it the shadows on the real table that the scene's wrist picture lacks during the reach?

Run from the workshop folder, for one moment (a number of frames before the
human closes the gripper on the red cube), after early_reach_check.py and
wrist_picture_parts.py have stored that moment's pictures:
    wrist_table_shadows_at_the_reach.py <before>                     (.venv-sapien)
    release_aim_check.py predict early-reach-<before>-table-shadows-pictures.npz early-reach-<before>-table-shadows-predictions.npz   (.venv)
    early_reach_check.py report <before> early-reach-<before>-table-shadows        (.venv-sapien)
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

Found before (wrist_picture_parts.py): with only the table of the real wrist
picture exchanged for the scene's, the model's reach for a near cube ends
about 10 mm higher. The scene's table is one recorded picture of the empty
table laid on the table plane. Its colour is within a few levels of the real
table's and the recording robot's leftover in it does not matter
(table_leftovers_at_the_reach.py). What it cannot have is what the hand and
the cubes do to the real table: their shadows beside the cubes and under the
hand.

Every picture here is the real wrist picture with only its table changed;
cubes and hand are the real ones, put on by wrist_picture_parts. The tables:
  real                          the real picture itself (checked pixel for pixel)
  scene                         the scene's table (checked against the stored
                                picture of wrist_picture_parts.py)
  scene, real around ...        the scene's table, with the real table kept
                                within AROUND_PX of the cubes, of the hand, or
                                of both - the scene's table with real shadows
  real, scene around ...        the real table, with the scene's table within
                                AROUND_PX of the cubes or of the hand - the
                                real table without those shadows
  scene with real smooth part   the scene's table with its smooth brightness
                                (blurred over SMOOTH_PX) exchanged for the
                                real table's
  real with scene smooth part   the other way round
Each is asked about with the real top picture. Nothing is simulated forward.

Writes analysis/early-reach-<before>-table-shadows-pictures.npz and -<n>.png.

A trial that writes nothing: a number of recordings as second argument takes
the pictures of only that many.
    wrist_table_shadows_at_the_reach.py <before> <recordings>
"""

from __future__ import annotations

import sys
from dataclasses import replace

import cv2
import numpy as np

from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS
from wrist_picture_parts import SceneInParts, WristParts, continued, grow, label

ONLY_TABLE = ("scene", "real", "real")
# How far around a cube or the hand the other table is used, and how wide the
# change from one table to the other is. Not measured: the shadows in the
# pictures looked at are about as wide as the large cube.
AROUND_PX = 60
SEAM_PX = 10.0
# Wide enough to wipe out the table's grain and the edges of shadows; the
# value of wrist_table_look.py. Not measured.
SMOOTH_PX = 25.0
REAL = "real table"
SCENE = "scene table"
SHEET_RECORDINGS = (2, 20)


def around(mask: np.ndarray, reach_px: int = AROUND_PX, seam_px: float = SEAM_PX) -> np.ndarray:
    """Per pixel, how much it belongs to the surroundings of the mask: 1 inside, 0 far away."""
    near = grow(mask, reach_px).astype(np.float32)
    return cv2.GaussianBlur(near, (0, 0), seam_px)[..., None]


def blended(inside: np.ndarray, outside: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return (weight * inside + (1.0 - weight) * outside).round().astype(np.uint8)


def smooth(picture: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(picture.astype(np.float32), (0, 0), SMOOTH_PX)


def tables(parts: WristParts) -> dict[str, np.ndarray]:
    """The table pictures tried for one moment, by name."""
    scene = parts.scene_table
    # The real picture is its own table: cubes and hand are put on top of it again.
    real = parts.real
    near_cubes, near_hand = around(parts.real_cubes), around(parts.real_hand)
    near_both = np.maximum(near_cubes, near_hand)
    real_under_things = continued(parts.real, parts.real_cubes | parts.real_hand)
    real_smooth, scene_smooth = smooth(real_under_things), smooth(scene)

    def with_smooth_part(picture: np.ndarray, own: np.ndarray, other: np.ndarray) -> np.ndarray:
        return np.clip(picture - own + other, 0, 255).round().astype(np.uint8)

    return {
        REAL: real,
        SCENE: scene,
        "scene table, real around the cubes": blended(real, scene, near_cubes),
        "scene table, real around the hand": blended(real, scene, near_hand),
        "scene table, real around cubes and hand": blended(real, scene, near_both),
        "real table, scene around the cubes": blended(scene, real, near_cubes),
        "real table, scene around the hand": blended(scene, real, near_hand),
        "scene table with the real table's smooth part": with_smooth_part(
            scene, scene_smooth, real_smooth),
        "real table with the scene table's smooth part": with_smooth_part(
            real_under_things, real_smooth, scene_smooth),
    }


def main(before: str, trial: str | None = None) -> None:
    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    stem = f"early-reach-{int(before)}-table-shadows"
    recorded = np.load(ANALYSIS / f"early-reach-{int(before)}-predictions.npz")
    real_wrist = {
        (number, frame): wrist for number, frame, wrist in zip(
            recorded["episode"].tolist(), recorded["frame"].tolist(), recorded["real_wrist"])}
    # Where wrist_picture_parts.py was run for this moment, its stored picture
    # with the scene's table must come out again.
    earlier_file = ANALYSIS / f"early-reach-{int(before)}-wrist-parts-pictures.npz"
    earlier_only_table: dict[int, np.ndarray] | None = None
    if earlier_file.exists():
        earlier = np.load(earlier_file)
        earlier_only_table = dict(zip(earlier["episode"].tolist(), earlier[label(ONLY_TABLE)]))
    print(f"{earlier_file.name}: "
          f"{'checked against' if earlier_only_table is not None else 'absent, no check against it'}",
          flush=True)

    rows: dict[str, list] = {
        name: [] for name in ("episode", "frame", "state", "red_cube", "table_height")}
    names: list[str] = []
    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        scene = SceneInParts(env)
        inner = scene.inner
        for number, frame, reading in scenes_before_closing(env, int(before)):
            parts = scene.taken_apart(real_wrist[(number, frame)])
            tried = tables(parts)
            names = list(tried)
            for name, table in tried.items():
                rows.setdefault(name, []).append(
                    replace(parts, scene_table=table).mixed(ONLY_TABLE))
            if earlier_only_table is not None and not np.array_equal(
                    rows[SCENE][-1], earlier_only_table[number]):
                raise RuntimeError(
                    f"recording {number}: the picture with the scene's table is not the stored "
                    f"one of wrist_picture_parts.py")
            if not np.array_equal(rows[REAL][-1], parts.real):
                raise RuntimeError(
                    f"recording {number}: the picture with the real table is not the real picture")
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            rows["red_cube"].append(inner.cubes["red"].pose.p[0].cpu().numpy().copy())
            rows["table_height"].append(float(inner.TABLE_SURFACE_Z))
            print(f"recording {number}: pictures taken", flush=True)
            if trial is not None and len(rows["episode"]) == int(trial):
                break
    finally:
        env.close()
    if trial is not None:
        print(f"trial of {len(rows['episode'])} recordings, nothing written; "
              f"{len(names)} tables: {names}", flush=True)
        return
    if rows["episode"] != recorded["episode"].tolist():
        raise RuntimeError("the real pictures are not of these recordings")

    pairs = [(f"real top + real cubes and hand on the {name}", "real_top", name) for name in names]
    np.savez_compressed(
        ANALYSIS / f"{stem}-pictures.npz", pairs=np.array(pairs),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures, "
          f"{int(before)} frames before the human closes the gripper", flush=True)

    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = []
        for name in names:
            tile = cv2.resize(rows[name][index], (320, 240), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            tiles.append(tile)
        cv2.imwrite(
            str(ANALYSIS / f"{stem}-{sheet}.png"),
            cv2.cvtColor(np.concatenate(
                [np.concatenate(tiles[i:i + 3], axis=1) for i in range(0, len(tiles), 3)], axis=0),
                cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
