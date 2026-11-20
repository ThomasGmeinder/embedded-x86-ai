"""Does the recording robot's leftover in the table picture move the model's answers during the reach?

Run from the workshop folder, for one moment (a number of frames before the
human closes the gripper on the red cube), after early_reach_check.py and
wrist_picture_parts.py have stored that moment's pictures:
    table_leftovers_at_the_reach.py <before>                         (.venv-sapien)
    release_aim_check.py predict early-reach-<before>-table-leftovers-pictures.npz early-reach-<before>-table-leftovers-predictions.npz   (.venv)
    early_reach_check.py report <before> early-reach-<before>-table-leftovers      (.venv-sapien)
PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim : ~/.cache/so101-stack-training
: <workshop>/vla_testing/diagnostics.

The scene shows one recorded picture of the table to both cameras. Near the
robot's base that picture is not table: it holds a black patch where the
recording robot stood (table_picture_leftovers.py). That script painted the
patch over and asked the model at the grasp and at the release, where the
wrist camera looks at the cubes. During the reach the hand is still near the
base and the wrist camera looks at the patch lying on the table beside its
jaw; wrist_picture_parts.py found that the scene's table in the wrist picture
keeps the arm higher there. This script asks at the reach.

Both cameras' pictures are taken with the table picture as configured and
with the patch painted over (table_picture_leftovers.without_robot). The
wrist picture is also made with only its table from the scene and cubes and
hand from the real picture, with either table picture; with the configured
one it must be the stored picture of wrist_picture_parts.py, pixel for pixel.
The wrist camera's view of the table alone is stored as well. Nothing is
simulated forward.

Writes analysis/early-reach-<before>-table-leftovers-pictures.npz,
-pixels.json and -<n>.png.

A trial that writes nothing: a number of recordings as second argument takes
the pictures of only that many and prints the pixel counts.
    table_leftovers_at_the_reach.py <before> <recordings>
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS
from table_picture_leftovers import AS_CONFIGURED, PAINTED_OVER, without_robot
from wrist_picture_parts import SceneInParts, label
from so101_smolvla.stack_recording_rig import RECORDING_RIG

ONLY_TABLE = ("scene", "real", "real")
# What the wrist camera sees with the scene's robot and cubes hidden. Not asked
# about; stored to see what the dark bits beside the jaw are made of.
TABLE_ALONE = "wrist camera's table alone"
SHORT = {AS_CONFIGURED: "as configured", PAINTED_OVER: "leftover painted over"}
SHEET_RECORDINGS = (2, 20)


def main(before: str, trial: str | None = None) -> None:
    import torch

    import so101_smolvla.sapien_stack_env  # noqa: F401  (registers the stacking tasks)
    from so101_sim.lerobot_env import So101SimEnv
    from so101_smolvla.sapien_config import environment_arguments
    from so101_smolvla.sapien_env_worker import _configure_robot_appearance

    stem = f"early-reach-{int(before)}-table-leftovers"
    recorded = np.load(ANALYSIS / f"early-reach-{int(before)}-predictions.npz")
    real_wrist = {
        (number, frame): wrist for number, frame, wrist in zip(
            recorded["episode"].tolist(), recorded["frame"].tolist(), recorded["real_wrist"])}
    earlier = np.load(ANALYSIS / f"early-reach-{int(before)}-wrist-parts-pictures.npz")
    earlier_only_table = dict(zip(earlier["episode"].tolist(), earlier[label(ONLY_TABLE)]))

    configured = cv2.cvtColor(cv2.imread(str(RECORDING_RIG.table_picture)), cv2.COLOR_BGR2RGB)
    painted, patch = without_robot(configured)
    tables = {AS_CONFIGURED: configured, PAINTED_OVER: painted}
    kinds = ("top", "wrist", "wrist with only the scene's table", TABLE_ALONE)
    rows: dict[str, list] = {
        name: [] for name in ("episode", "frame", "state", "red_cube", "table_height")}
    rows.update({f"{kind}, {SHORT[table]}": [] for table in tables for kind in kinds})
    changed: dict[str, list[int]] = {"top": [], "wrist": []}

    env = So101SimEnv(**environment_arguments("stack2-recording-rig"), episode_length=10)
    try:
        _configure_robot_appearance(env)
        env._env.reset(seed=0)
        scene = SceneInParts(env)
        inner = scene.inner
        own = inner._rgb_overlay_image
        shown = {table: torch.as_tensor(picture, dtype=own.dtype, device=own.device)
                 for table, picture in tables.items()}
        if not torch.equal(shown[AS_CONFIGURED], own):
            raise RuntimeError("the scene does not show the configured table picture as it is")
        for number, frame, reading in scenes_before_closing(env, int(before)):
            for table, picture in shown.items():
                inner._rgb_overlay_image = picture
                parts = scene.taken_apart(real_wrist[(number, frame)])
                rows[f"top, {SHORT[table]}"].append(scene.pictures()["top"])
                rows[f"wrist, {SHORT[table]}"].append(parts.scene)
                rows[f"wrist with only the scene's table, {SHORT[table]}"].append(
                    parts.mixed(ONLY_TABLE))
                rows[f"{TABLE_ALONE}, {SHORT[table]}"].append(parts.scene_table)
            inner._rgb_overlay_image = own
            if not np.array_equal(
                    rows[f"wrist with only the scene's table, {SHORT[AS_CONFIGURED]}"][-1],
                    earlier_only_table[number]):
                raise RuntimeError(
                    f"recording {number}: the wrist picture with the configured table is not "
                    f"the stored one of wrist_picture_parts.py")
            for camera in changed:
                differs = np.any(rows[f"{camera}, {SHORT[AS_CONFIGURED]}"][-1]
                                 != rows[f"{camera}, {SHORT[PAINTED_OVER]}"][-1], axis=2)
                changed[camera].append(int(differs.sum()))
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
        print(f"trial of {len(rows['episode'])} recordings, nothing written; pixels changed by "
              f"painting over: {changed}", flush=True)
        return
    if rows["episode"] != recorded["episode"].tolist():
        raise RuntimeError("the real pictures are not of these recordings")

    summary = {
        "frames_before_closing": int(before),
        "recordings": len(rows["episode"]),
        "patch_px_in_the_table_picture": int(patch.sum()),
        **{f"{camera}_picture_px_changed_by_painting_over (smallest, median, largest)":
           [int(min(values)), int(np.median(values)), int(max(values))]
           for camera, values in changed.items()},
        **{f"{camera}_pictures_changed": int((np.array(values) > 0).sum())
           for camera, values in changed.items()},
    }
    print(json.dumps(summary, indent=1), flush=True)
    (ANALYSIS / f"{stem}-pixels.json").write_text(
        json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    pairs = [("real top + real wrist", "real_top", "real_wrist")]
    for short in SHORT.values():
        pairs += [
            (f"real top + real wrist with only the scene's table, {short}",
             "real_top", f"wrist with only the scene's table, {short}"),
            (f"real top + scene wrist, {short}", "real_top", f"wrist, {short}"),
            (f"scene top + real wrist, {short}", f"top, {short}", "real_wrist"),
            (f"scene top + scene wrist, {short}", f"top, {short}", f"wrist, {short}"),
        ]
    np.savez_compressed(
        ANALYSIS / f"{stem}-pictures.npz", pairs=np.array(pairs),
        **{name: np.array(values) for name, values in rows.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures, "
          f"{int(before)} frames before the human closes the gripper", flush=True)

    for sheet, index in enumerate(SHEET_RECORDINGS):
        tiles = [
            np.concatenate([
                cv2.resize(rows[f"{kind}, {short}"][index], (320, 240),
                           interpolation=cv2.INTER_AREA)
                for kind in kinds], axis=1)
            for short in SHORT.values()
        ]
        cv2.imwrite(str(ANALYSIS / f"{stem}-{sheet}.png"),
                    cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(*sys.argv[1:])
