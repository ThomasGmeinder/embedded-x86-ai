"""Does the recording robot's leftover in the table picture move the model's aim?

Run with .venv-sapien, PYTHONPATH = <workshop>/src : /tmp/Xbotics-SO101-Sim :
~/.cache/so101-stack-training : <workshop>/vla_testing/diagnostics, from the
workshop folder:
    table_picture_leftovers.py release
    table_picture_leftovers.py grasp
then, with file names in analysis/ (<m> is release or grasp):
    release_aim_check.py predict table-leftovers-<m>-pictures.npz table-leftovers-<m>-predictions.npz
    release_aim_check.py report  table-leftovers-release-pictures.npz table-leftovers-release-predictions.npz table-leftovers-release-aim.json
    grasp_aim_check.py report    table-leftovers-grasp-predictions.npz table-leftovers-grasp-aim.json

The stack2-recording-rig scene shows one recorded picture of the table to
both cameras. That picture is the per-pixel median of top pictures of one
recording with robot and cube pixels left out. Where the recording robot
stood in every one of them (its base, its shoulder, a cable) the picture is
not table but a black patch. The scene's robot stands in the same place but
does not cover all of the patch, so the top picture shows black bits beside
the scene's arm, and the wrist camera sees the patch lying on the table when
it looks towards the base.

Tried here: the patch is painted over with the table around it (the largest
dark part of the table picture, with a margin for its soft edge and the
cable on it; cv2.inpaint). Both cameras' pictures are taken with the table
picture as configured and with the patch painted over, and the model is
asked about all four combinations. Nothing is simulated forward.

"release" is the moment of release_aim_check.py (25 frames before the human
lets the red cube go), "grasp" that of grasp_aim_check.py (25 frames before
the human closes the gripper).

Writes analysis/table-leftovers-<m>-pictures.npz, analysis/table-leftovers-<m>.json,
analysis/table-leftovers-<m>-table-picture.png and analysis/table-leftovers-<m>.png.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np

from grasp_aim_check import scenes_before_closing
from release_aim_check import ANALYSIS, recording_rig_scene, scene_facts, scenes_before_release
from wrist_picture_parts import DARKEST_TABLE_VALUE, grow
from so101_smolvla.stack_recording_rig import RECORDING_RIG

MOMENTS = {"release": scenes_before_release, "grasp": scenes_before_closing}
AS_CONFIGURED = "table picture as configured"
PAINTED_OVER = "robot leftover painted over"
# The patch's soft edge and the cable lying on it go with the patch.
LEFTOVER_MARGIN_PX = 8
INPAINT_RADIUS_PX = 5
SHEET_RECORDINGS = (2, 20)


def without_robot(table_picture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The table picture with the recording robot's patch painted over, and the patch."""
    dark = (table_picture.max(axis=2) < DARKEST_TABLE_VALUE).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    if count < 2:
        raise RuntimeError("the table picture has no dark part")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    patch = grow(labels == largest, LEFTOVER_MARGIN_PX)
    painted = cv2.inpaint(
        table_picture, patch.astype(np.uint8) * 255, INPAINT_RADIUS_PX, cv2.INPAINT_TELEA)
    return painted, patch


def main(moment: str) -> None:
    import torch

    configured = cv2.cvtColor(cv2.imread(str(RECORDING_RIG.table_picture)), cv2.COLOR_BGR2RGB)
    painted, patch = without_robot(configured)
    stem = f"table-leftovers-{moment}"
    cv2.imwrite(str(ANALYSIS / f"{stem}-table-picture.png"),
                cv2.cvtColor(painted, cv2.COLOR_RGB2BGR))
    tables = {AS_CONFIGURED: configured, PAINTED_OVER: painted}

    rows: dict[str, list] = {name: [] for name in ("episode", "frame", "state")}
    rows.update({f"{camera}, {table}": [] for table in tables for camera in ("top", "wrist")})
    facts: dict[str, list] = {}
    changed = {"top": [], "wrist": []}
    env = recording_rig_scene()
    try:
        inner = env._env.unwrapped
        own = inner._rgb_overlay_image
        if tuple(own.shape) != configured.shape:
            raise RuntimeError(
                f"the scene's table picture is {tuple(own.shape)}, the file is {configured.shape}")
        shown = {table: torch.as_tensor(picture, dtype=own.dtype, device=own.device)
                 for table, picture in tables.items()}
        if not torch.equal(shown[AS_CONFIGURED], own):
            raise RuntimeError("the scene does not show the configured table picture as it is")
        for number, frame, reading in MOMENTS[moment](env):
            rows["episode"].append(number)
            rows["frame"].append(frame)
            rows["state"].append(reading)
            for table, picture in shown.items():
                inner._rgb_overlay_image = picture
                pixels = env._format_raw_obs(inner.get_obs())["pixels"]
                for camera in ("top", "wrist"):
                    rows[f"{camera}, {table}"].append(pixels[camera].copy())
            inner._rgb_overlay_image = own
            for camera in changed:
                differs = np.any(rows[f"{camera}, {AS_CONFIGURED}"][-1]
                                 != rows[f"{camera}, {PAINTED_OVER}"][-1], axis=2)
                changed[camera].append(int(differs.sum()))
            if moment == "release":
                for name, value in scene_facts(env).items():
                    facts.setdefault(name, []).append(value)
            print(f"recording {number}: pictures taken", flush=True)
    finally:
        env.close()

    summary = {
        "recordings": len(rows["episode"]),
        "patch_px_in_the_table_picture": int(patch.sum()),
        **{f"{camera}_picture_px_changed_by_painting_over (10 %, median, 90 %)":
           [int(v) for v in np.percentile(values, [10, 50, 90])]
           for camera, values in changed.items()},
        **{f"{camera}_pictures_changed": int((np.array(values) > 0).sum())
           for camera, values in changed.items()},
    }
    print(json.dumps(summary, indent=1), flush=True)
    (ANALYSIS / f"{stem}.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")

    pairs = np.array([
        (f"top: {top}; wrist: {wrist}", f"top, {top}", f"wrist, {wrist}")
        for top in tables for wrist in tables])
    np.savez_compressed(
        ANALYSIS / f"{stem}-pictures.npz",
        pairs=pairs,
        **{name: np.array(values) for name, values in {**rows, **facts}.items()},
    )
    print(f"assembled {len(rows['episode'])} recordings, {len(pairs)} pairs of pictures", flush=True)

    tiles = [
        np.concatenate([
            cv2.resize(rows[f"{camera}, {table}"][index], (320, 240), interpolation=cv2.INTER_AREA)
            for camera in ("top", "wrist") for table in tables], axis=1)
        for index in SHEET_RECORDINGS if index < len(rows["episode"])
    ]
    cv2.imwrite(str(ANALYSIS / f"{stem}.png"),
                cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(sys.argv[1])
