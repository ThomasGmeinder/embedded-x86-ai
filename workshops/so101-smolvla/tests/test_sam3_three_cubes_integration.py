"""The real SAM 3 (the reference sam3-tracker-rocm) on the pictures of the real three-cube scene.

Skipped while the reference's environment .venv-sam3 or its compiled model
files are missing.
"""

import json
import os
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clip_reading import read_clip
from so101_smolvla import sapien_rollout, three_cube_rollout
from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.feeds_video import CAPTION_HEIGHT, ROLE_COLORS, TINT
from so101_smolvla.picture_editor import edit_pictures
from so101_smolvla.sam3_outlines import CAMERAS, PHRASE, Sam3Outlines, cube_colours
from so101_smolvla.sapien_config import (
    IMAGE_SHAPE,
    OUTLINE_CODES,
    SAPIEN_PYTHON,
    SAPIEN_SOURCE,
    THREE_CUBE_TASK,
)
from so101_smolvla.sapien_env_client import SapienPolicyEnvironment
from so101_smolvla.three_cube_rollout import OUTLINE_SOURCES

WORKSHOP = Path(__file__).resolve().parents[1]
SETTINGS = json.loads((WORKSHOP / "configs" / "sam3_reference.json").read_text())
READY = (WORKSHOP / ".venv-sam3" / "bin" / "python").is_file() and (
    Path(os.path.expanduser(SETTINGS["compiled_dir"])) / "ARTIFACT_MANIFEST.json"
).is_file()
# The least overlap of a SAM 3 outline with the simulator's outline of the
# same cube (measured with the PyTorch route of before: 0.968 to 0.995; the
# reference's own figures are printed by the first test).
OVERLAP_FLOOR = 0.9

pytestmark = [
    pytest.mark.skipif(
        not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
        reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
    ),
    pytest.mark.skipif(
        not READY,
        reason="run scripts/setup_sam3_env.sh and build the reference's model files",
    ),
]


def _overlap(left: np.ndarray, right: np.ndarray) -> float:
    """Shared pixels of two outlines as a share of the pixels either has."""
    return float((left & right).sum() / max(int((left | right).sum()), 1))


def _coloured(picture: np.ndarray, color: str) -> int:
    """Pixels of a picture that have a cube's colour."""
    return int(cube_colours(picture.reshape(-1, 3))[color].sum())


def _check_provenance(provenance: dict) -> None:
    assert provenance["phrase"] == PHRASE == "cube"
    assert provenance["route"] == "sam3-tracker-rocm"
    assert provenance["source_commit"] == SETTINGS["commit"]
    assert provenance["source"]["commit"] == SETTINGS["commit"]
    assert len(provenance["manifest_sha256"]) == 64
    assert provenance["miopen_find_mode"] == "FAST"
    assert provenance["full_detection_every_call"] is True
    assert provenance["max_objects_per_prompt"] is None
    assert provenance["image_size"] == 504


def _compare(label: str, taken, outlined, found) -> dict[str, dict[str, float]]:
    """Print SAM 3's cube maps beside the simulator's; the overlap per camera and cube."""
    overlaps = {}
    for camera in CAMERAS:
        truth = getattr(taken, f"{camera}_outline")
        sam3 = getattr(outlined, f"{camera}_outline")
        overlaps[camera] = {
            color: _overlap(truth == OUTLINE_CODES[color], sam3 == OUTLINE_CODES[color])
            for color in BIN_CUBE_COLORS
        }
        unused = found[camera].unused
        on_bin = int((truth[unused] == OUTLINE_CODES["bin"]).sum())
        print(
            f"{label}, {camera}: {found[camera].seconds:.2f} s; outlines "
            f"{[(o.number, o.pixels, o.cube) for o in found[camera].found]}; overlap "
            f"with the simulator's {({c: round(v, 3) for c, v in overlaps[camera].items()})}; "
            f"on no cube {int(unused.sum())} pixels, {on_bin} of them on the bin"
        )
    return overlaps


def test_real_sam3_outlines_the_cubes_and_the_pictures_are_painted_by_them():
    floor = OVERLAP_FLOOR
    with ExitStack() as stack:
        env = SapienPolicyEnvironment(horizon=6, task_mode=THREE_CUBE_TASK)
        stack.callback(env.close)
        sam3 = Sam3Outlines()
        stack.callback(sam3.close)
        _check_provenance(sam3.provenance)
        print(json.dumps(sam3.provenance))
        # On this seed the wrist camera sees all three cubes.
        first = env.reset(environment_seed=7)
        facts = env.set_target("green")
        scene_without_cubes, _ = env.scene_without_cubes()
        with pytest.raises(RuntimeError, match="no layout was started"):
            sam3.outline(first)

        sam3.start_layout()
        outlined, found = sam3.outline(first)
        overlaps = _compare("seed 7, first picture", first, outlined, found)
        for camera in CAMERAS:
            sam3_map = getattr(outlined, f"{camera}_outline")
            assert sam3_map.shape == IMAGE_SHAPE[:2] and sam3_map.dtype == np.uint8
            # SAM 3 outlines cubes only: its map knows no robot and no bin.
            assert set(np.unique(sam3_map)) <= {
                OUTLINE_CODES[name] for name in ("other", *BIN_CUBE_COLORS)
            }
            # Seconds on the graphics processor (measured: 2.2 to 3.6).
            assert 0.0 < found[camera].seconds < 60.0
        # An outline of SAM 3 lies on the simulator's outline of the same cube
        # (measured overlaps: 0.968 to 0.995).
        for color in BIN_CUBE_COLORS:
            assert overlaps["wrist"][color] > floor
        for color in ("red", "green"):
            assert overlaps["top"][color] > floor
        # The outline with no cube's colour inside lies on the bin (measured:
        # 98,092 of 98,113 pixels).
        unused = found["wrist"].unused
        assert unused.sum() > 5000
        assert (first.wrist_outline[unused] == OUTLINE_CODES["bin"]).mean() > 0.95

        # Green is named; red and blue are painted over by SAM 3's outlines.
        shown, count = edit_pictures(
            outlined,
            ("red", "blue"),
            scene_without_cubes,
            np.asarray(facts["cameras"]["top"]),
            np.asarray(facts["cameras"]["wrist"]),
            facts["table_height"],
        )
        print(f"painted by SAM 3's outlines: {count}")
        for camera in CAMERAS:
            before, after = getattr(first, camera), getattr(shown, camera)
            truth = getattr(first, f"{camera}_outline")
            sam3_map = getattr(outlined, f"{camera}_outline")
            changed = np.any(before != after, axis=2)
            arm = int((changed & (truth == OUTLINE_CODES["robot"])).sum())
            for color in BIN_CUBE_COLORS:
                taken, given = _coloured(before, color), _coloured(after, color)
                was_outlined = bool((sam3_map == OUTLINE_CODES[color]).any())
                print(
                    f"{camera}: pixels of {color}'s colour {taken} taken, {given} "
                    f"given; outlined by SAM 3: {was_outlined}"
                )
                if color == "green":
                    # The named cube keeps every pixel.
                    assert taken > 500 and given == taken
                    where = truth == OUTLINE_CODES["green"]
                    assert np.array_equal(after[where], before[where])
                elif was_outlined:
                    # A painted cube is gone (measured: 0 pixels left of
                    # 1,207 to 25,820).
                    assert taken > 500 and given < 0.02 * taken
                else:
                    # A cube SAM 3 did not outline is not painted (measured:
                    # the top camera's blue cube, 696 pixels, partly hidden).
                    assert given == taken
            # The arm keeps its pixels where it touches no painted cube
            # (measured: no changed pixel on it).
            print(f"{camera}: changed pixels on the arm {arm}")
            assert arm < 50
        assert _coloured(first.wrist, "red") > 500 and _coloured(first.wrist, "blue") > 500
        assert bool((outlined.wrist_outline == OUTLINE_CODES["blue"]).any())

        # The next picture of the layout: a cube keeps the number SAM 3
        # follows it by.
        held = env.step(first.state)
        outlined_next, found_next = sam3.outline(held.observation)
        overlaps = _compare("seed 7, second picture", held.observation, outlined_next, found_next)
        for camera in CAMERAS:
            numbers = {o.cube: o.number for o in found[camera].found if o.cube}
            numbers_next = {o.cube: o.number for o in found_next[camera].found if o.cube}
            assert numbers and numbers_next == numbers
        for color in BIN_CUBE_COLORS:
            assert overlaps["wrist"][color] > floor

        # A new layout starts new: SAM 3 outlines the cubes where they stand
        # now. On this seed both cameras see all three (measured overlaps:
        # 0.983 to 0.995).
        second = env.reset(environment_seed=3)
        sam3.start_layout()
        outlined, found = sam3.outline(second)
        overlaps = _compare("seed 3, first picture", second, outlined, found)
        for camera in CAMERAS:
            for color in BIN_CUBE_COLORS:
                assert overlaps[camera][color] > floor
            # An outline on no cube is not required (the PyTorch route of
            # before gave one on the bin in both cameras here; the reference
            # gives one in the top camera only, measured 2026-11-22). Where
            # there is one, it lies on the bin.
            unused = found[camera].unused
            truth = getattr(second, f"{camera}_outline")
            if unused.any():
                assert (truth[unused] == OUTLINE_CODES["bin"]).mean() > 0.95

        # A picture of another size is refused before it is sent.
        with pytest.raises(ValueError, match="top picture is uint8 of shape"):
            sam3.outline(replace(second, top=second.top[:100]))
        # A worker that is gone is reported with its exit status, not waited for.
        sam3._process.kill()
        with pytest.raises(RuntimeError, match="SAM 3 worker exited with status -9"):
            sam3.outline(second)
        with pytest.raises(RuntimeError, match="SAM 3 worker is closed"):
            sam3.start_layout()


def test_real_three_cube_batch_with_sam3_outlines_through_the_command_line(
    tmp_path, monkeypatch, capsys
):
    """The whole test with the real SAM 3 and scene; a stand-in for SmolVLA holds the arm."""
    policy = SimpleNamespace(
        config=SimpleNamespace(n_action_steps=2), reset=lambda: None
    )
    shown = []

    def hold_the_arm(policy, pre, post, observation, task_text):
        shown.append(observation)
        return observation.state

    monkeypatch.setattr(
        three_cube_rollout, "_load_policy", lambda: (tmp_path / "revision", policy, None, None)
    )
    monkeypatch.setattr(three_cube_rollout, "_action", hold_the_arm)
    destination = tmp_path / "batch"
    sapien_rollout.main(
        [
            "--task", THREE_CUBE_TASK,
            "--pictures", "others-painted",
            "--outlines", "sam3",
            "--steps", "3",
            "--environment-seed-base", "7",
            "--order", "green,red,blue",
            "--output", str(destination),
        ]
    )
    assert "succeeded 0 of 1" in capsys.readouterr().out

    rows = json.loads((destination / "summary.json").read_text(encoding="utf-8"))
    assert len(rows) == 1 and len(shown) == 9
    row = rows[0]
    assert row["outline_source"] == OUTLINE_SOURCES["sam3"]
    _check_provenance(row["sam3"])
    assert row["order"] == ["green", "red", "blue"]
    assert [plan["step"] for plan in row["plans"]] == [0, 2, 3, 5, 6, 8]
    for plan in row["plans"]:
        for camera in CAMERAS:
            asked = plan["sam3"][camera]
            print(
                f"plan at step {plan['step']}, {camera}: {asked['seconds']:.2f} s, "
                f"{[(o['number'], o['pixels'], o['cube']) for o in asked['outlines']]}"
            )
            assert asked["seconds"] > 0.0
            assert {o["cube"] for o in asked["outlines"]} <= {None, *BIN_CUBE_COLORS}
    print({pick["cube"]: pick["painted_pixels"] for pick in row["picks"]})

    folder = destination / "seed-00007"
    stored = np.load(folder / "policy-observations.npz")
    assert stored["step"].tolist() == [0, 2, 3, 5, 6, 8]
    assert np.array_equal(stored["top"][0], shown[0].top)
    assert np.array_equal(stored["wrist"][0], shown[0].wrist)
    # First plan, green named: where SAM 3 outlined the red cube, the camera
    # took the red cube and SmolVLA was given none (measured in the scene:
    # 0 pixels of its colour left of 1,207 and of 4,014).
    for camera in CAMERAS:
        taken, given = stored[f"{camera}_taken"][0], stored[camera][0]
        assert (stored[f"{camera}_outlines"][0] == 1).sum() > 500
        assert _coloured(taken, "red") > 500
        assert _coloured(given, "red") < 0.02 * _coloured(taken, "red")
        assert _coloured(given, "green") == _coloured(taken, "green") > 500
        legend = row["plans"][0]["outlines"][camera]
        assert legend[:3] == [
            {"label": "red cube", "role": "painted over"},
            {"label": "green cube", "role": "target"},
            {"label": "blue cube", "role": "painted over"},
        ]
    # The wrist camera's outline on the bin is shown and not used.
    assert row["plans"][0]["outlines"]["wrist"][3] == {
        "label": "no cube colour", "role": "not used",
    }
    assert (stored["wrist_outlines"][0] == 4).sum() > 5000
    assert row["picks"][0]["painted_pixels"]["wrist_mean"] > 1500
    assert row["picks"][0]["painted_pixels"]["top_mean"] > 1000
    # Between two plans the pictures go on as the cameras took them.
    assert _coloured(shown[1].wrist, "red") > 500

    # The video: a frame per step, a row per camera. In the middle picture
    # the red cube's outline has the tint of a painted-over cube and the
    # bin's the grey of an outline that is not used; the picture given to
    # SmolVLA shows no red cube there.
    row_height = IMAGE_SHAPE[0] + CAPTION_HEIGHT
    feeds = read_clip(folder / "feeds.mp4")
    assert feeds.shape == (9, 2 * row_height, 3 * IMAGE_SHAPE[1], 3)
    assert read_clip(destination / "overall-feeds.mp4").shape == feeds.shape
    for index, camera in enumerate(CAMERAS):
        pictures = feeds[
            0, index * row_height + CAPTION_HEIGHT : (index + 1) * row_height
        ].astype(int)
        middle, right = pictures[:, 640:1280], pictures[:, 1280:]
        taken_picture = stored[f"{camera}_taken"][0].astype(int)
        for number, role in ((1, "painted over"), (4, "not used")):
            where = stored[f"{camera}_outlines"][0] == number
            if not where.any():
                # The top camera's picture has no outline off the cubes here.
                assert (camera, number) == ("top", 4)
                continue
            taken = np.median(taken_picture[where], axis=0)
            tinted = TINT * np.asarray(ROLE_COLORS[role]) + (1.0 - TINT) * taken
            in_clip = np.median(middle[where], axis=0)
            print(
                f"{camera} camera, outline {number} ({int(where.sum())} pixels, {role}): "
                f"taken {taken}, in the clip {in_clip} (tint {np.round(tinted)})"
            )
            assert np.abs(in_clip - tinted).max() <= 15
        red = stored[f"{camera}_outlines"][0] == 1
        given = np.median(right[red], axis=0)
        assert given[0] - max(given[1], given[2]) < 10

    assert json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8")) == {
        "complete": True,
        "episodes": 1,
        "task_mode": THREE_CUBE_TASK,
        "pictures": "others-painted",
        "outlines": "sam3",
    }
    report = (destination / "REPORT.md").read_text(encoding="utf-8")
    assert "Outlines the picture editor paints by: SAM 3 outlines." in report
    assert "| 7 | 0 | user | green out - | red out - | blue out - | none | fail |" in report
