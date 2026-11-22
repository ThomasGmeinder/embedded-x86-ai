from __future__ import annotations

import numpy as np
import pytest

from so101_smolvla.feeds_video import (
    CAPTION_HEIGHT,
    ROLE_COLORS,
    TINT,
    Outline,
    PlanFeeds,
    caption,
    feed_frames,
    outline_picture,
)
from so101_smolvla.three_cube_rollout import OUTLINE_SOURCES, pick_place_command

SHAPE = (480, 640, 3)
ROW = SHAPE[0] + CAPTION_HEIGHT
SOURCE = OUTLINE_SOURCES["simulator"]
LEGEND = (
    Outline("red cube", "target"),
    Outline("green cube", "painted over"),
    Outline("blue cube", "left as it is"),
)


def _flat(value: int) -> np.ndarray:
    return np.full(SHAPE, value, dtype=np.uint8)


def _plan(step: int, taken: int, shown: int) -> PlanFeeds:
    return PlanFeeds(
        step=step,
        command=f"command at step {step}",
        taken=_flat(taken),
        outlines=np.zeros(SHAPE[:2], dtype=np.uint8),
        legend=(),
        shown=_flat(shown),
    )


def test_a_caption_is_a_dark_line_with_white_text_and_refuses_text_that_does_not_fit():
    line = caption("top camera, step 12", 640)
    assert line.shape == (CAPTION_HEIGHT, 640, 3) and line.dtype == np.uint8
    assert np.all(line[0] == 30) and np.all(line[:, 400:] == 30)
    assert line[:, :200].max() > 200
    # The longest captions of the three-cube test fit above a camera picture
    # (measured: 495 and 229 of 640 pixels).
    caption(f"wrist camera, step 1199: {pick_place_command('green')}", 640)
    for source in OUTLINE_SOURCES.values():
        caption(f"{source} at step 1150", 640)
    with pytest.raises(ValueError, match="needs .* pixels, the picture has 100"):
        caption("top camera, step 12", 100)


def test_an_outline_is_tinted_edged_and_named_in_the_colour_of_its_role():
    taken = _flat(220)
    outlines = np.zeros(SHAPE[:2], dtype=np.uint8)
    outlines[100:140, 100:140] = 1
    # The second outline touches the top and the right edge of the picture.
    outlines[2:42, 600:640] = 2
    drawn = outline_picture(taken, outlines, LEGEND)

    assert drawn.shape == SHAPE and drawn.dtype == np.uint8
    assert np.all(taken == 220)
    for (rows, columns), role in (
        ((slice(104, 136), slice(104, 136)), "target"),
        ((slice(6, 38), slice(604, 636)), "painted over"),
    ):
        tint = np.round(TINT * np.asarray(ROLE_COLORS[role]) + (1.0 - TINT) * 220)
        assert np.all(drawn[rows, columns] == tint.astype(np.uint8))
    # The edge of an outline is its role's colour.
    assert tuple(drawn[100, 120]) == ROLE_COLORS["target"]
    assert tuple(drawn[120, 100]) == ROLE_COLORS["target"]
    assert tuple(drawn[20, 600]) == ROLE_COLORS["painted over"]
    # The first name stands above its outline. The second outline leaves no
    # room above it, so its name stands below, inside the picture.
    assert np.any(drawn[81:98, 100:220] != 220)
    assert np.any(drawn[45:64, 460:640] != 220)
    # Nothing else is drawn: the blue cube is not in the picture.
    changed = np.any(drawn != taken, axis=2)
    assert not changed[150:].any()
    assert not changed[:, :98].any()
    assert not changed[70:, 230:].any()


def test_an_outline_the_editor_was_not_given_is_drawn_in_grey():
    taken = _flat(60)
    outlines = np.zeros(SHAPE[:2], dtype=np.uint8)
    outlines[200:300, 200:400] = 4
    legend = (*LEGEND, Outline("no cube colour", "not used"))
    drawn = outline_picture(taken, outlines, legend)

    assert ROLE_COLORS["not used"] == (150, 150, 150)
    tint = np.round(TINT * 150 + (1.0 - TINT) * 60).astype(np.uint8)
    assert np.all(drawn[206:294, 206:394] == tint)
    assert tuple(drawn[200, 300]) == ROLE_COLORS["not used"]
    # Its name stands above it; the picture below and beside it is untouched.
    assert np.any(drawn[181:198, 200:400] != 60)
    changed = np.any(drawn != taken, axis=2)
    assert not changed[304:].any() and not changed[:, :196].any()


def test_a_picture_without_outlines_stays_as_it_was_taken():
    taken = _flat(90)
    drawn = outline_picture(taken, np.zeros(SHAPE[:2], dtype=np.uint8), LEGEND)
    assert np.array_equal(drawn, taken) and drawn is not taken


def test_outlines_that_do_not_fit_their_picture_or_legend_are_refused():
    taken = _flat(90)
    outlines = np.zeros(SHAPE[:2], dtype=np.uint8)
    with pytest.raises(ValueError, match="invalid picture"):
        outline_picture(taken[:, :, 0], outlines, LEGEND)
    with pytest.raises(ValueError, match="outlines are uint8 of shape"):
        outline_picture(taken, outlines[:100], LEGEND)
    with pytest.raises(ValueError, match="outlines are bool"):
        outline_picture(taken, outlines.astype(bool), LEGEND)
    outlines[10:20, 10:20] = 4
    with pytest.raises(ValueError, match="outline number 4 has no entry"):
        outline_picture(taken, outlines, LEGEND)
    with pytest.raises(ValueError, match="unknown role 'hidden'"):
        outline_picture(taken, outlines, (*LEGEND, Outline("bin", "hidden")))


def test_a_frame_has_a_row_per_camera_of_camera_outlines_and_given_picture():
    frames = {
        "top": [_flat(10 + step) for step in range(5)],
        "wrist": [_flat(110 + step) for step in range(5)],
    }
    plans = {
        "top": [_plan(0, 50, 60), _plan(3, 51, 61)],
        "wrist": [_plan(0, 150, 160), _plan(3, 151, 161)],
    }
    video = list(feed_frames(frames, plans, SOURCE))

    assert len(video) == 5
    captions = []
    for step, frame in enumerate(video):
        assert frame.shape == (2 * ROW, 3 * 640, 3) and frame.dtype == np.uint8
        # The plan made at step 3 is in force from the frame after step 3 on.
        plan = 0 if step < 3 else 1
        for row, camera in enumerate(("top", "wrist")):
            pictures = frame[row * ROW + CAPTION_HEIGHT : (row + 1) * ROW]
            assert np.array_equal(pictures[:, :640], frames[camera][step])
            # No outline was given, so the middle shows what the camera took.
            assert np.array_equal(pictures[:, 640:1280], plans[camera][plan].taken)
            assert np.array_equal(pictures[:, 1280:], plans[camera][plan].shown)
        captions.append(frame[:CAPTION_HEIGHT])
        # Every picture has text above it.
        for third in range(3):
            assert frame[:CAPTION_HEIGHT, third * 640 : (third + 1) * 640].max() > 200
            assert frame[ROW : ROW + CAPTION_HEIGHT, third * 640 : (third + 1) * 640].max() > 200
    # The camera's caption counts the steps; the other two change with the plan.
    assert not np.array_equal(captions[0][:, :640], captions[1][:, :640])
    assert np.array_equal(captions[0][:, 640:], captions[2][:, 640:])
    assert not np.array_equal(captions[2][:, 640:1280], captions[3][:, 640:1280])
    assert not np.array_equal(captions[2][:, 1280:], captions[3][:, 1280:])


def test_the_middle_picture_shows_the_outlines_of_the_plan_in_force():
    outlines = np.zeros(SHAPE[:2], dtype=np.uint8)
    outlines[200:240, 300:340] = 2
    outlined = PlanFeeds(
        step=0,
        command=pick_place_command("red"),
        taken=_flat(220),
        outlines=outlines,
        legend=LEGEND,
        shown=_flat(220),
    )
    frame = next(
        feed_frames({"top": [_flat(220)]}, {"top": [outlined]}, SOURCE)
    )
    assert frame.shape == (ROW, 3 * 640, 3)
    middle = frame[CAPTION_HEIGHT:, 640:1280]
    assert np.array_equal(middle, outline_picture(outlined.taken, outlines, LEGEND))
    assert tuple(middle[200, 320]) == ROLE_COLORS["painted over"]


def test_frames_and_plans_that_do_not_belong_together_are_refused():
    frames = {"top": [_flat(1), _flat(2)], "wrist": [_flat(3), _flat(4)]}
    plans = {"top": [_plan(0, 5, 6)], "wrist": [_plan(0, 7, 8)]}
    with pytest.raises(ValueError, match="frames of cameras"):
        feed_frames(frames, {"top": plans["top"]}, SOURCE)
    with pytest.raises(ValueError, match="frames of cameras"):
        feed_frames({}, {}, SOURCE)
    with pytest.raises(ValueError, match="same number of frames"):
        feed_frames({**frames, "wrist": [_flat(3)]}, plans, SOURCE)
    with pytest.raises(ValueError, match="same number of frames"):
        feed_frames({"top": [], "wrist": []}, plans, SOURCE)
    for steps in ((), (1,), (0, 0), (0, 2, 1)):
        made = [_plan(step, 5, 6) for step in steps]
        with pytest.raises(ValueError, match="must start at 0 and go up"):
            feed_frames(frames, {**plans, "top": made}, SOURCE)
    small = PlanFeeds(
        step=0,
        command="command",
        taken=np.zeros((240, 320, 3), dtype=np.uint8),
        outlines=np.zeros((240, 320), dtype=np.uint8),
        legend=(),
        shown=np.zeros((240, 320, 3), dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="holds pictures of shape"):
        feed_frames(frames, {**plans, "wrist": [small]}, SOURCE)
