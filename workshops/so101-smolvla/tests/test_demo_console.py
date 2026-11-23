"""demo_console: the composed picture, drawn from a snapshot without any thread or server."""

from __future__ import annotations

import numpy as np

from so101_smolvla.demo_console import (
    DIM,
    FRAME_SHAPE,
    GREEN,
    GRIP_BAR,
    RED,
    SILENT_AFTER_S,
    TILES_W,
    VAD_BAR,
    Snapshot,
    VoiceLevel,
    compose,
    jpeg,
    voice_state,
)
from so101_smolvla.feeds_video import Outline, PlanFeeds
from so101_smolvla.sapien_config import IMAGE_SHAPE


def _snapshot(**fields) -> Snapshot:
    return Snapshot(seed=0, status="idle", outline_source="SAM 3 outlines", steps_per_pick=400, **fields)


def _bar_fill(frame: np.ndarray, bar: tuple, colour: tuple) -> int:
    """Pixels of ``colour`` along the middle row of a panel bar."""
    x, y, w, h = bar
    row = frame[y + h // 2, TILES_W + x + 1 : TILES_W + x + w - 1]
    return int(np.all(row == colour, axis=-1).sum())


def test_the_picture_has_the_six_tiles_and_the_panel_and_encodes_as_jpeg():
    frame = compose(_snapshot(), (), VoiceLevel(), 0.0)
    assert frame.shape == FRAME_SHAPE and frame.dtype == np.uint8
    assert jpeg(frame)[:2] == b"\xff\xd8"


def test_the_voice_meter_fills_with_the_level_green_while_speaking_and_says_why_it_is_silent():
    now = 100.0
    loud = VoiceLevel(available=True, level=0.5, speaking=True, time=now)
    frame = compose(_snapshot(), (), loud, now)
    fill = _bar_fill(frame, VAD_BAR, GREEN)
    assert abs(fill - 0.5 * (VAD_BAR[2] - 2)) <= 2
    quiet = VoiceLevel(available=True, level=0.25, speaking=False, time=now)
    frame = compose(_snapshot(), (), quiet, now)
    assert abs(_bar_fill(frame, VAD_BAR, DIM) - 0.25 * (VAD_BAR[2] - 2)) <= 2
    assert _bar_fill(frame, VAD_BAR, GREEN) == 0
    assert voice_state(VoiceLevel(), now) == ("no microphone; type a command", RED)
    assert voice_state(VoiceLevel(error="no device"), now) == ("no microphone: no device", RED)
    assert voice_state(quiet, now + SILENT_AFTER_S + 0.5) == ("no microphone signal", RED)
    assert voice_state(VoiceLevel(available=True, busy=True, time=now), now)[0] == "asking Gemma..."
    assert voice_state(loud, now)[0] == "hearing you"
    assert voice_state(quiet, now)[0] == "listening"
    # Without a microphone no bar is drawn at all.
    assert _bar_fill(compose(_snapshot(), (), VoiceLevel(), now), VAD_BAR, DIM) == 0


def test_the_gripper_bar_shows_the_opening_and_turns_green_on_contact():
    open_free = compose(_snapshot(opening=40.0), (), VoiceLevel(), 0.0)
    assert abs(_bar_fill(open_free, GRIP_BAR, DIM) - 0.4 * (GRIP_BAR[2] - 2)) <= 2
    assert _bar_fill(open_free, GRIP_BAR, GREEN) == 0
    touching = compose(_snapshot(opening=40.0, contact=True), (), VoiceLevel(), 0.0)
    assert abs(_bar_fill(touching, GRIP_BAR, GREEN) - 0.4 * (GRIP_BAR[2] - 2)) <= 2
    assert _bar_fill(compose(_snapshot(), (), VoiceLevel(), 0.0), GRIP_BAR, DIM) == 0  # no reading, no bar


def test_the_tiles_show_the_pictures_and_the_plan_in_force():
    rng = np.random.default_rng(0)
    taken = rng.integers(0, 255, IMAGE_SHAPE, dtype=np.uint8)
    outlines = np.zeros(IMAGE_SHAPE[:2], dtype=np.uint8)
    outlines[100:140, 100:140] = 1
    plan = PlanFeeds(step=50, command="pick_place(object: red cube, place: box)", taken=taken, outlines=outlines,
                     legend=(Outline("red cube", "target"),), shown=taken)
    empty = compose(_snapshot(), (), VoiceLevel(), 0.0)
    full = compose(
        _snapshot(pictures={"top": taken, "wrist": taken}, plans={"top": plan, "wrist": plan},
                  command=plan.command, cube="red", step=7),
        (("you: put the red cube in the box", "you"), ("-> pick_place(object: red cube, place: box)", "call")),
        VoiceLevel(), 0.0,
    )
    assert full.shape == empty.shape
    tiles_region = (slice(0, FRAME_SHAPE[0]), slice(0, TILES_W))
    assert not np.array_equal(full[tiles_region], empty[tiles_region])
    # The placeholder tiles are one flat colour (every channel constant); the pictures are not.
    assert np.ptp(empty[40:200, 10:300].reshape(-1, 3), axis=0).max() == 0
    assert np.ptp(full[40:200, 10:300].reshape(-1, 3), axis=0).max() > 0
    panel_region = (slice(0, FRAME_SHAPE[0]), slice(TILES_W, FRAME_SHAPE[1]))
    assert not np.array_equal(full[panel_region], empty[panel_region])  # the transcript is drawn
