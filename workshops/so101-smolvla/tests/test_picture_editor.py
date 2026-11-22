import numpy as np
import pytest

from so101_smolvla.picture_editor import (
    PAINT_MARGIN_PX,
    PaintCount,
    edit_pictures,
    paint_over,
)
from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.sapien_config import OUTLINE_CODES

OTHER, ROBOT, BIN = (OUTLINE_CODES[name] for name in ("other", "robot", "bin"))
RED, GREEN, BLUE = (OUTLINE_CODES[name] for name in ("red", "green", "blue"))
SHAPE = (120, 160)
TABLE = 200  # grey level of the scene without the objects
# A camera 0.5 m above the table that looks straight down on it.
LOOKING_DOWN = np.array(
    [[200.0, 0.0, 80.0], [0.0, 200.0, 60.0], [0.0, 0.0, 1.0]]
) @ np.array(
    [[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.5]]
)


def _scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A picture with a red and a green square, its outline map and the empty scene."""
    outline = np.full(SHAPE, OTHER, dtype=np.uint8)
    outline[40:60, 30:50] = RED
    outline[40:60, 100:120] = GREEN
    # The robot touches the red square on its right side, the bin lies below it.
    outline[40:60, 50:56] = ROBOT
    outline[62:70, 30:50] = BIN
    picture = np.full((*SHAPE, 3), TABLE, dtype=np.uint8)
    picture[outline == RED] = (180, 40, 40)
    picture[outline == GREEN] = (40, 180, 40)
    picture[outline == ROBOT] = (255, 140, 0)
    picture[outline == BIN] = (70, 70, 70)
    empty = np.full((*SHAPE, 3), TABLE, dtype=np.uint8)
    empty[outline == BIN] = (70, 70, 70)
    return picture, outline, empty


def test_only_the_removed_outline_and_its_rim_are_painted():
    picture, outline, empty = _scene()
    edited, painted, kept = paint_over(picture, outline, (RED,), empty, margin=3)

    # The red square shows the scene without it.
    assert np.all(edited[outline == RED] == TABLE)
    # The robot beside it, the green square and the bin keep every pixel.
    for code in (ROBOT, GREEN, BIN):
        assert np.array_equal(edited[outline == code], picture[outline == code])
    changed = np.any(edited != picture, axis=-1)
    assert np.array_equal(changed, outline == RED)
    # Painted are the square and a rim of at most three pixels around it that
    # lies on the table or the bin, not on the robot.
    assert painted > int((outline == RED).sum())
    assert kept == 0
    rows, columns = np.nonzero(changed)
    assert rows.min() == 40 and rows.max() == 59
    assert columns.min() == 30 and columns.max() == 49


def test_the_rim_removes_a_fringe_of_the_objects_colour_but_never_robot_pixels():
    picture, outline, empty = _scene()
    # One pixel of red has bled onto the table left of the square and one onto
    # the robot right of it.
    picture[50, 29] = (190, 120, 120)
    picture[50, 50] = (190, 120, 120)
    edited, _, _ = paint_over(picture, outline, (RED,), empty, margin=PAINT_MARGIN_PX)
    assert edited[50, 29].tolist() == [TABLE] * 3
    assert edited[50, 50].tolist() == [190, 120, 120]
    # Without a margin the fringe on the table stays.
    unwidened, painted, _ = paint_over(picture, outline, (RED,), empty, margin=0)
    assert unwidened[50, 29].tolist() == [190, 120, 120]
    assert painted == int((outline == RED).sum())


def test_pixels_without_paint_are_kept_and_counted():
    picture, outline, empty = _scene()
    paintable = np.ones(SHAPE, dtype=bool)
    paintable[:50] = False
    edited, painted, kept = paint_over(
        picture, outline, (RED,), empty, margin=0, paintable=paintable
    )
    upper = (outline == RED) & ~paintable
    lower = (outline == RED) & paintable
    assert np.array_equal(edited[upper], picture[upper])
    assert np.all(edited[lower] == TABLE)
    assert (painted, kept) == (int(lower.sum()), int(upper.sum()))


def test_paint_over_rejects_pictures_that_do_not_fit():
    picture, outline, empty = _scene()
    with pytest.raises(ValueError, match="outline map of shape"):
        paint_over(picture, outline[:-1], (RED,), empty, margin=1)
    with pytest.raises(ValueError, match="paint is"):
        paint_over(picture, outline, (RED,), empty[:, :-1], margin=1)
    with pytest.raises(ValueError, match="invalid picture"):
        paint_over(picture.astype(np.float32), outline, (RED,), empty, margin=1)
    with pytest.raises(ValueError, match="invalid margin"):
        paint_over(picture, outline, (RED,), empty, margin=-1)
    with pytest.raises(ValueError, match="paintable is"):
        paint_over(
            picture, outline, (RED,), empty, margin=1, paintable=np.ones(SHAPE)
        )


def _observation() -> tuple[PolicyObservation, np.ndarray]:
    picture, outline, empty = _scene()
    # The wrist camera sees the same scene with the two squares swapped.
    wrist_outline = outline.copy()
    wrist_outline[outline == RED] = GREEN
    wrist_outline[outline == GREEN] = RED
    wrist = picture.copy()
    wrist[wrist_outline == RED] = (180, 40, 40)
    wrist[wrist_outline == GREEN] = (40, 180, 40)
    return (
        PolicyObservation(
            state=np.zeros(6),
            top=picture,
            wrist=wrist,
            top_outline=outline,
            wrist_outline=wrist_outline,
        ),
        empty,
    )


def test_both_pictures_lose_the_named_objects_and_keep_the_rest():
    observation, empty = _observation()
    edited, count = edit_pictures(
        observation, ("red",), empty, LOOKING_DOWN, LOOKING_DOWN, table_height=0.0
    )
    for before, after, outline in (
        (observation.top, edited.top, observation.top_outline),
        (observation.wrist, edited.wrist, observation.wrist_outline),
    ):
        assert np.all(after[outline == RED] == TABLE)
        for code in (ROBOT, GREEN, BIN):
            assert np.array_equal(after[outline == code], before[outline == code])
    assert count.top >= 400 and count.wrist >= 400 and count.wrist_kept == 0
    # State and outline maps are those of the scene, and the observation that
    # was handed in is not changed.
    assert edited.state is observation.state
    assert edited.top_outline is observation.top_outline
    assert np.any(observation.top[observation.top_outline == RED] != TABLE)


def test_the_wrist_is_painted_with_the_scene_as_the_wrist_camera_sees_it():
    observation, empty = _observation()
    # The scene without the objects has a mark the top camera sees at
    # (row 45, column 105), inside the wrist picture's red square.
    empty[45, 105] = (10, 20, 30)
    # A wrist camera that stands 0.05 m further along x sees the table shifted
    # by 0.05 m * 200 px / 0.5 m = 20 columns to the left.
    shifted = LOOKING_DOWN.copy()
    shifted[:, 3] -= LOOKING_DOWN[:, 0] * 0.05
    same_view, _ = edit_pictures(
        observation, ("red",), empty, LOOKING_DOWN, LOOKING_DOWN, table_height=0.0
    )
    moved_view, _ = edit_pictures(
        observation, ("red",), empty, LOOKING_DOWN, shifted, table_height=0.0
    )
    assert same_view.wrist[45, 105].tolist() == [10, 20, 30]
    assert moved_view.wrist[45, 105].tolist() == [TABLE] * 3
    # The top picture is painted with the scene as the top camera took it.
    assert np.array_equal(same_view.top, moved_view.top)


def test_wrist_pixels_that_look_past_the_table_keep_the_object():
    observation, empty = _observation()
    # A wrist camera below the table plane looks at none of it.
    below = LOOKING_DOWN.copy()
    below[2, 3] = -0.5
    edited, count = edit_pictures(
        observation, ("red",), empty, LOOKING_DOWN, below, table_height=0.0
    )
    assert np.array_equal(edited.wrist, observation.wrist)
    assert count.wrist == 0
    assert count.wrist_kept >= int((observation.wrist_outline == RED).sum())
    assert count.top >= 400


def test_nothing_to_remove_returns_the_observation_as_it_is():
    observation, empty = _observation()
    edited, count = edit_pictures(
        observation, (), empty, LOOKING_DOWN, LOOKING_DOWN, table_height=0.0
    )
    assert edited is observation
    assert count == PaintCount()


def test_an_object_only_the_top_camera_sees_leaves_the_wrist_picture_alone():
    observation, empty = _observation()
    edited, count = edit_pictures(
        observation, ("blue",), empty, LOOKING_DOWN, LOOKING_DOWN, table_height=0.0
    )
    assert edited.wrist is observation.wrist
    assert np.array_equal(edited.top, observation.top)
    assert count == PaintCount()


def test_editing_needs_outline_maps_and_known_names():
    observation, empty = _observation()
    bare = PolicyObservation(
        state=observation.state, top=observation.top, wrist=observation.wrist
    )
    with pytest.raises(ValueError, match="no outline maps"):
        edit_pictures(bare, ("red",), empty, LOOKING_DOWN, LOOKING_DOWN, 0.0)
    with pytest.raises(ValueError, match="no outline is named 'purple'"):
        edit_pictures(observation, ("purple",), empty, LOOKING_DOWN, LOOKING_DOWN, 0.0)
