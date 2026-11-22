from __future__ import annotations

import numpy as np
import pytest

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.sam3_outlines import (
    COLOUR_SHARE,
    CameraOutlines,
    FoundOutline,
    cube_colour,
    cube_colours,
    read_outlines,
)
from so101_smolvla.sapien_config import IMAGE_SHAPE, OUTLINE_CODES

# The colours of the three cubes in the scene's top picture (medians measured
# in test_real_three_cube_outline_maps_show_what_the_pictures_show).
CUBES = {"red": (198, 135, 134), "green": (111, 190, 123), "blue": (185, 200, 211)}
# What else the scene shows: the table, the dark bin, the arm's orange paint
# (sapien_config.ROBOT_ORANGE) and a bluish-grey part of the top view.
NO_CUBE = {
    "table": (220, 220, 220),
    "bin": (70, 70, 70),
    "arm": (255, 89, 0),
    "bluish grey": (107, 111, 118),
    "black": (0, 0, 0),
}
RED = (slice(100, 140), slice(100, 140))
GREEN = (slice(100, 140), slice(300, 340))
BIN = (slice(300, 430), slice(220, 350))


def _pixels(*colours) -> np.ndarray:
    return np.asarray(colours, dtype=np.uint8)


def _scene() -> np.ndarray:
    picture = np.full(IMAGE_SHAPE, NO_CUBE["table"], dtype=np.uint8)
    picture[BIN] = NO_CUBE["bin"]
    picture[RED] = CUBES["red"]
    picture[GREEN] = CUBES["green"]
    return picture


def _mask(*areas) -> np.ndarray:
    mask = np.zeros(IMAGE_SHAPE[:2], dtype=bool)
    for area in areas:
        mask[area] = True
    return mask


def test_each_cube_colour_is_read_by_its_hue_and_nothing_else_of_the_scene_is_a_cube():
    assert tuple(cube_colours(_pixels(CUBES["red"]))) == BIN_CUBE_COLORS
    for cube, colour in CUBES.items():
        # In full light and in shade: a darker face keeps its hue.
        for shade in (1.0, 0.6):
            shaded = tuple(int(shade * level) for level in colour)
            read = cube_colours(_pixels(shaded))
            assert {name for name, inside in read.items() if inside[0]} == {cube}
    for name, colour in NO_CUBE.items():
        read = cube_colours(_pixels(colour))
        assert not any(inside[0] for inside in read.values()), name
    # Many pixels at once, each read by itself.
    read = cube_colours(_pixels(CUBES["red"], NO_CUBE["table"], CUBES["blue"]))
    assert read["red"].tolist() == [True, False, False]
    assert read["blue"].tolist() == [False, False, True]
    assert read["green"].tolist() == [False, False, False]
    with pytest.raises(ValueError, match="invalid pixels"):
        cube_colours(np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="invalid pixels"):
        cube_colours(np.zeros((4, 3), dtype=np.float64))


def test_an_outline_is_the_cube_whose_colour_enough_of_its_pixels_have():
    picture = _scene()
    assert COLOUR_SHARE == 0.5
    assert cube_colour(picture, _mask(RED)) == "red"
    assert cube_colour(picture, _mask(GREEN)) == "green"
    # Half cube and half table is still the cube; less than half is not.
    half = _mask(RED, (slice(100, 140), slice(140, 180)))
    assert cube_colour(picture, half) == "red"
    less = _mask(RED, (slice(100, 140), slice(140, 181)))
    assert cube_colour(picture, less) is None
    # An outline over two cubes of equal size is neither's by a majority.
    assert cube_colour(picture, _mask(RED, GREEN)) in ("red", "green")
    assert cube_colour(picture, _mask(RED, GREEN, (slice(0, 2), slice(0, 2)))) is None
    assert cube_colour(picture, _mask(BIN)) is None
    assert cube_colour(picture, _mask()) is None


def test_sam3s_outlines_become_the_editors_map_and_the_rest_is_kept_apart():
    picture = _scene()
    # A wide outline with no cube's colour that lies over the red cube too.
    wide = _mask((slice(60, 180), slice(60, 260)))
    masks = np.stack([_mask(BIN), _mask(RED), _mask(GREEN), wide])
    read = read_outlines(picture, masks, [2, 0, 3, 7], 6.4)

    assert isinstance(read, CameraOutlines)
    assert read.seconds == 6.4
    assert read.found == (
        FoundOutline(number=2, pixels=130 * 130, cube=None),
        FoundOutline(number=0, pixels=40 * 40, cube="red"),
        FoundOutline(number=3, pixels=40 * 40, cube="green"),
        FoundOutline(number=7, pixels=120 * 200, cube=None),
    )
    assert read.cubes.shape == IMAGE_SHAPE[:2] and read.cubes.dtype == np.uint8
    assert np.all(read.cubes[RED] == OUTLINE_CODES["red"])
    assert np.all(read.cubes[GREEN] == OUTLINE_CODES["green"])
    # No outline is the blue cube, the bin or the robot: all else is "other".
    assert set(np.unique(read.cubes)) == {
        OUTLINE_CODES["other"], OUTLINE_CODES["red"], OUTLINE_CODES["green"],
    }
    assert int((read.cubes != OUTLINE_CODES["other"]).sum()) == 2 * 40 * 40
    # The outlines on no cube, without the pixels a cube's outline holds.
    assert read.unused.dtype == bool
    assert np.all(read.unused[BIN])
    assert not read.unused[RED].any()
    assert int(read.unused.sum()) == 130 * 130 + 120 * 200 - 40 * 40

    nothing = read_outlines(picture, np.zeros((0, *IMAGE_SHAPE[:2]), dtype=bool), [], 2.0)
    assert nothing.found == ()
    assert np.all(nothing.cubes == OUTLINE_CODES["other"]) and not nothing.unused.any()


def test_outlines_that_do_not_fit_their_picture_are_refused():
    picture = _scene()
    masks = np.stack([_mask(RED)])
    with pytest.raises(ValueError, match="invalid picture"):
        read_outlines(picture.astype(np.float32), masks, [0], 1.0)
    with pytest.raises(ValueError, match="outlines are bool of shape"):
        read_outlines(picture, masks[:, :100], [0], 1.0)
    with pytest.raises(ValueError, match="outlines are uint8"):
        read_outlines(picture, masks.astype(np.uint8), [0], 1.0)
    with pytest.raises(ValueError, match="outlines are bool of shape"):
        read_outlines(picture, masks[0], [0], 1.0)
    with pytest.raises(ValueError, match="1 outlines with 2 numbers"):
        read_outlines(picture, masks, [0, 1], 1.0)
