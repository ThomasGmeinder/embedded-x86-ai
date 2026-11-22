import numpy as np
import pytest

from so101_smolvla.bin_layouts import (
    BIN_CUBE_COLORS,
    BIN_POSITIONS_TRIED,
    BIN_SCALE,
    bin_footprint_corners,
    bin_footprint_distance,
    draw_bin_layout,
)
from so101_smolvla.reachable_workspace import (
    REACHABLE_HIGH,
    REACHABLE_LOW,
    is_reachable_xy,
)
from so101_smolvla.stack_layouts import COLOR_CUBE_CLEARANCES, COLOR_CUBE_DISTANCE

# Half the length and width of the simulator's trained bin (80 x 100 mm).
TRAINED_HALF = np.array([0.040, 0.050])
# The test's bin, the reach of a trained bin's corner beyond the workspace and
# the room a trained layout keeps between a cube's centre and the bin.
HALF = TRAINED_HALF * np.array(BIN_SCALE[:2])
OUTREACH = float(np.linalg.norm(TRAINED_HALF))
KEEP_CLEAR = 0.040
# The table under the parts of the resting arm that hang lower than a cube:
# gripper and jaw, as measured in the scene on seeds 0 to 9.
RESTING_GRIPPER = np.stack(
    np.meshgrid(np.linspace(0.141, 0.187, 11), np.linspace(-0.009, 0.043, 11)),
    axis=-1,
).reshape(-1, 2)
# What the scene keeps between that and the bin, and between that and the
# centre of a 40 mm cube: 10 mm, and half the cube's diagonal more.
BIN_KEEP_OFF = 0.010
CUBE_KEEP_OFF = 0.020 * np.sqrt(2.0) + 0.010


def test_the_bin_of_the_test_is_one_and_a_half_trained_bins_long_and_wide():
    assert BIN_SCALE == (1.5, 1.5, 1.0)
    assert HALF.tolist() == pytest.approx([0.060, 0.075])
    assert BIN_CUBE_COLORS == ("red", "green", "blue")


def test_footprint_distance_is_zero_inside_and_measured_from_the_nearest_edge():
    centre = np.array([0.30, 0.05])
    points = np.array(
        [
            [0.30, 0.05],  # the centre
            [0.35, 0.12],  # inside, near a corner
            [0.37, 0.05],  # 10 mm beyond the short half
            [0.30, 0.145],  # 20 mm beyond the long half
            [0.39, 0.165],  # 30 and 40 mm beyond a corner
        ]
    )
    distance = bin_footprint_distance(points, centre, 0.0, HALF)
    assert distance == pytest.approx([0.0, 0.0, 0.010, 0.020, 0.050])
    # Turned a quarter, the long side lies along x.
    turned = bin_footprint_distance(points, centre, np.pi / 2, HALF)
    assert turned[2] == pytest.approx(0.0)
    assert turned[3] == pytest.approx(0.095 - 0.060)


def test_footprint_corners_turn_with_the_bin():
    centre = np.array([0.30, 0.05])
    straight = bin_footprint_corners(centre, 0.0, HALF)
    assert sorted(map(tuple, np.round(straight, 6))) == [
        (0.24, -0.025),
        (0.24, 0.125),
        (0.36, -0.025),
        (0.36, 0.125),
    ]
    quarter = bin_footprint_corners(centre, np.pi / 2, HALF)
    assert np.ptp(quarter[:, 0]) == pytest.approx(0.150)
    assert np.ptp(quarter[:, 1]) == pytest.approx(0.120)
    # Every corner lies on the footprint's edge.
    assert bin_footprint_distance(quarter, centre, np.pi / 2, HALF) == pytest.approx(
        np.zeros(4), abs=1e-12
    )


def test_every_drawn_layout_replays_and_keeps_bin_and_cubes_reachable_and_apart():
    # Every one of these 200 seeds has a layout.
    for seed in range(200):
        layout = draw_bin_layout(
            np.random.default_rng(seed), HALF, OUTREACH, KEEP_CLEAR
        )
        assert layout == draw_bin_layout(
            np.random.default_rng(seed), HALF, OUTREACH, KEEP_CLEAR
        )
        bin_xy, bin_yaw = np.asarray(layout["bin_xy"]), layout["bin_yaw"]
        assert is_reachable_xy(bin_xy)
        assert -np.pi <= bin_yaw <= np.pi
        corners = bin_footprint_corners(bin_xy, bin_yaw, HALF)
        assert np.all(corners >= REACHABLE_LOW - OUTREACH)
        assert np.all(corners <= REACHABLE_HIGH + OUTREACH)
        assert tuple(layout["cubes"]) == BIN_CUBE_COLORS
        cubes = np.asarray(list(layout["cubes"].values()))
        for cube, clearance in zip(cubes, COLOR_CUBE_CLEARANCES, strict=True):
            assert is_reachable_xy(cube, clearance)
        for left in range(3):
            for right in range(left + 1, 3):
                assert np.linalg.norm(cubes[left] - cubes[right]) >= COLOR_CUBE_DISTANCE
        assert np.all(
            bin_footprint_distance(cubes, bin_xy, bin_yaw, HALF) >= KEEP_CLEAR
        )


def test_bin_and_cubes_keep_off_the_table_under_the_resting_gripper():
    assert BIN_POSITIONS_TRIED == 1000
    spread = []
    # Every one of these 200 seeds has a layout (measured on 22 Nov 2026;
    # with 100 bin positions tried instead of 1,000 it is 129 of 200).
    for seed in range(200):
        layout = draw_bin_layout(
            np.random.default_rng(seed),
            HALF,
            OUTREACH,
            KEEP_CLEAR,
            keep_off=RESTING_GRIPPER,
            bin_keep_off=BIN_KEEP_OFF,
            cube_keep_off=CUBE_KEEP_OFF,
        )
        bin_xy, bin_yaw = np.asarray(layout["bin_xy"]), layout["bin_yaw"]
        assert np.all(
            bin_footprint_distance(RESTING_GRIPPER, bin_xy, bin_yaw, HALF)
            > BIN_KEEP_OFF
        )
        cubes = np.asarray(list(layout["cubes"].values()))
        nearest = np.linalg.norm(cubes[:, None, :] - RESTING_GRIPPER, axis=-1).min()
        assert nearest > CUBE_KEEP_OFF
        # The other rules of a layout still hold.
        assert np.all(
            bin_footprint_distance(cubes, bin_xy, bin_yaw, HALF) >= KEEP_CLEAR
        )
        for cube, clearance in zip(cubes, COLOR_CUBE_CLEARANCES, strict=True):
            assert is_reachable_xy(cube, clearance)
        spread.append(np.ptp(cubes, axis=0))
    # The cubes are still scattered over the workspace, not set in a row:
    # measured medians 92.2 mm in x and 100.9 mm in y.
    assert np.all(np.median(spread, axis=0) > 0.080)


def test_a_point_to_keep_off_moves_only_a_bin_that_would_stand_on_it():
    plain = draw_bin_layout(np.random.default_rng(3), HALF, OUTREACH, KEEP_CLEAR)
    centre = np.asarray(plain["bin_xy"])
    # A point in the middle of the bin, with no distance asked for, is enough.
    moved = draw_bin_layout(
        np.random.default_rng(3), HALF, OUTREACH, KEEP_CLEAR, keep_off=centre[None]
    )
    assert moved["bin_xy"] != plain["bin_xy"]
    assert (
        bin_footprint_distance(
            centre[None], np.asarray(moved["bin_xy"]), moved["bin_yaw"], HALF
        )[0]
        > 0.0
    )
    # A point far from the workspace changes nothing, and neither do no points.
    for keep_off in (np.array([[2.0, 2.0]]), np.empty((0, 2))):
        assert plain == draw_bin_layout(
            np.random.default_rng(3),
            HALF,
            OUTREACH,
            KEEP_CLEAR,
            keep_off=keep_off,
            bin_keep_off=BIN_KEEP_OFF,
            cube_keep_off=CUBE_KEEP_OFF,
        )


def test_a_cube_keeps_its_distance_from_a_point_to_keep_off():
    plain = draw_bin_layout(np.random.default_rng(3), HALF, OUTREACH, KEEP_CLEAR)
    red = np.asarray(plain["cubes"]["red"])
    # A point 10 mm beside the red cube's centre, kept 30 mm from every cube.
    point = red + np.array([0.010, 0.0])
    moved = draw_bin_layout(
        np.random.default_rng(3),
        HALF,
        OUTREACH,
        KEEP_CLEAR,
        keep_off=point[None],
        cube_keep_off=0.030,
    )
    cubes = np.asarray(list(moved["cubes"].values()))
    assert np.all(np.linalg.norm(cubes - point, axis=1) > 0.030)
    assert moved["cubes"]["red"] != plain["cubes"]["red"]


def test_a_bin_that_leaves_no_room_fails_instead_of_standing_on_a_cube():
    refused = "no bin position of 1000 is free and leaves room"
    # A bin 300 mm square reaches beyond the workspace wherever it stands.
    with pytest.raises(RuntimeError, match=refused):
        draw_bin_layout(
            np.random.default_rng(0), np.array([0.150, 0.150]), OUTREACH, KEEP_CLEAR
        )
    # No cube can stand half a metre from the bin.
    with pytest.raises(RuntimeError, match=refused):
        draw_bin_layout(np.random.default_rng(0), HALF, OUTREACH, 0.5)
    # No bin stands a quarter of a metre from the resting gripper.
    with pytest.raises(RuntimeError, match=refused):
        draw_bin_layout(
            np.random.default_rng(0),
            HALF,
            OUTREACH,
            KEEP_CLEAR,
            keep_off=RESTING_GRIPPER,
            bin_keep_off=0.25,
        )
    # Nor does a cube.
    with pytest.raises(RuntimeError, match=refused):
        draw_bin_layout(
            np.random.default_rng(0),
            HALF,
            OUTREACH,
            KEEP_CLEAR,
            keep_off=RESTING_GRIPPER,
            cube_keep_off=0.25,
        )
    # A draw that is told to try fewer positions says how many it tried.
    with pytest.raises(RuntimeError, match="no bin position of 7 is free"):
        draw_bin_layout(np.random.default_rng(0), HALF, OUTREACH, 0.5, 7)
    for limits in ({"bin_keep_off": -0.01}, {"cube_keep_off": -0.01}):
        with pytest.raises(ValueError, match="invalid bin placement limits"):
            draw_bin_layout(
                np.random.default_rng(0),
                HALF,
                OUTREACH,
                KEEP_CLEAR,
                keep_off=RESTING_GRIPPER,
                **limits,
            )
    for points in (np.zeros(2), np.zeros((4, 3)), np.array([[np.nan, 0.0]])):
        with pytest.raises(ValueError, match="points to keep off are of shape"):
            draw_bin_layout(
                np.random.default_rng(0), HALF, OUTREACH, KEEP_CLEAR, keep_off=points
            )
    with pytest.raises(ValueError, match="invalid bin half sizes"):
        draw_bin_layout(
            np.random.default_rng(0), np.array([0.06, -0.075]), OUTREACH, KEEP_CLEAR
        )
    with pytest.raises(ValueError, match="invalid bin half sizes"):
        draw_bin_layout(np.random.default_rng(0), np.array([0.06]), OUTREACH, KEEP_CLEAR)
    with pytest.raises(ValueError, match="invalid bin placement limits"):
        draw_bin_layout(np.random.default_rng(0), HALF, -0.01, KEEP_CLEAR)
    with pytest.raises(ValueError, match="invalid bin placement limits"):
        draw_bin_layout(np.random.default_rng(0), HALF, OUTREACH, KEEP_CLEAR, 0)
