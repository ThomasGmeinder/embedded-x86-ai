import json
from types import SimpleNamespace

import numpy as np
import pytest

from clip_reading import read_clip
from so101_smolvla import three_cube_rollout
from so101_smolvla.bin_layouts import (
    BIN_CUBE_COLORS,
    bin_footprint_corners,
    bin_footprint_distance,
)
from so101_smolvla.feeds_video import CAPTION_HEIGHT, ROLE_COLORS, TINT
from so101_smolvla.picture_editor import edit_pictures
from so101_smolvla.reachable_workspace import (
    REACHABLE_HIGH,
    REACHABLE_LOW,
    is_reachable_xy,
)
from so101_smolvla.sapien_config import (
    IMAGE_SHAPE,
    OUTLINE_CODES,
    SAPIEN_PYTHON,
    SAPIEN_SOURCE,
    THREE_CUBE_TASK,
)
from so101_smolvla.sapien_env_client import SapienPolicyEnvironment
from so101_smolvla.stack_layouts import COLOR_CUBE_CLEARANCES, COLOR_CUBE_DISTANCE

pytestmark = pytest.mark.skipif(
    not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
    reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
)

SENTENCE = "Pick up a cube and place in the bin"
# The simulator's trained bin is 80 x 100 x 30 mm; the test's is one and a
# half times as long and as wide.
BIN_HALF_SIZES = [0.060, 0.075, 0.015]
TRAINED_BIN_REACH = float(np.hypot(0.040, 0.050))


def _pixel(projection: list, point: list) -> np.ndarray:
    """Column and row at which a camera shows a point of the scene."""
    column, row, depth = np.asarray(projection) @ np.array([*point, 1.0])
    return np.array([column / depth, row / depth])


def test_real_three_cube_scene_has_three_cubes_a_larger_bin_and_a_valid_layout():
    env = SapienPolicyEnvironment(horizon=2, task_mode=THREE_CUBE_TASK)
    try:
        first = env.reset(environment_seed=3)
        facts = dict(env.facts)
        env.reset(environment_seed=3)
        assert env.facts["initial_layout"] == facts["initial_layout"]
        env.reset(environment_seed=4)
        assert env.facts["initial_layout"] != facts["initial_layout"]
    finally:
        env.close()

    # SmolVLA is given the sentence of the trained 40 mm bin task.
    assert env.task_text == SENTENCE
    assert env.provenance["task_text"] == SENTENCE
    assert env.provenance["task_mode"] == THREE_CUBE_TASK
    assert first.top.shape == IMAGE_SHAPE and first.wrist.shape == IMAGE_SHAPE

    # No cube is named yet, so no cube counts as placed.
    assert facts["target"] is None
    assert facts["success"] is False
    assert facts["table_height"] == 0.0

    assert facts["bin"]["half_sizes"] == pytest.approx(BIN_HALF_SIZES, abs=1e-6)
    assert facts["cube_half_size"] == pytest.approx(0.020, abs=1e-6)
    assert tuple(facts["cubes"]) == BIN_CUBE_COLORS
    layout = facts["initial_layout"]
    bin_xy, bin_yaw = np.asarray(layout["bin_xy"]), layout["bin_yaw"]
    assert facts["bin"]["position"][:2] == pytest.approx(bin_xy.tolist(), abs=1e-6)
    assert np.cos(facts["bin"]["yaw"] - bin_yaw) == pytest.approx(1.0, abs=1e-6)
    assert is_reachable_xy(bin_xy)
    corners = bin_footprint_corners(bin_xy, bin_yaw, BIN_HALF_SIZES[:2])
    assert np.all(corners >= REACHABLE_LOW - TRAINED_BIN_REACH - 1e-9)
    assert np.all(corners <= REACHABLE_HIGH + TRAINED_BIN_REACH + 1e-9)

    centres = []
    for color, clearance in zip(BIN_CUBE_COLORS, COLOR_CUBE_CLEARANCES, strict=True):
        cube = facts["cubes"][color]
        # A cube stands where the layout puts it, resting on the table.
        assert cube["position"][:2] == pytest.approx(layout["cubes"][color], abs=1e-6)
        assert cube["position"][2] == pytest.approx(0.020, abs=1e-4)
        assert is_reachable_xy(cube["position"][:2], clearance)
        assert cube["in_bin"] is False
        assert cube["above_table"] is False
        assert cube["grasped"] is False
        assert cube["touched_by_robot"] is False
        assert cube["speed"] == 0.0
        grip = facts["grip"][color]
        assert grip["contact"] is False
        assert 0.0 < grip["distance"] < 0.5
        centres.append(cube["position"][:2])
    centres = np.asarray(centres)
    for left in range(3):
        for right in range(left + 1, 3):
            assert (
                np.linalg.norm(centres[left] - centres[right])
                >= COLOR_CUBE_DISTANCE - 1e-6
            )
    # Half a cube and the simulator's two 10 mm margins lie between a cube's
    # centre and the bin.
    assert np.all(
        bin_footprint_distance(centres, bin_xy, bin_yaw, BIN_HALF_SIZES[:2])
        >= 0.040 - 1e-6
    )


def test_real_three_cube_layouts_start_with_bin_and_cubes_clear_of_the_resting_arm():
    """Nothing is pushed by the arm before it moves.

    The gripper rests lower than a cube and than the bin's walls at the near
    edge of the workspace. Set down there, the bin was thrown 44 mm and a
    cube 58 mm by the first five steps (measured on 22 Nov 2026 before bin
    and cubes were kept off that part of the table).
    """
    env = SapienPolicyEnvironment(horizon=5, task_mode=THREE_CUBE_TASK)
    try:
        for seed in range(10):
            first = env.reset(environment_seed=seed)
            stood = {
                color: np.asarray(cube["position"])
                for color, cube in env.facts["cubes"].items()
            }
            bin_stood = np.asarray(env.facts["bin"]["position"])
            bin_yaw = env.facts["bin"]["yaw"]
            for _ in range(5):
                held = env.step(first.state)
            for color in BIN_CUBE_COLORS:
                moved = np.asarray(held.facts["cubes"][color]["position"]) - stood[color]
                assert np.linalg.norm(moved) < 1e-4, (seed, color)
                assert held.facts["cubes"][color]["touched_by_robot"] is False
            # The bin only settles the 2.5 mm the simulator sets it down above
            # the table (so101_sim/tasks/place.py:380).
            bin_moved = np.asarray(held.facts["bin"]["position"]) - bin_stood
            assert np.linalg.norm(bin_moved[:2]) < 1e-4, seed
            assert bin_moved[2] == pytest.approx(-0.0025, abs=2e-4)
            assert np.cos(held.facts["bin"]["yaw"] - bin_yaw) > 1.0 - 1e-6
            assert held.facts["robot_touching_bin"] is False
    finally:
        env.close()


def test_real_three_cube_outline_maps_show_what_the_pictures_show():
    env = SapienPolicyEnvironment(horizon=2, task_mode=THREE_CUBE_TASK)
    try:
        # On this seed the arm hides no part of a cube from the top camera.
        first = env.reset(environment_seed=3)
        facts = dict(env.facts)
    finally:
        env.close()

    for outline in (first.top_outline, first.wrist_outline):
        assert outline.shape == IMAGE_SHAPE[:2]
        assert outline.dtype == np.uint8
        assert set(np.unique(outline)) <= set(OUTLINE_CODES.values())
    # The top camera sees the robot, the bin and all three cubes.
    assert set(np.unique(first.top_outline)) == set(OUTLINE_CODES.values())

    def shown(name: str) -> np.ndarray:
        return first.top[first.top_outline == OUTLINE_CODES[name]].astype(int)

    # A 40 mm cube covers more than a thousand pixels of the top picture
    # (measured: 1,227 to 1,365 on this seed).
    for color in BIN_CUBE_COLORS:
        assert 800 < len(shown(color)) < 2500
    # Each outline lies on its cube's colour: red, green and the simulator's
    # pale blue (measured medians 198/135/134, 111/190/123, 185/200/211).
    red, green, blue = (np.median(shown(color), axis=0) for color in BIN_CUBE_COLORS)
    assert red[0] - max(red[1], red[2]) > 40
    assert green[1] - max(green[0], green[2]) > 40
    assert blue[2] > blue[1] > blue[0] and blue[2] - blue[0] > 15
    # The bin is the trained dark grey and the table is light.
    assert np.median(shown("bin"), axis=0).max() < 110
    assert np.median(shown("other"), axis=0).min() > 200

    # Outline, cube position and camera agree: the middle of a cube's outline
    # is where the top camera shows the cube's centre (measured: 0.95 to
    # 1.48 pixels apart).
    for color in BIN_CUBE_COLORS:
        rows, columns = np.nonzero(first.top_outline == OUTLINE_CODES[color])
        middle = np.array([columns.mean(), rows.mean()])
        expected = _pixel(facts["cameras"]["top"], facts["cubes"][color]["position"])
        assert np.linalg.norm(middle - expected) < 3.0
    # The same holds for the wrist camera wherever it sees a whole cube side.
    for color in BIN_CUBE_COLORS:
        rows, columns = np.nonzero(first.wrist_outline == OUTLINE_CODES[color])
        if len(rows) == 0:
            continue
        expected = _pixel(facts["cameras"]["wrist"], facts["cubes"][color]["position"])
        if not (0 <= expected[0] < IMAGE_SHAPE[1] and 0 <= expected[1] < IMAGE_SHAPE[0]):
            continue
        assert first.wrist_outline[int(expected[1]), int(expected[0])] in (
            OUTLINE_CODES[color],
            OUTLINE_CODES["robot"],
        )


def test_real_three_cube_scene_can_be_shown_without_cubes_and_arm():
    env = SapienPolicyEnvironment(horizon=3, task_mode=THREE_CUBE_TASK)
    try:
        first = env.reset(environment_seed=0)
        empty_top, empty_wrist = env.scene_without_cubes()
        held = env.step(first.state)
    finally:
        env.close()

    assert empty_top.shape == IMAGE_SHAPE and empty_wrist.shape == IMAGE_SHAPE
    table = np.median(first.top[first.top_outline == OUTLINE_CODES["other"]], axis=0)
    for picture, taken, outline in (
        (empty_top, first.top, first.top_outline),
        (empty_wrist, first.wrist, first.wrist_outline),
    ):
        # Where a cube stood the light table shows, or the dark grey bin
        # behind it (on this seed 163 of the blue cube's 20,021 wrist pixels).
        for color in BIN_CUBE_COLORS:
            shown = picture[outline == OUTLINE_CODES[color]].astype(int)
            assert len(shown) > 500
            light = np.abs(shown - table).max(axis=1) <= 12
            dark_grey = (shown.max(axis=1) < 110) & (np.ptp(shown, axis=1) <= 6)
            assert np.all(light | dark_grey)
            assert light.mean() > 0.95
        # The bin and the table are drawn as before.
        for name in ("bin", "other"):
            where = outline == OUTLINE_CODES[name]
            same = np.abs(picture[where].astype(int) - taken[where]).max(axis=1) <= 2
            assert same.mean() > 0.99
    # The top camera sees the orange arm (11,991 orange pixels on this seed);
    # without it none is left, and the table shows where it stood (94.75 %;
    # the rest is the grey strip along the lower edge of the picture).
    arm = first.top_outline == OUTLINE_CODES["robot"]

    def orange(picture: np.ndarray) -> int:
        shown = picture[arm].astype(int)
        return int((shown[:, 0] - shown[:, 2] > 60).sum())

    assert orange(first.top) > 5000
    assert orange(empty_top) == 0
    assert (np.abs(empty_top[arm].astype(int) - table).max(axis=1) <= 12).mean() > 0.9

    # Cubes and arm are drawn again afterwards, and the outline maps of the
    # next observation still show them.
    assert np.abs(
        held.observation.top.astype(int) - first.top.astype(int)
    ).max(axis=2).mean() < 1.0
    for name in (*BIN_CUBE_COLORS, "robot"):
        before = int((first.top_outline == OUTLINE_CODES[name]).sum())
        after = int((held.observation.top_outline == OUTLINE_CODES[name]).sum())
        assert abs(after - before) <= 0.1 * before


def test_real_three_cube_scene_takes_the_named_cube_and_refuses_an_unknown_one():
    env = SapienPolicyEnvironment(horizon=3, task_mode=THREE_CUBE_TASK)
    try:
        first = env.reset(environment_seed=5)
        for color in ("green", "red", "blue"):
            facts = env.set_target(color)
            assert facts["target"] == color
            # The cube stands on the table, so naming it does not place it.
            assert facts["success"] is False
        with pytest.raises(RuntimeError, match="unknown cube 'purple'"):
            env.set_target("purple")
        # The refused name leaves the scene and its target as they were.
        held = env.step(first.state)
        assert held.facts["target"] == "blue"
        assert held.success is False
        assert all(not cube["in_bin"] for cube in held.facts["cubes"].values())
        # A new layout starts without a named cube.
        env.reset(environment_seed=5)
        assert env.facts["target"] is None
    finally:
        env.close()

    # A scene without several cubes has none to name.
    plain = SapienPolicyEnvironment(horizon=2, task_mode="pick-place")
    try:
        plain_first = plain.reset(environment_seed=0)
        assert plain_first.top_outline is None and plain_first.wrist_outline is None
        with pytest.raises(RuntimeError, match="no cube to name"):
            plain.set_target("green")
        with pytest.raises(RuntimeError, match="cannot show its scene without cubes"):
            plain.scene_without_cubes()
    finally:
        plain.close()


def test_real_three_cube_pictures_lose_the_other_cubes_and_keep_the_named_one():
    env = SapienPolicyEnvironment(horizon=2, task_mode=THREE_CUBE_TASK)
    try:
        # On this seed the wrist camera sees all three cubes.
        first = env.reset(environment_seed=7)
        facts = env.set_target("blue")
        scene_without_cubes, _ = env.scene_without_cubes()
    finally:
        env.close()
    edited, count = edit_pictures(
        first,
        ("red", "green"),
        scene_without_cubes,
        np.asarray(facts["cameras"]["top"]),
        np.asarray(facts["cameras"]["wrist"]),
        facts["table_height"],
    )
    table = np.median(first.top[first.top_outline == OUTLINE_CODES["other"]], axis=0)
    for before, after, outline, painted in (
        (first.top, edited.top, first.top_outline, count.top),
        (first.wrist, edited.wrist, first.wrist_outline, count.wrist),
    ):
        for color in ("red", "green"):
            where = outline == OUTLINE_CODES[color]
            assert where.sum() > 500
            # Where the cube was, the picture shows the light table.
            is_table = np.abs(after[where].astype(int) - table).max(axis=1) <= 12
            assert is_table.mean() > 0.98
        # The named cube and the arm keep every pixel.
        for name in ("blue", "robot"):
            where = outline == OUTLINE_CODES[name]
            assert where.sum() > 500
            assert np.array_equal(after[where], before[where])
        removed = np.isin(outline, [OUTLINE_CODES["red"], OUTLINE_CODES["green"]])
        assert painted >= int(removed.sum())
        # Nothing but the cubes and a rim of a few pixels around them changed.
        changed = np.any(after != before, axis=2)
        assert int((changed & ~removed).sum()) < 0.25 * int(removed.sum())
    # The wrist camera looks down at the table where these cubes stand.
    assert count.wrist_kept == 0


def test_real_three_cube_batch_runs_a_command_per_cube_and_writes_its_report(
    tmp_path, monkeypatch
):
    """The whole test in the real scene, with a stand-in for SmolVLA that holds the arm."""
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
    rows = three_cube_rollout.run_three_cube_batch(
        1,
        destination=destination,
        pictures="others-painted",
        steps_per_pick=3,
        environment_seed_base=7,
        order=("green", "red", "blue"),
    )

    assert len(rows) == 1 and len(shown) == 9
    row = rows[0]
    assert row["environment_seed"] == 7
    assert row["order"] == ["green", "red", "blue"]
    assert row["task"] == SENTENCE
    assert row["provenance"]["task_mode"] == THREE_CUBE_TASK
    # The arm is held where it started, so every cube stays on the table.
    assert row["success"] is False
    assert row["cubes_in_bin"] == {"red": False, "green": False, "blue": False}
    assert row["cube_heights_mm"] == pytest.approx(
        {"red": 20.0, "green": 20.0, "blue": 20.0}, abs=0.2
    )
    for pick in row["picks"]:
        assert pick["success"] is False and pick["in_bin_at_end"] is False
        assert pick["arm_from_start_deg"] < 0.5
        assert max(pick["others_moved_mm"].values()) < 0.5
        # The two cubes the command does not name are painted in both pictures.
        assert pick["painted_pixels"]["top_mean"] > 1500
        assert pick["painted_pixels"]["wrist_mean"] > 1500

    # While green is named the policy is shown no red cube: the top picture
    # it gets differs from the real one where the red cube stands.
    folder = destination / "seed-00007"
    stored = np.load(folder / "policy-observations.npz")
    assert stored["top"].shape == (6, *IMAGE_SHAPE)
    assert np.array_equal(stored["top"][0], shown[0].top)
    assert json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))["complete"]
    assert (folder / "top.mp4").stat().st_size > 0
    assert (destination / "overall-top.mp4").stat().st_size > 0

    # The layout's video has a frame per step and a row per camera: the
    # camera's picture, the outlines the picture editor was given and the
    # picture SmolVLA was given. The batch's video joins the layouts'.
    row_height = IMAGE_SHAPE[0] + CAPTION_HEIGHT
    feeds = read_clip(folder / "feeds.mp4")
    assert feeds.shape == (9, 2 * row_height, 3 * IMAGE_SHAPE[1], 3)
    assert read_clip(destination / "overall-feeds.mp4").shape == feeds.shape
    assert read_clip(folder / "rollout.mp4").shape == (
        9, IMAGE_SHAPE[0], 2 * IMAGE_SHAPE[1], 3,
    )
    assert row["outline_source"] == "simulator outlines"
    assert stored["step"].tolist() == [0, 2, 3, 5, 6, 8]
    assert [plan["step"] for plan in row["plans"]] == [0, 2, 3, 5, 6, 8]
    for camera in ("top", "wrist"):
        assert row["plans"][0]["outlines"][camera] == [
            {"label": "red cube", "role": "painted over"},
            {"label": "green cube", "role": "target"},
            {"label": "blue cube", "role": "painted over"},
        ]
    # First frame, while green is named. The arm is held, so the camera still
    # shows what it took when SmolVLA planned. Where the red cube's outline
    # lies, the camera's picture shows the red cube, the picture given to
    # SmolVLA does not, and the outline picture shows it under the tint of a
    # painted-over cube. The clip's colours are a few levels off the stored
    # pictures'.
    painted_over = np.asarray(ROLE_COLORS["painted over"])
    for index, camera in enumerate(("top", "wrist")):
        red = stored[f"{camera}_outlines"][0] == 1
        assert red.sum() > 500
        pictures = feeds[
            0, index * row_height + CAPTION_HEIGHT : (index + 1) * row_height
        ].astype(int)
        in_clip = [
            np.median(pictures[:, third * 640 : (third + 1) * 640][red], axis=0)
            for third in range(3)
        ]
        taken = np.median(stored[f"{camera}_taken"][0][red].astype(int), axis=0)
        given = np.median(stored[camera][0][red].astype(int), axis=0)
        tinted = TINT * painted_over + (1.0 - TINT) * taken
        print(
            f"{camera} camera, red cube's outline ({int(red.sum())} pixels): taken "
            f"{taken}, given {given}; in the clip {in_clip[0]}, {in_clip[1]} "
            f"(tint {np.round(tinted)}), {in_clip[2]}"
        )
        assert taken[0] - max(taken[1], taken[2]) > 40
        assert given[0] - max(given[1], given[2]) < 10
        assert np.abs(in_clip[0] - taken).max() <= 15
        assert np.abs(in_clip[1] - tinted).max() <= 15
        assert np.abs(in_clip[2] - given).max() <= 15
    assert json.loads((destination / "summary.json").read_text(encoding="utf-8")) == rows
    assert json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8")) == {
        "complete": True,
        "episodes": 1,
        "task_mode": THREE_CUBE_TASK,
        "pictures": "others-painted",
        "outlines": "simulator",
    }
    report = (destination / "REPORT.md").read_text(encoding="utf-8")
    assert "Outlines the picture editor paints by: simulator outlines." in report
    assert "All three cubes in the bin at the end: 0 of 1 layouts." in report
    assert "Cubes in the bin at the end: 0 of 3." in report
    assert "| 7 | 0 | user | green out - | red out - | blue out - | none | fail |" in report
