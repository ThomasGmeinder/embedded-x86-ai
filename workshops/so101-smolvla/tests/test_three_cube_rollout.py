from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from clip_reading import read_clip
from so101_smolvla import three_cube_rollout
from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.feeds_video import CAPTION_HEIGHT, ROLE_COLORS, TINT, Outline
from so101_smolvla.policy_env import PolicyObservation, PolicyStep
from so101_smolvla.sam3_outlines import CAMERAS, Sam3Outlines
from so101_smolvla.sapien_config import (
    IMAGE_SHAPE,
    OUTLINE_CODES,
    THREE_CUBE_TASK,
    task_spec,
)
from so101_smolvla.three_cube_rollout import (
    DEFAULT_STEPS_PER_PICK,
    OUTLINE_SOURCES,
    PICTURE_MODES,
    cube_outlines,
    cubes_to_paint,
    draw_order,
    parse_order,
    pick_place_command,
    run_three_cube_episode,
    write_three_cube_report,
)

TABLE = (220, 220, 220)
BIN = (70, 70, 70)
PAINT = {"red": (180, 40, 40), "green": (40, 180, 40), "blue": (40, 40, 180)}
# Where the stand-in scene draws each cube on the table and in the bin: rows
# and columns of a 40 pixel square, the same in both pictures.
ON_TABLE = {"red": (100, 100), "green": (100, 300), "blue": (100, 500)}
IN_BIN = {"red": (320, 240), "green": (320, 290), "blue": (370, 265)}
BIN_AREA = (slice(300, 430), slice(220, 350))
# A camera 0.5 m above the table that looks straight down on it.
CAMERA = (
    np.array([[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]])
    @ np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.5]])
).tolist()


def _square(corner: tuple[int, int]) -> tuple[slice, slice]:
    return slice(corner[0], corner[0] + 40), slice(corner[1], corner[1] + 40)


class StandInScene:
    """Three cubes that go into the bin on the second step of their command."""

    backend = "sapien"
    task_mode = THREE_CUBE_TASK
    task_text = task_spec(THREE_CUBE_TASK).instruction
    provenance = {"backend": "sapien"}

    def __init__(self, places: bool = True) -> None:
        self.places = places
        self.targets: list[str] = []
        self.in_bin = {color: False for color in BIN_CUBE_COLORS}
        self.steps_of_command = 0
        self.facts: dict[str, object] = {}

    def scene_without_cubes(self) -> tuple[np.ndarray, np.ndarray]:
        empty = np.full(IMAGE_SHAPE, TABLE, dtype=np.uint8)
        empty[BIN_AREA] = BIN
        return empty, empty.copy()

    def _observation(self) -> PolicyObservation:
        picture, _ = self.scene_without_cubes()
        outline = np.full(IMAGE_SHAPE[:2], OUTLINE_CODES["other"], dtype=np.uint8)
        outline[BIN_AREA] = OUTLINE_CODES["bin"]
        for color in BIN_CUBE_COLORS:
            where = _square((IN_BIN if self.in_bin[color] else ON_TABLE)[color])
            picture[where] = PAINT[color]
            outline[where] = OUTLINE_CODES[color]
        return PolicyObservation(
            state=np.zeros(6),
            top=picture,
            wrist=picture.copy(),
            top_outline=outline,
            wrist_outline=outline.copy(),
        )

    def _facts(self) -> dict[str, object]:
        target = self.targets[-1] if self.targets else None
        self.facts = {
            "table_height": 0.0,
            "target": target,
            "initial_layout": {"bin_xy": [0.3, 0.0], "bin_yaw": 0.0, "cubes": {}},
            "bin": {"position": [0.3, 0.0, 0.0], "yaw": 0.0},
            "cubes": {
                color: {
                    "position": [0.3, 0.0, 0.025]
                    if self.in_bin[color]
                    else [0.2, 0.05 * index, 0.02],
                    "in_bin": self.in_bin[color],
                    "above_table": self.in_bin[color],
                    "grasped": color == target and self.steps_of_command == 1,
                }
                for index, color in enumerate(BIN_CUBE_COLORS)
            },
            "grip": {
                color: {
                    "contact": color == target and self.steps_of_command == 1,
                    "distance": 0.1,
                    "finger_face_deg": 5.0,
                }
                for color in BIN_CUBE_COLORS
            },
            "cameras": {"top": CAMERA, "wrist": CAMERA},
            "success": target is not None and self.in_bin[target],
        }
        return self.facts

    def reset(self, environment_seed: int) -> PolicyObservation:
        self.seed = environment_seed
        self._facts()
        return self._observation()

    def set_target(self, cube: str) -> dict[str, object]:
        self.targets.append(cube)
        self.steps_of_command = 0
        return self._facts()

    def step(self, action: np.ndarray) -> PolicyStep:
        self.steps_of_command += 1
        if self.places and self.steps_of_command == 2:
            self.in_bin[self.targets[-1]] = True
        facts = self._facts()
        return PolicyStep(
            observation=self._observation(),
            commanded=action,
            achieved=action,
            success=bool(facts["success"]),
            facts=facts,
        )


class StandInPolicy:
    config = SimpleNamespace(n_action_steps=2)

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def _run(tmp_path, monkeypatch, pictures: str, **kwargs):
    """Run one layout in the stand-in scene; what the policy was shown, step by step."""
    shown: list[tuple[str, PolicyObservation, str]] = []
    scene = kwargs.pop("scene", None) or StandInScene()
    policy = StandInPolicy()

    def stand_in_action(policy, pre, post, observation, task_text):
        shown.append((scene.targets[-1], observation, task_text))
        return observation.state

    monkeypatch.setattr(three_cube_rollout, "_action", stand_in_action)
    result = run_three_cube_episode(
        scene,
        policy,
        None,
        None,
        Path("revision-1"),
        tmp_path / "seed-00004",
        environment_seed=4,
        policy_seed=10,
        steps_per_pick=3,
        pictures=pictures,
        **kwargs,
    )
    return result, shown, scene, policy


def _shows(picture: np.ndarray, color: str) -> bool:
    return bool(np.all(picture == PAINT[color], axis=-1).any())


def test_a_command_reads_as_the_tool_call_of_the_voice_control():
    assert pick_place_command("green") == "pick_place(object: green cube, place: box)"


def test_the_drawn_order_replays_and_every_order_of_three_cubes_occurs():
    orders = {draw_order(seed) for seed in range(60)}
    assert orders == set(permutations(BIN_CUBE_COLORS))
    assert draw_order(7) == draw_order(7)
    assert all(isinstance(color, str) for color in draw_order(7))


def test_the_users_order_names_every_cube_once():
    assert parse_order(" Green, red ,BLUE") == ("green", "red", "blue")
    for text in ("green,red", "green,red,red", "green,red,purple", "", "green red blue"):
        with pytest.raises(ValueError, match="must name each of red, green, blue once"):
            parse_order(text)


def test_each_picture_mode_names_the_cubes_it_paints():
    facts = {
        "cubes": {
            "red": {"in_bin": True},
            "green": {"in_bin": False},
            "blue": {"in_bin": False},
        }
    }
    assert set(PICTURE_MODES) == {"unedited", "others-painted", "table-others-painted"}
    assert cubes_to_paint("unedited", "green", facts) == ()
    assert cubes_to_paint("others-painted", "green", facts) == ("red", "blue")
    # The red cube lies in the bin and is left as it is.
    assert cubes_to_paint("table-others-painted", "green", facts) == ("blue",)
    assert cubes_to_paint("table-others-painted", "red", facts) == ("green", "blue")
    with pytest.raises(ValueError, match="unknown picture mode"):
        cubes_to_paint("blurred", "green", facts)


def test_a_layout_runs_one_command_per_cube_in_the_users_order(tmp_path, monkeypatch):
    order = ("green", "blue", "red")
    result, shown, scene, policy = _run(
        tmp_path, monkeypatch, "others-painted", order=order
    )

    # One target and one emptied action queue per command, in the given order,
    # and three steps each with the trained sentence.
    assert scene.targets == list(order)
    assert policy.resets == 3
    assert [target for target, _, _ in shown] == [
        color for color in order for _ in range(3)
    ]
    assert {text for _, _, text in shown} == {"Pick up a cube and place in the bin"}

    assert result["order"] == list(order)
    assert result["order_given_by"] == "user"
    assert result["pictures"] == "others-painted"
    assert result["task"] == "Pick up a cube and place in the bin"
    assert result["steps_per_pick"] == 3
    assert result["cubes_in_bin"] == {"red": True, "green": True, "blue": True}
    assert result["cubes_in_bin_count"] == 3
    assert result["success"] is True
    assert [pick["command"] for pick in result["picks"]] == [
        pick_place_command(color) for color in order
    ]
    for pick in result["picks"]:
        # The cube goes in on the second step of its command.
        assert pick["success"] is True
        assert pick["first_success_step"] == 1
        assert pick["in_bin_at_end"] is True
        assert pick["picked_up"] is True and pick["held"] is True
        assert pick["grip"] is True and pick["finger_face_deg"] == 5.0
        assert pick["arm_from_start_deg"] == 0.0
        assert set(pick["others_moved_mm"]) == set(BIN_CUBE_COLORS) - {pick["cube"]}
        assert max(pick["others_moved_mm"].values()) == 0.0
        assert pick["painted_pixels"]["top_mean"] >= 2 * 40 * 40

    folder = tmp_path / "seed-00004"
    assert json.loads((folder / "result.json").read_text(encoding="utf-8")) == result
    assert json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))["complete"]
    trajectory = np.load(folder / "trajectory.npz")
    assert trajectory["target"].shape == (9, 6)
    assert trajectory["success"].tolist() == [False, True, True] * 3
    for color in BIN_CUBE_COLORS:
        assert trajectory[f"cube_position_{color}"].shape == (9, 3)
    # What the policy was shown at the start of every chunk of two steps.
    stored = np.load(folder / "policy-observations.npz")
    assert stored["top"].shape == (6, *IMAGE_SHAPE)
    assert np.array_equal(stored["top"][0], shown[0][1].top)
    assert np.array_equal(stored["wrist"][1], shown[2][1].wrist)


def test_without_an_order_a_layout_draws_its_own_from_its_seed(tmp_path, monkeypatch):
    result, _, scene, _ = _run(tmp_path, monkeypatch, "unedited")
    assert tuple(result["order"]) == draw_order(4)
    assert result["order_given_by"] == "seeded draw"
    assert scene.targets == list(draw_order(4))


def test_others_painted_shows_the_policy_only_the_named_cube(tmp_path, monkeypatch):
    _, shown, _, _ = _run(
        tmp_path, monkeypatch, "others-painted", order=("green", "blue", "red")
    )
    for target, observation, _ in shown:
        for picture in (observation.top, observation.wrist):
            assert _shows(picture, target)
            for other in set(BIN_CUBE_COLORS) - {target}:
                assert not _shows(picture, other)
            # The bin is still there, and where a cube in it is painted over
            # the empty bin shows.
            assert np.all(picture[BIN_AREA][0, 0] == BIN)
    # During the last command the green and the blue cube lie in the bin.
    last = shown[-1][1].top
    assert np.all(last[_square(IN_BIN["green"])] == BIN)
    assert np.all(last[_square(ON_TABLE["green"])] == TABLE)


def test_table_others_painted_leaves_a_cube_in_the_bin_as_it_is(tmp_path, monkeypatch):
    _, shown, _, _ = _run(
        tmp_path, monkeypatch, "table-others-painted", order=("green", "blue", "red")
    )
    # First command: red and blue stand on the table and are painted over.
    first = shown[0][1].top
    assert _shows(first, "green")
    assert not _shows(first, "red") and not _shows(first, "blue")
    # Last command: green and blue lie in the bin and are shown.
    last = shown[-1][1].top
    assert np.all(last[_square(IN_BIN["green"])] == PAINT["green"])
    assert np.all(last[_square(IN_BIN["blue"])] == PAINT["blue"])
    assert _shows(last, "red")


def test_unedited_shows_the_policy_the_pictures_as_taken(tmp_path, monkeypatch):
    result, shown, _, _ = _run(tmp_path, monkeypatch, "unedited")
    first = shown[0][1]
    assert all(_shows(first.top, color) for color in BIN_CUBE_COLORS)
    assert all(_shows(first.wrist, color) for color in BIN_CUBE_COLORS)
    assert all(
        pick["painted_pixels"] == {"top_mean": 0.0, "wrist_mean": 0.0, "wrist_kept_mean": 0.0}
        for pick in result["picks"]
    )


def test_cube_outlines_number_the_cubes_and_say_what_is_done_with_each():
    outline_map = np.full(IMAGE_SHAPE[:2], OUTLINE_CODES["other"], dtype=np.uint8)
    outline_map[BIN_AREA] = OUTLINE_CODES["bin"]
    outline_map[:5, :5] = OUTLINE_CODES["robot"]
    outline_map[_square(ON_TABLE["red"])] = OUTLINE_CODES["red"]
    outline_map[_square(IN_BIN["blue"])] = OUTLINE_CODES["blue"]

    # The green cube is named and not in this picture; red is painted over.
    numbered, legend = cube_outlines(outline_map, "green", ("red",))
    assert numbered.shape == IMAGE_SHAPE[:2] and numbered.dtype == np.uint8
    assert np.all(numbered[_square(ON_TABLE["red"])] == 1)
    assert np.all(numbered[_square(IN_BIN["blue"])] == 3)
    # Table, bin and robot carry no number.
    assert int((numbered > 0).sum()) == 2 * 40 * 40
    assert legend == (
        Outline("red cube", "painted over"),
        Outline("green cube", "target"),
        Outline("blue cube", "left as it is"),
    )
    assert {outline.role for outline in legend} <= set(ROLE_COLORS)
    with pytest.raises(ValueError, match="hands out no outline maps"):
        cube_outlines(None, "green", ())

    # Outlines the editor was not given follow as a fourth entry. Here one
    # lies on the bin and over a corner of the red cube, which keeps its number.
    unused = np.zeros(IMAGE_SHAPE[:2], dtype=bool)
    unused[BIN_AREA] = True
    unused[130:150, 130:150] = True
    with_unused, legend = cube_outlines(outline_map, "green", ("red",), unused)
    assert legend[:3] == cube_outlines(outline_map, "green", ("red",))[1]
    assert legend[3] == Outline("no cube colour", "not used")
    assert np.all(with_unused[_square(ON_TABLE["red"])] == 1)
    assert np.all(with_unused[_square(IN_BIN["blue"])] == 3)
    assert np.all(with_unused[140:150, 140:150] == 4)
    assert int((with_unused == 4).sum()) == 130 * 130 - 40 * 40 + 20 * 20 - 10 * 10
    # Without a pixel of such an outline there is no fourth entry.
    same, legend = cube_outlines(outline_map, "green", ("red",), np.zeros_like(unused))
    assert np.array_equal(same, numbered) and len(legend) == 3
    with pytest.raises(ValueError, match="unused outlines are uint8"):
        cube_outlines(outline_map, "green", (), unused.astype(np.uint8))
    with pytest.raises(ValueError, match="unused outlines are bool of shape"):
        cube_outlines(outline_map, "green", (), unused[:100])


def _inside(picture: np.ndarray, corner: tuple[int, int]) -> np.ndarray:
    """The median colour of the middle of a cube's square in a picture."""
    middle = picture[corner[0] + 10 : corner[0] + 30, corner[1] + 10 : corner[1] + 30]
    return np.median(middle.reshape(-1, 3).astype(int), axis=0)


def _is(colour: np.ndarray, expected) -> bool:
    # A flat colour comes back from the clip a few levels off.
    return bool(np.abs(colour - np.asarray(expected)).max() <= 12)


def _tinted(role: str, under) -> np.ndarray:
    return TINT * np.asarray(ROLE_COLORS[role]) + (1.0 - TINT) * np.asarray(under)


def test_a_layout_writes_a_video_of_camera_outlines_and_given_pictures(
    tmp_path, monkeypatch
):
    order = ("green", "blue", "red")
    result, shown, _, _ = _run(tmp_path, monkeypatch, "others-painted", order=order)
    folder = tmp_path / "seed-00004"

    # A plan starts with every chunk of two steps, and with every command.
    starts = [0, 2, 3, 5, 6, 8]
    assert result["outline_source"] == OUTLINE_SOURCES["simulator"] == "simulator outlines"
    assert "sam3" not in result and all("sam3" not in plan for plan in result["plans"])
    assert [plan["step"] for plan in result["plans"]] == starts
    assert [plan["command"] for plan in result["plans"]] == [
        pick_place_command(color) for color in order for _ in range(2)
    ]
    for plan in result["plans"]:
        named = plan["command"].split("object: ")[1].split(",")[0]
        for camera in ("top", "wrist"):
            assert plan["outlines"][camera] == [
                {
                    "label": f"{color} cube",
                    "role": "target" if f"{color} cube" == named else "painted over",
                }
                for color in BIN_CUBE_COLORS
            ]
    stored = np.load(folder / "policy-observations.npz")
    assert stored["step"].tolist() == starts
    for camera in ("top", "wrist"):
        assert stored[f"{camera}_taken"].shape == (6, *IMAGE_SHAPE)
        assert stored[f"{camera}_outlines"].shape == (6, *IMAGE_SHAPE[:2])
        # The camera took all three cubes; the outlines carry their numbers.
        taken, outlines = stored[f"{camera}_taken"][0], stored[f"{camera}_outlines"][0]
        for number, color in enumerate(BIN_CUBE_COLORS, start=1):
            assert _shows(taken, color)
            assert np.all(outlines[_square(ON_TABLE[color])] == number)
        assert int((outlines > 0).sum()) == 3 * 40 * 40
    assert np.array_equal(stored["top"][0], shown[0][1].top)

    completion = json.loads((folder / "COMPLETE.json").read_text(encoding="utf-8"))
    assert "feeds.mp4" in completion["artifacts"]
    feeds = read_clip(folder / "feeds.mp4")
    row = IMAGE_SHAPE[0] + CAPTION_HEIGHT
    assert feeds.shape == (9, 2 * row, 3 * IMAGE_SHAPE[1], 3)
    # The two cameras side by side, a frame per step.
    assert read_clip(folder / "rollout.mp4").shape == (
        9, IMAGE_SHAPE[0], 2 * IMAGE_SHAPE[1], 3,
    )

    def panels(frame: int, camera: int):
        pictures = feeds[frame, camera * row + CAPTION_HEIGHT : (camera + 1) * row]
        return (
            pictures[:, index * IMAGE_SHAPE[1] : (index + 1) * IMAGE_SHAPE[1]]
            for index in range(3)
        )

    for camera in range(2):
        # After the first step of the command for green: all cubes on the table.
        taken, outlined, given = panels(0, camera)
        for color in BIN_CUBE_COLORS:
            assert _is(_inside(taken, ON_TABLE[color]), PAINT[color])
        # SmolVLA was given the green cube and the table where the others stand.
        assert _is(_inside(given, ON_TABLE["green"]), PAINT["green"])
        assert _is(_inside(given, ON_TABLE["red"]), TABLE)
        assert _is(_inside(given, ON_TABLE["blue"]), TABLE)
        # Every outline is tinted in the colour of what is done with its cube.
        assert _is(_inside(outlined, ON_TABLE["green"]), _tinted("target", PAINT["green"]))
        for other in ("red", "blue"):
            assert _is(
                _inside(outlined, ON_TABLE[other]), _tinted("painted over", PAINT[other])
            )
        # Away from the cubes all three show the table.
        for picture in (taken, outlined, given):
            assert _is(_inside(picture, (200, 50)), TABLE)

        # The plan of step 0 holds for the frame after step 1, on which the
        # camera already shows the green cube in the bin.
        taken, outlined, given = panels(1, camera)
        assert _is(_inside(taken, IN_BIN["green"]), PAINT["green"])
        assert _is(_inside(taken, ON_TABLE["green"]), TABLE)
        assert _is(_inside(given, ON_TABLE["green"]), PAINT["green"])
        assert _is(_inside(given, IN_BIN["green"]), BIN)

        # Last frame, the command for red: all cubes lie in the bin. SmolVLA
        # was given the red cube there and the empty bin where the others lie.
        taken, outlined, given = panels(8, camera)
        for color in BIN_CUBE_COLORS:
            assert _is(_inside(taken, IN_BIN[color]), PAINT[color])
        assert _is(_inside(given, IN_BIN["red"]), PAINT["red"])
        assert _is(_inside(given, IN_BIN["green"]), BIN)
        assert _is(_inside(given, IN_BIN["blue"]), BIN)
        assert _is(_inside(outlined, IN_BIN["red"]), _tinted("target", PAINT["red"]))
        assert _is(
            _inside(outlined, IN_BIN["green"]), _tinted("painted over", PAINT["green"])
        )


def test_the_video_says_which_cubes_were_left_as_they_are(tmp_path, monkeypatch):
    result, _, _, _ = _run(
        tmp_path, monkeypatch, "table-others-painted", order=("green", "blue", "red")
    )
    # Last plan, the command for red: green and blue lie in the bin and stay.
    assert result["plans"][-1]["outlines"]["top"] == [
        {"label": "red cube", "role": "target"},
        {"label": "green cube", "role": "left as it is"},
        {"label": "blue cube", "role": "left as it is"},
    ]
    feeds = read_clip(tmp_path / "seed-00004" / "feeds.mp4")
    pictures = feeds[8, CAPTION_HEIGHT : CAPTION_HEIGHT + IMAGE_SHAPE[0]]
    outlined, given = pictures[:, 640:1280], pictures[:, 1280:]
    assert _is(_inside(given, IN_BIN["green"]), PAINT["green"])
    assert _is(
        _inside(outlined, IN_BIN["green"]), _tinted("left as it is", PAINT["green"])
    )


class StandInSam3(Sam3Outlines):
    """SAM 3's client with the worker's replies made up from the picture.

    It outlines the red cube, the green cube and the bin, and never the blue
    cube. Which outline is which cube is read by the client itself.
    """

    provenance = {"model": "stand-in", "phrase": "cube"}
    NUMBERS = [5, 9, 2]
    SECONDS = {"top": 0.25, "wrist": 0.5}

    def __init__(self) -> None:
        self.requests: list[dict] = []

    def _ask(self, request: dict) -> dict:
        self.requests.append(request)
        if request["operation"] == "start":
            return {"started": True}
        assert request["operation"] == "outline"
        return {
            camera: {
                "masks": np.stack(
                    [
                        np.all(request[camera] == colour, axis=-1)
                        for colour in (PAINT["red"], PAINT["green"], BIN)
                    ]
                ),
                "numbers": list(self.NUMBERS),
                "seconds": self.SECONDS[camera],
            }
            for camera in CAMERAS
        }


def test_with_sam3_the_pictures_are_painted_by_its_outlines_of_every_plan(
    tmp_path, monkeypatch
):
    order = ("green", "blue", "red")
    sam3 = StandInSam3()
    result, shown, _, _ = _run(
        tmp_path, monkeypatch, "others-painted", order=order, sam3=sam3
    )
    folder = tmp_path / "seed-00004"
    stored = np.load(folder / "policy-observations.npz")

    # SAM 3 starts once for the layout and is asked for the pictures of the
    # six plans only, in their order.
    starts = [0, 2, 3, 5, 6, 8]
    operations = [request["operation"] for request in sam3.requests]
    assert operations == ["start"] + ["outline"] * 6
    assert stored["step"].tolist() == starts
    for index, request in enumerate(sam3.requests[1:]):
        for camera in CAMERAS:
            assert np.array_equal(request[camera], stored[f"{camera}_taken"][index])

    assert set(OUTLINE_SOURCES) == {"simulator", "sam3"}
    assert result["outline_source"] == OUTLINE_SOURCES["sam3"] == "SAM 3 outlines"
    assert result["sam3"] == StandInSam3.provenance
    for index, plan in enumerate(result["plans"]):
        assert plan["sam3"] == {
            camera: {
                "seconds": StandInSam3.SECONDS[camera],
                "outlines": [
                    {"number": 5, "pixels": 40 * 40, "cube": "red"},
                    {"number": 9, "pixels": 40 * 40, "cube": "green"},
                    {
                        "number": 2,
                        "pixels": int((stored[f"{camera}_outlines"][index] == 4).sum()),
                        "cube": None,
                    },
                ],
            }
            for camera in CAMERAS
        }
        for camera in CAMERAS:
            assert plan["outlines"][camera][3] == {
                "label": "no cube colour", "role": "not used",
            }
    # The bin's outline is on no cube: shown as the fourth entry, not painted.
    first_outlines = stored["top_outlines"][0]
    assert np.all(first_outlines[_square(ON_TABLE["red"])] == 1)
    assert np.all(first_outlines[_square(ON_TABLE["green"])] == 2)
    assert np.all(first_outlines[BIN_AREA] == 4)
    # SAM 3 did not outline the blue cube: it carries no number.
    assert not (first_outlines == 3).any()
    assert not first_outlines[_square(ON_TABLE["blue"])].any()

    # First plan, the command for green: red is painted over by SAM 3's
    # outline. Blue was to be painted too and stays, because SAM 3 missed it.
    for picture in (shown[0][1].top, shown[0][1].wrist):
        assert _shows(picture, "green")
        assert not _shows(picture, "red")
        assert _shows(picture, "blue")
        assert np.all(picture[BIN_AREA] == BIN)
    assert np.array_equal(stored["top"][0], shown[0][1].top)
    assert result["picks"][0]["painted_pixels"]["top_mean"] >= 40 * 40
    assert result["picks"][0]["painted_pixels"]["top_mean"] < 2 * 40 * 40
    # Between two plans SmolVLA uses no picture; the cameras' go on as taken.
    assert all(_shows(shown[1][1].top, color) for color in BIN_CUBE_COLORS)
    # Third plan, the command for blue (green lies in the bin): red and green
    # are painted over, on the table and in the bin.
    third = shown[3][1].top
    assert shown[3][0] == "blue" and _shows(third, "blue")
    assert not _shows(third, "red") and not _shows(third, "green")
    assert np.all(third[_square(IN_BIN["green"])] == BIN)

    # The video's middle picture shows all SAM 3 outlined: the cubes in the
    # colour of what is done with them, the bin in the grey of "not used",
    # the blue cube as the camera took it.
    feeds = read_clip(folder / "feeds.mp4")
    row = IMAGE_SHAPE[0] + CAPTION_HEIGHT
    for camera in range(2):
        pictures = feeds[0, camera * row + CAPTION_HEIGHT : (camera + 1) * row]
        outlined, given = pictures[:, 640:1280], pictures[:, 1280:]
        assert _is(_inside(outlined, ON_TABLE["green"]), _tinted("target", PAINT["green"]))
        assert _is(_inside(outlined, ON_TABLE["red"]), _tinted("painted over", PAINT["red"]))
        assert _is(_inside(outlined, ON_TABLE["blue"]), PAINT["blue"])
        assert _is(_inside(outlined, (385, 225)), _tinted("not used", BIN))
        assert _is(_inside(given, ON_TABLE["red"]), TABLE)
        assert _is(_inside(given, ON_TABLE["blue"]), PAINT["blue"])
        assert _is(_inside(given, (385, 225)), BIN)


def test_with_sam3_and_unedited_pictures_the_outlines_are_shown_and_nothing_is_painted(
    tmp_path, monkeypatch
):
    sam3 = StandInSam3()
    result, shown, _, _ = _run(tmp_path, monkeypatch, "unedited", sam3=sam3)
    stored = np.load(tmp_path / "seed-00004" / "policy-observations.npz")
    assert len(sam3.requests) == 7
    assert np.array_equal(stored["top"], stored["top_taken"])
    assert all(_shows(shown[0][1].top, color) for color in BIN_CUBE_COLORS)
    assert all(pick["painted_pixels"]["top_mean"] == 0.0 for pick in result["picks"])
    # No cube is painted: the two SAM 3 outlined are the target or left.
    assert {entry["role"] for entry in result["plans"][0]["outlines"]["top"]} == {
        "target", "left as it is", "not used",
    }


def test_a_layout_whose_cubes_stay_on_the_table_fails(tmp_path, monkeypatch):
    result, _, _, _ = _run(
        tmp_path, monkeypatch, "others-painted", scene=StandInScene(places=False)
    )
    assert result["success"] is False
    assert result["cubes_in_bin_count"] == 0
    assert all(
        pick["success"] is False
        and pick["first_success_step"] is None
        and pick["in_bin_at_end"] is False
        for pick in result["picks"]
    )


def _row(
    seed: int,
    order: tuple[str, ...],
    placed: tuple[bool, ...],
    outline_source: str = "simulator outlines",
) -> dict:
    in_bin = dict(zip(order, placed, strict=True))
    return {
        "pictures": "others-painted",
        "outline_source": outline_source,
        "task": "Pick up a cube and place in the bin",
        "steps_per_pick": 400,
        "environment_seed": seed,
        "policy_seed": 100 + seed,
        "order_given_by": "seeded draw",
        "picks": [
            {
                "cube": color,
                "in_bin_at_end": inside,
                "first_success_step": 210 + index if inside else None,
            }
            for index, (color, inside) in enumerate(in_bin.items())
        ],
        "cubes_in_bin": {color: in_bin[color] for color in BIN_CUBE_COLORS},
        "cubes_in_bin_count": sum(placed),
        "success": all(placed),
    }


def test_the_report_counts_layouts_cubes_and_commands(tmp_path):
    rows = [
        _row(0, ("green", "red", "blue"), (True, True, True)),
        _row(1, ("blue", "green", "red"), (True, False, True)),
    ]
    text = write_three_cube_report(rows, tmp_path).read_text(encoding="utf-8")
    assert "Pictures given to SmolVLA: the two cubes the command does not name" in text
    assert 'Sentence given to SmolVLA: "Pick up a cube and place in the bin".' in text
    assert "Steps per command: 400." in text
    assert "All three cubes in the bin at the end: 1 of 2 layouts." in text
    assert "Cubes in the bin at the end: 5 of 6." in text
    assert "Commands whose cube was in the bin when the command ended: 5 of 6." in text
    assert "Command 1 of a layout: 2 of 2." in text
    assert "Command 2 of a layout: 1 of 2." in text
    assert "Command 3 of a layout: 2 of 2." in text
    assert "Commands for the red cube: 2 of 2." in text
    assert "Commands for the green cube: 1 of 2." in text
    assert "Commands for the blue cube: 2 of 2." in text
    assert "Accumulated video: [all feeds](overall-feeds.mp4), [top](overall-top.mp4)" in text
    assert "Outlines the picture editor paints by: simulator outlines." in text
    assert "the outlines the picture editor was given (simulator outlines)" in text
    with_sam3 = write_three_cube_report(
        [_row(0, ("green", "red", "blue"), (True, True, True), "SAM 3 outlines")],
        tmp_path,
    ).read_text(encoding="utf-8")
    assert "Outlines the picture editor paints by: SAM 3 outlines." in with_sam3
    assert "the outlines the picture editor was given (SAM 3 outlines)" in with_sam3
    assert (
        "| 0 | 100 | seeded draw | green in 210 | red in 211 | blue in 212 | "
        "red, green, blue | pass |"
    ) in text
    assert (
        "| 1 | 101 | seeded draw | blue in 210 | green out - | red in 212 | "
        "red, blue | fail |"
    ) in text


def test_a_batch_is_refused_before_anything_is_loaded_or_written(tmp_path):
    from so101_smolvla.three_cube_rollout import run_three_cube_batch

    assert DEFAULT_STEPS_PER_PICK == 400
    destination = tmp_path / "batch"
    with pytest.raises(ValueError, match="unknown picture mode"):
        run_three_cube_batch(1, destination=destination, pictures="blurred")
    with pytest.raises(ValueError, match="unknown outline source 'sam4'"):
        run_three_cube_batch(
            1, destination=destination, pictures="unedited", outlines="sam4"
        )
    with pytest.raises(ValueError, match="count must be positive"):
        run_three_cube_batch(0, destination=destination, pictures="unedited")
    with pytest.raises(ValueError, match="steps per command must be positive"):
        run_three_cube_batch(
            1, destination=destination, pictures="unedited", steps_per_pick=0
        )
    assert not destination.exists()
    destination.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_three_cube_batch(1, destination=destination, pictures="unedited")


class StandInVoice:
    """Gemma's fields for a few sentences, without a server; asked sentences are kept."""

    provenance = {"route": "stand-in"}
    FIELDS = {
        "put the green cube in the bin": ("pick_place", "green cube", "bin"),
        "grab the blue one": ("pick_place", "blue one", ""),
        "put the cube in the box": ("pick_place", "cube", "box"),
        "pick up the red block": ("pick_place", "red block", ""),
        "learn the stack": ("learn_stack", "", ""),
    }

    def __init__(self) -> None:
        self.asked: list[tuple[str, bool]] = []

    def ask(self, sentence: str, spoken: bool = False):
        from so101_smolvla.voice.commands import CommandFields

        self.asked.append((sentence, spoken))
        command, obj, place = self.FIELDS[sentence]
        return CommandFields(sentence, "synthetic voice" if spoken else "typed", command, obj, place,
                             sentence.capitalize(), 0.5, 2.0 if spoken else None)


def test_sentences_through_the_voice_name_the_cubes_and_a_refused_one_moves_nothing(
    tmp_path, monkeypatch
):
    voice = StandInVoice()
    sentences = [
        {"say": "put the green cube in the bin", "cube": "green"},
        {"say": "put the cube in the box", "cube": None},
        {"say": "grab the blue one", "cube": "blue"},
        {"say": "learn the stack", "cube": None},
        {"say": "pick up the red block", "cube": "red"},
    ]
    result, shown, scene, policy = _run(
        tmp_path, monkeypatch, "others-painted", sentences=sentences, voice=voice
    )
    # Every sentence was given to Gemma typed, right before its pick; the two
    # that name no cube were refused and the scene was not asked for a target.
    assert voice.asked == [(entry["say"], False) for entry in sentences]
    assert scene.targets == ["green", "blue", "red"]
    assert policy.resets == 3 and len(shown) == 9
    assert result["order"] == ["green", None, "blue", None, "red"]
    assert result["order_given_by"] == "typed sentences through Gemma 4"
    assert result["voice"] == {"route": "stand-in"}
    assert [entry["cube"] for entry in result["sentences"]] == ["green", None, "blue", None, "red"]
    assert [entry["as_expected"] for entry in result["sentences"]] == [True, None, True, None, True]
    assert result["sentences"][1]["refusal"] == "names no cube colour"
    assert result["sentences"][3]["refusal"] == "not a pick command but 'learn_stack'"
    assert result["sentences"][0]["gemma"] == {
        "sentence": "put the green cube in the bin", "how": "typed", "command": "pick_place",
        "object": "green cube", "place": "bin", "heard": "Put the green cube in the bin",
        "seconds": 0.5, "spoken_s": None,
    }
    assert [pick["command"] for pick in result["picks"]] == [
        "pick_place(object: green cube, place: bin)",
        "pick_place(object: cube, place: box)",
        "pick_place(object: blue one)",
        "learn_stack()",
        "pick_place(object: red block)",
    ]
    refused = result["picks"][1]
    assert refused["cube"] is None and refused["refusal"] == "names no cube colour"
    assert refused["in_bin_at_end"] is False and refused["steps"] == 0
    assert refused["sentence"] == "put the cube in the box" and refused["expected_cube"] is None
    moved = result["picks"][2]
    assert moved["cube"] == "blue" and moved["in_bin_at_end"] is True
    assert moved["sentence"] == "grab the blue one" and moved["expected_cube"] == "blue"
    assert moved["refusal"] is None
    assert result["success"] is True
    # The video's captions carry Gemma's call, two plans per moved cube.
    assert [plan["command"] for plan in result["plans"]] == [
        "pick_place(object: green cube, place: bin)",
        "pick_place(object: green cube, place: bin)",
        "pick_place(object: blue one)",
        "pick_place(object: blue one)",
        "pick_place(object: red block)",
        "pick_place(object: red block)",
    ]

    report = write_three_cube_report([result], tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "Commands from: typed sentences through Gemma 4." in text
    assert "Sentences through Gemma 4: 5; 3 of 3 with a known cube gave that cube; 2 refused (nothing moved)." in text
    assert "| green in 1 | refused | blue in 1 |" in text
    assert "| 4 | put the cube in the box | typed | Put the cube in the box | `pick_place(object: cube, place: box)` | refused: names no cube colour | - | 0.50 |" in text
    assert "| 4 | grab the blue one | typed | Grab the blue one | `pick_place(object: blue one)` | blue | blue | 0.50 |" in text


def test_spoken_sentences_are_said_to_the_voice_and_say_so_in_the_result(tmp_path, monkeypatch):
    voice = StandInVoice()
    result, _, scene, _ = _run(
        tmp_path, monkeypatch, "unedited",
        sentences=[{"say": "grab the blue one", "cube": "blue"}], voice=voice, spoken=True,
    )
    assert voice.asked == [("grab the blue one", True)]
    assert scene.targets == ["blue"]
    assert result["order_given_by"] == "sentences in a synthetic voice through Gemma 4"
    assert result["sentences"][0]["gemma"]["how"] == "synthetic voice"
    assert result["sentences"][0]["gemma"]["spoken_s"] == 2.0


def test_sentences_need_the_voice_and_exclude_an_order_and_all_refused_is_a_fault(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="colours or as sentences, not both"):
        _run(tmp_path, monkeypatch, "unedited", order=("red", "green", "blue"),
             sentences=[{"say": "grab the blue one", "cube": "blue"}], voice=StandInVoice())
    with pytest.raises(ValueError, match="need the voice"):
        _run(tmp_path, monkeypatch, "unedited", sentences=[{"say": "grab the blue one", "cube": "blue"}])
    with pytest.raises(RuntimeError, match="every command was refused"):
        _run(tmp_path, monkeypatch, "unedited",
             sentences=[{"say": "learn the stack", "cube": None}], voice=StandInVoice())
    assert not (tmp_path / "seed-00004").exists()


def test_the_batch_checks_sentences_against_its_count_and_order(tmp_path):
    from so101_smolvla.three_cube_rollout import run_three_cube_batch

    sentences = [[{"say": "grab the blue one", "cube": "blue"}]]
    with pytest.raises(ValueError, match="colours or as sentences, not both"):
        run_three_cube_batch(1, destination=tmp_path / "a", pictures="unedited",
                             order=("red", "green", "blue"), sentences=sentences)
    with pytest.raises(ValueError, match="1 layouts of sentences for a batch of 2"):
        run_three_cube_batch(2, destination=tmp_path / "b", pictures="unedited", sentences=sentences)
    with pytest.raises(ValueError, match="spoken commands need sentences"):
        run_three_cube_batch(1, destination=tmp_path / "c", pictures="unedited", spoken=True)
    assert not any((tmp_path / name).exists() for name in "abc")
