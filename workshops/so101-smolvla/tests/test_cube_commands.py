"""cube_commands: which cube Gemma's fields name, and the sentences of the Gemma test."""

from __future__ import annotations

import json
import re

import pytest

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.cube_commands import (
    REFUSED_SENTENCES,
    TEMPLATES,
    CubeChoice,
    call_text,
    choose_cube,
    layout_sentences,
    pick_sentences,
    read_sentences,
)
from so101_smolvla.three_cube_rollout import draw_order, pick_place_command


def test_a_pick_of_one_colour_is_that_cube_whatever_else_the_object_says():
    for colour in BIN_CUBE_COLORS:
        for words in (
            f"{colour} cube", colour, f"the {colour} one", f"small {colour} block",
            f"{colour.upper()} Cube", f"big {colour} brick",
        ):
            assert choose_cube("pick_place", words) == CubeChoice(colour), words


def test_a_command_that_names_no_one_cube_is_refused_with_its_reason():
    assert choose_cube("pick_place", "cube") == CubeChoice(None, "names no cube colour")
    assert choose_cube("pick_place", "") == CubeChoice(None, "names no cube colour")
    assert choose_cube("pick_place", "red and green cubes") == CubeChoice(
        None, "names two colours, red and green"
    )
    assert choose_cube("pick_place", "red ball") == CubeChoice(None, "names a ball, not a cube")
    assert choose_cube("pick_place", "yellow cube") == CubeChoice(
        None, "names a yellow cube; the cubes are red, green and blue"
    )
    assert choose_cube("unknown", "") == CubeChoice(None, "not a pick command but 'unknown'")
    assert choose_cube("learn_stack", "") == CubeChoice(None, "not a pick command but 'learn_stack'")
    assert choose_cube("fetch_block", "red cube") == CubeChoice(
        None, "not a pick command but 'fetch_block'"
    )


def test_a_choice_is_a_cube_or_a_refusal_and_the_cube_is_one_of_the_scene():
    with pytest.raises(ValueError, match="either a cube or a refusal"):
        CubeChoice(None, None)
    with pytest.raises(ValueError, match="either a cube or a refusal"):
        CubeChoice("red", "no")
    with pytest.raises(ValueError, match="unknown cube"):
        CubeChoice("yellow")


def test_a_call_is_written_the_way_the_voice_notebook_writes_one():
    assert call_text("pick_place", "red cube", "box") == "pick_place(object: red cube, place: box)"
    assert call_text("pick_place", "red cube") == "pick_place(object: red cube)"
    assert call_text("pick_place", "", "box") == "pick_place(place: box)"
    assert call_text("unknown") == "unknown()"
    assert pick_place_command("green") == call_text("pick_place", "green cube", "box")


def test_the_sentences_cover_ten_phrasings_of_every_cube_and_name_no_other_cube():
    sentences = pick_sentences()
    assert len(TEMPLATES) == 10 and len(sentences) == 30
    assert len({sentence for sentence, _ in sentences}) == 30
    for colour in BIN_CUBE_COLORS:
        assert sum(cube == colour for _, cube in sentences) == 10
    for sentence, cube in sentences:
        words = set(re.findall(r"[a-z]+", sentence.lower()))
        assert words & set(BIN_CUBE_COLORS) == {cube}, sentence
        # No stacking: nothing is put on another cube.
        assert " on " not in f" {sentence} ", sentence
    for template in TEMPLATES:
        assert template.count("{colour}") == 1
    # The refused sentences are each one thing, and none is a pick of one cube.
    assert len({sentence for sentence, _ in REFUSED_SENTENCES}) == len(REFUSED_SENTENCES)
    assert len({reason for _, reason in REFUSED_SENTENCES}) == len(REFUSED_SENTENCES)


def test_a_batchs_built_in_sentences_follow_each_layouts_drawn_order_and_rotate_the_templates():
    layouts = layout_sentences(10, environment_seed_base=5)
    assert len(layouts) == 10
    used = []
    for index, layout in enumerate(layouts):
        assert [entry["cube"] for entry in layout] == list(draw_order(5 + index))
        for position, entry in enumerate(layout):
            template = TEMPLATES[(index + position) % 10]
            assert entry["say"] == template.format(colour=entry["cube"])
            used.append(template)
    assert all(used.count(template) == 3 for template in TEMPLATES)
    assert layout_sentences(1) == [
        [
            {"say": TEMPLATES[position].format(colour=cube), "cube": cube}
            for position, cube in enumerate(draw_order(0))
        ]
    ]


def test_a_users_sentence_file_is_a_list_of_layouts_of_sentences(tmp_path):
    path = tmp_path / "sentences.json"
    path.write_text(json.dumps([
        ["put the red cube in the bin", {"say": " grab the blue one ", "cube": "blue"}],
        [{"say": "pick up the green cube"}],
    ]))
    assert read_sentences(path) == [
        [
            {"say": "put the red cube in the bin", "cube": None},
            {"say": "grab the blue one", "cube": "blue"},
        ],
        [{"say": "pick up the green cube", "cube": None}],
    ]
    for bad in ([], [[]], {"say": "x"}, [["ok"], "not a list"]):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="expected a list of layouts"):
            read_sentences(path)
    for bad in ([[""]], [[{"cube": "red"}]], [[{"say": "x", "cube": "yellow"}]], [[3]]):
        path.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="a sentence is a string or"):
            read_sentences(path)
