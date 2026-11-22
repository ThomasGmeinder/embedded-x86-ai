"""Which cube a command names: Gemma's fields in, a cube or a refusal out.

The three-cube test (three_cube_rollout) can take its commands as sentences a
user types or says. Gemma 4 turns a sentence into the voice client's fields
``command``, ``object``, ``place`` and ``heard`` (so101_smolvla.voice.commands). This module
is the one place that reads those fields for the three-cube scene: the cube is
the one colour of the scene's three that the object names. It decides from
``command`` and ``object`` only. The place is recorded with the command and
not judged: this test ends every pick in the bin (SPEC.md section 6), and a
sentence that names another cube as the place is the stacking case, tested on
its own (decision 9).

The module also holds the sentences of the automated test: ten phrasings of
the same request, each for the three colours.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS

PICK = "pick_place"
# Words of an object that say nothing against a cube of the scene.
KIND_WORDS = frozenset({"cube", "cubes", "block", "blocks", "one", "brick", "piece"})
SIZE_WORDS = frozenset({"small", "little", "tiny", "big", "large"})
FILLER_WORDS = frozenset({"the", "a", "an", "this", "that"})
# Colours the scene has no cube of.
OTHER_COLOURS = frozenset(
    {"yellow", "orange", "purple", "white", "black", "pink", "brown", "grey", "gray"}
)

# Ten ways to ask for one cube, with the colour left open: the verbs pick up,
# put, grab, place, move, take; the place as box, bin, container or not named;
# and a filler word or two. None is among the examples in the instructions to
# the model (so101_smolvla/voice/intents.py) or in the voice notebook's
# table (so101_smolvla/voice/examples.py).
TEMPLATES = (
    "pick up the {colour} cube and put it in the box",
    "put the {colour} cube in the bin",
    "grab the {colour} one and drop it in the box",
    "robot, please place the {colour} cube into the container",
    "can you move the {colour} cube to the bin",
    "take the {colour} block and put it in the box",
    "pick up the {colour} cube",
    "hey robot, put the {colour} cube in the box please",
    "I'd like you to put the {colour} cube in the bin",
    "next, the {colour} cube goes into the box",
)
# Sentences that must move no cube, with what they are to the chooser.
REFUSED_SENTENCES = (
    ("put the cube in the box", "no colour"),
    ("put the red and the green cube in the box", "two colours"),
    ("put the red ball in the box", "not a cube"),
    ("put the yellow cube in the box", "no such cube"),
    ("is the red cube in the box", "a question"),
    ("do not pick up the red cube", "a negation"),
    ("learn the stack", "another command"),
)


@dataclass(frozen=True)
class CubeChoice:
    """The cube a command names, or why nothing moves."""

    cube: str | None
    refusal: str | None = None

    def __post_init__(self) -> None:
        if (self.cube is None) == (self.refusal is None):
            raise ValueError("a choice is either a cube or a refusal")
        if self.cube is not None and self.cube not in BIN_CUBE_COLORS:
            raise ValueError(f"unknown cube {self.cube!r}")


def call_text(command: str, object_words: str = "", place: str = "") -> str:
    """A command with its fields, written the way the voice notebook writes a call."""
    arguments = [
        f"{name}: {value}"
        for name, value in (("object", object_words), ("place", place))
        if value
    ]
    return f"{command}({', '.join(arguments)})"


def choose_cube(command: str, object_words: str) -> CubeChoice:
    """The one cube of the scene that a pick command's object names."""
    if command != PICK:
        return CubeChoice(None, f"not a pick command but {command!r}")
    words = re.findall(r"[a-z]+", object_words.lower())
    colours = []
    for word in words:
        if word in BIN_CUBE_COLORS and word not in colours:
            colours.append(word)
    if len(colours) > 1:
        return CubeChoice(None, f"names two colours, {' and '.join(colours)}")
    others = [
        word
        for word in words
        if word not in BIN_CUBE_COLORS
        and word not in KIND_WORDS | SIZE_WORDS | FILLER_WORDS
    ]
    other_colours = [word for word in others if word in OTHER_COLOURS]
    if other_colours:
        return CubeChoice(
            None,
            f"names a {other_colours[0]} cube; the cubes are "
            f"{', '.join(BIN_CUBE_COLORS[:-1])} and {BIN_CUBE_COLORS[-1]}",
        )
    if others:
        return CubeChoice(None, f"names a {' '.join(others)}, not a cube")
    if not colours:
        return CubeChoice(None, "names no cube colour")
    return CubeChoice(colours[0])


def pick_sentences() -> tuple[tuple[str, str], ...]:
    """Every template for every colour: (sentence, the cube it names)."""
    return tuple(
        (template.format(colour=colour), colour)
        for template in TEMPLATES
        for colour in BIN_CUBE_COLORS
    )


def layout_sentences(count: int, environment_seed_base: int = 0) -> list[list[dict]]:
    """The built-in sentences of a batch: per layout three, in the layout's drawn order.

    Layout ``i`` says its three cubes with templates ``i``, ``i + 1`` and
    ``i + 2`` (around the end of the list), so ten layouts use every template
    three times. Each entry is ``{"say": sentence, "cube": colour}``.
    """
    from so101_smolvla.three_cube_rollout import draw_order

    layouts = []
    for index in range(count):
        order = draw_order(environment_seed_base + index)
        layouts.append(
            [
                {
                    "say": TEMPLATES[(index + position) % len(TEMPLATES)].format(
                        colour=colour
                    ),
                    "cube": colour,
                }
                for position, colour in enumerate(order)
            ]
        )
    return layouts


def read_sentences(path: Path) -> list[list[dict]]:
    """A user's sentences: a JSON list of layouts, each a list of sentences.

    A sentence is a string, or ``{"say": sentence, "cube": colour}`` where
    the cube it should name is known; a bare string gets ``"cube": None``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data or not all(
        isinstance(layout, list) and layout for layout in data
    ):
        raise ValueError(f"{path}: expected a list of layouts, each a list of sentences")
    layouts = []
    for layout in data:
        entries = []
        for entry in layout:
            if isinstance(entry, str):
                entry = {"say": entry, "cube": None}
            if (
                not isinstance(entry, dict)
                or not isinstance(entry.get("say"), str)
                or not entry["say"].strip()
                or entry.get("cube") not in (None, *BIN_CUBE_COLORS)
            ):
                raise ValueError(
                    f"{path}: a sentence is a string or {{'say': sentence, 'cube': "
                    f"one of {', '.join(BIN_CUBE_COLORS)} or null}}, not {entry!r}"
                )
            entries.append({"say": entry["say"].strip(), "cube": entry.get("cube")})
        layouts.append(entries)
    return layouts
