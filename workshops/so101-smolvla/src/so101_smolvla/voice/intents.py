"""The commands Gemma can call, the instructions it is given, and what one answer holds.

A new command is three changes: its member in :class:`Intent`, one line in
:data:`INTENT_SYSTEM_PROMPT`, its name in the ``cmd`` rule of ``intent.gbnf``.
The grammar holds the shape of the answer; the instructions steer the choice.
Both texts are those measured in SPEC.md and are not edited lightly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class Intent(str, Enum):
    """The commands, plus STOP and UNKNOWN."""

    GESTURE_MIMIC = "gesture_mimic"
    FETCH_BLOCK = "fetch_block"
    PICK_PLACE = "pick_place"
    DANCE = "dance"
    GRIP = "grip"
    RELEASE = "release"
    WAVE = "wave"
    HOME = "home"
    LEARN_STACK = "learn_stack"
    REBUILD_STACK = "rebuild_stack"
    STOP = "stop"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "Intent":
        """The member named by value, or UNKNOWN."""
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ParsedCommand:
    """One answer of Gemma: the command, its parameters (object, place) and the words it heard."""

    intent: Intent
    params: dict[str, str] = field(default_factory=dict)
    transcript: str = ""

    @property
    def object_name(self) -> str | None:
        return self.params.get("object")

    @property
    def place(self) -> str:
        return self.params.get("place", "")


def command_from_json(text: str) -> ParsedCommand | None:
    """The answer in Gemma's JSON text, or None when the text is not one."""
    raw = (text or "").strip()
    if not raw.startswith("{"):
        return None
    try:
        data = json.loads(raw)
        command = data["command"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    params: dict[str, str] = {}
    obj = str(data.get("object", "") or "").strip()
    if obj:
        params["object"] = obj
    place = str(data.get("place", "") or "").strip()
    if place:
        params["place"] = place
    heard = str(data.get("heard", "") or "").strip()
    return ParsedCommand(Intent.from_string(str(command)), params=params, transcript=heard)


INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm on a crowded floor. Other people
are talking. Ignore background conversation, a second speaker, questions,
quotes, negations, and anything not addressed to the robot: those are
"unknown". Map a clear instruction to exactly one command and answer ONLY
with a JSON object {"command": "<name>", "object": "<object or empty string>", "place": "<place or empty string>", "heard": "<words>"}.
"object" is the thing to act on for pick_place (one to three plain words,
e.g. "ball", "water bottle", "small green bottle"); for every other command
use "".
"place" is where the user wants the object put for pick_place (one or two
plain words, e.g. "box"); use "" when the user names no place and for every
other command.
"heard" is the words the person said to the robot, and nothing else.
Use "" for "heard" when you cannot make out the words.

Commands:
- "gesture_mimic": the user wants the arm to copy / mirror / follow their own
  arm and hand movements ("copy me", "mirror my hand", "follow my arm").
- "fetch_block": the user wants the block/cube from its usual spot brought to
  them ("get me the block", "bring the cube here").
- "pick_place": the user names an object to pick up / grab / hand over, or
  to put somewhere ("pick up the ball", "grab the bottle and give it to me",
  "put the apple in the box"). Put the object name in "object" and the
  place, if the user names one, in "place".
- "dance": the user wants the robot to dance / groove / party.
- "grip": the user is handing the robot something to hold ("hold this",
  "take this pencil", "grab what I give you").
- "release": the user wants the jaws opened, here, so they can take the
  object or drop it ("release", "open", "drop it", "let go",
  "open your hand").
- "wave": the user wants a wave / greeting ("wave at me", "say hi").
- "home": the speaker says "go home". "home", "rest", and "reset" alone
  are "unknown".
- "learn_stack": the user tells the robot to learn / memorise the stack of
  cubes that stands on the table now ("learn the stack", "memorise this
  stack").
- "rebuild_stack": the user tells the robot to build the stack it learned
  again from the scattered cubes ("rebuild the stack", "build the stack
  again").
- "stop": the speaker says "stop robot". "stop", "halt", "freeze",
  "stop it", "abort", and "stop talking" are "unknown".
- "unknown": anything else.
"""
