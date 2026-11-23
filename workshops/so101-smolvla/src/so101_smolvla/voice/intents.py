"""The commands Gemma can call, the instructions it is given, and what one answer holds.

A new command is three changes: its member in :class:`Intent`, one line in
:data:`INTENT_SYSTEM_PROMPT`, its name in the ``cmd`` rule of ``intent.gbnf``.
The grammar holds the shape of the answer for ``llama-server``; the JSON schema
of :func:`response_schema`, the same shape for FastFlowLM, follows
:class:`Intent` and needs no change. The instructions steer the choice. The
instructions and the grammar are the texts measured in SPEC.md and are not
edited lightly. The "stop" entry is the wording of 2026-11-23, chosen by
measurement on both backends (TESTING_LEDGER.md, "the stop entry"); the
sentences that hold it are ``examples.STOP_ORDERS`` and the lists beside it.

One answer is a JSON list of four strings: the command, the object, the place
and the words heard, in that order (:data:`FIELDS`). Until 2026-11-23 it was
an object with those four names as keys; the list was chosen by measurement
for its fewer tokens, on the user's word to shorten the answer. The sentence
of the instructions that shows the shape, the grammar and the schema changed
with it; every other word of the instructions is as it was.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum

# The shape of one answer, as intent.gbnf writes it, for the schema of the FastFlowLM request.
WORD = "[a-z]{1,16}"
OBJECT_PATTERN = f"^$|^{WORD}( {WORD}){{0,2}}$"  # empty, or one to three lowercase words
PLACE_PATTERN = f"^$|^{WORD}( {WORD}){{0,1}}$"  # empty, or one or two lowercase words
HEARD_PATTERN = r"^[a-zA-Z0-9',.?!\- ]*$"  # the words said; empty when they cannot be made out


FIELDS = ("command", "object", "place", "heard")  # one answer, in the order it is written


def response_schema() -> dict:
    """The JSON schema of one answer: the grammar's shape, with the commands of :class:`Intent`.

    A list of four strings, the fields of :data:`FIELDS` by position. A list and not an object
    with named keys: the call is then 12 tokens of Gemma's instead of 21 (measured on
    FastFlowLM, 2026-11-23: 1.25 s to the complete call instead of 1.94 s; shorter key names
    saved nothing, a key is one token whatever its length)."""
    return {
        "type": "array",
        "prefixItems": [
            {"type": "string", "enum": [intent.value for intent in Intent]},
            {"type": "string", "pattern": OBJECT_PATTERN},
            {"type": "string", "pattern": PLACE_PATTERN},
            {"type": "string", "pattern": HEARD_PATTERN},
        ],
        "items": False,
        "minItems": len(FIELDS),
        "maxItems": len(FIELDS),
    }


def response_format() -> dict:
    """The ``response_format`` field of an OpenAI-style request, holding the answer to the schema."""
    return {"type": "json_schema", "json_schema": {"name": "intent", "schema": response_schema()}}


def schema_text() -> str:
    """The schema as one canonical text, for its checksum in a result's provenance."""
    return json.dumps(response_schema(), sort_keys=True, separators=(",", ":"))


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
    """The answer in Gemma's JSON text, a list of the four fields, or None when the text is not one."""
    raw = (text or "").strip()
    if not raw.startswith("["):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not data or not all(isinstance(item, str) for item in data):
        return None
    command, obj, place, heard = (*data, "", "", "")[: len(FIELDS)]
    params = {key: value.strip() for key, value in (("object", obj), ("place", place)) if value.strip()}
    return ParsedCommand(Intent.from_string(command), params=params, transcript=heard.strip())


# An answer is written in the order command, object, place, heard (the grammar and the
# schema fix it), so the call is complete while the words heard are still being written.
# The values of the first three hold no quote and no backslash (WORD above).
_CALL_SO_FAR = re.compile(r'\s*\[\s*"([a-z_]*)"\s*,\s*"([a-z ]*)"\s*,\s*"([a-z ]*)"\s*,')
_HEARD_SO_FAR = re.compile(r'\s*"([^"\\]*)')  # what follows the call


def call_from_partial(text: str) -> ParsedCommand | None:
    """The call (command, object, place) once the answer's text so far holds all of it; None before.

    The comma after the place's closing quote is what completes it. The words heard are not
    in the result: they follow.
    """
    found = _CALL_SO_FAR.match(text or "")
    if found is None:
        return None
    command, obj, place = found.groups()
    params = {key: value.strip() for key, value in (("object", obj), ("place", place)) if value.strip()}
    return ParsedCommand(Intent.from_string(command), params=params)


def heard_from_partial(text: str) -> str:
    """The words heard as far as the answer's text has them: all of them in a finished answer."""
    call = _CALL_SO_FAR.match(text or "")
    found = _HEARD_SO_FAR.match(text, call.end()) if call else None
    return found.group(1).strip() if found else ""


INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm on a crowded floor. Other people
are talking. Ignore background conversation, a second speaker, questions,
quotes, negations, and anything not addressed to the robot: those are
"unknown". Map a clear instruction to exactly one command and answer ONLY
with a JSON list ["<name>", "<object or empty string>", "<place or empty string>", "<words>"]: the command, then "object", then "place", then "heard".
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
- "stop": the user orders the robot to stop moving ("stop", "stop robot",
  "robot stop", "stop the robot", "stop now"). This is the emergency
  command: when in doubt between "stop" and "unknown" for such an order,
  answer "stop". Talk about stopping that is not an order to the robot
  ("stop talking", "don't stop", "why did it stop") is "unknown".
- "unknown": anything else.
"""
