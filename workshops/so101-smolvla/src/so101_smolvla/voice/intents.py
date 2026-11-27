"""The commands Gemma can call, the instructions it is given, and what one answer holds.

One answer is a line of plain words, then a colon and the words heard:

    stop: stop robot                                -> stop
    learn: learn the stack                          -> learn_stack
    pick red cube box: put the red cube in the box  -> pick_place(object: red cube, place: box)
    none: who built this stack                      -> unknown

Gemma writes about 15 tokens a second on the NPU, and a command can only be
acted on when it is written. So the command's part is the fewest tokens that
say it, each word a single token of Gemma's tokenizer, and it comes first: a
command without parameters is settled by its one word, a pick command by the
colon. The words heard follow and cost the call nothing. This is the answer of
``~/GitHub/voice_intent`` (its ``commands.py``, and the ``HEARD`` rule of its
notebook), taken over on 2026-11-27 on the user's word; measured there against
the JSON list this workshop had since 2026-11-23: a pick command known 1.43 s
after the request began instead of 1.83 s.

The objects are a closed list: only what is in :data:`SIZES`, :data:`COLOURS`,
:data:`OBJECTS` and :data:`PLACES` can be written. The lists are
``voice_intent``'s, with ``cylinder`` added for the sentences of
``examples.CASES``.

This module is the one place for all of it. The answer is held to its shape by
:data:`PATTERN` in a JSON schema for FastFlowLM (:func:`response_format`) and
by the grammar of :func:`grammar_text` for ``llama-server``; both are built
from the lists here. A new command without parameters is two changes: its word
and member in :data:`SIMPLE` (and :class:`Intent`), and one line in
:data:`INTENT_SYSTEM_PROMPT`.

The instructions are ``voice_intent``'s: this workshop's text of 2026-11-23
(the "stop" entry chosen by measurement, TESTING_LEDGER.md) with the answer's
shape replaced and the rules for negations, questions and remarks added. Three
things differ from that project's text, chosen by measurement on the NPU with
this workshop's 37 test sentences (TESTING_LEDGER.md, 2026-11-27):

- The examples of a pick answer show cubes of two colours, and the entry
  says never to leave out a colour. With ``voice_intent``'s examples, both of
  them a red cube, six spoken sentences about the green cube were answered
  "red cube" while the words heard said "green"; with examples that name no
  cube, the colour was left out. The examples of the words heard name no
  cube: with "put the blue cube in the bin" among them, ``llama-server``
  wrote that very sentence for three other spoken ones. A colour in an
  example still leaks: on ``llama-server`` two of the 37 spoken sentences
  are answered "blue cube" (one run; the known weak point of this wording).
- An object may carry two colours ("pick red green cube box"), so that an
  order that names two cubes can be written down and is then refused by
  ``cube_commands.choose_cube``; with one colour at most it became a pick of
  the first cube.
- ``cylinder`` among the objects (above).
"""

from __future__ import annotations

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


# What Gemma writes -> the command. One word, one token, and the beginning of no other answer.
SIMPLE = {
    "copy": Intent.GESTURE_MIMIC,
    "fetch": Intent.FETCH_BLOCK,
    "dance": Intent.DANCE,
    "grip": Intent.GRIP,
    "release": Intent.RELEASE,
    "wave": Intent.WAVE,
    "home": Intent.HOME,
    "learn": Intent.LEARN_STACK,
    "build": Intent.REBUILD_STACK,
    "stop": Intent.STOP,
}
PICK = "pick"  # then the object's words, then the place if one was named
NONE = "none"  # no command
SIZES = ("small", "large")
COLOURS = ("red", "green", "blue", "yellow")
OBJECTS = ("cube", "bottle", "pen", "ball", "cylinder")
PLACES = ("box", "bin")
HEARD_CHARACTERS = "a-zA-Z0-9',.?! -"  # the words heard: no colon, no quote

MOST_COLOURS = 2  # an object names one colour; two say that the user named two objects

# The command's words, then a colon and the words heard (empty when they cannot be made out).
WORDS_PATTERN = "(%s|%s|%s( %s)?%s (%s)( %s)?)" % (
    "|".join(SIMPLE), NONE, PICK, "| ".join(SIZES), ("( %s)?" % "| ".join(COLOURS)) * MOST_COLOURS,
    "|".join(OBJECTS), "| ".join(PLACES),
)
PATTERN = f"^{WORDS_PATTERN}: [{HEARD_CHARACTERS}]*$"


def response_format() -> dict:
    """The ``response_format`` field of an OpenAI-style request: FastFlowLM holds the answer to
    one JSON string of the allowed words (1.0.10 or later)."""
    return {"type": "json_schema", "json_schema": {"name": "command", "schema": {"type": "string", "pattern": PATTERN}}}


def grammar_text() -> str:
    """The same shape as a grammar for ``llama-server``, built from the same lists: a JSON
    string, as FastFlowLM writes it, so one reader reads both."""

    def one_of(words) -> str:
        return " | ".join(f'"{word}"' for word in words)

    colours = ' (" " colour)?' * MOST_COLOURS
    return "\n".join((
        "# Built by so101_smolvla.voice.intents.grammar_text from the lists of that module.",
        'root   ::= "\\"" words ": " said "\\""',
        f'words  ::= simple | "{NONE}" | pick',
        f"simple ::= {one_of(SIMPLE)}",
        f'pick   ::= "{PICK}" (" " size)?{colours} " " thing (" " place)?',
        f"size   ::= {one_of(SIZES)}",
        f"colour ::= {one_of(COLOURS)}",
        f"thing  ::= {one_of(OBJECTS)}",
        f"place  ::= {one_of(PLACES)}",
        "# What was said; empty when the words cannot be made out.",
        "said   ::= hchar*",
        'hchar  ::= [a-zA-Z0-9] | "\'" | "," | "." | "?" | "!" | "-" | " "',
        "",
    ))


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

    @property
    def call(self) -> tuple[str, str, str]:
        """(command, object, place): what the listening session's filter compares."""
        return self.intent.value, self.params.get("object", ""), self.place


def command_from_words(words: str) -> ParsedCommand | None:
    """The call of a complete answer's command words; None when they are no allowed answer."""
    parts = words.split()
    if len(parts) == 1 and parts[0] in SIMPLE:
        return ParsedCommand(SIMPLE[parts[0]])
    if parts == [NONE]:
        return ParsedCommand(Intent.UNKNOWN)
    if len(parts) < 2 or parts[0] != PICK:
        return None
    place = parts.pop() if parts[-1] in PLACES else ""
    thing = parts[1:]
    if not thing or thing[-1] not in OBJECTS:
        return None
    described = thing[:-1]
    if described and described[0] in SIZES:
        described = described[1:]
    if len(described) > MOST_COLOURS or any(word not in COLOURS for word in described):
        return None
    params = {"object": " ".join(thing)}
    if place:
        params["place"] = place
    return ParsedCommand(Intent.PICK_PLACE, params)


def _opened(text: str) -> str | None:
    """The answer's text after its opening quote; None before the quote is written."""
    text = (text or "").lstrip()
    return text[1:] if text.startswith('"') else None


def call_from_partial(text: str) -> ParsedCommand | None:
    """The call (command, object, place) once the answer's text so far settles it; None before.

    A command without parameters, and "none", is settled by its word, which is one token and
    the beginning of no other answer. A pick command is settled by the colon (or, in an answer
    without words heard, by the closing quote): until then a place can still follow. Words
    that are complete and no allowed answer are ``unknown``: no command is the stricter
    reading. The words heard are not in the result: they follow.
    """
    body = _opened(text)
    if body is None:
        return None
    words, colon, _ = body.partition(":")
    if colon or words.endswith('"'):
        return command_from_words(words.rstrip('"')) or ParsedCommand(Intent.UNKNOWN)
    word = words.strip()
    return command_from_words(word) if word in SIMPLE or word == NONE else None


def heard_from_partial(text: str) -> str:
    """The words heard as far as the answer's text has them: all of them in a finished answer."""
    body = _opened(text)
    if body is None:
        return ""
    _, _, heard = body.partition(":")
    return heard.rstrip().removesuffix('"').strip()


INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm on a crowded floor. Other people
are talking. Ignore background conversation, a second speaker, questions,
quotes, negations, and anything not addressed to the robot: those are
"none". Map a clear instruction to exactly one command and answer ONLY
with the command's words in lower case, nothing else.
First decide whether the speaker orders the robot to do something now.
A sentence that says NOT to do something ("do not rebuild the stack",
"don't pick up the cube", "never mind") is "none". A question ("who built
this stack", "is the ball in the box", "can it dance") is "none". A remark
about things ("the stack looks nice", "the cube is red") is "none".

Commands:
- "copy": the user wants the arm to copy / mirror / follow their own
  arm and hand movements ("copy me", "mirror my hand", "follow my arm").
- "fetch": the user wants the block/cube from its usual spot brought to
  them ("get me the block", "bring the cube here", "fetch the cube").
- "pick": the user names an object to pick up / grab / hand over, or
  to put somewhere ("pick up the ball", "grab the bottle and give it to me",
  "put the pen in the box"). Answer "pick", then the object as the user
  names it: its size if said (small, large), its colour if said (red, green,
  blue, yellow; both colours when the user names two), and what it is (cube, bottle, pen, ball, cylinder); then the
  place if the user names one (box, bin). Never leave out a colour the user says.
  Examples: "pick ball", "pick blue cube", "pick green cube box", "pick small yellow
  bottle bin".
- "dance": the user wants the robot to dance / groove / party / show its
  dance moves.
- "grip": the user is handing the robot something to hold ("hold this",
  "take this pencil", "grab what I give you").
- "release": the user wants the jaws opened, here, so they can take the
  object or drop it ("release", "open", "drop it", "let go",
  "open your hand").
- "wave": the user wants a wave / greeting ("wave at me", "say hi").
- "home": the speaker says "go home". "home", "rest", and "reset" alone
  are "none".
- "learn": the user orders the robot to learn / memorise the stack of
  cubes that stands on the table now ("learn the stack", "memorise this
  stack").
- "build": the user orders the robot to build the stack it learned
  again from the scattered cubes ("rebuild the stack", "build the stack
  again"). Talk about a stack that is no such order is "none".
- "stop": the user orders the robot to stop moving ("stop", "stop robot",
  "robot stop", "stop the robot", "stop now"). This is the emergency
  command: when in doubt between "stop" and "none" for such an order,
  answer "stop". Talk about stopping that is not an order to the robot
  ("stop talking", "don't stop", "why did it stop") is "none".
- "none": anything else.

After the command's words write a colon and then the words the person said, as you heard them, for example
"pick small yellow bottle bin: put the small yellow bottle in the bin", "pick large red ball: grab the large red ball",
"wave: wave at me", "none: who built this stack".
"""
