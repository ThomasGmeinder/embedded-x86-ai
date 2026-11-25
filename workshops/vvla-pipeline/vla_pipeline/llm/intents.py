# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Command intents shared by the LLM parser, orchestrator, and behaviors.

This revision makes the command set **extensible**: a command is an
``Intent`` plus a parameter dict (``ParsedCommand``), and behaviors register
themselves on a :class:`CommandRegistry` instead of being hardcoded in the
orchestrator's dispatch. Adding a new voice command =

1. add the enum member + one line in :data:`INTENT_SYSTEM_PROMPT` (Gemma) and
   one in :data:`COMMAND_ANSWERS` (the decision model),
2. add its name to ``config/intent.gbnf``'s ``cmd`` rule,
3. decorate a handler with ``@registry.register(Intent.MY_COMMAND)``.

``STOP`` is special: the orchestrator matches it with plain keywords on the
raw transcript (no LLM round trip) so the safety stop is as fast as possible,
and it interrupts any running behavior via the shared stop event.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class Intent(str, Enum):
    """Recognized voice-command intents, plus STOP and UNKNOWN sentinels."""

    GESTURE_MIMIC = "gesture_mimic"
    FETCH_BLOCK = "fetch_block"
    PICK_PLACE = "pick_place"
    DANCE = "dance"
    GRIP = "grip"
    RELEASE = "release"
    WAVE = "wave"
    HOME = "home"
    STOP = "stop"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "Intent":
        """Parse value into an Intent, or UNKNOWN if it doesn't match a member."""
        try:
            return cls(value.strip().lower())
        except ValueError:
            return cls.UNKNOWN


@dataclass
class ParsedCommand:
    """An intent plus its parameters (e.g. the object name for pick_place)."""

    intent: Intent
    params: dict[str, str] = field(default_factory=dict)
    transcript: str = ""

    @property
    def object_name(self) -> str | None:
        """Object name parameter, if any (used by pick_place)."""
        return self.params.get("object")


# The only spoken stop. A bare "stop" is crowd noise on a tradeshow floor.
# Matched as a phrase: a check for the word "stop" would also fire on
# "stop talking".
STOP_KEYWORDS = ("stop robot",)


# Gemma's answer is a JSON list of three strings, in this order (config/intent.gbnf fixes it).
# A list and not an object with named keys: fewer tokens before the call is complete (measured
# on the so101 workshop, 2026-11-23: 12 tokens to the call instead of 21). The words heard come
# last, so the call is complete, and acted on, while they are still being written.
FIELDS = ("command", "object", "heard")

# The values of the first two hold no quote and no backslash (the grammar's word rule). The
# comma after the object's closing quote is what completes the call.
_CALL_SO_FAR = re.compile(r'\s*\[\s*"([a-z_]*)"\s*,\s*"([a-z ]*)"\s*,')
_HEARD_SO_FAR = re.compile(r'\s*"([^"\\]*)')  # what follows the call


def call_from_partial(text: str) -> ParsedCommand | None:
    """The call (command, object) once the answer's text so far holds all of it; None before.

    The words heard are not in the result: they follow.
    """
    found = _CALL_SO_FAR.match(text or "")
    if found is None:
        return None
    command, obj = found.groups()
    params = {"object": obj.strip()} if obj.strip() else {}
    return ParsedCommand(Intent.from_string(command), params=params)


def heard_from_partial(text: str) -> str:
    """The words heard as far as the answer's text has them: all of them in a finished answer."""
    call = _CALL_SO_FAR.match(text or "")
    found = _HEARD_SO_FAR.match(text, call.end()) if call else None
    return found.group(1).strip() if found else ""


def command_from_json(text: str) -> ParsedCommand | None:
    """A command the speech front end already resolved, handed on as a JSON object, or None."""
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
    heard = str(data.get("heard", "") or "").strip()
    return ParsedCommand(Intent.from_string(str(command)), params=params, transcript=heard)


_KEYWORD_ROUTES = [
    (r"\bstop robot\b", "stop", None),
    (r"\b(release|open|drop)\b|\blet go\b", "release", None),
    (r"\b(copy|mirror|follow|mimic)\b", "gesture_mimic", None),
    (r"\b(block|cube)\b", "fetch_block", None),
    (r"\b(pick|grab|bring|fetch|get)\b", "pick_place", "OBJECT"),
    (r"\b(dance|groove|party|moves)\b", "dance", None),
    (r"\b(hold|take)\b", "grip", None),
    (r"\b(wave|hi|hello|greet)\b", "wave", None),
    (r"\bgo home\b", "home", None),
]
_OBJECT_WORDS = (
    "ball", "bottle", "cup", "mug", "phone", "apple", "banana", "book",
    "toothbrush", "remote", "orange", "teddy",
)


def keyword_parse(text: str) -> ParsedCommand:
    """Keyword stand-in used when the language model is not available."""
    low = (text or "").lower()
    for pattern, command, obj_mode in _KEYWORD_ROUTES:
        if re.search(pattern, low):
            params = {}
            if obj_mode == "OBJECT":
                obj = next((word for word in _OBJECT_WORDS if word in low), "")
                if obj:
                    params["object"] = obj
            return ParsedCommand(Intent.from_string(command), params=params, transcript=text)
    return ParsedCommand(Intent.UNKNOWN, transcript=text)


class CommandRegistry:
    """Maps intents to behavior entry points; the orchestrator just dispatches.

    Handlers receive ``(ctx, cmd: ParsedCommand)`` where ``ctx`` is the
    orchestrator's :class:`~vla_pipeline.main.PipelineContext` (arm, cameras,
    models, shared state, stop_check).
    """

    def __init__(self) -> None:
        self._handlers: dict[Intent, Callable] = {}

    def register(self, intent: Intent):
        """Return a decorator that registers fn as the handler for intent."""

        def deco(fn: Callable) -> Callable:
            """Register fn as the handler for intent, raising if one is already registered."""
            if intent in self._handlers:
                raise ValueError(f"handler for {intent} already registered")
            self._handlers[intent] = fn
            return fn

        return deco

    def get(self, intent: Intent) -> Callable | None:
        """Return the handler registered for intent, or None."""
        return self._handlers.get(intent)

    def intents(self) -> list[Intent]:
        """Return the list of intents that have a registered handler."""
        return list(self._handlers)


# Few-shot system prompt for the intent LLM. Kept here so the parser and any
# eval scripts share one source of truth. The grammar (config/intent.gbnf)
# guarantees the output shape; this prompt steers the choice.
INTENT_SYSTEM_PROMPT = """\
You are the command router for a robot arm on a crowded floor. Other people
are talking. Ignore background conversation, a second speaker, questions,
quotes, negations, and anything not addressed to the robot: those are
"unknown". Map a clear instruction to exactly one command and answer ONLY
with a JSON list ["<name>", "<object or empty string>", "<words>"]: the command, then "object", then "heard".
"object" is the thing to act on for pick_place (one or two plain words,
e.g. "ball", "water bottle"); for every other command use "".
"heard" is the words the person said to the robot, and nothing else.
Use "" for "heard" when you cannot make out the words.

Commands:
- "gesture_mimic": the user wants the arm to copy / mirror / follow their own
  arm and hand movements ("copy me", "mirror my hand", "follow my arm").
- "fetch_block": the user wants the block/cube from its usual spot brought to
  them ("get me the block", "bring the cube here").
- "pick_place": the user names an object to pick up / grab / hand over
  ("pick up the ball", "grab the bottle and give it to me"). Put the object
  name in "object".
- "dance": the user wants the robot to dance / groove / party.
- "grip": the user is handing the robot something to hold ("hold this",
  "take this pencil", "grab what I give you").
- "release": the user wants the jaws opened, here, so they can take the
  object or drop it ("release", "open", "drop it", "let go",
  "open your hand").
- "wave": the user wants a wave / greeting ("wave at me", "say hi").
- "home": the speaker says "go home". "home", "rest", and "reset" alone
  are "unknown".
- "stop": the speaker says "stop robot". "stop", "halt", "freeze",
  "stop it", "abort", and "stop talking" are "unknown".
- "unknown": anything else.
"""


# (sentence, the command it must give, the object it must name or None): the typed sentences
# each parser's --selftest asks its real model.
SELFTEST_CASES = [
    ("can you copy my arm movements", Intent.GESTURE_MIMIC, None),
    ("mirror my hand please", Intent.GESTURE_MIMIC, None),
    ("go grab the block and bring it to me", Intent.FETCH_BLOCK, None),
    ("fetch the cube", Intent.FETCH_BLOCK, None),
    ("pick up the ball", Intent.PICK_PLACE, "ball"),
    ("grab the bottle and hand it to me", Intent.PICK_PLACE, "bottle"),
    ("show me your best dance moves", Intent.DANCE, None),
    ("hold this pencil for me", Intent.GRIP, None),
    ("release", Intent.RELEASE, None),
    ("open your hand", Intent.RELEASE, None),
    ("drop it", Intent.RELEASE, None),
    ("wave hello", Intent.WAVE, None),
    ("go home", Intent.HOME, None),
    ("stop robot", Intent.STOP, None),
    ("what's the weather like today", Intent.UNKNOWN, None),
]


# ---- the questions for a decision model (d1-Omni-2B) ------------------------------------------------
# A decision model writes no text: it answers named questions about a sentence or a clip by
# picking one of the answers offered. So the instructions above become two pick-one questions.
# The model's own answers are the authority for a command; these texts steer the choice, as
# INTENT_SYSTEM_PROMPT does for Gemma.

COMMAND_QUESTION = "What does the speaker ask the robot arm to do?"
# The answers the pick-one question offers: (label, the command it stands for, the phrases
# offered with it). A command's label is its name. "No command" has four answers of its own,
# each a kind of sentence that must do nothing; every one of them stands for UNKNOWN.
#
# Chosen by measurement on typed sentences and spoken clips of a synthetic voice (2026-11-24,
# three rounds; with the pick rule of chosen_command). Of 214 sentences and clips the answers
# were chosen on, 187 are right (169 with one "unknown" answer; 108 of the 165 clips with a
# one-line description of each command). Of 204 held-out ones, in no answer's phrases, 163.
# - Phrases, not descriptions: the model matches what it hears against them.
# - "copy" does not lead gesture_mimic: the model hears a spoken "pick up" close to "copy",
#   and took "pick up the ball" for gesture_mimic while "copy me" led that answer.
# - pick_place has several "pick up ..." phrases and is offered first; grip names no object
#   ("take this pencil" there made a typed "pick up the pen" a grip).
# - stop and home say "both words": a bare "stop", "halt" or "abort" then goes to one_word.
#   (A bare "home" is still taken as home; no wording tried moved it without costing more.)
# - With one answer for "no command", a bare "stop", "stop talking" and "do not pick up the
#   ball" went to the nearest command. An answer for each kind gives them one to match.
COMMAND_ANSWERS = (
    ("pick_place", Intent.PICK_PLACE,
     "pick up the ball, pick up the bottle, pick up the cup, pick up the pen, pick up the toothbrush, "
     "grab the book, hand me the phone"),
    ("gesture_mimic", Intent.GESTURE_MIMIC, "mirror my arm, follow my hand, imitate my movements, copy my movements"),
    ("fetch_block", Intent.FETCH_BLOCK, "get me the block, bring the cube here, fetch the cube"),
    ("dance", Intent.DANCE, "dance, show me your dance moves"),
    ("grip", Intent.GRIP, "hold this, hold this for me, take this from me"),
    ("release", Intent.RELEASE, "release, open your hand, drop it, let go"),
    ("wave", Intent.WAVE, "wave at me, wave hello, say hi"),
    ("home", Intent.HOME, "go home: both words"),
    ("stop", Intent.STOP, "stop robot, robot stop: both words"),
    ("one_word", Intent.UNKNOWN, "one word alone, no order: stop, halt, freeze, abort, home, rest, reset"),
    ("about_stopping", Intent.UNKNOWN,
     "talk about stopping, no order to the robot: stop it, stop talking, why did it stop"),
    ("negation", Intent.UNKNOWN, "told not to do something: do not pick up the ball, don't dance, do not stop"),
    ("unknown", Intent.UNKNOWN, "any other talk or a question: the weather is nice, how does the robot work"),
)
OBJECT_QUESTION = "Which object does the person ask the robot to pick up, grab or hand over?"


def command_question() -> dict:
    """The question a decision model answers with one of :data:`COMMAND_ANSWERS`, and a probability for each."""
    return {
        "type": "choice",
        "instructions": COMMAND_QUESTION,
        "criteria": {label: phrases for label, _, phrases in COMMAND_ANSWERS},
    }


def chosen_command(
    command: dict, *, min_pick_probability: float | None = None, min_confidence: float = 0.0,
) -> Intent:
    """The command a decision model's answer stands for.

    The answer is ``{"choice": label, "probabilities": {label: 0..1}, "confidence": 0..1}``.
    The probabilities of the answers that stand for one command are added up (the four "no
    command" answers are one UNKNOWN), and the most probable command is taken.
    ``min_pick_probability`` puts a pick first: ``pick_place`` is taken when its probability
    reaches it, and otherwise the most probable of the other commands. A spoken "pick up the
    ball" is often the model's second choice, behind gesture_mimic or dance, with 0.10 to
    0.28; most sentences that are no pick stay below. Calibrated on the clips named at
    :data:`COMMAND_ANSWERS`; it costs some sentences that are no pick ("go grab the block and
    bring it to me" is a pick in three clips of five). A command the model is less confident
    of than ``min_confidence`` is ``unknown``. An answer without probabilities is taken by its
    choice.
    """
    stands_for = {label: intent for label, intent, _ in COMMAND_ANSWERS}
    probabilities = command.get("probabilities") or {}
    if not probabilities:
        if float(command.get("confidence", 0.0)) < min_confidence:
            return Intent.UNKNOWN
        return stands_for.get(str(command.get("choice", "")), Intent.UNKNOWN)
    summed: dict[Intent, float] = {}
    for label, probability in probabilities.items():
        intent = stands_for.get(label, Intent.UNKNOWN)
        summed[intent] = summed.get(intent, 0.0) + float(probability)
    if min_pick_probability is not None:
        if summed.get(Intent.PICK_PLACE, 0.0) >= min_pick_probability:
            return Intent.PICK_PLACE  # the model's own probability for a pick, not its confidence, decides
        summed.pop(Intent.PICK_PLACE, None)
    if not summed or float(command.get("confidence", 0.0)) < min_confidence:
        return Intent.UNKNOWN
    return max(summed, key=summed.get)


def object_question(object_names) -> dict:
    """The question a decision model answers with the object of a pick command.

    ``object_names`` are the names the detector knows. The answers offered are those names
    and nothing else: offered "none" or "other", the model takes it whenever the sound is
    hard (measured 2026-11-24: 15 of 30 spoken objects right with them, 21 without). A pick
    command with no object therefore comes back with a name the model was not sure of:
    :func:`command_from_answers` drops an object below ``min_object_confidence``.
    """
    return {
        "type": "choice",
        "instructions": OBJECT_QUESTION,
        "criteria": {name: None for name in dict.fromkeys(object_names)},
    }


def most_probable_thing(answer: dict, thing_of: Callable[[str], str]) -> dict:
    """An object answer with the probabilities of every name of one thing added up.

    The object question offers each name the detector knows, and one thing has several: a
    ball is also "football", "soccer ball", "sports ball". The model spreads its probability
    for a ball over them, and a single unrelated name can then be its first choice (measured
    2026-11-24: "pick up the ball" gave "handbag"; with the names of a thing added up, the
    ball in 20 of 20 clips instead of 6, and 93 of 120 spoken objects right instead of 79).
    ``thing_of`` gives the thing a name stands for. The result is ``{"choice": thing,
    "confidence": its summed probability}``; an answer without probabilities is taken by
    its choice.
    """
    probabilities = answer.get("probabilities") or {}
    if not probabilities:
        name = str(answer.get("choice", "") or "")
        return {"choice": thing_of(name) if name else "", "confidence": float(answer.get("confidence", 0.0))}
    summed: dict[str, float] = {}
    for name, probability in probabilities.items():
        thing = thing_of(name)
        summed[thing] = summed.get(thing, 0.0) + float(probability)
    best = max(summed, key=summed.get)
    return {"choice": best, "confidence": summed[best]}


def command_from_answers(
    command: dict, obj: dict | None = None, *, min_pick_probability: float | None = None,
    min_confidence: float = 0.0, min_object_confidence: float = 0.0,
) -> ParsedCommand:
    """A decision model's answers as a command: its command (:func:`chosen_command`), and its
    choice of object for a pick.

    An object the model is less confident of than ``min_object_confidence`` is left out, as
    when the speaker named none.
    """
    intent = chosen_command(command, min_pick_probability=min_pick_probability, min_confidence=min_confidence)
    params: dict[str, str] = {}
    if intent is Intent.PICK_PLACE and obj is not None:
        name = str(obj.get("choice", "") or "").strip()
        if name and float(obj.get("confidence", 0.0)) >= min_object_confidence:
            params["object"] = name
    return ParsedCommand(intent, params=params)
