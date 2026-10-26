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

1. add the enum member + one line in :data:`INTENT_SYSTEM_PROMPT`,
2. add its name to ``config/intent.gbnf``'s ``cmd`` rule,
3. decorate a handler with ``@registry.register(Intent.MY_COMMAND)``.

``STOP`` is special: the orchestrator matches it with plain keywords on the
raw transcript (no LLM round trip) so the safety stop is as fast as possible,
and it interrupts any running behavior via the shared stop event.
"""

from __future__ import annotations

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
    PLACE = "place"
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


# Transcript keywords that trigger STOP without an LLM round trip. Matched
# case-insensitively as whole words on the raw transcript - the safety stop
# must never wait on a language model.
STOP_KEYWORDS = ("stop", "halt", "freeze", "abort", "emergency")
_STOP_RE = re.compile(r"\b(" + "|".join(STOP_KEYWORDS) + r")\b", re.IGNORECASE)


def is_stop(transcript: str) -> bool:
    """True when the raw transcript contains a stop keyword."""
    return bool(_STOP_RE.search(transcript or ""))


# Whisper, fed the first burst of room noise the moment the mic goes hot,
# reliably hallucinates a few stock phrases ("you", "thank you.", "Bye.",
# subtitle credits). With the grammar forcing every transcript into one valid
# command, such a phantom utterance would dispatch a real behavior, so these
# are dropped before they reach the LLM. Phrases match the transcript
# lowercased and stripped of whitespace and trailing punctuation, whole.
DEFAULT_NOISE_PHRASES = (
    "",
    "you",
    "thank you",
    "thanks",
    "bye",
    "goodbye",
    "hello",
    "hi",
    "hmm",
    "uh",
    "um",
    "ah",
    "oh",
    "yeah",
    "okay",
    "ok",
    "the",
    "a",
    "so",
    "and",
    "thank you for watching",
    "thanks for watching",
    "please subscribe",
    "subtitles by the amara.org community",
    ".",
    "..",
    "...",
)

# Whisper's non-speech annotations: [BLANK_AUDIO], (sighs), *music*.
_NON_SPEECH_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*")


def _normalize_transcript(text: str) -> str:
    """Lowercase, strip surrounding whitespace and trailing punctuation/quotes."""
    return re.sub(r"[\s\.\,\!\?\"\'`]+$", "", (text or "").strip().lower()).strip()


def make_noise_gate(cfg: dict):
    """Build an ``is_noise(text) -> bool`` predicate from config.

    ``audio.noise_phrases`` (optional) replaces the defaults; a transcript with
    no alphabetic content is always noise.
    """
    phrases = (cfg.get("audio") or {}).get("noise_phrases")
    phrase_set = {_normalize_transcript(p) for p in (phrases if phrases is not None else DEFAULT_NOISE_PHRASES)}

    def is_noise(text: str) -> bool:
        """True for known silence-hallucination phrases and empty transcripts."""
        norm = _normalize_transcript(_NON_SPEECH_TAG.sub(" ", text or ""))
        if not norm or not any(c.isalpha() for c in norm):
            return True
        return norm in phrase_set

    return is_noise


is_noise = make_noise_gate({})


# A keyword stand-in for the LLM, for runs without llama-server (--mock, tests).
_KEYWORD_ROUTES = [
    (r"\b(stop|halt|freeze|abort)\b", "stop", None),
    (r"\b(release|open|drop)\b|\blet go\b", "release", None),
    (r"\b(copy|mirror|follow|mimic)\b", "gesture_mimic", None),
    (r"\b(block|cube)\b", "fetch_block", None),
    (r"\b(pick|grab|bring|fetch|get)\b", "pick_place", "OBJECT"),
    (r"\bplace\b|\b(set|put|lay)\b.*\b(down|table)\b", "place", None),
    (r"\b(dance|groove|party|moves)\b", "dance", None),
    (r"\b(hold|take)\b", "grip", None),
    (r"\b(wave|hi|hello|greet)\b", "wave", None),
    (r"\b(home|rest|reset)\b", "home", None),
]
_OBJECT_WORDS = ("ball", "bottle", "cup", "mug", "phone", "apple", "banana", "book", "toothbrush",
                 "remote", "orange", "teddy", "pen")


def offline_parse_text(text: str) -> dict:
    """Keyword routing → the same ``{"command", "object"}`` dict the LLM emits."""
    low = (text or "").lower()
    for pattern, command, obj_mode in _KEYWORD_ROUTES:
        if re.search(pattern, low):
            obj = ""
            if obj_mode == "OBJECT":
                obj = next((word for word in _OBJECT_WORDS if word in low), "")
            return {"command": command, "object": obj}
    return {"command": "unknown", "object": ""}


def keyword_parse(text: str) -> ParsedCommand:
    """:func:`offline_parse_text` as a :class:`ParsedCommand`: a parser that needs no LLM."""
    d = offline_parse_text(text)
    params = {"object": d["object"]} if d.get("object") else {}
    return ParsedCommand(Intent.from_string(d["command"]), params=params, transcript=text)


class CommandRegistry:
    """Maps intents to behavior entry points; the orchestrator just dispatches.

    Handlers receive ``(ctx, cmd: ParsedCommand)`` where ``ctx`` is a
    :class:`~vla_pipeline.dispatch.PipelineContext` (arm, cameras, models,
    shared state, stop_check).
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

    def add(self, intent: Intent, fn: Callable) -> None:
        """Non-decorator registration (same rules as :meth:`register`)."""
        self.register(intent)(fn)

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
You are the command router for a robot arm. Map the user's spoken request to
exactly one command and answer ONLY with a JSON object
{"command": "<name>", "object": "<object or empty string>"}.
"object" is the thing to act on for pick_place (one or two plain words,
e.g. "ball", "water bottle"); for every other command use "".

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
- "place": the user wants the held object set down on the table ("place it",
  "set it down", "put it down", "set it on the table").
- "wave": the user wants a wave / greeting ("wave at me", "say hi").
- "home": return to the rest/home position ("go home", "reset position",
  "rest now").
- "stop": stop moving immediately ("stop", "halt", "freeze").
- "unknown": anything else.
"""
