"""Sentences with the command, object and place each one should give: the voice notebook's table."""

from __future__ import annotations

from so101_smolvla.voice.intents import Intent

PICK = "pick with object and place"
STACK = "learn / rebuild the stack"
EXISTING = "commands the client had before this workshop"
UNKNOWN = "must stay unknown"

# (group, sentence, expected command, expected object, expected place): the sentences of the
# tool-call test of SPEC.md section 4.5. None of them is among the examples in the instructions
# to the model, except "pick up the ball", which is the client's own example.
CASES = [
    (PICK, "pick up the small red cube and put it in the box", "pick_place", "small red cube", "box"),
    (PICK, "put the red cube in the box", "pick_place", "red cube", "box"),
    (PICK, "put the blue ball in the box", "pick_place", "blue ball", "box"),
    (PICK, "grab the large green cylinder and place it in the box", "pick_place", "large green cylinder", "box"),
    (PICK, "pick up the yellow cylinder", "pick_place", "yellow cylinder", ""),
    (PICK, "take the small blue cube to the bin", "pick_place", "small blue cube", "bin"),
    (PICK, "place the large red ball in the bin", "pick_place", "large red ball", "bin"),
    (STACK, "learn the stack", "learn_stack", "", ""),
    (STACK, "rebuild the stack", "rebuild_stack", "", ""),
    (EXISTING, "stop robot", "stop", "", ""),
    (EXISTING, "pick up the ball", "pick_place", "ball", ""),
    (EXISTING, "go home", "home", "", ""),
    (EXISTING, "wave hello", "wave", "", ""),
    (EXISTING, "fetch the cube", "fetch_block", "", ""),
    (EXISTING, "show me your best dance moves", "dance", "", ""),
    (UNKNOWN, "do not rebuild the stack", "unknown", "", ""),
    (UNKNOWN, "the stack looks nice", "unknown", "", ""),
    (UNKNOWN, "I learned a lot today", "unknown", "", ""),
    (UNKNOWN, "who built this stack", "unknown", "", ""),
    (UNKNOWN, "the red cube is in the box", "unknown", "", ""),
    (UNKNOWN, "is the ball in the box", "unknown", "", ""),
]

# (sentence, expected command, expected object or None): the client's own list, from before this
# workshop. It names no place, so the place is not checked for these.
CLIENT_CASES = [
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


def same(a: str, b: str) -> bool:
    """Two argument values, compared after trimming and lower-casing."""
    return a.strip().lower() == b.strip().lower()
