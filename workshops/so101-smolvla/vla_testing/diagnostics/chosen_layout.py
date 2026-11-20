"""How the policy side hands one chosen cube layout to the diagnostic simulator worker.

Imported by both sides (same_tests_with_more_room.py with .venv,
chosen_layout_worker.py with .venv-sapien); it needs nothing but the standard
library. The worker is a child process of the policy side and inherits its
environment variables.
"""

from __future__ import annotations

import json
import os

VARIABLE = "SO101_CHOSEN_STACK_LAYOUT"


def choose(layout: dict[str, list[float]]) -> None:
    """Set the layout that simulator workers started from now on will place."""
    os.environ[VARIABLE] = json.dumps(layout)


def chosen() -> dict[str, list[float]]:
    """The layout chosen by the policy side; an error if it chose none."""
    try:
        return json.loads(os.environ[VARIABLE])
    except KeyError as exc:
        raise RuntimeError(f"{VARIABLE} is not set: no cube layout was chosen") from exc
