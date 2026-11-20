"""How the recording robot's arm joints stop short of their commands.

Imported by replay_recordings_fresh_scene.py and servo_stop_short.py
(.venv-sapien); it needs neither the simulator nor the model.

On the recording robot a joint's motor does not move the joint while the
command is closer than a fixed angle, the joint's stop-short. So a joint
comes to rest short of its command, on the side it came from.
servo_stop_short.py reads that angle for every arm joint from the recordings
and stores it in analysis/servo-stop-short-fit-40.json.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MEASURED = (
    Path.home() / ".cache" / "so101-stack-training" / "analysis" / "servo-stop-short-fit-40.json"
)


def held_target(target: np.ndarray, command: np.ndarray, stop_short: np.ndarray) -> np.ndarray:
    """What each joint's motor aims at after a new command.

    The target stays where it is while the command is within the joint's
    stop-short of it, and otherwise follows the command at that distance.
    """
    return np.clip(target, command - stop_short, command + stop_short)


def held_targets(commands: np.ndarray, start: np.ndarray, stop_short: np.ndarray) -> np.ndarray:
    """``held_target`` frame by frame; ``start`` is where the joints stand before the first command."""
    targets = np.empty_like(commands)
    target = np.asarray(start, dtype=np.float64).copy()
    for frame, command in enumerate(commands):
        target = held_target(target, command, stop_short)
        targets[frame] = target
    return targets
