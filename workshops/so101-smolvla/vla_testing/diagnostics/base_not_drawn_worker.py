"""Simulator worker whose recording-rig scene does not draw the robot's base.

Started by ``base_not_drawn_python`` with .venv-sapien. It serves the
repository's worker protocol unchanged. The only thing changed is the list of
robot links the recording-rig scene does not draw
(sapien_stack_env.RECORDING_RIG_ABSENT_LINKS): the base link is added to it.
Both cameras then show, where the drawn base was, the table picture, which
holds the recording robot's own base at that place. The base stays in the
physics.
"""

from __future__ import annotations

import so101_smolvla.sapien_stack_env as stacking_scene
from so101_smolvla.sapien_env_worker import serve

BASE_LINK = "base_link"

stacking_scene.RECORDING_RIG_ABSENT_LINKS = (
    *stacking_scene.RECORDING_RIG_ABSENT_LINKS, BASE_LINK)

if __name__ == "__main__":
    serve()
