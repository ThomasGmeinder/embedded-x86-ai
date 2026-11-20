"""Simulator worker that places the cubes where the policy side chose.

Started by ``chosen_layout_python`` with .venv-sapien. It serves the
repository's worker protocol unchanged. The only thing replaced is the draw of
the cube places in the stacking scene: the scene still draws its start pose,
the number the layout would have been drawn from, and each cube's turn from
the environment seed in the same order as in a random stacking test, so a test
with a chosen layout differs from the random test of the same seed in the cube
places alone.
"""

from __future__ import annotations

import so101_smolvla.sapien_stack_env as stacking_scene
from so101_smolvla.sapien_env_worker import serve

from chosen_layout import chosen


def _chosen_layout(task_mode: str, seed: int) -> dict[str, list[float]]:
    del task_mode, seed
    return chosen()


stacking_scene.draw_stack_layout = _chosen_layout

if __name__ == "__main__":
    serve()
