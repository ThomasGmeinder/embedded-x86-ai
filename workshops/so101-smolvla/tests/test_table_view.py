from __future__ import annotations

import numpy as np

from so101_smolvla.table_view import table_seen_from

HEIGHT, WIDTH = 48, 64
FOCAL_PX = 100.0
TABLE_HEIGHT = -0.0045
# Pixels are whole numbers; the horizon of a level camera lies between two rows.
INTRINSIC = np.array(
    [[FOCAL_PX, 0.0, WIDTH / 2], [0.0, FOCAL_PX, HEIGHT / 2 - 0.5], [0.0, 0.0, 1.0]]
)
LOOKING_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
LOOKING_ALONG_THE_TABLE = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _camera(turn: np.ndarray, position: tuple[float, float, float]) -> np.ndarray:
    return INTRINSIC @ np.column_stack([turn, -turn @ np.asarray(position)])


def _picture() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8)


def _levels_apart(first: np.ndarray, second: np.ndarray) -> int:
    return int(np.abs(first.astype(int) - second.astype(int)).max())


def test_a_camera_sees_its_own_table_picture_unchanged():
    picture = _picture()
    camera = _camera(LOOKING_DOWN, (0.2, 0.1, TABLE_HEIGHT + 1.0))
    painted, on_table = table_seen_from(picture, camera, camera, TABLE_HEIGHT)
    assert on_table.all()
    assert _levels_apart(painted, picture) <= 1


def test_a_camera_moved_sideways_sees_the_table_picture_shifted_and_its_edge_continued():
    picture = _picture()
    taken_by = _camera(LOOKING_DOWN, (0.2, 0.1, TABLE_HEIGHT + 1.0))
    # One metre above the table, 0.08 m to the side is 8 pixels at this lens.
    seen_by = _camera(LOOKING_DOWN, (0.28, 0.1, TABLE_HEIGHT + 1.0))
    painted, on_table = table_seen_from(picture, taken_by, seen_by, TABLE_HEIGHT)
    assert on_table.all()
    assert _levels_apart(painted[:, : WIDTH - 8], picture[:, 8:]) <= 1
    # Beyond what the first camera saw, its picture's last column goes on.
    assert _levels_apart(
        painted[:, WIDTH - 7 :], np.repeat(picture[:, -1:], 7, axis=1)
    ) <= 1


def test_pixels_above_the_horizon_do_not_look_at_the_table():
    taken_by = _camera(LOOKING_DOWN, (0.2, 0.1, TABLE_HEIGHT + 1.0))
    seen_by = _camera(LOOKING_ALONG_THE_TABLE, (0.0, 0.0, TABLE_HEIGHT + 0.5))
    painted, on_table = table_seen_from(_picture(), taken_by, seen_by, TABLE_HEIGHT)
    assert painted.shape == (HEIGHT, WIDTH, 3)
    assert not on_table[: HEIGHT // 2].any()
    assert on_table[HEIGHT // 2 :].all()
