"""One camera's picture of the table, seen from where another camera stands."""

from __future__ import annotations

import cv2
import numpy as np


def table_seen_from(
    picture: np.ndarray,
    taken_by: np.ndarray,
    seen_by: np.ndarray,
    table_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Paint the table plane, as one camera sees it, with another camera's picture of it.

    ``picture`` was taken by the camera ``taken_by``; the result has the same
    size and is what the camera ``seen_by`` sees if that picture lies on the
    horizontal plane at ``table_height``. A camera is its 3x4 projection:
    pixels from coordinates in the frame ``table_height`` is given in, with
    the depth in front of the camera positive. Where ``taken_by`` did not see
    the plane, the nearest edge of its picture is continued.

    Returns the painted picture and, per pixel, whether it looks at the plane
    in front of both cameras. The other pixels look past the plane and their
    paint means nothing.
    """
    height, width = picture.shape[:2]
    on_plane = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, table_height], [0.0, 0.0, 1.0]]
    )
    # Pixel of the seeing camera -> point on the plane -> pixel of the picture.
    to_plane = np.linalg.inv(seen_by @ on_plane)
    to_picture = taken_by @ on_plane @ to_plane
    painted = cv2.warpPerspective(
        picture,
        to_picture,
        (width, height),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64)
    )
    pixels = np.stack([columns, rows, np.ones_like(columns)], axis=-1)
    # The last coordinate of a pixel's plane point is one over its depth in
    # the seeing camera; times its depth in the other camera it is the last
    # coordinate of its place in the picture.
    in_front = ((pixels @ to_plane[2]) > 0.0) & ((pixels @ to_picture[2]) > 0.0)
    return painted, in_front
