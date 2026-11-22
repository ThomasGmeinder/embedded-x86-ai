"""Paint the scene without them over the objects a command does not name.

The policy is given one object to see. Every other object is painted over,
in the top and in the wrist picture, with what the camera sees without it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from so101_smolvla.policy_env import PolicyObservation
from so101_smolvla.sapien_config import OUTLINE_CODES
from so101_smolvla.table_view import table_seen_from

# Pixels by which an outline is widened before it is painted over, so that no
# rim of the object's colour stays around it.
PAINT_MARGIN_PX = 3
# What the widening may spread over. The robot and the objects that stay keep
# every pixel of theirs.
SPREAD_OVER = (OUTLINE_CODES["other"], OUTLINE_CODES["bin"])


@dataclass(frozen=True)
class PaintCount:
    """Pixels painted in each picture, and wrist pixels that could not be.

    A wrist pixel cannot be painted when it looks past the table, where the
    top camera's picture shows nothing; it keeps what the scene draws there.
    """

    top: int = 0
    wrist: int = 0
    wrist_kept: int = 0


def paint_over(
    picture: np.ndarray,
    outline: np.ndarray,
    remove: tuple[int, ...],
    paint: np.ndarray,
    margin: int,
    paintable: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """Replace the outlines to remove, widened by ``margin`` pixels, with ``paint``.

    ``outline`` says for every pixel of ``picture`` what it shows and
    ``remove`` lists the codes to paint over. ``paint`` is a picture of the
    same view without those objects. Where ``paintable`` is given and false,
    ``paint`` means nothing and the pixel is kept.

    Returns the edited picture, the number of pixels painted and the number
    of pixels kept for want of paint.
    """
    if picture.ndim != 3 or picture.shape[2] != 3 or picture.dtype != np.uint8:
        raise ValueError(f"invalid picture {picture.dtype} of shape {picture.shape}")
    if paint.shape != picture.shape or paint.dtype != np.uint8:
        raise ValueError(
            f"paint is {paint.dtype} of shape {paint.shape}, "
            f"the picture is of shape {picture.shape}"
        )
    if outline.shape != picture.shape[:2]:
        raise ValueError(
            f"outline map of shape {outline.shape} does not fit a picture "
            f"of shape {picture.shape}"
        )
    if margin < 0:
        raise ValueError(f"invalid margin {margin}")
    removed = np.isin(outline, remove)
    mask = removed
    if margin and removed.any():
        reach = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)
        )
        widened = cv2.dilate(removed.astype(np.uint8), reach).astype(bool)
        mask = removed | (widened & np.isin(outline, SPREAD_OVER))
    kept = 0
    if paintable is not None:
        if paintable.shape != outline.shape or paintable.dtype != bool:
            raise ValueError(
                f"paintable is {paintable.dtype} of shape {paintable.shape}, "
                f"expected bool of shape {outline.shape}"
            )
        kept = int((mask & ~paintable).sum())
        mask = mask & paintable
    edited = picture.copy()
    edited[mask] = paint[mask]
    return edited, int(mask.sum()), kept


def edit_pictures(
    observation: PolicyObservation,
    remove: tuple[str, ...],
    scene_without_objects: np.ndarray,
    top_camera: np.ndarray,
    wrist_camera: np.ndarray,
    table_height: float,
    margin: int = PAINT_MARGIN_PX,
) -> tuple[PolicyObservation, PaintCount]:
    """The observation with the named objects painted over in both pictures.

    ``scene_without_objects`` is the top camera's picture of the scene
    without them. It is the paint of the top picture as it is, and of the
    wrist picture as the wrist camera sees it lying on the table.
    ``top_camera`` and ``wrist_camera`` are the 3x4 projections that took the
    two pictures of ``observation``. With nothing to remove the observation
    comes back as it is.
    """
    if not remove:
        return observation, PaintCount()
    if observation.top_outline is None or observation.wrist_outline is None:
        raise ValueError("the scene hands out no outline maps to paint by")
    try:
        codes = tuple(OUTLINE_CODES[name] for name in remove)
    except KeyError as exc:
        raise ValueError(f"no outline is named {exc.args[0]!r}") from exc
    top, top_painted, _ = paint_over(
        observation.top, observation.top_outline, codes, scene_without_objects, margin
    )
    if not np.isin(observation.wrist_outline, codes).any():
        return replace(observation, top=top), PaintCount(top=top_painted)
    wrist_paint, on_table = table_seen_from(
        scene_without_objects,
        np.asarray(top_camera, dtype=np.float64),
        np.asarray(wrist_camera, dtype=np.float64),
        table_height,
    )
    wrist, wrist_painted, wrist_kept = paint_over(
        observation.wrist,
        observation.wrist_outline,
        codes,
        wrist_paint,
        margin,
        paintable=on_table,
    )
    return (
        replace(observation, top=top, wrist=wrist),
        PaintCount(top=top_painted, wrist=wrist_painted, wrist_kept=wrist_kept),
    )
