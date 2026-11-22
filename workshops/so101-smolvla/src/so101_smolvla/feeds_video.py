"""One video of what the cameras took, what was outlined and what SmolVLA was given.

A row per camera and three pictures per row: the camera's picture at every
step; the outlines the picture editor was given when SmolVLA last planned,
drawn over the picture they were found in; and the picture SmolVLA was given
then. SmolVLA looks at a picture only when it plans, so the last two hold
until its next plan. A line of text above every picture says what it is.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import cv2
import numpy as np

# Height of the line of text above every picture.
CAPTION_HEIGHT = 24
# What the picture editor does with an outlined object, and the colour its
# outline is drawn in. None of them is the colour of a cube. "not used" is an
# outline the editor was not given, because it is on no cube.
ROLE_COLORS = {
    "target": (255, 220, 0),
    "painted over": (230, 0, 230),
    "left as it is": (0, 210, 210),
    "not used": (150, 150, 150),
}
# Share of the role's colour laid over the pixels inside an outline.
TINT = 0.45
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.45


@dataclass(frozen=True)
class Outline:
    """An outlined object: its name and what the picture editor does with it."""

    label: str
    role: str


@dataclass(frozen=True)
class PlanFeeds:
    """One camera at the start of one plan of SmolVLA.

    ``step`` is the first step the plan covers, counted over the whole
    layout. ``outlines`` holds for every pixel of ``taken`` a zero or the
    number, from one, of the entry of ``legend`` the pixel belongs to.
    """

    step: int
    command: str
    taken: np.ndarray
    outlines: np.ndarray
    legend: tuple[Outline, ...]
    shown: np.ndarray


def _text_size(text: str) -> tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, _FONT, _FONT_SCALE, 1)
    return width, height


def caption(text: str, width: int) -> np.ndarray:
    """A dark line of ``width`` pixels with ``text`` in white."""
    needed, _ = _text_size(text)
    if needed + 8 > width:
        raise ValueError(
            f"caption {text!r} needs {needed + 8} pixels, the picture has {width}"
        )
    line = np.full((CAPTION_HEIGHT, width, 3), 30, dtype=np.uint8)
    cv2.putText(
        line, text, (4, CAPTION_HEIGHT - 8), _FONT, _FONT_SCALE,
        (255, 255, 255), 1, cv2.LINE_AA,
    )
    return line


def outline_picture(
    taken: np.ndarray, outlines: np.ndarray, legend: tuple[Outline, ...]
) -> np.ndarray:
    """``taken`` with every outline tinted, edged and named in its role's colour."""
    if taken.ndim != 3 or taken.shape[2] != 3 or taken.dtype != np.uint8:
        raise ValueError(f"invalid picture {taken.dtype} of shape {taken.shape}")
    if outlines.shape != taken.shape[:2] or outlines.dtype != np.uint8:
        raise ValueError(
            f"outlines are {outlines.dtype} of shape {outlines.shape}, "
            f"the picture is of shape {taken.shape}"
        )
    if int(outlines.max(initial=0)) > len(legend):
        raise ValueError(
            f"outline number {int(outlines.max())} has no entry among "
            f"the {len(legend)} of the legend"
        )
    for outline in legend:
        if outline.role not in ROLE_COLORS:
            raise ValueError(
                f"unknown role {outline.role!r}; expected one of {list(ROLE_COLORS)}"
            )
    drawn = taken.copy()
    names = []
    for number, outline in enumerate(legend, start=1):
        mask = outlines == number
        if not mask.any():
            continue
        colour = ROLE_COLORS[outline.role]
        drawn[mask] = np.round(
            TINT * np.asarray(colour) + (1.0 - TINT) * drawn[mask]
        ).astype(np.uint8)
        edges, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv2.drawContours(drawn, edges, -1, colour, 2)
        rows, columns = np.nonzero(mask)
        names.append((f"{outline.label}: {outline.role}", colour, rows, columns))
    # The names go on last, so that no tint or edge runs through one.
    for text, colour, rows, columns in names:
        width, height = _text_size(text)
        left = int(min(columns.min(), taken.shape[1] - width - 4))
        # Above the outline, or below it where the picture ends above.
        baseline = int(rows.min()) - 6
        if baseline - height < 2:
            baseline = min(int(rows.max()) + height + 8, taken.shape[0] - 4)
        cv2.rectangle(
            drawn, (left - 2, baseline - height - 3), (left + width + 2, baseline + 4),
            (30, 30, 30), -1,
        )
        cv2.putText(drawn, text, (left, baseline), _FONT, _FONT_SCALE, colour, 1, cv2.LINE_AA)
    return drawn


def feed_frames(
    frames: Mapping[str, Sequence[np.ndarray]],
    plans: Mapping[str, Sequence[PlanFeeds]],
    outline_source: str,
) -> Iterator[np.ndarray]:
    """A frame per step, with a row per camera of ``frames`` in its order.

    ``frames[camera][step]`` is the camera's picture after that step and
    ``plans[camera]`` the camera's plans in the order they were made, the
    first at step 0. ``outline_source`` says where the outlines come from
    and is written above them.
    """
    if not frames or set(frames) != set(plans):
        raise ValueError(
            f"frames of cameras {list(frames)}, plans of cameras {list(plans)}"
        )
    counts = {camera: len(pictures) for camera, pictures in frames.items()}
    if len(set(counts.values())) != 1 or min(counts.values()) < 1:
        raise ValueError(f"every camera needs the same number of frames, got {counts}")
    for camera, made in plans.items():
        steps = [plan.step for plan in made]
        if not steps or steps[0] != 0 or any(
            later <= earlier for earlier, later in zip(steps, steps[1:])
        ):
            raise ValueError(
                f"the plans of camera {camera!r} start at steps {steps}; "
                "they must start at 0 and go up"
            )
        shape = frames[camera][0].shape
        for plan in made:
            if plan.taken.shape != shape or plan.shown.shape != shape:
                raise ValueError(
                    f"a plan of camera {camera!r} holds pictures of shape "
                    f"{plan.taken.shape} and {plan.shown.shape}, its frames are {shape}"
                )
    return _compose(frames, plans, outline_source, next(iter(counts.values())))


def _held(plan: PlanFeeds, outline_source: str) -> np.ndarray:
    """The two pictures of a plan that stay until the next plan, side by side."""
    width = plan.taken.shape[1]
    outlined = np.concatenate(
        (
            caption(f"{outline_source} at step {plan.step}", width),
            outline_picture(plan.taken, plan.outlines, plan.legend),
        )
    )
    given = np.concatenate(
        (caption(f"given to SmolVLA at step {plan.step}", width), plan.shown)
    )
    return np.concatenate((outlined, given), axis=1)


def _compose(
    frames: Mapping[str, Sequence[np.ndarray]],
    plans: Mapping[str, Sequence[PlanFeeds]],
    outline_source: str,
    count: int,
) -> Iterator[np.ndarray]:
    in_force = {camera: -1 for camera in frames}
    held: dict[str, np.ndarray] = {}
    for step in range(count):
        rows = []
        for camera, pictures in frames.items():
            made = plans[camera]
            index = in_force[camera]
            while index + 1 < len(made) and made[index + 1].step <= step:
                index += 1
            if index != in_force[camera]:
                in_force[camera] = index
                held[camera] = _held(made[index], outline_source)
            picture = pictures[step]
            live = np.concatenate(
                (
                    caption(
                        f"{camera} camera, step {step}: {made[index].command}",
                        picture.shape[1],
                    ),
                    picture,
                )
            )
            rows.append(np.concatenate((live, held[camera]), axis=1))
        yield np.concatenate(rows)
