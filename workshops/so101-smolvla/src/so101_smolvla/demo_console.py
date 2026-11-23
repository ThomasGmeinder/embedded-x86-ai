"""The demo's one picture: the six tiles of the feeds video and a side panel with the meters.

Pure drawing, no state and no threads: ``compose`` takes an immutable
:class:`Snapshot` of the run, the transcript lines, the :class:`VoiceLevel`
and the clock, and returns one RGB picture. The tiles are the ones
``feeds_video`` writes into a layout's video (per camera: the scene as the
camera takes it, the outlines SmolVLA last planned on, tinted by what was done
with each, and the picture SmolVLA was given), at half size. The panel shows
the voice meter (the frame level, green while Silero counts the frame as
speech), the transcript, the command's progress and the gripper: the jaw
opening the simulator reports (0 to 100 % of travel), coloured while a jaw
touches the target cube, "grasped" while it holds it. The twin measures no
gripper current or force, so none is shown.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from so101_smolvla.bin_layouts import BIN_CUBE_COLORS
from so101_smolvla.feeds_video import CAPTION_HEIGHT, PlanFeeds, caption, outline_picture
from so101_smolvla.sapien_config import IMAGE_SHAPE

CAMERAS = ("top", "wrist")
SCALE = 0.5  # the tiles at half their size
TILE_W = int(IMAGE_SHAPE[1] * SCALE)
TILE_H = int((IMAGE_SHAPE[0] + CAPTION_HEIGHT) * SCALE)
TILES_W = 3 * TILE_W
TILES_H = len(CAMERAS) * TILE_H
PANEL_W = 400
FRAME_SHAPE = (TILES_H, TILES_W + PANEL_W, 3)
# The voice meter goes red when no frame arrived for this long.
SILENT_AFTER_S = 2.0
# Panel colours (RGB).
PANEL = (28, 28, 32)
TEXT = (235, 235, 235)
DIM = (120, 120, 128)
GREEN = (80, 200, 100)
RED = (220, 70, 70)
YELLOW = (230, 200, 60)
CYAN = (90, 200, 220)
PLACEHOLDER = (60, 60, 66)
TRANSCRIPT_COLOURS = {"you": TEXT, "call": CYAN, "note": DIM, "error": RED}
WRAP_AT = 50  # characters of the small font that fit the panel's width
# Where the two bars are drawn in the panel: x, y, width, height.
VAD_BAR = (16, 56, PANEL_W - 32, 12)
GRIP_BAR = (16, TILES_H - 44, PANEL_W - 32, 12)


@dataclass(frozen=True)
class VoiceLevel:
    """The microphone as the Listener last reported it."""

    available: bool = False  # the microphone streams
    error: str | None = None  # why it does not, when it does not
    level: float = 0.0  # 0..1, from the frame's level
    speaking: bool = False  # Silero counted the frame as speech
    busy: bool = False  # a request to Gemma is under way
    time: float = 0.0  # monotonic seconds of the last frame


@dataclass(frozen=True)
class Snapshot:
    """The run as it stands, replaced whole by its one owner (demo.DemoSession)."""

    seed: int
    status: str  # one line: what is happening
    outline_source: str
    command: str | None = None
    cube: str | None = None
    step: int | None = None  # within the command
    steps_per_pick: int = 0
    pictures: dict = field(default_factory=dict)  # camera -> the picture as taken, or absent
    plans: dict = field(default_factory=dict)  # camera -> PlanFeeds of the plan in force
    opening: float | None = None  # the gripper, 0..100 % of travel
    contact: bool = False
    grasped: bool = False
    in_bin: bool = False
    placed_at: int | None = None  # the step the simulator's rule first held
    cubes_in_bin: tuple = ()


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONT = _font(15)
SMALL = _font(13)


def _tile(picture: np.ndarray | None, text: str) -> np.ndarray:
    """One captioned picture at half size; a dark placeholder when there is no picture."""
    body = (
        np.full(IMAGE_SHAPE, PLACEHOLDER, dtype=np.uint8) if picture is None else picture
    )
    full = np.vstack([caption(text, IMAGE_SHAPE[1]), body])
    return cv2.resize(full, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)


def tiles(snapshot: Snapshot) -> np.ndarray:
    """The six tiles, a row per camera: as taken, the outlines, what SmolVLA was given."""
    rows = []
    for camera in CAMERAS:
        taken = snapshot.pictures.get(camera)
        plan: PlanFeeds | None = snapshot.plans.get(camera)
        step = "" if snapshot.step is None else f", step {snapshot.step}"
        live = _tile(taken, f"{camera} camera{step}: {snapshot.command or 'waiting for a command'}")
        if plan is None:
            outlined = _tile(None, f"{snapshot.outline_source}: no plan yet")
            given = _tile(None, "given to SmolVLA: no plan yet")
        else:
            outlined = _tile(
                outline_picture(plan.taken, plan.outlines, plan.legend),
                f"{snapshot.outline_source} at step {plan.step}",
            )
            given = _tile(plan.shown, f"given to SmolVLA at step {plan.step}")
        rows.append(np.hstack([live, outlined, given]))
    return np.vstack(rows)


def _bar(draw: ImageDraw.ImageDraw, rect: tuple, share: float, colour: tuple, mark: float | None = None) -> None:
    x, y, w, h = rect
    draw.rectangle([x, y, x + w, y + h], outline=DIM, width=1)
    fill = int(round(max(0.0, min(1.0, share)) * (w - 2)))
    if fill > 0:
        draw.rectangle([x + 1, y + 1, x + 1 + fill, y + h - 1], fill=colour)
    if mark is not None:
        mx = x + int(round(max(0.0, min(1.0, mark)) * w))
        draw.line([mx, y - 3, mx, y + h + 3], fill=TEXT, width=1)


def voice_state(voice: VoiceLevel, now: float) -> tuple[str, tuple]:
    """The one line above the voice meter and its colour."""
    if not voice.available:
        return (f"no microphone: {voice.error}" if voice.error else "no microphone; type a command"), RED
    if now - voice.time > SILENT_AFTER_S:
        return "no microphone signal", RED
    if voice.busy:
        return "asking Gemma...", YELLOW
    if voice.speaking:
        return "hearing you", GREEN
    return "listening", DIM


def panel(snapshot: Snapshot, transcript: tuple, voice: VoiceLevel, now: float) -> np.ndarray:
    """The side panel: voice meter, command, transcript, gripper."""
    image = Image.new("RGB", (PANEL_W, TILES_H), PANEL)
    draw = ImageDraw.Draw(image)
    # Voice
    draw.text((16, 14), "VOICE", font=FONT, fill=TEXT)
    state, colour = voice_state(voice, now)
    draw.text((80, 14), state, font=SMALL, fill=colour)
    if voice.available:
        _bar(draw, VAD_BAR, voice.level, GREEN if voice.speaking else DIM)
    # Command
    y = 84
    draw.text((16, y), "COMMAND", font=FONT, fill=TEXT)
    draw.text((16, y + 22), snapshot.status, font=SMALL, fill=CYAN)
    if snapshot.cube is not None and snapshot.step is not None and snapshot.steps_per_pick:
        _bar(draw, (16, y + 46, PANEL_W - 32, 6), (snapshot.step + 1) / snapshot.steps_per_pick, CYAN)
    # Transcript
    y = 150
    draw.line([16, y, PANEL_W - 16, y], fill=DIM, width=1)
    y += 10
    bottom = GRIP_BAR[1] - 64
    # Long lines wrap; the newest lines win the room.
    wrapped = [
        (line, kind)
        for text, kind in transcript
        for line in (textwrap.wrap(text, WRAP_AT, subsequent_indent="    ") or [""])
    ]
    room = max(0, (bottom - y) // 19)
    for text, kind in wrapped[-room:] if room else []:
        draw.text((16, y), text, font=SMALL, fill=TRANSCRIPT_COLOURS.get(kind, TEXT))
        y += 19
    # Gripper
    gy = GRIP_BAR[1]
    draw.line([16, gy - 40, PANEL_W - 16, gy - 40], fill=DIM, width=1)
    if snapshot.opening is None:
        draw.text((16, gy - 30), "gripper: no reading", font=SMALL, fill=DIM)
    else:
        state = "grasped" if snapshot.grasped else ("contact" if snapshot.contact else "open" if snapshot.opening > 10 else "closed")
        colour = GREEN if (snapshot.contact or snapshot.grasped) else TEXT
        draw.text((16, gy - 30), f"gripper: jaw opening {snapshot.opening:3.0f} % of travel, {state}", font=SMALL, fill=colour)
        _bar(draw, GRIP_BAR, snapshot.opening / 100.0, GREEN if (snapshot.contact or snapshot.grasped) else DIM)
    placed = ", ".join(snapshot.cubes_in_bin) or "none"
    draw.text((16, TILES_H - 24), f"layout seed {snapshot.seed}; in the bin: {placed}", font=SMALL, fill=DIM)
    return np.asarray(image)


def compose(snapshot: Snapshot, transcript: tuple, voice: VoiceLevel, now: float) -> np.ndarray:
    """The whole picture: the six tiles on the left, the panel on the right."""
    frame = np.hstack([tiles(snapshot), panel(snapshot, transcript, voice, now)])
    assert frame.shape == FRAME_SHAPE, frame.shape
    return frame


def jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    """The picture as JPEG bytes, for a widget."""
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("could not encode the frame")
    return encoded.tobytes()


def cubes_in_bin(facts: dict) -> tuple:
    return tuple(color for color in BIN_CUBE_COLORS if facts["cubes"][color]["in_bin"])
