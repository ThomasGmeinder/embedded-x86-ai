# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Twin console - one image of what the pipeline sees, hears and does.

Layout (1200 x 1100)::

    +-----------------+-----------------+-----------------+
    | mount pose      | mount hands     | arm detect      |   400 x 300 each
    +-----------------+-----------------+-----------------+
    |                                   | voice / intent  |
    |        3D view of the twin        | behavior log    |   800 x 800 + 400 x 800
    |                                   | gripper, joints |
    +-----------------------------------+-----------------+

The console only observes. Cameras are wrapped in :class:`FanoutCamera` so the
console and a running behavior both get every frame; the models are tapped in
place (:class:`ModelTap`) so the tiles show what the behavior computed, and
the console runs a model itself only while no behavior is using it. The
UserRobotControl's events (:meth:`TwinConsole.attach`) fill the voice panel.

:meth:`TwinConsole.compose` returns the frame: :meth:`TwinConsole.run` shows
it in a desktop window (``run_pipeline.sh --digital-twin``), and the digital
twin notebook shows it in a widget (:func:`jpeg`). All HighGUI calls happen
on the thread that calls :meth:`TwinConsole.run`.
"""

from __future__ import annotations

import logging
import textwrap
import threading
import time
from collections import deque

import cv2
import numpy as np

from so101_sim import MOTOR_SHORT
from vla_pipeline.robot.moveit_arm import GRIPPER_CONTACT_MA, GRIPPER_STALL_MA

logger = logging.getLogger(__name__)

WINDOW = "SO-101 twin console"
TILE_W, TILE_H = 400, 300
VIEW_W, VIEW_H = 800, 800  # tall enough for the raised arm (wave)
LOG_LINES = (VIEW_H - 250) // 20  # what fits under the voice lines and above the arm HUD
PANEL_W = 400
# The reset button, in the composed image. The notebook's reset button calls
# the same TwinConsole.reset_scene the ``r`` key calls.
RESET_BUTTON = (VIEW_W + 12, TILE_H + VIEW_H - 44, 110, 30)
IDLE_AFTER_S = 0.5  # a model untouched this long is run by the console itself
FPS = 15.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GRAY, DIM = (235, 235, 235), (150, 150, 150), (90, 90, 90)
GREEN, YELLOW, RED, CYAN = (80, 200, 80), (60, 200, 230), (60, 60, 230), (220, 200, 80)
ORANGE = (40, 140, 240)
RESET_LINE = "Reset. Arm parked, objects back on the table."
_LINE_COLOR = {"cyan": CYAN, "red": RED, "gray": GRAY}


def panel_line(event: str, **data) -> tuple[str, str] | None:
    """The voice-log line for one behavior event: ``(text, color name)``.

    The drawing process and the in-process console both use this, so the
    ``->`` line has one wording.
    """
    if event == "parsed":
        params = ", ".join(f"{k}={v}" for k, v in (data.get("params") or {}).items())
        text = f"-> {data['intent']}" + (f" ({params})" if params else "")
        return text, "cyan"
    if event == "ignored":
        return f"  (ignored: {data['reason']})", "gray"
    if event == "stop":
        return "STOP", "red"
    if event == "behavior_crash":
        return f"  crashed: {data['error']}", "red"
    if event == "behavior_end":
        how = "stopped" if data.get("stopped") else ("done" if data.get("ok") else "failed")
        return f"  {data['intent']} {how} after {data['took']:.1f}s", "gray"
    return None


POSE = "mount: YOLOv26s-pose"
HANDS = "mount: MediaPipe hands"
DETECT = "arm: YOLOv26s-detect"
VIEW = "digital twin simulated in MuJoCo"
VOICE = (
    "Voice: gemma4-e4b-it transcription",
    "and intent-to-command translation",
)


def with_processor(caption: str, processor: str) -> str:
    """``caption`` plus the processor it is running on, in brackets."""
    return f"{caption} [{processor}]"


def hits_reset_button(x: int, y: int) -> bool:
    """True when an image-space click lands on the reset button."""
    bx, by, bw, bh = RESET_BUTTON
    return bx <= x < bx + bw and by <= y < by + bh


def pointer_hits_reset(x: int, y: int, image_wh: tuple[int, int], window_rect: tuple[int, int, int, int]) -> bool:
    """True when a window click lands on the reset button.

    OpenCV reports the pointer either in image pixels or in the window, and
    ``getWindowImageRect`` may be screen coordinates. Accept the mapping that
    lands on the button.
    """
    if hits_reset_button(x, y):
        return True
    iw, ih = image_wh
    _rx, _ry, rw, rh = window_rect
    if rw <= 0 or rh <= 0 or iw <= 0 or ih <= 0:
        return False
    scaled = (int(x * iw / rw), int(y * ih / rh))
    shifted = (int((x - _rx) * iw / rw), int((y - _ry) * ih / rh))
    return hits_reset_button(*scaled) or hits_reset_button(*shifted)


class FanoutCamera:
    """One reader thread; every consumer thread gets each new frame once.

    ``read()`` blocks for the next frame this thread hasn't seen, so behaviors
    keep camera-like pacing while the console reads the same stream.

    The reader blocks on the camera: a device or a paced stand-in in its
    ``read()``, a latest-frame client (a ROS topic, the twin) in its
    ``next_frame(after)``, which returns the first frame newer than ``after``.
    A camera that stops giving frames ends the reader, unless it says when it
    will try its device again (``retry_at()``, wall-clock seconds).
    """

    def __init__(self, inner, timeout_s: float = 1.0, *, own: bool = True):
        self.inner = inner
        self._own = own
        self.role = getattr(inner, "role", "?")
        self._timeout = timeout_s
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._stamp = 0.0
        self._wall = 0.0
        self._seen: dict[int, int] = {}
        self.last_stamp: float | None = None  # capture time of the last read_after() frame
        self._running = True
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True, name=f"fanout-{self.role}")
        self._thread.start()

    def _next(self, after: float):
        """``(ok, frame, captured)``: the camera's next frame and when its scene was captured.

        A camera that knows the capture time (the twin's) says so in
        ``last_stamp``; otherwise it is the arrival time.
        """
        if callable(getattr(self.inner, "next_frame", None)):
            return self.inner.next_frame(after)
        ok, frame = self.inner.read()
        captured = getattr(self.inner, "last_stamp", None)
        return ok, frame, time.time() if captured is None else float(captured)

    def _pump(self) -> None:
        after = 0.0
        while self._running:
            try:
                ok, frame, captured = self._next(after)
            except Exception as e:
                logger.warning("[console] %s camera read failed: %s", self.role, e)
                ok, frame = False, None
            if ok and frame is not None:
                after = captured
                with self._cond:
                    self._frame, self._stamp, self._wall = frame, time.monotonic(), captured
                    self._seq += 1
                    self._cond.notify_all()
                continue
            retry = getattr(self.inner, "retry_at", None)
            retry_at = retry() if callable(retry) else None
            if retry_at is None:
                return  # closed, or a camera that will not give frames again
            self._closed.wait(max(retry_at - time.time(), 0.0))  # the camera's own reopen schedule

    def read(self) -> tuple[bool, np.ndarray | None]:
        tid = threading.get_ident()
        with self._cond:
            if not self._cond.wait_for(
                lambda: self._seq > self._seen.get(tid, 0), timeout=self._timeout
            ):
                return False, None
            self._seen[tid] = self._seq
            return True, self._frame

    def read_after(self, settled_at: float):
        """Next pumped frame captured (or, without a capture stamp, arrived) after ``settled_at``."""
        deadline = time.monotonic() + max(self._timeout, 0.5)
        with self._cond:
            while self._frame is None or self._wall <= settled_at:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=remaining)
            self._seen[threading.get_ident()] = self._seq
            self.last_stamp = self._wall
            return True, self._frame

    def latest(self) -> np.ndarray | None:
        """Newest frame without blocking, or None if the stream is stale."""
        with self._cond:
            if self._frame is None or time.monotonic() - self._stamp > self._timeout:
                return None
            return self._frame

    def ok(self) -> bool:
        return self.latest() is not None

    def close(self) -> None:
        """Stop the reader. A camera this fan-out owns is closed, which wakes a blocked read."""
        self._running = False
        self._closed.set()
        if self._own:
            close = getattr(self.inner, "close", None) or getattr(self.inner, "release", None)
            if close:
                close()
        self._thread.join(timeout=1.0)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class ModelTap:
    """Wrap ``obj.method`` in place: serialise calls and keep the last result.

    Patching the instance (not wrapping the object) also catches internal
    calls such as ``detector.find`` -> ``self.infer``.
    """

    def __init__(self, obj, method: str):
        self.obj = obj
        self.last = None
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._inner = getattr(obj, method)

        def tapped(*args, **kwargs):
            result = self.call(*args, **kwargs)
            self._stamp = time.monotonic()
            return result

        setattr(obj, method, tapped)

    def call(self, *args, **kwargs):
        """Run the model for the console; doesn't count as a behavior using it."""
        with self._lock:
            result = self._inner(*args, **kwargs)
        self.last = result
        return result

    def idle(self) -> bool:
        """True while no behavior has called the model recently."""
        return time.monotonic() - self._stamp > IDLE_AFTER_S



def _view_source(cfg: dict):
    """The twin's scene camera on ``cameras.scene.topic``."""
    if (cfg.get("robot") or {}).get("backend") != "sim":
        return None
    try:
        if (cfg.get("robot") or {}).get("use_ros2"):
            from vla_pipeline.vision.camera import Ros2Camera

            topic = ((cfg.get("cameras") or {}).get("scene") or {}).get(
                "topic", "/cameras/scene/image_raw"
            )
            camera = Ros2Camera("view", topic, stale_after_s=1.0)
            camera.topic = topic
            return camera
        logger.warning("[console] no 3D view: the twin view is the scene camera topic")
        return None
    except Exception as e:
        logger.warning("[console] no 3D view: %s", e)
        return None


def _fit(img: np.ndarray | None, w: int, h: int, label: str = "no signal") -> np.ndarray:
    if img is None:
        out = np.zeros((h, w, 3), np.uint8)
        cv2.putText(out, label, (w // 2 - 60, h // 2), FONT, 0.6, DIM, 1, cv2.LINE_AA)
        return out
    if img.shape[1] != w or img.shape[0] != h:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img


def _title(img: np.ndarray, text: str) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 17), FONT, 0.5, WHITE, 1, cv2.LINE_AA)


def _bar(img, x, y, w, h, frac, color, mark: float | None = None) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), DIM, 1)
    fill = int(np.clip(frac, 0.0, 1.0) * (w - 2))
    if fill > 0:
        cv2.rectangle(img, (x + 1, y + 1), (x + 1 + fill, y + h - 1), color, -1)
    if mark is not None:
        mx = x + int(np.clip(mark, 0.0, 1.0) * w)
        cv2.line(img, (mx, y - 3), (mx, y + h + 3), WHITE, 1)


PROCESSOR = {"VitisAIExecutionProvider": "NPU", "CPUExecutionProvider": "CPU"}


def provider_name(model, default: str = "CPU") -> str:
    """The processor a model runs on, for a tile title: NPU, CPU, or the provider's name.

    From the model's ``provider``, else its ONNX ``session``.
    """
    p = getattr(model, "provider", None)
    session = getattr(model, "session", None)
    if p is None and session is not None:
        p = session.get_providers()[0]
    p = str(p or default)
    return PROCESSOR.get(p, p.replace("ExecutionProvider", ""))


def jpeg(img: np.ndarray, quality: int = 80) -> bytes:
    """``img`` as JPEG bytes, what a notebook ``widgets.Image`` shows."""
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


class TwinConsole:
    """Observe a :class:`~vla_pipeline.user_robot_control.PipelineContext` and draw it in one image.

    Construction fans the context's cameras out, so a behavior and the
    console both get every frame, and routes ``ctx.say`` into the voice panel.
    ``view`` is the 3D view (default: the twin's, from the config);
    ``restore()`` parks the twin and puts the props back, True when it did
    (default: through the arm client).
    """

    def __init__(self, ctx, voice: str = "mic", view=None, hint: str = "space: stop    q: quit",
                 restore=None):
        self.ctx = ctx
        self.voice = voice  # "mic" | "typed"
        self.hint = hint
        self._restore = restore
        cfg = ctx.cfg
        if ctx.mount_camera is not None:
            ctx.mount_camera = FanoutCamera(ctx.mount_camera)
        if ctx.arm_camera is not None:
            ctx.arm_camera = FanoutCamera(ctx.arm_camera)
        self.pose = ModelTap(ctx.pose, "infer") if ctx.pose is not None else None
        self.hands = ModelTap(ctx.hands, "process") if ctx.hands is not None else None
        self.detector = (
            ModelTap(ctx.detector, "infer") if ctx.detector is not None else None
        )
        self.view = view if view is not None else _view_source(cfg)

        self._log: deque[tuple[str, tuple]] = deque(maxlen=LOG_LINES)
        self._lock = threading.Lock()
        self._level, self._speaking, self._level_t = 0.0, False, 0.0
        self._asr_pending = False
        self._behavior: str | None = None
        self._behavior_t0 = 0.0
        self._stop_flash = 0.0
        self._resetting = False
        self._tiles = None
        self._joints = None
        self._gripper_load = None  # milliamps
        self._on_reset = None
        self._say = ""
        self._heard = ""
        self._log_seq = 0
        # An arm in this process (the notebook) feeds apply_joints directly.
        # The window process has no arm; ConsoleLink feeds it.
        if ctx.arm is not None and hasattr(ctx.arm, "on_cached_joints"):
            ctx.arm.on_cached_joints = self.apply_joints

        say = ctx.say

        def _say(message: str) -> None:
            self._add(f"robot: {message}", GREEN)
            say(message)

        ctx.say = _say
        # Behaviors draw into this window, never their own.
        ctx.headless = True

    # ------------------------------------------------------------ inputs

    def attach(self, robot_voice) -> None:
        self._robot_voice = robot_voice
        robot_voice.listeners.append(self.on_event)

    def _restore_twin(self) -> bool:
        """Park the twin and restore its props. True when this process can."""
        if self._restore is not None:
            return bool(self._restore())
        restore = getattr(self.ctx.arm, "reset", None)
        return bool(restore and restore())

    @property
    def resetting(self) -> bool:
        """True from the click until the arm is parked and the props are back."""
        with self._lock:
            return self._resetting

    def use_tiles(self) -> None:
        """Draw published overlays instead of running the models in this process."""
        self._tiles = {"mount": None, "hands": None, "wrist": None}

    def set_tile(self, name: str, frame) -> None:
        """The latest overlay for ``name`` (mount, hands, or wrist)."""
        with self._lock:
            if self._tiles is not None:
                self._tiles[name] = frame

    def apply_state(self, state) -> None:
        """Paint one latched control state. The console does not ask the arm."""
        say = heard = logged = None
        with self._lock:
            if state.say and state.say != self._say:
                self._say = state.say
                say = state.say
            if state.heard and state.heard != self._heard:
                self._heard = state.heard
                heard = state.heard
            if state.log_seq > getattr(self, "_log_seq", 0) and state.log_text:
                self._log_seq = state.log_seq
                logged = (state.log_text, state.log_kind)
            if state.behavior == "idle":
                self._behavior = None
            elif state.behavior != self._behavior:
                self._behavior = state.behavior
                self._behavior_t0 = time.monotonic()
            self._resetting = state.resetting
        if heard:
            self._add(f"you: {heard}", WHITE)
        if logged:
            text, kind = logged
            if kind == "stop":
                self._stop_flash = time.monotonic()
            self._add(text, _LINE_COLOR.get(kind, GRAY))
        if say:
            self._add(f"robot: {say}", GREEN)

    def apply_joints(self, joints: dict, gripper_load: float | None = None) -> None:
        """The latest joint sample and gripper current in milliamps. Neither adds a voice-log line."""
        with self._lock:
            self._joints = dict(joints)
            self._gripper_load = gripper_load

    def reset_scene(self, on_busy=None, on_done=None) -> None:
        """Stop the behavior, park the arm with the jaws open, and restore the props.

        A second call while this one runs does nothing. ``on_busy`` runs once
        the button should show the reset in progress, before the arm moves, so
        the window can redraw. ``on_done`` runs after the arm is parked, while
        a new reset is still refused.

        When this console only draws, the call publishes a reset and returns.
        The control process coalesces a second click into one extra pass.
        """
        if getattr(self, "_on_reset", None) is not None:
            self._on_reset()
            return
        with self._lock:
            if self._resetting:
                return
            self._resetting = True
        try:
            if on_busy is not None:
                on_busy()
            self._run_reset()
            if on_done is not None:
                on_done()
        finally:
            with self._lock:
                self._resetting = False

    def _run_reset(self) -> None:
        """The reset itself. ``reset_scene`` is the only caller."""
        ctx = self.ctx
        robot_voice = getattr(self, "_robot_voice", None)
        holding = getattr(robot_voice, "holding", None)
        if holding is not None:
            holding.set()
        ctx.stop_event.set()
        if robot_voice is not None:
            robot_voice.discard()
            robot_voice.wait_idle(3.0)  # the running behavior gets 3 s to stop
            robot_voice.discard()
        try:
            state = getattr(ctx, "shared_state", None)
            if isinstance(state, dict):
                from vla_pipeline.behaviors.manipulation import forget_scene

                forget_scene(state)
            if not self._restore_twin():
                ctx.say("Reset is only available on the digital twin.")
                return
            ctx.say(RESET_LINE)
        except Exception as exc:
            logger.exception("reset")
            ctx.say(f"Reset failed: {exc}")
        finally:
            if robot_voice is not None:
                robot_voice.discard()
            ctx.stop_event.clear()
            if holding is not None:
                holding.clear()

    def on_level(self, rms: float, speaking: bool) -> None:
        """Microphone block level from the voice session."""
        with self._lock:
            self._level, self._speaking, self._level_t = rms, speaking, time.monotonic()

    def note_asr(self, busy: bool) -> None:
        """True while the voice model is decoding. The meter keeps running either way."""
        with self._lock:
            self._asr_pending = busy

    def on_event(self, event: str, **data) -> None:
        if event == "heard":
            self._asr_pending = False
            self._add(f"you: {data['text']}", WHITE)
        elif event == "behavior_start":
            self._behavior, self._behavior_t0 = data["intent"], time.monotonic()
        elif event == "behavior_end":
            self._behavior = None
            line = panel_line(event, took=time.monotonic() - self._behavior_t0, **data)
            if line is not None:
                text, kind = line
                self._add(text, _LINE_COLOR.get(kind, GRAY))
        else:
            line = panel_line(event, **data)
            if line is not None:
                text, kind = line
                if kind == "stop":
                    self._stop_flash = time.monotonic()
                self._add(text, _LINE_COLOR.get(kind, GRAY))

    def _add(self, text: str, color: tuple) -> None:
        lines = textwrap.wrap(text, 46, subsequent_indent="    ") or [""]
        with self._lock:
            self._log.extend((line, color) for line in lines)

    # ------------------------------------------------------------ tiles

    def _pose_tile(self, frame):
        if frame is None or self.pose is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.pose.idle():
            self.pose.call(frame)
        if self.pose.last is None:
            return _fit(frame, TILE_W, TILE_H)
        boxes, scores, kpts = self.pose.last
        return _fit(self.ctx.pose.draw(frame, boxes, scores, kpts), TILE_W, TILE_H)

    def _hands_tile(self, frame):
        if frame is None or self.hands is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.hands.idle():
            self.hands.call(frame)
        if self.hands.last is None:
            return _fit(frame, TILE_W, TILE_H)
        obs, results = self.hands.last
        return _fit(self.ctx.hands.draw(frame, obs, results), TILE_W, TILE_H)

    def _detect_tile(self, frame):
        if frame is None or self.detector is None:
            return _fit(frame, TILE_W, TILE_H)
        if self.detector.idle():
            self.detector.call(frame)
        dets = self.detector.last or []
        return _fit(self.ctx.detector.draw(frame, dets), TILE_W, TILE_H)

    def _view_tile(self):
        frame = None
        if self.view is not None:
            ok, frame = self.view.read()
            frame = frame if ok else None
        topic = getattr(self.view, "topic", None) if self.view is not None else None
        label = f"waiting for {topic}" if topic else ("waiting for the twin" if self.view is not None else "no 3D view")
        img = _fit(frame, VIEW_W, VIEW_H, label).copy()
        _title(img, VIEW)
        return img

    def _panel(self) -> np.ndarray:
        p = np.full((VIEW_H, PANEL_W, 3), 28, np.uint8)
        now = time.monotonic()
        y = 22
        for line in VOICE:
            cv2.putText(p, line, (12, y), FONT, 0.45, WHITE, 1, cv2.LINE_AA)
            y += 18
        with self._lock:
            level, speaking, level_t = self._level, self._speaking, self._level_t
            asr, log = self._asr_pending, list(self._log)
            resetting = self._resetting
        if self.voice == "typed":
            state, color = "typed commands (stdin)", GRAY
        elif level_t == 0.0 or now - level_t > 2.0:
            state, color = "no microphone signal", RED
        elif speaking:
            state, color = "hearing you", GREEN
        elif asr:
            state, color = "transcribing...", YELLOW
        else:
            state, color = "listening", GRAY
        cv2.putText(p, state, (12, y), FONT, 0.5, color, 1, cv2.LINE_AA)
        if self.voice == "mic":
            db = 20.0 * np.log10(max(level, 1e-6))
            _bar(p, 12, y + 12, PANEL_W - 24, 10, (db + 60.0) / 50.0, GREEN if speaking else DIM)
        y += 52

        if now - self._stop_flash < 1.0:
            cv2.rectangle(p, (0, y - 18), (PANEL_W, y + 8), RED, -1)
            cv2.putText(p, "STOP", (12, y), FONT, 0.6, WHITE, 2, cv2.LINE_AA)
        elif self._behavior:
            text = f"running: {self._behavior}  {now - self._behavior_t0:4.1f}s"
            cv2.putText(p, text, (12, y), FONT, 0.55, CYAN, 1, cv2.LINE_AA)
        else:
            cv2.putText(p, "idle - say a command", (12, y), FONT, 0.55, GRAY, 1, cv2.LINE_AA)
        y += 22
        cv2.line(p, (12, y), (PANEL_W - 12, y), DIM, 1)
        y += 24

        for text, color in log:
            cv2.putText(p, text, (12, y), FONT, 0.45, color, 1, cv2.LINE_AA)
            y += 20

        self._arm_hud(p, VIEW_H - 110)
        bx, by, bw, bh = RESET_BUTTON
        # RESET_BUTTON is in the composed image; this panel is its right half.
        local = (bx - VIEW_W, by - TILE_H, bw, bh)
        # Gray "resetting" while a reset is running; orange "reset" when it is free.
        label, scale, fill, ink = (
            ("resetting", 0.5, (70, 70, 70), WHITE)
            if resetting
            else ("reset", 0.55, ORANGE, (20, 20, 20))
        )
        cv2.rectangle(p, (local[0], local[1]), (local[0] + bw, local[1] + bh), fill, -1)
        (tw, th), _ = cv2.getTextSize(label, FONT, scale, 1)
        cv2.putText(
            p, label,
            (local[0] + (bw - tw) // 2, local[1] + (bh + th) // 2),
            FONT, scale, ink, 1, cv2.LINE_AA,
        )
        cv2.putText(p, self.hint, (local[0] + bw + 12, local[1] + 21), FONT, 0.42, DIM, 1, cv2.LINE_AA)
        return p

    def _arm_hud(self, p, y) -> None:
        """The gripper-current bar in milliamps, full at stall, then the joint positions.

        Both come only from :meth:`apply_joints`. ``n/a`` means no sample has
        carried a gripper current yet. The tick is the contact level.
        """
        with self._lock:
            load, joints = getattr(self, "_gripper_load", None), getattr(self, "_joints", None)
        cv2.line(p, (12, y - 16), (PANEL_W - 12, y - 16), DIM, 1)
        if load is None:
            cv2.putText(p, "gripper current: n/a", (12, y), FONT, 0.45, GRAY, 1, cv2.LINE_AA)
        else:
            grasp = load >= GRIPPER_CONTACT_MA
            text = f"gripper {load:4.0f} / {GRIPPER_STALL_MA:.0f} mA"
            cv2.putText(p, text, (12, y), FONT, 0.45, GREEN if grasp else WHITE, 1, cv2.LINE_AA)
            _bar(
                p, 12, y + 8, PANEL_W - 24, 8, load / GRIPPER_STALL_MA,
                GREEN if grasp else GRAY, mark=GRIPPER_CONTACT_MA / GRIPPER_STALL_MA,
            )
        if joints:
            parts = [f"{MOTOR_SHORT.get(n, n)} {v:+.0f}" for n, v in joints.items()]
            for i in range(0, len(parts), 3):
                cv2.putText(p, "   ".join(parts[i : i + 3]), (12, y + 36 + 18 * (i // 3)), FONT, 0.42, GRAY, 1, cv2.LINE_AA)

    def _webcam_tiles(self, frame) -> list[np.ndarray]:
        pose, hands = self._pose_tile(frame).copy(), self._hands_tile(frame).copy()
        _title(pose, with_processor(POSE, provider_name(self.pose.obj) if self.pose else "off"))
        _title(hands, with_processor(HANDS, provider_name(self.hands.obj) if self.hands else "off"))
        return [pose, hands]

    def webcam_tiles(self) -> np.ndarray:
        """The two webcam tiles, YOLO pose and MediaPipe hands, side by side."""
        frame = self.ctx.mount_camera.latest() if self.ctx.mount_camera is not None else None
        return np.hstack(self._webcam_tiles(frame))

    def compose(self) -> np.ndarray:
        """One 1200 x 1100 BGR frame of the whole console."""
        with self._lock:
            tiles = getattr(self, "_tiles", None)
            tiles = None if tiles is None else dict(tiles)
        if tiles is not None:
            pose = _fit(tiles.get("mount"), TILE_W, TILE_H).copy()
            hands = _fit(tiles.get("hands"), TILE_W, TILE_H).copy()
            detect = _fit(tiles.get("wrist"), TILE_W, TILE_H).copy()
            _title(pose, POSE)
            _title(hands, HANDS)
            _title(detect, DETECT)
            top = np.hstack([pose, hands, detect])
            bottom = np.hstack([self._view_tile(), self._panel()])
            return np.vstack([top, bottom])
        mount_frame = self.ctx.mount_camera.latest() if self.ctx.mount_camera is not None else None
        armcam = self.ctx.arm_camera.latest() if self.ctx.arm_camera is not None else None
        detect = self._detect_tile(armcam).copy()
        _title(detect, with_processor(DETECT, provider_name(self.detector.obj) if self.detector else "off"))
        top = np.hstack([*self._webcam_tiles(mount_frame), detect])
        bottom = np.hstack([self._view_tile(), self._panel()])
        return np.vstack([top, bottom])

    def close(self) -> None:
        """Stop the arm's feed, the camera fan-out threads (closing the cameras they wrap) and the 3D view."""
        arm = self.ctx.arm
        if arm is not None and getattr(arm, "on_cached_joints", None) == self.apply_joints:
            arm.on_cached_joints = None
        for cam in (self.ctx.mount_camera, self.ctx.arm_camera):
            if isinstance(cam, FanoutCamera):
                cam.close()
        if self.view is not None:
            (getattr(self.view, "close", None) or self.view.release)()

    # ------------------------------------------------------------ loop

    def run(self, shutdown: threading.Event, on_stop, on_reset=None) -> None:
        """Draw until ``shutdown`` is set or the window is closed.

        Space calls ``on_stop()``. The reset button and ``r`` both call
        :meth:`reset_scene`: park the arm and put the props back. The button
        stays gray until that returns, and a click or ``r`` during it is dropped.
        """
        if on_reset is not None:
            self._on_reset = on_reset
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        scale = min(1.0, 950 / (TILE_H + VIEW_H))  # fit a 1080p screen; drag to resize
        cv2.resizeWindow(WINDOW, int(TILE_W * 3 * scale), int((TILE_H + VIEW_H) * scale))
        clicked = {"reset": False}

        def on_mouse(event, x, y, flags, _param) -> None:
            if event != cv2.EVENT_LBUTTONDOWN or (self.resetting and getattr(self, "_on_reset", None) is None):
                return
            rect = cv2.getWindowImageRect(WINDOW)
            if pointer_hits_reset(x, y, (TILE_W * 3, TILE_H + VIEW_H), rect):
                clicked["reset"] = True

        def show_busy() -> None:
            cv2.imshow(WINDOW, self.compose())
            cv2.waitKey(1)

        def drop_clicks_from_the_reset() -> None:
            # One pump, while resetting is still true, so a click that landed
            # during the reset is delivered here and the callback drops it.
            cv2.waitKey(1)
            clicked["reset"] = False

        cv2.setMouseCallback(WINDOW, on_mouse)
        period = 1.0 / FPS
        try:
            while not shutdown.is_set():
                t0 = time.monotonic()
                try:
                    cv2.imshow(WINDOW, self.compose())
                except Exception:
                    logger.exception("[console] frame failed")
                key = cv2.waitKey(max(1, int((period - (time.monotonic() - t0)) * 1e3))) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    on_stop()
                elif key in (ord("r"), ord("R")) or clicked["reset"]:
                    clicked["reset"] = False
                    self.reset_scene(on_busy=show_busy, on_done=drop_clicks_from_the_reset)
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            shutdown.set()
            cv2.destroyWindow(WINDOW)
            if self.view is not None:
                (getattr(self.view, "close", None) or self.view.release)()
                self.view = None
