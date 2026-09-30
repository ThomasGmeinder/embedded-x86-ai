# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Pinch-to-tele-op: your hand flies an SO-101 arm to pick & place a cube.

The strongest "Physical AI" story on one chip: the NPU sees your hand, the CPU
turns it into a robot command, the iGPU runs the robot. Nothing in the cloud.

    Thread A (NPU)   webcam -> YOLO26-pose (anchor) -> hand-landmark model (21 pts)
                     -- BOTH neural nets run on the AIE, back-to-back
    Thread B (CPU)   21 landmarks -> palm position + pinch -> end-effector target
                     -- pure vector math + smoothing, no neural net, no IK here
    Thread C (iGPU)  MuJoCo SO-101: IK solves to the target, pinch grabs the cube

Move your hand to fly the gripper; lower onto the cube and pinch to grab, carry
it, open to drop it on the green target circle. The exact same command stream
would drive a real SO-101 — this is teleoperation, simulated.
"""

from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np

from .yolo_pipeline import preprocess_frame
from .pose_common import (
    postprocess_pose,
    draw_pose_colored,
    make_frame_source,
    _OneEuro,
    _put_dropping,
)
from .hand_landmarks import HandLandmarker, track_roi, roi_from_wrist
from .hand_to_arm import TeleopMapper, PINCH_GRAB, PINCH_RELEASE, DEPTH_SENSITIVITY
from .arm_teleop_sim import (
    ArmTeleopRenderer,
    X_REACH,
    X_NEAR,
    X_FAR,
    Z_TOP,
    Z_BOTTOM,
)
from .resource_monitor import NPU_ACTIVITY
from . import pipeline_registry

_WRIST = {"left": 9, "right": 10}
_ELBOW = {"left": 7, "right": 8}
_HAND_BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]

# Hold the grip through this many frames of lost hand tracking before releasing,
# so a short detection dropout doesn't drop the carried cube. Raised from 5:
# BlazeHand drops out more often under bright/uneven lighting (the workshop
# room), and a held cube surviving a longer blink reads as far more reliable.
_GRIP_GRACE = 15
# Coast the last hand OVERLAY through this many lost frames before blanking the
# skeleton + showing "show your hand". Raised from 3 so brief dropouts don't
# make the HUD flicker. NOTE: the end-effector TARGET is held INDEFINITELY while
# the hand is gone (see _cpu_loop) — the arm waits exactly where you left it and
# only moves again on a fresh detection, so it never snaps back to neutral.
_TARGET_GRACE = 12


class ArmTeleopPipeline:
    def __init__(
        self,
        session,
        frame_source,
        on_frame,
        conf=0.4,
        mirror=True,
        render_size=600,
        score_thresh=0.30,
        min_cutoff=1.5,
        beta=0.05,
        hand_session=None,
        feed_npu_gauge=True,
        control_hand="auto",
        min_person_area_frac=0.06,
        depth_sensitivity=None,
        enable_depth=False,
        randomize_target=True,
        x_near=None,
        x_far=None,
        z_top=None,
        z_bottom=None,
        on_stats=None,
    ):
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._frame_source = frame_source
        self._on_frame = on_frame
        self._on_stats = on_stats
        self._conf = conf
        self._mirror = mirror
        self._render_size = render_size
        self._score_thresh = score_thresh
        # Which hand drives the arm: "auto" = whichever hand is raised highest
        # (re-selectable mid-demo by raising the other hand), or "left"/"right"
        # to pin to one side of the (mirrored) video. Set live from the UI toggle.
        self._control_hand = control_hand
        # Presenter selection: pick the LARGEST person box (closest = fills the
        # most frame), and ignore any box smaller than this fraction of the frame
        # so background attendees in a crowded room are never locked onto.
        self._min_person_area_frac = float(min_person_area_frac)
        # Only feed the dashboard's NPU gauge when perception actually runs on the
        # NPU. On a CPU run we leave the gauge alone so it correctly shows the NPU
        # idle (otherwise the CPU inference times would masquerade as NPU work).
        self._feed_npu_gauge = feed_npu_gauge

        # When a VitisAI hand_session is passed (as the notebook does), the 21-pt
        # landmark model also runs on the AIE, so all perception is on the NPU and
        # the CPU thread is left with just the target/pinch math. With the default
        # hand_session=None, HandLandmarker builds a CPU session instead.
        self._hands = HandLandmarker(session=hand_session)
        # Reach (X) gets its own stronger smoothing (lower min_cutoff/beta) so the
        # opt-in depth axis is far less jumpy; lateral (Y) and height (Z) keep the
        # snappier default. When depth is off, reach is constant so this is inert.
        self._fx = _OneEuro(0.7, 0.03)
        self._fy = _OneEuro(min_cutoff, beta)
        self._fz = _OneEuro(min_cutoff, beta)
        # 3D reach box shared by the hand mapper and the renderer's reach test
        # (None -> the module defaults in arm_teleop_sim/hand_to_arm).
        self._x_near = X_NEAR if x_near is None else float(x_near)
        self._x_far = X_FAR if x_far is None else float(x_far)
        self._z_top = Z_TOP if z_top is None else float(z_top)
        self._z_bottom = Z_BOTTOM if z_bottom is None else float(z_bottom)
        self._depth_sensitivity = (
            DEPTH_SENSITIVITY if depth_sensitivity is None else float(depth_sensitivity)
        )
        # Reach (depth) is OFF by default: the reliable demo is lateral + height
        # (both clean image-plane axes). enable_depth=True opts into the
        # experimental in/out reach from the hand's apparent size.
        self._enable_depth = bool(enable_depth)
        # Target randomisation for the renderer: True (default) relocates the
        # green target each round for a live webcam demo; False pins it to one
        # fixed reachable spot so a pre-recorded clip's canned motion always
        # lands. Auto-derived from the source type by the notebook / standalone
        # entry points (video -> fixed, webcam -> moving).
        self._randomize_target = bool(randomize_target)
        # Maps hand landmarks -> end-effector target (lateral Y, height Z, and —
        # when enable_depth — auto-calibrated reach X); holds neutral-span state.
        self._mapper = TeleopMapper(
            x_near=self._x_near,
            x_far=self._x_far,
            z_top=self._z_top,
            z_bottom=self._z_bottom,
            depth_sensitivity=self._depth_sensitivity,
            enable_depth=self._enable_depth,
        )
        self._prev_lm_xy = None
        self._tracked_wi = None  # wrist keypoint idx (9/10) currently tracked
        self._target = None  # last smoothed hand-body target (held when lost)
        self._last_lm_xy = None  # last hand overlay (coasted through brief dropouts)
        self._last_pinch = 1.0  # last pinch value (coasted through brief dropouts)
        self._grip_latched = False  # debounced grip state (pinch hysteresis)
        self._lost_frames = 0  # consecutive frames with no hand detection

        self._pose_q: Queue = Queue(maxsize=1)
        self._latest = None
        self._stop = threading.Event()
        self._reset = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._npu_fps = self._cpu_ms = self._igpu_fps = self._cpu_fps = 0.0
        self._dashboard = None  # optional LiveDashboard shown under the demo

    def reset_object(self):
        self._reset.set()

    def set_control_hand(self, mode: str):
        """Switch the controlling hand live ("auto" | "left" | "right").

        Clears the current hand lock so the next frame re-acquires on the newly
        chosen side without the user having to pull their hand out of frame, and
        recaptures the depth neutral so reach re-calibrates to the new hand."""
        mode = str(mode).lower()
        if mode not in ("auto", "left", "right"):
            return
        self._control_hand = mode
        self._prev_lm_xy = None
        self._tracked_wi = None
        self._mapper.reset()

    def start(self):
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._npu_loop, name="npu-pose", daemon=True),
            threading.Thread(target=self._cpu_loop, name="cpu-hand", daemon=True),
            threading.Thread(target=self._render_loop, name="igpu-arm", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=3.0)
        self._threads = []
        if self._dashboard is not None:
            try:
                self._dashboard.stop()
            except Exception:
                pass

    # -- NPU: pose (wrist anchor + skeleton) AND hand landmarks ---------
    def _npu_loop(self):
        """Both neural nets run here, back-to-back on the one AIE: YOLO26-pose
        finds the person/hand anchor, then the 21-pt hand-landmark model runs on
        the cropped ROI. Keeping the hand detection in this thread means its time
        is counted as NPU work (not CPU) and ``_prev_lm_xy`` is owned by a single
        thread (no cross-thread tracking race)."""
        ema, last = 0.0, time.perf_counter()
        while not self._stop.is_set():
            frame = self._frame_source()
            if frame is None:
                self._stop.set()
                break
            if self._mirror:
                frame = cv2.flip(frame, 1)
            t0 = time.perf_counter()
            nchw, info = preprocess_frame(frame)
            outputs = self._session.run(None, {self._input_name: nchw})
            boxes, scores, kpts = postprocess_pose(
                outputs, frame.shape[:2], info, self._conf
            )
            pose_ms = (time.perf_counter() - t0) * 1000.0
            best, best_box, best_score = self._select_person(
                boxes, scores, kpts, frame.shape[:2]
            )

            # hand landmarks on the NPU (cropped ROI), timed as NPU work
            t1 = time.perf_counter()
            roi = self._acquire_roi(frame, best)
            det = (
                self._hands.detect(frame, roi, self._score_thresh)
                if roi is not None
                else None
            )
            self._prev_lm_xy = det[1] if det is not None else None
            hand_ms = (time.perf_counter() - t1) * 1000.0

            if self._feed_npu_gauge:
                NPU_ACTIVITY.record(
                    pose_ms + hand_ms
                )  # feed the live dashboard NPU gauge
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            ema = inst if ema == 0 else 0.2 * inst + 0.8 * ema
            with self._lock:
                self._npu_fps = ema
            _put_dropping(
                self._pose_q,
                (frame, best, best_box, best_score, roi, det, pose_ms, hand_ms),
            )

    def _select_person(self, boxes, scores, kpts, frame_shape):
        """Pick the presenter = the LARGEST person box (closest to the camera).

        In a crowded room a fully-visible background attendee can out-score the
        (often partly-cropped) presenter, so confidence is the wrong signal —
        proximity is. We also drop any box smaller than ``min_person_area_frac``
        of the frame so distant onlookers are ignored entirely."""
        if not len(scores):
            return None, None, 0.0
        fh, fw = frame_shape
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        if float(areas[i]) < self._min_person_area_frac * float(fw * fh):
            return None, None, 0.0
        return kpts[i], boxes[i], float(scores[i])

    def _pick_wrist(self, best):
        """Choose which wrist to drive from, honouring the control-hand mode.

        Returns ``(wrist_idx, elbow_idx)`` or ``None``. In "auto" we take the
        highest (raised) confident wrist; in "left"/"right" we take the wrist on
        that side of the (mirrored, selfie-view) frame — spatial side, so it
        matches what the user sees regardless of YOLO's anatomical L/R labels."""
        if best is None:
            return None
        cands = [
            (wi, ei)
            for wi, ei in (
                (_WRIST["left"], _ELBOW["left"]),
                (_WRIST["right"], _ELBOW["right"]),
            )
            if best[wi, 2] >= 0.2
        ]
        if not cands:
            return None
        if self._control_hand == "auto":
            return min(cands, key=lambda c: float(best[c[0], 1]))  # highest wrist
        if self._control_hand == "right":
            return max(cands, key=lambda c: float(best[c[0], 0]))  # right of frame
        return min(cands, key=lambda c: float(best[c[0], 0]))  # left of frame

    @staticmethod
    def _is_raised(best, wi, ei) -> bool:
        """True when the wrist is clearly above its elbow (an intentional raise)."""
        return float(best[wi, 1]) < float(best[ei, 1])

    def _acquire_roi(self, frame, best):
        h, w = frame.shape[:2]
        pick = self._pick_wrist(best)
        # Already tracking a hand: stay locked for stability, but allow switching
        # to the other hand the instant the user deliberately raises it (auto), or
        # whenever the fixed side is available — no need to pull the hand off-frame.
        if self._prev_lm_xy is not None:
            switch = (
                pick is not None
                and pick[0] != self._tracked_wi
                and (self._control_hand != "auto" or self._is_raised(best, *pick))
            )
            if not switch:
                return track_roi(self._prev_lm_xy, w, h)
        if pick is None:
            return None
        self._tracked_wi = pick[0]
        return roi_from_wrist(best[pick[0], :2], best[pick[1], :2], w, h)

    # -- CPU: landmarks -> end-effector target (vector math, no IK) -----
    def _cpu_loop(self):
        """Perception already happened on the NPU; this thread only turns the
        21 landmarks into an end-effector target + pinch (vector math + One-Euro
        smoothing). That's why the CPU time here is tiny."""
        ema, last = 0.0, time.perf_counter()
        while not self._stop.is_set():
            try:
                frame, best, best_box, best_score, roi, det, pose_ms, hand_ms = (
                    self._pose_q.get(timeout=0.2)
                )
            except Empty:
                continue
            now = time.perf_counter()
            inst = 1.0 / max(now - last, 1e-6)
            last = now
            ema = inst if ema == 0 else 0.2 * inst + 0.8 * ema
            t0 = time.perf_counter()
            h, w = frame.shape[:2]
            lm_xy = None
            pinch = 1.0
            if det is not None:
                lm_crop, lm_xy, _score = det
                tgt, pinch, _grip_raw = self._mapper.command(lm_crop, lm_xy, w, h)
                sx = self._fx(t0, tgt[0])
                sy = self._fy(t0, tgt[1])
                sz = self._fz(t0, tgt[2])
                self._target = np.array([sx, sy, sz])
                self._last_lm_xy = lm_xy
                self._last_pinch = pinch
                self._lost_frames = 0
                # pinch hysteresis: close below GRAB, only open above RELEASE.
                if pinch < PINCH_GRAB:
                    self._grip_latched = True
                elif pinch > PINCH_RELEASE:
                    self._grip_latched = False
            else:
                # Hand not detected this frame. HOLD-LAST-STATE: the end-effector
                # target (self._target) is deliberately NOT touched, so the arm
                # stays exactly where you last put it and only moves again on a
                # fresh detection — no snap back to neutral, no drift. This is the
                # single biggest reliability win for a flaky tracker (bright rooms,
                # motion blur): a dropped frame just means "stay put", not "jump".
                #
                # The OVERLAY (skeleton) and pinch coast for a slightly longer
                # window so the HUD doesn't flicker on brief dropouts, then blank
                # so a genuine loss still reads as "show your hand". The grip is
                # held longest so a carried cube survives a real blink.
                self._lost_frames += 1
                if self._lost_frames <= _TARGET_GRACE:
                    lm_xy = self._last_lm_xy
                    pinch = self._last_pinch
                else:
                    self._last_lm_xy = None
                if self._lost_frames > _GRIP_GRACE:
                    self._grip_latched = False
            grip = self._grip_latched
            cpu_ms = (time.perf_counter() - t0) * 1000.0
            with self._lock:
                self._cpu_ms = cpu_ms
                self._cpu_fps = ema
            self._latest = (
                frame,
                best,
                best_box,
                best_score,
                self._target,
                lm_xy,
                roi,
                pinch,
                grip,
                pose_ms,
                hand_ms,
                cpu_ms,
            )

    # -- iGPU: SO-101 render --------------------------------------------
    def _render_loop(self):
        renderer = ArmTeleopRenderer(
            width=self._render_size,
            height=self._render_size,
            x_near=self._x_near,
            x_far=self._x_far,
            z_top=self._z_top,
            z_bottom=self._z_bottom,
            randomize_target=self._randomize_target,
        )
        ema, last = 0.0, time.perf_counter()
        _target_dt = 1.0 / 30.0  # cap the DISPLAY push to ~30 fps (see note below)
        _last_push = 0.0
        try:
            while not self._stop.is_set():
                if self._reset.is_set():
                    renderer.reset_object()
                    self._reset.clear()
                latest = self._latest
                if latest is None:
                    time.sleep(0.01)
                    continue
                (
                    frame,
                    best,
                    best_box,
                    best_score,
                    target,
                    lm_xy,
                    roi,
                    pinch,
                    grip,
                    pose_ms,
                    hand_ms,
                    cpu_ms,
                ) = latest

                tgt = target if target is not None else np.array([X_REACH, 0.0, Z_TOP])
                rgb = renderer.render(tgt, grip, grab_radius=0.05)
                # Physics + render run EVERY iteration (keeps the sim realtime), but
                # throttle only the DISPLAY push to ~30 fps: pushing JPEGs into the
                # ipywidgets Image faster than the browser can show them backs up the
                # comm channel and makes the feed lag further behind each second.
                _now = time.perf_counter()
                if _now - _last_push < _target_dt:
                    continue
                _last_push = _now
                robot_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

                cam = frame.copy()
                if best_box is not None:
                    x1, y1, x2, y2 = best_box.astype(int).tolist()
                    # Dark-blue box + white filled label chip (matches the notebook default look).
                    cv2.rectangle(cam, (x1, y1), (x2, y2), (255, 42, 4), 2, cv2.LINE_AA)
                    label = f"person {best_score:.2f}"
                    (tw, th), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    outside = y1 - th - 3 >= 0
                    y2c = y1 - th - 3 if outside else y1 + th + 3
                    cv2.rectangle(
                        cam, (x1, y1), (x1 + tw, y2c), (255, 42, 4), -1, cv2.LINE_AA
                    )
                    ty = y1 - 2 if outside else y1 + th + 2
                    cv2.putText(
                        cam,
                        label,
                        (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                if best is not None:
                    draw_pose_colored(
                        cam, best
                    )  # ultralytics default palette, matches the notebook
                if roi is not None:
                    x0, y0, s = roi
                    cv2.rectangle(cam, (x0, y0), (x0 + s, y0 + s), (0, 220, 255), 2)
                if lm_xy is not None:
                    self._draw_hand(cam, lm_xy)
                hh = self._render_size
                cam = cv2.resize(cam, (int(cam.shape[1] * hh / cam.shape[0]), hh))
                composite = cv2.hconcat([cam, robot_bgr])

                now = time.perf_counter()
                inst = 1.0 / max(now - last, 1e-6)
                last = now
                ema = inst if ema == 0 else 0.2 * inst + 0.8 * ema
                with self._lock:
                    self._igpu_fps = ema
                    npu_fps = self._npu_fps
                    cpu_fps = self._cpu_fps
                self._emit_stats(
                    npu_fps, pose_ms, hand_ms, cpu_ms, ema
                )  # HTML line above the feed
                self._draw_cues(
                    composite,
                    pinch,
                    grip,
                    renderer.grasped,
                    renderer.placed,
                    tracked=lm_xy is not None,
                )  # scene cues stay on video
                try:
                    self._on_frame(composite)
                except Exception:
                    pass
        finally:
            renderer.close()

    @staticmethod
    def _draw_hand(img, lm_xy):
        pts = lm_xy.astype(int)
        for a, b in _HAND_BONES:
            cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 180), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)

    def _emit_stats(self, npu_fps, pose_ms, hand_ms, cpu_ms, igpu_fps):
        """Perf readout as an HTML line above the feed (matches Sections 5 & 7)."""
        if self._on_stats is None:
            return
        self._on_stats(
            f"<b>NPU (pose+hand)</b> &nbsp; <span style='color:#0a8'>{npu_fps:.1f} fps</span>"
            f" &nbsp; ({pose_ms + hand_ms:.1f} ms) &nbsp;&nbsp;"
            f"<b>CPU</b> {cpu_ms:.1f} ms &nbsp;&nbsp; <b>iGPU</b> {igpu_fps:.1f} fps"
        )

    def _draw_cues(self, img, pinch, grip, grasped, placed, tracked):
        """Only scene-anchored cues stay painted on the video (pinch meter, state,
        the placed counter); the perf numbers live in the HTML line above."""
        if not tracked:
            cv2.putText(
                img,
                "show your hand",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                "show your hand",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (60, 160, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            # pinch meter: empty bar = open hand, fills as you pinch; green once gripping
            x0, y0, wbar = 10, 28, 180
            openness = float(np.clip((pinch - 0.15) / (0.7 - 0.15), 0, 1))
            cv2.rectangle(img, (x0, y0), (x0 + wbar, y0 + 16), (60, 60, 60), -1)
            col = (60, 255, 90) if grip else (200, 200, 60)
            cv2.rectangle(
                img, (x0, y0), (x0 + int(wbar * (1 - openness)), y0 + 16), col, -1
            )
            if grasped:
                state = "CARRYING"
            elif grip:
                state = "GRIP"
            else:
                state = "pinch to grab"
            cv2.putText(
                img,
                state,
                (x0 + wbar + 8, y0 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                col,
                2,
                cv2.LINE_AA,
            )

        # placed counter (bottom)
        H = img.shape[0]
        txt = f"placed: {placed}"
        cv2.putText(
            img,
            txt,
            (12, H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            img,
            txt,
            (12, H - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (90, 255, 140),
            2,
            cv2.LINE_AA,
        )


def _resolve_randomize_target(source, randomize_target=None):
    """Decide whether the SO-101 target should move each round or stay fixed.

    A live webcam (int ``source``) keeps the moving target — great for an
    interactive demo. A pre-recorded video (str path) needs a deterministic
    drop, so its target is pinned to a single fixed spot; otherwise the canned
    motion would never land on a relocating target. An explicit
    ``randomize_target`` (True/False) always overrides this auto-derivation."""
    if randomize_target is None:
        return not isinstance(source, str)
    return bool(randomize_target)


def run_arm_teleop_in_notebook(
    session,
    source=0,
    conf=0.4,
    mirror=True,
    render_size=600,
    show_dashboard=True,
    npu_infer_ms=20.0,
    hand_session=None,
    feed_npu_gauge=True,
    control_hand="auto",
    min_person_area_frac=0.06,
    depth_sensitivity=None,
    enable_depth=False,
    x_near=None,
    x_far=None,
    z_top=None,
    z_bottom=None,
    randomize_target=None,
):
    """ipywidgets UI: move your hand to fly the SO-101, pinch to grab & place the cube.

    By default your hand flies the gripper on two reliable image-plane axes:
    left/right (lateral) and up/down (height), each One-Euro smoothed for a
    controllable pick-and-place from a single camera. Set ``enable_depth=True``
    to opt into the experimental in/out reach axis (depth from the hand's
    apparent size, auto-calibrated at start) — it is jittery, cross-couples with
    the other axes, and buys nothing when the cube and target are at the same
    depth, so it is off by default. ``depth_sensitivity`` and the ``x_near``/
    ``x_far``/``z_top``/``z_bottom`` reach-box bounds are exposed for live tuning
    (None -> defaults).

    The green target circle relocates to a fresh reachable spot each round for a
    live webcam (``source`` is an int), which keeps an interactive demo varied.
    For a pre-recorded video (``source`` is a str path) this is turned OFF
    automatically so the canned motion always lands on one fixed, deterministic
    spot. Leave ``randomize_target=None`` for that auto behavior, or pass
    True/False to force it either way.

    When ``show_dashboard`` is True, a live resource dashboard is rendered right
    under the demo so all three compute units (NPU/CPU/iGPU) are visible at once.
    It starts and stops with the demo.
    """
    import ipywidgets as widgets
    from IPython.display import display

    pipeline_registry.stop_active_pipelines()  # don't fight a previous demo (fps!)

    # width:100% so the feed scales with the browser window (like the dashboard)
    img = widgets.Image(
        format="jpeg", layout=widgets.Layout(width="100%", height="auto")
    )
    stats = widgets.HTML()
    status = widgets.HTML(value="<b style='color:#2a8'>● running</b>")
    stop_btn = widgets.Button(description="Stop", button_style="danger", icon="stop")
    reset_btn = widgets.Button(
        description="Reset cube", button_style="info", icon="refresh"
    )

    def on_frame(bgr):
        ok, buf = cv2.imencode(".jpg", bgr)
        if ok:
            img.value = buf.tobytes()

    def on_stats(html):
        stats.value = html

    frame_source, cap = make_frame_source(source)
    # Live webcam (int) -> moving target; pre-recorded video (str path) -> fixed
    # target so its canned motion always lands. Silent unless overridden.
    randomize_target = _resolve_randomize_target(source, randomize_target)
    pipe = ArmTeleopPipeline(
        session,
        frame_source,
        on_frame,
        conf=conf,
        mirror=mirror,
        render_size=render_size,
        hand_session=hand_session,
        feed_npu_gauge=feed_npu_gauge,
        control_hand=control_hand,
        min_person_area_frac=min_person_area_frac,
        depth_sensitivity=depth_sensitivity,
        enable_depth=enable_depth,
        randomize_target=randomize_target,
        x_near=x_near,
        x_far=x_far,
        z_top=z_top,
        z_bottom=z_bottom,
        on_stats=on_stats,
    )
    children = [widgets.HBox([stop_btn, reset_btn, status]), stats, img]
    if show_dashboard:
        from .live_dashboard import LiveDashboard

        pipe._dashboard = LiveDashboard(
            window_seconds=60.0, sample_period=1.0, npu_infer_ms=npu_infer_ms
        )
        children.append(pipe._dashboard.widget)

    def on_stop(_b):
        pipe.stop()
        cap.release()
        stop_btn.disabled = True
        status.value = "<b style='color:#a33'>■ stopped</b>"

    stop_btn.on_click(on_stop)
    reset_btn.on_click(lambda _b: pipe.reset_object())
    display(widgets.VBox(children))
    pipe.start()
    if pipe._dashboard is not None:
        pipe._dashboard.start(auto_display=False)
    pipeline_registry.register(pipe, cap)
    return pipe
