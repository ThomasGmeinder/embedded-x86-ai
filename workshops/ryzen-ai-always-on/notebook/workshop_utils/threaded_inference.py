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

import queue
import threading
import time
from collections import deque

import cv2
import numpy as np

from .resource_monitor import NPU_ACTIVITY
from .pose_common import make_frame_source
from .hand_landmarks import HandLandmarker, roi_from_wrist, track_roi
from . import pipeline_registry


# COCO-17 wrist/elbow indices used to seed the hand ROI (same anchors the teleop
# NPU loop uses). Left/right are anatomical labels straight from YOLO26-pose.
_WRIST = {"left": 9, "right": 10}
_ELBOW = {"left": 7, "right": 8}
# BlazeHand 21-point skeleton (matches arm_teleop_mirror._HAND_BONES so the hand
# overlay looks identical in §8 and §9).
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


def draw_hand_landmarks(img, lm_xy):
    """Overlay the 21-point hand skeleton (same styling as the teleop demo)."""
    pts = lm_xy.astype(int)
    for a, b in _HAND_BONES:
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), (0, 255, 180), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)


class PoseHandPerception:
    """Pose-gated hand-landmark perception, overlay only.

    This is the shared perception recipe used by the teleop demo (§9), distilled
    to just what a live *overlay* needs: run YOLO26-pose, pick the largest
    person, crop a hand ROI around a raised wrist, and run the 21-point hand
    model on that crop. §8 uses this to show a SECOND model landing on the same
    NPU; §9's ArmTeleopPipeline does the equivalent inline plus the arm mapper.

    ``hand_session`` (a VitisAI session) runs the hand model on the NPU; pass
    None to skip hand detection entirely (single-model pose overlay). The
    per-hand landmark tracking box (``_prev_lm_xy``) is owned here so hand
    detection stays where the pose ran — no cross-thread tracking race.
    """

    def __init__(
        self,
        hand_session=None,
        score_thresh=0.30,
        min_person_area_frac=0.06,
        presence_getter=None,
    ):
        self._score_thresh = float(score_thresh)
        self._min_person_area_frac = float(min_person_area_frac)
        self._hands = (
            HandLandmarker(session=hand_session) if hand_session is not None else None
        )
        self._prev_lm_xy = None
        self._tracked_wi = None
        # Optional live hand-presence knob. When set, this callable is read PER
        # FRAME in detect_hand so a §8 slider can gate hand detections in real
        # time (analogous to the pose ``conf`` slider). Left None (the teleop
        # case) we keep the fixed default so teleop is unaffected by the slider.
        self._presence_getter = presence_getter

    @property
    def has_hand(self) -> bool:
        return self._hands is not None

    def _select_person(self, boxes, scores, kpts, frame_shape):
        """Largest person box (closest to the camera), ignoring tiny background
        people — same presenter pick the teleop uses."""
        if not len(scores):
            return None
        fh, fw = frame_shape
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        i = int(np.argmax(areas))
        if float(areas[i]) < self._min_person_area_frac * float(fw * fh):
            return None
        return kpts[i]

    def _pick_wrist(self, best):
        """Highest (most-raised) confident wrist, with its elbow, or None."""
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
        return min(cands, key=lambda c: float(best[c[0], 1]))  # highest wrist

    def _acquire_roi(self, frame, best):
        h, w = frame.shape[:2]
        pick = self._pick_wrist(best)
        if self._prev_lm_xy is not None:
            switch = (
                pick is not None
                and pick[0] != self._tracked_wi
                and float(best[pick[0], 1]) < float(best[pick[1], 1])
            )  # other hand raised
            if not switch:
                return track_roi(self._prev_lm_xy, w, h)
        if pick is None:
            return None
        self._tracked_wi = pick[0]
        return roi_from_wrist(best[pick[0], :2], best[pick[1], :2], w, h)

    def detect_hand(self, frame, boxes, scores, kpts):
        """Pose already ran; run the hand model on the wrist ROI it implies.

        Returns ``(roi, lm_xy)`` where ``lm_xy`` is (21, 2) full-frame hand
        landmarks (or None if no confident hand). Timing is the caller's job so
        it can be attributed to the NPU gauge."""
        if self._hands is None:
            self._prev_lm_xy = None
            return None, None
        best = self._select_person(boxes, scores, kpts, frame.shape[:2])
        roi = self._acquire_roi(frame, best)
        # Read the live hand-presence threshold each frame (falls back to the
        # fixed default when no getter is wired), so the §8 slider takes effect
        # instantly with no NPU re-run.
        presence_thresh = self._score_thresh
        if self._presence_getter is not None:
            presence_thresh = float(self._presence_getter())
        det = (
            self._hands.detect(frame, roi, presence_thresh) if roi is not None else None
        )
        self._prev_lm_xy = det[1] if det is not None else None
        return roi, self._prev_lm_xy


class _WebcamDemo:
    """Non-blocking live webcam pose demo with a Stop button.

    Runs on two daemon threads — one fuses capture + inference (read →
    preprocess → NPU → postprocess), the other handles display (draw → JPEG
    encode → widget) — so the notebook cell returns immediately and the Stop
    button stays responsive (a blocking display loop would freeze the kernel
    and the button with it).
    """

    def __init__(
        self,
        session,
        preprocess_frame,
        postprocess_pose,
        draw_poses,
        source=0,
        label="",
        cap_width=1280,
        cap_height=720,
        feed_npu_gauge=True,
        hand_session=None,
        hand_on=True,
        hand_presence_getter=None,
    ):
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._preprocess = preprocess_frame
        self._postprocess = postprocess_pose
        self._draw = draw_poses
        self._source = source  # camera index (int) or video path (str/Path)
        self._label = label
        self._cap_width = cap_width
        self._cap_height = cap_height
        # Only feed the dashboard's NPU gauge when this session actually runs on
        # the NPU. On a CPU run we leave the gauge alone so it correctly shows the
        # NPU idle (the whole point of the CPU-vs-NPU swap).
        self._feed_npu_gauge = feed_npu_gauge

        # Optional SECOND model: a 21-point hand-landmark net, pose-gated on the
        # wrist ROI (§8). The live toggle (``hand_on``) flips it on/off mid-demo
        # so the audience watches the second model added to / removed from the
        # SAME NPU. Perception lives in the shared PoseHandPerception helper so
        # §8 and the §9 teleop run one code path.
        # ``hand_presence_getter`` (optional) is a callable read PER FRAME by the
        # perception so a live "hand presence" slider can gate hand detections
        # instantly, mirroring how the pose ``conf`` slider works.
        self._perception = (
            PoseHandPerception(
                hand_session=hand_session, presence_getter=hand_presence_getter
            )
            if hand_session is not None
            else None
        )
        self._hand_on = bool(hand_on) and self._perception is not None

        self._disp_q = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._cap = None
        self._threads = []
        self._dashboard = None  # optional LiveDashboard shown under the demo

        import ipywidgets as widgets

        self._w = widgets
        # Fixed 640px feed (matches the Section 5 benchmark widgets) instead of
        # scaling to the full window width, which got too large.
        self.img = widgets.Image(format="jpeg", width=640)
        self.fps = widgets.HTML()
        self.status = widgets.HTML("<b style='color:#2a8'>● running</b>")
        self.stop_btn = widgets.Button(
            description="Stop", button_style="danger", icon="stop"
        )
        self.stop_btn.on_click(lambda _b: self.stop())
        # Live on/off for the second (hand) model — only shown when a hand
        # session was supplied.
        self.hand_toggle = None
        if self._perception is not None:
            self.hand_toggle = widgets.ToggleButton(
                value=self._hand_on,
                description="hand model",
                icon="hand-paper-o",
                button_style="success" if self._hand_on else "",
            )
            self.hand_toggle.observe(self._on_hand_toggle, names="value")

    def _on_hand_toggle(self, change):
        self._hand_on = bool(change["new"])
        if self.hand_toggle is not None:
            self.hand_toggle.button_style = "success" if self._hand_on else ""

    # -- threads --------------------------------------------------------
    def _infer(self):
        # Capture + inference on ONE thread (read -> preprocess -> NPU ->
        # postprocess), exactly like the teleop's NPU loop. A separate
        # free-running capture thread + BUFFERSIZE=1 halved the rate here: the
        # tight MJPG-decode loop contends for the GIL/CPU and stretches each
        # session.run. Reading the source inline keeps the loop NPU/source
        # bound. The skeleton draw + JPEG encode run on the display thread.
        #
        # ``source`` is a camera index (live, capped at the webcam's frame rate)
        # or a video file path (decoded from disk -> not frame-rate capped, so
        # the NPU runs flat-out and you see its true throughput).
        try:
            frame_source, self._cap = make_frame_source(
                self._source, self._cap_width, self._cap_height
            )
        except Exception as exc:
            self.status.value = f"<b style='color:#a33'>■ {exc}</b>"
            return
        fps_window = deque(maxlen=30)
        t_last = time.perf_counter()
        while not self._stop.is_set():
            frame = frame_source()
            if frame is None:
                break
            nchw, info = self._preprocess(frame)
            t0 = time.perf_counter()
            outputs = self._session.run(None, {self._input_name: nchw})
            pose_ms = (time.perf_counter() - t0) * 1000
            boxes, scores, kpts = self._postprocess(outputs, frame.shape[:2], info)

            # Second model on the same NPU: pose-gated hand landmarks, timed as
            # NPU work so both models feed the one gauge. The live toggle can
            # switch it off without stopping the demo.
            roi = lm_xy = None
            hand_ms = 0.0
            if self._perception is not None and self._hand_on:
                t1 = time.perf_counter()
                roi, lm_xy = self._perception.detect_hand(frame, boxes, scores, kpts)
                hand_ms = (time.perf_counter() - t1) * 1000

            if self._feed_npu_gauge:
                NPU_ACTIVITY.record(
                    pose_ms + hand_ms
                )  # NPU runs only; keeps the gauge honest on CPU

            now = time.perf_counter()
            fps_window.append(1.0 / max(now - t_last, 1e-6))
            t_last = now
            npu_fps = sum(fps_window) / len(fps_window)

            if self._disp_q.full():
                try:
                    self._disp_q.get_nowait()
                except queue.Empty:
                    pass
            self._disp_q.put(
                (frame, boxes, scores, kpts, pose_ms, hand_ms, roi, lm_xy, npu_fps)
            )
        if self._cap is not None:
            self._cap.release()

    def _display(self):
        while not self._stop.is_set():
            try:
                frame, boxes, scores, kpts, pose_ms, hand_ms, roi, lm_xy, npu_fps = (
                    self._disp_q.get(timeout=0.5)
                )
            except queue.Empty:
                continue
            annotated = self._draw(frame, boxes, scores, kpts)
            if lm_xy is not None:
                draw_hand_landmarks(annotated, lm_xy)
            self._set_stats(
                npu_fps, pose_ms, hand_ms, len(boxes), hand_tracked=lm_xy is not None
            )  # HTML line above the feed
            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                continue
            self.img.value = jpg.tobytes()

    def _set_stats(self, fps, pose_ms, hand_ms, n_people, hand_tracked=False):
        """Perf readout as an HTML line above the feed (matches Section 5).

        Per-model latency is the teaching signal: a live webcam caps fps at its
        own frame rate, so toggling the second model does NOT move the fps
        number, only the per-model ms does. Pose ms is always shown; hand ms is
        shown only while the second model is active."""
        tag = self._label or "NPU"
        people = f"{n_people} {'person' if n_people == 1 else 'people'}"
        if self._perception is None:
            self.fps.value = (
                f"<b>{tag}</b> &nbsp; <span style='color:#0a8'>{fps:.1f} fps</span>"
                f" &nbsp; ({pose_ms:.1f} ms/frame) &nbsp; {people}"
            )
            return
        hand_bit = (
            f" &nbsp;&nbsp; <b>hand</b> {hand_ms:.1f} ms" if self._hand_on else ""
        )
        self.fps.value = (
            f"<b>{tag} (pose{'+hand' if self._hand_on else ''})</b> &nbsp; "
            f"<span style='color:#0a8'>{fps:.1f} fps</span> &nbsp; "
            f"<b>pose</b> {pose_ms:.1f} ms{hand_bit} &nbsp;&nbsp; {people}"
        )

    # -- control --------------------------------------------------------
    def start(self):
        self._stop.clear()
        self._threads = [
            threading.Thread(target=self._infer, name="infer", daemon=True),
            threading.Thread(target=self._display, name="disp", daemon=True),
        ]
        for t in self._threads:
            t.start()

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        if self._dashboard is not None:
            try:
                self._dashboard.stop()
            except Exception:
                pass
        try:
            self.stop_btn.disabled = True
            self.status.value = "<b style='color:#a33'>■ stopped</b>"
        except Exception:
            pass


def run_webcam_demo(
    session,
    preprocess_frame,
    postprocess_pose,
    draw_poses,
    source=0,
    label="",
    show_dashboard=True,
    npu_infer_ms=20.0,
    cap_width=1280,
    cap_height=720,
    cam_index=None,
    feed_npu_gauge=True,
    extra_widgets=None,
    hand_session=None,
    hand_on=True,
    hand_presence_getter=None,
):
    """Live pose demo with a Stop button. Returns immediately (the pipeline runs
    on background threads). Click **Stop** to end it cleanly.

    Pass ``hand_session`` (a VitisAI/NPU session for the 21-point hand model) to
    add a SECOND model on the same NPU: pose finds a raised wrist, the hand model
    reads the fingers in that crop, and both skeletons overlay the feed. A live
    "hand model" toggle turns the second model on/off without stopping the demo,
    so the audience sees it added to / removed from the one NPU. ``hand_on`` sets
    the toggle's start state.

    ``hand_presence_getter`` (optional) is a zero-arg callable read PER FRAME to
    gate hand detections on the model's hand-presence score. Left None (the
    default, and what the notebook uses), the perception keeps its fixed default
    threshold. There is intentionally no live slider for this: the hand model's
    presence score saturates near ~1.0 whenever a hand is visible, so a threshold
    slider almost never crosses it; ``HandLandmarker.detect``'s geometry check is
    what actually gates hands.

    ``source`` selects the input:
      * an **int** (default ``0``) → live webcam at that index. The live fps is
        capped by the webcam's own frame rate (many USB cams top out at 20–30
        fps), so this shows the real-world rate, not the NPU's ceiling.
      * a **path** (str/Path) to a video file → decoded from disk, which is NOT
        frame-rate capped, so the NPU runs flat-out and you see its true
        throughput (~40–50 fps in turbo). Great for showcasing the NPU when the
        webcam is the bottleneck. The video loops forever.

    ``cap_width``/``cap_height`` set the camera capture resolution (ignored for
    video files). The model letterboxes to 640 internally, so resolution affects
    only the displayed feed's sharpness, not pose accuracy.

    When ``show_dashboard`` is True, a live resource dashboard is rendered right
    under the feed so all three compute units (NPU/CPU/iGPU) are visible at once.
    """
    from IPython.display import display

    if cam_index is not None:  # backwards-compat alias for the old kwarg
        source = cam_index

    pipeline_registry.stop_active_pipelines()  # don't fight a previous demo (fps!)
    demo = _WebcamDemo(
        session,
        preprocess_frame,
        postprocess_pose,
        draw_poses,
        source=source,
        label=label,
        cap_width=cap_width,
        cap_height=cap_height,
        feed_npu_gauge=feed_npu_gauge,
        hand_session=hand_session,
        hand_on=hand_on,
        hand_presence_getter=hand_presence_getter,
    )

    top_row = [demo.stop_btn, demo.status]
    if demo.hand_toggle is not None:  # live on/off for the second (hand) model
        top_row.insert(1, demo.hand_toggle)
    children = [demo._w.HBox(top_row), demo.fps]
    if extra_widgets:  # e.g. live threshold sliders, shown above the feed
        children.extend(extra_widgets)
    children.append(demo.img)
    if show_dashboard:
        from .live_dashboard import LiveDashboard

        demo._dashboard = LiveDashboard(
            window_seconds=60.0, sample_period=1.0, npu_infer_ms=npu_infer_ms
        )
        children.append(demo._dashboard.widget)

    display(demo._w.VBox(children))
    demo.start()
    if demo._dashboard is not None:
        demo._dashboard.start(auto_display=False)
    pipeline_registry.register(demo, None)
    return demo


def make_live_pose_controls(postprocess_pose, draw_poses, conf=0.4, kpt_vis=0.3):
    """Build the live tuning widgets for the Section 8 pose + hand webcam demo.

    Keeps the ipywidgets boilerplate out of the notebook cell. Returns
    ``(postprocess_live, draw_live, control_widgets)``:

      * ``postprocess_live`` / ``draw_live`` wrap the notebook's ``postprocess_pose``
        / ``draw_poses`` and read the current control values **each frame**, so the
        feed reacts instantly to the sliders/checkboxes with no NPU re-run;
      * ``control_widgets`` is the list to hand straight to
        ``run_webcam_demo(extra_widgets=...)`` (a sliders row + a show/hide row).

    Slider ranges are matched to what the models actually output (measured over
    the workshop image + videos), so every part of the track does something:
      * ``conf`` 0.30-0.96: box scores top out at ~0.96 (never higher), so 0.96-0.99
        is a dead zone where everything vanishes; below ~0.30 nothing ever drops.
        The subject starts flickering out near 0.95 and secondary/distant people
        drop between ~0.5-0.9.
      * ``kpt vis`` 0.30-0.99: visibility is bimodal (occluded joints ~0.0, visible
        ~0.97-0.99) with real signal all the way up, so it bites hardest past ~0.9.

    Note: there is deliberately no "hand presence" slider. The hand model's
    presence score saturates near ~1.0 whenever a hand is visible, so a threshold
    slider almost never crosses it; the geometry check in ``HandLandmarker.detect``
    (landmarks inside the crop + sane palm size) is what actually gates hands.
    """
    import ipywidgets as widgets

    controls = {
        "conf": conf,
        "kpt_vis": kpt_vis,
        "boxes": True,
        "skeleton": True,
        "keypoints": True,
    }

    def postprocess_live(outputs, original_shape, info):
        return postprocess_pose(
            outputs, original_shape, info, conf_threshold=controls["conf"]
        )

    def draw_live(frame, boxes, scores, kpts):
        return draw_poses(
            frame,
            boxes,
            scores,
            kpts,
            kpt_vis=controls["kpt_vis"],
            show_boxes=controls["boxes"],
            show_skeleton=controls["skeleton"],
            show_keypoints=controls["keypoints"],
        )

    conf_slider = widgets.FloatSlider(
        value=conf,
        min=0.30,
        max=0.99,
        step=0.01,
        description="conf",
        continuous_update=True,
        readout_format=".2f",
    )
    kpt_slider = widgets.FloatSlider(
        value=kpt_vis,
        min=0.30,
        max=0.99,
        step=0.01,
        description="kpt vis",
        continuous_update=True,
        readout_format=".2f",
    )
    conf_slider.observe(lambda ch: controls.update(conf=ch["new"]), names="value")
    kpt_slider.observe(lambda ch: controls.update(kpt_vis=ch["new"]), names="value")

    box_cb = widgets.Checkbox(value=True, description="boxes", indent=False)
    skel_cb = widgets.Checkbox(value=True, description="skeleton", indent=False)
    kpt_cb = widgets.Checkbox(value=True, description="keypoints", indent=False)
    box_cb.observe(lambda ch: controls.update(boxes=ch["new"]), names="value")
    skel_cb.observe(lambda ch: controls.update(skeleton=ch["new"]), names="value")
    kpt_cb.observe(lambda ch: controls.update(keypoints=ch["new"]), names="value")

    control_widgets = [
        widgets.HBox([conf_slider, kpt_slider]),
        widgets.HBox([box_cb, skel_cb, kpt_cb]),
    ]
    return postprocess_live, draw_live, control_widgets
