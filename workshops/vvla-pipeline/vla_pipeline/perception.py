# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Perception process: cameras, YOLO, MediaPipe, Silero, and the llama-server child.

Each camera thread blocks on the next frame. It does not sleep to find one.
Measurements go out as observations. Joint targets stay in the control process.
llama-server is started once and is not pinned to this core.
"""

from __future__ import annotations

import logging
import signal
import threading
import time

from vla_pipeline.ipc import (
    OVERLAY_HANDS,
    OVERLAY_MOUNT,
    OVERLAY_WRIST,
    CameraModel,
    DetectionSignals,
    HandSignals,
    MountObservation,
    PerceptionStatus,
    WristObservation,
)
from vla_pipeline.main import apply_sim_overrides, build_parser
from vla_pipeline.wire import PerceptionLink

logger = logging.getLogger(__name__)

_OVERLAY_S = 1.0 / 15.0


class _Frames:
    """The newest frame of one camera. The worker waits on this condition."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._seq = 0
        self._stamp = 0.0
        self._frame = None
        self._info = None
        self._closed = False

    def image(self, msg) -> None:
        from vla_pipeline.vision.camera import from_image_msg

        frame = from_image_msg(msg)
        if frame is None:
            return
        with self._cond:
            self._frame = frame
            self._stamp = time.time()
            self._seq += 1
            self._cond.notify_all()

    def info(self, msg) -> None:
        with self._cond:
            self._info = msg

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()

    def wait(self, after: int):
        """The next frame after ``after``, or None once :meth:`close` is called."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or self._seq > after)
            if self._seq <= after:
                return None
            return self._seq, self._stamp, self._frame, self._info


def _hand(obs) -> HandSignals | None:
    if obs is None:
        return None
    return HandSignals(
        float(obs.pinch), (float(obs.thumb_px[0]), float(obs.thumb_px[1])),
        (float(obs.index_px[0]), float(obs.index_px[1])), str(obs.handedness),
        float(obs.hand_size), float(obs.roll), float(obs.roll_ti), float(obs.roll_ref),
        float(obs.roll_ti_weight), (float(obs.wrist_px[0]), float(obs.wrist_px[1])),
    )


def _camera(info, width: int, height: int) -> CameraModel | None:
    if info is None:
        return None
    k = [float(v) for v in info.k]
    if len(k) != 9:
        return None
    return CameraModel(
        int(info.width or width), int(info.height or height), tuple(k),
        tuple(float(v) for v in info.d), str(info.distortion_model or "plumb_bob"),
    )


def _mount_worker(slot, link, pose, hands, models, stop) -> None:
    from vla_pipeline.vision.yolo_pose_npu import pick_nearest

    seq = 0
    overlay_at = 0.0
    while not stop.is_set():
        item = slot.wait(seq)
        if item is None:
            return
        seq, stamp, frame, _info = item
        height, width = frame.shape[:2]
        with models:
            boxes, scores, kpts = pose.infer(frame) if pose is not None else ((), (), ())
            observed, raw = hands.process(frame) if hands is not None else (None, None)
        person = ()
        if pose is not None and len(boxes):
            person = tuple(tuple(float(v) for v in point) for point in kpts[pick_nearest(boxes)])
        link.publish_mount(MountObservation(
            "mount", stamp, seq, int(width), int(height), person, _hand(observed),
        ))
        now = time.monotonic()
        if now - overlay_at < _OVERLAY_S:
            continue
        overlay_at = now
        pose_image = pose.draw(frame, boxes, scores, kpts) if pose is not None else frame
        hand_image = hands.draw(frame, observed, raw) if hands is not None and observed is not None else frame
        link.publish_overlay(OVERLAY_MOUNT, pose_image)
        link.publish_overlay(OVERLAY_HANDS, hand_image)
        link.publish_status(link.status)


def _wrist_worker(slot, link, detector, models, cfg, stop) -> None:
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE, object_axis

    elong = float((cfg.get("pick_place") or {}).get("wrist_axis_min_elongation", 1.2))
    seq = 0
    overlay_at = 0.0
    while not stop.is_set():
        item = slot.wait(seq)
        if item is None:
            return
        seq, stamp, frame, info = item
        height, width = frame.shape[:2]
        with models:
            found = detector.infer(frame) if detector is not None else []
        signals = []
        for det in found:
            size = CLASS_SIZE.get(det.name)
            axis = None
            if size is not None and size[0] != size[1]:
                ends = object_axis(frame, det.box, elong)
                if ends is not None:
                    axis = (tuple(float(v) for v in ends[0]), tuple(float(v) for v in ends[1]))
            signals.append(DetectionSignals(
                det.name, float(det.score), tuple(float(v) for v in det.box), axis,
            ))
        link.publish_wrist(WristObservation(
            "arm", stamp, seq, int(width), int(height), _camera(info, width, height), tuple(signals),
        ))
        now = time.monotonic()
        if now - overlay_at < _OVERLAY_S:
            continue
        overlay_at = now
        image = detector.draw(frame, found) if detector is not None else frame
        link.publish_overlay(OVERLAY_WRIST, image)


def _subscribe(node, slot, topic: str) -> None:
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import CameraInfo, Image

    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    node.create_subscription(Image, topic, slot.image, qos)
    info = topic.removesuffix("image_raw") + "camera_info"
    node.create_subscription(CameraInfo, info, slot.info, qos)


def main() -> None:
    """Subscribe to the cameras, run the models, and publish observations."""
    from vla_pipeline.utils import npu_session

    npu_session.defer_sigint = False
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    from vla_pipeline.utils.config import ensure_dirs, load_config

    cfg = load_config(args.config)
    if args.digital_twin:
        apply_sim_overrides(cfg)
    ensure_dirs(cfg)

    from vla_pipeline.robot.ros2_control_arm import Spinner

    spin = Spinner("perception")
    link = PerceptionLink(spin.node)
    link.voice = "down"
    link.level = 0.0
    link.speaking = False
    link.busy = False

    def status() -> PerceptionStatus:
        return PerceptionStatus(link.voice, float(link.level), bool(link.speaking), bool(link.busy))

    link.status = status()
    stop = threading.Event()

    def _stop(_signum, _frame) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    pose = hands = detector = None
    try:
        from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

        pose = PoseEstimator(cfg, device=args.device)
    except Exception as exc:
        logger.warning("Pose model unavailable: %s", exc)
    try:
        from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

        detector = ObjectDetector(cfg, device=args.device)
    except Exception as exc:
        logger.warning("Detector unavailable: %s", exc)
    try:
        from vla_pipeline.vision.mediapipe_hands import HandTracker

        hands = HandTracker(cfg)
    except Exception as exc:
        logger.warning("Hand tracker unavailable: %s", exc)

    models = threading.Lock()
    slots = []
    workers = []
    cameras = cfg.get("cameras") or {}
    if (cameras.get("mount") or {}).get("topic"):
        slot = _Frames()
        _subscribe(spin.node, slot, cameras["mount"]["topic"])
        slots.append(slot)
        workers.append(threading.Thread(
            target=_mount_worker, args=(slot, link, pose, hands, models, stop),
            name="mount-perception", daemon=True,
        ))
    if (cameras.get("arm") or {}).get("topic"):
        slot = _Frames()
        _subscribe(spin.node, slot, cameras["arm"]["topic"])
        slots.append(slot)
        workers.append(threading.Thread(
            target=_wrist_worker, args=(slot, link, detector, models, cfg, stop),
            name="wrist-perception", daemon=True,
        ))
    for worker in workers:
        worker.start()

    voice = None
    if not args.text_commands:
        from vla_pipeline.llm.intents import command_from_json, is_stop
        from vla_pipeline.llm.voice import open_voice

        voice = open_voice(cfg, device=args.device)
        link.voice = "down" if voice.error else "up"
        link.status = status()
        link.publish_status(link.status)

        def on_text(text: str) -> None:
            if is_stop(text):
                link.offer_stop()
                return
            command = command_from_json(text)
            if command is None:
                return
            link.offer_command(
                command.intent.value,
                str((command.params or {}).get("object", "")),
                command.transcript or "",
            )

        def on_level(level: float, speaking: bool) -> None:
            link.level = level
            link.speaking = speaking
            link.status = status()
            link.publish_status(link.status)

        def on_busy(busy: bool) -> None:
            link.busy = busy
            link.status = status()
            link.publish_status(link.status)

        def on_drop(_reason: str) -> None:
            link.voice = "down"
            link.status = status()
            link.publish_status(link.status)

        if voice.error:
            logger.warning("Voice is unavailable (%s).", voice.error)
        else:
            voice.start(on_text, on_level=on_level, on_busy=on_busy, on_drop=on_drop, stop_check=stop.is_set)
    else:
        link.publish_status(status())

    stop.wait()
    for slot in slots:
        slot.close()
    for worker in workers:
        worker.join(timeout=2.0)
    if voice is not None:
        voice.close()
    if hands is not None:
        hands.close()
    spin.close()


if __name__ == "__main__":
    main()
