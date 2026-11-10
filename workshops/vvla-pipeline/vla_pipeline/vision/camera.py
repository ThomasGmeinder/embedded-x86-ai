# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Camera access layer: one ``CameraClient`` API for the two pipeline cameras.

Roles
-----
- ``mount`` - externally mounted, faces the human. Feeds gesture-mimic pose
  tracking and person finding.
- ``arm``   - mounted on the SO-101 end effector. Feeds object detection in
  the jaws/workspace and pick-and-place visual servoing.

Behaviors never own
a ``cv2.VideoCapture`` - they take a :class:`CameraClient` so the same code
runs against direct V4L2 devices (``cameras.<role>.use_ros2: false``) or a
ROS 2 image topic published by ``v4l2_camera`` or the twin's camera plugin.

Graceful failure: ``read()`` returns ``(False, None)`` instead of raising, so
behaviors can degrade (fetch/pick already handle a missing camera) while the
client retries the device in the background.

Independent test
----------------
    python -m vla_pipeline.vision.camera --role mount          # direct preview
    python -m vla_pipeline.vision.camera --role arm --ros2     # via topic
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class CameraClient(ABC):
    """Minimal frame source API used by every behavior."""

    role: str = "?"

    @abstractmethod
    def read(self) -> tuple[bool, np.ndarray | None]:
        """Latest BGR frame. ``(False, None)`` when unavailable (never raises)."""

    def ok(self) -> bool:
        """Return True if a frame is currently available."""
        return self.read()[0]

    def close(self) -> None:  # noqa: B027
        """No-op; override in clients that own a resource to release."""


class DirectCamera(CameraClient):
    """OpenCV capture with automatic reopen on failure (no ROS required)."""

    def __init__(
        self,
        role: str,
        device: int | str,
        width: int,
        height: int,
        fps: float,
        reopen_interval_s: float = 3.0,
    ):
        import cv2  # local: keep module importable without cv2 for unit tests

        self._cv2 = cv2
        self.role = role
        self._device = device
        self._size = (int(width), int(height))
        self._fps = float(fps)
        self._reopen_interval = float(reopen_interval_s)
        self._cap = None
        self._next_retry = 0.0
        self._open()

    def _open(self) -> None:
        """Open (or reopen) the V4L2 device."""
        cap = self._cv2.VideoCapture(self._device)
        if cap.isOpened():
            cap.set(self._cv2.CAP_PROP_FRAME_WIDTH, self._size[0])
            cap.set(self._cv2.CAP_PROP_FRAME_HEIGHT, self._size[1])
            cap.set(self._cv2.CAP_PROP_FPS, self._fps)
            self._cap = cap
            logger.info("[camera:%s] opened device %s", self.role, self._device)
        else:
            cap.release()
            self._cap = None
            self._next_retry = time.time() + self._reopen_interval
            logger.warning(
                "[camera:%s] device %s unavailable - will retry every %.0fs",
                self.role,
                self._device,
                self._reopen_interval,
            )

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame, reopening the device on failure or after the retry interval."""
        if self._cap is None:
            if time.time() >= self._next_retry:
                self._open()
            if self._cap is None:
                return False, None
        ok, frame = self._cap.read()
        if not ok:
            self._cap.release()
            self._cap = None
            self._next_retry = time.time() + self._reopen_interval
            return False, None
        return True, frame

    def retry_at(self) -> float | None:
        """When :meth:`read` next tries to reopen a lost device (``time.time()``); None while it is open."""
        return self._next_retry if self._cap is None else None

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """Discard buffered frames until a read blocks for a new exposure.

        ``settled_at`` is when the arm finished moving. A buffered frame
        returns immediately; the frame that blocks for the next exposure was
        captured after the move.
        """
        period = 1.0 / max(self._fps, 1.0)
        ok, frame = False, None
        for _ in range(8):
            started = time.perf_counter()
            ok, frame = self.read()
            if not ok or frame is None:
                return False, None
            if time.perf_counter() - started >= 0.5 * period:
                return True, frame
        return bool(ok), frame

    def close(self) -> None:
        """Release the OpenCV capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def from_image_msg(msg) -> np.ndarray | None:
    """Decode a ``bgr8``/``rgb8`` ``sensor_msgs/Image`` into a BGR ndarray (or None)."""
    if msg.encoding not in ("bgr8", "rgb8"):
        return None
    frame = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        frame = frame[..., ::-1]
    return frame.copy()


def to_image_msg(frame_bgr, stamp=None, frame_id: str = ""):
    """Pack a BGR ndarray into a ``sensor_msgs/Image`` (encoding ``bgr8``)."""
    from sensor_msgs.msg import Image

    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = frame_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = frame_bgr.tobytes()
    return msg


class Ros2Camera(CameraClient):
    """Latest-frame subscriber on a camera node's image topic.

    Keeps only the newest frame (best-effort QoS, depth 1) so vision can never
    back up behind the publisher, and reports ``(False, None)`` when the stream
    goes stale (``stale_after_s``), so a dead publisher looks like a dead
    camera, not like a frozen image.
    """

    def __init__(self, role: str, topic: str, stale_after_s: float = 1.0):
        self.role = role
        self._stale_after = float(stale_after_s)
        self._cond = threading.Condition()  # notified on every new frame, and on close
        self._frame: np.ndarray | None = None
        self._stamp = 0.0
        self._closed = False
        self._executor = None
        self._node = None
        self._setup_ros(topic)
        logger.info("[camera:%s] subscribed to %s", role, topic)

    def _setup_ros(self, topic: str) -> None:
        """Create the client node, the subscription, and a spin thread.

        Init rclpy if needed; make a node named ``camera_client_<role>``;
        subscribe to ``topic`` (``sensor_msgs/Image``) with best-effort /
        keep-last / depth-1 QoS routing into ``self._on_image``. The node
        spins on :class:`~vla_pipeline.robot.ros2_control_arm.Spinner`, which
        is the thread ``close`` can stop.
        """
        import rclpy
        from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import Image

        from vla_pipeline.robot.ros2_control_arm import Spinner

        if not rclpy.ok():
            rclpy.init()
        # Spinner is the thread that stops: executor.shutdown() does not end a
        # thread blocked in spin(), and close() used to destroy the node under it.
        self._spin = Spinner(f"camera_client_{self.role}")
        self._node = self._spin.node
        self._executor = None
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._node.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, msg) -> None:
        """Decode an incoming Image message into the cached latest frame."""
        frame = from_image_msg(msg)
        if frame is None:
            return
        with self._cond:
            self._frame = frame
            self._stamp = time.time()
            self._cond.notify_all()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the latest frame if it isn't stale, else (False, None)."""
        with self._cond:
            if self._frame is None or (time.time() - self._stamp) > self._stale_after:
                return False, None
            return True, self._frame

    def read_after(self, settled_at: float) -> tuple[bool, np.ndarray | None]:
        """First frame that arrives after ``settled_at``, within the stream's staleness limit."""
        ok, frame, _arrived = self.next_frame(settled_at, max(self._stale_after, 0.5))
        return ok, frame

    def next_frame(self, after: float, timeout: float | None = None) -> tuple[bool, np.ndarray | None, float]:
        """``(ok, frame, arrived)``: the first frame that arrives after ``after``;
        not ok once closed, or when ``timeout`` passes first."""
        with self._cond:
            self._cond.wait_for(lambda: self._closed or (self._frame is not None and self._stamp > after), timeout)
            if self._closed or self._frame is None or self._stamp <= after:
                return False, None, after
            return True, self._frame, self._stamp

    def close(self) -> None:
        """Wake a waiting reader, shut down the executor and destroy the subscriber node."""
        with self._cond:
            self._closed = True
            self._cond.notify_all()
        spin = getattr(self, "_spin", None)
        if spin is not None:
            spin.close()
            self._spin = None
            self._node = None
            self._executor = None


def make_camera(cfg: dict, role: str) -> CameraClient:
    """Build the CameraClient for ``cameras.<role>`` from the pipeline config."""
    c = (cfg.get("cameras") or {}).get(role)
    if c is None:
        raise KeyError(f"no cameras.{role} in the config")
    if c.get("use_ros2", False):
        return Ros2Camera(role, c["topic"], stale_after_s=c.get("stale_after_s", 1.0))
    if c.get("source", "device") == "sim":
        raise RuntimeError(f"cameras.{role} source sim is the ROS image topic; set use_ros2: true")
    return DirectCamera(
        role,
        c.get("device", 0),
        c.get("width", 640),
        c.get("height", 480),
        c.get("fps", 30),
    )


# =============================================================================
# Standalone component test (live preview)
# =============================================================================


def main() -> None:
    """CLI entry point: preview one camera role in a live OpenCV window."""
    import cv2

    from vla_pipeline.utils.config import load_config

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Camera client - component test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--role", choices=["mount", "arm"], default="mount")
    parser.add_argument(
        "--ros2",
        action="store_true",
        help="force the ROS 2 transport regardless of config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.ros2:
        cfg.setdefault("cameras", {}).setdefault(args.role, {})["use_ros2"] = True
    cam = make_camera(cfg, args.role)
    print(f"Previewing camera '{args.role}' - press 'q' to quit.")
    misses = 0
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                misses += 1
                if misses % 30 == 1:
                    print("  (no frames - camera offline, retrying...)")
                time.sleep(0.1)
                continue
            misses = 0
            cv2.imshow(f"camera:{args.role}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
