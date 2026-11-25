# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A reader with no frame waits on the camera's own event, not on a sleep.

``wait_for_frames`` is what the mimic loop, the two calibrations and the two
previews call after ``read()`` gave no frame. These stand-in cameras have the
two events real cameras have: a frame that arrives (a ROS image topic), and a
time at which a lost device is tried again.
"""

from __future__ import annotations

import threading
import time

import pytest

from vla_pipeline.vision.camera import wait_for_frames

pytestmark = [pytest.mark.sim, pytest.mark.timeout(30)]


class TopicCamera:
    """A latest-frame client: ``next_frame`` blocks until a frame arrives, as ``Ros2Camera`` does."""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._arrived = 0.0
        self.waited_for: list[float | None] = []

    def publish(self) -> None:
        with self._cond:
            self._arrived = time.time()
            self._cond.notify_all()

    def next_frame(self, after: float, timeout: float | None = None):
        self.waited_for.append(timeout)
        with self._cond:
            ok = self._cond.wait_for(lambda: self._arrived > after, timeout)
            return ok, object() if ok else None, self._arrived


class LostDevice:
    """A device that is gone and is tried again at ``retry_at()``, as ``DirectCamera`` does."""

    def __init__(self, in_s: float) -> None:
        self._at = time.time() + in_s

    def retry_at(self) -> float:
        return self._at

    def next_frame(self, after, timeout=None):  # pragma: no cover - a lost device is not asked
        raise AssertionError("a lost device was waited on for a frame")


def test_the_reader_is_released_by_the_frame_that_arrives():
    camera = TopicCamera()
    waiting = threading.Event()
    returned = threading.Event()

    def reader() -> None:
        waiting.set()
        wait_for_frames(camera, threading.Event(), bound_s=20.0)
        returned.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    assert waiting.wait(5)
    assert not returned.is_set()  # no frame yet: the reader is inside the camera's wait
    camera.publish()
    assert returned.wait(5), "the arriving frame did not release the reader"
    thread.join()
    assert camera.waited_for == [20.0]  # one bounded wait, not a series of short ones


def test_a_stop_ends_the_wait_for_a_lost_device():
    stop = threading.Event()
    returned = threading.Event()

    def reader() -> None:
        wait_for_frames(LostDevice(in_s=60.0), stop)
        returned.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    stop.set()  # the device is tried again in a minute; the stop does not wait for that
    assert returned.wait(5), "the stop did not end the wait for the device's reopen time"
    thread.join()


def test_a_device_whose_reopen_time_has_come_is_not_waited_for():
    started = time.monotonic()
    wait_for_frames(LostDevice(in_s=-1.0), threading.Event())
    assert time.monotonic() - started < 1.0


def test_a_camera_whose_read_blocks_is_not_waited_on_again():
    """The console's fan-out and a paced stand-in wait inside ``read()``: nothing more to wait for."""

    class BlockingRead:
        def read(self):
            return False, None

    started = time.monotonic()
    wait_for_frames(BlockingRead(), threading.Event())
    assert time.monotonic() - started < 1.0
