# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""TODO 2.3: the camera client's ROS wiring.

The pipeline's :class:`~vla_pipeline.vision.camera.Ros2Camera` keeps only the
newest frame and reports ``(False, None)`` when the stream goes stale. This
subclass is that client with YOUR ``_setup_ros``.

Reference: ``vla_pipeline/vision/camera.py`` (``Ros2Camera._setup_ros``).
Notebook: ``notebooks/02_ros2/01_cameras.ipynb``
Self-test: ``python -m vvla_workshop.selftest_ros``
"""

from __future__ import annotations

import threading

from vla_pipeline.vision import camera


class Ros2Camera(camera.Ros2Camera):
    """The pipeline's camera client, wired by your TODO."""

    def _setup_ros(self, topic: str) -> None:
        """Create the client node, the subscription, and a spin thread.

        Init rclpy if needed; make a node named ``camera_client_<role>``;
        subscribe to ``topic`` (``sensor_msgs/Image``) with best-effort /
        keep-last / depth-1 QoS routing into ``self._on_image``; then spin a
        ``SingleThreadedExecutor`` on a daemon thread so messages keep
        arriving while the caller's thread does vision work. Keep the node in
        ``self._node`` and the executor in ``self._executor``: ``close()``
        shuts them down.
        """
        # >>> TODO 2.3: Ros2Camera._setup_ros - notebooks/02_ros2/01_cameras.ipynb
        raise NotImplementedError(
            "TODO 2.3 (Ros2Camera._setup_ros): implement me - see notebooks/02_ros2/01_cameras.ipynb")
        # <<< TODO 2.3
