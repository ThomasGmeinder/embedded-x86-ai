# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A private rclpy node for the twin's subscribers.

:class:`Spinner` is the executor the twin session, the notebook observers,
and the ROS self-test share. The arm itself is
:class:`~vla_pipeline.robot.moveit_arm.MoveItArm`.
"""

from __future__ import annotations


class Spinner:
    """One rclpy node on its own executor thread, joined by :meth:`close`.

    A thread still inside ``executor.spin()`` when Python exits aborts the
    process ("terminate called without an active exception"), and
    ``executor.shutdown()`` does not end it. So the thread runs one callback
    at a time, and ``close()`` triggers a guard condition that wakes it to
    see the stop, then joins it before shutting the executor down.
    """

    def __init__(self, name: str):
        import threading

        import rclpy
        from rclpy.executors import SingleThreadedExecutor

        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(name)
        self._wake = self.node.create_guard_condition(lambda: None)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self.node)
        self._stop = threading.Event()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, daemon=True, name=name)
        self._thread.start()

    def add_node(self, node) -> None:
        """Spin ``node`` on this thread too; its owner still destroys it."""
        self._executor.add_node(node)
        self._wake.trigger()

    def _loop(self) -> None:
        from rclpy.executors import ExternalShutdownException

        while not self._stop.is_set():
            try:
                self._executor.spin_once()
            except ExternalShutdownException:
                return

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._wake.trigger()
        self._thread.join(timeout=2.0)
        self._executor.shutdown()
        try:
            self.node.destroy_node()
        except Exception:
            pass
