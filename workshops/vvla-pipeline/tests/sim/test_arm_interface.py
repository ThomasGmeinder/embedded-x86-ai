# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""``make_arm`` builds a MoveIt arm for the real bringup and for the twin."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.sim


def test_make_arm_builds_moveit_for_the_real_bringup_and_the_twin(monkeypatch):
    from vla_pipeline.robot import arm_interface, moveit_arm

    built = []
    monkeypatch.setattr(moveit_arm, "MoveItArm", lambda cfg, sim: built.append(sim) or "moveit")

    assert arm_interface.make_arm({"robot": {"backend": "real"}}) == "moveit"
    assert arm_interface.make_arm({"robot": {}}) == "moveit"
    assert arm_interface.make_arm({"robot": {"backend": "sim"}}) == "moveit"
    assert built == [False, False, True]


def test_pick_on_an_uncalibrated_real_arm_asks_for_calibration():
    """A real arm with no camera calibration refuses before it looks or moves."""
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_pick_place

    said = []
    looked = []
    ctx = SimpleNamespace(
        cfg={"robot": {"backend": "real"}, "cameras": {"arm": {}}, "pick_place": {}},
        arm=SimpleNamespace(node=object()),
        detector=SimpleNamespace(find=lambda *_a, **_k: looked.append(1)),
        say=said.append,
        stop_check=lambda: False,
    )
    assert run_pick_place(ctx, SimpleNamespace(object_name="ball")) is False
    assert said == [
        "The arm camera isn't calibrated. "
        "lerobot-calibrate, camera_calibration, and easy_handeye2 are not installed."
    ]
    assert looked == []


def test_pick_with_wrist_observations_does_not_require_a_local_detector():
    """Perception owns the detector. Pick reads its wrist observations."""
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_pick_place

    said = []
    ctx = SimpleNamespace(
        cfg={"robot": {"backend": "real"}, "cameras": {"arm": {}}, "pick_place": {}},
        arm=SimpleNamespace(sim=False),
        detector=None,
        mailbox=object(),
        say=said.append,
        stop_check=lambda: False,
    )
    assert run_pick_place(ctx, SimpleNamespace(object_name="ball")) is False
    assert said[0].startswith("The arm camera isn't calibrated.")


def test_a_wrist_image_without_calibration_is_not_called_offline():
    """A missing camera_info is not a dead camera."""
    import threading
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_pick_place

    said = []

    class _Box:
        def wait_after(self, *_args, **_kwargs):
            return SimpleNamespace(camera=None, detections=())

    ctx = SimpleNamespace(
        cfg={
            "robot": {"backend": "sim"},
            "cameras": {"arm": {"topic": "/cameras/arm/image_raw", "stale_after_s": 1.0}},
            "pick_place": {
                "table_z": 0.0,
                "pregrasp_above_m": 0.05,
                "wrist_align_pregrasp_above_m": 0.08,
                "wrist_align_classes": [],
                "wrist_axis_min_elongation": 1.2,
            },
        },
        arm=SimpleNamespace(sim=True, scan_states=lambda: ["scan_1"], go_to_named=lambda _name: "arrived"),
        detector=None,
        mailbox=_Box(),
        say=said.append,
        stop_check=lambda: False,
        stop_event=threading.Event(),
    )
    assert run_pick_place(ctx, SimpleNamespace(object_name="ball")) is False
    assert said == ["The wrist image is arriving, but its calibration message has not."]


def test_a_sim_camera_without_ros_is_refused():
    from vla_pipeline.vision.camera import make_camera

    cfg = {"cameras": {"arm": {"source": "sim", "use_ros2": False, "topic": "/cameras/arm/image_raw"}}}
    with pytest.raises(RuntimeError, match="use_ros2: true"):
        make_camera(cfg, "arm")
