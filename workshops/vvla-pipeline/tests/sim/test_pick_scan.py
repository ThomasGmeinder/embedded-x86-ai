# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pick scan stops at the first view whose pick arrives, and a quiet
wrist camera is reported, not raised."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim

SCANS = ["scan_1", "scan_2", "scan_3", "scan_near_1"]


class _Arm:
    """Records the named moves and picks. ``target_at`` maps a scan to the
    pick statuses its toothbrush detections return, best score first."""

    sim = True
    node = None

    def __init__(self, target_at):
        self.target_at = target_at
        self.at = None
        self.moves = []
        self.picks = []

    def scan_states(self):
        return list(SCANS)

    def go_to_named(self, name):
        self.moves.append(name)
        if name != "open":
            self.at = name
        return "arrived"

    def grasp_box(self, size, yaw):
        return size, yaw

    def remember(self, *_args):
        pass

    def pick_up(self, _name, centre, *_args):
        self.picks.append((self.at, centre))
        return self.target_at[self.at][centre]


def _ctx(monkeypatch, arm, *, info=True):
    from vla_pipeline.behaviors import manipulation
    from vla_pipeline.robot.moveit_arm import grasp_frame
    from vla_pipeline.vision import camera, yolo_detect_npu
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE

    tip = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=0.1), orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    # The twin reports the picked toothbrush where the jaws hold it.
    jaws = grasp_frame(CLASS_SIZE["toothbrush"]).position
    held = SimpleNamespace(free_joints=[SimpleNamespace(name="toothbrush", pose=SimpleNamespace(pose=SimpleNamespace(
        position=SimpleNamespace(x=tip.position.x + jaws.x, y=tip.position.y + jaws.y, z=tip.position.z + jaws.z),
    )))])

    def message(_node, msg_type, _topic):
        if msg_type.__name__ == "CameraInfo" and not info:
            return None
        return held if msg_type.__name__ == "FreeJointStateArray" else object()

    monkeypatch.setattr(manipulation, "_ros_message", message)
    monkeypatch.setattr(manipulation, "_optical_in_base", lambda _node: None)
    monkeypatch.setattr(manipulation, "tip_pose", lambda _arm: tip)
    monkeypatch.setattr(camera, "from_image_msg", lambda _image: arm.at)
    monkeypatch.setattr(
        yolo_detect_npu, "locate_detection",
        lambda box, *_args: {"xyz": box, "yaw": 0.0},
    )

    def infer(scan):
        statuses = arm.target_at.get(scan, {})
        return [
            SimpleNamespace(name="toothbrush", score=1.0 - 0.1 * rank, box=centre)
            for rank, centre in enumerate(statuses)
        ]

    said = []
    ctx = SimpleNamespace(
        cfg={
            "cameras": {"arm": {"topic": "/cameras/arm/image_raw"}},
            "pick_place": {
                "table_z": 0.0, "pregrasp_above_m": 0.06, "wrist_align_pregrasp_above_m": 0.1,
                "wrist_align_classes": ["toothbrush"], "wrist_axis_min_elongation": 1.6,
            },
        },
        arm=arm,
        detector=SimpleNamespace(infer=infer),
        say=said.append,
        stop_check=lambda: False,
        shared_state={},
    )
    return ctx, said


def test_the_scan_stops_at_the_first_view_whose_pick_arrives(monkeypatch):
    from vla_pipeline.behaviors.manipulation import run_pick_place

    arm = _Arm({"scan_2": {"a": "arrived"}, "scan_3": {"b": "arrived"}})
    ctx, said = _ctx(monkeypatch, arm)

    assert run_pick_place(ctx, SimpleNamespace(object_name="toothbrush")) is True
    assert [move for move in arm.moves if move != "open"] == ["scan_1", "scan_2"]
    assert arm.picks == [("scan_2", "a")]
    assert said == ["Got the toothbrush."]


def test_a_pick_with_no_plan_tries_the_next_detection_then_the_next_view(monkeypatch):
    from vla_pipeline.behaviors.manipulation import run_pick_place

    arm = _Arm({
        "scan_1": {"a": "NO_PLAN", "b": "NO_PLAN"},
        "scan_3": {"c": "arrived"},
        "scan_near_1": {"d": "arrived"},
    })
    ctx, said = _ctx(monkeypatch, arm)

    assert run_pick_place(ctx, SimpleNamespace(object_name="toothbrush")) is True
    assert [move for move in arm.moves if move != "open"] == ["scan_1", "scan_2", "scan_3"]
    assert arm.picks == [("scan_1", "a"), ("scan_1", "b"), ("scan_3", "c")]
    assert said == ["Got the toothbrush."]


def test_a_quiet_camera_info_says_the_camera_is_offline(monkeypatch):
    from vla_pipeline.behaviors.manipulation import run_pick_place

    arm = _Arm({"scan_1": {"a": "arrived"}})
    ctx, said = _ctx(monkeypatch, arm, info=False)

    assert run_pick_place(ctx, SimpleNamespace(object_name="toothbrush")) is False
    assert arm.picks == []
    assert said == ["The arm camera is offline"]


def test_no_target_in_any_view_says_so(monkeypatch):
    from vla_pipeline.behaviors.manipulation import run_pick_place

    arm = _Arm({})
    ctx, said = _ctx(monkeypatch, arm)

    assert run_pick_place(ctx, SimpleNamespace(object_name="toothbrush")) is False
    assert [move for move in arm.moves if move != "open"] == SCANS
    assert said == ["I can't see a toothbrush in the workspace."]
