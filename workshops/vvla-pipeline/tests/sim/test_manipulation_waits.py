# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""What the grasp and handover wait on: the arm owner's states and the stop event.

``ArmClient.wait_state`` paces the grasp loops on the owner's clock; the
handover and grip dwells wait on ``stop_event``, so a stop ends them at once.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.sim


def _ctx(arm, cfg=None, camera=None, pose=None):
    stop = threading.Event()
    return SimpleNamespace(
        arm=arm, cfg=cfg or {}, mount_camera=camera, pose=pose, arm_camera=None, shared_state={},
        stop_event=stop, stop_check=stop.is_set, say=lambda _m: None,
    )


def test_a_dry_run_arm_reaches_any_state_time_at_once():
    from vla_pipeline.robot.arm_interface import DryRunArm

    arm = DryRunArm()
    t0 = arm.state_time()
    assert arm.wait_state(t0 + 0.25) == t0 + 0.25
    assert arm.state_time() == t0 + 0.25
    assert arm.wait_state(t0 + 1.0, stop_check=lambda: True) is None


def test_the_twin_arm_waits_for_its_sim_time(pipeline_cfg):
    from so101_sim.world import SimWorld
    from vla_pipeline.robot.sim_arm import REST_POSE, SimArmClient

    world = SimWorld({"sim": dict(pipeline_cfg.get("sim") or {})}, rest_pose=REST_POSE)
    try:
        arm = SimArmClient(pipeline_cfg, world)
        t0 = arm.state_time()
        t = arm.wait_state(t0 + 0.1)
        assert t >= t0 + 0.1 and float(world.data.time) >= t0 + 0.1
        assert arm.wait_state(arm.state_time() + 10.0, stop_check=lambda: True) is None
    finally:
        world.close()


def test_a_stop_during_the_handover_keeps_the_object_in_the_jaws():
    """The release dwell ends on stop, and the jaws stay closed on the object."""
    from vla_pipeline.behaviors.manipulation import hand_over
    from vla_pipeline.robot.arm_interface import DryRunArm

    arm = DryRunArm()
    arm.send_joints({"gripper": 45.0})
    ctx = _ctx(arm)
    ctx.stop_event.set()
    lift = {"shoulder_lift": -40.0, "elbow_flex": 60.0}
    assert hand_over(ctx, {"handoff_reach": 20.0, "release_timeout_s": 6.0}, lift) is False
    assert arm.get_joints()["gripper"] == 45.0, "the jaws opened after a stop"


def test_a_stop_during_the_grip_countdown_closes_nothing(pipeline_cfg, monkeypatch):
    import vla_pipeline.behaviors.manipulation as manipulation
    from vla_pipeline.robot.arm_interface import REST_POSE, DryRunArm

    closes = []
    monkeypatch.setattr(manipulation, "close_with_feedback", lambda *a, **k: closes.append(a) or (False, 0.0))
    arm = DryRunArm()
    ctx = _ctx(arm, cfg=pipeline_cfg)
    ctx.stop_event.set()
    assert manipulation.run_grip(ctx) is False
    assert not closes, "the jaws closed after a stop"
    assert arm.get_joints()["shoulder_lift"] == REST_POSE["shoulder_lift"], "the arm did not park"


def test_closing_on_empty_jaws_ends_on_the_owner_clock(pipeline_cfg):
    """With nothing between the jaws the close runs to the floor, waits out its
    grace on the twin's sim time, and reports no grip."""
    from so101_sim.world import SimWorld
    from vla_pipeline.behaviors.manipulation import close_with_feedback
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR, REST_POSE
    from vla_pipeline.robot.sim_arm import SimArmClient

    world = SimWorld({"sim": dict(pipeline_cfg.get("sim") or {})}, rest_pose=REST_POSE)
    try:
        for obj in world.objects:
            world.hide(obj)
        arm = SimArmClient({"robot": {"max_relative_target": None}}, world=world)
        t0 = arm.state_time()
        gripped, hold = close_with_feedback(arm, pipeline_cfg["grip"], start_from=75.0)
        assert not gripped and hold == GRIPPER_CLOSED_FLOOR
        assert arm.state_time() - t0 >= float(pipeline_cfg["grip"].get("stall_grace_s", 0.6))
    finally:
        world.close()
