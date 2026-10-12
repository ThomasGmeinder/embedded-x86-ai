# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier 1: the workshop's ``common.manipulation`` behaviors on the twin.

Spoken commands go through the project's ``integration.handle_transcript``
and the workshop ``LlamaIntentParser`` against the mock llama-server; the
behaviors run on ``common.motion.SimArmClient`` and see through the
workshop's YOLOv26s on the twin's rendered arm camera (``--sim-eyes seg``:
ground-truth boxes from its segmentation).
"""

from __future__ import annotations

import pytest

from scripted import Person

pytestmark = [pytest.mark.sim, pytest.mark.slow]

PERSON_LEFT = Person(cx=0.3, wrist=(0.4, 0.45))


def _pick_ball(rig, pos=None, timeout: float = 60.0):
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    z0 = rig.put("sports_ball", pos)[2]
    rig.run([(0.3, "pick up the ball")], timeout=timeout, settle=1.0)
    return z0


@pytest.mark.parametrize("spot", [None, (0.22, -0.04)], ids=["default", "near"])
def test_workshop_pick_up_the_ball(workshop_rig, caplog, spot):
    rig = workshop_rig
    z0 = _pick_ball(rig, None if spot is None else [*spot, 0.015])

    assert "IK reach: sports ball" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    assert rig.said_any("Got the sports ball"), f"robot said {rig.said}"
    assert rig.trace.heights("sports_ball").max() - z0 > 0.03, "ball never rose 3 cm"
    assert rig.trace.grasped_seconds("sports_ball") >= 1.0, "ball not held for 1 s"
    assert not rig.world.is_grasped("sports_ball"), "ball was never released at the handoff"


def test_workshop_pick_up_toothbrush_aligns_wrist(workshop_rig, caplog):
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()  # the notebook console's scene: all four props
    z0 = rig.world.body_pos("toothbrush")[2]
    rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=1.0)

    assert "wrist alignment:" in caplog.text
    _, roll = rig.trace.series("wrist_roll")
    assert abs(roll).max() > 60.0, "wrist never turned across the toothbrush"
    assert rig.said_any("Got the toothbrush"), f"robot said {rig.said}"
    assert rig.trace.heights("toothbrush").max() - z0 > 0.03
    assert rig.trace.grasped_seconds("toothbrush") >= 1.0


def test_workshop_pick_ball_out_of_fixed_reach_says_so(workshop_rig):
    rig = workshop_rig
    rig.cfg["pick_place"]["reach"] = "fixed"
    z0 = _pick_ball(rig, [0.19, 0.0, 0.015])

    assert rig.said_any("couldn't get a grip on the sports ball"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.trace.heights("sports_ball").max() - z0 < 0.03, "ball was lifted"
