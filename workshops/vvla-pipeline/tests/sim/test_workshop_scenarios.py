# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier 1: the workshop's ``common.manipulation`` behaviors on the twin.

Spoken commands go through the project's ``integration.handle_transcript``
and the workshop ``LlamaIntentParser`` against the mock llama-server; the
behaviors run on ``common.motion.SimArmClient`` with ground-truth
detections from the twin's segmentation.
"""

from __future__ import annotations

import pytest

from scripted import Person

pytestmark = [pytest.mark.sim, pytest.mark.slow]

PERSON_LEFT = Person(cx=0.3, wrist=(0.4, 0.45))


def _pick_ball(rig, pos=None, timeout: float = 60.0):
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    z0 = rig.put("ball", pos)[2]
    rig.run([(0.3, "pick up the ball")], timeout=timeout, settle=1.0)
    return z0


@pytest.mark.parametrize("spot", [None, (0.22, -0.04)], ids=["default", "near"])
def test_workshop_pick_up_the_ball(workshop_rig, caplog, spot):
    rig = workshop_rig
    z0 = _pick_ball(rig, None if spot is None else [*spot, 0.015])

    assert "IK reach: sports ball" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    assert rig.said_any("Got the sports ball"), f"robot said {rig.said}"
    assert rig.trace.heights("ball").max() - z0 > 0.03, "ball never rose 3 cm"
    assert rig.trace.grasped_seconds("ball") >= 1.0, "ball not held for 1 s"
    assert not rig.world.is_grasped("ball"), "ball was never released at the handoff"


def test_workshop_pick_ball_out_of_fixed_reach_says_so(workshop_rig):
    rig = workshop_rig
    rig.cfg["pick_place"]["reach"] = "fixed"
    z0 = _pick_ball(rig, [0.19, 0.0, 0.015])

    assert rig.said_any("couldn't get a grip on the sports ball"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.trace.heights("ball").max() - z0 < 0.03, "ball was lifted"
