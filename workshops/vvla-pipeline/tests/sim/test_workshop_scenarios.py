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
    if pos is None:
        rig.default_scene()  # the notebook console's scene: all four props
        z0 = rig.world.body_pos("sports_ball")[2]
    else:
        z0 = rig.put("sports_ball", pos)[2]
    rig.run([(0.3, "pick up the ball")], timeout=timeout, settle=1.0)
    return z0


def _assert_others_in_place(rig, handled: str):
    moved = rig.disturbed(handled)
    assert not moved, f"the arm moved objects it was not asked to pick up (m): {moved}"


def _assert_held_out(rig, body: str, spoken: str, z0: float):
    assert rig.said_any(f"Got the {spoken}"), f"robot said {rig.said}"
    heights = rig.trace.heights(body)
    assert heights.max() - z0 > 0.03, f"{spoken} never rose 3 cm"
    assert heights.max() > 0.10, f"{spoken} peaked at {heights.max():.3f} m, below the other props"
    assert rig.trace.grasped_seconds(body) >= 1.0, f"{spoken} not held for 1 s"
    assert rig.world.is_grasped(body), f"{spoken} was dropped"
    end = rig.world.body_pos(body)
    assert abs(end[0] - 0.30) < 0.08 and abs(end[1]) < 0.08, (
        f"{spoken} ended at ({end[0]:.3f}, {end[1]:.3f}), not held out in front"
    )
    assert end[2] > 0.08, f"{spoken} was set back down ({end[2]:.3f} m)"
    _assert_others_in_place(rig, body)


@pytest.mark.parametrize("spot", [None, (0.22, -0.04)], ids=["default", "near"])
def test_workshop_pick_up_the_ball(workshop_rig, caplog, spot):
    rig = workshop_rig
    z0 = _pick_ball(rig, None if spot is None else [*spot, 0.015])

    assert "IK reach: sports ball" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    _assert_held_out(rig, "sports_ball", "sports ball", z0)


def test_workshop_pick_up_toothbrush_aligns_wrist(workshop_rig, caplog):
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()  # the notebook console's scene: all four props
    z0 = rig.world.body_pos("toothbrush")[2]
    rig.run([(0.3, "pick up the toothbrush")], timeout=90, settle=1.0)

    assert "wrist alignment:" in caplog.text
    _, roll = rig.trace.series("wrist_roll")
    assert abs(roll).max() > 60.0, "wrist never turned across the toothbrush"
    _assert_held_out(rig, "toothbrush", "toothbrush", z0)


def test_workshop_pick_up_the_bottle(workshop_rig, caplog):
    rig = workshop_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.default_scene()
    z0 = rig.world.body_pos("bottle")[2]
    rig.run([(0.3, "pick up the bottle")], timeout=90, settle=1.0)

    assert "IK reach: bottle" in caplog.text and "-> solved" in caplog.text, "IK reach not used"
    assert "using the fixed approach" not in caplog.text
    _assert_held_out(rig, "bottle", "bottle", z0)


def test_workshop_pick_ball_out_of_fixed_reach_says_so(workshop_rig):
    rig = workshop_rig
    rig.cfg["pick_place"]["reach"] = "fixed"
    z0 = _pick_ball(rig, [0.19, 0.0, 0.015])

    assert rig.said_any("couldn't get a grip on the sports ball"), rig.said
    assert not rig.said_any("Got the"), rig.said
    assert rig.trace.heights("sports_ball").max() - z0 < 0.03, "ball was lifted"
