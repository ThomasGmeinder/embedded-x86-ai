# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Tier 1: ``vla_pipeline`` behaviors driven by scripted user signals.

Spoken commands go through the real ``Dispatcher`` (grace window, stop fast
path, noise gate) and the real ``LlamaIntentParser`` HTTP client against the
mock llama-server; behaviors run unchanged on ``SimArmClient`` (the
``DirectArm`` transport over the twin). Assertions are on the simulated
physics: joint traces, gripper current, and whether objects are held.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripted import Hand, Person

pytestmark = pytest.mark.sim

BODY = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def _rest():
    from vla_pipeline.robot.arm_interface import REST_POSE

    return REST_POSE


def _assert_at(rig, pose: dict, tol: float = 4.0, joints=BODY):
    cur = rig.joints()
    off = {k: round(cur[k] - pose[k], 1) for k in joints if abs(cur[k] - pose[k]) > tol}
    assert not off, f"not at pose (deg off): {off}"


def _assert_within_limits(rig, limits: dict, slack: float = 2.0):
    for s in rig.trace.samples:
        for k, (lo, hi) in limits.items():
            v = s.joints[k]
            assert lo - slack <= v <= hi + slack, f"{k}={v:.1f} outside [{lo}, {hi}] at t={s.t:.2f}s"


# ---------------------------------------------------------------------------
# scripted behaviors
# ---------------------------------------------------------------------------
def test_wave(pipeline_rig, mock_llm):
    rig = pipeline_rig
    rig.run([(0.3, "wave at me")], timeout=25)

    # the LLM client sent a grammar-constrained, deterministic request
    p = mock_llm.last_payload
    assert p and p.get("grammar") and p.get("temperature") == 0.0
    assert rig.event_times("behavior_start", intent="wave")
    assert rig.event_times("behavior_end", intent="wave", ok=True)

    _, roll = rig.trace.series("wrist_roll")
    assert roll.max() > 35.0 and roll.min() < -35.0, "wrist never swung through +/-45"
    _assert_at(rig, _rest())


def test_home_from_anywhere(pipeline_rig):
    rig = pipeline_rig
    rig.world.reset({"shoulder_pan": 40.0, "shoulder_lift": -40.0, "elbow_flex": 50.0, "wrist_flex": 20.0,
                     "wrist_roll": 0.0})
    rig.run([(0.3, "go home")], timeout=15)
    assert rig.event_times("behavior_end", intent="home", ok=True)
    _assert_at(rig, _rest())


def test_dance_visits_moves_and_parks(pipeline_rig):
    from vla_pipeline.robot.arm_interface import JOINT_LIMITS

    rig = pipeline_rig
    rig.run([(0.3, "let's dance")], timeout=30)
    assert rig.event_times("behavior_end", intent="dance", ok=True)
    spans = {k: np.ptp(rig.trace.series(k)[1]) for k in BODY}
    assert sum(v > 20.0 for v in spans.values()) >= 2, f"dance barely moved: {spans}"
    _assert_within_limits(rig, JOINT_LIMITS)
    _assert_at(rig, _rest())


# ---------------------------------------------------------------------------
# dispatcher safety properties
# ---------------------------------------------------------------------------
def test_stop_mid_wave(pipeline_rig):
    """'stop' halts the move within a control tick; the wave never resumes, the arm parks."""
    rig = pipeline_rig
    rig.run([(0.3, "wave"), (1.6, "stop")], timeout=25)
    t_stop = rig.transcripts.sent_at("stop")
    assert rig.event_times("stop"), "stop fast path not taken"
    assert len(rig.event_times("parsed")) == 1, "stop must not reach the LLM"

    # the commanded goal leaves the wave target within 100 ms (halt = hold here)
    t, goal = rig.trace.series("elbow_flex", goal=True)
    before = goal[t < t_stop][-1]
    after = goal[(t > t_stop) & (t < t_stop + 0.15)]
    assert np.any(np.abs(after - before) > 1.0), "goal did not change within 150 ms of 'stop'"

    # no further wave strokes after the stop (wrist_roll goal never hits +/-45)
    tr, roll_goal = rig.trace.series("wrist_roll", t0=t_stop, goal=True)
    stroke = np.abs(np.abs(roll_goal) - 45.0) <= 2.0
    assert not np.any(stroke & (tr > t_stop + 0.15)), (
        f"wave resumed after stop: +/-45 roll goal at t-t_stop={np.round(tr[stroke] - t_stop, 3)}")
    assert rig.event_times("behavior_end", intent="wave", stopped=True)
    _assert_at(rig, _rest())


def test_noise_and_grace_window_cause_no_motion(pipeline_rig, mock_llm):
    rig = pipeline_rig
    start = rig.joints()
    rig.run([(0.2, "wave"), (1.3, "thank you."), (1.6, "you"), (1.9, "...")], timeout=10,
            startup_grace_s=1.0)
    reasons = [d["reason"] for _t, e, d in rig.events if e == "ignored"]
    assert reasons == ["grace", "noise", "noise", "noise"], reasons
    assert mock_llm.request_count == 0, "gated transcripts must never reach the LLM"
    _assert_at(rig, start, tol=1.0, joints=BODY + ("gripper",))


# ---------------------------------------------------------------------------
# grip: gripper current from contact physics
# ---------------------------------------------------------------------------
def _present():
    from vla_pipeline.behaviors.grip import PRESENT_POSE

    return PRESENT_POSE


@pytest.mark.xfail(reason=(
    "twin: grip.py locks at the first position stall (~70 deg) and the cube ends "
    "resting on the fixed jaw only - stall-vs-contact geometry needs calibrating"))
def test_grip_object_handed_over(pipeline_rig):
    rig = pipeline_rig
    # the "user" puts the cube between the jaws a second after the arm presents
    rig.when_near(_present(), lambda: rig.stage("cube", _present()), delay=1.0)
    rig.run([(0.3, "hold this for me")], timeout=25)

    assert rig.ctx.shared_state.get("grip_lock"), f"grip not reported; robot said {rig.said}"
    assert rig.world.is_grasped("cube"), f"cube not between both jaws: {rig.world.contact_report('cube')}"
    assert rig.trace.grasped_seconds("cube") >= 1.0, "cube not held for 1 s"
    # grip.py stops closing at the stall, so the holding current sits below
    # load_threshold - but well clear of the idle current
    cur = rig.trace.currents()
    assert cur.max() > 5 * np.median(cur[:25]), "squeezing the cube drew no extra current"
    assert rig.said_any("Object gripped")


def test_grip_empty_jaws_reports_nothing(pipeline_rig):
    # the position-stall check still fires on empty jaws (the close is commanded
    # below the 43 clamp floor); the arm-camera cross-check is what rejects it
    rig = pipeline_rig
    rig.run([(0.3, "hold this")], timeout=25)
    assert rig.trace.currents().max() < 120.0  # the physics agrees: nothing was squeezed
    assert not rig.ctx.shared_state.get("grip_lock"), "reported a grip with empty jaws"


# ---------------------------------------------------------------------------
# pick and place / fetch (segmentation detector + scripted person)
# ---------------------------------------------------------------------------
PERSON_LEFT = Person(cx=0.3, wrist=(0.4, 0.45))


def _descend_pose(cfg, pan: float = 0.0) -> dict:
    pp = cfg["behaviors"]["pick_place"]
    return {"shoulder_pan": pan, **pp["approach_pose"], "shoulder_lift": float(pp["descend_lift"]),
            "wrist_roll": 0.0}


@pytest.mark.slow
@pytest.mark.xfail(reason=(
    "twin: at wrist_roll 0 the wrist_cam image is rotated ~90 deg, so the pan "
    "visual servo steers on the wrong image axis - verify against the real arm camera"))
def test_pick_up_the_ball(pipeline_rig):
    rig = pipeline_rig
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.stage("ball", _descend_pose(rig.cfg, pan=8.0))
    z0 = rig.world.body_pos("ball")[2]
    rig.run([(0.3, "pick up the ball")], timeout=60)

    assert rig.detector.calls > 0, "behavior never looked through the arm camera"
    assert rig.said_any("Got the sports ball"), f"robot said {rig.said}"
    assert rig.trace.heights("ball").max() - z0 > 0.03, "ball never rose 3 cm"
    assert rig.trace.grasped_seconds("ball") >= 1.0, "ball not held for 1 s"
    assert not rig.world.is_grasped("ball"), "ball was never released at the handoff"
    # handed toward the person on the left of the operator camera (mirrored -> +pan)
    _, pan = rig.trace.series("shoulder_pan")
    assert pan.max() > 10.0


def test_pick_unknown_object_fails_safely(pipeline_rig):
    rig = pipeline_rig
    rig.stage("pen", _descend_pose(rig.cfg))
    rig.run([(0.3, "pick up the pen")], timeout=25)
    assert rig.said_any("don't know what a pen looks like"), rig.said
    assert not any(s.grasped.get("pen") for s in rig.trace.samples)
    _assert_at(rig, _rest())


@pytest.mark.slow
@pytest.mark.xfail(reason=(
    "twin: the block lifts ~3 mm then drops - same jaw/contact geometry issue "
    "as test_grip_object_handed_over"))
def test_fetch_block_hands_to_person(pipeline_rig):
    rig = pipeline_rig
    fb = rig.cfg["behaviors"]["fetch_block"]
    rig.pose.keys = [(0.0, [PERSON_LEFT])]
    rig.stage("cube", fb["block_pose"])
    z0 = rig.world.body_pos("cube")[2]
    rig.run([(0.3, "bring me the block")], timeout=60)

    assert rig.event_times("behavior_end", intent="fetch_block", ok=True)
    assert rig.trace.heights("cube").max() - z0 > 0.03, "block never lifted"
    assert rig.trace.grasped_seconds("cube") >= 1.0
    assert not rig.world.is_grasped("cube"), "block not released"
    _assert_at(rig, _rest())


# ---------------------------------------------------------------------------
# gesture mimic (scripted person + hand -> mapper -> twin)
# ---------------------------------------------------------------------------
def _mimic_limits():
    from vla_pipeline.robot.arm_interface import JOINT_LIMITS

    return JOINT_LIMITS


@pytest.mark.slow
def test_mimic_sweep_pinch_roll(pipeline_rig):
    rig = pipeline_rig
    floor = float(rig.cfg["sim"]["gripper_map"]["closed_cal"])
    rig.pose.keys = [
        (0.0, [Person(wrist=(0.5, 0.45))]),
        (2.0, [Person(wrist=(0.5, 0.45))]),
        (5.0, [Person(wrist=(0.8, 0.45))]),  # sweep to the right of the frame
        (6.0, [Person(wrist=(0.8, 0.45))]),
    ]
    rig.hands.keys = [
        (0.0, Hand(pinch=1.3, roll=0.0)),
        (6.0, Hand(pinch=1.3, roll=0.0)),
        (6.5, Hand(pinch=0.05, roll=0.0)),  # pinch -> gripper to the floor
        (8.0, Hand(pinch=0.05, roll=0.0)),
        (9.0, Hand(pinch=0.05, roll=60.0)),  # roll one way...
        (10.5, Hand(pinch=0.05, roll=60.0)),
        (11.5, Hand(pinch=0.05, roll=-60.0)),  # ...then the other
        (13.0, Hand(pinch=0.05, roll=-60.0)),
    ]
    rig.run([(0.3, "copy my movements"), (13.0, "stop")], timeout=30)

    tr = rig.trace
    _, pan_a = tr.series("shoulder_pan", 1.5, 2.0, goal=True)
    t, pan_s = tr.series("shoulder_pan", 2.0, 5.5, goal=True)
    _, pan_b = tr.series("shoulder_pan", 5.5, 6.0, goal=True)
    assert abs(pan_b.mean() - pan_a.mean()) > 25.0, "pan did not follow the wrist sweep"
    corr = np.corrcoef(t, pan_s)[0, 1]
    assert abs(corr) > 0.9, f"pan not steady through the sweep (r={corr:.2f})"

    _, grip = tr.series("gripper", 7.5, 8.0)
    assert grip.mean() < floor + 4.0, f"pinch did not close the gripper (got {grip.mean():.1f})"

    _, roll_p = tr.series("wrist_roll", 10.0, 10.5, goal=True)
    _, roll_n = tr.series("wrist_roll", 12.5, 13.0, goal=True)
    assert np.sign(roll_p.mean()) != np.sign(roll_n.mean()) and min(abs(roll_p.mean()), abs(roll_n.mean())) > 15
    _assert_within_limits(rig, _mimic_limits())
    _assert_at(rig, _rest())


@pytest.mark.slow
def test_mimic_hand_lost_then_person_lost(pipeline_rig):
    rig = pipeline_rig
    rig.pose.keys = [
        (0.0, [Person(wrist=(0.5, 0.45))]),
        (2.0, [Person(wrist=(0.5, 0.45))]),
        (4.5, [Person(wrist=(0.25, 0.45))]),  # wrist moves with NO hand seen
        (5.5, [Person(wrist=(0.25, 0.45))]),
        (5.6, []),  # person leaves: the arm must hold its pose
        (8.0, []),
    ]
    rig.hands.keys = [(0.0, Hand(pinch=1.3)), (1.5, Hand(pinch=1.3)), (1.6, None)]
    rig.run([(0.3, "mirror my arm"), (8.0, "stop")], timeout=20)

    tr = rig.trace
    _, a = tr.series("shoulder_pan", 1.6, 2.0, goal=True)
    _, b = tr.series("shoulder_pan", 5.0, 5.5, goal=True)
    assert abs(b.mean() - a.mean()) > 20.0, "YOLO wrist fallback did not drive pan"
    t, held = tr.series("shoulder_pan", 6.2, 7.9, goal=True)
    assert np.ptp(held) < 1.5, f"arm drifted with nobody in frame ({np.ptp(held):.1f} deg)"


def test_mimic_follows_nearest_person(pipeline_rig):
    rig = pipeline_rig
    far_right = Person(cx=0.75, scale=0.5, wrist=(0.85, 0.45))
    near_left = Person(cx=0.35, scale=1.1, wrist=(0.2, 0.45))
    rig.pose.keys = [(0.0, [far_right, near_left])]
    rig.hands.keys = [(0.0, None)]
    rig.run([(0.3, "follow me"), (3.5, "stop")], timeout=15)
    _, pan = rig.trace.series("shoulder_pan", 2.5, 3.4, goal=True)
    # mirrored mapping: a wrist on the LEFT of the image pans the arm positive
    assert pan.mean() > 20.0, f"followed the wrong person (pan {pan.mean():.1f})"
