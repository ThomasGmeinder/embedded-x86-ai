# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The speed a streamed joint is asked for: one it can still stop from.

A stream used to ask for full speed until the joint was one Servo period from
its target. Servo limits acceleration, so the joint arrived too fast, ran past
and came back: a 1 rad step took 1.7 to 5.5 s to settle (2026-11-26).
``tests/moveit/test_arm.py`` times the step on the twin; this is the law.
"""

from __future__ import annotations

import math

import pytest

from vla_pipeline.robot.moveit_arm import SERVO_PERIOD_S, STREAM_BRAKE, STREAM_SPEED, jog_velocities, jog_velocity

pytestmark = pytest.mark.sim

A, V = 7.05637, 2.14152  # the pan joint's max_acceleration and max_velocity
SLOW_A, SLOW_V = 11.81239, 1.12399  # the elbow's


def _one(error: float) -> float:
    return jog_velocities({"pan": error}, {"pan": A}, {"pan": V})["pan"]


def test_far_from_the_target_the_joint_is_asked_for_a_share_of_its_speed_limit():
    assert _one(1.0) == pytest.approx(STREAM_SPEED * V)
    assert _one(-1.0) == pytest.approx(-STREAM_SPEED * V)
    assert 0 < STREAM_SPEED < 1  # the arm's speed swings around the command; the peaks need room


@pytest.mark.parametrize("error", [0.2, 0.1, 0.05, 0.01, 0.002, -0.05])
def test_near_the_target_the_joint_can_stop_within_what_is_left(error):
    speed = _one(error)
    assert math.copysign(1.0, speed) == math.copysign(1.0, error)
    # Braking at STREAM_BRAKE of the joint's acceleration from this speed takes no more than the error.
    assert speed * speed / (2.0 * STREAM_BRAKE * A) <= abs(error) + 1e-12
    assert abs(speed) <= STREAM_SPEED * V + 1e-12


def test_the_speed_falls_as_the_target_comes_near_and_is_zero_on_it():
    speeds = [_one(error) for error in (0.3, 0.1, 0.03, 0.01, 0.001, 0.0)]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[-1] == 0.0


def test_a_last_small_step_is_not_asked_to_overshoot():
    # Closer than one Servo period at the braking speed: the velocity that closes the error exactly.
    error = 1e-6
    assert jog_velocity(error, A) == pytest.approx(min(error / SERVO_PERIOD_S, math.sqrt(2 * STREAM_BRAKE * A * error)))
    assert _one(error) == pytest.approx(jog_velocity(error, A))


def test_the_joints_of_one_stream_keep_their_proportions_and_none_is_over_its_share():
    # Far from both targets each joint alone would ask for far more than its limit. One factor
    # brings them down together, so they stay in the proportion of what each asked for.
    errors = {"pan": 1.0, "elbow": -0.4}
    limits_a = {"pan": A, "elbow": SLOW_A}
    limits_v = {"pan": V, "elbow": SLOW_V}
    alone = {name: jog_velocity(error, limits_a[name]) for name, error in errors.items()}
    asked = jog_velocities(errors, limits_a, limits_v)
    assert asked["pan"] / asked["elbow"] == pytest.approx(alone["pan"] / alone["elbow"])
    shares = {name: abs(asked[name]) / limits_v[name] for name in errors}
    assert max(shares.values()) == pytest.approx(STREAM_SPEED)
    assert all(share <= STREAM_SPEED + 1e-12 for share in shares.values())


def test_a_stream_that_is_on_its_target_asks_for_nothing():
    assert jog_velocities({"pan": 0.0, "elbow": 0.0}, {"pan": A, "elbow": SLOW_A}, {"pan": V, "elbow": SLOW_V}) == {
        "pan": 0.0, "elbow": 0.0,
    }
