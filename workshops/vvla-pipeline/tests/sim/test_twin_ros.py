# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The workshop behaviors on the notebook twin, over ROS 2.

The context is the section 5 console's (``TwinRig``): ``TwinArmClient``,
the arm camera behind ``FanoutCamera``, the YOLO eyes, and the keyword
parser through ``handle_transcript``. The checks mirror the in-process
``test_workshop_scenarios.py``. Every reading of the twin comes from
``/sim/scene`` and ``/so101/joint_state``.

Tier 2 (ROS 2): ``./run_sim_tests.sh``.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
import pytest

from twin_ros_rig import (
    BODY_JOINTS,
    Probe,
    TwinRig,
    assert_domain_free,
    default_scene_snapshot,
    join_test_domain,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(300)]
pytest.importorskip("rclpy")


@pytest.fixture(scope="module")
def probe(workshop_cfg):
    join_test_domain()
    p = Probe(workshop_cfg["robot"]["state_topic"])
    yield p
    p.close()


@pytest.fixture
def rig(workshop_cfg, eyes, probe):
    detector = eyes("workshop")
    if detector is None:
        pytest.skip("--sim-eyes seg reads the in-process twin's segmentation; this twin is over ROS")
    assert_domain_free("a twin already runs on the test domain")
    r = TwinRig(workshop_cfg, detector, scene=default_scene_snapshot(workshop_cfg))
    yield r
    r.close()
    assert_domain_free("the twin node outlived its test", wait=True)


def _settled_scene(rig) -> dict:
    """The first scene stamped now or later with every prop at rest."""
    return rig.twin.scene.latest(after=time.time(), settled=True, timeout=None)


def _hold_problem(scene) -> str | None:
    """Why the gripper's hold could trip, from the twin's own registers, or ``None``.

    The servo trips only at or above ``Overload_Torque``; ``Torque_Limit``
    caps its output below that, so a lower limit can never trip.
    """
    g = scene["gripper"]
    if g["tripped"]:
        return "the gripper's overload protection tripped"
    if g["torque_limit"] >= 10 * g["overload_torque"]:
        return (f"the hold Torque_Limit {g['torque_limit']} reaches Overload_Torque "
                f"{g['overload_torque']} %, so the grip trips after Protection_Time")
    return None


class _GripLog(logging.Handler):
    """The held gripper goal of every confirmed grasp (``_log_grip``).

    A handler on the logger itself, not ``caplog``: with ROS 2 sourced,
    ROS's launch_testing pytest plugin swaps in its own logger class and
    ``caplog`` captures nothing.
    """

    def __init__(self):
        super().__init__(logging.INFO)
        self.goals: list[float] = []

    def emit(self, record) -> None:
        m = re.match(r"grasp: jaw [-\d.]+, goal ([-\d.]+)", record.getMessage())
        if m:
            self.goals.append(float(m.group(1)))


@pytest.fixture
def grip_log():
    handler = _GripLog()
    log = logging.getLogger("common.manipulation")
    level = log.level
    log.setLevel(logging.INFO)
    log.addHandler(handler)
    yield handler
    log.removeHandler(handler)
    log.setLevel(level)


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_twin_pickup(rig, grip_log, record_property, text, body, spoken):
    """Held out in front, and held with a torque limit that cannot trip the
    gripper's overload protection (the in-process tests watch the hold itself)."""
    start = rig.twin.scene.latest(timeout=None)
    z0 = start["bodies"][body]["xyz"][2]

    rig.run(text)
    end = _settled_scene(rig)
    held = end["bodies"][body]

    holds = grip_log.goals
    record_property("hold_goal", holds[-1] if holds else None)
    print(f"twin grasp hold goal for {body}: {holds}")

    assert rig.said_any(f"Got the {spoken}"), rig.said
    assert held["grasped"], f"{spoken} is not between both fingers; robot said {rig.said}"
    problem = _hold_problem(end)
    assert problem is None, problem
    assert held["xyz"][2] - z0 > 0.03, f"{spoken} rose {1000 * (held['xyz'][2] - z0):.0f} mm"
    x, y, z = held["xyz"]
    assert abs(x - 0.30) < 0.08 and abs(y) < 0.08, f"{spoken} held at ({x:.3f}, {y:.3f}), not in front"
    assert z > 0.08, f"{spoken} was set back down ({z:.3f} m)"
    moved = {
        name: float(np.linalg.norm(np.subtract(end["bodies"][name]["xyz"], pos["xyz"])))
        for name, pos in start["bodies"].items()
        if name != body
    }
    assert all(d < 0.01 for d in moved.values()), f"other props moved (m): {moved}"


def test_twin_drop_it_opens_the_jaws(rig, probe):
    rig.run("pick up the ball")
    assert _settled_scene(rig)["bodies"]["sports_ball"]["grasped"], rig.said
    rig.run("drop it")
    released = _settled_scene(rig)
    assert rig.said_any("Released"), rig.said
    assert probe.joints(after=time.time())["gripper"] > 80.0
    assert not released["bodies"]["sports_ball"]["grasped"]


def test_twin_go_home_parks_the_arm(rig, probe):
    from common.motion import REST_POSE

    rig.twin.arm.go_to({"shoulder_pan": 35.0, "shoulder_lift": -40.0, "elbow_flex": 40.0})
    rig.run("go home")
    joints = probe.joints(after=time.time())
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, off


def test_twin_wave_swings_the_base(rig, probe):
    """``run_wave`` swings ``shoulder_pan`` between WAVE_LEFT (-25) and WAVE_RIGHT (+25)."""
    pans: list[float] = []
    unsubscribe = probe.subscribe(lambda: pans.append(probe.joints()["shoulder_pan"]))
    try:
        rig.run("wave at me")
    finally:
        unsubscribe()
    pan = np.array(pans)
    assert pan.max() - pan.min() > 40.0, f"shoulder_pan spanned {pan.max() - pan.min():.0f} deg"


def test_twin_dance_finishes_parked(rig, probe):
    from common.motion import REST_POSE

    rig.run("dance")
    joints = probe.joints(after=time.time())
    off = {k: round(joints[k] - REST_POSE[k], 1) for k in BODY_JOINTS if abs(joints[k] - REST_POSE[k]) > 4.0}
    assert not off, off


def test_twin_mimic_drives_the_arm_then_stop_holds(rig, probe):
    """Mimic commands reach the twin, the pinch drives the gripper even after a
    grip was locked, and "stop" from another thread halts them.

    The scripted person moves the arm only a few degrees (in-process too), so
    the check is that the server takes the commands, not how far they reach.
    The scripted wrist orbits and the pinch cycles once per
    ``SCRIPT_PERIOD_S``, so each check waits at most one cycle.
    """
    from common.fixtures import SCRIPT_PERIOD_S
    from common.motion import DEFAULT_CONTROL_HZ, motion_budget

    rig.ctx.shared_state.update(grip_lock=True, grip_lock_pos=45.0)
    start = probe.joints()
    started = time.time()
    worker = rig.start("copy my movements")

    def reach(s) -> float:
        return max(abs(s["goal"].get(k, start[k]) - start[k]) for k in BODY_JOINTS)

    probe.motion.wait(
        lambda s: s["cmd_time"] >= started and reach(s) > 2.0, SCRIPT_PERIOD_S,
        "a mimic goal 2 deg from where the arm started",
    )
    assert "grip_lock" not in rig.ctx.shared_state, "mimic left the grip locked"
    jaw: list[float] = []

    def pinch_moved(s) -> bool:
        if s["cmd_time"] >= started and "gripper" in s["goal"]:
            jaw.append(float(s["goal"]["gripper"]))
        return bool(jaw) and max(jaw) - min(jaw) > 5.0

    probe.motion.wait(pinch_moved, SCRIPT_PERIOD_S, "the pinch moving the gripper goal 5 deg")
    rig.stop()
    worker.join()
    ended = time.time()
    tick_s = 1.0 / float(rig.cfg["robot"].get("control_fps", DEFAULT_CONTROL_HZ))
    last = probe.motion.latest()
    rest = probe.motion.wait(
        lambda s: s["stamp"] > ended and not any(s["moving"].values()),
        motion_budget(float(last["eta_s"]), tick_s), "the servos finishing after stop",
    )
    assert rest["cmd_time"] <= ended, "the server took a command after mimic had stopped"
