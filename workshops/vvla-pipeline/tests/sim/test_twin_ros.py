# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The pipeline's behaviors on the notebook twin, over ROS 2.

The context is the section 4 console's (``TwinRig``): ``MoveItArm`` on the
MuJoCo bringup, the arm camera behind ``FanoutCamera``, the YOLO eyes, and
the keyword parser through ``handle_transcript``. Prop poses come from
``/free_joint_states``. Joints are radians under the URDF names.

Tier 2 (ROS 2): ``./run_sim_tests.sh``.
"""

from __future__ import annotations

import os
import threading

import numpy as np
import pytest

from twin_ros_rig import (
    TEST_DOMAIN_ID,
    TwinRig,
    assert_domain_free,
    default_scene_snapshot,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(300)]
pytest.importorskip("rclpy")


@pytest.fixture(scope="session")
def _rig_session(request):
    """One MoveItPy for every behavior test. Each test parks it; none constructs another."""
    from vla_pipeline.utils.config import load_config
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    if request.config.getoption("--sim-eyes") == "seg":
        pytest.skip("--sim-eyes seg reads the in-process twin's segmentation; this twin is over ROS")
    cfg = load_config()
    cfg["dance"]["duration_s"] = 5.0
    cfg.setdefault("sim", {})["require_hw_gl"] = False
    os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    # The NPU session must not be the process's first user of the LARGE_DATA
    # transport. A DDS participant has to exist before it, or this process
    # never discovers the bringup. The default context stays free for MoveItPy.
    from vvla_workshop.twin_ros import use_private_domain

    use_private_domain(TEST_DOMAIN_ID, private_context=True)
    assert_domain_free("a twin already runs on the test domain")
    device = request.config.getoption("--sim-yolo-device")
    detector = ObjectDetector(cfg, device=device)
    providers = detector.session.get_providers()
    if device == "npu" and "VitisAIExecutionProvider" not in providers:
        pytest.fail(
            f"--sim-yolo-device npu, but the detector runs on {providers}: "
            "source scripts/ryzen_ai_env.sh with RYZEN_AI_WHEELS set",
            pytrace=False,
        )
    session = TwinRig(cfg, detector, scene=default_scene_snapshot(cfg))
    try:
        yield session
    finally:
        session.close()
        assert_domain_free("the twin node outlived its test", wait=True)


@pytest.fixture
def rig(_rig_session):
    """The process's MoveIt twin, parked and with the previous behavior's state cleared."""
    _rig_session.said.clear()
    _rig_session._crashes.clear()
    _rig_session.ctx.shared_state.clear()
    _rig_session.ctx.stop_event.clear()
    _rig_session.ctx._stops = 0
    _rig_session.twin.reset_scene()
    yield _rig_session


def _scene_after(rig, stamp: float, *, settled: bool = False) -> dict:
    """The next scene on the twin's clock after ``stamp``."""
    scene = rig.twin.scene.latest(after=stamp, settled=settled, timeout=15.0)
    assert scene is not None, "no scene after the behavior"
    return scene


def _parked_home(arm) -> None:
    """The arm joints sit on the SRDF ``home`` pose."""
    joints = arm.get_joints()
    for name, value in arm.named_state("home").items():
        assert joints[name] == pytest.approx(value, abs=0.08), (name, joints[name], value)


def _pans(arm):
    """Subscribe to the base joint for the duration of a ``with`` block."""
    from sensor_msgs.msg import JointState

    samples: list[float] = []

    def on_state(msg) -> None:
        names = list(msg.name)
        if "shoulder_pan_joint" in names:
            samples.append(msg.position[names.index("shoulder_pan_joint")])

    sub = arm.node.create_subscription(JointState, "/joint_states", on_state, 10)
    return samples, sub


@pytest.mark.parametrize(
    "text, body, spoken",
    [
        ("pick up the ball", "sports_ball", "sports ball"),
        ("pick up the bottle", "bottle", "bottle"),
        ("pick up the toothbrush", "toothbrush", "toothbrush"),
    ],
    ids=["ball", "bottle", "toothbrush"],
)
def test_twin_pickup(rig, record_property, text, body, spoken):
    """Once the props have stopped, the object is still up and the others have not moved.

    One sample taken while a slipped object is still in the air is not a hold.
    The free-joint topic does not report contact, so the stopped height is the check.
    """
    start = rig.twin.scene.latest(timeout=5.0)
    assert start is not None, "the twin published no scene"
    z0 = start["bodies"][body]["xyz"][2]

    rig.run(text)
    end = _scene_after(rig, start["stamp"], settled=True)
    held = end["bodies"][body]
    record_property("held_xyz", held["xyz"])

    assert rig.said_any(f"Got the {spoken}"), rig.said
    # The pickup's shoulder lift raises the object about 45 mm. It does not
    # climb back to the pre-grasp height. The same 30 mm bar is the bringup hold.
    rise = held["xyz"][2] - z0
    assert rise > 0.03, f"{spoken} rose {1000 * rise:.0f} mm ({held['xyz']})"
    moved = {
        name: float(np.linalg.norm(np.subtract(end["bodies"][name]["xyz"], pos["xyz"])))
        for name, pos in start["bodies"].items()
        if name != body
    }
    assert all(d < 0.02 for d in moved.values()), f"other props moved (m): {moved}"


def test_twin_drop_it_opens_the_jaws(rig):
    start = rig.twin.scene.latest(timeout=5.0)
    assert start is not None
    rig.run("pick up the ball")
    held = _scene_after(rig, start["stamp"], settled=True)
    above = float(rig.ctx.cfg["pick_place"]["pregrasp_above_m"])
    rise = held["bodies"]["sports_ball"]["xyz"][2] - start["bodies"]["sports_ball"]["xyz"][2]
    assert rise > above - 0.015, (rise, rig.said)
    rig.run("drop it")
    released = _scene_after(rig, held["stamp"], settled=True)
    assert rig.said_any("Released"), rig.said
    jaw = rig.twin.arm.gripper_joint
    assert rig.twin.arm.get_joints()[jaw] == pytest.approx(rig.twin.arm.named_state("open")[jaw], abs=0.05)
    assert released["bodies"]["sports_ball"]["xyz"][2] < held["bodies"]["sports_ball"]["xyz"][2] - 0.03


def test_twin_go_home_parks_the_arm(rig):
    assert rig.twin.arm.go_to({"shoulder_pan_joint": 0.6}) == "arrived"
    rig.run("go home")
    _parked_home(rig.twin.arm)


def test_twin_wave_swings_the_base(rig):
    """``run_wave`` swings the base between the left and right wave poses (±25 deg)."""
    samples, sub = _pans(rig.twin.arm)
    try:
        rig.run("wave at me")
    finally:
        rig.twin.arm.node.destroy_subscription(sub)
    assert samples, "no joint states during the wave"
    span = max(samples) - min(samples)
    assert span > 0.70, f"shoulder_pan_joint spanned {span:.3f} rad"


def test_twin_dance_finishes_parked(rig):
    rig.run("dance")
    _parked_home(rig.twin.arm)


def test_twin_mimic_drives_the_arm_then_stop_holds(rig):
    """Mimic moves the arm, the pinch moves the gripper after a locked grip, and stop parks at home."""
    from sensor_msgs.msg import JointState

    from vvla_workshop.fixtures import SCRIPT_PERIOD_S

    rig.ctx.shared_state.update(grip_lock=True, grip_lock_pos=45.0)
    start = rig.twin.arm.get_joints()["shoulder_pan_joint"]
    cond = threading.Condition()
    pans: list[float] = []
    jaws: list[float] = []

    def on_state(msg) -> None:
        names = list(msg.name)
        jaw = rig.twin.arm.gripper_joint
        if "shoulder_pan_joint" not in names or jaw not in names:
            return
        with cond:
            pans.append(msg.position[names.index("shoulder_pan_joint")])
            jaws.append(msg.position[names.index(jaw)])
            cond.notify_all()

    sub = rig.twin.arm.node.create_subscription(JointState, "/joint_states", on_state, 10)
    worker = rig.start("copy my movements")
    try:
        with cond:
            assert cond.wait_for(
                lambda: any(abs(pan - start) > 0.03 for pan in pans), SCRIPT_PERIOD_S,
            ), "mimic did not move the base"
        assert "grip_lock" not in rig.ctx.shared_state, "mimic left the grip locked"
        with cond:
            assert cond.wait_for(
                lambda: bool(jaws) and max(jaws) - min(jaws) > 0.05, SCRIPT_PERIOD_S,
            ), "pinch did not move the gripper"
    finally:
        rig.twin.arm.node.destroy_subscription(sub)
    rig.stop()
    worker.join()
    _parked_home(rig.twin.arm)
