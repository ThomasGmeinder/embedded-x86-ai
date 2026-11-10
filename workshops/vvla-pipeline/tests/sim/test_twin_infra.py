# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The notebook twin's plumbing: the MuJoCo MoveIt bringup and ``vvla_workshop.twin_ros``.

Tier 2 (ROS 2): ``./run_sim_tests.sh``.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys

import pytest

from twin_ros_rig import (
    TEST_DOMAIN_ID,
    assert_domain_free,
    default_scene_snapshot,
    quat_close,
    wait_exit,
    xml_poses,
)

pytestmark = [pytest.mark.sim, pytest.mark.hw, pytest.mark.slow, pytest.mark.timeout(240)]
pytest.importorskip("rclpy")


@pytest.fixture
def no_twin():
    """No twin on the test domain before the test, and none left after it."""
    assert_domain_free("a twin already runs on the test domain")
    yield
    assert_domain_free("a twin node outlived its test", wait=True)


@pytest.fixture(scope="session")
def _moveit_twin():
    """The one MoveItPy for this process. A second constructor segfaults on destruction."""
    from vla_pipeline.utils.config import load_config
    from vvla_workshop.twin_ros import TwinSession

    # Same adjustments as the pipeline_cfg fixture. A session fixture cannot
    # depend on that function-scoped fixture.
    cfg = load_config()
    cfg["dance"]["duration_s"] = 5.0
    cfg.setdefault("sim", {})["require_hw_gl"] = False
    assert_domain_free("a twin already runs on the test domain")
    os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    session = TwinSession(cfg)
    yield session
    session.close()
    assert_domain_free("a twin node outlived its test", wait=True)


@pytest.fixture
def twin(_moveit_twin):
    """The process's MoveIt twin, parked, without constructing another MoveItPy.

    Reset takes longer than the camera's staleness window, so wait for a frame
    captured after the arm is parked. ``read()`` would otherwise report no frame.
    """
    import time

    mark = time.time()
    _moveit_twin.arm.reset()
    cam = _moveit_twin.armcam
    ok, _frame, _stamp = cam.frame_arrived_after(mark, 5.0)
    if not ok:
        raise AssertionError(
            f"no arm camera frame after reset; joint stamp {_moveit_twin.arm.state_time():.3f}; "
            f"last frame stamp {cam._stamp:.3f}; last arrival {cam._arrived:.3f}"
        )
    yield _moveit_twin


def test_frames_are_stamped_after_the_arm_settles(twin):
    """A frame taken after the arm arrives carries a later capture stamp, the twin's clock."""
    from vla_pipeline.console import FanoutCamera

    assert twin.arm.go_to({"shoulder_pan_joint": 0.35}) == "arrived"
    settled_at = twin.arm.state_time()
    ok, frame = twin.armcam.read_after(settled_at)
    assert ok and frame is not None
    assert twin.armcam.last_stamp > settled_at

    fan = FanoutCamera(twin.armcam, own=False)
    try:
        assert twin.arm.go_to({"shoulder_pan_joint": -0.35}) == "arrived"
        settled_at = twin.arm.state_time()
        ok, frame = fan.read_after(settled_at)
        assert ok and frame is not None
        assert fan._wall > settled_at
    finally:
        fan.close()


def test_the_props_are_on_free_joint_states(twin):
    """Each prop is a free joint, at rest, and not reported as grasped."""
    from so101_sim import PROP_BODIES as PROPS

    scene = twin.scene.latest(settled=True, timeout=5.0)
    assert scene is not None, "no settled scene"
    assert set(PROPS) <= scene["bodies"].keys()
    for name in PROPS:
        assert scene["bodies"][name]["xyz"]
        assert not scene["bodies"][name]["grasped"], name


def test_a_goal_sent_as_soon_as_the_session_is_ready_is_taken(twin):
    """The session is ready once the controllers are; a goal sent at once arrives."""
    assert twin.arm.go_to({"shoulder_pan_joint": 0.26}) == "arrived"
    assert twin.arm.get_joints()["shoulder_pan_joint"] == pytest.approx(0.26, abs=0.05)


_OTHER_DOMAIN = r"""
import sys, threading
import rclpy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
rclpy.init()
node = rclpy.create_node("other_domain_probe")
pub = node.create_publisher(JointState, "/so101/joint_state", 1)


def publish():
    msg = JointState()
    msg.name = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    msg.position = [0.0] * 6
    pub.publish(msg)


def report():
    sys.stdin.read()  # the test closes stdin once the twin is up
    print("SEEN", " ".join(sorted(name for name, _types in node.get_topic_names_and_types())), flush=True)
    rclpy.shutdown()


node.create_timer(0.05, publish)
threading.Thread(target=report, daemon=True).start()
print("UP", flush=True)
try:
    rclpy.spin(node)
except ExternalShutdownException:
    pass
"""


def test_the_twin_domain_is_private(pipeline_cfg, no_twin):
    """A joint-state publisher on another domain is invisible to the twin, and the
    twin to it: once the twin and its clients are up, that domain lists none of its topics."""
    from vvla_workshop.twin_ros import TwinSession

    env = {**os.environ, "ROS_DOMAIN_ID": str(TEST_DOMAIN_ID + 1)}
    other = subprocess.Popen(
        [sys.executable, "-c", _OTHER_DOMAIN], env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        for line in other.stdout:
            if line.startswith("UP"):
                break
        else:
            pytest.fail("the other domain's publisher exited before it started")
        os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
        session = TwinSession(pipeline_cfg)
        try:
            joints = session.arm.get_joints()
            assert joints["shoulder_lift_joint"] == pytest.approx(-1.68424, abs=0.05), joints
            assert session.scene.latest(timeout=5.0) is not None
            other.stdin.close()
            out = other.stdout.read()
            other.wait()
        finally:
            session.close()
    finally:
        if other.poll() is None:
            other.kill()
    seen = next(line for line in out.splitlines() if line.startswith("SEEN")).split()[1:]
    assert "/sim/props_settled" not in seen and "/cameras/arm/image_raw" not in seen, seen


def test_a_second_twin_on_the_domain_is_refused(pipeline_cfg, twin):
    from vvla_workshop.twin_ros import TwinSession

    before = twin.scene.latest(timeout=5.0)
    assert before is not None, "the twin published no scene"
    with pytest.raises(RuntimeError, match="already runs on ROS domain"):
        TwinSession(pipeline_cfg)
    # The header stamp is the twin's clock, so a newer scene is after that stamp.
    assert twin.scene.latest(after=before["stamp"], timeout=5.0) is not None, "the first twin stopped"
    assert twin.armcam.read()[0]


def test_the_node_dies_with_its_kernel(no_twin):
    from jupyter_client import KernelManager

    km = KernelManager(kernel_name="vvla-workshop")
    env = os.environ.copy()
    env["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    km.start_kernel(env=env)
    kc = km.client()
    kc.start_channels()
    out: list[str] = []
    try:
        kc.wait_for_ready(timeout=None)
        code = (
            "from vla_pipeline.utils.config import load_config\n"
            "from vvla_workshop.twin_ros import TwinSession\n"
            "twin = TwinSession(load_config())\n"
            "print('TWIN_PID', twin.pid, twin.controllers_pid)\n"
        )

        def hook(msg):
            if msg["msg_type"] == "stream":
                out.append(msg["content"]["text"])
            elif msg["msg_type"] == "error":
                out.append("\n".join(msg["content"]["traceback"]))

        reply = kc.execute_interactive(code, timeout=None, output_hook=hook)
        assert reply["content"]["status"] == "ok", "".join(out)
        pid, controllers = map(int, next(t for t in "".join(out).split("\n") if t.startswith("TWIN_PID")).split()[1:])
        os.kill(km.provisioner.process.pid, signal.SIGKILL)
        assert wait_exit(pid), "the twin node outlived its kernel"
        assert wait_exit(controllers), "the controllers outlived their kernel"
    finally:
        kc.stop_channels()
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass


def test_reset_service_restores_the_scene_and_parks_the_arm(pipeline_cfg, no_twin):
    """Reset puts the arm on the SRDF home pose and the props back on the scene."""
    from vvla_workshop.twin_ros import TwinSession

    os.environ["TWIN_ROS_DOMAIN_ID"] = str(TEST_DOMAIN_ID)
    moved = default_scene_snapshot(pipeline_cfg)
    moved["bodies"]["sports_ball"][:3] = [0.22, -0.04, 0.015]
    home_props = xml_poses(pipeline_cfg)
    session = TwinSession(pipeline_cfg, scene=moved)
    try:
        start = session.scene.latest(timeout=5.0)
        assert start is not None
        assert start["bodies"]["sports_ball"]["xyz"][:2] == pytest.approx([0.22, -0.04], abs=0.01)
        assert session.arm.go_to({
            "shoulder_pan_joint": 0.52,
            "shoulder_lift_joint": -0.70,
            "elbow_flex_joint": 0.70,
        }) == "arrived"
        assert session.reset_scene()
        joints = session.arm.get_joints()
        for name, value in session.arm.named_state("home").items():
            assert joints[name] == pytest.approx(value, abs=0.05), (name, joints[name], value)
        open_jaws = session.arm.named_state("open")
        jaw = session.arm.gripper_joint
        assert joints[jaw] == pytest.approx(open_jaws[jaw], abs=0.05), joints[jaw]
        scene = session.scene.latest(settled=True, timeout=5.0)
        assert scene is not None, "the scene did not settle after reset"
        for name, qpos in home_props.items():
            got = scene["bodies"][name]
            assert got["xyz"] == pytest.approx(list(qpos[:3]), abs=0.01), name
            assert quat_close(got["quat"], qpos[3:7]), name
            assert not got["grasped"], name
        assert session.arm.go_to({"shoulder_pan_joint": 0.35}) == "arrived"
        assert session.arm.get_joints()["shoulder_pan_joint"] == pytest.approx(0.35, abs=0.05)
    finally:
        session.close()


def _line(proc: subprocess.Popen, prefix: str) -> str:
    """The first stdout line starting with ``prefix``. The process exiting first is a failure."""
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith(prefix):
            return line.rstrip()
    pytest.fail(f"process exited before {prefix!r}")


_DISCOVERY_SUB = """
import os, sys
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"
from vvla_workshop.twin_ros import use_private_domain
use_private_domain(int(sys.argv[1]))
import rclpy
from std_msgs.msg import String
node = rclpy.create_node("twin_discovery_sub")
def on_probe(msg):
    print("SEEN", flush=True)
node.create_subscription(String, "/twin_probe", on_probe, 10)
print("READY", flush=True)
rclpy.spin(node)
"""

_DISCOVERY_PUB = """
import os, sys
from pathlib import Path
os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
os.environ["FASTDDS_BUILTIN_TRANSPORTS"] = "LARGE_DATA"
from vvla_workshop.twin_ros import use_private_domain
use_private_domain(int(sys.argv[1]))
assert os.environ.get("ROS_AUTOMATIC_DISCOVERY_RANGE") != "LOCALHOST"
assert os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"].endswith("twin_localhost.xml")
assert os.environ.get("FASTDDS_BUILTIN_TRANSPORTS") == "LARGE_DATA"
import rclpy
from std_msgs.msg import String
node = rclpy.create_node("twin_discovery_pub")
inodes = set()
for entry in Path("/proc/self/fd").iterdir():
    try:
        target = os.readlink(entry)
    except OSError:
        continue
    if target.startswith("socket:[") and target.endswith("]"):
        inodes.add(target[len("socket:["):-1])
listening = 0
for name in ("tcp", "tcp6"):
    table = Path(f"/proc/self/net/{name}")
    if not table.exists():
        continue
    for row in table.read_text().splitlines()[1:]:
        parts = row.split()
        if len(parts) > 9 and parts[3] == "0A" and parts[9] in inodes:
            listening += 1
print(f"TCP {listening}", flush=True)
pub = node.create_publisher(String, "/twin_probe", 10)
def tick():
    msg = String()
    msg.data = "hi"
    pub.publish(msg)
node.create_timer(0.05, tick)
rclpy.spin(node)
"""


def test_localhost_discovery_keeps_large_data_transports():
    """Two processes on the private domain see each other, and LARGE_DATA stays on.

    The publisher is started with ``ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST``.
    That range would disable builtin transports, so the private-domain setup
    clears it. A TCP listen socket is what ``FASTDDS_BUILTIN_TRANSPORTS=LARGE_DATA``
    opens and that range does not.
    """
    domain = "94"
    sub = subprocess.Popen(
        [sys.executable, "-c", _DISCOVERY_SUB, domain],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy(),
    )
    pub = None
    try:
        assert _line(sub, "READY") == "READY"
        pub = subprocess.Popen(
            [sys.executable, "-c", _DISCOVERY_PUB, domain],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy(),
        )
        tcp = _line(pub, "TCP")
        assert int(tcp.split()[1]) >= 1, tcp
        assert _line(sub, "SEEN") == "SEEN"
    finally:
        for proc in (pub, sub):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait()
