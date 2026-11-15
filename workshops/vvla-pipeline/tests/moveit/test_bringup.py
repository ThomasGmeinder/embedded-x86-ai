# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The MoveIt 2 bringup on the MuJoCo twin: ``launch/so101_control.launch.py
hardware_type:=mujoco``, ros2_so_arm's controllers, move_group and MoveItPy.

The static tests hold the MoveIt config to the pipeline's values while both
exist; the launch tests run the bringup once for the module. A launch service
replaces Python's logger class, so these run in their own pytest process
(``run_sim_tests.sh``), away from the twin suite's ``caplog`` assertions.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import conftest
from conftest import next_message as _next
from so101_sim import PROP_BODIES as PROPS
from so101_sim import MOTOR_NAMES
from vla_pipeline.utils.config import REPO_ROOT

pytestmark = pytest.mark.sim
# The session's one bringup, named through conftest: a fixture imported here would be a second one.
launch = pytest.mark.launch(fixture=conftest.twin_bringup)
MOVEIT_CONFIG = REPO_ROOT / "so_arm101_moveit_config" / "config"


def _srdf_state(name: str) -> dict:
    root = ET.parse(MOVEIT_CONFIG / "so_arm101.srdf").getroot()
    return {j.get("name"): float(j.get("value")) for j in root.find(f"group_state[@name='{name}']")}


# ----------------------------------------------------------------------------
# the config agrees with the pipeline
# ----------------------------------------------------------------------------
def test_the_srdf_named_states_are_the_pipelines_poses(pipeline_cfg):
    from vla_pipeline.robot.arm_interface import REST_POSE, radians_from_calibrated

    home = radians_from_calibrated(pipeline_cfg, REST_POSE)
    srdf = _srdf_state("home")
    # home is the manipulator group. The jaw is the gripper group's open and close states.
    assert srdf == pytest.approx({name: home[name] for name in srdf}, abs=1e-5)
    root = ET.parse(MOVEIT_CONFIG / "so_arm101.srdf").getroot()
    names = {state.get("name") for state in root.findall("group_state")}
    for i in range(1, 9):
        assert f"scan_{i}" in names
    for i in range(1, 6):
        assert f"scan_near_{i}" in names


def test_the_reset_keyframe_is_home_with_the_jaws_open_and_every_prop_in_place():
    import mujoco

    from so101_sim.config import DEFAULT_SCENE

    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    key = m.key("home")
    arm = {**_srdf_state("home"), **_srdf_state("open")}
    for joint, value in arm.items():
        adr = m.joint(joint).qposadr[0]
        assert key.qpos[adr] == pytest.approx(value, abs=1e-5), joint
        assert key.ctrl[m.actuator(joint).id] == pytest.approx(value, abs=1e-5), joint
    for prop in PROPS:
        adr = m.joint(f"{prop}_j").qposadr[0]
        assert key.qpos[adr:adr + 7] == pytest.approx(m.qpos0[adr:adr + 7], abs=1e-5), prop


def test_the_global_speed_is_the_joint_limit_scale():
    """Every plan uses joint_limits.yaml's default scales unless that plan overrides them."""
    import yaml

    from vla_pipeline.robot.moveit_arm import planning_scales

    doc = yaml.safe_load((MOVEIT_CONFIG / "joint_limits.yaml").read_text())
    assert planning_scales(limits=doc) == (
        doc["default_velocity_scaling_factor"],
        doc["default_acceleration_scaling_factor"],
    )
    assert planning_scales(0.25, limits=doc) == (0.25, doc["default_acceleration_scaling_factor"])
    request = yaml.safe_load((MOVEIT_CONFIG / "moveit_cpp.yaml").read_text())["plan_request_params"]
    assert "max_velocity_scaling_factor" not in request
    assert "max_acceleration_scaling_factor" not in request


# ----------------------------------------------------------------------------
# the bringup on the twin
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def moveit(arm):
    """The session arm's MoveItPy, on the twin."""
    return arm.robot


@launch
def test_the_twin_publishes_its_clock_joints_cameras_and_props(ros_node):
    from mujoco_ros2_control_msgs.msg import FreeJointStateArray
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState

    first = _next(ros_node, Clock, "/clock").clock
    later = _next(ros_node, Clock, "/clock", lambda m: (m.clock.sec, m.clock.nanosec) > (first.sec, first.nanosec))
    assert (later.clock.sec, later.clock.nanosec) > (first.sec, first.nanosec)
    joints = _next(ros_node, JointState, "/joint_states")
    assert set(joints.name) == {f"{j}_joint" for j in (*MOTOR_NAMES[:5], "gripper")}
    # The mount camera is the real webcam, so the twin does not publish it.
    for role in ("arm", "scene"):
        image = _next(ros_node, Image, f"/cameras/{role}/image_raw")
        info = _next(ros_node, CameraInfo, f"/cameras/{role}/camera_info")
        assert (image.width, image.height) == (640, 480) == (info.width, info.height), role
    props = _next(ros_node, FreeJointStateArray, "/free_joint_states")
    assert {p.name for p in props.free_joints} == set(PROPS)


@launch
def test_moveitpy_plans_and_executes_from_home_to_ready(arm):
    from vla_pipeline.robot import moveit_arm

    arm.reset()
    planner = arm.robot.get_planning_component(moveit_arm.PLANNING_GROUP)
    planner.set_start_state_to_current_state()
    planner.set_goal_state(configuration_name="ready")
    plan = planner.plan()
    assert plan, "no plan from home to ready"
    status = arm.robot.execute(plan.trajectory, controllers=[])
    assert status.status == "SUCCEEDED", status.status
    ready = _srdf_state("ready")
    reached = moveit_arm.current_joints(arm.robot)
    for joint, value in ready.items():
        assert reached[joint] == pytest.approx(value, abs=0.01), joint


@launch
def test_reset_world_puts_the_arm_and_every_prop_back(ros_node, arm):
    from control_msgs.msg import JointTrajectoryControllerState
    from mujoco_ros2_control_msgs.msg import FreeJointState, FreeJointStateArray
    from mujoco_ros2_control_msgs.srv import SetFreeJointState

    from vla_pipeline.robot import moveit_arm

    planner = arm.robot.get_planning_component(moveit_arm.PLANNING_GROUP)
    planner.set_start_state_to_current_state()
    planner.set_goal_state(configuration_name="scan_6")
    plan = planner.plan()
    assert plan and arm.robot.execute(plan.trajectory, controllers=[]).status == "SUCCEEDED"
    moved = FreeJointState(name="sports_ball")
    moved.pose.pose.position.x, moved.pose.pose.position.y, moved.pose.pose.position.z = 0.25, 0.12, 0.05
    reply = moveit_arm._call(ros_node, SetFreeJointState, "/mujoco_ros2_control_node/set_free_joint_state",
                             SetFreeJointState.Request(free_joints=[moved]), 15.0)
    assert reply.success, reply.message

    arm.reset()

    home = _srdf_state("home")
    held = _next(ros_node, JointTrajectoryControllerState, "/joint_trajectory_controller/controller_state")
    assert dict(zip(held.joint_names, held.reference.positions)) == pytest.approx(home, abs=1e-3)
    import mujoco

    from so101_sim.config import DEFAULT_SCENE

    m = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    props = _next(ros_node, FreeJointStateArray, "/free_joint_states")
    for p in props.free_joints:
        adr = m.joint(f"{p.name}_j").qposadr[0]
        xyz = (p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z)
        assert np.linalg.norm(np.subtract(xyz, m.qpos0[adr:adr + 3])) < 0.002, p.name


def _jaw_down_tip(centre, size, yaw_deg: float):
    """``TIP_LINK`` in ``base_link`` when the class centre sits in :func:`grasp_frame`."""
    from geometry_msgs.msg import Pose
    from scipy.spatial.transform import Rotation

    from vla_pipeline.robot.moveit_arm import grasp_frame

    frame = grasp_frame(size)
    held = np.array([frame.position.x, frame.position.y, frame.position.z])
    jaw = Rotation.from_quat([frame.orientation.x, frame.orientation.y, frame.orientation.z, frame.orientation.w])
    turn = Rotation.from_euler("z", yaw_deg, degrees=True)
    # The grasp frame is the object centre in the tip. Invert it, then yaw about vertical.
    tip_in_object = jaw.inv().apply(-held)
    xyz = np.asarray(centre, float) + turn.apply(tip_in_object)
    quat = (turn * jaw.inv()).as_quat()
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = (float(v) for v in xyz)
    pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (float(v) for v in quat)
    return pose


@launch
def test_pick_ik_solves_the_top_down_grasps_on_the_default_objects(moveit, pipeline_cfg):
    """pick_ik from home on the jaw-down grasp at each prop in the scene file,
    and on that grasp turned about the vertical. The solve rate is written to
    ``_artifacts/moveit/pick_ik_solve_rate.json``. The book is the known miss."""
    import mujoco
    from moveit.core.robot_state import RobotState

    from so101_sim.assets import require
    from so101_sim.config import DEFAULT_SCENE
    from vla_pipeline.robot import moveit_arm
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE

    require()
    scene = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    robot = moveit.get_robot_model()
    home = {**_srdf_state("home"), **_srdf_state("open")}
    yaws = range(0, 360, 30)
    spoken = {"sports_ball": "sports ball"}
    rates = {}
    for prop in PROPS:
        centre = np.array(scene.body(prop).pos, float)
        size = CLASS_SIZE[spoken.get(prop, prop)]
        solved = []
        for yaw in yaws:
            seed = RobotState(robot)
            seed.joint_positions = home
            seed.update()
            target = _jaw_down_tip(centre, size, yaw)
            solved.append(bool(seed.set_from_ik(moveit_arm.PLANNING_GROUP, target, moveit_arm.TIP_LINK, 0.1)))
        rates[prop] = {"solved": sum(solved), "of": len(solved), "yaws_deg": [y for y, ok in zip(yaws, solved) if ok]}
    out = REPO_ROOT / "_artifacts" / "moveit" / "pick_ik_solve_rate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rates, indent=2) + "\n")
    print("pick_ik solve rate:", json.dumps(rates))
    for prop in ("sports_ball", "bottle", "toothbrush"):
        assert 0 in rates[prop]["yaws_deg"], f"pick_ik does not reach the jaw-down grasp of the {prop}"


@launch
def test_a_kernel_from_its_kernel_json_creates_moveitpy_on_the_twin(ros_node):
    """The notebooks' kernel, started the way VS Code starts it, reads the twin's arm state through MoveItPy."""
    from jupyter_client.manager import start_new_kernel
    from sensor_msgs.msg import JointState

    from notebook_harness import KERNEL, kernel_env

    code = (
        "import json\n"
        "from vla_pipeline.robot import moveit_arm\n"
        "robot = moveit_arm.moveit_py('kernel_probe', sim=True)\n"
        "print(json.dumps(moveit_arm.current_joints(robot)))\n"
        "robot.shutdown()\n"
    )
    km, client = start_new_kernel(kernel_name=KERNEL, env=kernel_env(), startup_timeout=120)
    lines = []
    try:
        reply = client.execute_interactive(
            code, timeout=120,
            output_hook=lambda msg: msg["msg_type"] == "stream" and lines.append(msg["content"]["text"]),
        )
        assert reply["content"]["status"] == "ok", reply["content"]
    finally:
        client.stop_channels()
        km.shutdown_kernel()
    seen = json.loads("".join(lines).strip().splitlines()[-1])
    live = _next(ros_node, JointState, "/joint_states")
    assert seen == pytest.approx(dict(zip(live.name, live.position)), abs=0.01)
