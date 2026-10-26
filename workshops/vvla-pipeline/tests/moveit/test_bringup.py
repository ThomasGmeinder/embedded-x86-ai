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
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

import conftest
from conftest import next_message as _next
from so101_sim.mapping import BODY_JOINTS as BODY
from vla_pipeline.robot.servo_profile import PROPS
from vla_pipeline.utils.config import REPO_ROOT

pytestmark = pytest.mark.sim
# The session's one bringup, named through conftest: a fixture imported here would be a second one.
LAUNCH = pytest.mark.launch(fixture=conftest.twin_bringup)
MOVEIT_CONFIG = REPO_ROOT / "so_arm101_moveit_config" / "config"


def _srdf_state(name: str) -> dict:
    root = ET.parse(MOVEIT_CONFIG / "so_arm101.srdf").getroot()
    return {j.get("name"): float(j.get("value")) for j in root.find(f"group_state[@name='{name}']")}


def _twin_maps(cfg):
    from so101_sim.config import load_sim_config
    from so101_sim.mapping import JointMap

    sim = load_sim_config(cfg)
    return sim, JointMap(sim["joint_map"], sim["gripper_map"])


def _radians(jmap, pose: dict) -> dict:
    return {f"{j}_joint": jmap.to_rad(j, pose[j]) for j in BODY}


# ----------------------------------------------------------------------------
# the config agrees with the pipeline
# ----------------------------------------------------------------------------
def test_the_srdf_named_states_are_the_pipelines_poses(pipeline_cfg):
    from vla_pipeline.robot.arm_interface import REST_POSE

    sim, jmap = _twin_maps(pipeline_cfg)
    pp = pipeline_cfg["pick_place"]
    expected = {"home": _radians(jmap, REST_POSE), "ready": _radians(jmap, {**pp["scan_pose"], "shoulder_pan": 0.0})}
    for i, pan in enumerate(pp["scan_pans"], 1):
        expected[f"scan_{i}"] = _radians(jmap, {**pp["scan_pose"], "shoulder_pan": pan})
    for i, pan in enumerate(pp["scan_near_pans"], 1):
        expected[f"scan_near_{i}"] = _radians(jmap, {**pp["scan_near_pose"], "shoulder_pan": pan})
    expected["close"] = {"gripper_joint": jmap.to_rad("gripper", sim["gripper_map"]["closed_cal"])}
    for name, joints in expected.items():
        assert _srdf_state(name) == pytest.approx(joints, abs=1e-5), name


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


def test_the_planning_limits_are_the_servo_envelope():
    import yaml

    table = json.loads((REPO_ROOT / "config" / "servo_profiles.json").read_text())["joints"]
    limits = yaml.safe_load((MOVEIT_CONFIG / "joint_limits.yaml").read_text())["joint_limits"]
    for motor, row in table.items():
        joint = limits[f"{motor}_joint"]
        assert joint["max_velocity"] == pytest.approx(math.radians(row["speed_deg_s"]), abs=1e-5), motor
        assert joint["max_acceleration"] == pytest.approx(math.radians(row["accel_deg_s2"]), abs=1e-5), motor


# ----------------------------------------------------------------------------
# the bringup on the twin
# ----------------------------------------------------------------------------
@pytest.fixture(scope="module")
def moveit(arm):
    """The session arm's MoveItPy, on the twin."""
    return arm.robot


@LAUNCH
def test_the_twin_publishes_its_clock_joints_cameras_and_props(ros_node):
    from mujoco_ros2_control_msgs.msg import FreeJointStateArray
    from rosgraph_msgs.msg import Clock
    from sensor_msgs.msg import CameraInfo, Image, JointState

    first = _next(ros_node, Clock, "/clock").clock
    later = _next(ros_node, Clock, "/clock", lambda m: (m.clock.sec, m.clock.nanosec) > (first.sec, first.nanosec))
    assert (later.clock.sec, later.clock.nanosec) > (first.sec, first.nanosec)
    joints = _next(ros_node, JointState, "/joint_states")
    assert set(joints.name) == {f"{j}_joint" for j in (*BODY, "gripper")}
    for role in ("arm", "mount"):
        image = _next(ros_node, Image, f"/cameras/{role}/image_raw")
        info = _next(ros_node, CameraInfo, f"/cameras/{role}/camera_info")
        assert (image.width, image.height) == (640, 480) == (info.width, info.height), role
    props = _next(ros_node, FreeJointStateArray, "/free_joint_states")
    assert {p.name for p in props.free_joints} == set(PROPS)


@LAUNCH
def test_moveitpy_plans_and_executes_from_home_to_ready(ros_node, moveit):
    from vla_pipeline.robot import moveit_arm

    moveit_arm.reset_twin(ros_node)
    arm = moveit.get_planning_component(moveit_arm.PLANNING_GROUP)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="ready")
    plan = arm.plan()
    assert plan, "no plan from home to ready"
    status = moveit.execute(plan.trajectory, controllers=[])
    assert status.status == "SUCCEEDED", status.status
    ready = _srdf_state("ready")
    reached = moveit_arm.current_joints(moveit)
    for joint, value in ready.items():
        assert reached[joint] == pytest.approx(value, abs=0.01), joint


@LAUNCH
def test_reset_world_puts_the_arm_and_every_prop_back(ros_node, moveit):
    from control_msgs.msg import JointTrajectoryControllerState
    from mujoco_ros2_control_msgs.msg import FreeJointState, FreeJointStateArray
    from mujoco_ros2_control_msgs.srv import SetFreeJointState

    from vla_pipeline.robot import moveit_arm

    arm = moveit.get_planning_component(moveit_arm.PLANNING_GROUP)
    arm.set_start_state_to_current_state()
    arm.set_goal_state(configuration_name="scan_6")
    plan = arm.plan()
    assert plan and moveit.execute(plan.trajectory, controllers=[]).status == "SUCCEEDED"
    moved = FreeJointState(name="sports_ball")
    moved.pose.pose.position.x, moved.pose.pose.position.y, moved.pose.pose.position.z = 0.25, 0.12, 0.05
    reply = moveit_arm._call(ros_node, SetFreeJointState, "/mujoco_ros2_control_node/set_free_joint_state",
                             SetFreeJointState.Request(free_joints=[moved]), 15.0)
    assert reply.success, reply.message

    moveit_arm.reset_twin(ros_node)

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


def _hand_built_grasp(kin, pp: dict, prop: str):
    """The hand-built IK's top-down grasp of ``prop`` where the scene file has it,
    from the inputs the pipeline's pickup gives it, and the jaw width: ``(grasp, width)``."""
    from vla_pipeline.robot.arm_interface import GRASP_LIMITS

    m = kin.world.model
    body = m.body(prop)
    size = m.geom_size[body.geomadr[0]]
    table_z = float(pp["table_z"])
    p = np.array(body.pos, float)
    p[2] = min(p[2], table_z + float(pp["max_grasp_height_m"]))
    kw = {"above_m": float(pp["pregrasp_above_m"]), "table_z": table_z, "wrist_roll": 0.0}
    if prop == "bottle":
        kw.update(width=2 * size[0], obstacle=(*body.pos[:2], size[0], table_z + 2 * size[1]), limits=GRASP_LIMITS)
    elif prop in pp["wrist_align_classes"]:
        yaw = 2.0 * math.atan2(body.quat[3], body.quat[0])
        kw.update(width=float(pp["wrist_align_grasp_width_m"]), limits=GRASP_LIMITS,
                  object_axis=np.array([-math.sin(yaw), math.cos(yaw), 0.0]),
                  above_m=float(pp["wrist_align_pregrasp_above_m"]))
        p[2] = max(p[2], table_z + 0.011)
    else:
        kw.update(width=2 * float(min(size[0], size[1])))
    found = kin.solve_grasp(p, **kw)
    return (None if found is None else found[1]), kw["width"]


@LAUNCH
def test_pick_ik_solves_the_top_down_grasps_on_the_default_objects(moveit, pipeline_cfg):
    """pick_ik from home on the grasps the hand-built IK finds for the default
    objects, and on each turned about the vertical through its grasp point.
    The solve rate is measured and written to ``_artifacts/moveit/pick_ik_solve_rate.json``;
    an object the hand-built IK has no grasp for is recorded as such."""
    from geometry_msgs.msg import Pose
    from moveit.core.robot_state import RobotState
    from scipy.spatial.transform import Rotation

    from so101_sim.kinematics import shared_kinematics
    from vla_pipeline.robot import moveit_arm
    from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR

    model = moveit.get_robot_model()
    _sim, jmap = _twin_maps(pipeline_cfg)
    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    world = kin.world
    home = {**_srdf_state("home"), **_srdf_state("open")}
    yaws = range(0, 360, 30)
    rates = {}
    for prop in PROPS:
        grasp, width = _hand_built_grasp(kin, pipeline_cfg["pick_place"], prop)
        if grasp is None:
            rates[prop] = {"hand_built_grasp": None}
            continue
        state = RobotState(model)
        state.joint_positions = {**home, **_radians(jmap, grasp)}
        state.update()
        tip = state.get_pose(moveit_arm.TIP_LINK)
        centre = world.grasp_point(grasp, width)
        tip_xyz = np.array([tip.position.x, tip.position.y, tip.position.z])
        tip_rot = Rotation.from_quat([tip.orientation.x, tip.orientation.y, tip.orientation.z, tip.orientation.w])
        solved = []
        for yaw in yaws:
            turn = Rotation.from_euler("z", yaw, degrees=True)
            xyz = centre + turn.apply(tip_xyz - centre)
            q = (turn * tip_rot).as_quat()
            target = Pose()
            target.position.x, target.position.y, target.position.z = (float(v) for v in xyz)
            target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = (float(v) for v in q)
            seed = RobotState(model)
            seed.joint_positions = home
            seed.update()
            solved.append(bool(seed.set_from_ik(moveit_arm.PLANNING_GROUP, target, moveit_arm.TIP_LINK, 0.1)))
        rates[prop] = {"solved": sum(solved), "of": len(solved), "yaws_deg": [y for y, ok in zip(yaws, solved) if ok]}
    out = REPO_ROOT / "_artifacts" / "moveit" / "pick_ik_solve_rate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rates, indent=2) + "\n")
    print("pick_ik solve rate:", json.dumps(rates))
    compared = {prop: rate for prop, rate in rates.items() if "yaws_deg" in rate}
    assert compared, "the hand-built IK grasps none of the default objects"
    for prop, rate in compared.items():
        assert 0 in rate["yaws_deg"], f"pick_ik does not reach the hand-built IK's own grasp of the {prop}"


@LAUNCH
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
