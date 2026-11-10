# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""A YOLO box from the wrist camera, located two ways, against the prop.

The pick grounds a spoken name in the arm camera. This runs that detector on
the twin's wrist image at the scan poses. The locate is the box centre's ray
at the class centre height. The printed ray is that same pixel on the table.
The truth is the prop on ``/free_joint_states``. A sphere has to land within
10 mm, a bottle within 5 mm, and a flat object within 25 mm.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import conftest
from conftest import next_message
from vla_pipeline.robot.arm_interface import radians_from_calibrated

pytestmark = pytest.mark.sim
launch = pytest.mark.launch(fixture=conftest.twin_bringup)
# Spoken class, the prop body, and the shape the hand-built locator fits.
OBJECTS = (
    ("sports ball", "sports_ball", "sphere"),
    ("bottle", "bottle", "bottle"),
    ("book", "book", "flat"),
    ("toothbrush", "toothbrush", "flat"),
)
# Sphere 10 mm, bottle 5 mm, flat 25 mm.
TOL_MM = {"sphere": 10.0, "bottle": 5.0, "flat": 25.0}


def _truth(ros_node) -> dict:
    from mujoco_ros2_control_msgs.msg import FreeJointStateArray

    props = next_message(ros_node, FreeJointStateArray, "/free_joint_states")
    return {
        p.name: np.array([p.pose.pose.position.x, p.pose.pose.position.y, p.pose.pose.position.z])
        for p in props.free_joints
    }


def _still(ros_node, body: str) -> dict:
    """Prop poses once ``body``'s linear speed is under 2 cm/s."""
    from mujoco_ros2_control_msgs.msg import FreeJointStateArray

    def stopped(msg) -> bool:
        for item in msg.free_joints:
            if item.name != body:
                continue
            vel = item.twist.twist.linear
            return vel.x * vel.x + vel.y * vel.y + vel.z * vel.z < 0.02 * 0.02
        return False

    props = next_message(ros_node, FreeJointStateArray, "/free_joint_states", stopped)
    pos, quat = {}, {}
    for item in props.free_joints:
        pos[item.name] = np.array([
            item.pose.pose.position.x, item.pose.pose.position.y, item.pose.pose.position.z,
        ])
        ori = item.pose.pose.orientation
        quat[item.name] = np.array([ori.x, ori.y, ori.z, ori.w])
    return pos, quat


def _bottom(xyz, quat, half) -> float:
    """Lowest corner of a box of half-sizes ``half`` at ``xyz`` with ``quat``."""
    rotation = Rotation.from_quat(quat).as_matrix()
    low = float("inf")
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                corner = xyz + rotation @ (np.array([sx, sy, sz]) * half)
                low = min(low, float(corner[2]))
    return low


@launch
def test_a_wrist_camera_detection_locates_the_prop(arm, ros_node, pipeline_cfg):
    from sensor_msgs.msg import CameraInfo, Image

    from vla_pipeline.behaviors.manipulation import _optical_in_base
    from vla_pipeline.vision.camera import from_image_msg
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector, locate_detection, table_point

    detector = ObjectDetector(pipeline_cfg)
    table_z = float(pipeline_cfg["pick_place"]["table_z"])
    found = {}
    lines = [f"detector {detector.provider} cutoff {detector.conf_threshold}"]
    for name in arm.scan_states():
        if len(found) == len(OBJECTS):
            break
        assert arm.go_to_named(name) == "arrived", name
        image = next_message(ros_node, Image, "/cameras/arm/image_raw")
        info = next_message(ros_node, CameraInfo, "/cameras/arm/camera_info")
        frame = from_image_msg(image)
        assert frame is not None, image.encoding
        pose = _optical_in_base(ros_node)
        truth = _truth(ros_node)
        for spoken, body, shape in OBJECTS:
            if spoken in found:
                continue
            dets, resolved = detector.find(frame, spoken)
            if not dets or resolved is None:
                continue
            box = dets[0].box
            centre = (0.5 * (float(box[0]) + float(box[2])), 0.5 * (float(box[1]) + float(box[3])))
            hit = table_point(centre, info, pose, table_z)
            located = locate_detection(box, info, pose, table_z, resolved)
            where = truth[body]
            ray_mm = None if hit is None else 1000 * float(np.linalg.norm(hit[:2] - where[:2]))
            fit_mm = None if located is None else 1000 * float(np.linalg.norm(located["xyz"][:2] - where[:2]))
            found[spoken] = (name, float(dets[0].score), ray_mm, fit_mm)
            lines.append(
                f"{spoken:12} {name:12} score {dets[0].score:.2f}  "
                f"ray {ray_mm if ray_mm is None else round(ray_mm, 1)} mm  "
                f"locate {fit_mm if fit_mm is None else round(fit_mm, 1)} mm  "
                f"truth ({where[0]:.3f}, {where[1]:.3f})"
            )
    missed = [spoken for spoken, _, _ in OBJECTS if spoken not in found]
    lines.append("not seen: " + (", ".join(missed) if missed else "none"))
    outside = [
        f"{spoken} fit {None if spoken not in found else found[spoken][3]} mm, bar {TOL_MM[shape]} mm"
        for spoken, _, shape in OBJECTS
        if spoken not in found or found[spoken][3] is None or found[spoken][3] >= TOL_MM[shape]
    ]
    report = "\n".join(lines)
    print(report)
    assert not outside, report


# Spoken name, class, prop body, shape, and the scan that sees it.
LIFTS = (
    ("ball", "sports ball", "sports_ball", "sphere", "scan_1"),
    ("bottle", "bottle", "bottle", "bottle", "scan_3"),
    ("book", "book", "book", "flat", "scan_5"),
    ("toothbrush", "toothbrush", "toothbrush", "flat", "scan_7"),
)


@launch
def test_the_arm_lifts_each_object_it_sees(arm, ros_node, pipeline_cfg):
    """``run_pick_place`` then ``run_place`` for each default object.

    The book is the known miss: the fingers cannot clear the table and still
    meet a book 12 mm thick.
    """
    import threading
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_pick_place, run_place
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    detector = ObjectDetector(pipeline_cfg)
    lines = []
    for spoken, class_name, body, _shape, scan in LIFTS:
        if class_name == "book":
            lines.append(f"{class_name:12} {scan:8} no grasp")
            continue
        arm.reset()
        before = _truth(ros_node)[body]
        said: list[str] = []
        ctx = SimpleNamespace(
            cfg=pipeline_cfg, arm=arm, detector=detector,
            say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
            shared_state={},
        )
        held = run_pick_place(ctx, SimpleNamespace(object_name=spoken))
        if not held:
            lines.append(f"{class_name:12} {scan:8} not picked")
            continue
        if body == "toothbrush":
            _settled, quats = _still(ros_node, body)
            from vla_pipeline.behaviors.manipulation import tip_pose

            tip = tip_pose(arm).orientation
            fingers = Rotation.from_quat([tip.x, tip.y, tip.z, tip.w]).apply([0.0, 1.0, 0.0])
            brush = Rotation.from_quat(quats[body]).apply([0.0, 1.0, 0.0])
            aligned = abs(float(np.dot(fingers, brush)))
            degrees = float(np.degrees(np.arccos(np.clip(aligned, -1.0, 1.0))))
            if degrees > 10.0:
                lines.append(
                    f"{class_name:12} {scan:8} fingers not along the brush ({degrees:.1f} deg)"
                )
                continue
        rise = 1000 * float(_truth(ros_node)[body][2] - before[2])
        if not run_place(ctx):
            lines.append(f"{class_name:12} {scan:8} rose {rise:.1f} mm  not set down")
            continue
        back = 1000 * abs(float(_truth(ros_node)[body][2] - before[2]))
        lines.append(f"{class_name:12} {scan:8} rose {rise:.1f} mm  back {back:.1f} mm")
    report = "\n".join(lines)
    print(report)
    failed = [
        line for line in lines
        if not line.startswith("book")
        and (
            "not " in line
            or float(line.split("rose")[1].split("mm")[0]) <= 30.0
            or float(line.split("back")[1].split("mm")[0]) > 15.0
        )
    ]
    assert not failed, report


# The six orders of the three objects the grasp can hold. The book is the known miss.
ORDERS = (
    ("ball", "bottle", "toothbrush"),
    ("ball", "toothbrush", "bottle"),
    ("bottle", "ball", "toothbrush"),
    ("bottle", "toothbrush", "ball"),
    ("toothbrush", "ball", "bottle"),
    ("toothbrush", "bottle", "ball"),
)


@pytest.mark.parametrize("order", ORDERS, ids=["-".join(order) for order in ORDERS])
@launch
def test_one_pick_and_place_order_without_a_reset(arm, ros_node, pipeline_cfg, order):
    """Pick and place each object, then the next, with no reset between them.

    The scene starts at the home keyframe once. Each object rises more than
    30 mm and returns within 15 mm in z of where it started.
    """
    import threading
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_pick_place, run_place
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    bodies = {"ball": "sports_ball", "bottle": "bottle", "toothbrush": "toothbrush"}
    arm.reset()
    detector = ObjectDetector(pipeline_cfg)
    said: list[str] = []
    ctx = SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, detector=detector,
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
        shared_state={},
    )
    for spoken in order:
        body = bodies[spoken]
        before = _truth(ros_node)[body]
        assert run_pick_place(ctx, SimpleNamespace(object_name=spoken)) is True, said
        rise = float(_still(ros_node, body)[0][body][2] - before[2])
        assert rise > 0.03, f"{spoken} rose {1000 * rise:.1f} mm"
        assert run_place(ctx) is True, said
        back = float(_still(ros_node, body)[0][body][2] - before[2])
        print(f"{spoken} rose {1000 * rise:.1f} mm  back {1000 * back:.1f} mm")
        assert abs(back) < 0.015, f"{spoken} ended {1000 * back:.1f} mm off its start height"


@launch
def test_empty_jaws_reach_the_close(arm, ros_node, pipeline_cfg, bag):
    """Nothing between the jaws: the close reaches the SRDF pose and does not stall.

    The jaw is the ``/joint_states`` sample that is at that pose. The band is
    the gripper action's ``goal_tolerance`` (0.01 rad, the controller default;
    the yaml does not set it).
    """
    from sensor_msgs.msg import JointState

    from vla_pipeline.behaviors.manipulation import present_and_close

    arm.reset()
    stalled = present_and_close(arm, pipeline_cfg)
    closed = arm.named_state("close")[arm.gripper_joint]
    jaws = arm.gripper_joint
    msg = next_message(
        ros_node,
        JointState,
        "/joint_states",
        lambda m: abs(dict(zip(m.name, m.position)).get(jaws, 9.0) - closed) < 0.01,
    )
    jaw = dict(zip(msg.name, msg.position))[jaws]
    print(f"empty stall {stalled} jaw {jaw:.4f} rad  close {closed:.4f} rad")
    assert stalled is False
    assert abs(jaw - closed) < 0.01
    assert bag.count("/joint_states") > 0
    assert bag.count("/free_joint_states") > 0


class _PersonOnTheLeft:
    """Torso keypoints left of centre, the input ``person_pan_target`` already takes."""

    kpt_threshold = 0.3

    def primary_person(self, frame):
        kpts = np.zeros((17, 3), np.float32)
        height, width = frame.shape[:2]
        x = 0.3 * width
        kpts[0] = (x, 0.4 * height, 0.9)
        kpts[5] = (x - 0.05 * width, 0.5 * height, 0.9)
        kpts[6] = (x + 0.05 * width, 0.5 * height, 0.9)
        return kpts, 0.9


@launch
def test_the_handover_aims_the_pan(arm, pipeline_cfg):
    """The fetch extend at the pan a person on the left would set."""
    from vla_pipeline.behaviors.manipulation import extend_handoff, person_pan_target

    pan = person_pan_target(_PersonOnTheLeft(), np.zeros((480, 640, 3), np.uint8))
    assert pan is not None and pan > 0.0
    arm.reset()
    assert extend_handoff(arm, pipeline_cfg, pan)
    aimed = radians_from_calibrated(pipeline_cfg, {"shoulder_pan": pan})["shoulder_pan_joint"]
    reached = arm.get_joints()["shoulder_pan_joint"]
    print(f"handover pan {pan:.1f} deg  joint {reached:.4f} rad  target {aimed:.4f} rad")
    assert abs(reached - aimed) < 0.05


@launch
def test_the_mount_camera_aims_the_handover(arm, ros_node, pipeline_cfg):
    """One mount-camera frame decides the handover pan, and the arm reaches it."""
    from sensor_msgs.msg import Image

    from vla_pipeline.behaviors.manipulation import extend_toward_person
    from vla_pipeline.vision.camera import from_image_msg
    from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

    arm.reset()
    image = next_message(ros_node, Image, pipeline_cfg["cameras"]["mount"]["topic"])
    frame = from_image_msg(image)
    assert frame is not None, image.encoding
    pose = PoseEstimator(pipeline_cfg)
    seen = pose.primary_person(frame)
    pan = extend_toward_person(arm, pipeline_cfg, pose, frame)
    aimed = radians_from_calibrated(pipeline_cfg, {"shoulder_pan": pan})["shoulder_pan_joint"]
    reached = arm.get_joints()["shoulder_pan_joint"]
    print(
        f"mount person {None if seen is None else round(seen[1], 2)}  "
        f"pan {pan:.1f} deg  joint {reached:.4f} rad  target {aimed:.4f} rad"
    )
    assert abs(reached - aimed) < 0.05


@launch
def test_a_recognized_person_aims_the_handover(arm, pipeline_cfg):
    """The pose model on the workshop's stand-in photo aims the fetch extend.

    That photo is the webcam stand-in in ``tests/sim/notebook_harness.py``.
    The twin's mount camera has nobody in frame, so this is the frame that
    does contain a person.
    """
    import cv2
    from importlib.util import find_spec
    from pathlib import Path

    from vla_pipeline.behaviors.manipulation import extend_toward_person
    from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

    spec = find_spec("ultralytics")
    photo = Path(spec.submodule_search_locations[0]) / "assets" / "zidane.jpg"
    frame = cv2.imread(str(photo))
    assert frame is not None, photo
    pose = PoseEstimator(pipeline_cfg)
    arm.reset()
    pan = extend_toward_person(arm, pipeline_cfg, pose, frame)
    aimed = radians_from_calibrated(pipeline_cfg, {"shoulder_pan": pan})["shoulder_pan_joint"]
    reached = arm.get_joints()["shoulder_pan_joint"]
    print(f"photo pan {pan:.1f} deg  joint {reached:.4f} rad  target {aimed:.4f} rad")
    assert abs(pan) > 1.0
    assert abs(reached - aimed) < 0.05


@launch
def test_run_grip_on_empty_jaws_does_not_lock(arm, ros_node, pipeline_cfg):
    """``run_grip`` on this arm: nothing in the jaws, so the close does not lock."""
    import threading
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_grip

    said = []
    state: dict = {}
    arm.reset()
    held = run_grip(SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, shared_state=state, arm_camera=None,
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
    ))
    closed = arm.named_state("close")[arm.gripper_joint]
    jaws = arm.gripper_joint
    msg = next_message(
        ros_node,
        JointState,
        "/joint_states",
        lambda m: abs(dict(zip(m.name, m.position)).get(jaws, 9.0) - closed) < 0.01,
    )
    jaw = dict(zip(msg.name, msg.position))[jaws]
    print(f"grip held {held} jaw {jaw:.4f} rad  close {closed:.4f} rad  said {said}")
    assert held is False
    assert state.get("grip_lock") is not True
    assert abs(jaw - closed) < 0.01
    assert said == ["No object detected."]


@launch
def test_mimic_without_a_pinch_keeps_the_held_ball_and_release_drops_it(arm, ros_node, pipeline_cfg):
    """A wrist with no hand reading does not open the jaws, so a held ball stays up.
    ``run_release`` then opens the jaws and the ball returns to the table."""
    import threading
    from types import SimpleNamespace

    import numpy as np

    from vla_pipeline.behaviors.gesture_mimic import mimic_tick
    from vla_pipeline.behaviors.manipulation import run_pick_place, run_release
    from vla_pipeline.utils.gesture_map import mapper_from_config
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector
    from vla_pipeline.vision.yolo_pose_npu import KP

    class _Pose:
        kpt_threshold = 0.3

        def infer(self, frame):
            kpts = np.zeros((1, 17, 3), np.float32)
            kpts[0, KP["r_wrist"]] = (120.0, 240.0, 0.9)
            kpts[0, KP["r_elbow"]] = (120.0, 180.0, 0.9)
            boxes = np.array([[80.0, 100.0, 200.0, 400.0]], np.float32)
            scores = np.array([0.9], np.float32)
            return boxes, scores, kpts

    said = []
    state: dict = {}
    ctx = SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, detector=ObjectDetector(pipeline_cfg),
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
        shared_state=state, pose=_Pose(), hands=None,
        mapper=mapper_from_config(pipeline_cfg),
    )
    arm.reset()
    before = _truth(ros_node)["sports_ball"]
    assert run_pick_place(ctx, SimpleNamespace(object_name="ball")) is True
    jaw = arm.get_joints()[arm.gripper_joint]
    signals = mimic_tick(ctx, np.zeros((480, 640, 3), np.uint8))
    assert signals.joints is not None and "gripper" not in signals.joints
    held, _quat = _still(ros_node, "sports_ball")
    jaw_after = arm.get_joints()[arm.gripper_joint]
    ball = held["sports_ball"]
    print(f"held jaw {jaw:.4f} after {jaw_after:.4f}  ball z {ball[2]:.3f}")
    assert abs(jaw_after - jaw) < 0.05
    assert ball[2] - before[2] > 0.03

    assert run_release(ctx) is True
    placed, _quat = _still(ros_node, "sports_ball")
    back = placed["sports_ball"] - before
    print(
        f"release said {said[-1]}  "
        f"back z {1000 * back[2]:.1f} mm  xy {1000 * float(np.linalg.norm(back[:2])):.1f} mm"
    )
    assert said[-1] == "Released."
    assert not state.get("grip_lock")
    assert abs(back[2]) < 0.015


@launch
def test_the_ball_is_picked_from_a_near_spot(arm, ros_node, pipeline_cfg):
    """The ball moved closer to the base is still found and held up."""
    import threading
    from types import SimpleNamespace

    from mujoco_ros2_control_msgs.msg import FreeJointState
    from mujoco_ros2_control_msgs.srv import SetFreeJointState

    from vla_pipeline.behaviors.manipulation import run_pick_place
    from vla_pipeline.robot.moveit_arm import _call
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    arm.reset()
    moved = FreeJointState(name="sports_ball")
    moved.pose.pose.position.x = 0.22
    moved.pose.pose.position.y = -0.04
    moved.pose.pose.position.z = 0.015
    moved.pose.pose.orientation.w = 1.0
    reply = _call(
        ros_node, SetFreeJointState, "/mujoco_ros2_control_node/set_free_joint_state",
        SetFreeJointState.Request(free_joints=[moved]), 15.0,
    )
    assert reply.success, reply.message
    before = _truth(ros_node)["sports_ball"]
    assert abs(before[0] - 0.22) < 0.02 and abs(before[1] + 0.04) < 0.02, before
    said = []
    held = run_pick_place(SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, detector=ObjectDetector(pipeline_cfg),
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
        shared_state={},
    ), SimpleNamespace(object_name="ball"))
    up, _quat = _still(ros_node, "sports_ball")
    rise = float(up["sports_ball"][2] - before[2])
    print(f"near pick held {held} rise {1000 * rise:.1f} mm  said {said}")
    assert held is True
    assert rise > 0.03


@pytest.mark.parametrize("label,xy", [("right", (0.25, -0.10))])
@launch
def test_the_ball_is_picked_from_another_spot(arm, ros_node, pipeline_cfg, label, xy):
    """The ball at the old right table spot is still found and held up.

    Every other prop stays within 1 cm. The far spot (0.32, 0) is not here:
    its grasp has no inverse kinematics. See the plan's spot note.
    """
    import threading
    from types import SimpleNamespace

    from mujoco_ros2_control_msgs.msg import FreeJointState
    from mujoco_ros2_control_msgs.srv import SetFreeJointState

    from vla_pipeline.behaviors.manipulation import run_pick_place
    from vla_pipeline.robot.moveit_arm import _call
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    arm.reset()
    moved = FreeJointState(name="sports_ball")
    moved.pose.pose.position.x = float(xy[0])
    moved.pose.pose.position.y = float(xy[1])
    moved.pose.pose.position.z = 0.015
    moved.pose.pose.orientation.w = 1.0
    reply = _call(
        ros_node, SetFreeJointState, "/mujoco_ros2_control_node/set_free_joint_state",
        SetFreeJointState.Request(free_joints=[moved]), 15.0,
    )
    assert reply.success, reply.message
    before = _truth(ros_node)
    ball = before["sports_ball"]
    assert abs(ball[0] - xy[0]) < 0.02 and abs(ball[1] - xy[1]) < 0.02, ball
    said = []
    held = run_pick_place(SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, detector=ObjectDetector(pipeline_cfg),
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
        shared_state={},
    ), SimpleNamespace(object_name="ball"))
    up, _quat = _still(ros_node, "sports_ball")
    rise = float(up["sports_ball"][2] - ball[2])
    shifted = {
        name: 1000 * float(np.linalg.norm(up[name][:2] - before[name][:2]))
        for name in before
        if name != "sports_ball"
    }
    print(f"{label} pick held {held} rise {1000 * rise:.1f} mm  said {said}  others {shifted}")
    assert held is True
    assert rise > 0.03
    assert any("Got the sports ball" in line for line in said), said
    assert all(mm < 10.0 for mm in shifted.values()), shifted


@launch
def test_run_fetch_block_without_a_block_says_so(arm, ros_node, pipeline_cfg):
    """The taught spot is empty, so fetch does not take the ball from the table."""
    import threading
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import run_fetch_block

    said = []
    arm.reset()
    before = _truth(ros_node)
    fetched = run_fetch_block(SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, say=said.append, stop_check=lambda: False,
        stop_event=threading.Event(),
    ))
    after = _truth(ros_node)
    moved = {
        name: 1000 * float(np.linalg.norm(after[name][:2] - start[:2]))
        for name, start in before.items()
        if float(np.linalg.norm(after[name][:2] - start[:2])) >= 0.02
    }
    print(f"fetch {fetched} said {said} moved {moved}")
    assert fetched is False
    assert said == ["I couldn't get a grip on the block - is it in its usual spot?"]
    assert not moved


@launch
def test_a_held_ball_is_handed_over(arm, ros_node, pipeline_cfg):
    """Pick the ball, then the fetch handover extends, opens, and the ball returns to the table.

    The taught pose closes sideways, so a free ball cannot sit between those jaws.
    This is the handover of a ball the pickup is already holding.
    """
    import threading
    from types import SimpleNamespace

    from vla_pipeline.behaviors.manipulation import hand_over_held, run_pick_place
    from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

    said = []
    ctx = SimpleNamespace(
        cfg=pipeline_cfg, arm=arm, detector=ObjectDetector(pipeline_cfg),
        say=said.append, stop_check=lambda: False, stop_event=threading.Event(),
        shared_state={},
    )
    arm.reset()
    before = _truth(ros_node)["sports_ball"]
    assert run_pick_place(ctx, SimpleNamespace(object_name="ball")) is True
    held = _truth(ros_node)["sports_ball"]
    assert float(held[2] - before[2]) > 0.03
    assert hand_over_held(ctx) is True
    placed, _quat = _still(ros_node, "sports_ball")
    back = 1000 * float(abs(placed["sports_ball"][2] - before[2]))
    print(
        f"held {1000 * float(held[2] - before[2]):.1f} mm up  "
        f"after handover within {back:.1f} mm of the start height"
    )
    assert back < 15.0
