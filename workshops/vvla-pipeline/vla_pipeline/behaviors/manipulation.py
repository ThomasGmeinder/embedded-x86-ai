# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Manipulation behaviors: grip, pick_place, place, fetch_block.

The handlers take ``(ctx, cmd)`` (a :class:`~vla_pipeline.user_robot_control.PipelineContext`);
:func:`register_manipulation` adds them to the registry:

- **grip** - present the open jaws and close them on the gripper action. A stall
  is contact and locks the grip; an empty close reaches the goal and does not.
  "release",
  "open", or "drop it" opens the jaws; gesture_mimic hands them to the pinch.
- **pick_place** - ground the spoken object name to a COCO class, locate it
  on the table, and grasp it with a MoveIt Task Constructor pick. The jaws
  stay closed until "release", "open", or "drop it".
- **place** - "place it" or "set it down" lowers the held object and opens
  the jaws.
- **fetch_block** - fetch the block from its taught pose, then hand it over.

The handover finds you with YOLO-pose on the mount camera, aims the pan
joint at you, extends, opens the jaws and parks. Settings: ``grip``,
``pick_place`` and ``fetch_block`` in ``config/pipeline.yaml``.
"""

from __future__ import annotations

import logging
import time
import numpy as np

from vla_pipeline.llm.intents import Intent
from vla_pipeline.robot.arm_interface import radians_from_calibrated
from vla_pipeline.vision.yolo_pose_npu import KP

logger = logging.getLogger(__name__)

# Pose presenting the gripper toward the user, jaws open.
PRESENT_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -35.0,
    "elbow_flex": 30.0,
    "wrist_flex": 10.0,
    "wrist_roll": 0.0,
    "gripper": 80.0,
}

def run_grip(ctx, cmd=None, wait_s: float = 4.0) -> bool:
    """Take an object from the user and lock the grip; "grip" again releases."""
    state = ctx.shared_state
    if ctx.stop_check():
        return False
    if state.get("grip_lock"):
        ctx.say("I'm already holding something. Say release to let go.")
        return False
    # The gripper action reports the stall. An empty close reaches the goal.
    stalled = present_and_close(ctx.arm, ctx.cfg)
    if not stalled:
        ctx.say("No object detected.")
        return False
    state["grip_lock"] = True
    state["grip_lock_pos"] = ctx.arm.named_state("close")[ctx.arm.gripper_joint]
    ctx.say("Object gripped. Say 'release' to let go.")
    return True

# ----------------------------------------------------------------------------
# Handover: find the person, extend, release, park
# ----------------------------------------------------------------------------


def pan_from_kpts(kpts, threshold: float, width: int) -> float | None:
    """The pan angle facing a person's torso, from keypoints the perception process measured."""
    if kpts is None or len(kpts) == 0 or width <= 0:
        return None
    arr = np.asarray(kpts, dtype=float)
    xs = [arr[i, 0] for i in (KP["l_shoulder"], KP["r_shoulder"], KP["nose"]) if arr[i, 2] > threshold]
    if not xs:
        return None
    cx = (float(np.mean(xs)) / width) * 2.0 - 1.0  # [-1, 1]
    return float(np.clip(-cx * 60.0, -80.0, 80.0))  # mirrored: person on the left -> pan left


def person_pan_target(pose, frame: np.ndarray) -> float | None:
    """The pan angle facing the nearest person's torso, or None."""
    person = pose.primary_person(frame)
    if person is None:
        return None
    kpts, _ = person
    return pan_from_kpts(kpts, pose.kpt_threshold, frame.shape[1])


def _pan_from_observations(ctx, seek_s: float) -> tuple[float, bool]:
    """Aim the pan joint from mount observations newer than the last one, until ``seek_s``."""
    threshold = float((ctx.cfg.get("yolo_pose") or {}).get("kpt_threshold", 0.3))
    pan, found = 0.0, False
    last_seq = 0
    deadline = time.monotonic() + seek_s
    while not ctx.stop_check():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        observation = ctx.mailbox.wait_new("mount", last_seq, remaining, ctx.stop_event)
        if observation is None:
            break
        last_seq = observation.seq
        target = pan_from_kpts(observation.kpts, threshold, observation.width)
        if target is None:
            continue
        pan, found = target, True
        ctx.arm.send_joints(radians_from_calibrated(ctx.cfg, {"shoulder_pan": pan}))
    logger.info("[handoff] person %s (pan=%.1f)", "found" if found else "NOT found", pan)
    return pan, found


def find_person_pan(ctx, seek_s: float = 4.0) -> tuple[float, bool]:
    """Track the person for ``seek_s``, one new frame at a time, and aim the
    pan joint at them. A camera that delivers no fresh frame ends the search."""
    pan, found = 0.0, False
    if getattr(ctx, "mailbox", None) is not None:
        return _pan_from_observations(ctx, seek_s)
    if ctx.pose is None or ctx.mount_camera is None:
        logger.info("[handoff] no mount camera - handing over straight ahead")
        return pan, found
    deadline = time.monotonic() + seek_s
    while time.monotonic() < deadline and not ctx.stop_check():
        ok, frame = _read_settled(ctx.mount_camera, time.time())
        if not ok or frame is None:
            logger.info("[handoff] the mount camera delivered no fresh frame")
            break
        target = person_pan_target(ctx.pose, frame)
        if target is not None:
            pan, found = target, True
            ctx.arm.send_joints(radians_from_calibrated(ctx.cfg, {"shoulder_pan": pan}))
    logger.info("[handoff] person %s (pan=%.1f)", "found" if found else "NOT found", pan)
    return pan, found


# ----------------------------------------------------------------------------
# fetch_block and pick_place
# ----------------------------------------------------------------------------




def _fetch_urdf(ctx) -> bool:
    """The taught fetch on an arm whose joints are URDF names.

    The close is the gripper action. No stall means the taught spot is empty.
    """
    from vla_pipeline.behaviors.simple import _go_to, _rest

    fb = ctx.cfg["fetch_block"]
    # Joint interpolation returns immediately when the arm is already at the taught pose,
    # so a ball placed in the open jaws is not swept away by a second approach.
    # The jaw stays where it is. block_pose's gripper value is the taught opening;
    # travelling there before the close knocks a ball out of the open jaws.
    target = radians_from_calibrated(ctx.cfg, fb["block_pose"])
    target.pop(ctx.arm.gripper_joint, None)
    if ctx.arm.go_to_mtc(target) != "arrived" or ctx.stop_check():
        return False
    # Close at once. A ball sitting in the open jaws falls if the jaws travel
    # to the taught opening again before this close.
    stalled = ctx.arm.grip(ctx.arm.named_state("close")[ctx.arm.gripper_joint])
    if not stalled:
        ctx.say("I couldn't get a grip on the block - is it in its usual spot?")
        ctx.arm.go_to_named("open")
        _rest(ctx)
        return False
    if _go_to(ctx, fb["lift_pose"]) != "arrived" or not hand_over_held(ctx):
        ctx.say("I dropped the block.")
        _rest(ctx)
        return False
    return True


def hand_over_held(ctx) -> bool:
    """Extend toward the person, open, and park. The object is already in the jaws."""
    from vla_pipeline.behaviors.simple import _rest

    if not extend_handoff(ctx.arm, ctx.cfg) or ctx.arm.go_to_named("open") != "arrived":
        return False
    _rest(ctx)
    return True


def run_fetch_block(ctx, cmd=None) -> bool:
    """Grasp the block at its taught pose, lift it, hand it over.

    The pose is open loop - nothing looks for the block - so the grasp is
    checked twice, after closing and after the lift; if the block isn't held,
    the arm says so and parks instead of handing over empty jaws.
    """
    return _fetch_urdf(ctx)


_DESCENT_LATERAL_M = 0.005


def _read_settled(camera, settled_at: float):
    """One frame captured after the arm finished moving at ``settled_at``."""
    read_after = getattr(camera, "read_after", None)
    if callable(read_after):
        return read_after(settled_at)
    period = 1.0 / 30.0
    ok, frame = False, None
    for _ in range(8):
        started = time.perf_counter()
        ok, frame = camera.read()
        if not ok or frame is None:
            return False, None
        if time.perf_counter() - started >= 0.5 * period:
            return True, frame
    return bool(ok), frame


def _ros_message(node, msg_type, topic: str):
    """The next message on ``topic``, or None when none arrives within ``REPLY_S``.

    The wrist camera publishes best-effort. A reliable subscriber never
    matches that publisher, so the wait would end with an empty queue.
    """
    import queue

    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

    from vla_pipeline.robot.moveit_arm import REPLY_S, clear_rcutils_error

    # The planner leaves its typesupport probe set. Creating this subscription
    # looks up another typesupport and librcutils prints the overwrite.
    clear_rcutils_error()
    got: queue.Queue = queue.Queue()
    qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
    sub = node.create_subscription(msg_type, topic, got.put, qos)
    try:
        return got.get(timeout=REPLY_S)
    except queue.Empty:
        logger.warning("no %s on %s within %.1f s", msg_type.__name__, topic, REPLY_S)
        return None
    finally:
        node.destroy_subscription(sub)


def _optical_in_base(node):
    """The wrist optical frame's rotation and origin in ``base_link``.

    The listener's subscriptions hold the listener, so ``del`` never frees
    them. Each one left behind takes every ``/tf`` message on ``node``.
    """
    from rclpy.duration import Duration
    from rclpy.time import Time
    from scipy.spatial.transform import Rotation
    from tf2_ros import Buffer, TransformListener

    from vla_pipeline.robot.moveit_arm import REPLY_S, clear_rcutils_error

    clear_rcutils_error()
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    try:
        stamped = buffer.lookup_transform(
            "base_link", "wrist_cam_optical_frame", Time(), timeout=Duration(seconds=REPLY_S),
        )
    finally:
        listener.unregister()
    rotation = Rotation.from_quat([
        stamped.transform.rotation.x, stamped.transform.rotation.y,
        stamped.transform.rotation.z, stamped.transform.rotation.w,
    ]).as_matrix()
    origin = np.array([
        stamped.transform.translation.x, stamped.transform.translation.y, stamped.transform.translation.z,
    ])
    return rotation, origin


def _camera_info(model):
    """A ``CameraInfo`` built from the perception process's camera model."""
    from sensor_msgs.msg import CameraInfo

    info = CameraInfo()
    info.width = int(model.width)
    info.height = int(model.height)
    info.k = [float(v) for v in model.k]
    info.d = [float(v) for v in model.d]
    info.distortion_model = model.distortion_model or "plumb_bob"
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    fx, _, cx, _, fy, cy, _, _, _ = info.k
    info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return info


def _hold_out(ctx, spoken: str) -> bool:
    """Walk the scan views until one sees ``spoken`` and its pick arrives, then hold it.

    Every object a view detects goes into the planning scene before that
    view's pick plans.
    """
    from sensor_msgs.msg import CameraInfo, Image

    from vla_pipeline.vision.camera import from_image_msg
    from vla_pipeline.vision.yolo_detect_npu import CLASS_SIZE, locate_detection, resolve_class

    arm, cfg, detector = ctx.arm, ctx.cfg, ctx.detector
    if getattr(ctx, "mailbox", None) is None and hasattr(ctx, "arm_camera") and ctx.arm_camera is None:
        ctx.say("The arm camera is offline")
        return False
    cam = cfg["cameras"]["arm"]
    # The twin tests leave backend at real and construct the arm with sim=True.
    # arm.sim is that flag. A real arm has no wrist_cam model, so it refuses
    # until cameras.arm intrinsics and extrinsics are set.
    if (
        not getattr(arm, "sim", False)
        and cfg.get("robot", {}).get("backend", "real") == "real"
        and not (cam.get("intrinsics") and cam.get("extrinsics"))
    ):
        ctx.say(
            "The arm camera isn't calibrated. "
            "lerobot-calibrate, camera_calibration, and easy_handeye2 are not installed."
        )
        return False
    if resolve_class(spoken) is None:
        ctx.say(f"I don't know what a {spoken} looks like.")
        return False
    resolved = resolve_class(spoken)
    pp = cfg["pick_place"]
    table_z = float(pp["table_z"])
    lift = float(pp["wrist_align_pregrasp_above_m"] if resolved in pp["wrist_align_classes"] else pp["pregrasp_above_m"])
    image_topic = cfg["cameras"]["arm"]["topic"]
    info_topic = image_topic.removesuffix("image_raw") + "camera_info"
    saw = False
    held = None
    for scan in arm.scan_states():
        if ctx.stop_check():
            return False
        if arm.go_to_named("open") != "arrived" or arm.go_to_named(scan) != "arrived":
            return False
        frame = None
        if getattr(ctx, "mailbox", None) is not None:
            stale = float(cam.get("stale_after_s", 1.0))
            observation = ctx.mailbox.wait_after("arm", time.time(), stale, ctx.stop_event)
            if ctx.stop_check():
                return False
            if observation is None:
                ctx.say("The arm camera is offline")
                return False
            if observation.camera is None:
                ctx.say("The wrist image is arriving, but its calibration message has not.")
                return False
            info = _camera_info(observation.camera)
            seen = [det for det in observation.detections if det.name in CLASS_SIZE]
        else:
            image = _ros_message(arm.node, Image, image_topic)
            info = None if image is None else _ros_message(arm.node, CameraInfo, info_topic)
            if image is None or info is None:
                ctx.say("The arm camera is offline")
                return False
            frame = from_image_msg(image)
            if frame is None:
                continue
            seen = [det for det in detector.infer(frame) if det.name in CLASS_SIZE]
        if not seen:
            continue
        optical = _optical_in_base(arm.node)
        tries = []
        for det in seen:
            size = CLASS_SIZE[det.name]
            axis = getattr(det, "axis", None)
            located = locate_detection(
                det.box, info, optical, table_z, det.name, frame,
                float(pp["wrist_axis_min_elongation"]),
                **({} if axis is None else {"axis": axis}),
            )
            if located is None:
                continue
            box = arm.grasp_box(size, located["yaw"])
            if det.name == resolved and box is None:
                ctx.say(f"The {spoken} is too wide for my fingers.")
                return False
            if det.name == resolved:
                saw = True
            if box is None:
                arm.remember(det.name, located["xyz"], size, located["yaw"])
                continue
            grip_size, yaw = box
            arm.remember(det.name, located["xyz"], grip_size, yaw)
            if det.name == resolved:
                tries.append((det.score, located["xyz"], grip_size, yaw))
        # The highest score first. A pose with no inverse kinematics is skipped
        # for the next one, then for the next view. The arm is still at this
        # view with the jaws open, above the object, where the pick starts.
        for _score, centre, grip_size, yaw in sorted(tries, key=lambda item: -item[0]):
            how = arm.pick_up(resolved, centre, grip_size, lift, yaw)
            if how == "arrived":
                held = (resolved, grip_size, lift)
                break
            if how == "missed":
                arm.go_to_named("open")
                ctx.say(f"I couldn't get a grip on the {spoken}.")
                return False
            if how != "NO_PLAN":
                ctx.say(f"I couldn't pick up the {spoken}.")
                return False
        if held is not None:
            break
    if held is None:
        ctx.say(
            f"I can't see a {spoken} in the workspace." if not saw
            else f"I couldn't pick up the {spoken}."
        )
        return False
    _resolved, size, lift = held
    where = _prop_position(arm, resolved)
    centre, _orientation = _held_object_pose(arm, size)
    if where is not None:
        logger.info("%s is %.1f mm from the jaws", resolved, 1000 * float(np.linalg.norm(where - centre)))
    if where is not None and float(np.linalg.norm(where - centre)) > 0.03:
        arm.go_to_named("open")
        ctx.say(f"I dropped the {spoken}.")
        return False
    grasp = tip_pose(arm)
    state = getattr(ctx, "shared_state", None)
    if state is not None:
        held = grasp.orientation
        state["picked"] = {
            "name": spoken,
            "object": resolved,
            "size": size,
            "lift": lift,
            "tip_z": grasp.position.z - lift,
            "hold": (held.x, held.y, held.z, held.w),
        }
        state["grip_lock"] = True
    ctx.say(f"Got the {resolved}.")
    return True


_TWIN_BODY = {
    "sports ball": "sports_ball",
    "bottle": "bottle",
    "book": "book",
    "toothbrush": "toothbrush",
}


def _held_object_pose(arm, size):
    """The object's centre and orientation while it sits in the jaws."""
    from scipy.spatial.transform import Rotation

    from vla_pipeline.robot.moveit_arm import grasp_frame

    tip = tip_pose(arm)
    grasp = grasp_frame(size)
    tip_rotation = Rotation.from_quat([
        tip.orientation.x, tip.orientation.y, tip.orientation.z, tip.orientation.w,
    ])
    grasp_rotation = Rotation.from_quat([
        grasp.orientation.x, grasp.orientation.y, grasp.orientation.z, grasp.orientation.w,
    ])
    centre = np.array([tip.position.x, tip.position.y, tip.position.z]) + tip_rotation.apply(
        [grasp.position.x, grasp.position.y, grasp.position.z]
    )
    return centre, (tip_rotation * grasp_rotation).as_quat()


def _prop_position(arm, class_name: str):
    """Where the twin's prop is, or None when this arm is not the twin."""
    from mujoco_ros2_control_msgs.msg import FreeJointStateArray

    body = _TWIN_BODY.get(class_name)
    if body is None or not getattr(arm, "sim", False):
        return None
    msg = _ros_message(arm.node, FreeJointStateArray, "/free_joint_states")
    if msg is None:
        return None
    for item in msg.free_joints:
        if item.name == body:
            position = item.pose.pose.position
            return np.array([position.x, position.y, position.z])
    return None


def _set_down(ctx) -> bool:
    """Set the held object back on the table and open the jaw."""
    from geometry_msgs.msg import PoseStamped

    picked = (getattr(ctx, "shared_state", None) or {}).get("picked") or {}
    name = picked.get("name")
    if picked.get("object") is None or picked.get("size") is None:
        ctx.say("I'm not holding anything I picked up.")
        return False
    pp = ctx.cfg["pick_place"]
    tip = tip_pose(ctx.arm)
    pose = PoseStamped()
    pose.header.frame_id = "base_link"
    pose.pose.position.x = float(tip.position.x)
    pose.pose.position.y = float(tip.position.y)
    pose.pose.position.z = (
        float(pp["table_z"]) + 0.5 * float(picked["size"][2]) + float(pp["set_down_clearance_m"])
    )
    pose.pose.orientation.w = 1.0
    if ctx.arm.place_down(picked["object"], pose, picked["size"]) != "arrived":
        ctx.say(f"I can't set the {name} down.")
        return False
    where = _prop_position(ctx.arm, picked["object"])
    target = np.array([pose.pose.position.x, pose.pose.position.y, pose.pose.position.z])
    landed = where is None or float(np.linalg.norm(where - target)) <= 0.04
    state = ctx.shared_state
    for key in ("grip_lock", "grip_lock_pos", "picked"):
        state.pop(key, None)
    if not landed:
        ctx.say(f"I dropped the {name}.")
        return False
    ctx.say("Done.")
    return True


def run_pick_place(ctx, cmd) -> bool:
    """Find the named object, grasp it, and hold it out. The jaws stay closed."""
    name = (cmd.object_name if cmd is not None else "") or ""
    # The detector runs in the perception process. Its wrist observations are
    # the mailbox. A local detector is only the in-process path.
    if ctx.detector is None and getattr(ctx, "mailbox", None) is None:
        ctx.say("Object detection isn't available - can't do pick and place.")
        return False
    if not name:
        ctx.say("What should I pick up? Try: pick up the ball.")
        return False
    return _hold_out(ctx, name)


def run_release(ctx, cmd=None) -> bool:
    """Open the jaws where they are. The arm stays put."""
    state = getattr(ctx, "shared_state", None)
    picked = (state or {}).get("picked") or {}
    if state is not None:
        for key in ("grip_lock", "grip_lock_pos", "picked"):
            state.pop(key, None)
    opened = ctx.arm.go_to_named("open") == "arrived"
    if picked.get("object") is not None:
        ctx.arm.let_go(picked["object"])
    ctx.say("Released.")
    return opened and not ctx.stop_check()


def run_place(ctx, cmd=None) -> bool:
    """Lower the object the pickup is holding and open the jaws."""
    return _set_down(ctx)


def release_to_pinch(state: dict) -> bool:
    """Hand the jaws to the pinch (gesture mimic). A locked grip ends. True
    when a grip was locked."""
    if not state.pop("grip_lock", None):
        return False
    for key in ("grip_lock_pos", "picked", "table_objects"):
        state.pop(key, None)
    return True


def forget_scene(state: dict) -> None:
    """What a scene reset forgets: the locked grip, the picked object, and the known objects."""
    for key in ("grip_lock", "grip_lock_pos", "picked", "table_objects"):
        state.pop(key, None)


def tip_pose(arm, joints: dict | None = None):
    """Where the tip link is, with ``joints`` (URDF radians) on top of the current state."""
    from moveit.core.robot_state import RobotState

    from vla_pipeline.robot.moveit_arm import TIP_LINK

    state = RobotState(arm.robot.get_robot_model())
    state.joint_positions = {**arm.get_joints(), **(joints or {})}
    state.update()
    return state.get_pose(TIP_LINK)


def extend_handoff(arm, cfg: dict, pan_deg: float = 0.0) -> bool:
    """Aim the shoulder pan at ``pan_deg`` and leave every other joint where it is.

    A straight tip move that keeps the gripper's orientation has no inverse
    kinematics from the home pose or from the pose after a pick, so this is
    one joint move of the pan only. The wrist joints stay, which is what keeps
    a held object from being tipped out. The MoveIt arm reports URDF radians.
    The in-process twin reports calibrated degrees under the short names.
    """
    now = arm.get_joints()
    if "shoulder_pan_joint" in now:
        goal = {joint: float(now[joint]) for joint in arm.arm_joints}
        goal["shoulder_pan_joint"] = float(
            radians_from_calibrated(cfg, {"shoulder_pan": float(pan_deg)})["shoulder_pan_joint"]
        )
    else:
        goal = {joint: float(value) for joint, value in now.items() if joint != "gripper"}
        goal["shoulder_pan"] = float(pan_deg)
    return arm.go_to_mtc(goal) == "arrived"


def extend_toward_person(arm, cfg: dict, pose, frame) -> float:
    """Aim the fetch extend at the person in ``frame``. Straight ahead when there is nobody.

    Returns the pan in degrees.
    """
    pan = person_pan_target(pose, frame)
    if pan is None:
        pan = 0.0
    if not extend_handoff(arm, cfg, pan):
        raise RuntimeError("the handover extend did not arrive")
    return pan


def present_and_close(arm, cfg: dict) -> bool:
    """Present the open jaws, then close them. True when the close stalls on contact."""
    target = radians_from_calibrated(cfg, PRESENT_POSE)
    opening = target.pop(arm.gripper_joint)
    if arm.go_to_mtc(target) != "arrived":
        raise RuntimeError("the present pose did not arrive")
    arm.grip(opening)
    return bool(arm.grip(arm.named_state("close")[arm.gripper_joint]))


def register_manipulation(registry) -> None:
    """Add grip, release, place, pick_place and fetch_block to ``registry``."""
    registry.add(Intent.GRIP, run_grip)
    registry.add(Intent.RELEASE, run_release)
    registry.add(Intent.PLACE, run_place)
    registry.add(Intent.PICK_PLACE, run_pick_place)
    registry.add(Intent.FETCH_BLOCK, run_fetch_block)
