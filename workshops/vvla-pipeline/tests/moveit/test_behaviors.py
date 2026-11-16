# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Home, wave, dance and a streamed mimic setpoint on MoveItArm.

The pose tables stay in the behaviors, in calibrated degrees. ``radians_from_calibrated``
converts them, and the arm plays them at the durations those behaviors use,
on the trajectory controller. There is no separate person tracker: mimic and
tracking are both streamed setpoints, which is the last test.
"""

from __future__ import annotations

import threading

import pytest

import conftest
from conftest import PAN, next_message, off

from vla_pipeline.robot.arm_interface import radians_from_calibrated

pytestmark = pytest.mark.sim
launch = pytest.mark.launch(fixture=conftest.twin_bringup)


@launch
def test_home_parks_at_the_named_rest_pose(arm):
    from sensor_msgs.msg import JointState

    assert arm.go_to_rest() == "arrived"
    joints = arm.get_joints()
    assert off(joints, arm.named_state("home")) < 0.01
    jaw = next_message(arm.node, JointState, "/joint_states")
    position = dict(zip(jaw.name, jaw.position))[arm.gripper_joint]
    assert abs(position - arm.named_state("close")[arm.gripper_joint]) < 0.01


@launch
def test_run_wave_reaches_both_sides_and_parks(arm, pipeline_cfg):
    """``run_wave`` itself, one cycle, on this arm."""
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    from vla_pipeline.behaviors.simple import WAVE_LEFT, WAVE_RIGHT, run_wave

    seen: list[float] = []

    def remember(msg):
        position = dict(zip(msg.name, msg.position))
        if PAN in position:
            seen.append(position[PAN])

    sub = arm.node.create_subscription(JointState, "/joint_states", remember, 10)
    stop = threading.Event()
    cfg = {**pipeline_cfg, "wave": {"cycles": 1}}
    try:
        run_wave(SimpleNamespace(cfg=cfg, arm=arm, stop_event=stop, stop_check=stop.is_set))
    finally:
        arm.node.destroy_subscription(sub)
    left = radians_from_calibrated(cfg, WAVE_LEFT)[PAN]
    right = radians_from_calibrated(cfg, WAVE_RIGHT)[PAN]
    print(f"wave pan {min(seen):.3f} .. {max(seen):.3f}  left {left:.3f}  right {right:.3f}")
    assert min(seen) < left + 0.05
    assert max(seen) > right - 0.05
    assert off(arm.get_joints(), arm.named_state("home")) < 0.05


@launch
def test_the_wave_reaches_both_sides_in_its_own_durations(arm, pipeline_cfg):
    """The wave's raise takes 2 s and each swing 0.4 s, the same as run_wave.

    The clock is the stamp of the joint state that is at the target. The
    state this process has handled when the action result arrives can be older.
    """
    from sensor_msgs.msg import JointState

    from vla_pipeline.behaviors.simple import WAVE_LEFT, WAVE_RIGHT, WAVE_UP

    def lasted(target, duration, accept):
        started = arm.state_time()
        assert arm.go_to(target, duration_s=duration) == "arrived"
        msg = next_message(arm.node, JointState, "/joint_states", lambda m: accept(dict(zip(m.name, m.position))))
        took = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9 - started
        assert took >= duration - 0.05, f"the keyframe ended {took:.3f} s after it was sent"

    raised = radians_from_calibrated(pipeline_cfg, WAVE_UP)
    lasted(raised, 2.0, lambda pos: off(pos, raised) < 0.05)
    for swing, duration in ((WAVE_LEFT, 0.4), (WAVE_RIGHT, 0.4)):
        target = radians_from_calibrated(pipeline_cfg, swing)
        lasted(target, duration, lambda pos, pan=target[PAN]: abs(pos[PAN] - pan) < 0.02)


@launch
def test_run_dance_plays_its_first_keyframe_and_parks(arm, pipeline_cfg):
    """``run_dance`` itself, one keyframe, on this arm."""
    import random
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    from vla_pipeline.behaviors.dance import DANCE_MOVES, run_dance

    order = list(DANCE_MOVES)
    random.Random(0).shuffle(order)
    first = DANCE_MOVES[order[0]][0]
    target = radians_from_calibrated(pipeline_cfg, first)
    name = next(iter(target))
    seen: list[float] = []

    def remember(msg, name=name):
        position = dict(zip(msg.name, msg.position))
        if name in position:
            seen.append(position[name])

    sub = arm.node.create_subscription(JointState, "/joint_states", remember, 10)
    stop = threading.Event()
    try:
        run_dance(
            SimpleNamespace(cfg=pipeline_cfg, arm=arm, stop_event=stop, stop_check=stop.is_set),
            duration_s=0.2, seed=0,
        )
    finally:
        arm.node.destroy_subscription(sub)
    print(f"dance {order[0]} {name} saw {min(seen):.3f}..{max(seen):.3f}  goal {target[name]:.3f}")
    assert any(abs(value - target[name]) < 0.05 for value in seen)
    assert off(arm.get_joints(), arm.named_state("home")) < 0.05


@launch
def test_a_stop_during_the_dance_cancels_the_move(arm, pipeline_cfg):
    """A stop during the first dance move ends ``run_dance``."""
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    from vla_pipeline.behaviors.dance import run_dance

    stop = threading.Event()
    worker = threading.Thread(target=lambda: run_dance(
        SimpleNamespace(cfg=pipeline_cfg, arm=arm, stop_event=stop, stop_check=stop.is_set),
        duration_s=30.0, seed=0,
    ))
    worker.start()
    next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN]) > 0.2,
    )
    stop.set()
    worker.join(30)
    assert not worker.is_alive()


@launch
def test_a_dance_keyframe_plays_for_nine_tenths_of_a_beat(arm, pipeline_cfg):
    """run_dance gives each keyframe ``0.9 * 60/bpm`` seconds."""
    from vla_pipeline.behaviors.dance import DANCE_MOVES

    beat = 0.9 * 60.0 / float(pipeline_cfg["dance"]["bpm"])
    target = radians_from_calibrated(pipeline_cfg, DANCE_MOVES["sway"][0])
    started = arm.state_time()
    assert arm.go_to(target, duration_s=beat) == "arrived"
    took = arm.state_time() - started
    assert took >= beat - 0.05
    # The beat is shorter than the wrist's travel, so the check is the pan,
    # which is what the sway moves from side to side.
    assert abs(arm.get_joints()[PAN] - target[PAN]) < 0.05, f"the sway's pan was still {took:.2f} s into a {beat:.2f} s beat"


@launch
def test_a_stop_during_the_wave_holds_short_of_the_swing(arm, pipeline_cfg):
    from sensor_msgs.msg import JointState

    from vla_pipeline.behaviors.simple import WAVE_LEFT, WAVE_RIGHT

    arm.go_to(radians_from_calibrated(pipeline_cfg, WAVE_LEFT), duration_s=0.4)
    start = arm.get_joints()[PAN]
    goal = radians_from_calibrated(pipeline_cfg, WAVE_RIGHT)[PAN]
    stop = threading.Event()
    result: list = []
    mover = threading.Thread(
        target=lambda: result.append(arm.go_to({PAN: goal}, duration_s=3.0, stop_event=stop)),
        daemon=True,
    )
    mover.start()
    next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN] - start) > 0.05,
    )
    stop.set()
    mover.join()
    assert result == ["stopped"]
    assert abs(arm.get_joints()[PAN] - goal) > 0.2


@launch
def test_mimic_parks_at_home_when_it_stops(arm, pipeline_cfg):
    """The park after mimic is the SRDF home pose on this arm."""
    from types import SimpleNamespace

    from vla_pipeline.behaviors.gesture_mimic import park_after_mimic

    assert arm.go_to(radians_from_calibrated(pipeline_cfg, {"shoulder_pan": 25.0}), duration_s=1.0) == "arrived"
    park_after_mimic(SimpleNamespace(arm=arm, cfg=pipeline_cfg))
    assert off(arm.get_joints(), arm.named_state("home")) < 0.05


@launch
def test_mimic_parks_at_home_when_it_stops_mid_stream(arm, pipeline_cfg):
    """The park abandons a streamed setpoint the arm is still more than
    ``REPLY_S`` of travel from, at the lift's ``max_velocity``, and parks at home."""
    from types import SimpleNamespace

    from vla_pipeline.behaviors.gesture_mimic import park_after_mimic
    from vla_pipeline.robot.moveit_arm import PLANNING_GROUP, REPLY_S, moveit_configs

    lift = "shoulder_lift_joint"
    speed = float(moveit_configs().joint_limits["robot_description_planning"]["joint_limits"][lift]["max_velocity"])
    group = arm.robot.get_robot_model().get_joint_model_group(PLANNING_GROUP)
    upper = dict(zip(group.active_joint_model_names, group.active_joint_model_bounds))[lift][0].max_position
    start = arm.get_joints()[lift]
    far = start + 1.2 * REPLY_S * speed
    assert far < upper, "the lift has no setpoint that far from here"
    arm.send_joints({lift: far})
    park_after_mimic(SimpleNamespace(arm=arm, cfg=pipeline_cfg))
    assert off(arm.get_joints(), arm.named_state("home")) < 0.05


@launch
def test_mimic_tick_streams_the_wrist_as_a_urdf_setpoint(arm, pipeline_cfg):
    """One ``mimic_tick`` on a visible right wrist streams radians on this arm."""
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    import numpy as np

    from vla_pipeline.behaviors.gesture_mimic import mimic_tick
    from vla_pipeline.utils.gesture_map import mapper_from_config
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

    signals = mimic_tick(SimpleNamespace(
        pose=_Pose(), hands=None, mapper=mapper_from_config(pipeline_cfg),
        arm=arm, cfg=pipeline_cfg,
    ), np.zeros((480, 640, 3), np.uint8))
    target = radians_from_calibrated(pipeline_cfg, signals.joints)[PAN]
    seen = next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN] - target) < 0.05,
    )
    reached = dict(zip(seen.name, seen.position))[PAN]
    print(f"mimic pan {reached:.4f} rad  target {target:.4f} rad")
    assert abs(reached - target) < 0.05


@launch
def test_mimic_follows_generated_poses_to_its_range_ends_and_parks(arm, pipeline_cfg):
    """``run_mimic`` on generated YOLO-pose people and MediaPipe hands, one frame
    per 1/30 s of the arm's clock, from the start the twin resets to. The person
    reaches high left toward the camera, then low right away from it, where
    mimic's elbow setpoint is past the URDF limit. The session ends there, and
    the arm parks at home.

    Each pose is held for the frames mimic's smoothing takes to settle on it,
    plus the time joint_limits.yaml gives the arm for that move: Servo scales
    every joint to the one furthest from its speed limit, and the shoulder lift's
    limit is the lowest."""
    import math
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    import numpy as np

    from vla_pipeline.behaviors.gesture_mimic import mimic_tick, run_mimic
    from vla_pipeline.robot.moveit_arm import PLANNING_GROUP, moveit_configs
    from vla_pipeline.utils.gesture_map import mapper_from_config
    from vla_pipeline.vision.mediapipe_hands import HandObservation
    from vla_pipeline.vision.yolo_pose_npu import KP

    width, height, period = 640, 480, 1.0 / 30.0
    neutral = 0.16
    # (wrist x, wrist y as frame fractions, hand size, hand direction deg, pinch)
    keyframes = [
        (0.5, 0.5, neutral, 0.0, 1.4),
        (0.1, 0.15, neutral * 1.3, -60.0, 0.1),
        (0.5, 0.5, neutral, 0.0, 1.4),
        (0.9, 0.85, neutral * 0.7, 60.0, 1.4),
    ]
    shown = [keyframes[0]]

    class _Camera:
        def __init__(self):
            self.index = -1

        def read(self):
            arm.wait_state(arm.state_time() + period)
            self.index = min(self.index + 1, len(script) - 1)
            shown[0] = script[self.index]
            return True, np.zeros((height, width, 3), np.uint8)

    camera = _Camera()

    class _Pose:
        provider = "generated"
        kpt_threshold = 0.3

        def infer(self, frame):
            x, y = shown[0][0] * width, shown[0][1] * height
            shoulder = (0.55 * width, 0.35 * height)
            kpts = np.zeros((1, 17, 3), np.float32)
            kpts[0, KP["nose"]] = (0.5 * width, 0.2 * height, 0.9)
            kpts[0, KP["l_shoulder"]] = (0.45 * width, 0.35 * height, 0.9)
            kpts[0, KP["r_shoulder"]] = (*shoulder, 0.9)
            kpts[0, KP["r_elbow"]] = ((shoulder[0] + x) / 2, (shoulder[1] + y) / 2 + 0.05 * height, 0.9)
            kpts[0, KP["r_wrist"]] = (x, y, 0.95)
            boxes = np.array([[0.3 * width, 0.1 * height, 0.7 * width, 0.95 * height]], np.float32)
            return boxes, np.array([0.93], np.float32), kpts

    class _Hands:
        def process(self, frame):
            x, y, size, direction, pinch = shown[0]
            wrist = (int(x * width), int(y * height))
            return HandObservation(
                pinch=pinch, thumb_px=wrist, index_px=wrist, handedness="Right",
                hand_size=size, roll=direction, roll_ref=direction, wrist_px=wrist,
            ), None

        def draw(self, frame, obs, raw):
            return frame

    class _Clocked:
        """The mapper on a clock of one ``period`` per frame."""

        def __init__(self):
            self.mapper, self.t = mapper_from_config(pipeline_cfg), 0.0

        def step(self, *args, **kwargs):
            self.t += period
            return self.mapper.step(*args, t=self.t, **kwargs)

        def reset(self):
            self.mapper.reset()

    arm.reset()
    mimic = pipeline_cfg["mimic"]
    # The smoothing at its slowest is a low-pass of time constant 1/(2 pi min_cutoff);
    # ln(1e4) of those leave a step 1e-4 short. The hand-size baseline comes first.
    tau = 1.0 / (2.0 * math.pi * float(mimic["smoothing"]["min_cutoff"]))
    hold = int(mimic["depth_baseline_frames"]) + math.ceil(math.log(1e4) * tau / period)
    limits = moveit_configs().joint_limits["robot_description_planning"]["joint_limits"]
    sent: list[dict] = []
    offline = SimpleNamespace(
        pose=_Pose(), hands=_Hands(), mapper=_Clocked(), cfg=pipeline_cfg,
        arm=SimpleNamespace(send_joints=sent.append, send_gripper=lambda _jaw: None, gripper_joint=arm.gripper_joint),
    )
    blank = np.zeros((height, width, 3), np.uint8)
    previous = arm.get_joints()
    script: list[tuple] = []
    for key in keyframes:
        shown[0] = key
        sent.clear()
        for _ in range(hold):
            mimic_tick(offline, blank)
        steady = {name: sent[-1][name] for name in arm.arm_joints}
        settle = next(i for i, joints in enumerate(sent) if off(joints, steady) < 0.01)
        # A long move cruises at max_velocity after a ramp of max_velocity/max_acceleration.
        move_s = max(
            abs(steady[name] - previous[name]) / float(limits[name]["max_velocity"])
            + float(limits[name]["max_velocity"]) / float(limits[name]["max_acceleration"])
            for name in arm.arm_joints
        )
        script += [key] * (settle + math.ceil(move_s / period))
        previous = steady

    commanded: list[dict] = []

    def tick(ctx, frame):
        signals = mimic_tick(ctx, frame)
        if signals.joints:
            commanded.append(radians_from_calibrated(ctx.cfg, signals.joints))
        return signals

    pans: list[float] = []

    def remember(msg):
        position = dict(zip(msg.name, msg.position))
        if PAN in position:
            pans.append(position[PAN])

    sub = arm.node.create_subscription(JointState, "/joint_states", remember, 10)
    stop = threading.Event()
    try:
        run_mimic(SimpleNamespace(
            cfg=pipeline_cfg, arm=arm, pose=_Pose(), hands=_Hands(), mount_camera=camera, mapper=None,
            mailbox=None,  # the in-process camera path; PipelineContext defaults this to None
            stop_check=stop.is_set, say=print, shared_state={}, headless=True, max_frames=len(script),
        ), tick=tick)
    finally:
        arm.node.destroy_subscription(sub)

    elbow = "elbow_flex_joint"
    group = arm.robot.get_robot_model().get_joint_model_group(PLANNING_GROUP)
    upper = dict(zip(group.active_joint_model_names, group.active_joint_model_bounds))[elbow][0].max_position
    assert commanded[-1][elbow] > upper, "the last generated pose no longer sends the elbow past the URDF limit"
    left, right = max(c[PAN] for c in commanded), min(c[PAN] for c in commanded)
    print(f"mimic pan {min(pans):.3f} .. {max(pans):.3f}  commanded {right:.3f} .. {left:.3f}")
    assert max(pans) > left - 0.05 and min(pans) < right + 0.05
    assert off(arm.get_joints(), arm.named_state("home")) < 0.05


@launch
def test_find_person_pan_aims_the_streamed_pan(arm, pipeline_cfg):
    """A person left of centre streams that pan, in radians, on this arm."""
    from sensor_msgs.msg import JointState
    from types import SimpleNamespace

    import numpy as np

    from vla_pipeline.behaviors.manipulation import find_person_pan, person_pan_target

    class _Pose:
        kpt_threshold = 0.3

        def primary_person(self, frame):
            kpts = np.zeros((17, 3), np.float32)
            height, width = frame.shape[:2]
            x = 0.3 * width
            kpts[0] = (x, 0.4 * height, 0.9)
            kpts[5] = (x - 0.05 * width, 0.5 * height, 0.9)
            kpts[6] = (x + 0.05 * width, 0.5 * height, 0.9)
            return kpts, 0.9

    class _Cam:
        def read_after(self, _settled_at):
            return True, np.zeros((480, 640, 3), np.uint8)

    frame = np.zeros((480, 640, 3), np.uint8)
    pan = person_pan_target(_Pose(), frame)
    found_pan, found = find_person_pan(SimpleNamespace(
        pose=_Pose(), mount_camera=_Cam(), arm=arm, cfg=pipeline_cfg, stop_check=lambda: False,
    ), seek_s=0.2)
    assert found and found_pan == pytest.approx(pan)
    target = radians_from_calibrated(pipeline_cfg, {"shoulder_pan": pan})[PAN]
    seen = next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN] - target) < 0.05,
    )
    reached = dict(zip(seen.name, seen.position))[PAN]
    print(f"track pan {found_pan:.1f} deg  joint {reached:.4f} rad  target {target:.4f} rad")
    assert abs(reached - target) < 0.05


@launch
def test_a_mimic_setpoint_and_a_tracking_pan_stream_to_the_arm(arm, pipeline_cfg):
    """Gesture mimic and person tracking both stream one setpoint. The mapper
    owns the mimic pose; the tracking pan is a setpoint on the same topic."""
    from sensor_msgs.msg import JointState

    from vla_pipeline.utils.gesture_map import mapper_from_config

    mimic = radians_from_calibrated(pipeline_cfg, mapper_from_config(pipeline_cfg).step((120.0, 240.0), 640, 480, t=0.0))
    arm.send_joints(mimic)
    next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN] - mimic[PAN]) < 0.05,
    )
    tracked = arm.get_joints()[PAN] + 0.25
    arm.send_joints({PAN: tracked})
    seen = next_message(
        arm.node, JointState, "/joint_states",
        lambda msg: abs(dict(zip(msg.name, msg.position))[PAN] - tracked) < 0.02,
    )
    assert abs(dict(zip(seen.name, seen.position))[PAN] - tracked) < 0.02
