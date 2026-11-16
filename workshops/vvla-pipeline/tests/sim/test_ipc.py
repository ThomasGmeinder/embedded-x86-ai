# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The process contract: a late, duplicate, or malformed message does nothing.

Two processes share the encoded bytes. The child sends a command, the parent
stops, and the child's late copy of that command is rejected.
"""

from __future__ import annotations

import multiprocessing

import pytest

from vla_pipeline.ipc import (
    ClickOutbox,
    ContractError,
    Ledger,
    Mailbox,
    MountObservation,
    SpeechOutbox,
    VoiceCommand,
    decode_command,
    decode_mount,
    encode_command,
    encode_mount,
)

pytestmark = pytest.mark.sim


def test_a_bad_payload_is_rejected():
    with pytest.raises(ContractError):
        decode_command("{")
    with pytest.raises(ContractError):
        decode_command('{"epoch": 1, "command_seq": 1, "command": "wave"}')
    with pytest.raises(ContractError):
        decode_command(
            '{"epoch": 0, "command_seq": 1, "command": "wave", "object": "", "heard": "wave"}'
        )


def test_a_command_from_an_old_epoch_is_rejected_and_a_sequence_is_accepted_once():
    ledger = Ledger()
    assert ledger.claim_stop(1) == 2
    assert ledger.claim_stop(1) is None
    assert ledger.accept_command(1, 1) is False
    assert ledger.accept_command(2, 1) is True
    assert ledger.accept_command(2, 1) is False
    assert ledger.accept_command(2, 2) is True


def test_two_resets_coalesce_into_one_extra_pass():
    ledger = Ledger()
    assert ledger.claim_reset(1) == 2
    assert ledger.claim_reset(2) == "again"
    assert ledger.claim_reset(1) is None
    assert ledger.finish_reset(run_again=True) == 3
    assert ledger.resetting is True
    assert ledger.finish_reset(run_again=False) is None
    assert ledger.resetting is False
    assert ledger.epoch == 3


def test_an_observation_whose_stamp_went_backwards_is_ignored():
    ledger = Ledger()
    assert ledger.accept_observation("mount", 1, 10.0) is True
    assert ledger.accept_observation("mount", 1, 11.0) is False
    assert ledger.accept_observation("mount", 2, 9.0) is False
    assert ledger.accept_observation("mount", 3, 11.0) is True


def _child(conn) -> None:
    conn.send(encode_command(VoiceCommand(1, 1, "wave", "", "wave at me")))
    conn.recv()
    conn.send(encode_command(VoiceCommand(1, 1, "wave", "", "wave at me")))
    conn.close()


def test_a_command_spoken_before_the_epoch_is_kept_and_a_stop_goes_first():
    speech = SpeechOutbox()
    speech.offer_command("pick_place", "boy", "Pick up the boy")
    assert speech.take() is None
    speech.offer_stop()
    speech.set_epoch(1)
    assert speech.take() == ("stop", 1)
    assert speech.take() is None
    speech.set_epoch(2)
    sent = speech.take()
    assert sent[0] == "command" and sent[1] == 2 and sent[3:] == ("pick_place", "boy", "Pick up the boy")


def test_the_joint_sample_carries_the_gripper_current():
    from vla_pipeline.ipc import decode_joints, encode_joints
    from vla_pipeline.robot.moveit_arm import GRIPPER_STALL_MA, torque_nm_to_milliamps

    raw = encode_joints({"gripper_joint": 0.4}, 2080.0)
    joints, load = decode_joints(raw)
    assert joints == {"gripper_joint": 0.4}
    assert load == 2080.0
    assert torque_nm_to_milliamps(0.0) == pytest.approx(6.0)
    assert torque_nm_to_milliamps(1.91) == pytest.approx(GRIPPER_STALL_MA)


def test_control_forwards_the_arms_gripper_current_without_reading_the_arm():
    from types import SimpleNamespace

    from vla_pipeline.ipc import decode_joints
    from vla_pipeline.wire import ControlLink

    sent = []
    link = ControlLink.__new__(ControlLink)
    link.ctx = SimpleNamespace(arm=None)
    link._joints_pub = SimpleNamespace(publish=lambda msg: sent.append(msg.data))
    link.set_joints({"gripper_joint": 0.4}, 142.0)
    assert decode_joints(sent[-1]) == ({"gripper_joint": 0.4}, 142.0)
    link.set_joints({"gripper_joint": 0.4})
    assert decode_joints(sent[-1]) == ({"gripper_joint": 0.4}, None)


def _console_with(arm):
    from types import SimpleNamespace

    from vla_pipeline.console import TwinConsole

    ctx = SimpleNamespace(
        cfg={}, arm=arm,
        mount_camera=None, arm_camera=None, pose=None, hands=None, detector=None,
        say=lambda _message: None,
    )
    return TwinConsole(ctx, voice="typed")


def test_the_console_draws_only_what_the_arm_feeds_it():
    import numpy as np

    from vla_pipeline.console import GREEN, PANEL_W
    from vla_pipeline.robot.moveit_arm import GRIPPER_STALL_MA

    class _Arm:
        on_cached_joints = None

        def get_gripper_load(self):
            raise AssertionError("the console read the arm")

        def get_joints(self):
            raise AssertionError("the console read the arm")

    arm = _Arm()
    console = _console_with(arm)
    assert arm.on_cached_joints == console.apply_joints
    blank = np.zeros((200, PANEL_W, 3), np.uint8)
    console._arm_hud(blank, 40)
    # A closed jaw sits near 2080 mA. The bar is full only at stall, 2500 mA.
    arm.on_cached_joints({"gripper_joint": 0.4}, 2080.0)
    fed = np.zeros((200, PANEL_W, 3), np.uint8)
    console._arm_hud(fed, 40)
    bar = fed[49:55, 14:PANEL_W - 14]
    green = (bar == GREEN).all(axis=-1).any(axis=0)
    assert green.any(), "a current over the contact level draws no green bar"
    assert not green[-1], "2080 mA fills the bar, which is full only at stall"
    assert not (blank[49:55, 14:PANEL_W - 14] == GREEN).all(axis=-1).any()
    assert GRIPPER_STALL_MA == 2500.0
    console.close()
    assert arm.on_cached_joints is None


def test_the_window_console_has_no_arm_and_takes_the_link_feed():
    console = _console_with(None)
    console.apply_joints({"gripper_joint": 0.4}, 30.0)
    assert console._gripper_load == 30.0
    console.close()


def test_a_joint_sample_does_not_add_a_console_line_and_a_new_sentence_does():
    import threading
    from collections import deque

    from vla_pipeline.console import TwinConsole
    from vla_pipeline.ipc import ControlState

    console = TwinConsole.__new__(TwinConsole)
    console._lock = threading.Lock()
    console._say = ""
    console._heard = ""
    console._behavior = None
    console._behavior_t0 = 0.0
    console._resetting = False
    console._joints = None
    console._log = deque()
    console.apply_state(ControlState(1, "idle", False, "Ready.", "", {}, 0.0))
    after_ready = len(console._log)
    assert after_ready > 0
    console.apply_joints({"shoulder_pan_joint": 0.1})
    console.apply_joints({"shoulder_pan_joint": 0.2})
    assert len(console._log) == after_ready
    console.apply_state(ControlState(1, "idle", False, "Stopping.", "", {}, 1.0))
    assert len(console._log) > after_ready


def test_the_action_line_follows_the_transcript_and_the_gripper_label_fits():
    import threading
    from collections import deque

    from vla_pipeline.console import TwinConsole, panel_line
    from vla_pipeline.ipc import ControlState

    text, kind = panel_line("parsed", intent="pick_place", params={"object": "ball"})
    assert text == "-> pick_place (object=ball)" and kind == "cyan"
    console = TwinConsole.__new__(TwinConsole)
    console._lock = threading.Lock()
    console._say = ""
    console._heard = ""
    console._behavior = None
    console._behavior_t0 = 0.0
    console._resetting = False
    console._joints = None
    console._log_seq = 0
    console._log = deque()
    console.apply_state(ControlState(
        1, "pick_place", False, "", "pick up the ball", {}, 1.0,
        1, "cyan", "-> pick_place (object=ball)",
    ))
    logged = [line for line, _color in console._log]
    assert logged[0].startswith("you:")
    assert "-> pick_place (object=ball)" in logged
    console.apply_joints({
        "shoulder_pan_joint": 0.1, "shoulder_lift_joint": 0.2, "elbow_flex_joint": -0.3,
        "wrist_flex_joint": 0.4, "wrist_roll_joint": -0.5, "gripper_joint": 0.84,
    })
    assert console._joints["gripper_joint"] == 0.84
    assert len(console._log) == len(logged)


def test_a_reset_wakes_the_ledger():
    from types import SimpleNamespace

    from vla_pipeline.ipc import Ledger
    from vla_pipeline.wire import ControlLink

    ledger = Ledger()
    woken = []
    real_wake = ledger.wake
    ledger.wake = lambda: woken.append(real_wake())
    link = ControlLink.__new__(ControlLink)
    link.ledger = ledger
    link._epoch_pub = SimpleNamespace(publish=lambda _msg: None)
    link._stop = lambda: None
    link.publish_state = lambda: None
    link._after_stop(2)
    assert woken == [None]


def test_a_reset_click_replaces_an_unsent_stop_and_waits_for_an_epoch():
    clicks = ClickOutbox()
    clicks.offer("stop")
    clicks.offer("reset")
    assert clicks.take() is None
    clicks.set_epoch(1)
    assert clicks.take() == ("reset", 1)
    clicks.offer("stop")
    assert clicks.take() is None
    clicks.set_epoch(2)
    assert clicks.take() == ("stop", 2)


def test_a_command_delivered_after_the_stop_does_not_start():
    parent, child = multiprocessing.Pipe()
    proc = multiprocessing.Process(target=_child, args=(child,))
    proc.start()
    child.close()
    ledger = Ledger()
    first = decode_command(parent.recv())
    assert ledger.accept_command(first.epoch, first.command_seq) is True
    assert ledger.claim_stop(first.epoch) == 2
    parent.send("stopped")
    late = decode_command(parent.recv())
    assert ledger.accept_command(late.epoch, late.command_seq) is False
    proc.join(timeout=5)
    assert proc.exitcode == 0


def test_a_silent_perception_parks_the_mimic():
    import threading

    from vla_pipeline.behaviors.gesture_mimic import run_mimic

    class _Arm:
        def __init__(self) -> None:
            self.parked = False

        def stop(self) -> None:
            return None

        def go_to_rest(self) -> None:
            self.parked = True

    class _Ctx:
        pass

    ctx = _Ctx()
    ctx.mailbox = Mailbox(Ledger())
    ctx.pose = None
    ctx.mount_camera = None
    ctx.mapper = type("Mapper", (), {"reset": lambda self: None})()
    ctx.cfg = {"cameras": {"mount": {"stale_after_s": 0.05}}, "yolo_pose": {}}
    ctx.shared_state = {}
    ctx.stop_event = threading.Event()
    ctx.stop_check = ctx.stop_event.is_set
    ctx.headless = True
    ctx.max_frames = 0
    ctx.arm = _Arm()
    ctx.say = lambda _message: None
    run_mimic(ctx)
    assert ctx.arm.parked is True


def test_typed_text_is_the_raw_transcript():
    """A typed line is not a voice command. The control process parses the words."""
    from types import SimpleNamespace

    from vla_pipeline.wire import ControlLink

    heard = []
    link = ControlLink.__new__(ControlLink)
    link.robot_voice = SimpleNamespace(on_transcript=heard.append)
    link._on_text(SimpleNamespace(data="  pick up the toothbrush  "))
    link._on_text(SimpleNamespace(data="   "))
    assert heard == ["pick up the toothbrush"]


def test_a_mailbox_returns_one_new_frame_and_then_times_out():
    box = Mailbox(Ledger())
    frame = MountObservation("mount", 1.0, 1, 4, 4, (), None)
    assert box.offer(frame) is True
    assert box.offer(frame) is False
    assert box.wait_new("mount", 0, 0.1).seq == 1
    assert box.wait_new("mount", 1, 0.05) is None
    assert decode_mount(encode_mount(frame)).seq == 1
