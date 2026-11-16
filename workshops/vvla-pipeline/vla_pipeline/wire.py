# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""ROS topics for :mod:`vla_pipeline.ipc`. One encoder, one decoder, one drop log.

A payload that does not decode is logged once and dropped. Control is the only
writer of the epoch. Perception and the console publish nothing until they have
seen one.
"""

from __future__ import annotations

import logging
import threading
import time

from vla_pipeline.ipc import (
    CONSOLE_RESET,
    EPOCH,
    JOINTS,
    MOUNT,
    OVERLAY_HANDS,
    OVERLAY_MOUNT,
    OVERLAY_WRIST,
    STATE,
    STATUS,
    VOICE_COMMAND,
    VOICE_STOP,
    VOICE_TEXT,
    WRIST,
    ClickOutbox,
    ConsoleReset,
    ContractError,
    ControlState,
    Ledger,
    Mailbox,
    SpeechOutbox,
    PerceptionStatus,
    VoiceCommand,
    VoiceStop,
    decode_command,
    decode_joints,
    decode_mount,
    decode_reset,
    decode_state,
    decode_status,
    decode_stop,
    decode_wrist,
    encode_command,
    encode_joints,
    encode_mount,
    encode_reset,
    encode_state,
    encode_status,
    encode_stop,
    encode_wrist,
)

logger = logging.getLogger(__name__)


def _qos(reliable: bool, depth: int, latched: bool):
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

    return QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=depth,
        durability=DurabilityPolicy.TRANSIENT_LOCAL if latched else DurabilityPolicy.VOLATILE,
    )


class _Drops:
    """Log each distinct contract failure once."""

    def __init__(self) -> None:
        self._seen: set[tuple[str, str]] = set()

    def decode(self, fn, raw: str):
        try:
            return fn(raw)
        except ContractError as exc:
            key = (fn.__name__, str(exc))
            if key not in self._seen:
                self._seen.add(key)
                logger.warning("dropped %s: %s", fn.__name__, exc)
            return None


def _string(data: str):
    from std_msgs.msg import String

    return String(data=data)


def _epoch_msg(epoch: int):
    from std_msgs.msg import UInt64

    return UInt64(data=int(epoch))


class ControlLink:
    """Control's publishers and the checks on the way in.

    ``ctx.request_stop`` bumps the epoch, then stops. A stop message whose
    epoch is already old does not stop a second time.
    """

    def __init__(self, node, ctx, robot_voice, restore) -> None:
        from std_msgs.msg import String, UInt64

        self.node = node
        self.ctx = ctx
        self.robot_voice = robot_voice
        self._restore = restore
        self.ledger = Ledger()
        self.mailbox = Mailbox(self.ledger)
        ctx.mailbox = self.mailbox
        self._drops = _Drops()
        self.behavior = "idle"
        self.say = ""
        self.heard = ""
        self.joints: dict = {}
        self.log_seq = 0
        self.log_kind = ""
        self.log_text = ""
        self._behavior_t0 = time.monotonic()
        self._reset_lock = threading.Lock()
        self._epoch_pub = node.create_publisher(UInt64, EPOCH, _qos(True, 1, True))
        self._state_pub = node.create_publisher(String, STATE, _qos(True, 1, True))
        self._joints_pub = node.create_publisher(String, JOINTS, _qos(True, 1, True))
        node.create_subscription(String, VOICE_COMMAND, self._on_command, _qos(True, 16, True))
        node.create_subscription(String, VOICE_TEXT, self._on_text, _qos(True, 16, False))
        node.create_subscription(String, VOICE_STOP, self._on_stop, _qos(True, 1, True))
        node.create_subscription(String, CONSOLE_RESET, self._on_reset, _qos(True, 1, True))
        node.create_subscription(String, MOUNT, self._on_mount, _qos(False, 1, False))
        node.create_subscription(String, WRIST, self._on_wrist, _qos(False, 1, False))
        self._stop = ctx.request_stop

        def request_stop() -> None:
            new = self.ledger.claim_local_stop()
            self._after_stop(new)

        ctx.request_stop = request_stop
        say = ctx.say

        def said(message: str) -> None:
            self.say = message
            say(message)
            self.publish_state()

        ctx.say = said
        ctx.listeners.append(self._on_event)
        self.publish_epoch()

    def publish_epoch(self) -> None:
        with self.ledger._lock:
            epoch = self.ledger.epoch
        self._epoch_pub.publish(_epoch_msg(epoch))

    def _after_stop(self, epoch: int) -> None:
        self._epoch_pub.publish(_epoch_msg(epoch))
        self._stop()
        self.ledger.wake()
        self.publish_state()

    def publish_state(self) -> None:
        with self.ledger._lock:
            epoch, resetting = self.ledger.epoch, self.ledger.resetting
        self._state_pub.publish(_string(encode_state(ControlState(
            epoch, self.behavior, resetting, self.say, self.heard, dict(self.joints), time.time(),
            self.log_seq, self.log_kind, self.log_text,
        ))))

    def set_joints(self, joints: dict, gripper_load: float | None = None) -> None:
        """The arm's joint sample and gripper current in milliamps, sent on to the console. It does not republish the voice log."""
        self.joints = dict(joints)
        self._joints_pub.publish(_string(encode_joints(self.joints, gripper_load)))

    def _remember_line(self, event: str, **data) -> None:
        from vla_pipeline.console import panel_line

        line = panel_line(event, **data)
        if line is None:
            return
        text, kind = line
        self.log_seq += 1
        self.log_kind = kind
        self.log_text = text

    def _on_event(self, event: str, **data) -> None:
        if event == "heard":
            self.heard = str(data.get("text") or "")
        elif event == "parsed":
            self._remember_line(event, **data)
        elif event == "behavior_start":
            self.behavior = str(data.get("intent") or "idle")
            self._behavior_t0 = time.monotonic()
        elif event == "behavior_end":
            self.behavior = "idle"
            self._remember_line(event, took=time.monotonic() - self._behavior_t0, **data)
        elif event in ("stop", "ignored", "behavior_crash"):
            self._remember_line(event, **data)
        else:
            return
        self.publish_state()

    def _on_command(self, msg) -> None:
        command = self._drops.decode(decode_command, msg.data)
        if command is None or not self.ledger.accept_command(command.epoch, command.command_seq):
            return
        self.robot_voice.on_transcript(command.body())

    def _on_text(self, msg) -> None:
        """A typed line. The control process parses it, the same way stdin does."""
        text = msg.data.strip()
        if text:
            self.robot_voice.on_transcript(text)

    def _on_stop(self, msg) -> None:
        stop = self._drops.decode(decode_stop, msg.data)
        if stop is None:
            return
        new = self.ledger.claim_stop(stop.epoch)
        if new is None:
            return
        self._after_stop(new)
        self.ctx.say("Stopping.")

    def _on_mount(self, msg) -> None:
        observation = self._drops.decode(decode_mount, msg.data)
        if observation is not None:
            self.mailbox.offer(observation)

    def _on_wrist(self, msg) -> None:
        observation = self._drops.decode(decode_wrist, msg.data)
        if observation is not None:
            self.mailbox.offer(observation)

    def _on_reset(self, msg) -> None:
        reset = self._drops.decode(decode_reset, msg.data)
        if reset is None:
            return
        outcome = self.ledger.claim_reset(reset.epoch)
        if not isinstance(outcome, int):
            return
        self._after_stop(outcome)
        threading.Thread(target=self._reset_body, daemon=True, name="reset").start()

    def _reset_body(self) -> None:
        with self._reset_lock:
            self._run_reset()
            extra = self.ledger.finish_reset(run_again=True)
            if extra is not None:
                self._after_stop(extra)
                self._run_reset()
                self.ledger.finish_reset(run_again=False)
            self.publish_state()

    def _run_reset(self) -> None:
        from vla_pipeline.behaviors.manipulation import forget_scene
        from vla_pipeline.console import RESET_LINE

        ctx = self.ctx
        voice = self.robot_voice
        voice.holding.set()
        stops = ctx.stops_heard()
        try:
            voice.discard()
            voice.wait_idle(3.0)
            voice.discard()
            if isinstance(ctx.shared_state, dict):
                forget_scene(ctx.shared_state)
            if not self._restore():
                ctx.say("Reset is only available on the digital twin.")
                return
            ctx.say(RESET_LINE)
        except Exception as exc:
            logger.exception("reset")
            ctx.say(f"Reset failed: {exc}")
        finally:
            voice.discard()
            if ctx.stops_heard() == stops:
                ctx.stop_event.clear()
            voice.holding.clear()


class PerceptionLink:
    """Perception's view of the epoch, and the messages it sends."""

    def __init__(self, node) -> None:
        from std_msgs.msg import String, UInt64

        self._drops = _Drops()
        self._speech = SpeechOutbox()
        self._waiting = False
        node.create_subscription(UInt64, EPOCH, self._on_epoch, _qos(True, 1, True))
        self._command = node.create_publisher(String, VOICE_COMMAND, _qos(True, 16, True))
        self._stop = node.create_publisher(String, VOICE_STOP, _qos(True, 1, True))
        self._mount = node.create_publisher(String, MOUNT, _qos(False, 1, False))
        self._wrist = node.create_publisher(String, WRIST, _qos(False, 1, False))
        self._status = node.create_publisher(String, STATUS, _qos(True, 1, True))
        from sensor_msgs.msg import Image

        image = _qos(False, 1, False)
        self._mount_image = node.create_publisher(Image, OVERLAY_MOUNT, image)
        self._hands_image = node.create_publisher(Image, OVERLAY_HANDS, image)
        self._wrist_image = node.create_publisher(Image, OVERLAY_WRIST, image)

    def _on_epoch(self, msg) -> None:
        self._speech.set_epoch(int(msg.data))
        self._flush()

    def offer_stop(self) -> None:
        """Keep this stop until an epoch exists, then publish it ahead of any command."""
        self._speech.offer_stop()
        self._flush()

    def offer_command(self, command: str, obj: str, heard: str) -> None:
        """Keep the latest command. It is published only after an outstanding stop."""
        self._speech.offer_command(command, obj, heard)
        self._flush()

    def _flush(self) -> None:
        if self._speech.waiting() and not self._waiting:
            self._waiting = True
            logger.info("voice is holding the latest command until the control epoch arrives")
        while True:
            item = self._speech.take()
            if item is None:
                return
            if item[0] == "stop":
                self._stop.publish(_string(encode_stop(VoiceStop(item[1]))))
                continue
            _, epoch, seq, command, obj, heard = item
            self._command.publish(_string(encode_command(
                VoiceCommand(epoch, seq, command, obj, heard),
            )))

    def publish_mount(self, observation) -> None:
        self._mount.publish(_string(encode_mount(observation)))

    def publish_wrist(self, observation) -> None:
        self._wrist.publish(_string(encode_wrist(observation)))

    def publish_status(self, status: PerceptionStatus) -> None:
        self._status.publish(_string(encode_status(status)))

    def publish_overlay(self, topic: str, frame) -> None:
        from vla_pipeline.vision.camera import to_image_msg

        publisher = {
            OVERLAY_MOUNT: self._mount_image,
            OVERLAY_HANDS: self._hands_image,
            OVERLAY_WRIST: self._wrist_image,
        }[topic]
        publisher.publish(to_image_msg(frame))


class ConsoleLink:
    """The console paints the latest state and publishes stop and reset."""

    def __init__(self, node, console) -> None:
        from sensor_msgs.msg import Image
        from std_msgs.msg import String, UInt64

        self.console = console
        self._drops = _Drops()
        self._clicks = ClickOutbox()
        self._stop = node.create_publisher(String, VOICE_STOP, _qos(True, 1, True))
        self._reset = node.create_publisher(String, CONSOLE_RESET, _qos(True, 1, True))
        self._text = node.create_publisher(String, VOICE_TEXT, _qos(True, 16, False))
        node.create_subscription(UInt64, EPOCH, self._on_epoch, _qos(True, 1, True))
        node.create_subscription(String, STATE, self._on_state, _qos(True, 1, True))
        node.create_subscription(String, JOINTS, self._on_joints, _qos(True, 1, True))
        node.create_subscription(String, STATUS, self._on_status, _qos(True, 1, True))
        image = _qos(False, 1, False)
        node.create_subscription(Image, OVERLAY_MOUNT, lambda msg: self._on_image("mount", msg), image)
        node.create_subscription(Image, OVERLAY_HANDS, lambda msg: self._on_image("hands", msg), image)
        node.create_subscription(Image, OVERLAY_WRIST, lambda msg: self._on_image("wrist", msg), image)

    def publish_stop(self) -> None:
        """Remember one stop and send it once this console has an epoch."""
        self._clicks.offer("stop")
        self._flush()

    def publish_reset(self) -> None:
        """Remember one reset. It replaces an unsent stop, because a reset stops too."""
        self._clicks.offer("reset")
        self._flush()

    def publish_text(self, text: str) -> None:
        """A typed command. The control process parses it with the keyword parser."""
        self._text.publish(_string(text))

    def _on_epoch(self, msg) -> None:
        self._clicks.set_epoch(int(msg.data))
        self._flush()

    def _flush(self) -> None:
        item = self._clicks.take()
        if item is None:
            return
        action, epoch = item
        if action == "reset":
            self._reset.publish(_string(encode_reset(ConsoleReset(epoch))))
        else:
            self._stop.publish(_string(encode_stop(VoiceStop(epoch))))

    def _on_state(self, msg) -> None:
        state = self._drops.decode(decode_state, msg.data)
        if state is None:
            return
        self._clicks.set_epoch(state.epoch)
        self.console.apply_state(state)
        self._flush()

    def _on_joints(self, msg) -> None:
        sample = self._drops.decode(decode_joints, msg.data)
        if sample is not None:
            joints, load = sample
            self.console.apply_joints(joints, load)

    def _on_status(self, msg) -> None:
        status = self._drops.decode(decode_status, msg.data)
        if status is None:
            return
        self.console.on_level(status.level, status.speaking)
        self.console.note_asr(status.busy)

    def _on_image(self, name: str, msg) -> None:
        from vla_pipeline.vision.camera import from_image_msg

        frame = from_image_msg(msg)
        if frame is not None:
            self.console.set_tile(name, frame)
