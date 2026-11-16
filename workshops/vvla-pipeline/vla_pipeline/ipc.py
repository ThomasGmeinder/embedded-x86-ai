# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""The one contract between the twin, the control process, and perception.

Control is the only writer of the epoch. A command, a stop, or a reset whose
epoch is not the current one is a no-op, so a message that arrives late is
discarded. A command sequence is accepted once. An observation whose sequence
went backwards, or whose stamp went backwards, is discarded.

Nothing here talks to ROS. The wire encodes these values and publishes them.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass

EPOCH = "/control/epoch"
STATE = "/control/state"
JOINTS = "/control/joints"
VOICE_COMMAND = "/voice/command"
VOICE_STOP = "/voice/stop"
CONSOLE_RESET = "/console/reset"
MOUNT = "/perception/mount"
WRIST = "/perception/wrist"
STATUS = "/perception/status"
OVERLAY_MOUNT = "/perception/overlay/mount"
OVERLAY_HANDS = "/perception/overlay/hands"
OVERLAY_WRIST = "/perception/overlay/wrist"

_COMMAND_FIELDS = ("epoch", "command_seq", "command", "object", "heard")
_STOP_FIELDS = ("epoch",)
_RESET_FIELDS = ("epoch",)
_STATE_FIELDS = (
    "epoch", "behavior", "resetting", "say", "heard", "joints", "stamp",
    "log_seq", "log_kind", "log_text",
)
_HAND_FIELDS = (
    "pinch", "thumb_px", "index_px", "handedness", "hand_size", "roll",
    "roll_ti", "roll_ref", "roll_ti_weight", "wrist_px",
)
_MOUNT_FIELDS = ("role", "stamp", "seq", "width", "height", "kpts", "hand")
_DETECTION_FIELDS = ("name", "score", "box", "axis")
_CAMERA_FIELDS = ("width", "height", "k", "d", "distortion_model")
_WRIST_FIELDS = ("role", "stamp", "seq", "width", "height", "camera", "detections")
_STATUS_FIELDS = ("voice", "level", "speaking", "busy")


class ContractError(ValueError):
    """A payload is not a message this contract accepts."""


def _object(raw: str, fields: tuple[str, ...]) -> dict:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError("not JSON") from exc
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ContractError("unexpected fields")
    return data


def _epoch(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("epoch")
    return value


def _seq(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ContractError("sequence")
    return value


def _stamp(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("stamp")
    return float(value)


def _text(value) -> str:
    if not isinstance(value, str):
        raise ContractError("text")
    return value


def _pair(value) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ContractError("point")
    return (float(value[0]), float(value[1]))


def dumps(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


@dataclass(frozen=True)
class VoiceCommand:
    epoch: int
    command_seq: int
    command: str
    object: str
    heard: str

    def body(self) -> str:
        """The command JSON ``UserRobotControl`` already accepts."""
        return dumps({"command": self.command, "object": self.object, "heard": self.heard})


@dataclass(frozen=True)
class VoiceStop:
    epoch: int


@dataclass(frozen=True)
class ConsoleReset:
    epoch: int


@dataclass(frozen=True)
class ControlState:
    epoch: int
    behavior: str
    resetting: bool
    say: str
    heard: str
    joints: dict
    stamp: float
    log_seq: int = 0
    log_kind: str = ""
    log_text: str = ""


@dataclass(frozen=True)
class HandSignals:
    pinch: float
    thumb_px: tuple[float, float]
    index_px: tuple[float, float]
    handedness: str
    hand_size: float
    roll: float
    roll_ti: float
    roll_ref: float
    roll_ti_weight: float
    wrist_px: tuple[float, float]


@dataclass(frozen=True)
class MountObservation:
    role: str
    stamp: float
    seq: int
    width: int
    height: int
    kpts: tuple
    hand: HandSignals | None


@dataclass(frozen=True)
class DetectionSignals:
    name: str
    score: float
    box: tuple
    axis: tuple | None


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: tuple
    d: tuple
    distortion_model: str


@dataclass(frozen=True)
class WristObservation:
    role: str
    stamp: float
    seq: int
    width: int
    height: int
    camera: CameraModel | None
    detections: tuple


@dataclass(frozen=True)
class PerceptionStatus:
    voice: str
    level: float
    speaking: bool
    busy: bool


def encode_command(message: VoiceCommand) -> str:
    return dumps({
        "epoch": message.epoch,
        "command_seq": message.command_seq,
        "command": message.command,
        "object": message.object,
        "heard": message.heard,
    })


def decode_command(raw: str) -> VoiceCommand:
    data = _object(raw, _COMMAND_FIELDS)
    return VoiceCommand(
        _epoch(data["epoch"]), _seq(data["command_seq"]), _text(data["command"]),
        _text(data["object"]), _text(data["heard"]),
    )


def encode_stop(message: VoiceStop) -> str:
    return dumps({"epoch": message.epoch})


def decode_stop(raw: str) -> VoiceStop:
    return VoiceStop(_epoch(_object(raw, _STOP_FIELDS)["epoch"]))


def encode_reset(message: ConsoleReset) -> str:
    return dumps({"epoch": message.epoch})


def decode_reset(raw: str) -> ConsoleReset:
    return ConsoleReset(_epoch(_object(raw, _RESET_FIELDS)["epoch"]))


def encode_joints(joints: dict, gripper_load: float | None = None) -> str:
    if not isinstance(joints, dict):
        raise ContractError("joints")
    if gripper_load is not None and (isinstance(gripper_load, bool) or not isinstance(gripper_load, (int, float))):
        raise ContractError("gripper_load")
    return dumps({
        "joints": {str(k): float(v) for k, v in joints.items()},
        "gripper_load": None if gripper_load is None else float(gripper_load),
    })


def decode_joints(raw: str) -> tuple[dict, float | None]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError("not JSON") from exc
    if not isinstance(data, dict):
        raise ContractError("joints")
    if "joints" in data:
        joints = data["joints"]
        load = data.get("gripper_load")
    else:
        joints, load = data, None
    if not isinstance(joints, dict) or any(
        not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float))
        for k, v in joints.items()
    ):
        raise ContractError("joints")
    if load is not None and (isinstance(load, bool) or not isinstance(load, (int, float))):
        raise ContractError("gripper_load")
    return {k: float(v) for k, v in joints.items()}, None if load is None else float(load)


def encode_state(message: ControlState) -> str:
    return dumps({
        "epoch": message.epoch,
        "behavior": message.behavior,
        "resetting": message.resetting,
        "say": message.say,
        "heard": message.heard,
        "joints": message.joints,
        "stamp": message.stamp,
        "log_seq": message.log_seq,
        "log_kind": message.log_kind,
        "log_text": message.log_text,
    })


def decode_state(raw: str) -> ControlState:
    data = _object(raw, _STATE_FIELDS)
    joints = data["joints"]
    if not isinstance(joints, dict) or any(not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, (int, float)) for k, v in joints.items()):
        raise ContractError("joints")
    if not isinstance(data["resetting"], bool):
        raise ContractError("resetting")
    return ControlState(
        _epoch(data["epoch"]), _text(data["behavior"]), data["resetting"],
        _text(data["say"]), _text(data["heard"]), {k: float(v) for k, v in joints.items()},
        _stamp(data["stamp"]), int(data["log_seq"]), _text(data["log_kind"]), _text(data["log_text"]),
    )


def _hand(data) -> HandSignals | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != set(_HAND_FIELDS):
        raise ContractError("hand")
    return HandSignals(
        float(data["pinch"]), _pair(data["thumb_px"]), _pair(data["index_px"]),
        _text(data["handedness"]), float(data["hand_size"]), float(data["roll"]),
        float(data["roll_ti"]), float(data["roll_ref"]), float(data["roll_ti_weight"]),
        _pair(data["wrist_px"]),
    )


def encode_mount(message: MountObservation) -> str:
    hand = None if message.hand is None else {
        "pinch": message.hand.pinch,
        "thumb_px": list(message.hand.thumb_px),
        "index_px": list(message.hand.index_px),
        "handedness": message.hand.handedness,
        "hand_size": message.hand.hand_size,
        "roll": message.hand.roll,
        "roll_ti": message.hand.roll_ti,
        "roll_ref": message.hand.roll_ref,
        "roll_ti_weight": message.hand.roll_ti_weight,
        "wrist_px": list(message.hand.wrist_px),
    }
    return dumps({
        "role": message.role,
        "stamp": message.stamp,
        "seq": message.seq,
        "width": message.width,
        "height": message.height,
        "kpts": message.kpts,
        "hand": hand,
    })


def decode_mount(raw: str) -> MountObservation:
    data = _object(raw, _MOUNT_FIELDS)
    if data["role"] != "mount":
        raise ContractError("role")
    kpts = data["kpts"]
    if not isinstance(kpts, list):
        raise ContractError("kpts")
    return MountObservation(
        "mount", _stamp(data["stamp"]), _seq(data["seq"]),
        int(data["width"]), int(data["height"]), tuple(tuple(point) for point in kpts),
        _hand(data["hand"]),
    )


def _camera(data) -> CameraModel | None:
    if data is None:
        return None
    if not isinstance(data, dict) or set(data) != set(_CAMERA_FIELDS):
        raise ContractError("camera")
    if not isinstance(data["k"], list) or len(data["k"]) != 9:
        raise ContractError("camera")
    if not isinstance(data["d"], list):
        raise ContractError("camera")
    return CameraModel(
        int(data["width"]), int(data["height"]), tuple(float(v) for v in data["k"]),
        tuple(float(v) for v in data["d"]), _text(data["distortion_model"]),
    )


def _detection(data) -> DetectionSignals:
    if not isinstance(data, dict) or set(data) != set(_DETECTION_FIELDS):
        raise ContractError("detection")
    box = data["box"]
    if not isinstance(box, list) or len(box) != 4:
        raise ContractError("box")
    axis = data["axis"]
    if axis is not None:
        if not isinstance(axis, list) or len(axis) != 2:
            raise ContractError("axis")
        axis = (_pair(axis[0]), _pair(axis[1]))
    return DetectionSignals(_text(data["name"]), float(data["score"]), tuple(float(v) for v in box), axis)


def encode_wrist(message: WristObservation) -> str:
    camera = None if message.camera is None else {
        "width": message.camera.width,
        "height": message.camera.height,
        "k": list(message.camera.k),
        "d": list(message.camera.d),
        "distortion_model": message.camera.distortion_model,
    }
    detections = [
        {
            "name": item.name,
            "score": item.score,
            "box": list(item.box),
            "axis": None if item.axis is None else [list(item.axis[0]), list(item.axis[1])],
        }
        for item in message.detections
    ]
    return dumps({
        "role": message.role,
        "stamp": message.stamp,
        "seq": message.seq,
        "width": message.width,
        "height": message.height,
        "camera": camera,
        "detections": detections,
    })


def decode_wrist(raw: str) -> WristObservation:
    data = _object(raw, _WRIST_FIELDS)
    if data["role"] != "arm":
        raise ContractError("role")
    if not isinstance(data["detections"], list):
        raise ContractError("detections")
    return WristObservation(
        "arm", _stamp(data["stamp"]), _seq(data["seq"]),
        int(data["width"]), int(data["height"]), _camera(data["camera"]),
        tuple(_detection(item) for item in data["detections"]),
    )


def encode_status(message: PerceptionStatus) -> str:
    if message.voice not in ("up", "down"):
        raise ContractError("voice")
    return dumps({
        "voice": message.voice,
        "level": message.level,
        "speaking": message.speaking,
        "busy": message.busy,
    })


def decode_status(raw: str) -> PerceptionStatus:
    data = _object(raw, _STATUS_FIELDS)
    if data["voice"] not in ("up", "down"):
        raise ContractError("voice")
    if not isinstance(data["speaking"], bool) or not isinstance(data["busy"], bool):
        raise ContractError("status")
    level = data["level"]
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise ContractError("level")
    return PerceptionStatus(data["voice"], float(level), data["speaking"], data["busy"])


class SpeechOutbox:
    """The one unsent stop and the one unsent command, on the perception side.

    A newer command replaces the unsent one. A stop is kept beside it and is
    sent first. The command is sent only with an epoch newer than that stop,
    so the two cannot be reordered into "command survived the stop."
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0
        self._stop = False
        self._command: tuple[str, str, str] | None = None
        self._stop_at = 0
        self._seq = 0

    def set_epoch(self, epoch: int) -> None:
        with self._lock:
            self._epoch = int(epoch)

    def offer_stop(self) -> None:
        with self._lock:
            self._stop = True

    def offer_command(self, command: str, obj: str, heard: str) -> None:
        with self._lock:
            self._command = (command, obj, heard)

    def take(self) -> tuple | None:
        """One message that can be published now, or None.

        ``("stop", epoch)`` or ``("command", epoch, seq, command, object, heard)``.
        """
        with self._lock:
            if self._epoch < 1:
                return None
            if self._stop and self._stop_at != self._epoch:
                self._stop = False
                self._stop_at = self._epoch
                return ("stop", self._epoch)
            if self._command is not None and self._epoch > self._stop_at:
                command, obj, heard = self._command
                self._command = None
                self._seq += 1
                return ("command", self._epoch, self._seq, command, obj, heard)
            return None

    def waiting(self) -> bool:
        with self._lock:
            return self._epoch < 1 and (self._stop or self._command is not None)


class ClickOutbox:
    """The one unsent console click: a reset replaces a stop, and it is sent once per epoch."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._epoch = 0
        self._action: str | None = None
        self._sent_at = 0

    def set_epoch(self, epoch: int) -> None:
        with self._lock:
            self._epoch = int(epoch)

    def offer(self, action: str) -> None:
        if action not in ("stop", "reset"):
            raise ContractError("click")
        with self._lock:
            if action == "reset" or self._action != "reset":
                self._action = action

    def take(self) -> tuple[str, int] | None:
        with self._lock:
            if self._epoch < 1 or self._action is None or self._sent_at == self._epoch:
                return None
            action = self._action
            self._action = None
            self._sent_at = self._epoch
            return action, self._epoch


class Ledger:
    """Control's epoch, and the checks every incoming message has to pass.

    The lock is the only lock these decisions take. ``wake`` is notified after
    a stop or a new observation so a behavior waiting on a frame returns.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self.epoch = 1
        self.resetting = False
        self._reset_again = False
        self._command_seq = 0
        self._obs_seq: dict[str, int] = {}
        self._obs_stamp: dict[str, float] = {}

    def claim_local_stop(self) -> int:
        """Bump the epoch for a stop that started in this process. The new epoch."""
        with self._lock:
            self.epoch += 1
            return self.epoch

    def claim_stop(self, epoch: int) -> int | None:
        """Bump the epoch when ``epoch`` is current. None when it is stale."""
        with self._lock:
            if epoch != self.epoch:
                return None
            self.epoch += 1
            return self.epoch

    def claim_reset(self, epoch: int) -> int | str | None:
        """Start a reset, remember one more, or ignore a stale click.

        An int is the new epoch and this caller runs the reset. ``"again"``
        means a reset is already running and exactly one more will follow.
        None means the epoch did not match.
        """
        with self._lock:
            if epoch != self.epoch:
                return None
            if self.resetting:
                self._reset_again = True
                return "again"
            self.epoch += 1
            self.resetting = True
            return self.epoch

    def finish_reset(self, *, run_again: bool) -> int | None:
        """End the reset. The new epoch when one coalesced click still has to run."""
        with self._lock:
            self.resetting = False
            again = self._reset_again
            self._reset_again = False
            if again and run_again:
                self.epoch += 1
                self.resetting = True
                return self.epoch
            return None

    def accept_command(self, epoch: int, command_seq: int) -> bool:
        """True the first time this sequence arrives during the current epoch."""
        with self._lock:
            if epoch != self.epoch or command_seq <= self._command_seq:
                return False
            self._command_seq = command_seq
            return True

    def _accept_observation(self, role: str, seq: int, stamp: float) -> bool:
        """Caller holds the lock. Does not notify: the mailbox stores the message first."""
        prev_seq = self._obs_seq.get(role)
        prev_stamp = self._obs_stamp.get(role)
        if prev_seq is not None and seq <= prev_seq:
            return False
        if prev_stamp is not None and stamp < prev_stamp:
            return False
        self._obs_seq[role] = seq
        self._obs_stamp[role] = stamp
        return True

    def accept_observation(self, role: str, seq: int, stamp: float) -> bool:
        """True when ``seq`` and ``stamp`` are both newer than the last accepted one.

        A newer sequence with an older stamp is rejected and does not advance
        the sequence, so a later valid frame is still accepted.
        """
        with self._lock:
            return self._accept_observation(role, seq, stamp)

    def wake(self) -> None:
        """Return a behavior that is waiting on the next observation."""
        with self._lock:
            self._cond.notify_all()


class Mailbox:
    """The latest accepted observation of each camera. Waiters block on the ledger."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self._latest: dict[str, MountObservation | WristObservation] = {}
        self._closed = False

    def offer(self, observation) -> bool:
        with self.ledger._lock:
            if not self.ledger._accept_observation(observation.role, observation.seq, observation.stamp):
                return False
            self._latest[observation.role] = observation
            self.ledger._cond.notify_all()
            return True

    def close(self) -> None:
        with self.ledger._lock:
            self._closed = True
            self.ledger._cond.notify_all()

    def _wait(self, ready, timeout: float | None):
        with self.ledger._lock:
            if not self.ledger._cond.wait_for(ready, timeout):
                return None
            return True

    def wait_new(self, role: str, after_seq: int, timeout: float, stop_event=None):
        """The next observation with a higher sequence, or None when ``timeout`` passes or the stop is set."""
        def ready() -> bool:
            if self._closed or (stop_event is not None and stop_event.is_set()):
                return True
            current = self._latest.get(role)
            return current is not None and current.seq > after_seq

        if self._wait(ready, timeout) is None:
            return None
        if stop_event is not None and stop_event.is_set():
            return None
        with self.ledger._lock:
            current = self._latest.get(role)
        if current is None or current.seq <= after_seq:
            return None
        return current

    def wait_after(self, role: str, stamp: float, timeout: float, stop_event=None):
        """The next observation whose stamp is strictly after ``stamp``."""
        def ready() -> bool:
            if self._closed or (stop_event is not None and stop_event.is_set()):
                return True
            current = self._latest.get(role)
            return current is not None and current.stamp > stamp

        if self._wait(ready, timeout) is None:
            return None
        if stop_event is not None and stop_event.is_set():
            return None
        with self.ledger._lock:
            current = self._latest.get(role)
        if current is None or current.stamp <= stamp:
            return None
        return current
