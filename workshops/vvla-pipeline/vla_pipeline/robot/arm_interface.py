# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""SO-101 arm access layer: one ``ArmClient`` API, three transports.

Behaviors never touch hardware directly - they talk to an :class:`ArmClient`:

- :class:`DirectArm`   - owns the Feetech serial bus via LeRobot's
  ``SO101Follower`` (same path as ``amd_accel_native_node``). Used for
  component tests and ``robot.use_ros2: false``.
- :class:`DryRunArm`   - no hardware; integrates commands so behaviors can be
  tested end-to-end on any machine.
- ``Ros2ArmClient`` (in :mod:`vla_pipeline.robot.robot_node`) - publishes
  joint commands to the ROS 2 node that owns the bus.

Joint convention (degrees, LeRobot SO-101 calibration space):
``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper``.

New in this revision
--------------------
``get_gripper_load()`` - best-effort effort/current feedback for the gripper
servo, used by the grip behavior for contact-based closing. On the Feetech
STS3215 bus it reads ``Present_Current`` (falling back to ``Present_Load``);
capability is **detected at runtime** and the method returns ``None`` on
transports/firmware that can't provide it, in which case grip falls back to
position-stall detection.

Independent test
----------------
    python -m vla_pipeline.robot.arm_interface --dry-run
    python -m vla_pipeline.robot.arm_interface --port /dev/ttyACM0 --id my_awesome_follower_arm
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

logger = logging.getLogger(__name__)

MOTOR_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# LeRobot observation/action keys are "<motor>.pos". Precompute the pairing once
# so the per-tick read loop in DirectArm.get_joints doesn't rebuild these strings
# on every control tick.
_MOTOR_POS_KEYS = [(m, f"{m}.pos") for m in MOTOR_NAMES]

# Gripper floor (calibration space). Commanding the gripper below its physical
# closed point drives it into the mechanical stop and stalls it there - a
# leading suspect for the gripper servo (id=6, last on the chain) tripping off
# the bus. 43 is the jaws' measured physical closed point on the rig (the same
# value the workshop uses - see workshop/common/motion.py); keep every
# commanded gripper position at or above it so the servo never sits jammed shut.
GRIPPER_CLOSED_FLOOR = 43.0

# Safe rest pose (degrees) - identical to amd_accel_native_node.REST_POSE.
REST_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -96.5,
    "elbow_flex": 95.9,
    "wrist_flex": 57.9,
    "wrist_roll": 86.8,
    "gripper": GRIPPER_CLOSED_FLOOR,  # was 0.3 (jammed shut) - rest a few % open
}

# Software joint limits (degrees). These act as a hard backstop that every
# command is clamped to (see clamp_joints, called inside go_to). They are set
# WIDE ENOUGH to encompass the LeRobot calibrated travel so that calibration -
# not this backstop - governs normal range of motion. Callers that derive a
# sweep range from calibration intersect it with these limits, so if a limit is
# tighter than the calibration it silently caps the sweep (this previously held
# elbow_flex to -10° instead of its calibrated ~-93°).
#
# Body-joint values below carry a small margin over the calibrated half-spans
# measured on the reference arm (shoulder_pan ~±111°, shoulder_lift ~±104°,
# elbow_flex ~±94°) so a slightly different per-machine calibration still fits.
# wrist_roll is left conservative on purpose: LeRobot does NOT range-sweep it
# (its calibration spans the full encoder range, ~±180°), so its real safe
# travel is unknown and the backstop must stay tight.
JOINT_LIMITS = {
    "shoulder_pan": (-115.0, 115.0),
    "shoulder_lift": (-120.0, 110.0),
    "elbow_flex": (-100.0, 120.0),
    "wrist_flex": (-110.0, 90.0),
    "wrist_roll": (-100.0, 100.0),  # NOT range-swept by LeRobot - keep tight
    # Calibrated jaw span, matching the workshop rig: 43 = physically closed
    # (below jams the jaws into the stop), 95 = fully open. NOT a scaled angle.
    "gripper": (GRIPPER_CLOSED_FLOOR, 95.0),
}


def clamp_joints(joints: dict[str, float]) -> dict[str, float]:
    """Clamp each joint's value to its JOINT_LIMITS range."""
    out = {}
    for name, value in joints.items():
        lo, hi = JOINT_LIMITS.get(name, (-180.0, 180.0))
        out[name] = float(min(max(value, lo), hi))
    return out


class ArmClient(ABC):
    """Transport-agnostic arm API used by every behavior."""

    # Set True by transports whose relative-target clamp is disabled
    # (``max_relative_target: null``). In that mode go_to() sends the goal in a
    # single write and lets the servo's onboard controller drive there (see
    # move_to), which is smooth; with the clamp on, go_to() streams an
    # interpolated trajectory because the clamp throttles each command to a few
    # degrees.
    _direct_goal: bool = False

    @abstractmethod
    def send_joints(self, joints: dict[str, float]) -> None:
        """Command absolute joint targets (degrees); partial dicts allowed."""

    @abstractmethod
    def get_joints(self) -> dict[str, float]:
        """Read current joint positions (degrees)."""

    def get_gripper_load(self) -> float | None:
        """Gripper servo effort/current (arbitrary positive units), if readable.

        Returns ``None`` when this transport can't provide effort feedback -
        callers must treat that as "use position-stall detection instead".
        """
        return None

    def go_to(
        self,
        target: dict[str, float],
        duration_s: float = 2.0,
        fps: float = 30.0,
        stop_check=None,
        max_step_deg: float = 0.0,
        resend_s: float | None = 0.5,
    ) -> None:
        """Linearly interpolate from the current pose to ``target``.

        ``stop_check`` (optional callable → bool) aborts the motion mid-way -
        used by the safety stop so long moves react immediately.

        ``max_step_deg`` (default 0 = disabled) caps the per-tick command so it
        never jumps more than this many degrees from the *actual* servo position.
        When a gravity-loaded joint lags the planned trajectory, the command
        waits for it instead of building up a positional deficit that triggers
        the LeRobot ``max_relative_target`` slew limiter on every subsequent
        tick (the "clamping cascade" that causes twitchy motion).  A good value
        is slightly below ``max_relative_target`` (e.g. 6.0 when the limiter
        is 8.0) so the hardware limiter never fires under normal operation.

        When the clamp is disabled (``_direct_goal``), there is nothing to stream
        around: the goal is sent in one write and the servo interpolates in
        firmware, so this delegates to move_to(). ``duration_s``/``fps``/
        ``max_step_deg`` no longer apply; ``stop_check`` still does.
        """
        if self._direct_goal:
            return self.move_to(target, stop_check=stop_check, resend_s=resend_s)

        start = self.get_joints()
        target = clamp_joints({**start, **target})
        steps = max(int(duration_s * fps), 1)
        # Precompute per-joint travel and the sleep interval once: the per-tick
        # pose is start + a*delta, so there's no reason to recompute the
        # subtraction (or 1/fps) on every tick.
        deltas = {k: target[k] - start[k] for k in target}
        interval = 1.0 / fps
        for i in range(1, steps + 1):
            if stop_check and stop_check():
                return
            a = i / steps
            # Desired pose at this fraction of the trajectory.
            desired = {k: start[k] + a * deltas[k] for k in target}
            if max_step_deg > 0:
                # Re-read actual position and cap each joint's command so we
                # never ask the servo to jump farther than max_step_deg from
                # where it physically is RIGHT NOW.
                actual = self.get_joints()
                pose = {}
                for k in desired:
                    goal = desired[k]
                    cur = actual.get(k, goal)
                    delta = goal - cur
                    if abs(delta) > max_step_deg:
                        goal = cur + max_step_deg * (1.0 if delta > 0 else -1.0)
                    pose[k] = goal
            else:
                pose = desired
            self.send_joints(pose)
            time.sleep(interval)

    def move_to(
        self,
        target: dict[str, float],
        timeout_s: float = 6.0,
        tolerance_deg: float = 2.5,
        poll_hz: float = 20.0,
        progress_eps_deg: float = 1.0,
        settle_dwell_s: float = 0.35,
        resend_s: float | None = 0.5,
        gripper_settle_s: float = 2.0,
        stop_check=None,
    ) -> None:
        """Command an absolute goal and let the servos drive there themselves.

        Unlike go_to(), this does NOT stream interpolated setpoints - it sends
        the target once and waits until the arm has SETTLED. The servo's onboard
        controller produces the motion, which is smooth and continuous rather
        than the stepped, host-interpolated trajectory.

        "Settled" is judged on the BODY joints only (the gripper is excluded -
        see below) and means the move is finished, which is *either* of:
          • every body joint within ``tolerance_deg`` of the goal, or
          • no body joint has gotten ``progress_eps_deg`` closer to its goal for
            ``settle_dwell_s`` - i.e. the arm stopped making progress and is now
            holding or stalled (a hard range limit or a servo's steady-state
            error that keeps the reported value off the target).

        Progress (not instantaneous motion) is tracked on purpose: a servo
        straining at a stop reports a jittery position that never looks perfectly
        still, so a velocity-style "stopped moving" check (or an exact tolerance
        match) makes such a joint hunt for the whole ``timeout_s``, straining the
        servo - and that sustained strain is what trips a joint (the gravity-
        loaded elbow) into thermal/overcurrent protection.

        The gripper is excluded from settling because it runs on 0-100
        normalisation and its position readback can sit far from the commanded
        value even after it has physically moved; polling it never converges. In
        a mixed move it travels alongside the body joints; a gripper-only move is
        given a fixed ``gripper_settle_s`` window to travel instead of being
        polled. Grasp confirmation is ``get_gripper_load``, not position.

        Needs the relative-target clamp OFF (``max_relative_target: null``) so
        the whole goal reaches the servo in one write. ``target`` is range-clamped
        to JOINT_LIMITS. ``stop_check`` aborts and halts the arm where it is.
        """
        target = clamp_joints(target)
        self.send_joints(target)
        # Settle on the body joints only. The gripper runs on 0-100 normalisation
        # and its position readback can sit far from the commanded value even when
        # it has physically moved, so polling it never converges. It's an
        # open/close actuator, not a positioned joint, so it's commanded without
        # waiting on its position (grasp confirmation is get_gripper_load(), not
        # position). In a mixed move it travels alongside the body joints; a
        # GRIPPER-ONLY move has no body joint to gate on, so give it a fixed
        # ``gripper_settle_s`` window to finish travelling - without it the move
        # returns instantly and the next command cuts the gripper off mid-stroke.
        wait = [k for k in target if k != _GRIPPER_NAME]
        if not wait:
            if _GRIPPER_NAME in target:
                time.sleep(gripper_settle_s)
            return
        deadline = time.time() + timeout_s
        last_send = time.time()
        interval = 1.0 / poll_hz

        def _dist(c: dict[str, float]) -> float:
            """Max remaining distance to target across the joints being waited on."""
            return max(
                (abs(c.get(k, target[k]) - target[k]) for k in wait), default=0.0
            )

        ref_dist = _dist(self.get_joints())  # distance at the start of this window
        ref_time = time.time()
        started = False  # has it closed in meaningfully yet?
        while time.time() < deadline:
            if stop_check and stop_check():
                try:  # halt at the current position instead of finishing the move
                    self.send_joints(self.get_joints())
                except Exception:
                    pass
                return
            time.sleep(interval)
            dist = _dist(self.get_joints())

            # (1) Reached the goal - done. (Empty `wait`, i.e. a gripper-only
            # move, gives dist 0.0 and returns on the first poll.)
            if dist <= tolerance_deg:
                return

            # (2) Still closing in fast: it gained more than progress_eps_deg
            # toward the goal within this window, so reset the window and keep
            # waiting.
            if ref_dist - dist > progress_eps_deg:
                started = True
                ref_dist = dist
                ref_time = time.time()
                continue

            # (3) A whole settle_dwell_s window passed WITHOUT that much progress.
            # If it had been moving, it's now settled - as close as it can get,
            # whether that's the goal, a range limit, or a servo straining at its
            # stop (a slow creep counts as settled, so the arm doesn't sit there
            # straining for the full timeout). If it never started, the command
            # may have dropped, so re-send and give it a fresh window. (Re-sending
            # only before it moves is deliberate: re-writing the goal mid-move
            # restarts the servo's onboard profile. The ROS server de-dups too.)
            if time.time() - ref_time >= settle_dwell_s:
                if started:
                    return
                # resend_s=None disables re-sending entirely: the goal was sent
                # once and we just wait it out. Re-writing the goal restarts the
                # servo's onboard profile, which on a stop-triggered park looked
                # like the arm repeatedly lurching toward rest - so callers that
                # want a single clean move (go_to_rest on stop) pass None.
                if resend_s is not None and time.time() - last_send >= resend_s:
                    self.send_joints(target)
                    last_send = time.time()
                ref_dist = dist
                ref_time = time.time()
        logger.warning(
            "move_to: %s didn't settle within %.1fs (closest approach %.1f° "
            "from goal - joint may be stuck or a target unreachable)",
            {k: round(v, 1) for k, v in target.items()},
            timeout_s,
            _dist(self.get_joints()),
        )

    def stream_to(
        self,
        target: dict[str, float],
        duration_s: float,
        fps: float = 60.0,
        stop_check=None,
    ) -> None:
        """Move the joints in ``target`` to their goals in LOCKSTEP over
        ``duration_s`` by streaming a host-interpolated trajectory.

        Every joint covers its whole travel over the same duration, so they all
        arrive together and each stays at a fixed fraction of the others' progress
        throughout - e.g. a joint travelling twice as far moves twice as fast and
        is always at 2× another joint's progress. Use this (rather than go_to /
        move_to) only when joints must stay coordinated through the move: the
        servos follow host setpoints here instead of running their own smooth
        point-to-point profile, so keep each joint's speed (its travel ÷
        ``duration_s``) within the servo's comfortable range or it will lag and
        break the lockstep. ``stop_check`` aborts the trajectory where it is.
        """
        start = self.get_joints()
        goal = clamp_joints(target)
        base = {k: start.get(k, goal[k]) for k in goal}
        deltas = {k: goal[k] - base[k] for k in goal}
        steps = max(int(duration_s * fps), 1)
        interval = 1.0 / fps
        for i in range(1, steps + 1):
            if stop_check and stop_check():
                return
            a = i / steps  # 0 → 1, same for every joint
            self.send_joints({k: base[k] + a * deltas[k] for k in goal})
            time.sleep(interval)

    def go_to_rest(
        self, duration_s: float = 3.0, stop_check=None, resend_s: float | None = 0.5
    ) -> None:
        """Bring the whole arm to REST_POSE. Routes through go_to (→ move_to in
        direct-goal mode), so it inherits the same smooth, settle-aware motion.

        ``resend_s=None`` makes it a single, smooth, no-resend move - used when
        parking on a *stop* so the arm travels home once instead of repeatedly
        lurching (each resend restarts the servo's onboard profile)."""
        self.go_to(
            REST_POSE, duration_s=duration_s, stop_check=stop_check, resend_s=resend_s
        )

    def close(self) -> None:  # noqa: B027 - optional override
        """No-op; override in transports that need to release resources."""


class DryRunArm(ArmClient):
    """No-hardware arm: commands are integrated and echoed."""

    def __init__(self, verbose: bool = True):
        self._joints = dict(REST_POSE)
        self._verbose = verbose

    def send_joints(self, joints: dict[str, float]) -> None:
        """Update the integrated joint state and print it if verbose."""
        self._joints.update(clamp_joints(joints))
        if self._verbose:
            pretty = " ".join(f"{k}={v:6.1f}" for k, v in self._joints.items())
            print(f"\r[dry-run] {pretty}", end="", flush=True)

    def get_joints(self) -> dict[str, float]:
        """Return the integrated joint state."""
        return dict(self._joints)


# Transient Feetech serial errors ("There is no status packet!") show up when
# several joints accelerate at once and the bus voltage briefly sags. LeRobot's
# MotorsBus takes a per-call ``num_retry`` that defaults to 0 - hence the
# "after 1 tries" in the traceback - but the reads that fail are issued *inside*
# LeRobot's send_action/get_observation, where we can't pass that kwarg. So the
# transport retries the whole bus call a few times (below); a transient clears
# on the next attempt, and only a sustained failure (a dead link) re-raises.
BUS_RETRIES = 3
BUS_RETRY_DELAY_S = 0.004  # one 30 Hz control tick has ~33 ms of slack

# Direct-goal mode (relative-target clamp disabled): a move is a single
# Goal_Position write and the servo's onboard controller drives there. Give it a
# speed/accel profile so that one move is smooth and at a sane speed instead of a
# full-speed slam - important because each STS3215 can pull ~3.5 A under hard
# acceleration (≈20 A for six at once, the likely source of the original brownout).
# Raw STS3215 register units: speed ≈ steps/s (1000 ≈ 88°/s, ~the test's old
# 90°/s); accel is the ramp rate. Written best-effort - register names vary by
# firmware/LeRobot version, so the writes are guarded and motion still works
# (at the servo default) if they don't take.
DIRECT_MODE_SPEED_RAW = 1000
DIRECT_MODE_ACCEL_RAW = 50


class DirectArm(ArmClient):
    """Feetech bus via LeRobot ``SO101Follower`` (same as the production node)."""

    # Effort registers to probe, in preference order. STS3215 exposes both;
    # Present_Current is the cleaner contact signal when the firmware reports it.
    _EFFORT_REGISTERS = ("Present_Current", "Present_Load")

    def __init__(
        self, port: str, robot_id: str, max_relative_target: float | None = 8.0
    ):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            cameras={},
            max_relative_target=max_relative_target,
        )
        # Don't torque-off on disconnect. LeRobot's disconnect() writes
        # Torque_Enable=0 to every motor before closing the port; with the joints
        # lightly loaded we'd rather they hold the rest pose than go limp, and that
        # per-motor write was the one failing at shutdown (id=3, "after 6 tries")
        # and leaving the bus mid-transaction - which is the most likely reason the
        # next startup's handshake couldn't find id=6 at the end of the chain.
        # Guarded so it's a no-op if a LeRobot version lacks the flag.
        if hasattr(config, "disable_torque_on_disconnect"):
            config.disable_torque_on_disconnect = False
        self._robot = SO101Follower(config)
        self._robot.connect()
        logger.info("SO101Follower connected on %s (id=%s)", port, robot_id)
        self._effort_register: str | None | bool = False  # False = not probed yet
        self._coordinated_joints: list[
            str
        ] = []  # joints last scaled by set_coordinated_speeds

        # max_relative_target=None means the slew clamp is off, so a single
        # Goal_Position write reaches the servo unthrottled - point-to-point
        # "go to this pose" mode. Tell the base class to route go_to() through
        # move_to(), and give the servos a smooth speed/accel profile.
        self._direct_goal = max_relative_target is None
        if self._direct_goal:
            self.set_motion_profile(
                speed=DIRECT_MODE_SPEED_RAW, acceleration=DIRECT_MODE_ACCEL_RAW
            )

    def set_motion_profile(
        self, speed: int | None = None, acceleration: int | None = None
    ) -> None:
        """Set the servos' onboard speed/accel so one Goal_Position write makes a
        smooth, profiled move (the basis of move_to in direct-goal mode).

        Best-effort: register names differ across Feetech firmware and LeRobot
        versions, so each write is guarded and logged. If a register isn't found
        the move still happens - just at the servo's default speed/accel.
        """
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return

        def _write(candidates: tuple[str, ...], value: int, what: str) -> None:
            """Try each candidate register name until one accepts the write."""
            for name in candidates:
                try:
                    bus.sync_write(
                        name, {m: int(value) for m in MOTOR_NAMES}, normalize=False
                    )
                    logger.info(
                        "Servo %s set to %d (register '%s').", what, value, name
                    )
                    return
                except Exception:
                    continue
            logger.warning(
                "Could not set servo %s (tried %s) - using firmware "
                "default; move may be faster/abrupter than intended.",
                what,
                ", ".join(candidates),
            )

        if acceleration is not None:
            _write(
                ("Maximum_Acceleration", "Acceleration"), acceleration, "acceleration"
            )
        if speed is not None:
            _write(
                ("Goal_Velocity", "Maximum_Speed_Limit", "Goal_Speed"), speed, "speed"
            )

    def _sync_write_guarded(
        self, candidates: tuple[str, ...], values: dict[str, int], what: str
    ) -> bool:
        """sync_write ``values`` (per-joint, raw) trying each register name until
        one works. Returns True on success; logs once and returns False if none
        of the candidate registers exist on this firmware/LeRobot version.

        Each register is retried a few times before moving on: these writes share
        the bus with the 60 Hz control loop running on another thread, so a write
        can collide with a control tick and fail transiently. Without the retry a
        single collision on the *correct* register (the first candidate) falls
        straight through to the non-existent fallbacks and spuriously warns -
        which silently drops coordination. The retry rides the collision out."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return False
        payload = {m: max(int(v), 1) for m, v in values.items()}
        for name in candidates:
            for attempt in range(BUS_RETRIES):
                try:
                    bus.sync_write(name, payload, normalize=False)
                    return True
                except Exception:
                    time.sleep(BUS_RETRY_DELAY_S)
        logger.warning("Could not set per-joint %s - coordination may be off.", what)
        return False

    def set_joint_speeds(self, speeds: dict[str, int] | None = None) -> None:
        """Set the onboard move speed AND acceleration per joint (raw units).
        ``None`` restores the default direct-mode profile on the joints that
        ``set_coordinated_speeds`` last scaled - only those, not all six. Writing
        all six (gripper included) is a longer bus transaction that reliably
        collides with the control loop and fails; restoring just the 1-2 joints
        that were actually changed is short and matches the call that set them."""
        if speeds is None:
            joints = self._coordinated_joints
            if joints:
                self._sync_write_guarded(
                    ("Goal_Velocity", "Maximum_Speed_Limit", "Goal_Speed"),
                    {m: DIRECT_MODE_SPEED_RAW for m in joints},
                    "speed",
                )
                self._sync_write_guarded(
                    ("Maximum_Acceleration", "Acceleration"),
                    {m: DIRECT_MODE_ACCEL_RAW for m in joints},
                    "acceleration",
                )
            self._coordinated_joints = []
            return
        self._sync_write_guarded(
            ("Goal_Velocity", "Maximum_Speed_Limit", "Goal_Speed"), speeds, "speed"
        )

    def set_coordinated_speeds(self, travels: dict[str, float]) -> None:
        """Scale per-joint speed AND acceleration by travel so joints finish a move
        at the same instant and stay in lockstep the whole way: the longest-travel
        joint runs the default profile, the rest run a proportionally smaller copy.

        Scaling *both* speed and accel matters - it makes each joint's motion
        profile an exact scaled copy of the others (twice the travel → twice the
        speed → twice the accel → twice the position at every instant), so a joint
        travelling 2× as far is at 2× the progress throughout, ramps included, not
        just at cruise. With a single goal each, no streaming, so it stays smooth.

        Writes ONLY the joints in ``travels`` (the ones being coordinated). The
        non-paired joints keep their existing profile - writing all six here makes
        the transaction long enough to reliably collide with the control loop and
        fail. ``set_joint_speeds(None)`` restores exactly this set afterwards.
        """
        max_t = max(travels.values(), default=0.0)
        if max_t <= 0:
            return
        speeds = {
            m: max(int(DIRECT_MODE_SPEED_RAW * t / max_t), 50)
            for m, t in travels.items()
        }
        accels = {
            m: max(int(DIRECT_MODE_ACCEL_RAW * t / max_t), 1)
            for m, t in travels.items()
        }
        self._sync_write_guarded(
            ("Goal_Velocity", "Maximum_Speed_Limit", "Goal_Speed"), speeds, "speed"
        )
        self._sync_write_guarded(
            ("Maximum_Acceleration", "Acceleration"), accels, "acceleration"
        )
        self._coordinated_joints = list(travels)

    def _bus_retry(self, op, what: str):
        """Run a bus transaction, retrying briefly on transient serial errors.

        A transient drop ("There is no status packet!") clears on the next
        attempt; a genuinely dead link exhausts the retries and re-raises the
        last error so the caller (the control loop, or a behavior) can react
        rather than acting on stale state.
        """
        last_err: Exception | None = None
        for attempt in range(1, BUS_RETRIES + 1):
            try:
                return op()
            except Exception as e:
                last_err = e
                if attempt < BUS_RETRIES:
                    logger.debug(
                        "%s failed (attempt %d/%d): %s - retrying",
                        what,
                        attempt,
                        BUS_RETRIES,
                        e,
                    )
                    time.sleep(BUS_RETRY_DELAY_S)
        raise last_err  # type: ignore[misc]

    def send_joints(self, joints: dict[str, float]) -> None:
        """Write joints to the bus as an absolute-position action, retrying on transient errors."""
        action = {f"{k}.pos": v for k, v in clamp_joints(joints).items()}
        # send_action reads Present_Position (to clamp against
        # max_relative_target) and then writes Goal_Position; either transaction
        # can drop a packet mid-move, so retry the whole call.
        self._bus_retry(lambda: self._robot.send_action(action), "send_action")

    def get_joints(self) -> dict[str, float]:
        """Read current joint positions from the bus, retrying on transient errors."""
        obs = self._bus_retry(self._robot.get_observation, "get_observation")
        return {m: float(obs[key]) for m, key in _MOTOR_POS_KEYS}

    # ------------------------------------------------------------------
    # Effort feedback (runtime capability detection)
    # ------------------------------------------------------------------

    def _read_effort(self, reg: str):
        """Raw gripper value from effort register ``reg`` (None if unreadable)."""
        bus = getattr(self._robot, "bus", None)
        if bus is None:
            return None
        raw = bus.sync_read(reg, ["gripper"], normalize=False)
        return raw["gripper"] if isinstance(raw, dict) else raw

    def _probe_effort_register(self) -> str | None:
        """Find which effort register (if any) this firmware exposes."""
        for reg in self._EFFORT_REGISTERS:
            try:
                if self._read_effort(reg) is not None:
                    logger.info("Gripper effort feedback available via %s", reg)
                    return reg
            except Exception:
                continue
        logger.info(
            "No gripper effort register readable - grip will use "
            "position-stall detection."
        )
        return None

    def get_gripper_load(self) -> float | None:
        """Return the gripper's effort/current reading, probing for a working register on first use."""
        if self._effort_register is False:
            self._effort_register = self._probe_effort_register()
        if not self._effort_register:
            return None
        try:
            v = int(self._read_effort(self._effort_register))
            # Feetech load/current registers are sign-magnitude in bit 10.
            if v > 1023:
                v -= 1024
            return float(abs(v))
        except Exception as e:
            logger.debug("Effort read failed (%s) - disabling effort feedback.", e)
            self._effort_register = None
            return None

    def close(self) -> None:
        """Park the arm at rest and disconnect from the bus."""
        try:
            # Park the arm before releasing the bus (REST_POSE is within
            # JOINT_LIMITS, so send_joints' clamp is a no-op here).
            if self._direct_goal:
                # One write; the servo profiles its way to rest. Re-writing the
                # same goal would just restart that profile, so send once and
                # give it a moment to arrive. (The caller has usually already
                # parked via go_to_rest, so this is mostly a safety re-assert.)
                # Torque stays on through disconnect (see __init__), so the arm
                # holds this pose rather than going limp.
                self.send_joints(REST_POSE)
                time.sleep(1.0)
            else:
                # Clamp on: each command is throttled to a few degrees, so we
                # have to feed REST_POSE repeatedly to creep there.
                for _ in range(60):
                    self.send_joints(REST_POSE)
                    time.sleep(1 / 30)
        finally:
            self._robot.disconnect()
            logger.info("Robot disconnected.")


def make_arm(cfg: dict) -> ArmClient:
    """Build the ArmClient described by ``cfg['robot']``."""
    r = cfg["robot"]
    if not r.get("enable_motors", True):
        return DryRunArm()
    if r.get("use_ros2", False):
        from vla_pipeline.robot.robot_node import Ros2ArmClient

        return Ros2ArmClient(cfg)
    return DirectArm(r["motor_port"], r["robot_id"], r.get("max_relative_target", 8.0))


# =============================================================================
# Calibration
# =============================================================================
# LeRobot stores per-robot calibration at
#   ~/.cache/huggingface/lerobot/calibration/robots/<robot_type>/<robot_id>.json
# Calibration is INTERACTIVE - it prompts you to move the arm and press ENTER -
# so it can only run in a foreground terminal, never in the backgrounded server
# node. The launcher detects a missing file and runs calibrate_arm() first.


def calibration_path(robot_id: str) -> Path | None:
    """Return the existing calibration file for ``robot_id``, or None."""
    base = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots"
    hits = sorted(base.glob(f"*/{robot_id}.json"))
    return hits[0] if hits else None


def calibration_exists(robot_id: str) -> bool:
    """Return True if a calibration file exists for robot_id."""
    return calibration_path(robot_id) is not None


# Feetech STS3215 encoder resolution (12-bit). LeRobot normalizes degrees as
# ``(raw - mid) * 360 / (resolution - 1)`` with ``mid = (range_min+range_max)/2``.
_STS3215_RESOLUTION = 4096
_DEG_PER_COUNT = 360.0 / (_STS3215_RESOLUTION - 1)

# The gripper uses 0..100 normalization; body joints run in DEGREES mode.
# See lerobot so_follower + motors_bus._normalize.
_GRIPPER_NAME = "gripper"


def load_calibration(robot_id: str) -> dict[str, dict] | None:
    """Load and parse the LeRobot calibration JSON for ``robot_id``.

    Returns a dict ``{motor_name: {id, drive_mode, homing_offset, range_min,
    range_max}}`` or ``None`` if no calibration file exists / it can't be read.
    """
    path = calibration_path(robot_id)
    if path is None:
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Could not read calibration %s: %s", path, e)
        return None
    if not isinstance(data, dict):
        logger.warning("Unexpected calibration format in %s", path)
        return None
    return data


def calibration_limits_and_rest(
    robot_id: str,
    full_range_eps: int = 3500,
) -> tuple[dict[str, tuple[float, float]], dict[str, float], dict[str, dict]]:
    """Derive per-joint software limits and a rest pose from the calibration file.

    The SO-101 follower runs body joints in DEGREES mode: the calibrated zero is
    the **midpoint** of the recorded raw range, and the usable travel each way is
    ``±(range_max - range_min)/2`` converted to degrees. So:

        rest[joint]  = 0.0                         (midpoint maps to 0°)
        limits[joint] = (-half_span_deg, +half_span_deg)

    The gripper uses 0..100 normalization, so its rest is the 50 midpoint and its
    limits are (0, 100).

    A joint that was **not range-swept** during calibration (notably
    ``wrist_roll`` - LeRobot deliberately skips it) keeps the near-full encoder
    range, giving ≈±180°; we detect that (span > ``full_range_eps`` counts) and
    flag it via the returned ``meta`` so callers can warn that its rest/limits
    come from the default range, not a real sweep.

    Returns ``(limits, rest, meta)`` where ``meta[joint]`` is the raw calibration
    entry augmented with ``span_counts`` and ``swept`` (bool).
    """
    cal = load_calibration(robot_id)
    limits: dict[str, tuple[float, float]] = {}
    rest: dict[str, float] = {}
    meta: dict[str, dict] = {}
    if not cal:
        return limits, rest, meta

    for name, entry in cal.items():
        try:
            rmin = float(entry["range_min"])
            rmax = float(entry["range_max"])
        except (KeyError, TypeError, ValueError):
            continue
        span_counts = abs(rmax - rmin)
        swept = span_counts <= full_range_eps  # full encoder range => not swept
        if name == _GRIPPER_NAME:
            # 0..100 normalization: midpoint is 50, range is the full [0,100].
            limits[name] = (0.0, 100.0)
            rest[name] = 50.0
        else:
            half_span_deg = 0.5 * span_counts * _DEG_PER_COUNT
            limits[name] = (-half_span_deg, half_span_deg)
            rest[name] = 0.0  # DEGREES mode: calibrated midpoint == 0°
        meta[name] = {**entry, "span_counts": span_counts, "swept": swept}
    return limits, rest, meta


def calibrate_arm(cfg: dict, force: bool = False) -> None:
    """Run LeRobot's interactive SO-101 calibration (foreground / TTY only).

    Connecting already triggers calibration when the file is missing or
    mismatched; ``force`` re-runs it even when a valid file exists. Either way
    this must run in a real terminal because it calls ``input()`` to step you
    through moving the arm.
    """
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Calibration needs an interactive terminal (it asks you to move the "
            "arm and press ENTER) but stdin is not a TTY. Run it in the "
            "foreground:\n    python -m vla_pipeline.robot.arm_interface --calibrate"
        )
    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    r = cfg["robot"]
    existing = calibration_path(r["robot_id"])
    if existing and not force:
        logger.info(
            "Calibration already exists at %s - nothing to do "
            "(use --calibrate --force to redo).",
            existing,
        )
        return

    config = SO101FollowerConfig(
        port=r["motor_port"],
        id=r["robot_id"],
        cameras={},
        max_relative_target=r.get("max_relative_target", 8.0),
    )
    robot = SO101Follower(config)
    print(f"\nCalibrating SO-101 '{r['robot_id']}' on {r['motor_port']}.")
    print("Follow the prompts: move the arm as instructed and press ENTER.\n")
    robot.connect()  # auto-calibrates when the file is missing/mismatched
    if force:
        robot.calibrate()  # explicit recalibration even if a file existed
    robot.disconnect()
    saved = calibration_path(r["robot_id"])
    print(f"\nCalibration complete: {saved}")


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: component-test the arm, show calibration, or run interactive calibration."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="SO-101 arm interface - component test / calibration"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="no hardware, just print commands"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="run interactive calibration (foreground) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --calibrate: recalibrate even if a file exists",
    )
    parser.add_argument(
        "--config", default=None, help="pipeline.yaml (for --calibrate)"
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="my_awesome_follower_arm")
    parser.add_argument(
        "--show-calibration",
        action="store_true",
        help="print calibration-derived limits/rest for --id and exit",
    )
    args = parser.parse_args()

    if args.show_calibration:
        limits, rest, meta = calibration_limits_and_rest(args.id)
        if not limits:
            print(f"No calibration found for '{args.id}'.")
            return
        print(f"Calibration-derived limits/rest for '{args.id}':")
        for m in MOTOR_NAMES:
            if m in limits:
                lo, hi = limits[m]
                swept = meta[m].get("swept", True)
                tag = "" if swept else "  (NOT range-swept - default range)"
                print(
                    f"  {m:<14} limits=({lo:7.1f}, {hi:7.1f})  rest={rest[m]:6.1f}{tag}"
                )
        return

    if args.calibrate:
        from vla_pipeline.utils.config import load_config

        calibrate_arm(load_config(args.config), force=args.force)
        return

    arm: ArmClient = DryRunArm() if args.dry_run else DirectArm(args.port, args.id)
    try:
        print("Current joints:", {k: round(v, 1) for k, v in arm.get_joints().items()})
        print("Gripper effort feedback:", arm.get_gripper_load())
        print("Wiggling gripper (open → close → open)...")
        arm.go_to({"gripper": 60.0}, duration_s=1.5)
        arm.go_to({"gripper": 5.0}, duration_s=1.5)
        arm.go_to({"gripper": 30.0}, duration_s=1.0)
        print("\nReturning to rest pose...")
        arm.go_to_rest()
        print("Arm interface test PASSED")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
