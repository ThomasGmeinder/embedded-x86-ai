# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Direct camera-frame → SO-101 joint mapping for gesture mimic.

This is the mapping used by :mod:`vla_pipeline.behaviors.gesture_mimic`. Unlike
an analytic-IK retargeter, it does **not** try to
reproduce the human arm's geometry. It treats the camera image as a control
surface and maps a handful of human signals straight onto the joints, the same
way ``test_ros2_arm_motion`` commands them - absolute joint targets in LeRobot
calibration degrees, every command clamped to :data:`JOINT_LIMITS`.

The signals (all per frame)
---------------------------
- **YOLO wrist keypoint (x, y)** in the frame → where the hand is:
  horizontal position → **left/right** (x), vertical position → **up/down** (y).
  This is the *primary* position cue and the one the robot falls back to when the
  MediaPipe hand is lost (see :meth:`CameraMimicMapper.step`).
- **MediaPipe hand size** (apparent span, fraction of frame height) → how close
  the hand is → **forward/back** (z): a big/near hand reaches the robot forward,
  a small/far hand pulls it back. The neutral size auto-calibrates from the first
  few frames (or set ``hand_size_neutral`` in config).
- **MediaPipe directed thumb→index vector** (with the knuckle-line reference and
  the wrist→knuckle angle as fallbacks, all unwrapped for continuity) →
  robot **wrist_roll**.
- **MediaPipe thumb↔index distance** (``pinch``, calibrated min..max) →
  **gripper**, as a *continuous* ramp: closed for the first ``grip_close_frac``
  of the range, then opening linearly up to ``grip_open_pos`` (≈90 %).

How those become joints (the SO-101 mechanics, per the arm's behavior)
----------------------------------------------------------------------
Everything is an offset from a neutral "ready" pose (hand centered, mid-depth,
palm forward, hand open). Four normalized axes in ``[-1, 1]`` drive the offsets:

    shoulder_pan  = pan0  + pan_gain * lr
    shoulder_lift = lift0 + sl_up_gain * up + sl_fwd_gain * fwd
    elbow_flex    = elb0  + ef_up_gain * up + ef_fwd_gain * fwd      (ef_fwd < 0)
    wrist_flex    = wf0   + wf_up_gain * up
    wrist_roll    = roll0 + roll_gain * roll                          (roll0 = 0)
    gripper       = pinch → closed below grip_close_frac, then linear to ~90

so that, matching the described arm:

- **shoulder_pan** alone is left/right (x).
- **up/down (y)** is carried by **elbow_flex at full authority (100 %)**,
  **shoulder_lift at reduced authority (its lower ~50 %)**, and **wrist_flex** as
  a fine-tune that counter-rotates to keep the gripper roughly level - wrist_flex
  is essentially an extension of elbow_flex. ``ef_up_gain`` is therefore the
  largest of the three up/down gains and ``sl_up_gain`` the smallest.
- **forward/back (z)** drives **shoulder_lift the rest of the way** - once
  up/down has used its lower half, reaching forward extends shoulder_lift through
  its upper half (50 %→100 %) while **elbow_flex goes negative** to reach out;
  pulling back reverses it. ``shoulder_lift``'s forward gain is therefore its
  largest term. shoulder_lift and elbow_flex carry both contributions, summed.
- **wrist_roll**'s forward-most pose is the calibration midpoint (``0``); it
  follows the **directed thumb→index vector**, falling back to the steady
  knuckle-line reference (then wrist→knuckle) when the fingers are pinched too
  close to read the line, and is unwrapped so it only moves on a real rotation.

The default gains are sized so the full input envelope stays inside
:data:`JOINT_LIMITS` with margin (see the ``--selftest`` range check); they are
all exposed under ``behaviors.gesture_mimic`` in ``config/pipeline.yaml`` for
on-hardware tuning.

Independent test (pure math - no camera, no hardware)
-----------------------------------------------------
    python -m vla_pipeline.utils.gesture_map --selftest
"""

from __future__ import annotations

import argparse
import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from vla_pipeline.robot.arm_interface import JOINT_LIMITS, clamp_joints


# =============================================================================
# One Euro filter (Casiez et al. 2012) - compact, dependency-free copy so this
# module smooths on its own without importing the retarget pipeline it replaces.
# =============================================================================


class _LowPass:
    """Exponential moving average with a configurable smoothing factor alpha."""

    def __init__(self) -> None:
        self.y: float | None = None

    def __call__(self, x: float, alpha: float) -> float:
        self.y = x if self.y is None else alpha * x + (1.0 - alpha) * self.y
        return self.y


class OneEuro:
    """Adaptive low-pass: heavy smoothing when slow, low lag when fast."""

    def __init__(
        self, min_cutoff: float = 1.2, beta: float = 0.02, d_cutoff: float = 1.0
    ):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()
        self._t_prev: float | None = None
        self._x_prev: float | None = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        """Smoothing factor for a given cutoff frequency and time step."""
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: float | None = None) -> float:
        t = time.perf_counter() if t is None else t
        if self._t_prev is None:
            self._t_prev, self._x_prev = t, x
            self._x(x, 1.0)
            self._dx(0.0, 1.0)
            return x
        dt = max(t - self._t_prev, 1e-4)
        dx = (x - (self._x_prev if self._x_prev is not None else x)) / dt
        edx = self._dx(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        out = self._x(x, self._alpha(cutoff, dt))
        self._t_prev, self._x_prev = t, x
        return out

    def reset(self) -> None:
        """Reset the filter to its just-constructed state."""
        self.__init__(self.min_cutoff, self.beta, self.d_cutoff)


class VecOneEuro:
    """Independent One Euro filters over a fixed set of named channels."""

    def __init__(self, names: tuple[str, ...], **kw):
        self._f = {n: OneEuro(**kw) for n in names}

    def __call__(
        self, values: dict[str, float], t: float | None = None
    ) -> dict[str, float]:
        t = time.perf_counter() if t is None else t
        return {n: self._f[n](v, t) for n, v in values.items()}

    def reset(self) -> None:
        """Reset every per-channel filter."""
        for f in self._f.values():
            f.reset()


# =============================================================================
# Tunables
# =============================================================================


@dataclass
class MimicGains:
    """Neutral anchor pose (deg) and per-axis gains (deg per unit of axis).

    Defaults are chosen so the whole ``[-1, 1]^4`` input cube maps inside
    :data:`JOINT_LIMITS` with margin. The neutral pose matches the working,
    forward-facing poses used by the other behaviors (e.g. pick_place's
    ``scan_pose``), and ``wrist_roll0 = 0`` is the forward-most / calibration
    midpoint as specified.
    """

    # Neutral "ready" pose: RE-CENTERED on each joint's mid-range so the input
    # envelope can drive every actuator to BOTH of its limits (see gains below).
    # Midpoints from JOINT_LIMITS: pan 0, lift -5, elbow 10, wrist_flex -10,
    # roll 0. The arm therefore rests centered rather than reaching forward.
    shoulder_pan0: float = 0.0
    shoulder_lift0: float = -5.0
    elbow_flex0: float = 10.0
    wrist_flex0: float = -10.0
    wrist_roll0: float = 0.0  # forward-most == calibration midpoint

    # Gains are OVER-DRIVEN (~30%+) past each joint's half-span so every limit is
    # reachable with an easy single/two-axis motion instead of needing the hand at
    # an exact frame corner / depth extreme. The excess is absorbed by the safety
    # clamp in joints_from_axes (so a range of hand positions all map to the limit
    # - intentional saturation). The whole cube still clamps in range (--selftest).
    # left/right: shoulder_pan alone overshoots the ±115 limit so it saturates.
    pan_gain: float = 150.0

    # forward/back (z): shoulder_lift is the PRIMARY driver - a near hand reaches
    # it to its +110 limit on depth alone (neutral -5 + 110), a far hand pushes
    # past -120 - with elbow_flex going negative to reach out.
    sl_fwd_gain: float = 110.0
    ef_fwd_gain: float = -75.0

    # up/down (y): elbow_flex at FULL authority leads (ef_up the largest up term);
    # shoulder_lift adds more up travel; wrist_flex counter-rotates and over-swings
    # to reach both its limits. Each pair's sum overshoots the half-span so the
    # corners saturate at the clamp rather than falling short.
    sl_up_gain: float = 58.0
    ef_up_gain: float = 78.0
    wf_up_gain: float = -135.0  # overshoots ±100 about the -10 neutral

    # wrist roll (about the auto-centered midpoint): overshoots the ±100 limit.
    roll_gain: float = 128.0


@dataclass
class MimicNorm:
    """How raw camera signals normalize to the ``[-1, 1]`` control axes."""

    # Wrist position within the frame. center/half define the window that maps
    # to the full [-1, 1]; e.g. x_half=0.35 means the central 70% of the width.
    x_center: float = 0.5
    x_half: float = 0.35
    y_center: float = 0.5
    y_half: float = 0.30
    deadzone: float = 0.05  # |axis| below this snaps to 0 (kills jitter)
    mirror: bool = True  # mount camera faces the user → mirror L/R

    # Depth from hand size. 0 neutral = auto-calibrate from the first
    # ``depth_baseline_frames`` valid readings. depth_rel_span = fractional size
    # change that reaches the full forward/back travel (0.4 → +40% size = +1).
    hand_size_neutral: float = 0.0
    depth_rel_span: float = 0.40
    depth_baseline_frames: int = 8

    # Roll. The primary cue is the DIRECTED thumb→index vector (degrees in
    # (-180,180]). The roll is measured RELATIVE to a center that AUTO-CALIBRATES
    # to whatever orientation your hand is in the first time it's reliably seen
    # (and after each reset) - so however you naturally hold your hand becomes
    # wrist_roll 0, and rotating from there drives the joint. This fixes roll
    # "not working" when a fixed 0° center left the mapping saturated.
    # ``half`` is the tilt each way (deg) that reaches the wrist_roll extreme.
    # Set ``roll_auto_center: false`` (and ``roll_center_deg``) to pin it instead.
    roll_auto_center: bool = True
    roll_center_deg: float = 0.0  # used only when roll_auto_center is False
    roll_half_deg: float = 75.0
    roll_ti_min_weight: float = 0.4
    roll_invert: bool = False

    # Gripper from the thumb↔index distance (``pinch``) - a CONTINUOUS ramp. The
    # pinch is first normalized to [0,1] against the calibrated min/max
    # (``grip_min``..``grip_max`` from --calibrate-grip), then mapped to the
    # gripper opening:
    #   norm <= grip_close_frac           → CLOSED   (held at GRIPPER_CLOSED_FLOOR)
    #   grip_close_frac < norm <= 1.0     → smoothly OPENS, linearly, up to
    #                                        grip_open_pos at norm == 1.0
    # So fingers within the closest 10% of the range keep the gripper shut, and
    # spreading them past that opens it progressively to ~90%.
    grip_min: float = 0.10  # calibrated: fingers touching  → norm 0
    grip_max: float = 1.40  # calibrated: spread wide apart → norm 1
    grip_close_frac: float = 0.10  # norm at/below this → fully closed
    grip_open_pos: float = 90.0  # gripper opening (0-100) at norm == 1.0


def _axis(value: float, center: float, half: float, deadzone: float = 0.0) -> float:
    """Normalize ``value`` about ``center`` (±``half`` → ±1) with a deadzone."""
    a = (value - center) / (half if half else 1e-6)
    a = float(np.clip(a, -1.0, 1.0))
    if deadzone > 0.0 and abs(a) < deadzone:
        return 0.0
    return a


# Closed sits at GRIPPER_CLOSED_FLOOR so it never jams into the mechanical stop
# (see arm_interface.GRIPPER_CLOSED_FLOOR).
from vla_pipeline.robot.arm_interface import GRIPPER_CLOSED_FLOOR


def grip_opening(
    pinch_norm: float,
    close_frac: float,
    open_pos: float,
    closed_pos: float = GRIPPER_CLOSED_FLOOR,
) -> float:
    """Continuous thumb↔index distance → gripper opening (0-100 scale).

    ``pinch_norm`` is the calibrated distance in [0,1]. At/below ``close_frac``
    the gripper is fully closed (``closed_pos``, the safe floor); from there to
    1.0 it opens *linearly* up to ``open_pos`` (≈90%). The result is clamped to
    the gripper's joint limits.
    """
    lo, hi = JOINT_LIMITS["gripper"]
    if pinch_norm <= close_frac:
        return float(np.clip(closed_pos, lo, hi))
    span = max(1.0 - close_frac, 1e-6)
    frac = (pinch_norm - close_frac) / span  # 0 at close_frac → 1 at norm 1.0
    opening = closed_pos + frac * (open_pos - closed_pos)
    return float(np.clip(opening, lo, hi))


# =============================================================================
# The mapper
# =============================================================================


class CameraMimicMapper:
    """Per-frame: camera signals → smoothed, range-safe SO-101 joint targets.

    Pure geometry lives in :meth:`joints_from_axes` (no time, no smoothing) so
    it can be swept exhaustively in tests; :meth:`step` adds normalization,
    auto depth calibration, One Euro smoothing, and the gripper.
    """

    AXES = ("lr", "up", "fwd", "roll")

    def __init__(
        self,
        gains: MimicGains | None = None,
        norm: MimicNorm | None = None,
        min_cutoff: float = 1.5,
        beta: float = 0.03,
        d_cutoff: float = 1.0,
        grip_start_pos: float | None = None,
    ):
        self.g = gains or MimicGains()
        self.n = norm or MimicNorm()
        self._filt = VecOneEuro(
            self.AXES, min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff
        )

        # State held across frames so a momentarily missing signal "holds".
        self._raw = {a: 0.0 for a in self.AXES}
        self._size_buf: deque[float] = deque(
            maxlen=max(self.n.depth_baseline_frames, 1)
        )
        self._size_neutral: float | None = (
            self.n.hand_size_neutral if self.n.hand_size_neutral > 0 else None
        )
        # Continuous gripper opening (0-100), held across frames - and across
        # reset() - so a momentary hand dropout never moves the gripper; it only
        # changes when a NEW thumb↔index distance is read. Defaults to fully open.
        self._grip_pos: float = (
            float(grip_start_pos)
            if grip_start_pos is not None
            else self.n.grip_open_pos
        )
        # Roll continuity state: the previous RAW roll angle (deg) and a running
        # offset to the directed thumb→index reading captured the first time a
        # reliable reading appears. Unwrapping against the previous angle means
        # the commanded roll only changes when the hand actually rotates - it no
        # longer jumps when the thumb↔index reading wraps or the cue switches.
        self._roll_prev_deg: float | None = None
        self._roll_ref_offset: float | None = None  # roll_ti − roll_ref at lock-in
        # Auto-centered roll origin: the (unwrapped) source angle captured the
        # first time a reliable roll is seen. Rotation is measured relative to it,
        # so whatever orientation the hand starts in becomes wrist_roll 0.
        self._roll_center_deg: float | None = None

    # ------------------------------------------------------------------ pure
    def joints_from_axes(
        self, lr: float, up: float, fwd: float, roll: float
    ) -> dict[str, float]:
        """The mapping table → joint targets (deg), clamped to JOINT_LIMITS."""
        g = self.g
        joints = {
            "shoulder_pan": g.shoulder_pan0 + g.pan_gain * lr,
            "shoulder_lift": g.shoulder_lift0 + g.sl_up_gain * up + g.sl_fwd_gain * fwd,
            "elbow_flex": g.elbow_flex0 + g.ef_up_gain * up + g.ef_fwd_gain * fwd,
            "wrist_flex": g.wrist_flex0 + g.wf_up_gain * up,
            "wrist_roll": g.wrist_roll0 + g.roll_gain * roll,
        }
        return clamp_joints(joints)

    # ------------------------------------------------------------ depth helper
    def _depth_axis(self, hand_size: float | None) -> float | None:
        """hand size → forward axis in [-1, 1], or None until neutral is known."""
        if hand_size is None or hand_size <= 0.0:
            return None
        if self._size_neutral is None:
            self._size_buf.append(hand_size)
            if len(self._size_buf) < self._size_buf.maxlen:
                return 0.0  # still building the baseline → no forward command yet
            self._size_neutral = float(np.median(self._size_buf))
        rel = hand_size / max(self._size_neutral, 1e-6) - 1.0
        return float(np.clip(rel / max(self.n.depth_rel_span, 1e-6), -1.0, 1.0))

    # ------------------------------------------------------------------- frame
    def step(
        self,
        wrist_xy: tuple[float, float] | None,
        frame_w: int,
        frame_h: int,
        hand_size: float | None = None,
        hand_roll_deg: float | None = None,
        roll_ti_deg: float | None = None,
        roll_ref_deg: float | None = None,
        roll_ti_weight: float = 0.0,
        pinch: float | None = None,
        gripper_override: float | None = None,
        t: float | None = None,
    ) -> dict[str, float]:
        """Map one frame's signals to joint targets.

        Any signal may be ``None`` (not detected this frame) - its axis simply
        holds its previous value, so when the MediaPipe hand is lost the robot
        keeps following the YOLO wrist (``wrist_xy``) for x/y while depth, roll
        and the gripper freeze at their last commanded value. ``gripper_override``
        forces the gripper (used by the grip-lock so a held object stays gripped).

        Wrist roll uses the directed thumb→index vector (``roll_ti_deg``) as the
        primary cue and the knuckle-line reference (``roll_ref_deg``, then the
        wrist→knuckle ``hand_roll_deg``) as a fallback. The chosen angle is
        unwrapped against the previous frame so the command only changes when the
        hand actually rotates - it doesn't flip on a wrap or a cue switch.
        """
        n = self.n

        # Left/right + up/down from the YOLO wrist position in the frame.
        if wrist_xy is not None and frame_w > 0 and frame_h > 0:
            lr = _axis(wrist_xy[0] / frame_w, n.x_center, n.x_half, n.deadzone)
            if n.mirror:
                lr = -lr
            # Image y grows downward → invert so "hand high" is +up.
            up = _axis(
                1.0 - wrist_xy[1] / frame_h, 1.0 - n.y_center, n.y_half, n.deadzone
            )
            self._raw["lr"], self._raw["up"] = lr, up

        # Forward/back from hand size (holds last when the hand is unseen).
        fwd = self._depth_axis(hand_size)
        if fwd is not None:
            self._raw["fwd"] = fwd

        # Wrist roll: pick a continuous source angle (directed thumb→index when
        # reliable, else the steady knuckle-line reference / wrist→knuckle), unwrap
        # it for frame-to-frame continuity, then normalize to the [-1,1] axis.
        roll_axis = self._roll_axis(
            roll_ti_deg, roll_ref_deg, roll_ti_weight, hand_roll_deg
        )
        if roll_axis is not None:
            if n.roll_invert:
                roll_axis = -roll_axis
            self._raw["roll"] = roll_axis

        smoothed = self._filt(self._raw, t)
        joints = self.joints_from_axes(**smoothed)

        # Gripper: forced lock wins; otherwise a CONTINUOUS opening from the
        # calibrated thumb↔index distance - closed for the first grip_close_frac
        # of the range, then ramping linearly to grip_open_pos. The opening holds
        # whenever pinch is None (hand lost), so the gripper never resets on its
        # own - it only moves when a NEW index↔thumb distance is measured.
        if gripper_override is not None:
            joints["gripper"] = float(
                np.clip(gripper_override, *JOINT_LIMITS["gripper"])
            )
        else:
            if pinch is not None:
                pinch_norm = self._pinch_norm(pinch)
                self._grip_pos = grip_opening(
                    pinch_norm, n.grip_close_frac, n.grip_open_pos
                )
            joints["gripper"] = self._grip_pos

        return joints

    # ------------------------------------------------------------ roll helper
    @staticmethod
    def _unwrap_deg(prev: float, angle: float) -> float:
        """Return ``angle`` shifted by whole turns to be nearest ``prev``.

        Keeps a directed angle continuous across the ±180 seam so a hand that
        rotates smoothly produces a smoothly changing value (and a small jitter
        across the seam never looks like a flip).
        """
        return prev + (angle - prev + 180.0) % 360.0 - 180.0

    def _roll_axis(self, roll_ti_deg, roll_ref_deg, ti_weight, hand_roll_deg):
        """Pick + unwrap a roll source → roll axis in [-1,1] (None if no cue).

        Priority: the directed thumb→index vector when its reliability
        ``ti_weight`` is high enough; otherwise the steady knuckle-line reference
        (kept on the thumb→index scale via a locked-in offset so switching cues
        doesn't jump); otherwise the wrist→knuckle fallback. The result is
        unwrapped against the previous frame so it only moves on real rotation.
        """
        n = self.n

        # Maintain the offset between the two MediaPipe cues whenever BOTH are
        # seen, so the reference can stand in for the thumb→index line on the same
        # scale (no jump when the line gets too short to trust).
        if (
            roll_ti_deg is not None
            and roll_ref_deg is not None
            and ti_weight >= n.roll_ti_min_weight
        ):
            self._roll_ref_offset = roll_ti_deg - roll_ref_deg

        # Choose this frame's raw source angle (directed, degrees).
        source = None
        if roll_ti_deg is not None and ti_weight >= n.roll_ti_min_weight:
            source = roll_ti_deg
        elif roll_ref_deg is not None and self._roll_ref_offset is not None:
            source = roll_ref_deg + self._roll_ref_offset  # reference on the ti scale
        elif roll_ref_deg is not None:
            source = roll_ref_deg
        elif hand_roll_deg is not None:
            source = hand_roll_deg
        if source is None:
            return None

        # Unwrap for continuity (so a smooth rotation stays smooth across ±180).
        if self._roll_prev_deg is not None:
            source = self._unwrap_deg(self._roll_prev_deg, source)
        self._roll_prev_deg = source

        # Center: auto-calibrate to the first reliable angle (then measure
        # rotation relative to it), or use the fixed configured center.
        if n.roll_auto_center:
            if self._roll_center_deg is None:
                self._roll_center_deg = source
            center = self._roll_center_deg
        else:
            center = n.roll_center_deg
        return _axis(source, center, n.roll_half_deg)

    # ------------------------------------------------------------ pinch helper
    def _pinch_norm(self, pinch: float) -> float:
        """Normalize a raw pinch to [0,1] against calibrated grip_min..grip_max."""
        n = self.n
        span = max(n.grip_max - n.grip_min, 1e-6)
        return float(np.clip((pinch - n.grip_min) / span, 0.0, 1.0))

    def reset(self) -> None:
        """Forget smoothing/baseline state (call when the person leaves view)."""
        self._filt.reset()
        self._raw = {a: 0.0 for a in self.AXES}
        if self.n.hand_size_neutral <= 0:
            self._size_neutral = None
            self._size_buf.clear()
        # Drop roll continuity AND the auto-center so the next reading re-seats
        # cleanly: whatever orientation the hand returns in becomes the new 0.
        self._roll_prev_deg = None
        self._roll_ref_offset = None
        self._roll_center_deg = None
        # Keep the last grip OPENING so a held object isn't dropped on a dropout -
        # the gripper only moves when a new thumb↔index distance is measured.


def mapper_from_config(cfg: dict) -> CameraMimicMapper:
    """Build a :class:`CameraMimicMapper` from ``behaviors.gesture_mimic`` config."""
    g = (cfg.get("behaviors", {}) or {}).get("gesture_mimic", {}) or {}

    def gf(key, default):
        """Read key from the gesture_mimic config section as a float, falling back to default."""
        return float(g.get(key, default))

    gains = MimicGains(
        shoulder_pan0=gf("shoulder_pan0", MimicGains.shoulder_pan0),
        shoulder_lift0=gf("shoulder_lift0", MimicGains.shoulder_lift0),
        elbow_flex0=gf("elbow_flex0", MimicGains.elbow_flex0),
        wrist_flex0=gf("wrist_flex0", MimicGains.wrist_flex0),
        wrist_roll0=gf("wrist_roll0", MimicGains.wrist_roll0),
        pan_gain=gf("pan_gain", MimicGains.pan_gain),
        sl_fwd_gain=gf("sl_fwd_gain", MimicGains.sl_fwd_gain),
        ef_fwd_gain=gf("ef_fwd_gain", MimicGains.ef_fwd_gain),
        sl_up_gain=gf("sl_up_gain", MimicGains.sl_up_gain),
        ef_up_gain=gf("ef_up_gain", MimicGains.ef_up_gain),
        wf_up_gain=gf("wf_up_gain", MimicGains.wf_up_gain),
        roll_gain=gf("roll_gain", MimicGains.roll_gain),
    )
    norm = MimicNorm(
        x_center=gf("x_center", MimicNorm.x_center),
        x_half=gf("x_half", MimicNorm.x_half),
        y_center=gf("y_center", MimicNorm.y_center),
        y_half=gf("y_half", MimicNorm.y_half),
        deadzone=gf("deadzone", MimicNorm.deadzone),
        mirror=bool(g.get("mirror", MimicNorm.mirror)),
        hand_size_neutral=gf("hand_size_neutral", MimicNorm.hand_size_neutral),
        depth_rel_span=gf("depth_rel_span", MimicNorm.depth_rel_span),
        depth_baseline_frames=int(
            g.get("depth_baseline_frames", MimicNorm.depth_baseline_frames)
        ),
        roll_auto_center=bool(g.get("roll_auto_center", MimicNorm.roll_auto_center)),
        roll_center_deg=gf("roll_center_deg", MimicNorm.roll_center_deg),
        roll_half_deg=gf("roll_half_deg", MimicNorm.roll_half_deg),
        roll_ti_min_weight=gf("roll_ti_min_weight", MimicNorm.roll_ti_min_weight),
        roll_invert=bool(g.get("roll_invert", MimicNorm.roll_invert)),
        grip_min=gf("grip_min", MimicNorm.grip_min),
        grip_max=gf("grip_max", MimicNorm.grip_max),
        grip_close_frac=gf("grip_close_frac", MimicNorm.grip_close_frac),
        grip_open_pos=gf("grip_open_pos", MimicNorm.grip_open_pos),
    )
    sm = g.get("smoothing", {}) or {}
    # Starting grip opening (0-100); default = fully open (grip_open_pos).
    gsp = g.get("grip_start_pos", None)
    grip_start_pos = float(gsp) if gsp is not None else None
    return CameraMimicMapper(
        gains=gains,
        norm=norm,
        min_cutoff=float(sm.get("min_cutoff", 1.5)),
        beta=float(sm.get("beta", 0.03)),
        d_cutoff=float(sm.get("d_cutoff", 1.0)),
        grip_start_pos=grip_start_pos,
    )


# =============================================================================
# Standalone self-test (pure math - no camera, no hardware)
# =============================================================================


def _selftest() -> int:
    """Run the built-in geometry/range/continuity checks and print PASS/FAIL for each."""
    failures = 0  # noqa
    m = CameraMimicMapper()
    g = m.g

    # 1. Range safety: the whole input cube stays inside JOINT_LIMITS.
    grid = np.linspace(-1.0, 1.0, 9)
    worst = {}
    out_of_range = 0
    for lr in grid:
        for up in grid:
            for fwd in grid:
                for roll in grid:
                    j = m.joints_from_axes(lr, up, fwd, roll)
                    for name, v in j.items():
                        lo, hi = JOINT_LIMITS[name]
                        margin = min(v - lo, hi - v)
                        if name not in worst or margin < worst[name]:
                            worst[name] = margin
                        if v < lo - 1e-6 or v > hi + 1e-6:
                            out_of_range += 1
    ok = out_of_range == 0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] range safety: {out_of_range} of "
        f"{grid.size**4} cube samples outside JOINT_LIMITS"
    )
    print(
        "           tightest margins (deg): "
        + ", ".join(f"{k}={worst[k]:.1f}" for k in sorted(worst))
    )
    failures += not ok

    # 2. Neutral: all axes 0 → the neutral anchor pose.
    j0 = m.joints_from_axes(0.0, 0.0, 0.0, 0.0)
    ok = (
        abs(j0["shoulder_pan"] - g.shoulder_pan0) < 1e-6
        and abs(j0["shoulder_lift"] - g.shoulder_lift0) < 1e-6
        and abs(j0["elbow_flex"] - g.elbow_flex0) < 1e-6
        and abs(j0["wrist_flex"] - g.wrist_flex0) < 1e-6
        and abs(j0["wrist_roll"] - g.wrist_roll0) < 1e-6
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] neutral axes → anchor pose "
        f"(SL={j0['shoulder_lift']:.0f}, EF={j0['elbow_flex']:.0f}, WR={j0['wrist_roll']:.0f})"
    )
    failures += not ok

    # 3. Forward/back: forward raises shoulder_lift AND lowers elbow_flex.
    jf = m.joints_from_axes(0.0, 0.0, +1.0, 0.0)
    jb = m.joints_from_axes(0.0, 0.0, -1.0, 0.0)
    ok = (
        jf["shoulder_lift"] > g.shoulder_lift0
        and jf["elbow_flex"] < g.elbow_flex0
        and jb["shoulder_lift"] < g.shoulder_lift0
        and jb["elbow_flex"] > g.elbow_flex0
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] forward → SL+{jf['shoulder_lift']-g.shoulder_lift0:+.0f} "
        f"EF{jf['elbow_flex']-g.elbow_flex0:+.0f}; back → SL{jb['shoulder_lift']-g.shoulder_lift0:+.0f} "
        f"EF{jb['elbow_flex']-g.elbow_flex0:+.0f}"
    )
    failures += not ok

    # 4. Up/down: elbow_flex is the PRIMARY driver (largest authority); all three
    #    joints move together and the wrist counter-rotates to stay level.
    ju = m.joints_from_axes(0.0, +1.0, 0.0, 0.0)
    ok = (
        math.copysign(1, ju["shoulder_lift"] - g.shoulder_lift0)
        == math.copysign(1, g.sl_up_gain)
        and math.copysign(1, ju["elbow_flex"] - g.elbow_flex0)
        == math.copysign(1, g.ef_up_gain)
        and math.copysign(1, ju["wrist_flex"] - g.wrist_flex0)
        == math.copysign(1, g.wf_up_gain)
        and abs(g.ef_up_gain) > abs(g.sl_up_gain)
    )  # elbow leads, shoulder reduced
    print(
        f"  [{'PASS' if ok else 'FAIL'}] up → SL{ju['shoulder_lift']-g.shoulder_lift0:+.0f} "
        f"EF{ju['elbow_flex']-g.elbow_flex0:+.0f} WF{ju['wrist_flex']-g.wrist_flex0:+.0f} "
        f"(elbow leads: |ef|={abs(g.ef_up_gain):.0f} > |sl|={abs(g.sl_up_gain):.0f})"
    )
    failures += not ok

    # 4b. Forward/back: shoulder_lift is the PRIMARY driver of z (its forward gain
    #     is its largest term, so reaching forward extends it into its upper half).
    ok = abs(g.sl_fwd_gain) > abs(g.sl_up_gain)
    print(
        f"  [{'PASS' if ok else 'FAIL'}] z split: shoulder_lift fwd gain "
        f"{abs(g.sl_fwd_gain):.0f} > its up gain {abs(g.sl_up_gain):.0f} "
        f"(lower half = up/down, upper half = forward)"
    )
    failures += not ok

    # 5. Left/right: pan tracks lr (sign only, mirror handled in step()).
    jr = m.joints_from_axes(+1.0, 0.0, 0.0, 0.0)
    jl = m.joints_from_axes(-1.0, 0.0, 0.0, 0.0)
    ok = jr["shoulder_pan"] > 0 > jl["shoulder_pan"]
    print(
        f"  [{'PASS' if ok else 'FAIL'}] pan: lr+1 → {jr['shoulder_pan']:+.0f}, lr-1 → {jl['shoulder_pan']:+.0f}"
    )
    failures += not ok

    # 6. Wrist roll about the forward-most midpoint (0).
    jp = m.joints_from_axes(0.0, 0.0, 0.0, +1.0)
    jn = m.joints_from_axes(0.0, 0.0, 0.0, -1.0)
    ok = (
        abs(jp["wrist_roll"] - g.roll_gain) < 1e-6
        and abs(jn["wrist_roll"] + g.roll_gain) < 1e-6
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] wrist_roll: roll±1 → {jp['wrist_roll']:+.0f}/{jn['wrist_roll']:+.0f} about 0"
    )
    failures += not ok

    # 7. Depth axis: a near (bigger) hand pushes forward, a far (smaller) pulls back.
    m2 = CameraMimicMapper(norm=MimicNorm(hand_size_neutral=0.20, depth_rel_span=0.40))
    near = m2._depth_axis(0.28)  # +40% → +1
    far = m2._depth_axis(0.12)  # −40% → −1
    ok = near is not None and far is not None and near > 0.9 and far < -0.9
    print(
        f"  [{'PASS' if ok else 'FAIL'}] depth: near hand → {near:+.2f}, far hand → {far:+.2f}"
    )
    failures += not ok

    # 8. Auto depth baseline: holds 0 until enough frames, then centers on median.
    m3 = CameraMimicMapper(
        norm=MimicNorm(hand_size_neutral=0.0, depth_baseline_frames=4)
    )
    vals = [m3._depth_axis(0.20) for _ in range(3)]  # building baseline
    settled = m3._depth_axis(0.20)  # 4th completes baseline
    after = m3._depth_axis(0.28)  # now reads forward
    ok = all(v == 0.0 for v in vals) and abs(settled) < 1e-6 and after > 0.5
    print(
        f"  [{'PASS' if ok else 'FAIL'}] auto-baseline: building={vals}, settled={settled:.2f}, near→{after:+.2f}"
    )
    failures += not ok

    # 9. Gripper is a CONTINUOUS ramp: closed within grip_close_frac, then opens
    #    linearly to grip_open_pos at full spread.
    cf, op = MimicNorm.grip_close_frac, MimicNorm.grip_open_pos
    floor = GRIPPER_CLOSED_FLOOR
    o_shut = grip_opening(0.0, cf, op)
    o_edge = grip_opening(cf, cf, op)  # exactly at the close edge
    o_mid = grip_opening(cf + (1 - cf) / 2, cf, op)  # halfway up the ramp
    o_full = grip_opening(1.0, cf, op)
    mid_expected = floor + 0.5 * (op - floor)
    ok = (
        o_shut == floor
        and o_edge == floor
        and abs(o_mid - mid_expected) < 1e-6
        and abs(o_full - op) < 1e-6
        # monotonic non-decreasing across the range
        and all(
            grip_opening(a, cf, op) <= grip_opening(b, cf, op) + 1e-9
            for a, b in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:])
        )
    )
    print(
        f"  [{'PASS' if ok else 'FAIL'}] gripper ramp: shut={o_shut:.0f}, edge={o_edge:.0f}, "
        f"mid={o_mid:.0f}, full={o_full:.0f} (closed≤{cf:.0%} then →{op:.0f})"
    )
    failures += not ok

    # 9b. Gripper HOLDS its opening when the hand is lost (pinch=None) and across
    #     reset() - it must never reset on its own.
    mh = CameraMimicMapper()
    mh.step((320, 240), 640, 480, pinch=mh.n.grip_min, t=0.0)  # → closed
    held = mh.step(None, 640, 480, pinch=None, t=0.1)["gripper"]  # hand lost
    mh.reset()
    held_after_reset = mh.step(None, 640, 480, pinch=None, t=0.2)["gripper"]
    ok = abs(held - floor) < 1e-6 and abs(held_after_reset - floor) < 1e-6
    print(
        f"  [{'PASS' if ok else 'FAIL'}] gripper holds on hand-loss: "
        f"held={held:.0f}, after reset={held_after_reset:.0f}"
    )
    failures += not ok

    # 9c. Roll AUTO-CENTERS: whatever angle the hand starts at maps to 0, and
    #     rotating from there drives the joint. (Fixes roll sitting saturated when
    #     a fixed 0° center didn't match how the hand is held.)
    mr = CameraMimicMapper()  # roll_auto_center is on by default
    base = 120.0  # an awkward starting orientation
    j0 = mr.step((320, 240), 640, 480, roll_ti_deg=base, roll_ti_weight=1.0, t=0.0)
    j_rot = mr.step(
        (320, 240),
        640,
        480,
        roll_ti_deg=base + mr.n.roll_half_deg,
        roll_ti_weight=1.0,
        t=0.1,
    )
    j_rotn = mr.step(
        (320, 240),
        640,
        480,
        roll_ti_deg=base - mr.n.roll_half_deg,
        roll_ti_weight=1.0,
        t=0.2,
    )
    ok = (
        abs(j0["wrist_roll"]) < 1e-6  # first reading → centered at 0
        and j_rot["wrist_roll"] > 0  # rotate one way → positive
        and j_rotn["wrist_roll"] < 0
    )  # rotate the other → negative
    print(
        f"  [{'PASS' if ok else 'FAIL'}] roll auto-center: start({base:.0f})→{j0['wrist_roll']:+.0f}, "
        f"+turn→{j_rot['wrist_roll']:+.0f}, -turn→{j_rotn['wrist_roll']:+.0f}"
    )
    failures += not ok

    # 9d. Low thumb↔index reliability falls back to the steady knuckle reference
    #     instead of swinging to the (misleading) short-line angle.
    mq = CameraMimicMapper()
    # Seat the center with a reliable reading at the reference orientation.
    mq.step(
        (320, 240),
        640,
        480,
        roll_ti_deg=0.0,
        roll_ref_deg=0.0,
        roll_ti_weight=1.0,
        t=0.0,
    )
    # Now ti reads far off but is unreliable; ref stays near center → small output.
    j_lo = mq.step(
        (320, 240),
        640,
        480,
        roll_ti_deg=90.0,
        roll_ref_deg=0.0,
        roll_ti_weight=0.0,
        t=0.1,
    )
    ok = abs(j_lo["wrist_roll"]) < 20.0  # ignored the misleading ti angle
    print(
        f"  [{'PASS' if ok else 'FAIL'}] roll low-weight → reference: "
        f"ti=90 ignored, wrist_roll={j_lo['wrist_roll']:+.0f}"
    )
    failures += not ok

    # 9e. Roll does NOT flip when the directed angle crosses the ±180 seam - the
    #     unwrap keeps it continuous (the wrist_roll-flipping bug fix). With
    #     auto-center, 170 seats the center (→0); -170 is a +20° rotation across
    #     the seam, so the command must nudge POSITIVE a little (not jump ~-340°).
    md = CameraMimicMapper(
        norm=MimicNorm(roll_half_deg=200.0)
    )  # wide so we stay unclamped
    a = md.step((320, 240), 640, 480, roll_ti_deg=170.0, roll_ti_weight=1.0, t=0.0)[
        "wrist_roll"
    ]
    b = md.step((320, 240), 640, 480, roll_ti_deg=-170.0, roll_ti_weight=1.0, t=0.1)[
        "wrist_roll"
    ]
    ok = abs(a) < 1e-6 and b > a and (b - a) < 60.0
    print(
        f"  [{'PASS' if ok else 'FAIL'}] roll seam continuity: 170°→{a:+.0f}, then -170°→{b:+.0f} "
        f"(Δ={b - a:+.0f}, no flip)"
    )
    failures += not ok

    # 9e. Reach-all-limits: each joint's commanded value gets within 10° of BOTH
    #     its JOINT_LIMITS somewhere in the input cube (the "hit all limits" goal).
    reach = {
        name: [float("inf"), float("-inf")]
        for name in JOINT_LIMITS
        if name != "gripper"
    }
    for lr in (-1.0, 1.0):
        for up in (-1.0, 1.0):
            for fwd in (-1.0, 1.0):
                for roll in (-1.0, 1.0):
                    j = m.joints_from_axes(lr, up, fwd, roll)
                    for nme, v in j.items():
                        if nme == "gripper":
                            continue
                        reach[nme][0] = min(reach[nme][0], v)
                        reach[nme][1] = max(reach[nme][1], v)
    near = 10.0
    unreached = []
    for nme, (lo_lim, hi_lim) in JOINT_LIMITS.items():
        if nme == "gripper":
            continue
        rmin, rmax = reach[nme]
        if rmin - lo_lim > near or hi_lim - rmax > near:
            unreached.append(
                f"{nme}[{rmin:.0f},{rmax:.0f}] vs [{lo_lim:.0f},{hi_lim:.0f}]"
            )
    ok = not unreached
    print(
        f"  [{'PASS' if ok else 'FAIL'}] reach all limits (within {near:.0f}°): "
        + ("all joints reach both limits" if ok else "; ".join(unreached))
    )
    failures += not ok

    # 10. Mirror + deadzone via step(): centered hand → neutral; image-right → robot-left.
    m4 = CameraMimicMapper()
    jc = m4.step((320, 240), 640, 480, t=0.0)  # dead center
    m4.reset()
    jright = m4.step((620, 240), 640, 480, t=0.0)  # far image-right
    ok = abs(jc["shoulder_pan"]) < 1e-6 and jright["shoulder_pan"] < 0  # mirrored
    print(
        f"  [{'PASS' if ok else 'FAIL'}] step mirror/deadzone: center pan={jc['shoulder_pan']:+.1f}, "
        f"image-right pan={jright['shoulder_pan']:+.1f}"
    )
    failures += not ok

    print(
        f"\ngesture_map selftest: {'PASSED' if failures == 0 else f'{failures} FAILURES'}"
    )
    return failures


def main() -> None:
    """CLI entry point: run the selftest suite."""
    parser = argparse.ArgumentParser(
        description="Camera-frame → SO-101 mimic mapping - component test"
    )
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        raise SystemExit(_selftest())
    parser.print_help()


if __name__ == "__main__":
    main()
