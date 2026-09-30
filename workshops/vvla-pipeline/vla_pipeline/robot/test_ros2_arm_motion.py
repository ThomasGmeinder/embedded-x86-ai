# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Full-speed motion test for the SO-101 arm over the ROS 2 nodes.

Drives **each joint end to end** - from its calibrated minimum to its
calibrated maximum - **at full speed**. The arm always starts and finishes at
the physical rest pose (``REST_POSE``), and returns to rest between sweeps. This
is a simple node check that every ROS 2 joint node commands its servo through
its whole range - no pass/fail assertions, just visible motion.

    client.go_to() → command_topic → So101ServerNode → Feetech bus → servos

Sweep order
-----------
1. **paired proximal move (coordinated)** - shoulder_lift and elbow_flex move
   TOGETHER away from rest (shoulder_lift to 50% of its travel, elbow_flex to
   100%) and return together. Each runs its own smooth onboard move, but the
   servo speeds are scaled to the travel so both finish at the same instant and
   elbow_flex is always at 2× shoulder_lift's progress.
2. **every remaining joint, one at a time** - shoulder_pan, wrist_flex,
   wrist_roll, gripper - each swept end to end on its own, returning to its rest
   before the next.

"End of range" is the end *away from* rest: if a joint's rest angle is negative
it sweeps to its max, otherwise to its min.

Start pose
----------
The arm starts from ``REST_POSE`` - its safe parked pose. We do NOT start at the
calibration-derived midpoint.

Range
-----
Range comes from the LeRobot calibration file (``calibration_limits_and_rest``),
so each joint sweeps exactly as far as it was calibrated to travel. The hard
``JOINT_LIMITS`` in ``arm_interface`` still clamp every command as a backstop.

Full speed
----------
Moves are commanded with **no per-tick step cap** (``max_step_deg=0``): the
command goes straight at the target and the servo moves as fast as it can.
Durations are derived from travel distance at ``FULL_SPEED_DEG_PER_S``.

Calibration
-----------
Before any motion the test checks for a LeRobot calibration file. If one exists
it asks whether to recalibrate (default: keep it); if none exists, calibration
is required and runs first. See ``--recalibrate`` / ``--skip-calibration``.

By default the test spins the bus-owning server node **in-process**. If a server
is already running elsewhere (``python -m vla_pipeline.robot.robot_node
--server``), pass ``--external-server`` to attach to it instead.

Usage:

    python -m vla_pipeline.robot.test_ros2_arm_motion
    python -m vla_pipeline.robot.test_ros2_arm_motion --yes
    python -m vla_pipeline.robot.test_ros2_arm_motion --external-server

SAFETY: this moves a real arm at full speed. Keep the workspace clear, keep a
hand near the power switch. The test always parks at REST_POSE on exit
(including on Ctrl-C).
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time

from vla_pipeline.robot.arm_interface import (
    GRIPPER_CLOSED_FLOOR,
    JOINT_LIMITS,
    MOTOR_NAMES,
    REST_POSE,
    _GRIPPER_NAME,
    calibrate_arm,
    calibration_exists,
    calibration_limits_and_rest,
    calibration_path,
)
from vla_pipeline.robot.robot_node import (
    Ros2ArmClient,
    So101ServerNode,
    _require_ros2,
)
from vla_pipeline.utils.config import load_config

logger = logging.getLogger("test_ros2_arm_motion")

# Interpolation rate of go_to(). Oversampling the server control loop (tried at
# 120 Hz) did NOT smooth the sweeps - the limiter is the server's effective
# command rate, not setpoint granularity - so this stays at the control-loop
# rate. Raising it again only helps once the server loop can sustain a higher
# rate (check the PROFILE output from robot_node first).
FPS = 30.0

# Full speed. A comfortable "full" sweep speed for the SO-101 is ~90°/s. Move
# durations are derived from this so the trajectory is streamed at FPS, but with
# NO per-tick step cap the servo runs at its own full speed toward the target.
FULL_SPEED_DEG_PER_S = 90.0

# No per-tick command cap: full speed straight at the target.
FULL_SPEED_STEP_DEG = 0.0

# Choreography:
#   1. shoulder_lift and elbow_flex move TOGETHER away from rest - shoulder_lift
#      to 50% of its travel, elbow_flex to 100% - then return to rest together,
#   2. every remaining joint is then swept end to end, one at a time.
# The two big proximal joints are exercised as a paired move first; the distal
# joints are tested individually afterward.
PAIRED_JOINTS = ("shoulder_lift", "elbow_flex")

# Fraction of the rest → far-end travel each paired joint moves to.
# PAIRED_FRACTION is the default; PAIRED_FRACTIONS overrides it per joint.
PAIRED_FRACTION = 0.5
PAIRED_FRACTIONS = {"shoulder_lift": 0.5, "elbow_flex": 1.0}

# Floor on move duration so tiny moves still interpolate over several ticks.
MIN_DURATION_S = 0.4

# Pause between moves so each sweep reads as a distinct, settled motion.
SETTLE_S = 0.6

# Back the sweep extremes off the calibrated limits by this much so the arm
# stops just short of its hard stops instead of straining against them. The
# servo's own steady-state error already eats ~1-2° at a range edge; this keeps
# it from sitting there pushing (which is what heats a gravity-loaded joint).
# Units are degrees for the arm joints and the same magnitude for the gripper's
# 0-100 scale (a few % off each end).
SWEEP_MARGIN_DEG = 3.0


def _duration_for(travel_deg: float) -> float:
    """Seconds to cover ``travel_deg`` at full speed, with a sensible floor."""
    return max(abs(travel_deg) / FULL_SPEED_DEG_PER_S, MIN_DURATION_S)


def _sweep_range(
    joint: str, limits: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Reachable (lo, hi) for ``joint``: calibration range ∩ hard JOINT_LIMITS,
    inset by SWEEP_MARGIN_DEG so the sweep stops short of the hard stops."""
    cal_lo, cal_hi = limits.get(joint, JOINT_LIMITS.get(joint, (-90.0, 90.0)))
    hard_lo, hard_hi = JOINT_LIMITS.get(joint, (-180.0, 180.0))
    lo, hi = max(cal_lo, hard_lo), min(cal_hi, hard_hi)
    # Inset both ends, but never past each other (collapse to the midpoint for a
    # range narrower than 2× the margin).
    mid = (lo + hi) / 2.0
    swept_lo, swept_hi = (
        min(lo + SWEEP_MARGIN_DEG, mid),
        max(hi - SWEEP_MARGIN_DEG, mid),
    )
    # Keep the gripper clear of its closed stop. Its calibrated low end sits at the
    # mechanical stop; commanding there stalls the servo (suspect for id=6 dropping
    # off the bus), so floor its low end a few % open. Body joints are untouched.
    if joint == _GRIPPER_NAME:
        swept_lo = min(max(swept_lo, GRIPPER_CLOSED_FLOOR), swept_hi)
    return swept_lo, swept_hi


def _far_end(joint: str, limits: dict[str, tuple[float, float]]) -> float:
    """The end of ``joint``'s range that is *away from* its rest position.

    Rule: if rest is negative the far end is the maximum (hi); if rest is
    positive (or zero) the far end is the minimum (lo). So a joint resting near
    one extreme sweeps out toward the opposite extreme.
    """
    lo, hi = _sweep_range(joint, limits)
    return hi if REST_POSE.get(joint, 0.0) < 0 else lo


def _partial(joint: str, frac: float, limits: dict[str, tuple[float, float]]) -> float:
    """Angle ``frac`` of the way from ``joint``'s rest pose toward its far end."""
    rest = REST_POSE.get(joint, 0.0)
    return rest + frac * (_far_end(joint, limits) - rest)


def _move(arm: Ros2ArmClient, targets: dict[str, float]) -> None:
    """Drive one or more joints to absolute targets at full speed (no step cap).

    Duration is sized to the joint that has the farthest to travel, so a paired
    move runs until the slowest joint arrives.
    """
    current = arm.get_joints()
    travel = max((abs(t - current.get(j, t)) for j, t in targets.items()), default=0.0)
    arm.go_to(
        targets,
        duration_s=_duration_for(travel),
        fps=FPS,
        max_step_deg=FULL_SPEED_STEP_DEG,
    )
    time.sleep(SETTLE_S)


def _home(arm: Ros2ArmClient, home: dict[str, float]) -> None:
    """Bring the whole arm to the rest pose at full speed and let it settle."""
    arm.go_to(home, duration_s=2.0, fps=FPS, max_step_deg=FULL_SPEED_STEP_DEG)
    time.sleep(1.0)


def run_motion_test(
    arm: Ros2ArmClient,
    limits: dict[str, tuple[float, float]] | None = None,
    node=None,
) -> None:
    """Paired proximal move, then each remaining joint one by one.

    The arm always starts and finishes at ``REST_POSE``.

        1. ``PAIRED_JOINTS`` (shoulder_lift, elbow_flex) move TOGETHER away from
           rest - shoulder_lift to 50% of its travel, elbow_flex to 100% - and
           then return to rest together,
        2. every remaining joint is swept end to end, one at a time, returning to
           its own rest after each.

    The paired joints are coordinated by SPEED, not by streaming: each runs its
    own smooth onboard move, but the servo speeds are scaled to the travel so both
    finish at the same instant (elbow twice the travel → twice the speed → always
    at 2× the shoulder's progress). This keeps the motion smooth - streaming a
    joint trajectory fights the servo's onboard profile and comes out jittery.
    Per-joint speed is a server-side setting, so it needs ``node`` (the in-process
    So101ServerNode); without it the paired move still runs, just uncoordinated
    (both at the default speed, so the longer-travel joint finishes later).

    Full speed throughout. ``limits`` come from the calibration file and define
    how far each joint sweeps; missing entries fall back to ``JOINT_LIMITS``.
    """
    limits = limits or dict(JOINT_LIMITS)
    home = dict(REST_POSE)

    logger.info("Homing to rest pose at full speed...")
    _home(arm, home)

    # 1) Move the paired joints out, then back, coordinated so they arrive
    #    together. Speed ∝ travel (set once from the out-travel; the return has
    #    the same magnitudes), then restore the default speed afterward.
    out = {
        j: _partial(j, PAIRED_FRACTIONS.get(j, PAIRED_FRACTION), limits)
        for j in PAIRED_JOINTS
    }
    back = {j: REST_POSE.get(j, 0.0) for j in PAIRED_JOINTS}
    detail = ", ".join(
        "%s → %.1f° (%.0f%%)"
        % (j, out[j], PAIRED_FRACTIONS.get(j, PAIRED_FRACTION) * 100)
        for j in PAIRED_JOINTS
    )

    cur = arm.get_joints()
    travels = {j: abs(out[j] - cur.get(j, out[j])) for j in PAIRED_JOINTS}
    if node is not None:
        node.set_coordinated_speeds(travels)
    elif len(PAIRED_JOINTS) > 1:
        logger.warning(
            "No in-process node - paired move runs uncoordinated "
            "(both joints at default speed)."
        )
    try:
        logger.info("Paired move (coordinated, arrive together): %s", detail)
        _move(arm, out)
        logger.info(
            "  returning %s to rest (coordinated)...", " + ".join(PAIRED_JOINTS)
        )
        _move(arm, back)
    finally:
        if node is not None:
            node.reset_joint_speeds()  # restore default speed for the sweeps

    # 2) Sweep every remaining joint end to end, one at a time.
    singles = [j for j in MOTOR_NAMES if j not in PAIRED_JOINTS]
    for joint in singles:
        lo, hi = _sweep_range(joint, limits)
        rest_j = REST_POSE.get(joint, 0.0)
        logger.info("Sweeping %-13s  rest → %.1f° → %.1f° → rest", joint, lo, hi)
        _move(arm, {joint: lo})  # drive this one joint to one end
        _move(arm, {joint: hi})  # then all the way to the other end
        logger.info("  returning %s to rest before next joint...", joint)
        _move(arm, {joint: rest_j})

    _home(arm, home)  # settle whole arm at rest

    logger.info(
        "Done - %s moved together (lockstep) out and back, then %d joints one by one.",
        " + ".join(PAIRED_JOINTS),
        len(singles),
    )


def _spin_server_in_process(cfg: dict):
    """Start a real-hardware So101ServerNode on a background executor.

    Returns (node, executor, thread) so the caller can tear it down.
    """
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    if not rclpy.ok():
        rclpy.init()

    node = So101ServerNode(cfg, dry_run=False)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    # Give the control loop a beat to publish its first state message so the
    # client's startup timeout has something to latch onto.
    time.sleep(0.5)
    return node, executor, thread


def _ensure_calibration(
    cfg: dict, recalibrate: bool = False, skip: bool = False
) -> bool:
    """Calibration gate run before any motion.

    Returns False if calibration was needed but couldn't be completed.
    """
    robot_id = cfg["robot"]["robot_id"]
    have = calibration_exists(robot_id)

    if have:
        path = calibration_path(robot_id)
        if skip:
            logger.info("Using existing calibration: %s", path)
            return True
        if recalibrate:
            logger.info(
                "Recalibration forced - redoing calibration for '%s'.", robot_id
            )
            calibrate_arm(cfg, force=True)
            return True
        print(f"\n  A calibration file already exists for '{robot_id}':\n    {path}")
        ans = input("  Recalibrate before testing? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            calibrate_arm(cfg, force=True)
        else:
            logger.info("Keeping existing calibration.")
        return True

    if skip:
        logger.warning(
            "No calibration file for '%s' and --skip-calibration set - "
            "joint angles may be meaningless. Proceeding anyway.",
            robot_id,
        )
        return True
    if not sys.stdin.isatty():
        logger.error(
            "No calibration for '%s' and no interactive terminal to run "
            "it. Calibrate first in a foreground shell:\n"
            "    python -m vla_pipeline.robot.arm_interface --calibrate",
            robot_id,
        )
        return False
    print(
        f"\n  No calibration found for '{robot_id}'. Calibration is required "
        "before moving the arm."
    )
    try:
        calibrate_arm(cfg)
    except Exception as e:
        logger.error("Calibration failed: %s", e)
        return False
    return calibration_exists(robot_id)


def main() -> None:
    """CLI entry point: calibrate if needed, confirm with the user, then run the full-speed motion sweep."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Full-speed SO-101 motion test over the ROS 2 nodes: first a coordinated shoulder_lift+elbow_flex move, then every remaining joint swept end to end one at a time."
    )
    parser.add_argument("--config", default=None, help="path to pipeline config")
    parser.add_argument(
        "--external-server",
        action="store_true",
        help="attach to a server already running elsewhere instead of spinning one",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="skip the 'this moves real hardware' confirmation prompt",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="force recalibration even if a calibration file exists "
        "(skips the interactive prompt and just recalibrates)",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="never calibrate, even if no calibration file exists "
        "(not recommended - angles may be meaningless)",
    )
    args = parser.parse_args()

    _require_ros2()
    cfg = load_config(args.config)

    if not _ensure_calibration(
        cfg, recalibrate=args.recalibrate, skip=args.skip_calibration
    ):
        sys.exit(1)

    if not args.yes:
        print("\n  WARNING: This test moves the PHYSICAL SO-101 arm at FULL SPEED.")
        print("     Clear the workspace and keep a hand near the power switch.")
        if input("     Type 'go' to continue: ").strip().lower() != "go":
            print("Aborted.")
            sys.exit(1)

    import rclpy

    server = executor = spin_thread = None
    arm = None
    try:
        if not args.external_server:
            logger.info("Spinning So101ServerNode in-process (real hardware)...")
            server, executor, spin_thread = _spin_server_in_process(cfg)
        else:
            logger.info("Using external server (expecting --server running elsewhere).")
            if not rclpy.ok():
                rclpy.init()

        arm = Ros2ArmClient(cfg)
        limits, _rest, _meta = calibration_limits_and_rest(cfg["robot"]["robot_id"])
        if limits:
            logger.info(
                "Using calibration-derived sweep range for %d joints.", len(limits)
            )
        else:
            logger.warning(
                "No calibration limits available - falling back to "
                "hard-coded JOINT_LIMITS."
            )
        run_motion_test(arm, limits=limits or None, node=server)

    except KeyboardInterrupt:
        logger.warning("Interrupted -- parking arm and shutting down.")
    finally:
        if arm is not None:
            try:
                arm.go_to_rest(duration_s=3.0)
            except Exception as e:
                logger.error("Rest-park via client failed: %s", e)
            try:
                arm.close()
            except Exception as e:
                logger.error("Client close failed: %s", e)

        if server is not None:
            try:
                if executor is not None:
                    executor.shutdown()
                server.destroy_node()
            except Exception as e:
                logger.error("Server shutdown failed: %s", e)

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

    print("\nFull-speed motion test complete.")


if __name__ == "__main__":
    main()
