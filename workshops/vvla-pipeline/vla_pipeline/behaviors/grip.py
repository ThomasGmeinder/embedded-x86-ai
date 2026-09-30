# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Grip behavior: take an object from the user, grasp it, and lock the grip.

What changed (and why the pen used to slip)
-------------------------------------------
The old close loop terminated as soon as the *commanded* position reached 0°.
A pen stalls the jaws at only ~3-6°, and with ``close_step_deg: 1.5`` the
command races past that to 0° in a couple of ticks - the loop could exit
before ``stall_ticks`` consecutive stalled samples accumulated, log "no
object detected", then relax the gripper to 30° and drop the pen. Fixes:

1. **Grace window at full close** - after the command reaches 0° we keep
   sampling the *measured* position for ``stall_grace_s``; a thin object
   shows up as a persistent measured-vs-commanded gap and is now detected.
2. **Effort/current feedback when available** - ``ArmClient.get_gripper_load()``
   (runtime-detected on the Feetech bus) gives direct contact sensing:
   sustained load above ``load_threshold`` stops the close immediately, which
   is gentler and faster than position stall. Falls back automatically.
3. **Arm-camera verification** - a reference frame of the empty jaws is taken
   while presenting; after the grasp, a jaw-ROI difference confirms an object
   is actually held. Result is reported via the feedback callback (spoken /
   logged) and returned to the caller.

While ``grip_lock`` is set, gesture-mimic holds that gripper position, so a
gripped pencil stays gripped while the arm mirrors the human - e.g. to write.
Saying "grip" again releases.

Independent test
----------------
    python -m vla_pipeline.behaviors.grip --dry-run            # simulated 22° object
    python -m vla_pipeline.behaviors.grip --dry-run --thin     # simulated 4° pen
    python -m vla_pipeline.behaviors.grip                      # real arm
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vla_pipeline.robot.arm_interface import ArmClient, DryRunArm, make_arm
from vla_pipeline.utils.config import load_config

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


# =============================================================================
# Arm-camera jaw verification (model-free: reference-frame difference)
# =============================================================================


def jaw_roi_pixels(frame: np.ndarray, roi_frac) -> np.ndarray:
    """Crop frame to the jaw ROI (fractional coordinates)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = roi_frac
    return frame[int(y1 * h) : int(y2 * h), int(x1 * w) : int(x2 * w)]


def object_in_jaws(
    reference: np.ndarray, frame: np.ndarray, roi_frac, diff_threshold: float
) -> bool:
    """True if the jaw ROI differs significantly from the empty reference."""
    import cv2

    a = cv2.cvtColor(jaw_roi_pixels(reference, roi_frac), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(jaw_roi_pixels(frame, roi_frac), cv2.COLOR_BGR2GRAY)
    if a.shape != b.shape or a.size == 0:
        return False
    diff = (
        float(
            np.mean(
                cv2.absdiff(
                    cv2.GaussianBlur(a, (5, 5), 0), cv2.GaussianBlur(b, (5, 5), 0)
                )
            )
        )
        / 255.0
    )
    logger.debug("[grip] jaw ROI diff %.3f (threshold %.3f)", diff, diff_threshold)
    return diff >= diff_threshold


# =============================================================================
# Feedback-based close (shared with pick_place)
# =============================================================================


def close_with_feedback(
    arm: ArmClient, g: dict, stop_check=None, start_from: float | None = None
) -> tuple[bool, float]:
    """Close the gripper until contact. Returns ``(gripped, hold_position)``.

    Contact = sustained effort above ``load_threshold`` (when the transport
    provides effort feedback) OR measured-vs-commanded position stall - with
    a grace window after full close so thin objects are still detected.
    """
    step = float(g["close_step_deg"])
    eps = float(g["stall_eps_deg"])
    need_ticks = int(g["stall_ticks"])
    extra = float(g["settle_extra_deg"])
    load_threshold = float(g.get("load_threshold", 120.0))
    load_ticks_needed = int(g.get("load_ticks", 3))
    grace_s = float(g.get("stall_grace_s", 0.6))
    tick_s = 0.05

    use_load = arm.get_gripper_load() is not None
    logger.info(
        "[grip] closing with %s feedback...",
        "effort/current" if use_load else "position-stall",
    )

    commanded = (
        start_from
        if start_from is not None
        else float(arm.get_joints().get("gripper", PRESENT_POSE["gripper"]))
    )
    stall_count = load_count = 0
    gripped = False
    grace_deadline: float | None = None

    while True:
        if stop_check and stop_check():
            break
        if commanded > 0.0:
            commanded = max(commanded - step, 0.0)
            arm.send_joints({"gripper": commanded})
            if commanded == 0.0:
                grace_deadline = time.time() + grace_s
        elif grace_deadline is not None and time.time() >= grace_deadline:
            break  # fully closed, grace expired → nothing between the jaws
        time.sleep(tick_s)

        if use_load:
            load = arm.get_gripper_load()
            if load is None:  # feedback died mid-grasp → fall back
                use_load = False
            elif load >= load_threshold:
                load_count += 1
                if load_count >= load_ticks_needed:
                    gripped = True
                    break
            else:
                load_count = 0

        measured = float(arm.get_joints().get("gripper", commanded))
        if measured - commanded > eps:
            stall_count += 1
            if stall_count >= need_ticks:
                gripped = True
                break
        else:
            stall_count = 0

    if not gripped:
        return False, 0.0
    # Hold slightly tighter than where contact happened (bounded at closed).
    measured = float(arm.get_joints().get("gripper", commanded))
    hold = max(min(measured, commanded) - extra, 0.0)
    arm.send_joints({"gripper": hold})
    return True, hold


# =============================================================================
# Behavior
# =============================================================================


def run_grip(
    cfg: dict,
    arm: ArmClient,
    shared_state: dict | None = None,
    stop_check=None,
    wait_s: float = 4.0,
    arm_camera=None,
    feedback=None,
) -> bool:
    """Run the grip sequence. Returns True if an object was captured.

    ``arm_camera``: optional :class:`~vla_pipeline.vision.camera.CameraClient`
    for visual before/after verification. ``feedback``: optional ``str →``
    callable for spoken/UI status (defaults to logging).
    """
    g = cfg["behaviors"]["grip"]
    shared_state = shared_state if shared_state is not None else {}
    say = feedback or (lambda m: logger.info("[grip] %s", m))
    roi = g.get("jaw_roi", [0.30, 0.45, 0.70, 0.95])
    diff_thr = float(g.get("jaw_diff_threshold", 0.06))

    # A repeated "grip" while locked = release.
    if shared_state.get("grip_lock"):
        say("Releasing the object.")
        shared_state.pop("grip_lock", None)
        shared_state.pop("grip_lock_pos", None)
        arm.go_to(
            {**PRESENT_POSE, "gripper": 80.0}, duration_s=2.0, stop_check=stop_check
        )
        time.sleep(1.5)
        arm.go_to_rest()
        return False

    say("Place the object between the jaws.")
    arm.go_to(PRESENT_POSE, duration_s=2.5, stop_check=stop_check)

    # Empty-jaws reference for visual verification (before the user reaches in).
    reference = None
    if arm_camera is not None:
        ok, ref = arm_camera.read()
        reference = ref if ok else None

    for remaining in range(int(wait_s), 0, -1):
        if stop_check and stop_check():
            arm.go_to_rest()
            return False
        logger.info("[grip] closing in %d...", remaining)
        time.sleep(1.0)

    # Optional pre-check: is anything visibly between the jaws yet?
    if reference is not None:
        ok, now = arm_camera.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            say("I don't see anything between the jaws yet - closing carefully.")

    gripped, hold = close_with_feedback(arm, g, stop_check=stop_check)

    # Visual confirmation after the grasp.
    if gripped and reference is not None:
        ok, now = arm_camera.read()
        if ok and not object_in_jaws(reference, now, roi, diff_thr):
            logger.warning(
                "[grip] contact detected but camera sees empty jaws - "
                "reporting failure."
            )
            gripped = False

    if gripped:
        shared_state["grip_lock"] = True
        shared_state["grip_lock_pos"] = hold
        say(
            f"Object gripped at {hold:.1f} degrees - grip locked. "
            "Say 'gesture mimic' to use it; say 'grip' again to release."
        )
    else:
        say("No object detected.")
        arm.go_to({"gripper": 30.0}, duration_s=0.8)
    return gripped


# =============================================================================
# Standalone component test
# =============================================================================


class _SimulatedObjectArm(DryRunArm):
    """Dry-run arm whose gripper stalls at ``OBJECT_AT`` - emulates an object."""

    OBJECT_AT = 22.0

    def get_joints(self) -> dict[str, float]:
        """Return simulated joints, with gripper floored at OBJECT_AT to emulate a stalled grasp."""
        joints = super().get_joints()
        joints["gripper"] = max(joints["gripper"], self.OBJECT_AT)
        return joints


class _SimulatedThinObjectArm(_SimulatedObjectArm):
    """A pen: stalls at 4° - the case the old loop-exit logic dropped."""

    OBJECT_AT = 4.0


class _SimulatedLoadArm(_SimulatedObjectArm):
    """Object at 15° that also reports rising servo current near contact."""

    OBJECT_AT = 15.0

    def get_gripper_load(self) -> float | None:
        """Return a simulated effort reading that rises once the gripper stalls against the object."""
        measured = self.get_joints()["gripper"]
        target = self._joints["gripper"]
        return 40.0 + 80.0 * max(measured - target, 0.0)  # grows once stalled


def main() -> None:
    """CLI entry point: run the grip behavior standalone, optionally against a simulated object."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Grip behavior - component test")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--dry-run", action="store_true", help="simulate an object stalling the gripper"
    )
    parser.add_argument(
        "--thin", action="store_true", help="dry-run: simulate a pen (4° stall)"
    )
    parser.add_argument(
        "--load", action="store_true", help="dry-run: simulate effort feedback"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dry_run:
        arm: ArmClient = (
            _SimulatedLoadArm()
            if args.load
            else _SimulatedThinObjectArm()
            if args.thin
            else _SimulatedObjectArm()
        )
    else:
        arm = make_arm(cfg)
    state: dict = {}
    try:
        gripped = run_grip(
            cfg, arm, shared_state=state, wait_s=2.0 if args.dry_run else 4.0
        )
        print(
            f"\ngrip {'PASSED (object locked)' if gripped else 'completed (no object)'}"
        )
        print(f"shared state: {state}")
        if args.dry_run and not gripped:
            raise SystemExit(1)
    finally:
        arm.close()


if __name__ == "__main__":
    main()
