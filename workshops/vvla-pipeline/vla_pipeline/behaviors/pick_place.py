# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Voice-grounded pick-and-place: "pick up the ball" → find it → grasp →
lift it clear → hold it out. The jaws stay closed.

Pipeline
--------
1. **GROUND** - the object name parsed from the voice command
   (``ParsedCommand.object_name``) is resolved to a COCO class
   (:func:`vla_pipeline.vision.yolo_detect_npu.resolve_class`). Unknown names
   fail fast with a spoken explanation instead of guessing.
2. **SCAN** - walk ``scan_pans`` until one settled frame holds a usable box.
   Only a sweep that stored nothing continues at the near pose. Each stored
   frame is captured after the move and kept with the joints that took it.
3. **CHOOSE** - keep boxes ``locate_on_table`` can accept, rank them by area
   then earliest pan, and locate the first one that fixes on the table.
4. **REACH** (``reach: ik``) - one ``solve_grasp`` from that fix. The jaws
   stay open. The hand moves to the pose just above the object, then down
   the vertical, staying within 5 mm of it. The gripper closes only after
   that descent. An object with no solution stops the pick. ``reach: fixed``,
   or the operator camera, uses the taught ``approach_pose`` +
   ``descend_lift`` at the pan the scan stopped on.
   The camera model is ``cameras.arm.intrinsics`` / ``mount``
   (``scripts/calibrate_arm_camera.py``), the twin's until calibrated.
5. **GRASP** - close with the same feedback-based loop the grip behavior
   uses (:func:`vla_pipeline.behaviors.grip.close_with_feedback`):
   step closed at ``Torque_Limit`` 500 until the current crosses
   ``load_threshold`` (with no current, until the jaws stall), and keep the
   goal ``hold_past_contact`` below the jaw while both fingers seat. The
   limit drops to ``hold_torque_limit`` after the first lift step: on the
   table the hold force lets the object settle against one jaw, and the
   lift would leave it behind. The goal stays inside the object, so the
   torque limit is the grip force; a goal at the measured angle would let
   go.
6. **OFFER** - lift the grasp point above the other props and hold the
   object out toward the operator. The jaws stay closed, so the object can
   be taken or left. "release", "open", or "drop it" opens the jaws.

Failure handling: every stage reports through the ``feedback`` callback
("I can't see a ball", "I see two cups, taking the clearest one", "the grasp
slipped, trying once more") and the behavior retries the grasp
``retries`` times from a fresh frame at the pre-grasp before going home. It returns False
whenever it gives up, so the dispatcher reports the command as failed.

Independent test
----------------
    python -m vla_pipeline.behaviors.pick_place --object ball --dry-run-arm
    python -m vla_pipeline.behaviors.pick_place --object "cell phone"
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from vla_pipeline.behaviors.fetch_block import park_open
from vla_pipeline.behaviors.grip import close_with_feedback, ease_grip, holding
from vla_pipeline.robot.arm_interface import (
    GRIPPER_CLOSED_FLOOR,
    GRIPPER_TORQUE_LIMIT,
    JOINT_LIMITS,
    ArmClient,
    DryRunArm,
    make_arm,
)
from vla_pipeline.utils.config import load_config
from vla_pipeline.vision.yolo_detect_npu import ObjectDetector

logger = logging.getLogger(__name__)

# Grasp point high enough that a carried ball (its centre is the grasp
# point) clears the bottle, the tallest prop on the table.
_CARRY_Z = 0.12
# Out toward the operator, who stands in front of the arm.
_OFFER_XY = (0.30, 0.0)
_OFFER_ERR_M = 0.015


def _log_grip(arm, label: str, hold: float, plan=None) -> None:
    """Log the jaw against its held goal, the current, how far the hand landed
    from ``plan``'s grasp pose, and what the fingers touch (the twin only)."""
    here = arm.get_joints()
    jaw = float(here.get("gripper", hold))
    load = arm.get_gripper_load()
    parts = [
        f"jaw {jaw:.1f}, goal {hold:.1f} (margin {jaw - hold:+.1f})",
        "current n/a" if load is None else f"current {load:.0f}",
    ]
    if plan is not None:
        try:
            from so101_sim.kinematics import shared_kinematics

            kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
            want = {**here, **plan.grasp}
            got = {**here, "gripper": want["gripper"]}
            off = kin.world.grasp_point(got, 0.04) - kin.world.grasp_point(want, 0.04)
            parts.append(f"hand {1000 * float(np.linalg.norm(off)):.1f} mm from the planned grasp")
        except Exception as e:
            logger.debug("grasp offset unavailable: %s", e)
    report = arm.grip_report()
    if report:
        parts.append(report)
    logger.info("%s: %s", label, "; ".join(parts))


def _offer(arm, hold, stop_check, limits, say, resolved, still_holding, ease=lambda: None) -> str:
    """Lift above the other props and hold the object out. The jaws stay closed.

    Two joint-space legs from the pose that is holding the object, keeping
    its wrist roll. ``ease()`` runs after the first step, once the object
    hangs from the fingers. Returns ``offered``, ``dropped``, ``stopped``, or
    ``unreachable``.
    """
    try:
        from so101_sim.kinematics import joint_line, shared_kinematics
    except Exception as e:
        logger.warning("offer unavailable: %s", e)
        return "unreachable"

    def stopped() -> bool:
        return bool(stop_check and stop_check())

    pending = [ease]

    def follow(steps, duration_s: float) -> bool:
        for step in steps:
            arm.go_to(step, duration_s=duration_s / len(steps), stop_check=stop_check)
            while pending:
                pending.pop()()
            if stopped():
                return False
        return True

    kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    here = arm.get_joints()
    roll = float(here.get("wrist_roll", 0.0))
    origin = kin.world.grasp_point(here, 0.04)
    targets = (
        np.array([origin[0], origin[1], _CARRY_Z]),
        np.array([_OFFER_XY[0], _OFFER_XY[1], _CARRY_Z]),
    )
    logger.info(
        "offer: lift to %.2f m, then out to (%.2f, %.2f)",
        _CARRY_Z, _OFFER_XY[0], _OFFER_XY[1],
    )
    seed = here
    for leg, (target, duration_s) in enumerate(zip(targets, (2.5, 2.0)), 1):
        pose, err, _tilt = kin.ik(
            target, 0.04, hold, seed, roll, limits=limits, one_branch=True,
        )
        if err > _OFFER_ERR_M:
            logger.info("offer: no pose within 15 mm (err %.0f mm)", 1000 * err)
            return "unreachable"
        # From the measured jaw, the first steps would let go of the squeeze.
        if not follow(joint_line({**seed, "gripper": hold}, {**pose, "gripper": hold}), duration_s):
            return "stopped"
        time.sleep(0.3)
        _log_grip(arm, f"offer check {leg}", hold)
        if not still_holding():
            return "dropped"
        seed = arm.get_joints()
    say(f"Got the {resolved}.")
    return "offered"


_DESCENT_LATERAL_M = 0.005


def _read_settled(camera, settled_at: float):
    """One frame captured after the arm finished moving at ``settled_at``."""
    read_after = getattr(camera, "read_after", None)
    if callable(read_after):
        return read_after(settled_at)
    try:
        return camera.read(fresh=True)
    except TypeError:
        pass
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


def _shape_of(resolved: str) -> str:
    from so101_sim.kinematics import FLAT_CLASSES, ROUND_CLASSES

    if resolved in ROUND_CLASSES:
        return "sphere"
    if resolved in FLAT_CLASSES:
        return "flat"
    if resolved == "bottle":
        return "bottle"
    return "upright"


def _box_usable(box, shape: str, width: int, height: int) -> bool:
    """True when ``locate_on_table`` will accept this box's edges."""
    x1, y1, x2, y2 = (float(v) for v in box)
    if x1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return False
    if shape == "flat" and y1 <= 1:
        return False
    return True


def _rank_views(views: list) -> list:
    """Largest usable box first, then the earliest pan."""
    return sorted(views, key=lambda view: (-view["area"], view["order"]))


class _GraspPlan:
    def __init__(self, poses, xyz, beside=None):
        self.poses = poses
        self.xyz = np.asarray(xyz, float)
        self.beside = beside  # the last pose moved sideways to the fixed finger's gap, or None

    @property
    def grasp(self) -> dict:
        """The pose the jaw closes at."""
        return self.beside if self.beside is not None else self.poses[-1]


def _log_sighting(pp, cam, frame, det, resolved, joints) -> None:
    """Log one detection with its table fix and the joints that took the frame."""
    height, width = frame.shape[:2]
    shape = _shape_of(resolved)
    if det.area / float(height * width) < float(pp["min_box_area_frac"]):
        keep = "too small"
    elif not _box_usable(det.box, shape, width, height):
        keep = "clipped"
    else:
        keep = "kept"
    fix = "no table fix"
    try:
        from so101_sim.kinematics import shared_kinematics

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
        K, dist = kin.intrinsics((cam or {}).get("intrinsics"), width, height)
        loc = kin.locate_on_table(
            det.box, joints, K, dist, (cam or {}).get("mount"),
            float(pp.get("table_z", -0.005)), (width, height), shape=shape,
        )
        if loc is not None:
            fix = "xyz (%.3f, %.3f, %.3f) m, %.0f mm wide, %.0f mm tall" % (
                *loc["xyz"], 1000 * loc["width"], 1000 * loc["height"],
            )
    except Exception as exc:
        fix = f"locate failed: {exc}"
    logger.info(
        "scan sighting: %s box %s score %.2f %s | %s | joints pan %.1f lift %.1f elbow %.1f flex %.1f roll %.1f",
        resolved, tuple(round(float(v), 1) for v in det.box), float(det.score), keep, fix,
        float(joints.get("shoulder_pan", 0.0)), float(joints.get("shoulder_lift", 0.0)),
        float(joints.get("elbow_flex", 0.0)), float(joints.get("wrist_flex", 0.0)),
        float(joints.get("wrist_roll", 0.0)),
    )


def _sweep_views(arm, camera, find, name, pose, pans, pp, stop_check, *, near=False, first=True, cam=None, accept=None):
    """Walk the pans until one settled frame can be grasped.

    A miss, a box cut by the image edge, or a box whose grasp does not solve
    does not end the sweep. ``accept(batch)`` returns true when that pan's
    boxes produced a grasp. With no ``accept``, the first usable box stops
    the sweep. Returns ``(views, status, saw_small, saw_large)``.
    """
    views, saw_small, saw_large = [], False, False
    min_frac = float(pp["min_box_area_frac"])
    for i, pan in enumerate(pans):
        arm.go_to(
            {**pose, "shoulder_pan": float(pan)},
            duration_s=2.5 if first and i == 0 else 1.2,
            stop_check=stop_check,
        )
        if stop_check and stop_check():
            return views, "stopped", saw_small, saw_large
        logger.info("scan near pan %.1f" if near else "scan pan %.1f", float(pan))
        settled_at = time.time()
        ok, frame = _read_settled(camera, settled_at)
        joints = dict(arm.get_joints())
        if not ok or frame is None:
            continue
        dets, resolved = find(frame, name)
        if resolved is None:
            return views, "unknown", saw_small, saw_large
        if not dets:
            continue
        shape = _shape_of(resolved)
        height, width = frame.shape[:2]
        frame_area = float(height * width)
        batch = []
        for det in dets:
            _log_sighting(pp, cam, frame, det, resolved, joints)
            if det.area / frame_area < min_frac:
                saw_small = True
                continue
            saw_large = True
            if not _box_usable(det.box, shape, width, height):
                continue
            batch.append({
                "frame": frame,
                "det": det,
                "dets": list(dets),
                "resolved": resolved,
                "joints": joints,
                "area": float(det.area),
                "order": i,
                "settled_at": settled_at,
            })
        if not batch:
            continue
        if accept is None or accept(batch):
            views.extend(batch)
            break
        logger.info("scan sighting not graspable at pan %.1f, continuing", float(pan))
    return views, "ok", saw_small, saw_large


def _search_views(arm, camera, find, name, pp, stop_check, arm_camera, cam=None, accept=None):
    """High sweep first. The near pose runs only when that sweep stored nothing."""
    base = dict(pp["scan_pose"])
    pans = pp.get("scan_pans") if camera is arm_camera else None
    pans = list(pans or [base.get("shoulder_pan", 0.0)])
    views, status, saw_small, saw_large = _sweep_views(
        arm, camera, find, name, base, pans, pp, stop_check, cam=cam, accept=accept,
    )
    if status != "ok" or views or camera is not arm_camera or not pp.get("scan_near_pose"):
        return views, status, saw_small, saw_large
    near = dict(pp["scan_near_pose"])
    near_pans = list(pp.get("scan_near_pans") or [near.get("shoulder_pan", 0.0)])
    more, status, small, large = _sweep_views(
        arm, camera, find, name, near, near_pans, pp, stop_check,
        near=True, first=False, cam=cam, accept=accept,
    )
    return views + more, status, saw_small or small, saw_large or large


def _path_within(kin, pre, path, width, table_z) -> bool:
    top = kin.world.grasp_point(pre, width)
    for pose in path:
        point = kin.world.grasp_point(pose, width)
        if float(np.linalg.norm(point[:2] - top[:2])) > _DESCENT_LATERAL_M:
            return False
        if kin.lowest_point(pose) < table_z:
            return False
    return True


def _descent_poses(kin, pre, grasp, width, table_z, limits=None):
    """Joint-space steps from ``pre`` to ``grasp``.

    When those steps leave the vertical by more than 5 mm, use the Cartesian
    waypoints so the bound still holds. ``None`` when neither path stays above
    the table inside that band.
    """
    from so101_sim.kinematics import joint_line

    path = joint_line(pre, grasp)
    if _path_within(kin, pre, path, width, table_z):
        return path
    cart = kin.waypoints(pre, grasp, width, limits=limits)
    if cart and len(cart) >= 2 and _path_within(kin, pre, cart, width, table_z):
        logger.info("IK reach: joint line leaves the 5 mm band, descending on Cartesian steps")
        return cart
    logger.info("IK reach: descent leaves the vertical")
    return None


def _plan_from_view(cfg: dict, view):
    """One table locate and one grasp solve from a stored frame. ``None`` if either fails."""
    pp = cfg["behaviors"]["pick_place"]
    try:
        from so101_sim.kinematics import principal_axis_in_box, shared_kinematics

        kin = shared_kinematics(GRIPPER_CLOSED_FLOOR)
    except Exception as e:
        logger.warning("IK reach unavailable: %s", e)
        return None
    frame, det, resolved, joints = view["frame"], view["det"], view["resolved"], view["joints"]
    height, width = frame.shape[:2]
    cam = cfg["cameras"]["arm"]
    K, dist = kin.intrinsics(cam.get("intrinsics"), width, height)
    table_z = float(pp.get("table_z", -0.005))
    shape = _shape_of(resolved)
    loc = kin.locate_on_table(
        det.box, joints, K, dist, cam.get("mount"), table_z, (width, height), shape=shape,
    )
    if loc is None:
        logger.info("IK reach: no table fix for the %s box %s", resolved, det.box)
        return None
    logger.info("IK locate: %s", resolved)
    p = loc["xyz"].copy()
    p[2] = min(p[2], table_z + float(pp.get("max_grasp_height_m", 0.03)))
    obstacle = (
        (*loc["xyz"][:2], 0.5 * loc["width"], table_z + loc["height"])
        if shape in ("bottle", "upright")
        else None
    )
    oriented = resolved in pp.get("wrist_align_classes", [])
    object_axis = None
    if oriented:
        axis = principal_axis_in_box(
            frame, det.box, min_elongation=float(pp.get("wrist_axis_min_elongation", 1.6)),
        )
        if axis is not None:
            found = kin.table_axis(
                axis["center"], axis["direction"], axis["length_px"], joints, K, dist,
                cam.get("mount"), table_z,
            )
            if found is not None:
                centre, object_axis = found
                offset = (
                    float(pp.get("wrist_align_center_offset_m", 0.018))
                    if abs(float(joints.get("wrist_roll", 0.0))) < 30.0
                    else 0.0
                )
                p[:2] = centre[:2] + np.asarray(object_axis[:2], float) * offset
        p[2] = max(p[2], table_z + 0.011)
    grasp_width = (
        float(pp.get("wrist_align_grasp_width_m", 0.024)) if oriented else loc["width"]
    )
    above_m = float(
        pp.get("wrist_align_pregrasp_above_m", 0.10) if oriented else pp.get("pregrasp_above_m", 0.06)
    )
    # The pipeline arm's backstop is JOINT_LIMITS. A standing or oriented
    # grasp is only collision-free inside that envelope.
    limits = JOINT_LIMITS if oriented or obstacle else None
    logger.info("IK solve")
    solved = kin.solve_grasp(
        p, above_m=above_m, width=grasp_width, table_z=table_z, wrist_roll=0.0,
        limits=limits, obstacle=obstacle, object_axis=object_axis,
    )
    if solved is not None and oriented and abs(float(solved[1]["wrist_roll"])) >= 30.0:
        jaw = kin.world.jaw_axis(solved[1])
        p = p + jaw * float(pp.get("wrist_align_fixed_jaw_bias_m", 0.024))
        logger.info("IK solve")
        biased = kin.solve_grasp(
            p, above_m=above_m, width=grasp_width, table_z=table_z,
            wrist_roll=float(solved[1]["wrist_roll"]), limits=limits,
        )
        if biased is not None:
            solved = biased
    logger.info(
        "IK reach: %s at (%.3f, %.3f, %.3f) m, %.0f mm wide -> %s",
        resolved, *p, 1000 * grasp_width, "solved" if solved else "out of reach",
    )
    if solved is None:
        return None
    pregrasp, grasp = solved
    descent = _descent_poses(kin, pregrasp, grasp, grasp_width, table_z, limits)
    if descent is None:
        return None
    beside = None
    if not (oriented or obstacle):
        beside = kin.grasp_beside(
            grasp, p, float(pp.get("fixed_jaw_gap_m", 0.005)), grasp_width, table_z=table_z, limits=limits,
        )
    return _GraspPlan([pregrasp, *descent], p, beside)


def _view_here(arm, camera, find, name, pp, cam=None):
    """One fresh frame at the pose the arm is already holding, or ``None``."""
    settled_at = time.time()
    ok, frame = _read_settled(camera, settled_at)
    if not ok or frame is None:
        return None
    joints = dict(arm.get_joints())
    dets, resolved = find(frame, name)
    if resolved is None or not dets:
        return None
    for det in dets:
        _log_sighting(pp, cam, frame, det, resolved, joints)
    shape = _shape_of(resolved)
    height, width = frame.shape[:2]
    frame_area = float(height * width)
    min_frac = float(pp["min_box_area_frac"])
    usable = [
        det for det in dets
        if det.area / frame_area >= min_frac and _box_usable(det.box, shape, width, height)
    ]
    if not usable:
        return None
    det = max(usable, key=lambda item: item.area)
    return {
        "frame": frame, "det": det, "dets": list(dets), "resolved": resolved,
        "joints": joints, "area": float(det.area), "order": 0, "settled_at": settled_at,
    }


def _follow(arm, poses, stop_check, duration_s: float) -> bool:
    if not poses:
        return True
    step = duration_s / len(poses)
    for pose in poses:
        arm.go_to(pose, duration_s=step, stop_check=stop_check)
        if stop_check and stop_check():
            return False
    return True


def _execute_plan(arm, plan, stop_check) -> bool:
    """Move to the pose just above the object, then straight down, then
    sideways. Each leg starts once the arm has arrived at the end of the last,
    so a lagging joint does not cut the corner onto the object. The jaw stays
    open."""
    from so101_sim.kinematics import joint_line

    legs = [(joint_line(arm.get_joints(), plan.poses[0]), 1.5), (plan.poses[1:], 1.5)]
    if plan.beside is not None:
        legs.append(([plan.beside], 0.5))
    for poses, duration_s in legs:
        if not _follow(arm, poses, stop_check, duration_s):
            return False
        # 0.5° is about 2 mm at the jaws. With the default 1° of progress the
        # last degree never counts as moving, and the goal is resent until the timeout.
        arm.move_to(poses[-1], tolerance_deg=0.5, progress_eps_deg=0.1, stop_check=stop_check)
        if stop_check and stop_check():
            return False
    return True


def _return_along(arm, plan, stop_check) -> bool:
    """Walk the descent back to the pre-grasp. The gripper is already open."""
    back = list(reversed(plan.poses))
    return _follow(arm, back if plan.beside is not None else back[1:], stop_check, 1.5)


def run_pick_place(
    cfg: dict,
    arm: ArmClient,
    detector: ObjectDetector,
    pose,  # PoseEstimator | None - kept for the dispatcher; the offer does not search for a person
    arm_camera,  # CameraClient | None
    operator_camera,  # CameraClient | None
    object_name: str,
    shared_state: dict | None = None,
    stop_check=None,
    feedback=None,
) -> bool:
    """Pick the named object and hold it out. The jaws stay closed."""
    pp = cfg["behaviors"]["pick_place"]
    g = cfg["behaviors"]["grip"]
    say = feedback or (lambda m: logger.info("[pick] %s", m))
    shared_state = shared_state if shared_state is not None else {}

    def stopped() -> bool:
        """Return True if a stop has been requested."""
        return bool(stop_check and stop_check())

    if not object_name:
        say("What should I pick up? Try: pick up the ball.")
        return False

    camera = arm_camera if (arm_camera and arm_camera.ok()) else operator_camera
    if camera is None or not camera.ok():
        say("I can't see anything - both cameras are offline.")
        return False
    if camera is not arm_camera:
        say("Arm camera is offline - using the operator camera (coarser aim).")

    say(f"Looking for a {object_name}.")
    cams = cfg.get("cameras") or {}
    cam = cams.get("arm") if camera is arm_camera else cams.get("operator")
    use_ik = pp.get("reach", "fixed") == "ik" and camera is arm_camera
    plan = None
    saw_ungraspable = False

    def accept(batch):
        nonlocal plan, saw_ungraspable
        saw_ungraspable = True
        for view in _rank_views(batch):
            plan = _plan_from_view(cfg, view)
            if plan is not None:
                return True
        plan = None
        return False

    views, status, saw_small, saw_large = _search_views(
        arm, camera, detector.find, object_name, pp, stop_check, arm_camera, cam,
        accept if use_ik else None,
    )
    if status == "stopped" or stopped():
        return False
    if status == "unknown":
        say(f"Sorry - I don't know what a {object_name} looks like.")
        arm.go_to_rest()
        return False
    if not views:
        if saw_ungraspable:
            say(f"The {object_name} is out of reach.")
        elif saw_small and not saw_large:
            say(f"The {object_name} I can see is too small to grab safely.")
        else:
            say(f"I can't see a {object_name} in the workspace.")
        arm.go_to_rest()
        return False
    ranked = _rank_views(views)
    winner = ranked[0]
    resolved = winner["resolved"]
    by_score = sorted(winner["dets"], key=lambda det: det.score, reverse=True)
    if len(by_score) > 1 and by_score[0].score / max(by_score[1].score, 1e-6) < float(pp["ambiguous_ratio"]):
        say(f"I see {len(by_score)} of those - taking the clearest one.")

    if use_ik and plan is None:
        say(f"The {resolved} is out of reach.")
        arm.go_to_rest()
        return False

    attempts = 1 + int(pp.get("retries", 1))
    gripped, hold = False, 0.0
    for attempt in range(attempts):
        if stopped():
            return False
        if use_ik:
            if not _execute_plan(arm, plan, stop_check):
                return False
        else:
            _reach_fixed(pp, arm, stop_check)
        if stopped():
            return False
        gripped, hold = close_with_feedback(
            arm, g, stop_check=stop_check, start_from=75.0, ease=False
        )
        if stopped():
            return False
        if gripped:
            _log_grip(arm, "grasp", hold, plan)
        if gripped and holding(arm, g, hold):
            break
        gripped = False
        arm.set_gripper_torque_limit(GRIPPER_TORQUE_LIMIT)
        arm.go_to({"gripper": 75.0}, duration_s=0.6, stop_check=stop_check)
        if attempt + 1 >= attempts or not use_ik:
            continue
        say("The grasp slipped - trying once more.")
        if not _return_along(arm, plan, stop_check):
            return False
        fresh = _view_here(arm, camera, detector.find, object_name, pp, cam)
        if fresh is None:
            logger.info("retry: no usable box from the pre-grasp - giving up")
            break
        plan = _plan_from_view(cfg, fresh)
        if plan is None:
            logger.info("retry: the fresh box gives no grasp - giving up")
            break
    if stopped():
        return False
    if not gripped:
        say(f"I couldn't get a grip on the {resolved}. Giving up for now.")
        park_open(arm)
        return False

    status = _offer(
        arm, hold, stop_check, JOINT_LIMITS, say, resolved,
        lambda: holding(arm, g, hold), lambda: ease_grip(arm, g),
    )
    ease_grip(arm, g)
    if status == "stopped":
        return False
    if status == "dropped":
        say(f"I dropped the {resolved}.")
        park_open(arm)
        return False
    if status != "offered":
        say(f"I can't lift the {resolved} clear of the other objects.")
        return False
    shared_state["grip_lock"] = True
    shared_state["grip_lock_pos"] = hold
    say("Done.")
    return True


def _reach_fixed(pp: dict, arm: ArmClient, stop_check=None) -> None:
    """Jaws open over the object at the configured reach, then down onto it."""
    above = {**pp["approach_pose"], "shoulder_pan": arm.get_joints()["shoulder_pan"]}
    arm.go_to({**above, "gripper": 75.0}, duration_s=1.5, stop_check=stop_check)
    arm.go_to(
        {"shoulder_lift": float(pp["descend_lift"])},
        duration_s=1.5,
        stop_check=stop_check,
    )


# =============================================================================
# Standalone component test
# =============================================================================


def main() -> None:
    """CLI entry point: run pick-and-place standalone for a named object."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Voice-grounded pick-and-place - component test"
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--object", required=True, help='spoken object name, e.g. "ball"'
    )
    parser.add_argument(
        "--dry-run-arm",
        action="store_true",
        help="real cameras + detection, simulated arm",
    )
    parser.add_argument("--device", choices=["npu", "cpu"], default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    from vla_pipeline.vision.camera import make_camera
    from vla_pipeline.vision.yolo_pose_npu import PoseEstimator

    arm: ArmClient = DryRunArm() if args.dry_run_arm else make_arm(cfg)
    detector = ObjectDetector(cfg, device=args.device)
    pose = PoseEstimator(cfg, device=args.device)
    arm_cam = make_camera(cfg, "arm")
    operator_cam = make_camera(cfg, "operator")
    try:
        ok = run_pick_place(
            cfg,
            arm,
            detector,
            pose,
            arm_cam,
            operator_cam,
            args.object,
            feedback=lambda m: print(f"{m}"),
        )
        print(f"\npick_place {'PASSED' if ok else 'FAILED'}")
        raise SystemExit(0 if ok else 1)
    finally:
        arm_cam.close()
        operator_cam.close()
        arm.close()


if __name__ == "__main__":
    main()
