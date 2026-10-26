# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Which table spots a pick finds and plans a grasp for.

Each object is put alone on a grid of table spots. At every spot the stack's
own scan (``_search_views``) walks its views with the arm set straight to
each one, the twin's segmentation gives the boxes (exact, and hidden where
the jaws cover the image), and the stack's own ``_plan_from_view`` locates
the box and solves the grasp. The pickup judges YOLO's boxes, which near the
image border reach farther out than the segmentation, so a view counts only
when its box also keeps ``_DETECTOR_SLACK_PX`` more than the stack's
``EDGE_MARGIN_PX`` from the border. A spot is

- ``found``: the scan planned a grasp there;
- ``seen``: a usable box was seen, but no locate or grasp came of it;
- ``missed``: the open jaws reach the spot from above, and no view gave a
  usable box;
- ``out``: the open jaws do not reach the spot from above.
"""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

STACKS = {
    "pipeline": {
        "manipulation": "vla_pipeline.behaviors.manipulation",
        "motion": "vla_pipeline.robot.arm_interface",
        "detection": "vla_pipeline.vision.yolo_detect_npu",
    },
    "workshop": {
        "manipulation": "common.manipulation",
        "motion": "common.motion",
        "detection": "common.imaging",
    },
}

# Resting centre height on the table, and the spoken name.
OBJECTS = {
    "ball": ("sports_ball", 0.015, "ball"),
    "bottle": ("bottle", 0.0375, "bottle"),
    "toothbrush radial": ("toothbrush", 0.005, "toothbrush"),
    "toothbrush across": ("toothbrush", 0.005, "toothbrush"),
}

# The open jaws, 4 cm apart, reaching a point this high above the table.
_REACH_WIDTH_M = 0.04
_REACH_HEIGHT_M = 0.016
# How much farther out than the segmentation box YOLO's box reached within
# 30 px of the image border on the twin's scan views (90th percentile).
_DETECTOR_SLACK_PX = 12.0


@dataclass
class Spot:
    xy: tuple[float, float]
    result: str
    view: str = ""
    error_mm: float | None = None
    reasons: list[str] = field(default_factory=list)


def grid(radii, angles_deg) -> list[tuple[float, float]]:
    """Table spots at each distance from the base and direction (0 deg = straight out)."""
    return [
        (round(r * math.cos(math.radians(a)), 4), round(r * math.sin(math.radians(a)), 4))
        for r in radii for a in angles_deg
    ]


def _yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw))


def _object_quat(kind: str, xy) -> tuple[float, float, float, float]:
    """The toothbrush's long axis is its body y axis."""
    toward = math.atan2(xy[1], xy[0])
    if kind == "toothbrush radial":
        return _yaw_quat(toward - 0.5 * math.pi)
    if kind == "toothbrush across":
        return _yaw_quat(toward)
    return (1.0, 0.0, 0.0, 0.0)


class _Reasons(logging.Handler):
    """The stack's own lines on why a sighting gave no grasp."""

    KEEP = ("IK reach:", "scan sighting:")

    def __init__(self):
        super().__init__(logging.INFO)
        self.lines: list[str] = []

    def emit(self, record) -> None:
        message = record.getMessage()
        if message.startswith(self.KEEP):
            self.lines.append(message)


class _TeleportArm:
    """``go_to`` sets the twin's arm to the pose at once, clamped as the
    stack's arm clamps it, and puts the object back."""

    def __init__(self, world, put, clamp, limits=None):
        self.world = world
        self._put = put
        self._clamp = clamp
        self._limits = limits
        self._pose = dict(world.rest_pose)

    def go_to(self, target, duration_s=0.0, stop_check=None, limits=None):
        self._pose.update(self._clamp(target, limits if limits is not None else self._limits))
        self.world.reset(self._pose)
        self._put()

    def get_joints(self) -> dict:
        return self.world.joint_cal()


class Coverage:
    """One stack's scan and grasp planning on a twin whose physics does not run."""

    def __init__(self, stack: str, cfg: dict, scan_limits: dict | None = None):
        """``scan_limits``: the joint limits a scan move is clamped to (default: the arm's own)."""
        from scripted import SimDetector
        from so101_sim.camera import SimCamera
        from so101_sim.detect import SegDetector
        from so101_sim.kinematics import shared_kinematics
        from so101_sim.render import RenderService
        from so101_sim.world import SimWorld

        names = STACKS[stack]
        self.mod = importlib.import_module(names["manipulation"])
        motion = importlib.import_module(names["motion"])
        detection = importlib.import_module(names["detection"])
        if stack == "pipeline":
            from vla_pipeline.behaviors._ctx import flatten_cfg

            cfg = flatten_cfg(cfg)
        self.ctx = SimpleNamespace(cfg=cfg)
        self.pp = cfg["pick_place"]
        self.cam = (cfg.get("cameras") or {}).get("arm") or {}
        self.limits = motion.GRASP_LIMITS
        self._clamp = motion.clamp_joints
        self._scan_limits = scan_limits
        sim = dict(cfg.get("sim") or {})
        sim["gripper_map"] = {**(sim.get("gripper_map") or {}), "closed_cal": motion.GRIPPER_CLOSED_FLOOR}
        self.world = SimWorld({"sim": sim}, rest_pose=motion.REST_POSE, start=False)
        self.renderer = RenderService(self.world, require_hw=False)
        self.camera = SimCamera(self.world, "arm", renderer=self.renderer, segmentation=True)
        self.detector = SimDetector(SegDetector(self.camera), detection.Detection, detection.resolve_class)
        self.kin = shared_kinematics(motion.GRIPPER_CLOSED_FLOOR)

    def close(self) -> None:
        self.camera.release()
        self.renderer.close()
        self.world.close()

    def reaches(self, xy) -> bool:
        """The open jaws reach a point just above the table at ``xy`` from above."""
        table_z = float(self.pp.get("table_z", -0.005))
        p = np.array([xy[0], xy[1], table_z + _REACH_HEIGHT_M])
        return self.kin.solve_grasp(
            p, above_m=float(self.pp.get("pregrasp_above_m", 0.06)), width=_REACH_WIDTH_M,
            table_z=table_z, limits=self.limits,
        ) is not None

    def _scan(self, kind: str, xy, walk):
        """Run ``walk(arm, find, spoken, accept)`` with only this object on the table at ``xy``.

        Returns ``(seen, found, reasons)``: whether ``accept`` ever got a
        usable box, the planned view (``plan`` and ``view``) if one planned,
        and the stack's log lines about the sightings.
        """
        body, z, spoken = OBJECTS[kind]
        world = self.world
        quat = _object_quat(kind, xy)

        def put():
            for other in world.objects:
                if other != body:
                    world.hide(other)
            world.place(body, (xy[0], xy[1], z), quat)

        from so101_sim.kinematics import EDGE_MARGIN_PX, box_cut

        seen, found = [], {}

        def robust(view) -> bool:
            height, width = view["frame"].shape[:2]
            cut, top = box_cut(view["det"].box, (width, height), EDGE_MARGIN_PX + _DETECTOR_SLACK_PX)
            return not cut and not (self.mod._shape_of(view["resolved"]) == "flat" and top)

        def accept(batch):
            batch = [view for view in batch if robust(view)]
            if not batch:
                return False
            seen.append(True)
            for view in self.mod._rank_views(batch):
                plan = self.mod._plan_from_view(self.ctx, view)
                if plan is not None:
                    found["plan"], found["view"] = plan, view
                    return True
            return False

        reasons = _Reasons()
        log = logging.getLogger(self.mod.__name__)
        level = log.level
        log.setLevel(logging.INFO)
        log.addHandler(reasons)
        try:
            walk(_TeleportArm(world, put, self._clamp, self._scan_limits), self.detector.find, spoken, accept)
        finally:
            log.removeHandler(reasons)
            log.setLevel(level)
        return bool(seen), found, reasons.lines

    def view(self, kind: str, xy, pose: dict, pan: float) -> str:
        """``planned``, ``usable`` or ``none``: one view, at ``pose`` turned to ``pan``."""
        seen, found, _ = self._scan(
            kind, xy,
            lambda arm, find, spoken, accept: self.mod._sweep_views(
                arm, self.camera, find, spoken, pose, [pan], self.pp, None, cam=self.cam, accept=accept,
            ),
        )
        return "planned" if found else ("usable" if seen else "none")

    def spot(self, kind: str, xy) -> Spot:
        seen, found, reasons = self._scan(
            kind, xy,
            lambda arm, find, spoken, accept: self.mod._search_views(
                arm, self.camera, find, spoken, self.pp, None, self.cam, accept,
            ),
        )
        if found:
            view = found["view"]
            near = view["joints"].get("wrist_flex", 0.0) > self.pp["scan_pose"]["wrist_flex"] + 1.0
            label = f"{'near' if near else 'high'} pan {view['joints']['shoulder_pan']:.0f}"
            error = 1000 * float(np.linalg.norm(np.subtract(found["plan"].xyz[:2], xy)))
            return Spot(tuple(xy), "found", label, round(error, 1))
        if not self.reaches(xy):
            return Spot(tuple(xy), "out", reasons=reasons[-2:])
        return Spot(tuple(xy), "seen" if seen else "missed", reasons=reasons[-2:])


COLORS = {"found": "#2e9e44", "seen": "#e8c51c", "missed": "#f07c1a", "out": "#bdbdbd"}


def draw(results: dict[str, list[Spot]], path, title: str, marks: dict | None = None) -> None:
    """One panel per object: the table seen from above, the arm base at the origin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = list(results)
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 4.6), squeeze=False)
    for ax, kind in zip(axes[0], kinds):
        spots = results[kind]
        for result, color in COLORS.items():
            pts = [s.xy for s in spots if s.result == result]
            if pts:
                xs, ys = zip(*pts)
                # x forward drawn up, y to the arm's left drawn left
                ax.scatter([-y for y in ys], xs, s=70, marker="s", color=color,
                           label=f"{result} ({len(pts)})")
        for name, (x, y) in (marks or {}).items():
            ax.plot(-y, x, "kx")
            ax.annotate(name, (-y, x), textcoords="offset points", xytext=(4, 4), fontsize=7)
        ax.plot(0, 0, "ko")
        ax.set_title(kind)
        ax.set_aspect("equal")
        ax.set_xlabel("left (m)  <-  y  ->  right")
        ax.set_ylabel("x, out from the base (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
