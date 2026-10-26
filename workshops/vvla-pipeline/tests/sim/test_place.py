# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Both stacks' copies of the known objects, "place it" and "drop it", without a scene.

The pickup's scan keeps a list of the objects it detects on the table (one
footprint each); "place it" sets a picked object that is still in the jaws
down on the point nearest the drop position (where the pickup holds it out)
that keeps its spacing from every known object, lies off the arm's way from
its parked pose into the first scan view, and lets the arm's model pass every
known object through the put-down. It looks at nothing; table no footprint
covers counts as free. "drop it" opens the jaws where they are and
adds the area under them. The detector here is a fake that reports the balls
a test puts on the table, so the list and the choice of spot can be checked
exactly.
"""

from __future__ import annotations

import copy
import importlib
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.sim

STACKS = {
    "pipeline": ("vla_pipeline.behaviors.manipulation", "vla_pipeline.robot.arm_interface"),
    "workshop": ("common.manipulation", "common.motion"),
}
# Grasp point heights the pickups log ("fingers .. mm high") with the object on the table.
GRASP_Z = {"sports ball": 0.021, "bottle": 0.049, "toothbrush": 0.021}
# The wrist roll the pickup holds each object out with: the toothbrush grasp turns it.
HELD_ROLL = {"sports ball": 0.0, "bottle": 0.0, "toothbrush": 70.0}
_HOLD = 40.0
_TABLE_Z = -0.005
_SIZE = (640, 480)


def _stack(stack: str, request):
    manipulation, motion = (importlib.import_module(name) for name in STACKS[stack])
    cfg = request.getfixturevalue(f"{stack}_cfg")
    if stack == "pipeline":
        from vla_pipeline.behaviors._ctx import flatten_cfg

        cfg = flatten_cfg(cfg)
    return manipulation, motion, copy.deepcopy(cfg)


def _kin(motion):
    from so101_sim.kinematics import shared_kinematics

    return shared_kinematics(motion.GRIPPER_CLOSED_FLOOR)


class _Arm:
    """Records every goal; the jaw reads ``jaw`` until a goal moves it."""

    def __init__(self, jaw: float, pose: dict | None = None):
        self.pose = {"shoulder_pan": 0.0, "shoulder_lift": 10.0, "elbow_flex": 10.0,
                     "wrist_flex": 60.0, "wrist_roll": 0.0, **(pose or {}), "gripper": jaw}
        self.start = dict(self.pose)
        self.goals = []

    def get_joints(self) -> dict:
        return dict(self.pose)

    def go_to(self, target, duration_s=0.0, stop_check=None, limits=None):
        self.goals.append(dict(target))
        self.pose.update(target)

    def move_to(self, target, stop_check=None, limits=None, **_):
        self.go_to(target)

    def set_gripper_torque_limit(self, value) -> None:
        pass


class _Detector:
    """Boxes of balls resting at ``balls`` (x, y) as seen from ``joints``, and ``extra`` boxes."""

    def __init__(self, kin, joints, balls=(), extra=()):
        self.kin, self.joints, self.balls, self.extra = kin, joints, list(balls), list(extra)

    def infer(self, frame, conf=None):
        K = self.kin.default_intrinsics(*_SIZE)
        out = [SimpleNamespace(name=name, score=0.9, box=np.array(box, np.float32)) for name, box in self.extra]
        for x, y in self.balls:
            uv = self.kin.project(self.kin._surface("sphere", (x, y, 0.02), _TABLE_Z), self.joints, K)
            out.append(SimpleNamespace(name="sports ball", score=0.9, box=np.array([*uv.min(axis=0), *uv.max(axis=0)], np.float32)))
        return out


def _ctx(cfg, arm, state, detector=None):
    said = []
    ctx = SimpleNamespace(cfg=cfg, arm=arm, shared_state=state, say=said.append, stop_check=lambda: False,
                          detector=detector)
    return ctx, said


def _held_out(kin, motion, roll: float) -> dict:
    """Where the pickup holds an object out (``_offer``'s second leg)."""
    pose, err, _ = kin.ik(np.array([0.30, 0.0, 0.12]), 0.04, _HOLD, None, roll, limits=motion.GRASP_LIMITS)
    assert err < 0.015, f"no hold-out pose with roll {roll}"
    return {**pose, "gripper": _HOLD}


def _picked(name: str = "sports ball") -> dict:
    return {"grip_lock": True, "grip_lock_pos": _HOLD, "picked": {"name": name, "grasp_z": GRASP_Z[name]}}


def _scan_joints(cfg, pan: float) -> dict:
    return {**cfg["pick_place"]["scan_pose"], "shoulder_pan": pan}


def _scan_frame(manipulation, kin, cfg, joints, balls=(), extra=()):
    """One scan frame through the stack's observer, into a fresh list."""
    objects = manipulation.TableObjects()
    detector = _Detector(kin, joints, balls, extra)
    ctx, _ = _ctx(cfg, None, {}, detector)
    manipulation._observer(ctx, objects, detector.infer)(np.zeros((_SIZE[1], _SIZE[0], 3), np.uint8), joints)
    return objects


def _known(manipulation, name, xy, radius, height):
    return manipulation._Footprint(name, np.asarray(xy, float), radius, height)


def test_place_phrases_route_to_place_and_drop_it_stays_release():
    """The keyword router (the mock llama-server, the app's --no-llm brain)."""
    from common.fixtures import offline_parse_text

    for phrase in ("place it", "set it down", "put it down", "set it on the table"):
        assert offline_parse_text(phrase)["command"] == "place", phrase
    for phrase in ("drop it", "release", "open", "let go"):
        assert offline_parse_text(phrase)["command"] == "release", phrase
    assert offline_parse_text("pick up the ball and place it")["command"] == "pick_place"


@pytest.mark.parametrize("stack", list(STACKS))
def test_a_scan_frame_adds_each_detected_object_where_it_rests(stack, request):
    manipulation, motion, cfg = _stack(stack, request)
    kin = _kin(motion)
    objects = _scan_frame(manipulation, kin, cfg, _scan_joints(cfg, -10.0), balls=[(0.30, 0.02)])
    assert len(objects.items) == 1, str(objects)
    ball = objects.items[0]
    assert np.linalg.norm(ball.centre - [0.30, 0.02]) < 0.01, str(objects)
    assert 0.015 < ball.radius < 0.03 and ball.half == 0.0, str(objects)
    assert ball.evidence == manipulation._FITTED, "a sphere has a shape model"


@pytest.mark.parametrize("stack", list(STACKS))
def test_a_flat_object_is_a_strip_along_its_length(stack, request):
    manipulation, motion, cfg = _stack(stack, request)
    kin = _kin(motion)
    joints = _scan_joints(cfg, -10.0)
    top = _TABLE_Z + 0.012
    brush = np.array([[0.24, -0.012, top], [0.36, -0.012, top], [0.36, 0.012, top], [0.24, 0.012, top]])
    uv = kin.project(brush, joints, kin.default_intrinsics(*_SIZE))
    box = (*uv.min(axis=0), *uv.max(axis=0))
    objects = _scan_frame(manipulation, kin, cfg, joints, extra=[("toothbrush", box)])
    strip = objects.items[0]
    assert np.linalg.norm(strip.centre - [0.30, 0.0]) < 0.01, str(objects)
    assert strip.half + strip.radius >= 0.055, f"a 12 cm toothbrush covers only {str(objects)}"
    assert strip.evidence == manipulation._FITTED, "a toothbrush is a flat class"


@pytest.mark.parametrize("stack", list(STACKS))
def test_the_jaws_in_their_own_corner_are_not_an_object(stack, request):
    manipulation, motion, cfg = _stack(stack, request)
    jaws = ("knife", (400, 290, 640, 476))  # how YOLO named the open jaws in the twin
    objects = _scan_frame(manipulation, _kin(motion), cfg, _scan_joints(cfg, -10.0), extra=[jaws])
    assert objects.items == [], str(objects)


@pytest.mark.parametrize("stack", list(STACKS))
def test_an_object_seen_again_replaces_its_old_footprint(stack, request):
    manipulation, _, _ = _stack(stack, request)
    objects = manipulation.TableObjects()
    objects.see(_known(manipulation, "sports ball", (0.30, 0.06), 0.02, 0.04))
    objects.see(_known(manipulation, "bottle", (0.20, 0.20), 0.02, 0.085))
    objects.see(_known(manipulation, "orange", (0.31, 0.07), 0.02, 0.04))  # the ball again, another name
    assert sorted(f.name for f in objects.items) == ["bottle", "orange"], str(objects)
    objects.remove_near((0.20, 0.21), 0.04)
    assert [f.name for f in objects.items] == ["orange"], str(objects)


@pytest.mark.parametrize("stack", list(STACKS))
def test_a_sighting_replaces_only_what_is_known_less_well(stack, request):
    """The scan once named the toothbrush the arm had just laid down a
    "remote", a class with no shape model, and its 10 cm guess blocked the
    table: the arm's own record stays. A ball seen where a drop left only a
    guess replaces it."""
    m, _, _ = _stack(stack, request)
    objects = m.TableObjects()
    laid = m._held_at("toothbrush", (0.31, -0.01))
    objects.see(laid)
    objects.see(m._Footprint("remote", np.array([0.297, -0.003]), 0.05, m._UNSURE_HEIGHT_M, evidence=m._GUESSED))
    objects.see(m._Footprint("toothbrush", np.array([0.30, 0.0]), 0.012, 0.012, evidence=m._FITTED))
    assert [id(f) for f in objects.items] == [id(laid)], str(objects)

    dropped = m._Footprint("sports ball", np.array([0.20, 0.10]), m._DROP_SPREAD_M, 0.04)
    objects.see(dropped)
    seen = m._Footprint("sports ball", np.array([0.22, 0.11]), 0.02, 0.04, evidence=m._FITTED)
    objects.see(seen)
    assert [id(f) for f in objects.items] == [id(laid), id(seen)], str(objects)


@pytest.mark.parametrize("stack", list(STACKS))
def test_a_reset_forgets_the_grip_the_pickup_and_the_objects(stack, request):
    manipulation, _, _ = _stack(stack, request)
    state = {**_picked(), "table_objects": manipulation.TableObjects(), "other": 1}
    manipulation.forget_scene(state)
    assert state == {"other": 1}


@pytest.mark.parametrize("stack", list(STACKS))
def test_the_pinch_taking_a_held_grip_forgets_the_objects(stack, request):
    manipulation, _, _ = _stack(stack, request)
    state = {**_picked(), "table_objects": manipulation.TableObjects()}
    assert manipulation.release_to_pinch(state) is True
    assert state == {}
    kept = {"table_objects": manipulation.TableObjects()}
    assert manipulation.release_to_pinch(kept) is False, "no grip was locked"
    assert "table_objects" in kept


@pytest.mark.parametrize("stack", list(STACKS))
def test_drop_it_opens_where_it_is_and_the_table_below_joins_the_objects(stack, request):
    manipulation, motion, cfg = _stack(stack, request)
    kin = _kin(motion)
    arm = _Arm(jaw=_HOLD + 10.0, pose=_held_out(kin, motion, 0.0))
    objects = manipulation.TableObjects()
    state = {**_picked(), "table_objects": objects}
    ctx, said = _ctx(cfg, arm, state)
    under = kin.world.grasp_point(arm.pose, 0.04)[:2]
    assert manipulation.run_release(ctx) is True
    assert arm.goals == [{"gripper": motion.JOINT_LIMITS["gripper"][1]}], arm.goals
    assert said == ["Released."], said
    assert not {"grip_lock", "grip_lock_pos", "picked"} & set(state), state
    assert len(objects.items) == 1 and np.linalg.norm(objects.items[0].centre - under) < 1e-6, str(objects)
    assert objects.items[0].radius >= 0.05, "a dropped object can roll farther than its own size"


@pytest.mark.parametrize("stack", list(STACKS))
def test_drop_it_opens_a_handed_over_object_where_it_is(stack, request):
    manipulation, motion, cfg = _stack(stack, request)
    arm = _Arm(jaw=_HOLD + 10.0)
    state = {"grip_lock": True, "grip_lock_pos": _HOLD}  # "hold this": no pickup recorded
    ctx, said = _ctx(cfg, arm, state)
    assert manipulation.run_release(ctx) is True
    assert arm.goals == [{"gripper": motion.JOINT_LIMITS["gripper"][1]}], arm.goals
    assert said == ["Released."], said


def _refused(arm, said, line: str) -> None:
    assert arm.goals == [], f"the arm moved: {arm.goals}"
    assert said == [line], said


@pytest.mark.parametrize("stack", list(STACKS))
def test_place_refuses_a_handed_over_object(stack, request):
    manipulation, _, cfg = _stack(stack, request)
    arm = _Arm(jaw=_HOLD + 10.0)
    state = {"grip_lock": True, "grip_lock_pos": _HOLD}
    ctx, said = _ctx(cfg, arm, state)
    assert manipulation.run_place(ctx) is False
    _refused(arm, said, "I'm not holding anything I picked up.")
    assert state["grip_lock"], "a refused place kept the grip"


@pytest.mark.parametrize("stack", list(STACKS))
def test_place_with_empty_jaws_says_so_and_clears_the_grip(stack, request):
    manipulation, _, cfg = _stack(stack, request)
    arm = _Arm(jaw=_HOLD)  # closed onto the held goal: the object slipped out
    state = _picked()
    ctx, said = _ctx(cfg, arm, state)
    assert manipulation.run_place(ctx) is False
    _refused(arm, said, "I'm not holding the sports ball any more.")
    assert not {"grip_lock", "grip_lock_pos", "picked"} & set(state), state


def _place(stack, request, known, name="sports ball"):
    """Run "place it" from the hold-out pose with the known objects ``known(manipulation, drop)`` lists."""
    manipulation, motion, cfg = _stack(stack, request)
    kin = _kin(motion)
    held_out = _held_out(kin, motion, HELD_ROLL[name])
    drop = kin.world.grasp_point(held_out, 0.04)[:2]
    arm = _Arm(jaw=_HOLD + 10.0, pose=held_out)
    objects = manipulation.TableObjects()
    for footprint in known(manipulation, drop):
        objects.see(footprint)
    state = {**_picked(name), "table_objects": objects}
    ctx, said = _ctx(cfg, arm, state)
    ok = manipulation.run_place(ctx)
    return SimpleNamespace(ok=ok, said=said, state=state, arm=arm, kin=kin, motion=motion, cfg=cfg,
                           drop=drop, objects=objects, manipulation=manipulation)


def _set_down_at(kin, arm) -> np.ndarray:
    """Where the grasp point was when the jaws opened, and check they opened only there."""
    from so101_sim.kinematics import OPEN_GRIPPER

    opened = next(i for i, goal in enumerate(arm.goals) if goal.get("gripper") == OPEN_GRIPPER)
    down = arm.goals[opened - 1]
    assert down["gripper"] == _HOLD, "the jaws opened before the object was down"
    return kin.world.grasp_point(down, 0.04)


def _line(a: dict, b: dict) -> list:
    """``b`` reached from ``a`` on the straight joint-space line, 1 degree apart."""
    n = max(1, int(np.ceil(max(abs(b[k] - a[k]) for k in b))))
    return [{k: a[k] + (b[k] - a[k]) * i / n for k in b} for i in range(1, n + 1)]


def _swept(arm) -> list:
    """Every pose the arm passes on its goals: lockstep moves run the joint-space line."""
    pose, out = dict(arm.start), []
    for goal in arm.goals:
        out += _line(pose, {**pose, **goal})
        pose = out[-1]
    return out


def _closest(kin, poses, xy, radius: float, height: float) -> float:
    """The arm's closest approach over ``poses`` to an upright cylinder on the table (m)."""
    return min(kin.hand_clearance(q, (xy[0], xy[1], radius, _TABLE_Z + height)) for q in poses)


@pytest.mark.parametrize("stack", list(STACKS))
def test_place_sets_down_under_the_drop_position_with_nothing_known_there(stack, request):
    r = _place(stack, request, lambda m, drop: [])
    assert r.ok is True, r.said
    assert r.said == ["Putting the sports ball down.", "Done."], r.said
    end = _set_down_at(r.kin, r.arm)
    assert np.linalg.norm(end[:2] - r.drop) < 0.005, f"set down at {end}, drop position {r.drop}"
    assert end[2] == pytest.approx(GRASP_Z["sports ball"] + r.cfg["pick_place"]["set_down_clearance_m"], abs=0.005)
    assert r.arm.goals[-1] == {**r.motion.REST_POSE, "gripper": r.motion.JOINT_LIMITS["gripper"][1]}, r.arm.goals[-1]
    assert not {"grip_lock", "grip_lock_pos", "picked"} & set(r.state), r.state
    assert len(r.objects.items) == 1 and np.linalg.norm(r.objects.items[0].centre - end[:2]) < 0.005, str(r.objects)


@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("neighbour", [("sports ball", 0.02, 0.04), ("bottle", 0.02, 0.085)], ids=["ball", "standing bottle"])
def test_place_beside_an_object_under_the_drop_position_never_touches_it(stack, neighbour, request):
    """The held ball goes down near the object, the spacing kept, and the arm
    passes the object at every pose of the put-down: carried over, down, the
    jaws opening, back up and home."""
    name, radius, height = neighbour
    r = _place(stack, request, lambda m, drop: [_known(m, name, drop, radius, height)])
    assert r.ok is True, r.said
    end = _set_down_at(r.kin, r.arm)[:2]
    gap = float(np.linalg.norm(end - r.drop))
    spacing = 0.02 + radius + r.cfg["pick_place"]["place_clearance_m"]
    assert spacing - 0.005 <= gap <= spacing + 0.06, f"set down {gap:.3f} m from the {name} (spacing {spacing:.3f})"
    near = _closest(r.kin, _swept(r.arm), r.drop, radius, height)
    assert near > 0.0, f"the arm went {-1000 * near:.1f} mm into the {name}"


@pytest.mark.parametrize("stack", list(STACKS))
def test_place_keeps_the_object_off_the_arms_way_out_of_park(stack, request):
    """A tray over the drop position leaves the nearest free table toward the
    base, where the arm unfolds from its parked pose to the ready pose and its
    open jaw would sweep a standing bottle over. The bottle goes down where
    that move passes it."""
    tray = np.array([0.33, 0.0])
    r = _place(stack, request, lambda m, drop: [_known(m, "tray", tray, 0.08, 0.02)], name="bottle")
    assert r.ok is True, r.said
    end = _set_down_at(r.kin, r.arm)[:2]
    pp = r.cfg["pick_place"]
    park = {**r.motion.REST_POSE, "gripper": r.motion.JOINT_LIMITS["gripper"][1]}
    way_out = [park, *_line(park, {**r.manipulation._ready(pp), "gripper": park["gripper"]})]
    _half, radius, height = r.manipulation._HELD_FOOTPRINT_M["bottle"]
    assert _closest(r.kin, way_out, end, radius, height) > 0.0, f"set down at {end}, on the arm's way out of park"
    far, gap = float(pp["place_far_m"]["bottle"]), float(np.linalg.norm(end - r.drop))
    nearer = [
        xy for xy in (r.drop + np.array([dx, dy]) for dx in np.arange(-0.18, 0.19, 0.01) for dy in np.arange(-0.18, 0.19, 0.01))
        if np.linalg.norm(xy - r.drop) < gap and pp["place_near_m"] <= np.hypot(*xy) <= far
        and np.linalg.norm(xy - tray) >= radius + 0.08 + pp["place_clearance_m"]
    ]
    assert any(_closest(r.kin, way_out, xy, radius, height) < 0.0 for xy in nearer), (
        "no free spot nearer the drop position lies on the arm's way out of park: the scene tests nothing"
    )


@pytest.mark.parametrize("stack", list(STACKS))
def test_every_configured_scan_view_is_inside_the_arms_joint_range(stack, request):
    """A pan past the joint's range ends where the joint stops, not where the
    scan's coverage was measured: a configured -115 deg stopped at -110."""
    m, motion, cfg = _stack(stack, request)
    pp = cfg["pick_place"]
    kin = _kin(motion)
    high, pans = m._high_sweep(pp)
    views = [{**high, "shoulder_pan": float(p)} for p in pans]
    views += [{**pp["scan_near_pose"], "shoulder_pan": float(p)} for p in pp["scan_near_pans"]]
    outside = []
    for view in views:
        for joint, (lo, hi) in kin._limits.items():
            sent_lo, sent_hi = motion.GRASP_LIMITS[joint]
            if not max(lo, sent_lo) <= view[joint] <= min(hi, sent_hi):
                outside.append(f"{joint} {view[joint]:.0f} at pan {view['shoulder_pan']:.0f}")
    assert not outside, f"scan views outside the arm's range: {outside}"


@pytest.mark.parametrize("stack", list(STACKS))
def test_every_transit_move_passes_above_a_standing_bottle(stack, request):
    """Rest, ready, work: only the way between the parked pose and the ready
    pose dips toward the table (the place keeps objects off it). From ready
    to the scan views and between them, from the hold-out back to ready, and
    from the carry height over a put-down spot back to ready, the whole arm
    stays above a standing bottle's top."""
    from so101_sim.kinematics import OPEN_GRIPPER

    m, motion, cfg = _stack(stack, request)
    pp = cfg["pick_place"]
    kin = _kin(motion)
    ready = m._ready(pp)
    high, pans = m._high_sweep(pp)
    views = [{**high, "shoulder_pan": float(p)} for p in pans]
    views += [{**pp["scan_near_pose"], "shoulder_pan": float(p)} for p in pp["scan_near_pans"]]
    moves = [(ready, views[0]), *zip(views, views[1:]), (views[-1], ready)]
    for name, roll in HELD_ROLL.items():
        held = _held_out(kin, motion, roll)
        moves.append((held, ready))
        for xy in ((0.30, 0.0), (0.25, 0.12), (0.25, -0.12)):
            path = m._put_down_path(kin, held, _HOLD, xy, {"name": name, "grasp_z": GRASP_Z[name]}, pp)
            assert path is not None, f"no put-down path for the {name} at {xy}"
            moves.append((path[2][-1], {**ready, "gripper": OPEN_GRIPPER}))
    top = _TABLE_Z + 0.085 + m._ARM_CLEARANCE_M
    low = {}
    for a, b in moves:
        for q in [a, *_line(a, b)]:
            z = kin.lowest_point(q)
            if z < top:
                low[(round(a["shoulder_pan"]), round(b["shoulder_pan"]))] = round(1000 * z)
    assert not low, f"moves that dip below a standing bottle (mm, by start and end pan): {low}"


@pytest.mark.parametrize("stack", list(STACKS))
def test_the_toothbrush_fits_lengthwise_between_two_objects(stack, request):
    """Balls 11 cm either side of the line out through the drop position: the
    toothbrush as a 15.6 cm disc would not fit between them, laid pointing away
    from the base it does, and its jaws open across it without touching them."""
    def flanked(m, drop):
        across = np.array([-drop[1], drop[0]]) / np.hypot(*drop)
        return [_known(m, f"ball {side:+.0f}", drop + side * 0.11 * across, 0.02, 0.04) for side in (1.0, -1.0)]

    r = _place(stack, request, flanked, name="toothbrush")
    assert r.ok is True, r.said
    end = _set_down_at(r.kin, r.arm)[:2]
    across = np.array([-r.drop[1], r.drop[0]]) / np.hypot(*r.drop)
    assert abs(float((end - r.drop) @ across)) < 0.01, f"set down at {end}, not between the balls"
    for ball in r.objects.items[:2]:
        near = _closest(r.kin, _swept(r.arm), ball.centre, ball.radius, ball.height)
        assert near > 0.0, f"the arm went {-1000 * near:.1f} mm into the {ball.name}"


@pytest.mark.parametrize("stack", list(STACKS))
def test_place_with_no_room_near_the_drop_position_stays_holding(stack, request):
    """Something known covers all the table near the drop position: the arm
    says so, holds the object out where "drop it" lets it go, and does not move."""
    r = _place(stack, request, lambda m, drop: [_known(m, "tray", drop, 0.30, 0.02)])
    assert r.ok is False
    _refused(r.arm, r.said, "I can't find a free spot for the sports ball.")
    assert r.state["grip_lock"] and r.state["picked"]["name"] == "sports ball"


@pytest.mark.parametrize("stack", list(STACKS))
@pytest.mark.parametrize("name", list(GRASP_Z))
def test_each_object_has_a_put_down_path_under_the_drop_position(stack, name, request):
    """From the hold-out pose: carried to the carry height over the point, a
    descent that ends with the grasp point back at its resting height straight
    below, and a rise of the open jaws back up; the toothbrush ends pointing
    away from the base."""
    from so101_sim.kinematics import OPEN_GRIPPER

    manipulation, motion, cfg = _stack(stack, request)
    pp = cfg["pick_place"]
    kin = _kin(motion)
    here = _held_out(kin, motion, HELD_ROLL[name])
    drop = kin.world.grasp_point(here, 0.04)[:2]
    path = manipulation._put_down_path(kin, here, _HOLD, drop, {"name": name, "grasp_z": GRASP_Z[name]}, pp)
    assert path is not None, f"no put-down path for the {name} at {drop}"
    carry, descent, rise = path
    above = carry[-1]
    end = kin.world.grasp_point(descent[-1], 0.04)
    want = np.array([*drop, GRASP_Z[name] + float(pp["set_down_clearance_m"])])
    assert np.linalg.norm(end - want) < 0.005, f"{name}: set down at {end}"
    assert kin.world.grasp_point(above, 0.04)[2] > 0.11
    assert all(pose["gripper"] == _HOLD for pose in [*carry, *descent])
    assert rise[-1] == {**above, "gripper": OPEN_GRIPPER}, "the open jaws rise back to the carry height"
    if name in pp["wrist_align_classes"]:
        radial = np.array([*drop, 0.0]) / np.hypot(*drop)
        across = abs(float(kin.world.jaw_axis(descent[-1]) @ radial))
        assert across < 0.05, f"toothbrush not laid pointing away from the base ({across:.2f})"
