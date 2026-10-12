# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.

"""Where is the object, and which joint angles reach it - for the real arm and the twin.

The SO-101 MJCF gives forward kinematics in calibrated degrees (the same
units the LeRobot follower reports), so this module is not twin-only:

* :meth:`Kinematics.camera_pose` - the wrist camera's pose from the joint
  angles: FK to the gripper, then the camera's mount on the gripper.
* :meth:`Kinematics.locate_on_table` - an object detector box turned into a
  3D position: the camera rays through the box's pixels are intersected with
  the table plane (objects rest on the table, so one camera is enough).
* :meth:`Kinematics.solve_grasp` - damped least-squares IK for a top-down
  grasp: the point between the jaws on the object, the closing axis
  horizontal, every arm part above the table.

Camera frames follow OpenCV (x right, y down, z forward); a mount is the
camera frame in the ``gripper`` body frame - what ``cv2.calibrateHandEye``
returns for an eye-in-hand camera. Until the real camera is calibrated the
twin's ``wrist_cam`` (pose and intrinsics) is the default.
"""

from __future__ import annotations

import math
import threading

import numpy as np

from so101_sim.world import TABLE_Z

__all__ = [
    "Kinematics",
    "TABLE_Z",
    "ROUND_CLASSES",
    "rpy_to_matrix",
    "matrix_to_rpy",
    "solve_mount",
    "shared_kinematics",
]

IK_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
OPEN_GRIPPER = 75.0
# detector classes that rest on the table as a sphere; others are treated as upright
ROUND_CLASSES = frozenset({"sports ball", "orange", "apple"})

_CV_FROM_MJ = np.diag([1.0, -1.0, -1.0])  # MuJoCo cameras look down -z with y up
_CLEARANCE_BODIES = ("lower_arm", "wrist", "gripper", "camera_mount", "moving_jaw_so101_v1")
_AXIS_WEIGHT = 0.05  # metres of grasp-point error worth one unit of closing-axis tilt
_SEEDS = ((9.0, 20.0, 54.0), (-15.0, 20.0, 54.0), (30.0, -20.0, 75.0), (-40.0, 60.0, 60.0))
_RING = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 72, endpoint=False)])
_golden = math.pi * (3.0 - math.sqrt(5.0))
_SPHERE = np.array([
    [math.cos(_golden * i) * math.sqrt(1 - z * z), math.sin(_golden * i) * math.sqrt(1 - z * z), z]
    for i, z in enumerate(np.linspace(-1, 1, 600))
])


def rpy_to_matrix(rpy_deg) -> np.ndarray:
    """``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)`` from ``[roll, pitch, yaw]`` in degrees."""
    r, p, y = (math.radians(float(v)) for v in rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(R) -> list[float]:
    """Inverse of :func:`rpy_to_matrix` (degrees)."""
    R = np.asarray(R, float)
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(R[2, 0]) < 0.999999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal lock: fold yaw into roll
        roll, yaw = math.atan2(-R[1, 2], R[1, 1]), 0.0
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


class Kinematics:
    """FK, camera rays and grasp IK on the SO-101 model (no physics thread)."""

    def __init__(self, world=None, sim_cfg: dict | None = None, camera: str = "wrist_cam"):
        if world is None:
            from so101_sim.world import SimWorld

            world = SimWorld(sim_cfg, start=False)
        self.world = world
        self.model = world.model
        self._lock = threading.RLock()
        import mujoco

        m = self.model
        cid = m.camera(camera).id
        R_body_cam = np.zeros(9)
        mujoco.mju_quat2Mat(R_body_cam, m.cam_quat[cid])
        R_body_cam = R_body_cam.reshape(3, 3) @ _CV_FROM_MJ
        # camera body pose in the gripper frame (any rest pose will do)
        d = world.fk({})
        Rg, tg = self._body(d, "gripper")
        Rb, tb = self._body(d, m.body(m.cam_bodyid[cid]).name)
        R_g_body, t_g_body = Rg.T @ Rb, Rg.T @ (tb - tg)
        self.default_mount = (R_g_body @ R_body_cam, t_g_body + R_g_body @ m.cam_pos[cid])
        self._sensor = (m.cam_sensorsize[cid].copy(), m.cam_intrinsic[cid].copy(), m.cam_resolution[cid].copy())
        self._limits = {}
        for i, name in enumerate(IK_JOINTS):  # the first four of MOTOR_NAMES
            lo, hi = (world.jmap.to_cal(name, v) for v in world._jrange[i])
            self._limits[name] = (min(lo, hi), max(lo, hi))
        self._clear_geoms = [
            g for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and m.body(m.geom_bodyid[g]).name in _CLEARANCE_BODIES
        ]

    # ------------------------------------------------------------------
    # frames
    # ------------------------------------------------------------------
    @staticmethod
    def _body(d, name: str):
        b = d.body(name)
        return b.xmat.reshape(3, 3).copy(), b.xpos.copy()

    def gripper_pose(self, joints: dict):
        """``(R, t)`` of the ``gripper`` body in the world (base) frame."""
        with self._lock:
            return self._body(self.world.fk(dict(joints)), "gripper")

    def camera_pose(self, joints: dict, mount=None):
        """``(R, t)`` world-from-camera (OpenCV axes) at ``joints``.

        ``mount`` is ``(R, t)`` of the camera in the gripper frame, a config
        dict ``{pos, rpy}``, or ``None`` for the twin's ``wrist_cam``.
        """
        Rm, tm = self.mount(mount)
        Rg, tg = self.gripper_pose(joints)
        return Rg @ Rm, tg + Rg @ tm

    def mount(self, mount=None):
        """Normalise a mount spec to ``(R, t)`` (see :meth:`camera_pose`)."""
        if mount is None:
            return self.default_mount
        if isinstance(mount, dict):
            if mount.get("pos") is None or mount.get("rpy") is None:
                return self.default_mount
            return rpy_to_matrix(mount["rpy"]), np.asarray(mount["pos"], float)
        R, t = mount
        return np.asarray(R, float), np.asarray(t, float).reshape(3)

    def default_intrinsics(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """3x3 camera matrix of the twin's ``wrist_cam`` at ``width`` x ``height``."""
        sensor, intr, res = self._sensor
        W, H = int(width or res[0]), int(height or res[1])
        fx, fy = intr[0] / sensor[0] * W, intr[1] / sensor[1] * H
        cx, cy = W / 2.0 + intr[2] / sensor[0] * W, H / 2.0 + intr[3] / sensor[1] * H
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    def intrinsics(self, spec: dict | None, width: int, height: int):
        """``(K, dist)`` from a ``cameras.arm.intrinsics`` dict (or the twin's).

        A calibration taken at another resolution is scaled to the frame.
        """
        if not spec or spec.get("fx") is None:
            return self.default_intrinsics(width, height), None
        sx = width / float(spec.get("width") or width)
        sy = height / float(spec.get("height") or height)
        K = np.array([[spec["fx"] * sx, 0.0, spec["cx"] * sx],
                      [0.0, spec["fy"] * sy, spec["cy"] * sy],
                      [0.0, 0.0, 1.0]])
        dist = spec.get("dist")
        return K, (np.asarray(dist, float) if dist is not None and any(dist) else None)

    def project(self, points, joints: dict, K=None, mount=None) -> np.ndarray:
        """World points -> pixels (no distortion); for tests and calibration checks."""
        K = self.default_intrinsics() if K is None else np.asarray(K, float)
        R, t = self.camera_pose(joints, mount)
        pc = (np.atleast_2d(points) - t) @ R
        uv = pc[:, :2] / pc[:, 2:3]
        return uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]

    # ------------------------------------------------------------------
    # localization
    # ------------------------------------------------------------------
    @staticmethod
    def _normalized(pixels, K, dist=None) -> np.ndarray:
        """Pixels -> undistorted normalized image coordinates ((N, 2))."""
        px = np.asarray(pixels, float).reshape(-1, 2)
        if dist is not None:
            import cv2

            return cv2.undistortPoints(px.reshape(-1, 1, 2), K, np.asarray(dist, float)).reshape(-1, 2)
        return (px - [K[0, 2], K[1, 2]]) / [K[0, 0], K[1, 1]]

    def rays(self, pixels, joints: dict, K=None, dist=None, mount=None):
        """Camera centre and unit world directions through ``pixels`` ((N, 2))."""
        K = self.default_intrinsics() if K is None else np.asarray(K, float)
        xy = self._normalized(pixels, K, dist)
        R, t = self.camera_pose(joints, mount)
        d = np.column_stack([xy, np.ones(len(xy))]) @ R.T
        return t, d / np.linalg.norm(d, axis=1, keepdims=True)

    @staticmethod
    def _surface(shape: str, params, table_z: float) -> np.ndarray:
        """Sample points whose projection outlines the object."""
        if shape == "sphere":
            cx, cy, r = params
            return np.array([cx, cy, table_z + r]) + r * _SPHERE
        cx, cy, R, H = params
        ring = np.column_stack([R * _RING, np.zeros(len(_RING))]) + [cx, cy, table_z]
        return np.vstack([ring + [0.0, 0.0, f * H] for f in np.linspace(0.0, 1.0, 9)])

    def locate_on_table(
        self,
        box,
        joints: dict,
        K=None,
        dist=None,
        mount=None,
        table_z: float = TABLE_Z,
        image_size: tuple[int, int] | None = None,
        shape: str = "upright",
    ) -> dict | None:
        """Object centre ``xyz`` (m), ``width`` and ``height`` from a box, or ``None``.

        ``box`` is ``(x1, y1, x2, y2)`` pixels in a frame of ``image_size``
        ``(w, h)`` taken at ``joints``. The box bottom's ray meets the table
        at the object's front edge; from there a resting object - a sphere
        (``shape="sphere"``) or an upright cylinder (``"upright"``) - is
        fitted so its projected outline matches the box edges. A box cut by
        the image's sides or bottom gives ``None``; a cut top edge is ignored.
        """
        x1, y1, x2, y2 = (float(v) for v in box)
        top_cut = False
        if image_size is not None:
            W, H = image_size
            if x1 <= 1 or x2 >= W - 1 or y2 >= H - 1:
                return None
            top_cut = y1 <= 1
        K = self.default_intrinsics(*(image_size or (None, None))) if K is None else np.asarray(K, float)
        uc, vc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        edges = [(x1, vc), (x2, vc), (uc, y1), (uc, y2)]
        o, d = self.rays(edges, joints, K, dist, mount)
        bottom = d[3]
        if o[2] <= table_z or bottom[2] >= -1e-6:
            return None
        R, t = self.camera_pose(joints, mount)
        front = o + bottom * ((table_z - o[2]) / bottom[2])
        half_w = 0.5 * (x2 - x1) / K[0, 0] * float((front - o) @ R[:, 2])
        horiz = np.array([bottom[0], bottom[1], 0.0])
        horiz /= max(np.linalg.norm(horiz), 1e-9)
        c0 = front + horiz * half_w
        params = np.array([c0[0], c0[1], half_w] if shape == "sphere" else [c0[0], c0[1], half_w, 2 * half_w])
        n = self._normalized(edges, K, dist)
        obs = np.array([n[0, 0], n[1, 0], n[2, 1], n[3, 1]])
        lo = np.array([-np.inf, -np.inf, 0.004] + ([] if shape == "sphere" else [0.004]))
        hi = np.array([np.inf, np.inf, 0.15] + ([] if shape == "sphere" else [0.4]))

        def outline(p):
            pc = (self._surface(shape, p, table_z) - t) @ R
            if (pc[:, 2] <= 1e-3).any():
                return None
            xy = pc[:, :2] / pc[:, 2:3]
            top = xy[:, 1].min()
            if top_cut:  # the sides are what's left inside the image; the top lies above it
                xy = xy[xy[:, 1] >= obs[2]]
                if not len(xy):
                    return None
                top = obs[2] + max(0.0, top - obs[2])
            return np.array([xy[:, 0].min(), xy[:, 0].max(), top, xy[:, 1].max()]) - obs

        r = outline(params)
        if r is None:
            return None
        lam = 1e-3
        for _ in range(40):
            J = np.empty((len(r), len(params)))
            for j in range(len(params)):
                dp = np.zeros(len(params))
                dp[j] = 1e-4
                rj = outline(params + dp)
                if rj is None:
                    return None
                J[:, j] = (rj - r) / 1e-4
            step = -np.linalg.solve(J.T @ J + lam * np.eye(len(params)), J.T @ r)
            p_new = np.clip(params + step, lo, hi)
            r_new = outline(p_new)
            if r_new is not None and r_new @ r_new < r @ r:
                params, r, lam = p_new, r_new, lam * 0.3
                if np.abs(step).max() < 1e-6:
                    break
            else:
                lam *= 10.0
                if lam > 1e6:
                    break
        if np.abs(r).max() * K[0, 0] > 3.0:  # the model can't explain the box
            return None
        if shape == "sphere":
            cx, cy, rad = params
            return {"xyz": np.array([cx, cy, table_z + rad]), "width": 2 * rad, "height": 2 * rad, "shape": shape}
        cx, cy, rad, h = params
        return {"xyz": np.array([cx, cy, table_z + 0.5 * h]), "width": 2 * rad, "height": h, "shape": shape}

    # ------------------------------------------------------------------
    # IK
    # ------------------------------------------------------------------
    def _pose(self, q, gripper: float) -> dict:
        return {**dict(zip(IK_JOINTS, (float(v) for v in q))), "wrist_roll": 0.0, "gripper": float(gripper)}

    def _eval(self, q, target, width: float, gripper: float):
        pose = self._pose(q, gripper)
        p = self.world.grasp_point(pose, width)
        axis_z = float(self.world.jaw_axis(pose)[2])
        return np.append(p - target, _AXIS_WEIGHT * axis_z)

    def lowest_point(self, joints: dict) -> float:
        """Lowest z (m) of the arm's collision geometry from the lower arm out."""
        with self._lock:
            d = self.world.fk(dict(joints))
            m = self.model
            low = math.inf
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
            for g in self._clear_geoms:
                c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
                corners = (c + signs * h) @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                low = min(low, float(corners[:, 2].min()))
            return low

    def ik(self, target, width: float = 0.04, gripper: float = OPEN_GRIPPER, seed: dict | None = None,
           iters: int = 80) -> tuple[dict, float, float]:
        """Best top-down pose for grasp point ``target``: ``(pose, pos_err_m, axis_z)``."""
        target = np.asarray(target, float)
        with self._lock:
            sx, sy = self.world.fk({}).body("shoulder").xpos[:2]
            pan0 = -math.degrees(math.atan2(target[1] - sy, target[0] - sx))
            seeds = [tuple(seed[k] for k in IK_JOINTS)] if seed else []
            seeds += [(pan0, *s) for s in _SEEDS]
            lo = np.array([self._limits[k][0] for k in IK_JOINTS])
            hi = np.array([self._limits[k][1] for k in IK_JOINTS])
            best = None
            for s in seeds:
                q = np.clip(np.array(s, float), lo, hi)
                lam = 0.01
                r = self._eval(q, target, width, gripper)
                for _ in range(iters):
                    if np.linalg.norm(r[:3]) < 3e-4 and abs(r[3]) < 0.005 * _AXIS_WEIGHT:
                        break
                    J = np.empty((4, 4))
                    for j in range(4):
                        dq = np.zeros(4)
                        dq[j] = 0.25
                        J[:, j] = (self._eval(q + dq, target, width, gripper) - r) / 0.25
                    step = -J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(4), r)
                    step *= min(1.0, 8.0 / max(np.abs(step).max(), 1e-9))
                    q_new = np.clip(q + step, lo, hi)
                    r_new = self._eval(q_new, target, width, gripper)
                    if r_new @ r_new < r @ r:
                        q, r, lam = q_new, r_new, max(lam * 0.5, 1e-4)
                    else:
                        lam *= 4.0
                        if lam > 10.0:
                            break
                cand = (float(r @ r), q.copy(), r.copy())
                if best is None or cand[0] < best[0]:
                    best = cand
                if np.linalg.norm(r[:3]) < 1e-3 and abs(r[3]) < 0.02 * _AXIS_WEIGHT:
                    break
            _, q, r = best
            return self._pose(q, gripper), float(np.linalg.norm(r[:3])), float(r[3] / _AXIS_WEIGHT)

    def solve_grasp(
        self,
        p,
        above_m: float = 0.06,
        width: float = 0.04,
        table_z: float = TABLE_Z,
        clearance_m: float = 0.003,
        tol_m: float = 0.003,
        max_tilt: float = 0.3,
        max_raise_m: float = 0.015,
        gripper: float = OPEN_GRIPPER,
    ) -> tuple[dict, dict] | None:
        """``(pregrasp, grasp)`` joint poses for an object centred at ``p``, or ``None``.

        The grasp puts ``p`` between the open jaws with the closing axis
        horizontal (within ``max_tilt``); if the jaw tips would touch the
        table it grasps up to ``max_raise_m`` higher. The pre-grasp is the
        same grasp ``above_m`` straight above. ``None`` means out of reach.
        """
        p = np.asarray(p, float)

        def ok(pose, err, tilt):
            return err <= tol_m and abs(tilt) <= max_tilt and self.lowest_point(pose) >= table_z + clearance_m

        grasp = None
        seed = None
        for raise_m in np.arange(0.0, max_raise_m + 1e-9, 0.003):
            pose, err, tilt = self.ik(p + [0.0, 0.0, raise_m], width, gripper, seed)
            if err <= tol_m and abs(tilt) <= max_tilt:
                seed = pose
                if ok(pose, err, tilt):
                    grasp = pose
                    break
        if grasp is None:
            return None
        pre, err, tilt = self.ik(self.world.grasp_point(grasp, width) + [0.0, 0.0, above_m], width, gripper, grasp)
        if not ok(pre, err, tilt):
            return None
        return pre, grasp

    def waypoints(self, start: dict, end: dict, width: float = 0.04, step_m: float = 0.015) -> list[dict]:
        """Poses that move the grasp point along the straight line from
        ``start``'s to ``end``'s, ``end`` included.

        Servos interpolate each joint on its own, so one move between two
        poses only roughly follows the line; short steps keep the jaws on it.
        """
        with self._lock:
            a, b = self.world.grasp_point(start, width), self.world.grasp_point(end, width)
            n = max(1, math.ceil(float(np.linalg.norm(b - a)) / step_m))
            out, seed = [], start
            for i in range(1, n):
                pose, err, _ = self.ik(a + (b - a) * (i / n), width, end["gripper"], seed)
                if err > 0.003:
                    return [end]
                out.append(pose)
                seed = pose
            return [*out, end]


def solve_mount(gripper_poses, board_poses):
    """Eye-in-hand calibration: the camera's mount on the gripper.

    ``gripper_poses`` are ``(R, t)`` gripper-in-base from FK, ``board_poses``
    the matching ``(R, t)`` board-in-camera from ``cv2.solvePnP``, for a
    board that stays put on the table. Returns ``(R, t, spread_m)``: the
    mount (camera in gripper, as :meth:`Kinematics.camera_pose` takes it)
    and how far the board's base-frame position wanders across the views -
    a few mm for a good calibration.
    """
    import cv2

    if len(gripper_poses) < 3 or len(gripper_poses) != len(board_poses):
        raise ValueError("need at least 3 matching gripper/board poses")
    Rg = [np.asarray(R, float) for R, _ in gripper_poses]
    tg = [np.asarray(t, float).reshape(3, 1) for _, t in gripper_poses]
    Rb = [np.asarray(R, float) for R, _ in board_poses]
    tb = [np.asarray(t, float).reshape(3, 1) for _, t in board_poses]
    R, t = cv2.calibrateHandEye(Rg, tg, Rb, tb, method=cv2.CALIB_HAND_EYE_PARK)
    t = t.reshape(3)
    origins = np.array([Rg[i] @ (R @ tb[i].reshape(3) + t) + tg[i].reshape(3) for i in range(len(Rg))])
    spread = float(np.linalg.norm(origins - origins.mean(axis=0), axis=1).max())
    return R, t, spread


_shared: dict = {}
_shared_lock = threading.Lock()


def shared_kinematics(gripper_closed_cal: float | None = None) -> Kinematics:
    """One :class:`Kinematics` per process and gripper floor (loading takes ~0.3 s).

    ``gripper_closed_cal`` is the stack's calibrated gripper value with the
    jaws touching (its ``GRIPPER_CLOSED_FLOOR``); it sets where the open
    moving jaw sits, and so the grasp point between the jaws.
    """
    with _shared_lock:
        if gripper_closed_cal not in _shared:
            cfg = None if gripper_closed_cal is None else {"gripper_map": {"closed_cal": float(gripper_closed_cal)}}
            _shared[gripper_closed_cal] = Kinematics(sim_cfg=cfg)
        return _shared[gripper_closed_cal]
