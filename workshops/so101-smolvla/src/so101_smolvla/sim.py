"""MuJoCo SO-101 scene. This module owns physics, cameras, and the grasp reading."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from so101_smolvla.grasp_geometry import finger_face_angle_deg

ROOT = Path(__file__).resolve().parents[2]
SCENE = ROOT / "assets" / "robotstudio_so101" / "scene_task.xml"

ARM = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER = "gripper"
JOINTS = ARM + (GRIPPER,)

PHYSICS_HZ = 480
TICK_STEPS = 16
TICK_S = TICK_STEPS / PHYSICS_HZ
HOLD_S = 1.5
HOLD_STEPS = int(round(HOLD_S * PHYSICS_HZ))

IMAGE_W = 640
IMAGE_H = 480

# Home pose, radians. Gripper is overwritten with the measured open end.
HOME_ARM = np.array([0.0, -0.6, 0.8, 0.4, 0.0], dtype=np.float64)

LIFT_MARGIN_M = 0.04


def _name_id(model: mujoco.MjModel, kind: mujoco.mjtObj, name: str) -> int:
    index = mujoco.mj_name2id(model, kind, name)
    if index < 0:
        raise KeyError(name)
    return int(index)


class ArmSim:
    """One MuJoCo model. Callers pass commands in the joint contract."""

    def __init__(self, scene: Path | None = None) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(scene or SCENE))
        self.model.opt.timestep = 1.0 / PHYSICS_HZ
        self.data = mujoco.MjData(self.model)
        self.joint_qadr = {
            name: int(self.model.jnt_qposadr[_name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)])
            for name in JOINTS
        }
        self.act_id = {
            name: _name_id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in JOINTS
        }
        self.cam_id = {
            "top": _name_id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "top"),
            "wrist": _name_id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"),
        }
        cube_joint = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qadr = int(self.model.jnt_qposadr[cube_joint])
        self.cube_body = int(self.model.jnt_bodyid[cube_joint])
        self.cube_geom = _name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.box_body = _name_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "place_box")
        self.box_xy = np.array(self.model.body_pos[self.box_body][:2], dtype=np.float64)
        # Outer half-size of bin_2, 80 mm by 100 mm.
        self.bin_half = np.array([0.04, 0.05], dtype=np.float64)
        self.tip_a = _name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "moving_jaw_sph_tip1")
        self.tip_b = _name_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1")
        self.site = _name_id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
        self.jaw_geoms = self._jaw_geoms()
        grip = self.act_id[GRIPPER]
        low, high = (float(v) for v in self.model.actuator_ctrlrange[grip])
        self.closed_rad, self.open_rad = self._measure_ends(low, high)
        self.renderer = mujoco.Renderer(self.model, height=IMAGE_H, width=IMAGE_W)
        self.reset()

    def _jaw_geoms(self) -> set[int]:
        found = set()
        for index in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, index) or ""
            if "jaw" in name or name.startswith("camera_box"):
                found.add(index)
        return found

    def _gap(self) -> float:
        return float(np.linalg.norm(self.data.geom_xpos[self.tip_a] - self.data.geom_xpos[self.tip_b]))

    def _measure_ends(self, low: float, high: float) -> tuple[float, float]:
        gaps = {}
        for value in (low, high):
            mujoco.mj_resetData(self.model, self.data)
            self.data.ctrl[:] = 0.0
            self.data.ctrl[self.act_id[GRIPPER]] = value
            for _ in range(PHYSICS_HZ):
                mujoco.mj_step(self.model, self.data)
            gaps[value] = self._gap()
        closed = min(gaps, key=gaps.get)
        open_end = max(gaps, key=gaps.get)
        return float(closed), float(open_end)

    def gripper_to_rad(self, percent: float) -> float:
        span = self.open_rad - self.closed_rad
        return self.closed_rad + (float(percent) / 100.0) * span

    def rad_to_gripper(self, radians: float) -> float:
        span = self.open_rad - self.closed_rad
        return (float(radians) - self.closed_rad) / span * 100.0

    def read_state(self) -> np.ndarray:
        out = np.zeros(6, dtype=np.float64)
        for index, name in enumerate(ARM):
            out[index] = np.degrees(self.data.qpos[self.joint_qadr[name]])
        out[5] = self.rad_to_gripper(self.data.qpos[self.joint_qadr[GRIPPER]])
        return out

    def commanded(self) -> np.ndarray:
        out = np.zeros(6, dtype=np.float64)
        for index, name in enumerate(ARM):
            out[index] = np.degrees(self.data.ctrl[self.act_id[name]])
        out[5] = self.rad_to_gripper(self.data.ctrl[self.act_id[GRIPPER]])
        return out
        out = np.zeros(6, dtype=np.float64)
        for index, name in enumerate(ARM):
            out[index] = np.degrees(self.data.qpos[self.joint_qadr[name]])
        out[5] = self.rad_to_gripper(self.data.qpos[self.joint_qadr[GRIPPER]])
        return out

    def command(self, state: np.ndarray) -> None:
        values = np.asarray(state, dtype=np.float64).reshape(6)
        if not np.isfinite(values).all():
            raise ValueError("non-finite command")
        for index, name in enumerate(ARM):
            low, high = self.model.actuator_ctrlrange[self.act_id[name]]
            radians = np.deg2rad(values[index])
            self.data.ctrl[self.act_id[name]] = float(np.clip(radians, low, high))
        grip = self.gripper_to_rad(float(np.clip(values[5], 0.0, 100.0)))
        low, high = self.model.actuator_ctrlrange[self.act_id[GRIPPER]]
        self.data.ctrl[self.act_id[GRIPPER]] = float(np.clip(grip, low, high))

    def hold_pose(self, state: np.ndarray) -> None:
        """Put the arm on a contract pose without stepping.

        ``command`` owns the degree and gripper-percent conversion. This copies
        those clipped radians into ``qpos`` so the picture and the state match.
        """
        self.command(state)
        for name in JOINTS:
            joint = _name_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.data.qpos[self.joint_qadr[name]] = self.data.ctrl[self.act_id[name]]
            self.data.qvel[int(self.model.jnt_dofadr[joint])] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def command_radians(self, radians: np.ndarray) -> None:
        values = np.asarray(radians, dtype=np.float64).reshape(6)
        for index, name in enumerate(JOINTS):
            low, high = self.model.actuator_ctrlrange[self.act_id[name]]
            self.data.ctrl[self.act_id[name]] = float(np.clip(values[index], low, high))

    def step(self, n: int = TICK_STEPS) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)

    def reset(self, cube_xy: tuple[float, float] | None = None, gripper_open: bool = True) -> None:
        mujoco.mj_resetData(self.model, self.data)
        arm = HOME_ARM.copy()
        grip = self.open_rad if gripper_open else self.closed_rad
        q = np.concatenate([arm, [grip]])
        for index, name in enumerate(JOINTS):
            self.data.qpos[self.joint_qadr[name]] = q[index]
            self.data.ctrl[self.act_id[name]] = q[index]
        if cube_xy is None:
            cube = self.model.qpos0[self.cube_qadr : self.cube_qadr + 3]
        else:
            cube = (cube_xy[0], cube_xy[1], 0.02)
        self.data.qpos[self.cube_qadr : self.cube_qadr + 3] = cube
        self.data.qpos[self.cube_qadr + 3 : self.cube_qadr + 7] = (1.0, 0.0, 0.0, 0.0)
        mujoco.mj_forward(self.model, self.data)

    def _point_inside_geom(self, point: np.ndarray, gid: int) -> bool:
        local = self.data.geom_xmat[gid].reshape(3, 3).T @ (point - self.data.geom_xpos[gid])
        size = self.model.geom_size[gid]
        kind = int(self.model.geom_type[gid])
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return bool(np.all(np.abs(local) <= size))
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(np.linalg.norm(local)) <= float(size[0])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            along = float(np.clip(local[2], -size[1], size[1]))
            return float(np.linalg.norm(local - np.array([0.0, 0.0, along]))) <= float(size[0])
        return False

    def jaw_contains_cube_center(self) -> str | None:
        """The cube center is inside a jaw solid. Contact at the surface is not this."""
        point = self.cube_pos()
        for gid in self.jaw_geoms:
            if self._point_inside_geom(point, gid):
                return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or str(gid)
        return None

    def render(self, which: str) -> np.ndarray:
        buried = self.jaw_contains_cube_center()
        if buried is not None:
            raise RuntimeError(f"cube center is inside {buried}")
        self.renderer.update_scene(self.data, camera=self.cam_id[which])
        frame = self.renderer.render()
        return np.ascontiguousarray(frame)

    def cube_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[self.cube_qadr : self.cube_qadr + 3], dtype=np.float64)

    def finger_face_deg(self) -> float:
        """Angle between the fingertip line and the nearest vertical face of the cube."""
        return finger_face_angle_deg(
            self.data.geom_xpos[self.tip_a],
            self.data.geom_xpos[self.tip_b],
            self.data.xmat[self.cube_body],
        )

    def jaw_contact(self) -> bool:
        for index in range(self.data.ncon):
            geom1 = int(self.data.contact[index].geom1)
            geom2 = int(self.data.contact[index].geom2)
            pair = {geom1, geom2}
            if self.cube_geom in pair and (pair - {self.cube_geom}) & self.jaw_geoms:
                return True
        return False

    def gripper_force(self) -> float:
        return float(self.data.actuator_force[self.act_id[GRIPPER]])

    def held(self, empty_force: float) -> bool:
        return self.jaw_contact() and abs(self.gripper_force()) > abs(empty_force) + 0.05

    def above_table(self) -> bool:
        return float(self.cube_pos()[2]) > LIFT_MARGIN_M

    def place_bin(self, xy: tuple[float, float], yaw: float) -> None:
        """Move the tray. ``box_xy`` follows this body, which is the place target."""
        self.model.body_pos[self.box_body][0] = float(xy[0])
        self.model.body_pos[self.box_body][1] = float(xy[1])
        half = 0.5 * float(yaw)
        self.model.body_quat[self.box_body] = (np.cos(half), 0.0, 0.0, np.sin(half))
        self.box_xy = np.array(xy, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def in_box(self) -> bool:
        origin = self.data.xpos[self.box_body]
        rotation = self.data.xmat[self.box_body].reshape(3, 3)
        local = rotation.T @ (self.cube_pos() - origin)
        return (
            abs(local[0]) < self.bin_half[0]
            and abs(local[1]) < self.bin_half[1]
            and float(self.cube_pos()[2]) < 0.06
        )

    def close(self) -> None:
        self.renderer.close()
