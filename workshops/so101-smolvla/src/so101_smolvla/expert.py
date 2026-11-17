"""Scripted pick-and-place expert. It may read the cube pose. The policy never does."""

from __future__ import annotations

import mujoco
import numpy as np

from so101_smolvla.sim import ARM, HOLD_STEPS, PHYSICS_HZ, TICK_STEPS, ArmSim


class Expert:
    def __init__(self, sim: ArmSim) -> None:
        self.sim = sim
        model = sim.model
        self._body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._dofs = []
        self._qadr = []
        for name in ARM:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._dofs.append(int(model.jnt_dofadr[joint]))
            self._qadr.append(sim.joint_qadr[name])

    def _midpoint(self) -> np.ndarray:
        data = self.sim.data
        return 0.5 * (data.geom_xpos[self.sim.tip_a] + data.geom_xpos[self.sim.tip_b])

    def _ik_once(self, target: np.ndarray) -> None:
        sim = self.sim
        model, data = sim.model, sim.data
        mujoco.mj_forward(model, data)
        point = self._midpoint()
        mujoco.mj_jac(model, data, self._jacp, self._jacr, point, self._body)
        tool = data.site_xmat[sim.site].reshape(3, 3)[:, 0]
        error = np.concatenate(
            [target - point, 0.35 * np.cross(tool, np.array([0.0, 0.0, -1.0]))]
        )
        jac = np.vstack([self._jacp[:, self._dofs], 0.35 * self._jacr[:, self._dofs]])
        step = jac.T @ np.linalg.solve(jac @ jac.T + 2e-2 * np.eye(6), error)
        for index, address in enumerate(self._qadr):
            name = ARM[index]
            low, high = model.jnt_range[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            data.ctrl[sim.act_id[name]] = float(np.clip(data.qpos[address] + 0.7 * step[index], low, high))

    def _apply(self, grip_rad: float) -> None:
        state = np.zeros(6, dtype=np.float64)
        for index, name in enumerate(ARM):
            state[index] = np.degrees(self.sim.data.ctrl[self.sim.act_id[name]])
        state[5] = self.sim.rad_to_gripper(grip_rad)
        self.sim.command(state)

    def _go(self, points: list[np.ndarray], grip: float, repeats: int = 20) -> None:
        for target in points:
            for _ in range(repeats):
                self._ik_once(target)
                self._apply(grip)
                self.sim.step(6)

    def run(self, cube_xy: tuple[float, float], azimuth: float = 0.0, on_tick=None) -> bool:
        """Pick the cube, hold it for 3 s, and place it in the box.

        ``azimuth`` shifts the approach so repeated takes are not one spline.
        ``on_tick`` runs after every 1/30 s control tick.
        """
        sim = self.sim
        sim.reset(cube_xy=cube_xy)
        offset = 0.004 * np.array([np.cos(azimuth), np.sin(azimuth)])
        cube = np.asarray(cube_xy, dtype=np.float64) + offset
        opened = sim.open_rad
        closed = sim.closed_rad

        def tick() -> None:
            sim.step(TICK_STEPS)
            if on_tick is not None:
                on_tick()

        def go(points: list[np.ndarray], grip: float, repeats: int = 12) -> None:
            for target in points:
                for _ in range(repeats):
                    self._ik_once(target)
                    self._apply(grip)
                    tick()

        def dwell(grip: float, count: int) -> None:
            self._apply(grip)
            for _ in range(count):
                tick()

        go([np.array([cube[0], cube[1], 0.12]), np.array([cube[0], cube[1], 0.045])], opened)
        dwell(closed, PHYSICS_HZ // TICK_STEPS)
        go([np.array([cube[0], cube[1], 0.16])], closed)
        if not (sim.jaw_contact() and sim.above_table()):
            return False
        dwell(closed, HOLD_STEPS // TICK_STEPS)
        if not (sim.jaw_contact() and sim.above_table()):
            return False
        box = sim.box_xy
        go(
            [
                np.array([cube[0], cube[1], 0.16]),
                np.array([box[0], box[1], 0.16]),
                np.array([box[0], box[1], 0.06]),
            ],
            closed,
            repeats=16,
        )
        dwell(opened, PHYSICS_HZ // TICK_STEPS)
        return bool(sim.in_box())
