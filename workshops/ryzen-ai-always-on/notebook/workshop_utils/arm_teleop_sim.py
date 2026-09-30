# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Portions of this file consist of AI-generated content. AI-assisted
# content has been reviewed and validated by the authors.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""SO-101 tele-op renderer (iGPU/EGL) for the pinch-to-pick-and-place demo.

Your hand is the controller: hand position drives the arm's end-effector
(inverse kinematics, not joint-copy — see ``arm_ik.ArmIK``), and a **pinch** of
your thumb + index closes the gripper to pick up a cube. Carry it by moving your
hand; open your fingers to drop it onto the green target circle to score a place.

The arm + gripper are driven **kinematically** (IK solves qpos, joints are pinned
through the physics step) — smooth and jitter-free. The cube is a real free body
under gravity: when you pinch near it we kinematically attach it to the tool tip
(robust for a live demo, looks exactly like a grip); releasing lets it fall.

The robot is The Robot Studio's **SO-101** (5-DOF arm + 1-DOF gripper jaw), a
low-cost LeRobot-ecosystem arm — the exact same hand-driven command stream would
drive the physical SO-101. Model from MuJoCo Menagerie (``robots/so101/``).
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path

# Render off-screen on the iGPU via EGL. Without this MuJoCo defaults to GLX,
# which needs an X display and a thread-current GL context and fails in our
# background render thread with "mujoco.FatalError: gladLoadGL error".
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from .arm_ik import ArmIK

_REPO_ROOT = Path(__file__).resolve().parents[2]
TELEOP_SCENE = _REPO_ROOT / "robots" / "so101" / "scene_teleop.xml"

_ARM_JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
_FINGER_JOINTS = ["gripper"]  # SO-101 has a single moving jaw
_EE_BODY = "gripper"  # end-effector body
_EE_SITE = "gripperframe"  # tool tip site (between the jaws)
# Jaw angle (rad) for fully open vs closed-on-cube. The cube is pinned to the
# tip kinematically, so closure is cosmetic; these just look right.
_JAW_OPEN, _JAW_CLOSED = 1.2, 0.15

# Hand-target workspace (tool tip, world metres). Tuned against the SO-101 reach
# map (robots/so101/scene_teleop.xml; the whole region is reachable with margin).
# Full 3D control: your hand drives all three axes (lateral Y, height Z, reach X).
X_REACH = 0.20  # neutral/idle depth: tip sits on the cube+pad at rest
X_NEAR, X_FAR = (
    0.13,
    0.30,
)  # reach band: hand far -> pulled in, hand close -> reached out
# (X_FAR spans the whole tabletop; the SO-101 reaches well past it)
Y_LEFT, Y_RIGHT = 0.13, -0.13  # mirrored: your left -> robot's left
Z_TOP, Z_BOTTOM = 0.12, 0.02  # hand up -> gripper up, hand down -> down to grab
# (the cube centre is z=0.016, so the tip must reach low to grab)
# Mid height: the idle/centred height and a safe carry height (clears the cube
# top while sliding, yet stays inside the grab radius).
Z_CARRY = 0.06

# Fixed target spot used when target randomisation is OFF (a pre-recorded clip,
# whose canned motion can only score if the drop location never moves). Pinned
# at x==X_REACH (the tool-tip plane) like the random targets, at the far end of
# the sampled lateral band (|y|<=0.13 with margin) so it stays reachable and a
# good carry from the cube's home at y=-0.13.
TARGET_FIXED_XY = (X_REACH, 0.11)

# Placement scoring (cube resting on the green target circle, not held). The disc
# radius is 0.04, and a cube set on the tabletop sits at z~+0.016, so the XY catch
# radius matches the disc and the Z ceiling only needs to admit a cube resting on
# the surface (well below a cube still held up at the carry height).
_PLACE_XY_TOL = 0.05
_PLACE_Z_MAX = 0.05

# A dropped cube is "pickable" only if it settles within the arm's lateral reach
# band (so it can be grabbed again). If it lands/rolls outside this, we respawn
# it at the start so the demo never gets stuck with an unreachable cube.
_PICK_X_TOL = 0.05
_PICK_Y_MARGIN = 0.03

_CUBE_HELD = np.array([0.20, 0.85, 1.0, 1.0])  # cyan tint while gripped


class ArmTeleopRenderer:
    def __init__(
        self,
        width: int = 600,
        height: int = 600,
        scene_path: str | Path = TELEOP_SCENE,
        camera: str = "demo",
        substeps: int = 4,
        x_near: float = X_NEAR,
        x_far: float = X_FAR,
        z_top: float = Z_TOP,
        z_bottom: float = Z_BOTTOM,
        randomize_target: bool = True,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height, width)
        self.cam = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        self.substeps = substeps
        # 3D reach box the tool tip can command (kept in sync with the hand
        # mapper's bands so the cube-reachability rescue matches reality).
        self.x_near, self.x_far = float(x_near), float(x_far)
        self.z_top, self.z_bottom = float(z_top), float(z_bottom)
        # Target randomisation: True relocates the target each round (live webcam
        # demo); False pins it once to TARGET_FIXED_XY so a pre-recorded clip's
        # canned motion always lands on a known spot (see _randomize_target).
        self.randomize_target = bool(randomize_target)

        self._home_key = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key)

        self.ik = ArmIK(self.model, _ARM_JOINTS, _EE_BODY, site=_EE_SITE)
        self.hand_bid = self.ik.hand_bid
        # qpos / dof addresses we hold fixed across the physics step.
        self._arm_qadr = list(self.ik.qpos_addr)
        self._arm_dof = [
            int(
                self.model.jnt_dofadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                ]
            )
            for n in _ARM_JOINTS
        ]
        self._fin_qadr, self._fin_dof = [], []
        for n in _FINGER_JOINTS:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                self._fin_qadr.append(int(self.model.jnt_qposadr[jid]))
                self._fin_dof.append(int(self.model.jnt_dofadr[jid]))

        self.cube_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cj = int(self.model.body_jntadr[self.cube_bid])
        self._cube_q = int(self.model.jnt_qposadr[cj])
        self._cube_v = int(self.model.jnt_dofadr[cj])
        self._cube_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_g"
        )
        self._cube_rgba0 = self.model.geom_rgba[self._cube_gid].copy()

        self.target_bid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._target_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_g"
        )
        self._target_rgba0 = self.model.geom_rgba[self._target_gid].copy()

        # Cache the cube's reference pose (qpos0, the XML body pose) so respawns
        # always drop it back at the start, independent of the 'home' keyframe.
        self._cube_home = self.model.qpos0[self._cube_q : self._cube_q + 7].copy()
        self.data.qpos[self._cube_q : self._cube_q + 7] = self._cube_home

        mujoco.mj_forward(self.model, self.data)
        self._target_xy = self.data.xpos[self.target_bid][:2].copy()

        self.grasped = False
        self.placed = 0
        self._placed_latch = False
        # After a successful place, leave the cube on the target briefly (so the
        # score + colours read), then auto-respawn it at the start for another go.
        self._respawn_delay = 0.7
        self._respawn_at: float | None = None
        self.tip = self.ik.ee_pos(self.data)

        # Start each session with the target placed: a fresh random reachable
        # spot for a live demo, or the single fixed spot for a pre-recorded clip.
        self._target_fixed_placed = False
        self._randomize_target()

    def _respawn_cube(self) -> None:
        """Return the cube to its start pose without touching the placed count."""
        self.data.qpos[self._cube_q : self._cube_q + 7] = self._cube_home
        self.data.qvel[self._cube_v : self._cube_v + 6] = 0.0
        self.grasped = False
        self._placed_latch = False

    def _randomize_target(self) -> None:
        """Relocate the green target circle to a fresh, reachable spot for a new round.

        Depth is pinned at ``X_REACH`` (the tool tip lives on that plane), so the
        target must stay at x==X_REACH and on the table surface; only the lateral
        Y varies. We sample within the reach band with margin and far enough from
        the cube's pickup spot (y=-0.13) that there is always a real carry, and we
        force a visible move by keeping the new y at least ~0.04 from the last one.
        ``target`` is a static world-child body (no joint), so writing body_pos +
        mj_forward is all that's needed to move it.

        When ``self.randomize_target`` is False (a pre-recorded clip, whose canned
        motion can only score if the drop location is deterministic) the target is
        pinned ONCE to ``TARGET_FIXED_XY`` and never moved again — subsequent
        reset/respawn calls are a no-op so the fixed spot always stays put."""
        if not self.randomize_target:
            if self._target_fixed_placed:
                return
            self.model.body_pos[self.target_bid][:2] = TARGET_FIXED_XY  # keep the XML z
            mujoco.mj_forward(self.model, self.data)
            self._target_xy = self.data.xpos[self.target_bid][:2].copy()
            self._target_fixed_placed = True
            return
        prev_y = (
            float(self._target_xy[1])
            if getattr(self, "_target_xy", None) is not None
            else None
        )
        y_t = random.uniform(-0.01, 0.11)
        for _ in range(16):  # nudge until it visibly moves
            if prev_y is None or abs(y_t - prev_y) >= 0.04:
                break
            y_t = random.uniform(-0.01, 0.11)
        self.model.body_pos[self.target_bid][:2] = (X_REACH, y_t)  # keep the XML z
        mujoco.mj_forward(self.model, self.data)
        self._target_xy = self.data.xpos[self.target_bid][:2].copy()

    def reset_object(self) -> None:
        self._respawn_cube()
        self._randomize_target()
        self._respawn_at = None
        self.placed = 0

    def _set_fingers(self, opening: float) -> None:
        # opening: 1.0 = fully open jaw, 0.0 = closed on the cube.
        t = float(np.clip(opening, 0.0, 1.0))
        v = _JAW_CLOSED + t * (_JAW_OPEN - _JAW_CLOSED)
        for qa in self._fin_qadr:
            self.data.qpos[qa] = v

    def _cube_pickable(self, cube) -> bool:
        """True if the cube is within the arm's lateral grab reach (re-grabbable)."""
        y_lo, y_hi = (
            min(Y_LEFT, Y_RIGHT) - _PICK_Y_MARGIN,
            max(Y_LEFT, Y_RIGHT) + _PICK_Y_MARGIN,
        )
        return (
            abs(float(cube[0]) - X_REACH) < _PICK_X_TOL
            and y_lo <= float(cube[1]) <= y_hi
            and float(cube[2]) < _PLACE_Z_MAX
        )

    def _reachable(self, cube, grab_radius: float) -> bool:
        """True if the tool tip can actually get within ``grab_radius`` of the cube.

        The tip now roams a 3D reach box (reach X in ``[x_near, x_far]``, lateral
        Y in ``[Y_RIGHT, Y_LEFT]``, height Z in ``[z_bottom, z_top]``), so the
        closest it can approach the cube is the box-clamped point. This mirrors
        the real grab test in ``render()`` and keeps the auto-respawn from firing
        on a cube that is in fact reachable in 3D (the old lateral-only test,
        pinned at ``X_REACH``/``Z_CARRY``, would have wrongly rescued it)."""
        cube = np.asarray(cube, float)
        best_tip = np.array(
            [
                float(
                    np.clip(
                        cube[0],
                        min(self.x_near, self.x_far),
                        max(self.x_near, self.x_far),
                    )
                ),
                float(np.clip(cube[1], min(Y_LEFT, Y_RIGHT), max(Y_LEFT, Y_RIGHT))),
                float(
                    np.clip(
                        cube[2],
                        min(self.z_bottom, self.z_top),
                        max(self.z_bottom, self.z_top),
                    )
                ),
            ]
        )
        return float(np.linalg.norm(best_tip - cube)) < grab_radius

    def _pin_cube(self, pos) -> None:
        self.data.qpos[self._cube_q : self._cube_q + 3] = pos
        self.data.qpos[self._cube_q + 3 : self._cube_q + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[self._cube_v : self._cube_v + 6] = 0.0

    def render(self, target: np.ndarray, grip: bool, grab_radius: float = 0.05):
        # 0) auto-respawn the cube a beat after a successful place, and relocate
        #    the target so the next round has a fresh spot to carry to.
        if self._respawn_at is not None and time.monotonic() >= self._respawn_at:
            self._respawn_cube()
            self._randomize_target()
            self._respawn_at = None

        # 1) IK the tip to the hand target; close the gripper on a pinch.
        # More iters + a larger step make the tip track the hand tightly each frame
        # (fewer iters made the arm visibly trail/"loosely follow" the target).
        self.ik.solve(
            self.data, np.asarray(target, float), iters=8, damping=0.1, step=0.8
        )
        self._set_fingers(0.0 if grip else 1.0)  # closed on the cube vs open
        for dof in self._arm_dof + self._fin_dof:
            self.data.qvel[dof] = 0.0
        mujoco.mj_kinematics(self.model, self.data)

        # 2) tool tip + grasp decision.
        self.tip = self.ik.ee_pos(self.data)
        dist = float(np.linalg.norm(self.data.xpos[self.cube_bid] - self.tip))
        if grip and (self.grasped or dist < grab_radius):
            self.grasped = True
        elif not grip:
            self.grasped = False
        if self.grasped:
            self._pin_cube(self.tip)

        # While the cube is sitting idle at its start (not grasped, not mid-place),
        # glue it to the exact home pose. The open jaws hovering over the cube would
        # otherwise nudge this free body a few cm out of the arm's (thin) lateral
        # reach, stranding it "just out of reach" — the reason a manual reset was
        # needed. Gluing keeps it perfectly grabbable until you actually pinch it.
        home_xy = self._cube_home[:2]
        cube_xy = self.data.xpos[self.cube_bid][:2]
        glue_home = (
            not self.grasped
            and self._respawn_at is None
            and not self._placed_latch
            and abs(float(cube_xy[0]) - float(home_xy[0])) < 0.03
            and abs(float(cube_xy[1]) - float(home_xy[1])) < 0.03
        )
        if glue_home:
            self._pin_cube(self._cube_home[:3])

        # 3) step physics for the cube while the arm/gripper stay put.
        hold = {qa: self.data.qpos[qa] for qa in self._arm_qadr + self._fin_qadr}
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            for qa, v in hold.items():
                self.data.qpos[qa] = v
            for dof in self._arm_dof + self._fin_dof:
                self.data.qvel[dof] = 0.0
            if self.grasped:
                self._pin_cube(self.tip)
            elif glue_home:
                self._pin_cube(self._cube_home[:3])

        # 4) placement scoring: cube resting on the green target circle, not held.
        cube = self.data.xpos[self.cube_bid]
        on_target = (
            np.linalg.norm(cube[:2] - self._target_xy) < _PLACE_XY_TOL
            and cube[2] < _PLACE_Z_MAX
            and not self.grasped
        )
        if on_target and not self._placed_latch:
            self.placed += 1
            self._placed_latch = True
            self._respawn_at = time.monotonic() + self._respawn_delay
        elif not on_target:
            self._placed_latch = False

        # 4b) rescue: if a dropped cube settles out of the arm's reach, return it
        #     to the start so the demo can't get stuck with an unreachable cube.
        if not self.grasped and self._respawn_at is None and not self._placed_latch:
            cube_speed = float(
                np.linalg.norm(self.data.qvel[self._cube_v : self._cube_v + 3])
            )
            if cube_speed < 0.02 and not self._reachable(cube, grab_radius):
                self._respawn_cube()

        # 5) feedback colours.
        self.model.geom_rgba[self._cube_gid] = (
            _CUBE_HELD if self.grasped else self._cube_rgba0
        )
        self.model.geom_rgba[self._target_gid] = (
            np.array([0.4, 1.0, 0.5, 0.8]) if self._placed_latch else self._target_rgba0
        )

        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()

    def close(self) -> None:
        try:
            self.renderer.close()
        except Exception:
            pass
