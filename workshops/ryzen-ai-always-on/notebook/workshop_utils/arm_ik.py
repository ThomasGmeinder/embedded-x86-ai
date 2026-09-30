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

"""Damped-least-squares inverse kinematics for a robot arm chain.

Treat the goal as a 3D position for one body (the end-effector) and let IK solve
the joint angles: noise averages across the chain, joint limits are respected,
and the joints arrange themselves naturally. Used here to drive the SO-101
gripper to the hand-tele-op target.

Pure NumPy + MuJoCo Jacobians (``mj_jacSite`` when a tool-tip site is given,
else ``mj_jacBody``); no extra dependency.
"""

from __future__ import annotations

import numpy as np
import mujoco


class ArmIK:
    """Position IK for a named joint chain to a 3D world target.

    Generic over ``joint_names``; driven here with the SO-101's 5 arm joints
    (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll)."""

    def __init__(self, model, joint_names, hand_body, site=None):
        self.model = model
        self._qadr, self._dof, self._lo, self._hi = [], [], [], []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._qadr.append(int(model.jnt_qposadr[jid]))
            self._dof.append(int(model.jnt_dofadr[jid]))
            lo, hi = model.jnt_range[jid]
            self._lo.append(float(lo))
            self._hi.append(float(hi))
        self._dof = np.array(self._dof, dtype=int)
        self.hand_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, hand_body)
        # Optionally track a site (e.g. the gripper tip) instead of the body
        # origin — lets the workspace targets be the actual tool tip.
        self._site_id = (
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
            if site is not None
            else -1
        )

    @property
    def qpos_addr(self):
        return list(self._qadr)

    def ee_pos(self, data):
        """World position of the controlled point (site tip if set, else body)."""
        if self._site_id >= 0:
            return data.site_xpos[self._site_id].copy()
        return data.xpos[self.hand_bid].copy()

    def hand_pos(self, data):
        return data.xpos[self.hand_bid].copy()

    def solve(self, data, target, iters=8, damping=0.12, step=0.8, tol=5e-3):
        """Iterate DLS so the end-effector reaches ``target`` (world XYZ). Mutates qpos."""
        m = self.model
        jacp = np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, data)
            mujoco.mj_comPos(m, data)
            if self._site_id >= 0:
                err = target - data.site_xpos[self._site_id]
            else:
                err = target - data.xpos[self.hand_bid]
            if float(err @ err) < tol * tol:
                break
            if self._site_id >= 0:
                mujoco.mj_jacSite(m, data, jacp, None, self._site_id)
            else:
                mujoco.mj_jacBody(m, data, jacp, None, self.hand_bid)
            J = jacp[:, self._dof]  # 3 x n
            dq = J.T @ np.linalg.solve(J @ J.T + (damping**2) * np.eye(3), err)
            for i, qa in enumerate(self._qadr):
                data.qpos[qa] = min(
                    max(data.qpos[qa] + step * dq[i], self._lo[i]), self._hi[i]
                )
