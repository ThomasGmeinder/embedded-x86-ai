"""How the SAPIEN robot's gripper stands to a cube, in the terms of ``grasp_geometry``."""

from __future__ import annotations

import numpy as np

from so101_smolvla.grasp_geometry import finger_face_angle_deg


def grip_facts(agent, cube) -> dict[str, object]:
    """The gripper against one cube of a single-scene environment, as it stands now.

    ``contact`` is a jaw touching the cube, ``distance`` the gripper tip's
    distance from the cube's centre in metres, ``finger_face_deg`` the angle
    between the fingertip line and the cube's nearest vertical face.
    """
    tip_a, tip_b = (
        tip.pose.p[0].detach().cpu().numpy().astype(np.float64)
        for tip in (agent.finger1_tip, agent.finger2_tip)
    )
    pose = cube.pose
    centre = pose.p[0].detach().cpu().numpy().astype(np.float64)
    rotation = pose.to_transformation_matrix()[0, :3, :3].detach().cpu().numpy()
    return {
        "contact": bool(agent.is_touching(cube)[0]),
        "distance": float(np.linalg.norm((tip_a + tip_b) / 2 - centre)),
        "finger_face_deg": finger_face_angle_deg(tip_a, tip_b, rotation),
    }
