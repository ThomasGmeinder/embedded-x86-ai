"""Authoritative configuration for the isolated SAPIEN policy backend."""

from __future__ import annotations

from pathlib import Path

PROTOCOL_VERSION = 1
SAPIEN_COMMIT = "239b38e076387b11d5c0b0da29487edcbde5f905"
SAPIEN_SOURCE = Path("/tmp/Xbotics-SO101-Sim")
SAPIEN_PYTHON = Path(__file__).resolve().parents[2] / ".venv-sapien" / "bin" / "python"

TASK_ID = "SO101PickPlaceCube40-v1"
TASK_TEXT = "Pick up a cube and place in the bin"
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
ACTION_SHAPE = (6,)

ENVIRONMENT_ARGUMENTS = {
    "task": TASK_ID,
    "task_description": None,
    "control_mode": "pd_joint_pos",
    "unit_convention": "real",
    "obs_type": "pixels_agent_pos",
    "sim_backend": "physx_cpu",
    "auto_reset": False,
    "observation_width": IMAGE_WIDTH,
    "observation_height": IMAGE_HEIGHT,
    "render_mode": "rgb_array",
}
