"""Authoritative configuration for the isolated SAPIEN policy backend."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from so101_smolvla.reachable_workspace import (
    REACHABLE_CENTER,
    REACHABLE_HALF_EXTENT,
)

PROTOCOL_VERSION = 2
SAPIEN_COMMIT = "239b38e076387b11d5c0b0da29487edcbde5f905"
SAPIEN_SOURCE = Path("/tmp/Xbotics-SO101-Sim")
SAPIEN_PYTHON = Path(__file__).resolve().parents[2] / ".venv-sapien" / "bin" / "python"

@dataclass(frozen=True)
class TaskSpec:
    environment_id: str
    instruction: str
    output_label: str


TASK_SPECS = {
    "pick-place": TaskSpec(
        environment_id="SO101PickPlaceCube40-v1",
        instruction="Pick up a cube and place in the bin",
        output_label="random-sapien",
    ),
    "pick-place-cube20": TaskSpec(
        environment_id="SO101PickPlaceCube20-v1",
        instruction="Pick up a small cube and place in the bin",
        output_label="random-sapien-cube20",
    ),
    "stack2-training": TaskSpec(
        environment_id="SO101Stack2Training-v1",
        instruction="Stack the smaller cube on the larger one",
        output_label="random-stacking-sapien",
    ),
    # Diagnostic: the stack2-training cubes and sentence, seen through the
    # simulator's own top camera and started from the simulator's rest pose.
    "stack2-sim-view": TaskSpec(
        environment_id="SO101Stack2SimView-v1",
        instruction="Stack the smaller cube on the larger one",
        output_label="random-stacking-simview-sapien",
    ),
    # The stack2-training cubes, sentence and start poses, shown and driven as
    # the rig of the real stacking recordings did (stack_recording_rig.py).
    "stack2-recording-rig": TaskSpec(
        environment_id="SO101Stack2RecordingRig-v1",
        instruction="Stack the smaller cube on the larger one",
        output_label="random-stacking-recording-rig-sapien",
    ),
    "stack3-color": TaskSpec(
        environment_id="SO101Stack3Color-v1",
        instruction="",
        output_label="random-stacking-color-sapien",
    ),
    # Three 40 mm cubes (red, green, blue) and a larger bin. One cube is named
    # per command; the sentence stays the trained one of the 40 mm bin task.
    "pick-place-three-cubes": TaskSpec(
        environment_id="SO101PickPlaceThreeCubes-v1",
        instruction="Pick up a cube and place in the bin",
        output_label="random-three-cubes-sapien",
    ),
}
# Bin tasks of the pinned simulator; both draw objects from the central
# reachable workspace.
PICK_PLACE_TASKS = frozenset({"pick-place", "pick-place-cube20"})
THREE_CUBE_TASK = "pick-place-three-cubes"
TASK_ID = TASK_SPECS["pick-place"].environment_id
TASK_TEXT = TASK_SPECS["pick-place"].instruction
# What a pixel of an outline map shows. A scene that hands out outline maps
# numbers its pixels with these; the picture editor reads them.
OUTLINE_CODES = {
    "other": 0,
    "robot": 1,
    "bin": 2,
    "red": 3,
    "green": 4,
    "blue": 5,
}
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
ACTION_SHAPE = (6,)

ROBOT_ORANGE = (1.0, 0.35, 0.0, 1.0)
MOTOR_BLACK = (0.05, 0.05, 0.05, 1.0)
# Arm links whose printed parts are painted orange.
ORANGE_ROBOT_LINKS = frozenset(
    {
        "base_link",
        "shoulder_link",
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
    }
)
# The hand: everything past the wrist-roll joint. It is not painted, so it
# keeps the black the simulator gives the whole robot in its training pictures.
HAND_LINKS = frozenset(
    {
        "gripper_link",
        "moving_jaw_so101_v1_link",
        "wrist_camera_mount_link",
    }
)

BASE_ENVIRONMENT_ARGUMENTS = {
    "control_mode": "pd_joint_pos",
    "unit_convention": "real",
    "obs_type": "pixels_agent_pos",
    "sim_backend": "physx_cpu",
    "auto_reset": False,
    "observation_width": IMAGE_WIDTH,
    "observation_height": IMAGE_HEIGHT,
    "render_mode": "rgb_array",
}


def task_spec(task_mode: str) -> TaskSpec:
    try:
        return TASK_SPECS[task_mode]
    except KeyError as exc:
        raise ValueError(
            f"unsupported SAPIEN task {task_mode!r}; expected one of "
            f"{sorted(TASK_SPECS)}"
        ) from exc


def environment_arguments(task_mode: str) -> dict[str, object]:
    spec = task_spec(task_mode)
    arguments = {
        **BASE_ENVIRONMENT_ARGUMENTS,
        "task": spec.environment_id,
        "task_description": spec.instruction,
    }
    if task_mode in PICK_PLACE_TASKS:
        arguments.update(
            {
                "spawn_box_pos": REACHABLE_CENTER.tolist(),
                "spawn_box_half_size": float(REACHABLE_HALF_EXTENT[0]),
            }
        )
    return arguments


ENVIRONMENT_ARGUMENTS = environment_arguments("pick-place")
