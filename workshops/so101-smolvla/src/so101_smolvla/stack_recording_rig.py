"""Measured rig of the real stacking recordings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

WORKSHOP = Path(__file__).resolve().parents[2]
CONFIG_PATH = WORKSHOP / "configs" / "sapien_stack_recording_rig.json"
# The arm joints in the order of the robot's state and action numbers.
ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
# No joint of a calibrated arm reads farther than this from its true angle.
LARGEST_ZERO_POINT_DEG = 30.0


@dataclass(frozen=True)
class RecordingRig:
    """Where the recording robot's cameras and table were, and where each of
    its arm joints had its zero.

    The top camera is given in the robot base frame. The wrist camera is
    given on the simulated robot's ``wrist_camera_optical_frame`` link, which
    moves with the hand. A joint zero point is what has to be added to a
    recorded joint number to get the angle of the simulated robot, in the
    order of ``ARM_JOINTS``.

    The recording robot's arm had weight, and its motors held it with a
    position error. ``arm_drive_stiffness`` (N*m/rad) and
    ``arm_drive_damping`` (N*m*s/rad) are the joint drives under which the
    simulated arm, given its weight, follows commands as the recorded arm did,
    in the order of ``ARM_JOINTS``.
    """

    top_camera_position: tuple[float, float, float]
    top_camera_orientation_wxyz: tuple[float, float, float, float]
    top_camera_vertical_fov_rad: float
    wrist_camera_position: tuple[float, float, float]
    wrist_camera_orientation_wxyz: tuple[float, float, float, float]
    wrist_camera_vertical_fov_rad: float
    arm_zero_points_rad: tuple[float, float, float, float, float]
    arm_drive_stiffness: tuple[float, float, float, float, float]
    arm_drive_damping: tuple[float, float, float, float, float]
    table_below_robot_base: float
    table_picture: Path


def _camera(
    document: dict, name: str, frame: str
) -> tuple[tuple[float, ...], tuple[float, ...], float]:
    """Return position, orientation and vertical field of view of one camera."""
    camera = document[f"{name}_camera"]
    position = np.asarray(camera[f"position_{frame}_m"], dtype=np.float64)
    orientation = np.asarray(camera[f"orientation_{frame}_wxyz"], dtype=np.float64)
    field_of_view = np.deg2rad(float(camera["vertical_field_of_view_deg"]))
    if position.shape != (3,) or not np.isfinite(position).all():
        raise RuntimeError(f"invalid {name} camera position in {CONFIG_PATH}")
    if orientation.shape != (4,) or abs(np.linalg.norm(orientation) - 1.0) > 1e-6:
        raise RuntimeError(
            f"{name} camera orientation in {CONFIG_PATH} is not a unit quaternion"
        )
    if not 0.0 < field_of_view < np.pi:
        raise RuntimeError(f"invalid {name} camera field of view in {CONFIG_PATH}")
    return tuple(position.tolist()), tuple(orientation.tolist()), float(field_of_view)


def _arm_zero_points(document: dict) -> tuple[float, ...]:
    """Return the zero point of every arm joint in radians, in joint order."""
    zero_points = document["joint_zero_points_deg"]
    if set(zero_points) != set(ARM_JOINTS):
        raise RuntimeError(
            f"joint zero points in {CONFIG_PATH} must name exactly {ARM_JOINTS}"
        )
    degrees = np.asarray([zero_points[joint] for joint in ARM_JOINTS], dtype=np.float64)
    if not (np.abs(degrees) <= LARGEST_ZERO_POINT_DEG).all():
        raise RuntimeError(f"invalid joint zero point in {CONFIG_PATH}")
    return tuple(np.deg2rad(degrees).tolist())


def _arm_drives(document: dict) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return stiffness and damping of every arm joint's drive, in joint order."""
    drives = {
        name: drive for name, drive in document["arm_drives"].items() if name != "source"
    }
    if set(drives) != set(ARM_JOINTS):
        raise RuntimeError(f"arm drives in {CONFIG_PATH} must name exactly {ARM_JOINTS}")
    stiffness = np.asarray(
        [drives[joint]["stiffness_Nm_per_rad"] for joint in ARM_JOINTS], dtype=np.float64
    )
    damping = np.asarray(
        [drives[joint]["damping_Nms_per_rad"] for joint in ARM_JOINTS], dtype=np.float64
    )
    # A drive without stiffness lets the arm fall; one without damping swings.
    if not (np.isfinite(stiffness).all() and (stiffness > 0.0).all()):
        raise RuntimeError(f"invalid arm drive stiffness in {CONFIG_PATH}")
    if not (np.isfinite(damping).all() and (damping > 0.0).all()):
        raise RuntimeError(f"invalid arm drive damping in {CONFIG_PATH}")
    return tuple(stiffness.tolist()), tuple(damping.tolist())


def _load_rig() -> RecordingRig:
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    top = _camera(document, "top", "in_robot_base")
    wrist = _camera(document, "wrist", "on_optical_frame")
    stiffness, damping = _arm_drives(document)
    table_drop = float(document["table_below_robot_base_m"])
    # The picture's place is the measurement file's to name, relative to the workshop.
    picture = WORKSHOP / document["table_picture"]
    if not np.isfinite(table_drop) or table_drop < 0.0:
        raise RuntimeError(f"invalid table height in {CONFIG_PATH}")
    if not picture.is_file():
        raise RuntimeError(f"table picture {picture} is missing")
    return RecordingRig(
        top_camera_position=top[0],
        top_camera_orientation_wxyz=top[1],
        top_camera_vertical_fov_rad=top[2],
        wrist_camera_position=wrist[0],
        wrist_camera_orientation_wxyz=wrist[1],
        wrist_camera_vertical_fov_rad=wrist[2],
        arm_zero_points_rad=_arm_zero_points(document),
        arm_drive_stiffness=stiffness,
        arm_drive_damping=damping,
        table_below_robot_base=table_drop,
        table_picture=picture,
    )


RECORDING_RIG = _load_rig()
