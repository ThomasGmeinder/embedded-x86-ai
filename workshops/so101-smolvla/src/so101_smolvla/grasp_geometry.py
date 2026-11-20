"""How the gripper meets a cube. One measure for every simulator and test."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# The largest angle an upright cube allows: the jaws close onto two edges.
WORST_FINGER_FACE_DEG = 45.0
FINGER_ANGLE_NOTE = (
    "The finger angle is the angle between the fingertip line and the nearest cube face",
    "at the first jaw contact. Zero is square onto the face. If the jaws never touch,",
    "the angle is taken at the closest approach.",
)


def finger_face_angle_deg(
    fingertip_a: np.ndarray, fingertip_b: np.ndarray, cube_rotation: np.ndarray
) -> float:
    """Angle between the fingertip line and the nearest vertical face of a cube.

    Zero means the jaws close straight onto a face. The fingertips and the
    cube's rotation matrix are given in one frame whose third axis is
    vertical; only the horizontal parts count. The rotation is the cube's
    orientation as it is now, so a turn during the test is included.
    """
    pinch = (
        np.asarray(fingertip_b, dtype=np.float64)[:2]
        - np.asarray(fingertip_a, dtype=np.float64)[:2]
    )
    pinch_norm = float(np.linalg.norm(pinch))
    if pinch_norm < 1e-9:
        return 90.0
    pinch = pinch / pinch_norm
    axes = np.asarray(cube_rotation, dtype=np.float64).reshape(3, 3)[:2, :2]
    errors = []
    for column in range(2):
        face = axes[:, column]
        face_norm = float(np.linalg.norm(face))
        if face_norm < 1e-9:
            continue
        cosine = abs(float(np.dot(pinch, face / face_norm)))
        errors.append(float(np.degrees(np.arccos(min(1.0, cosine)))))
    return min(errors) if errors else 90.0


@dataclass(frozen=True)
class GripSample:
    """How the gripper stands to a cube at one step of a test."""

    step: int
    # Gripper tip (halfway between the fingertips) to the cube's centre, metres.
    distance: float
    finger_face_deg: float
    # Reading of the wrist roll joint, degrees.
    roll_deg: float
    # A jaw touches the cube.
    contact: bool

    def as_result(self) -> dict[str, object]:
        """The entries every test result carries for its grip."""
        return {
            "grip": self.contact,
            "grip_step": self.step,
            "finger_face_deg": round(self.finger_face_deg, 2),
            "grip_roll_deg": round(self.roll_deg, 2),
        }


class GripMoment:
    """The step at which a test's grip is read.

    It is the first step at which a jaw touches the cube, or the closest
    approach of the gripper tip when the jaws never touch it.
    """

    def __init__(self) -> None:
        self._first_contact: GripSample | None = None
        self._closest: GripSample | None = None

    def observe(self, sample: GripSample) -> None:
        if sample.contact and self._first_contact is None:
            self._first_contact = sample
        if self._closest is None or sample.distance < self._closest.distance:
            self._closest = sample

    def chosen(self) -> GripSample:
        if self._closest is None:
            raise RuntimeError("no step of the test was observed")
        if self._first_contact is not None:
            return self._first_contact
        return self._closest


def finger_angle_summary(angles: Iterable[float]) -> str:
    """One report line on the finger-to-face angles of a run's tests."""
    values = [float(angle) for angle in angles]
    if not values:
        raise ValueError("a run without tests has no finger angles")
    return (
        f"Finger-to-face angle at grip: mean {float(np.mean(values)):.1f}°, "
        f"median {float(np.median(values)):.1f}°, max {max(values):.1f}°. Ideal is 0°."
    )
