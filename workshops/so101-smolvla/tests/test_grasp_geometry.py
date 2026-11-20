import numpy as np
import pytest

from so101_smolvla.grasp_geometry import (
    WORST_FINGER_FACE_DEG,
    GripMoment,
    GripSample,
    finger_angle_summary,
    finger_face_angle_deg,
)


def _turned(yaw_deg: float) -> np.ndarray:
    """Rotation matrix of a cube turned about the vertical."""
    cosine, sine = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def _sample(step: int, distance: float, contact: bool, angle: float = 0.0) -> GripSample:
    return GripSample(
        step=step, distance=distance, finger_face_deg=angle, roll_deg=0.0, contact=contact
    )


@pytest.mark.parametrize(
    ("yaw_deg", "expected"),
    ((0.0, 0.0), (90.0, 0.0), (180.0, 0.0), (30.0, 30.0), (60.0, 30.0), (-20.0, 20.0), (45.0, 45.0)),
)
def test_finger_angle_is_the_turn_of_the_cube_against_the_fingertip_line(yaw_deg, expected):
    # The fingertips lie on a line along x, at different heights.
    left, right = np.array([0.10, 0.0, 0.03]), np.array([0.14, 0.0, 0.01])
    angle = finger_face_angle_deg(left, right, _turned(yaw_deg))
    assert angle == pytest.approx(expected, abs=1e-6)
    assert angle <= WORST_FINGER_FACE_DEG + 1e-6
    # Which fingertip is named first makes no difference.
    assert finger_face_angle_deg(right, left, _turned(yaw_deg)) == pytest.approx(angle)


def test_finger_angle_follows_the_fingertip_line_as_well_as_the_cube():
    # Fingertip line turned by 25 degrees, cube square to the axes.
    direction = np.array([np.cos(np.deg2rad(25.0)), np.sin(np.deg2rad(25.0)), 0.0])
    origin = np.array([0.2, 0.1, 0.02])
    assert finger_face_angle_deg(origin, origin + 0.03 * direction, np.eye(3)) == (
        pytest.approx(25.0, abs=1e-6)
    )
    # Turning the cube by the same 25 degrees brings the jaws square onto a face.
    assert finger_face_angle_deg(origin, origin + 0.03 * direction, _turned(25.0)) == (
        pytest.approx(0.0, abs=1e-6)
    )


def test_fingertips_above_each_other_have_no_fingertip_line():
    tip = np.array([0.2, 0.0, 0.02])
    assert finger_face_angle_deg(tip, tip + [0.0, 0.0, 0.03], np.eye(3)) == 90.0


def test_grip_is_read_at_the_first_jaw_contact():
    moment = GripMoment()
    for sample in (
        _sample(0, 0.10, False),
        _sample(1, 0.03, True, angle=12.0),
        _sample(2, 0.01, True, angle=3.0),
        _sample(3, 0.005, False),
    ):
        moment.observe(sample)
    chosen = moment.chosen()
    assert (chosen.step, chosen.contact, chosen.finger_face_deg) == (1, True, 12.0)
    assert chosen.as_result() == {
        "grip": True,
        "grip_step": 1,
        "finger_face_deg": 12.0,
        "grip_roll_deg": 0.0,
    }


def test_grip_without_any_contact_is_read_at_the_closest_approach():
    moment = GripMoment()
    for sample in (_sample(0, 0.10, False), _sample(1, 0.02, False), _sample(2, 0.06, False)):
        moment.observe(sample)
    chosen = moment.chosen()
    assert (chosen.step, chosen.contact) == (1, False)


def test_a_grip_that_was_never_observed_is_an_error_not_a_value():
    with pytest.raises(RuntimeError, match="no step"):
        GripMoment().chosen()
    with pytest.raises(ValueError, match="no finger angles"):
        finger_angle_summary([])


def test_report_line_gives_mean_median_and_largest_angle():
    assert finger_angle_summary([0.0, 10.0, 35.0]) == (
        "Finger-to-face angle at grip: mean 15.0°, median 10.0°, max 35.0°. Ideal is 0°."
    )
