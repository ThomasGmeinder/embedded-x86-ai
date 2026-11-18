from so101_smolvla.sapien_config import MOTOR_BLACK, ROBOT_ORANGE
from so101_smolvla.sapien_env_worker import _robot_part_color


def test_printed_robot_parts_are_orange_and_motors_stay_black():
    assert _robot_part_color(
        "upper_arm_link", "/assets/upper_arm_so101_v1.stl"
    ) == ROBOT_ORANGE
    assert _robot_part_color(
        "upper_arm_link", "/assets/sts3215_03a_v1.stl"
    ) == MOTOR_BLACK


def test_external_camera_rig_is_not_recolored():
    assert _robot_part_color(
        "top_camera_mount_link", "/assets/top_camera_mount.stl"
    ) is None
