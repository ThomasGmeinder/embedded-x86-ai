from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from so101_smolvla import stack_recording_rig
from so101_smolvla.policy_env import (
    PolicyObservation,
    PolicyStep,
    random_stacking_output_directory,
)
from so101_smolvla.sapien_config import IMAGE_SHAPE, task_spec
from so101_smolvla.sapien_rollout import STACK_COMMANDS, _write_stack_report
from so101_smolvla.sapien_stack_session import (
    StackCommandSession,
    parse_stack_command,
    target_can_support,
)
from so101_smolvla.stack_initial_states import (
    TRAINING_INITIAL_STATES,
    draw_stack_initial_state,
)
from so101_smolvla.stack_layouts import (
    COLOR_CUBE_CLEARANCES,
    COLOR_CUBE_DISTANCE,
    TRAINING_CUBE_CLEARANCES,
    TRAINING_CUBE_DISTANCE,
    TRAINING_CUBE_SIZES,
    draw_stack_layout,
)
from so101_smolvla.reachable_workspace import is_reachable_xy


def _observation() -> PolicyObservation:
    return PolicyObservation(
        state=np.zeros(6),
        top=np.zeros((2, 2, 3), dtype=np.uint8),
        wrist=np.zeros((2, 2, 3), dtype=np.uint8),
    )


class FakePolicy:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


class FakeStackEnvironment:
    def __init__(self, succeed: bool = True) -> None:
        self.succeed = succeed
        self.command: tuple[str, str] | None = None
        self.facts = {
            "cube_positions": {
                "red": [0.4, 0.0, 0.02],
                "green": [0.5, 0.0, 0.02],
                "blue": [0.6, 0.0, 0.02],
            },
            "table_supported": {"red": True, "green": True, "blue": True},
            "interfaces": {
                f"{source}_on_{target}": {"supported": False}
                for source in ("red", "green", "blue")
                for target in ("red", "green", "blue")
                if source != target
            },
            "grasped": {"red": False, "green": False, "blue": False},
            "grip": {
                color: {"contact": False, "distance": 0.2, "finger_face_deg": 40.0}
                for color in ("red", "green", "blue")
            },
        }
        self.steps = 0

    def set_command(self, source: str, target: str) -> dict[str, object]:
        self.command = (source, target)
        return self.facts

    def step(self, action: np.ndarray) -> PolicyStep:
        assert self.command is not None
        source, target = self.command
        self.steps += 1
        # The gripper comes closer with every step and turns towards the
        # face of the cube it was told to move.
        self.facts["grip"][source] = {
            "contact": self.succeed,
            "distance": 0.2 / (1 + self.steps),
            "finger_face_deg": 40.0 / (1 + self.steps),
        }
        if self.succeed:
            target_position = self.facts["cube_positions"][target]
            self.facts["cube_positions"][source] = [
                target_position[0],
                target_position[1],
                target_position[2] + 0.04,
            ]
            self.facts["table_supported"][source] = False
            self.facts["interfaces"][f"{source}_on_{target}"]["supported"] = True
        return PolicyStep(
            observation=_observation(),
            commanded=action,
            achieved=action,
            success=self.succeed,
            facts=self.facts,
        )


class DisplacedWithoutGraspEnvironment(FakeStackEnvironment):
    def step(self, action: np.ndarray) -> PolicyStep:
        self.facts["cube_positions"]["green"][0] += 0.03
        return super().step(action)


def test_measured_initial_states_are_complete_and_replay_by_index():
    assert TRAINING_INITIAL_STATES.shape == (200, 6)
    assert np.isfinite(TRAINING_INITIAL_STATES).all()
    assert np.array_equal(draw_stack_initial_state(17), TRAINING_INITIAL_STATES[17])
    with pytest.raises(ValueError, match="invalid stacking state index"):
        draw_stack_initial_state(200)


def test_task_specs_keep_training_and_color_tasks_isolated():
    baseline = task_spec("stack2-training")
    extension = task_spec("stack3-color")
    assert baseline.instruction == "Stack the smaller cube on the larger one"
    assert baseline.environment_id != extension.environment_id
    assert extension.instruction == ""


@pytest.mark.parametrize("task_mode", ("stack2-sim-view", "stack2-recording-rig"))
def test_two_cube_variants_keep_the_trained_stacking_sentence(tmp_path, task_mode):
    baseline = task_spec("stack2-training")
    variant = task_spec(task_mode)
    assert variant.instruction == baseline.instruction
    assert variant.environment_id != baseline.environment_id
    assert variant.output_label != baseline.output_label
    assert STACK_COMMANDS[task_mode] == STACK_COMMANDS["stack2-training"]
    directory = random_stacking_output_directory(
        tmp_path, "sapien", task_mode, "stamp"
    )
    assert directory.name == f"{variant.output_label}-stamp"


def test_recording_rig_is_complete_and_its_table_picture_fits_the_camera():
    rig = stack_recording_rig.RECORDING_RIG
    document = json.loads(stack_recording_rig.CONFIG_PATH.read_text(encoding="utf-8"))
    assert rig.top_camera_position == tuple(
        document["top_camera"]["position_in_robot_base_m"]
    )
    assert np.linalg.norm(rig.top_camera_orientation_wxyz) == pytest.approx(
        1.0, abs=1e-6
    )
    assert np.rad2deg(rig.top_camera_vertical_fov_rad) == pytest.approx(
        document["top_camera"]["vertical_field_of_view_deg"]
    )
    assert rig.wrist_camera_position == tuple(
        document["wrist_camera"]["position_on_optical_frame_m"]
    )
    assert np.linalg.norm(rig.wrist_camera_orientation_wxyz) == pytest.approx(
        1.0, abs=1e-6
    )
    assert np.rad2deg(rig.wrist_camera_vertical_fov_rad) == pytest.approx(
        document["wrist_camera"]["vertical_field_of_view_deg"]
    )
    assert np.rad2deg(rig.arm_zero_points_rad) == pytest.approx(
        [
            document["joint_zero_points_deg"][joint]
            for joint in stack_recording_rig.ARM_JOINTS
        ]
    )
    assert rig.arm_drive_stiffness == tuple(
        document["arm_drives"][joint]["stiffness_Nm_per_rad"]
        for joint in stack_recording_rig.ARM_JOINTS
    )
    assert rig.arm_drive_damping == tuple(
        document["arm_drives"][joint]["damping_Nms_per_rad"]
        for joint in stack_recording_rig.ARM_JOINTS
    )
    assert rig.table_below_robot_base == document["table_below_robot_base_m"]
    picture = cv2.imread(str(rig.table_picture))
    assert picture is not None and picture.shape == IMAGE_SHAPE


@pytest.mark.parametrize(
    ("change", "message"),
    (
        (lambda rig: rig["top_camera"].update(orientation_in_robot_base_wxyz=[1, 1, 0, 0]),
         "top camera orientation .* not a unit quaternion"),
        (lambda rig: rig["wrist_camera"].update(orientation_on_optical_frame_wxyz=[1, 1, 0, 0]),
         "wrist camera orientation .* not a unit quaternion"),
        (lambda rig: rig["wrist_camera"].update(position_on_optical_frame_m=[0.0, 0.0]),
         "invalid wrist camera position"),
        (lambda rig: rig["wrist_camera"].update(vertical_field_of_view_deg=180.0),
         "invalid wrist camera field of view"),
        (lambda rig: rig["joint_zero_points_deg"].pop("elbow_flex"),
         "joint zero points .* must name exactly"),
        (lambda rig: rig["joint_zero_points_deg"].update(gripper=1.0),
         "joint zero points .* must name exactly"),
        (lambda rig: rig["joint_zero_points_deg"].update(elbow_flex=45.0),
         "invalid joint zero point"),
        (lambda rig: rig["joint_zero_points_deg"].update(shoulder_pan=float("nan")),
         "invalid joint zero point"),
        (lambda rig: rig["arm_drives"].pop("elbow_flex"),
         "arm drives .* must name exactly"),
        (lambda rig: rig["arm_drives"]["elbow_flex"].update(stiffness_Nm_per_rad=0.0),
         "invalid arm drive stiffness"),
        (lambda rig: rig["arm_drives"]["elbow_flex"].update(damping_Nms_per_rad=-1.0),
         "invalid arm drive damping"),
        (lambda rig: rig.update(table_below_robot_base_m=-0.01), "invalid table height"),
        (lambda rig: rig.update(table_picture="absent.png"), "is missing"),
    ),
)
def test_recording_rig_rejects_an_invalid_measurement(
    tmp_path, monkeypatch, change, message
):
    document = json.loads(stack_recording_rig.CONFIG_PATH.read_text(encoding="utf-8"))
    change(document)
    broken = tmp_path / "rig.json"
    broken.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(stack_recording_rig, "CONFIG_PATH", broken)
    with pytest.raises(RuntimeError, match=message):
        stack_recording_rig._load_rig()


def test_training_cube_sizes_match_the_real_kit_assets():
    assert TRAINING_CUBE_SIZES == {"red": 0.020, "blue": 0.040}


@pytest.mark.parametrize("task_mode", ("stack2-training", "stack3-color"))
def test_layout_seed_replays_and_every_cube_stays_reachable(task_mode):
    for seed in range(100):
        first = draw_stack_layout(task_mode, seed)
        assert first == draw_stack_layout(task_mode, seed)
        points = np.asarray(list(first.values()))
        if task_mode == "stack2-training":
            minimum_distance = TRAINING_CUBE_DISTANCE
            clearances = TRAINING_CUBE_CLEARANCES
        else:
            minimum_distance = COLOR_CUBE_DISTANCE
            clearances = COLOR_CUBE_CLEARANCES
        assert all(
            is_reachable_xy(point, clearance)
            for point, clearance in zip(points, clearances, strict=True)
        )
        for left in range(len(points)):
            for right in range(left + 1, len(points)):
                assert np.linalg.norm(points[left] - points[right]) >= minimum_distance


def test_two_cubes_may_stand_as_close_as_in_the_tests_that_gave_10_stacks_in_20():
    # Layout seed 1 takes its first candidate, 54.6 mm apart. A rule that asks
    # for more room skips it and so changes the random stacking tests.
    assert (TRAINING_CUBE_DISTANCE, COLOR_CUBE_DISTANCE) == (0.040, 0.060)
    layout = draw_stack_layout("stack2-training", 1)
    assert layout["red"] == pytest.approx(
        [0.27202995890044357, 0.12235170189116999], abs=1e-12
    )
    assert layout["blue"] == pytest.approx(
        [0.2189612946132063, 0.10935044411174152], abs=1e-12
    )


def test_command_parser_normalizes_and_rejects_invalid_requests():
    parsed = parse_stack_command(" Stack green cube on RED ")
    assert parsed.canonical == "put green cube on red cube"
    with pytest.raises(ValueError, match="unknown cube color"):
        parse_stack_command("put purple cube on red cube")
    with pytest.raises(ValueError, match="must differ"):
        parse_stack_command("put red cube on red cube")
    with pytest.raises(ValueError, match="expected"):
        parse_stack_command("please move green")


def test_second_command_requires_an_independently_supported_target():
    environment = FakeStackEnvironment()
    command = parse_stack_command("put blue cube on green cube")
    environment.facts["table_supported"]["green"] = False
    assert not target_can_support(command, environment.facts)
    environment.facts["interfaces"]["green_on_red"]["supported"] = True
    assert target_can_support(command, environment.facts)


def test_persistent_session_keeps_scene_and_resets_only_policy_queue():
    environment = FakeStackEnvironment()
    policy = FakePolicy()
    session = StackCommandSession(
        environment=environment,
        policy=policy,
        observation=_observation(),
        action=lambda _observation, _task: np.zeros(6),
        max_steps_per_command=2,
    )
    first = session.execute("put green cube on red cube")
    second = session.execute("put blue cube on green cube")
    assert first.success and second.success
    assert session.completed_interfaces == {"green_on_red", "blue_on_green"}
    assert policy.resets == 2


def test_partial_session_reports_no_lift_and_stops_at_horizon():
    environment = FakeStackEnvironment(succeed=False)
    session = StackCommandSession(
        environment=environment,
        policy=FakePolicy(),
        observation=_observation(),
        action=lambda _observation, _task: np.zeros(6),
        max_steps_per_command=2,
    )
    trace = session.execute("put green cube on red cube")
    assert not trace.success
    assert trace.failure_stage == "no lift"
    assert len(trace.target) == 2
    # The jaws never touch the cube: the grip is read at the closest approach,
    # the second of the two steps.
    assert not trace.grip.contact
    assert trace.grip.step == 1
    assert trace.grip.finger_face_deg == pytest.approx(40.0 / 3)


def test_session_reads_the_grip_at_the_first_jaw_contact_with_the_commanded_cube():
    environment = FakeStackEnvironment()
    command = np.zeros(6)
    command[stack_recording_rig.ARM_JOINTS.index("wrist_roll")] = -17.5
    session = StackCommandSession(
        environment=environment,
        policy=FakePolicy(),
        observation=_observation(),
        action=lambda _observation, _task: command,
        max_steps_per_command=3,
    )
    trace = session.execute("put green cube on red cube")
    assert trace.grip.contact
    assert trace.grip.step == 0
    assert trace.grip.finger_face_deg == pytest.approx(20.0)
    # The wrist roll is the joint reading of that step.
    assert trace.grip.roll_deg == -17.5


def test_stack_report_gives_the_finger_angle_of_every_command(tmp_path):
    def stage(command: str, success: bool, contact: bool, angle: float, roll: float):
        return {
            "canonical_command": command,
            "success": success,
            "grip": contact,
            "finger_face_deg": angle,
            "grip_roll_deg": roll,
        }

    rows = [
        {
            "task_mode": "stack3-color",
            "environment_seed": 0,
            "policy_seed": 0,
            "success": True,
            "failure_stage": None,
            "commands": [
                stage("put green cube on red cube", True, True, 4.0, -12.5),
                stage("put blue cube on green cube", True, True, 10.0, 3.0),
            ],
        },
        {
            "task_mode": "stack3-color",
            "environment_seed": 1,
            "policy_seed": 1,
            "success": False,
            "failure_stage": "first stack failed",
            "commands": [stage("put green cube on red cube", False, False, 40.0, 0.5)],
        },
    ]
    text = _write_stack_report(rows, tmp_path).read_text(encoding="utf-8")
    assert "Complete stacks: 1/2" in text
    assert "Finger-to-face angle at grip: mean 18.0°, median 10.0°, max 40.0°. Ideal is 0°." in text
    assert (
        "| 0 | 0 | pass | put green cube on red cube: pass, "
        "put blue cube on green cube: pass | - | pass, pass | 4.0, 10.0 | -12.5, 3.0 |"
    ) in text
    assert (
        "| 1 | 1 | fail | put green cube on red cube: fail | first stack failed "
        "| fail | 40.0 | 0.5 |"
    ) in text


def test_displaced_cube_without_grasp_is_not_reported_as_dropped():
    environment = DisplacedWithoutGraspEnvironment(succeed=False)
    session = StackCommandSession(
        environment=environment,
        policy=FakePolicy(),
        observation=_observation(),
        action=lambda _observation, _task: np.zeros(6),
        max_steps_per_command=1,
    )
    trace = session.execute("put green cube on red cube")
    assert not trace.selected
    assert trace.failure_stage == "cube displaced without grasp"


def test_random_stacking_output_names_and_rejections(tmp_path):
    baseline = random_stacking_output_directory(
        tmp_path, "sapien", "stack2-training", "stamp"
    )
    extension = random_stacking_output_directory(
        tmp_path, "sapien", "stack3-color", "stamp"
    )
    assert baseline.name == "random-stacking-sapien-stamp"
    assert extension.name == "random-stacking-color-sapien-stamp"
    with pytest.raises(ValueError, match="only the SAPIEN"):
        random_stacking_output_directory(
            tmp_path, "mujoco", "stack2-training", "stamp"
        )
    with pytest.raises(ValueError, match="unsupported stacking task"):
        random_stacking_output_directory(tmp_path, "sapien", "other", "stamp")
