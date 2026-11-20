import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from so101_smolvla.grasp_geometry import WORST_FINGER_FACE_DEG
from so101_smolvla.reachable_workspace import is_reachable_xy
from so101_smolvla.sapien_config import (
    HAND_LINKS,
    IMAGE_SHAPE,
    MOTOR_BLACK,
    ROBOT_ORANGE,
    SAPIEN_PYTHON,
    SAPIEN_SOURCE,
)
from so101_smolvla.sapien_env_client import SapienPolicyEnvironment
from so101_smolvla.stack_recording_rig import RECORDING_RIG
from so101_smolvla.table_view import table_seen_from


pytestmark = pytest.mark.skipif(
    not SAPIEN_PYTHON.is_file() or not (SAPIEN_SOURCE / ".git").is_dir(),
    reason="run scripts/setup_sapien_env.sh to install the real SAPIEN backend",
)


def test_real_sapien_reset_step_and_clean_shutdown():
    env = SapienPolicyEnvironment(horizon=2)
    try:
        first = env.reset(environment_seed=0)
        outcome = env.step(first.state)
        assert first.state.shape == (6,)
        assert first.top.shape == IMAGE_SHAPE
        assert first.wrist.shape == IMAGE_SHAPE
        assert outcome.observation.top.shape == IMAGE_SHAPE
        assert outcome.observation.wrist.shape == IMAGE_SHAPE
        assert outcome.achieved.shape == (6,)
        assert np.isfinite(outcome.achieved).all()
        assert Path(SAPIEN_SOURCE).is_dir()
        assert env.provenance["backend"] == "sapien"
        assert env.provenance["robot_appearance"] == {
            "printed_parts_rgba": list(ROBOT_ORANGE),
            "motors_rgba": list(MOTOR_BLACK),
            "hand_links": sorted(HAND_LINKS),
            # The colour the pinned simulator gives the robot in training.
            "hand_rgba": [0.05, 0.05, 0.05, 1.0],
        }
    finally:
        env.close()
    env.close()


def _orange_pixels(picture: np.ndarray) -> int:
    rgb = picture.astype(np.int32)
    return int(
        ((rgb[..., 0] - rgb[..., 2] > 90) & (rgb[..., 1] - rgb[..., 2] > 40)).sum()
    )


# The humans' median command halfway between grasp and release in the real
# stacking recordings: the arm held out over the table, clear of the cubes.
CARRYING_POSE = np.array([-12.7, 21.8, -16.6, 81.8, -1.5, 2.7])


def _dark_columns(picture: np.ndarray) -> np.ndarray:
    """Picture columns of the dark hand along the lowest row of a wrist picture."""
    return np.flatnonzero(picture[-1].max(axis=1) < 80)


def _recorded_table_share(
    wrist: np.ndarray, facts: dict, table: np.ndarray
) -> tuple[float, float]:
    """How much of a wrist picture is the recorded table seen from the hand.

    Returns the share of the picture that looks at the table, and the share
    of those pixels outside the dark hand that are, within one level, the
    table picture's pixel of the same spot on the table.
    """
    painted, on_table = table_seen_from(
        table,
        np.asarray(facts["top_camera"]["projection"]),
        np.asarray(facts["wrist_camera"]["projection"]),
        -RECORDING_RIG.table_below_robot_base,
    )
    same = np.abs(wrist.astype(int) - painted).max(axis=2) <= 1
    hand = wrist.max(axis=2) < 80
    return float(on_table.mean()), float(same[on_table & ~hand].mean())


def test_real_sapien_shows_an_orange_arm_with_a_black_hand():
    env = SapienPolicyEnvironment(horizon=2, task_mode="stack2-sim-view")
    try:
        first = env.reset(environment_seed=3)
    finally:
        env.close()
    # The top camera sees the painted arm.
    assert _orange_pixels(first.top) > 5000
    # The wrist camera sees only the hand: no orange, and the two dark jaws
    # fill the lower part of its picture.
    assert _orange_pixels(first.wrist) == 0
    assert (first.wrist.max(axis=2) < 80).sum() > 20000


@pytest.mark.parametrize(
    ("task_mode", "cube_size", "sentence"),
    (
        ("pick-place", 0.040, "Pick up a cube and place in the bin"),
        ("pick-place-cube20", 0.020, "Pick up a small cube and place in the bin"),
    ),
)
def test_real_sapien_bin_tasks_place_the_right_cube_in_the_reachable_workspace(
    task_mode, cube_size, sentence
):
    env = SapienPolicyEnvironment(horizon=2, task_mode=task_mode)
    try:
        env.reset(environment_seed=3)
        cube = np.asarray(env.facts["item_position"])
        # A cube resting on the table has its centre half its size above it.
        assert cube[2] == pytest.approx(cube_size / 2, abs=1e-3)
        assert is_reachable_xy(cube[:2])
        assert env.task_text == sentence
        assert env.provenance["task_text"] == sentence
        assert env.provenance["task_mode"] == task_mode
        # The scene reports how the gripper stands to the cube. The
        # finger-to-face angle of an upright cube lies between square (0)
        # and edge-on (45).
        grip = env.facts["grip"]
        assert isinstance(grip["contact"], bool)
        assert 0.0 < grip["distance"] < 0.5
        assert 0.0 <= grip["finger_face_deg"] <= WORST_FINGER_FACE_DEG + 1e-6
    finally:
        env.close()


@pytest.mark.xfail(
    strict=True,
    reason=(
        "known defect, measured 20 Nov 2026 with "
        "vla_testing/diagnostics/cube_in_resting_hand.py: the bin tasks place "
        "the cube where the resting hand is on 12 of 200 seeds (40 mm cube) "
        "and 3 of 200 (20 mm cube); the stacking scene on 0 of 200"
    ),
)
def test_real_sapien_bin_task_does_not_place_the_cube_into_the_resting_hand():
    env = SapienPolicyEnvironment(horizon=2, task_mode="pick-place")
    try:
        # Seed 49: the cube's centre is placed 3.4 mm from the midpoint of
        # the resting fingertips.
        first = env.reset(environment_seed=49)
        placed = np.asarray(env.facts["item_position"])
        held = env.step(first.state)
        pushed = np.linalg.norm(np.asarray(held.facts["item_position"]) - placed)
        assert held.facts["grip"]["contact"] is False
        assert pushed < 0.0005
    finally:
        env.close()


@pytest.mark.parametrize(
    ("task_mode", "colors", "sizes"),
    (
        ("stack2-training", {"red", "blue"}, {"red": 0.020, "blue": 0.040}),
        ("stack2-recording-rig", {"red", "blue"}, {"red": 0.020, "blue": 0.040}),
        (
            "stack3-color",
            {"red", "green", "blue"},
            {"red": 0.04, "green": 0.04, "blue": 0.04},
        ),
    ),
)
def test_real_sapien_stacking_tasks_replay_seed_and_export_facts(
    task_mode, colors, sizes
):
    env = SapienPolicyEnvironment(horizon=3, task_mode=task_mode)
    try:
        first = env.reset(environment_seed=17)
        first_layout = env.facts["initial_layout"]
        first_state_index = env.facts["initial_state_source_index"]
        first_state_real = np.asarray(env.facts["initial_state_real"])
        assert np.allclose(first.state, first_state_real, atol=0.5)
        env.reset(environment_seed=17)
        assert env.facts["initial_layout"] == first_layout
        assert env.facts["initial_state_source_index"] == first_state_index
        assert env.facts["initial_state_real"] == first_state_real.tolist()
        assert set(env.facts["cube_positions"]) == colors
        assert env.facts["cube_sizes"] == sizes
        assert first.top.shape == IMAGE_SHAPE
        assert first.wrist.shape == IMAGE_SHAPE
        outcome = env.step(first.state)
        assert set(outcome.facts["grasped"]) == colors
        # Every cube reports how the gripper stands to it. The arm holds its
        # start pose, away from the cubes.
        assert set(outcome.facts["grip"]) == colors
        for color in colors:
            grip = outcome.facts["grip"][color]
            assert grip["contact"] is False
            assert grip["distance"] > sizes[color]
            assert 0.0 <= grip["finger_face_deg"] <= WORST_FINGER_FACE_DEG + 1e-6
        if task_mode == "stack3-color":
            facts = env.set_command("green", "red")
            assert facts["active_source"] == "green"
            assert facts["active_target"] == "red"
    finally:
        env.close()


def test_real_sapien_sim_view_changes_only_camera_and_start_pose():
    """Same seed: same cubes in the same place, simulator camera and rest pose."""
    observations = {}
    facts = {}
    for task_mode in ("stack2-training", "stack2-sim-view", "pick-place-cube20"):
        env = SapienPolicyEnvironment(horizon=2, task_mode=task_mode)
        try:
            observations[task_mode] = env.reset(environment_seed=17)
            facts[task_mode] = dict(env.facts)
            if task_mode == "stack2-sim-view":
                with pytest.raises(RuntimeError, match="use stack2-training"):
                    env.reset(
                        environment_seed=0,
                        stack_replay={
                            "state_index": 33,
                            "layout": {"red": [0.2, 0.0], "blue": [0.3, 0.0]},
                        },
                    )
        finally:
            env.close()

    training, sim_view = facts["stack2-training"], facts["stack2-sim-view"]
    assert sim_view["initial_layout"] == training["initial_layout"]
    assert sim_view["cube_sizes"] == training["cube_sizes"]
    for color in ("red", "blue"):
        assert np.allclose(
            sim_view["cube_positions"][color],
            training["cube_positions"][color],
            atol=1e-4,
        )
    assert sim_view["initial_state_source_index"] is None
    assert training["initial_state_source_index"] is not None
    # The simulator's bin task starts at the same rest pose plus 0.02 rad
    # (1.15 degree) seeded noise per joint; five standard deviations bound it.
    assert np.allclose(
        observations["stack2-sim-view"].state,
        sim_view["initial_state_real"],
        atol=0.5,
    )
    assert np.allclose(
        observations["stack2-sim-view"].state[:5],
        observations["pick-place-cube20"].state[:5],
        atol=5 * np.rad2deg(0.02),
    )
    assert not np.allclose(
        observations["stack2-training"].state[:5],
        observations["stack2-sim-view"].state[:5],
        atol=0.5,
    )
    assert not np.array_equal(
        observations["stack2-training"].top, observations["stack2-sim-view"].top
    )


def test_real_sapien_recording_rig_shows_and_drives_the_measured_rig():
    """Same seed as stack2-training: same cubes and start pose, measured rig."""
    rig = RECORDING_RIG
    table = cv2.cvtColor(cv2.imread(str(rig.table_picture)), cv2.COLOR_BGR2RGB)
    training = SapienPolicyEnvironment(horizon=60, task_mode="stack2-training")
    try:
        training_first = training.reset(environment_seed=17)
        training_facts = dict(training.facts)
        # The simulator's own arm has no weight: held out over the table it
        # comes to rest exactly on its command.
        for _ in range(60):
            weightless = training.step(CARRYING_POSE)
        assert np.allclose(weightless.achieved[:5], CARRYING_POSE[:5], atol=0.1)
    finally:
        training.close()

    env = SapienPolicyEnvironment(horizon=100, task_mode="stack2-recording-rig")
    try:
        first = env.reset(environment_seed=17)
        facts = dict(env.facts)
        assert facts["initial_layout"] == training_facts["initial_layout"]
        assert facts["initial_state_source_index"] == (
            training_facts["initial_state_source_index"]
        )
        assert facts["table_supported"] == {"red": True, "blue": True}
        for color, size in facts["cube_sizes"].items():
            # A cube rests on the table, which lies below the robot base.
            assert facts["cube_positions"][color][2] == pytest.approx(
                size / 2 - rig.table_below_robot_base, abs=1e-3
            )

        # The top camera stands where it was measured and has the measured lens.
        assert np.allclose(
            facts["top_camera"]["position"], rig.top_camera_position, atol=1e-4
        )
        assert facts["top_camera"]["focal_length_px"] == pytest.approx(
            IMAGE_SHAPE[0] / 2 / np.tan(rig.top_camera_vertical_fov_rad / 2),
            rel=1e-4,
        )

        # The wrist camera sits on the hand where it was measured, with the
        # measured lens. It sat closer to the jaws than the simulator's own
        # wrist camera: along the lowest picture row the dark hand lies
        # farther to the right and less of it is in view.
        assert np.allclose(
            facts["wrist_camera"]["position_on_optical_frame"],
            rig.wrist_camera_position,
            atol=1e-5,
        )
        assert np.allclose(
            facts["wrist_camera"]["orientation_on_optical_frame_wxyz"],
            rig.wrist_camera_orientation_wxyz,
            atol=1e-5,
        )
        assert facts["wrist_camera"]["focal_length_px"] == pytest.approx(
            IMAGE_SHAPE[0] / 2 / np.tan(rig.wrist_camera_vertical_fov_rad / 2),
            rel=1e-4,
        )
        jaws = _dark_columns(first.wrist)
        simulator_jaws = _dark_columns(training_first.wrist)
        assert jaws.mean() > simulator_jaws.mean() + 30
        assert len(jaws) < len(simulator_jaws) - 50

        # The top picture is the recorded table wherever no robot or cube is
        # drawn.
        table_pixels = np.all(first.top == table, axis=2)
        assert 0.5 < table_pixels.mean() < 1.0
        assert np.array_equal(first.top[0], table[0])
        assert not np.array_equal(first.top, training_first.top)

        # The wrist camera shows that same table from where the hand is, not
        # the top camera's picture as it is and not the simulator's flat
        # table: outside the hand, all but the two cubes is the table
        # picture's pixel of the same spot on the table (measured: 0.944).
        looks_at_table, recorded_table = _recorded_table_share(
            first.wrist, facts, table
        )
        assert looks_at_table > 0.9
        assert recorded_table > 0.9
        assert np.all(first.wrist == table, axis=2).mean() < recorded_table / 2
        assert not np.array_equal(first.wrist, training_first.wrist)

        # The model reads the recording robot's joint numbers; the simulated
        # arm stands at those numbers plus the recording robot's zero points.
        zero_points = np.rad2deg(rig.arm_zero_points_rad)
        assert np.abs(zero_points).max() > 5.0
        assert np.allclose(first.state, facts["initial_state_real"], atol=0.5)
        assert np.allclose(first.state, training_first.state, atol=0.5)
        assert np.allclose(
            facts["simulated_arm_deg"], first.state[:5] + zero_points, atol=0.01
        )

        # The arm has its weight, and the measured joint drives are still in
        # force after the reset.
        drives = facts["arm_drives"]
        assert drives["has_weight"]
        assert drives["stiffness"] == pytest.approx(rig.arm_drive_stiffness, rel=1e-5)
        assert drives["damping"] == pytest.approx(rig.arm_drive_damping, rel=1e-5)

        # Held out over the table, the weighted arm comes to rest below its
        # command at the two joints that carry it, as the recording robot's
        # did: lift and elbow read more than commanded (measured in this
        # scene: 1.70 and 2.04 degrees; the recorded elbow read 0.6 to 3.5
        # degrees more at the grasp). Pan, wrist flex and wrist roll carry
        # next to no weight and rest on their command.
        for _ in range(60):
            held = env.step(CARRYING_POSE)
        behind = held.achieved - CARRYING_POSE
        assert 0.8 < behind[1] < 2.6
        assert 1.2 < behind[2] < 3.0
        assert np.allclose(behind[[0, 3, 4]], 0.0, atol=0.2)
        assert np.allclose(
            held.facts["simulated_arm_deg"], held.achieved[:5] + zero_points, atol=0.01
        )

        # The hand has moved, and the recorded table in its picture with it
        # (measured: 0.980 of the picture outside the hand).
        assert not np.allclose(
            held.facts["wrist_camera"]["projection"],
            facts["wrist_camera"]["projection"],
            atol=1.0,
        )
        assert _recorded_table_share(held.observation.wrist, held.facts, table)[1] > 0.9

        # A commanded joint number moves the simulated joint by the same
        # amount. Both readings are taken after the arm has come to rest.
        target = CARRYING_POSE.copy()
        target[0] += 3.0
        for _ in range(40):
            moved = env.step(target)
        assert moved.achieved[0] - held.achieved[0] == pytest.approx(3.0, abs=0.1)
        assert np.allclose(
            moved.facts["simulated_arm_deg"],
            moved.achieved[:5] + zero_points,
            atol=0.01,
        )
    finally:
        env.close()


def _play_stored_test(env, test: dict) -> dict[str, np.ndarray]:
    """Start a stored test and play its commands; what the scene showed."""
    first = env.reset(environment_seed=test["environment_seed"])
    env.set_command("red", "blue")
    arm, red, blue = [], [], []
    for command in test["commands"]:
        outcome = env.step(np.asarray(command))
        arm.append(outcome.achieved)
        red.append(outcome.facts["cube_positions"]["red"])
        blue.append(outcome.facts["cube_positions"]["blue"])
    return {
        "first top picture": first.top,
        "first wrist picture": first.wrist,
        "last top picture": outcome.observation.top,
        "last wrist picture": outcome.observation.wrist,
        "arm": np.asarray(arm),
        "red cube": np.asarray(red),
        "blue cube": np.asarray(blue),
    }


def test_real_sapien_test_runs_the_same_whatever_ran_before_it():
    """A test starts in an environment that has run nothing before.

    The simulator's own reset moves robot and cubes back but keeps its physics
    scene. Measured with these commands on 20 Nov 2026
    (vla_testing/diagnostics/episode_history_check.py): after the first 100
    steps of another test, in which no cube is touched, the same seed with the
    same commands left its course at step 179, when the hand has the red cube.
    """
    stored = json.loads(
        (Path(__file__).parent / "data" / "stack_test_commands.json").read_text(
            encoding="utf-8"
        )
    )
    env = SapienPolicyEnvironment(horizon=200, task_mode=stored["task_mode"])
    try:
        alone = _play_stored_test(env, stored["later_test"])
        _play_stored_test(env, stored["earlier_test"])
        after_another = _play_stored_test(env, stored["later_test"])
    finally:
        env.close()

    # The commands take the red cube off the table, so the hand's contact
    # with it is part of what has to repeat.
    assert alone["red cube"][-1][2] - alone["red cube"][0][2] > 0.01
    # The step from which arm or cube differ, the picture row for a picture.
    first_difference = {}
    for name, course in alone.items():
        rows = np.flatnonzero(
            np.any(
                (course != after_another[name]).reshape(len(course), -1), axis=1
            )
        )
        if len(rows):
            first_difference[name] = int(rows[0])
    assert first_difference == {}


def test_real_sapien_stacking_accepts_an_exact_training_episode_replay():
    replay = {
        "state_index": 33,
        "layout": {
            "red": [0.21880966, -0.07736267],
            "blue": [0.31014021, 0.02937031],
        },
    }
    env = SapienPolicyEnvironment(horizon=2, task_mode="stack2-training")
    try:
        observation = env.reset(environment_seed=0, stack_replay=replay)
        assert env.facts["initial_state_source_index"] == 33
        assert env.facts["initial_layout"] == replay["layout"]
        assert np.allclose(
            observation.state,
            env.facts["initial_state_real"],
            atol=0.5,
        )
    finally:
        env.close()
