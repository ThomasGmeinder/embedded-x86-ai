import json
from pathlib import Path

import numpy as np

from so101_smolvla.layouts import (
    _clear_of_bin,
    _corners,
    _in_fingers,
    _wrist_clips_cube,
    draw_layout,
)
from so101_smolvla.policy_env import random_output_directory
from so101_smolvla.sim import ArmSim
from so101_smolvla.track import POWER_ON
from so101_smolvla.record import _write_clip, accumulate
from so101_smolvla.vla_test import (
    VLA_TESTING,
    GripWatch,
    cli,
    prepare,
    run_scenario,
    scenarios,
    write_report,
)


class _Pose:
    """One simulator reading: the cube is either up, gripped, both, or neither."""

    def __init__(self, up: bool, gripping: bool) -> None:
        self.tip_a = 0
        self.tip_b = 1
        self.data = type("Data", (), {"geom_xpos": np.zeros((2, 3))})()
        self._up = up
        self._gripping = gripping

    def cube_pos(self) -> np.ndarray:
        return np.zeros(3)

    def finger_face_deg(self) -> float:
        return 0.0

    def read_state(self) -> np.ndarray:
        return np.zeros(6)

    def jaw_contact(self) -> bool:
        return self._gripping

    def above_table(self) -> bool:
        return self._up

    def held(self, empty_force: float) -> bool:
        return self._gripping


def test_a_squeeze_without_a_lift_is_not_held():
    watch = GripWatch()
    watch.observe(_Pose(up=False, gripping=True))
    assert watch.picked_up is False
    assert watch.held is False
    watch.observe(_Pose(up=True, gripping=True))
    assert watch.picked_up is True
    assert watch.held is True


def test_results_go_to_vla_testing_not_scripted(tmp_path):
    root = tmp_path / "vla_testing"
    calls = []

    def predict(state, top, wrist):
        calls.append(1)
        assert top.shape == (480, 640, 3)
        return np.repeat(POWER_ON.reshape(1, 6), 2, axis=0)

    sim = ArmSim()
    try:
        row = run_scenario(sim, predict, "scene", None, root, steps=2)
    finally:
        sim.close()
    assert (root / "scene" / "top.mp4").is_file()
    assert (root / "scene" / "wrist.mp4").is_file()
    assert (root / "scene" / "policy-observations.npz").is_file()
    saved = json.loads((root / "scene" / "result.json").read_text(encoding="utf-8"))
    assert saved["scenario"] == "scene"
    assert saved["cube_start_xy"] == row["cube_start_xy"]
    assert saved["backend"] == "mujoco"
    assert "scripted_testing" not in root.parts
    assert VLA_TESTING.name == "vla_testing"
    assert "scripted_testing" not in VLA_TESTING.parts
    assert calls


def test_a_seed_places_the_cube_outside_the_tray_and_is_written(tmp_path):
    sim = ArmSim()
    try:
        first = draw_layout(sim, 17)
        assert draw_layout(sim, 17) == first
        assert draw_layout(sim, 18) != first
        bins = set()
        for seed in range(40):
            layout = draw_layout(sim, seed)
            bins.add(tuple(round(value, 4) for value in layout["bin_xy"]))
            assert _clear_of_bin(
                np.array(layout["cube_xy"]),
                np.array(layout["bin_xy"]),
                layout["bin_yaw"],
                sim.bin_half,
                0.02,
            )
            assert _in_fingers(np.array(layout["bin_xy"]))
            assert float(np.hypot(*layout["bin_xy"])) <= 0.38
            for corner in _corners(np.array(layout["cube_xy"]), layout["cube_yaw"], np.array([0.02, 0.02])):
                assert _in_fingers(corner)
            assert not _wrist_clips_cube(sim, np.array(layout["cube_xy"]), layout["cube_yaw"], 0.02)
        assert len(bins) > 30
        assert not _in_fingers(np.array([0.4573, -0.1392]))
        sim.hold_pose(POWER_ON)
        assert _wrist_clips_cube(sim, np.array([0.2909, 0.1118]), np.deg2rad(15.7), 0.02)
        xy, yaw = first["cube_xy"], first["cube_yaw"]

        def predict(state, top, wrist):
            return np.repeat(POWER_ON.reshape(1, 6), 2, axis=0)

        run_scenario(
            sim,
            predict,
            "seed-00017",
            xy,
            tmp_path,
            steps=2,
            yaw_rad=yaw,
            layout_seed=17,
            policy_seed=10017,
        )
    finally:
        sim.close()
    saved = json.loads((tmp_path / "seed-00017" / "result.json").read_text(encoding="utf-8"))
    assert saved["layout_seed"] == 17
    assert saved["policy_seed"] == 10017
    assert "finger_face_deg" in saved
    assert "trajectory" in saved
    assert saved["grip"] in (True, False)
    assert saved["cube_start_xy"] == [round(xy[0], 4), round(xy[1], 4)]
    assert saved["cube_yaw_deg"] == round(float(np.degrees(yaw)), 2)


def test_scenarios_are_the_yaml_list_plus_the_scene():
    names = [name for name, _xy in scenarios()]
    assert names[0] == "scene"
    assert "cube-0.14-0.18" in names
    assert len(names) == 7


def test_prepare_yaws_the_cube_about_vertical():
    sim = ArmSim()
    try:
        prepare(sim, (0.18, -0.06), yaw_rad=np.pi / 2)
        quat = sim.data.qpos[sim.cube_qadr + 3 : sim.cube_qadr + 7]
        xy = sim.cube_pos()[:2]
    finally:
        sim.close()
    assert abs(xy[0] - 0.18) < 1e-4
    assert abs(xy[1] + 0.06) < 1e-4
    assert abs(quat[0] - np.cos(np.pi / 4)) < 1e-6
    assert abs(quat[3] - np.sin(np.pi / 4)) < 1e-6
    assert abs(quat[1]) < 1e-6 and abs(quat[2]) < 1e-6


def test_finger_face_angle_follows_the_cube_yaw():
    sim = ArmSim()
    try:
        prepare(sim, (0.18, -0.06), yaw_rad=0.0)
        square = sim.finger_face_deg()
        prepare(sim, (0.18, -0.06), yaw_rad=np.pi / 4)
        turned = sim.finger_face_deg()
    finally:
        sim.close()
    assert 0.0 <= square <= 90.0
    assert 0.0 <= turned <= 90.0
    assert abs(square - turned) > 1.0


def test_the_report_links_the_video_and_scores_each_seed(tmp_path):
    rows = [
        {
            "layout_seed": 0,
            "policy_seed": 10000,
            "in_box": True,
            "picked_up": True,
            "held": False,
            "grip": True,
            "finger_face_deg": 0.0,
            "trajectory": {
                "shoulder_lift": {"reversal_rate": 0.05},
                "elbow_flex": {"reversal_rate": 0.1},
            },
        },
        {
            "layout_seed": 1,
            "policy_seed": 10001,
            "in_box": False,
            "picked_up": False,
            "held": False,
            "grip": False,
            "finger_face_deg": 35.0,
            "trajectory": {
                "shoulder_lift": {"reversal_rate": 0.4},
                "elbow_flex": {"reversal_rate": 0.5},
            },
        },
    ]
    path = write_report(rows, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "overall-top.mp4" in text
    assert "overall-wrist.mp4" in text
    assert "| Cube in the bin at the end | 1 | 1 |" in text
    assert "| Cube picked up | 1 | 1 |" in text
    assert "| Cube held | 0 | 2 |" in text
    assert "Ideal is 0°" in text
    assert "| 1 | 10001 | fail | fail | fail | fail | 35.0 | 40.0% | 50.0% |" in text


def test_random_directory_carries_the_timestamp():
    root = Path("/tmp/vla")
    assert random_output_directory(root, "mujoco", "20261118-090200") == (
        root / "random-MuJoCo-20261118-090200"
    )
    assert random_output_directory(root, "sapien", "20261118-090200") == (
        root / "random-sapien-20261118-090200"
    )


def test_prepare_moves_only_the_requested_cube():
    sim = ArmSim()
    try:
        default = prepare(sim, None)
        moved = prepare(sim, (0.14, 0.18))
    finally:
        sim.close()
    assert abs(moved[0] - 0.14) < 1e-4
    assert abs(moved[1] - 0.18) < 1e-4
    assert default != moved


def test_accumulate_joins_every_clip(tmp_path):
    frame = np.zeros((2, 480, 640, 3), dtype=np.uint8)
    folders = []
    for name in ("a", "b"):
        folder = tmp_path / name
        _write_clip(frame, folder / "top.mp4")
        _write_clip(frame, folder / "wrist.mp4")
        folders.append(folder)
    accumulate(folders, tmp_path, "overall")
    assert (tmp_path / "overall-top.mp4").is_file()
    assert (tmp_path / "overall-wrist.mp4").is_file()

def test_cli_routes_sapien_without_constructing_mujoco(monkeypatch):
    calls = []

    def fake_run_batch(count, **kwargs):
        calls.append((count, kwargs))
        return [{"success": True}, {"success": False}]

    monkeypatch.setattr("so101_smolvla.sapien_rollout.run_batch", fake_run_batch)

    cli(
        [
            "random",
            "2",
            "fixed",
            "--backend",
            "sapien",
            "--steps",
            "3",
            "--policy-seed-base",
            "70",
        ]
    )

    assert calls[0][0] == 2
    assert calls[0][1]["steps"] == 3
    assert calls[0][1]["policy_seed_base"] == 70
    assert calls[0][1]["destination"].name == "random-sapien-fixed"


def test_cli_forwards_steps_to_mujoco_random(monkeypatch):
    calls = []

    class FakeSim:
        def close(self):
            pass

    def fake_run_random(sim, predict, **kwargs):
        calls.append(kwargs)
        return [{"in_box": True}]

    monkeypatch.setattr("so101_smolvla.vla_test.ArmSim", FakeSim)
    monkeypatch.setattr(
        "so101_smolvla.vla_test.load_checkpoint",
        lambda: (object(), object(), object()),
    )
    monkeypatch.setattr("so101_smolvla.vla_test.run_random", fake_run_random)

    cli(["random", "10", "fixed", "--backend", "mujoco", "--steps", "300"])

    assert calls == [
        {
            "count": 10,
            "stamp": "fixed",
            "policy_seed_base": 10_000,
            "steps": 300,
        }
    ]
