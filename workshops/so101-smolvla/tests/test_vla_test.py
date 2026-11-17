import json

import numpy as np

from so101_smolvla.sim import ArmSim
from so101_smolvla.track import POWER_ON
from so101_smolvla.vla_test import VLA_TESTING, draw_cube, prepare, run_scenario, scenarios


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
    saved = json.loads((root / "scene" / "result.json").read_text(encoding="utf-8"))
    assert saved["scenario"] == "scene"
    assert saved["cube_start_xy"] == row["cube_start_xy"]
    assert "scripted_testing" not in root.parts
    assert VLA_TESTING.name == "vla_testing"
    assert "scripted_testing" not in VLA_TESTING.parts
    assert calls


def test_a_seed_draws_the_same_cube_twice_and_is_written(tmp_path):
    first = draw_cube(17)
    assert draw_cube(17) == first
    assert draw_cube(18) != first
    xy, yaw = first
    assert abs(xy[0] - 0.27) <= 0.10
    assert abs(xy[1] - 0.045) <= 0.10

    def predict(state, top, wrist):
        return np.repeat(POWER_ON.reshape(1, 6), 2, axis=0)

    sim = ArmSim()
    try:
        run_scenario(sim, predict, "seed-00017", xy, tmp_path, steps=2, yaw_rad=yaw, seed=17)
    finally:
        sim.close()
    saved = json.loads((tmp_path / "seed-00017" / "result.json").read_text(encoding="utf-8"))
    assert saved["seed"] == 17
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
