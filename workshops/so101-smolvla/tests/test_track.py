import numpy as np

from so101_smolvla.sim import ArmSim
from so101_smolvla.track import (
    POWER_ON,
    hold_power_on,
    play,
    rollout,
    trajectory_metrics,
)


def test_power_on_pose_is_radians_and_leaves_the_cube():
    sim = ArmSim()
    try:
        sim.reset()
        cube = sim.cube_pos().copy()
        pose = hold_power_on(sim)
        shoulder = sim.data.qpos[sim.joint_qadr["shoulder_lift"]]
        assert abs(shoulder - np.deg2rad(-100.0)) < 1e-5
        assert abs(shoulder) < 3.0
        assert np.allclose(pose, POWER_ON, atol=1e-4)
        assert np.allclose(sim.commanded(), POWER_ON, atol=1e-4)
        assert np.allclose(sim.cube_pos(), cube)
    finally:
        sim.close()


def test_target_past_the_hinge_is_a_clip():
    sim = ArmSim()
    try:
        hold_power_on(sim)
        target = POWER_ON.copy()
        target[1] = -115.0
        played = play(sim, target.reshape(1, 6))
        limit = np.degrees(sim.model.actuator_ctrlrange[sim.act_id["shoulder_lift"]][0])
        assert played["target"][0, 1] == -115.0
        assert abs(played["commanded"][0, 1] - limit) < 1e-4
        assert abs(played["commanded"][0, 1] - played["target"][0, 1]) > 2.0
    finally:
        sim.close()


def test_rollout_takes_a_new_picture_when_the_chunk_is_used():
    sim = ArmSim()
    calls = []

    def predict(state, top, wrist):
        calls.append(np.array(state))
        assert top.shape == (480, 640, 3)
        assert wrist.shape == (480, 640, 3)
        return np.repeat(POWER_ON.reshape(1, 6), 4, axis=0)

    try:
        hold_power_on(sim)
        out = rollout(sim, predict, steps=5, chunk_size=2)
    finally:
        sim.close()
    assert out["calls"] == 3
    assert len(calls) == 3
    assert out["policy_top"].shape == (3, 480, 640, 3)
    assert out["policy_wrist"].shape == (3, 480, 640, 3)
    assert out["top"].shape[0] == 5
    assert out["wrist"].shape == (5, 480, 640, 3)
    assert out["target"].shape == (5, 6)
    assert out["commanded"].shape == (5, 6)
    assert out["achieved"].shape == (5, 6)
    assert out["chunk_index"].tolist() == [0, 0, 1, 1, 2]


def test_trajectory_metrics_count_only_adjacent_moving_reversals():
    targets = np.zeros((7, 6), dtype=np.float64)
    targets[:, 1] = [0.0, 1.0, 2.0, 1.0, 0.95, 2.0, 3.0]
    measured = trajectory_metrics(targets, movement_epsilon=0.1)["shoulder_lift"]
    assert measured["moving_steps"] == 5
    assert measured["reversal_opportunities"] == 3
    assert measured["reversals"] == 1
    assert measured["reversal_rate"] == 1 / 3
    assert measured["median_step"] == 1.0
