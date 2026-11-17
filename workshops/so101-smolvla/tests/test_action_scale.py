from so101_smolvla.action_scale import dataset_stats


def test_checkpoint_scale_is_used_for_state_and_action():
    mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    stats = dataset_stats(
        {
            "so100.buffer.action.mean": mean,
            "so100.buffer.action.std": std,
            "so100-red.buffer.action.mean": [9.0] * 6,
        }
    )
    assert stats["action"]["mean"] is mean
    assert stats["observation.state"]["std"] is std
    assert set(stats) == {"action", "observation.state"}
