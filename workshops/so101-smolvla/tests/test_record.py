import numpy as np

from so101_smolvla.record import load_episode


def test_load_episode_copies_each_array_once(tmp_path):
    state = np.zeros((4, 6), np.float32)
    action = np.ones((4, 6), np.float32)
    top = np.arange(4 * 8 * 8 * 3, dtype=np.uint8).reshape(4, 8, 8, 3)
    wrist = np.flip(top, axis=0).copy()
    dest = tmp_path / "ep.npz"
    np.savez_compressed(dest, state=state, action=action, top=top, wrist=wrist)

    loaded = load_episode(dest)
    dest.unlink()

    assert loaded["state"].shape == (4, 6)
    assert np.array_equal(loaded["action"], action)
    assert np.array_equal(loaded["top"], top)
    assert np.array_equal(loaded["wrist"], wrist)
    loaded["top"][0, 0, 0, 0] = np.uint8(0)
    assert loaded["top"][0, 0, 0, 0] == 0
