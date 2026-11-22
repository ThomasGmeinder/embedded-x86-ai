from __future__ import annotations

import numpy as np
import pytest

from clip_reading import read_clip
from so101_smolvla.record import _write_clip, accumulate


def _flat(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def test_a_clip_has_the_size_and_the_number_of_its_frames(tmp_path):
    # Two camera pictures side by side, the size of rollout.mp4.
    frames = np.stack([_flat(480, 1280, value) for value in (40, 120, 200)])
    _write_clip(frames, tmp_path / "wide" / "rollout.mp4")
    written = read_clip(tmp_path / "wide" / "rollout.mp4")
    assert written.shape == (3, 480, 1280, 3)
    # Each frame comes back as the grey it was written with.
    for frame, value in zip(written, (40, 120, 200), strict=True):
        print(f"grey {value} read back as {frame.min()} to {frame.max()}")
        assert np.abs(frame.astype(int) - value).max() <= 3

    _write_clip(np.stack([_flat(480, 640, 90)] * 2), tmp_path / "camera.mp4")
    assert read_clip(tmp_path / "camera.mp4").shape == (2, 480, 640, 3)


def test_a_clip_is_written_from_frames_handed_out_one_at_a_time(tmp_path):
    handed_out = []

    def frames():
        for value in (10, 20, 30, 40):
            handed_out.append(value)
            yield _flat(1008, 1920, value)

    _write_clip(frames(), tmp_path / "feeds.mp4")
    assert handed_out == [10, 20, 30, 40]
    assert read_clip(tmp_path / "feeds.mp4").shape == (4, 1008, 1920, 3)


def test_a_clip_refuses_frames_it_cannot_encode(tmp_path):
    path = tmp_path / "clip.mp4"
    with pytest.raises(ValueError, match="no frame to write"):
        _write_clip([], path)
    with pytest.raises(ValueError, match="invalid frame float64"):
        _write_clip([np.zeros((480, 640, 3))], path)
    with pytest.raises(ValueError, match="invalid frame uint8 of shape"):
        _write_clip([np.zeros((480, 640), dtype=np.uint8)], path)
    with pytest.raises(ValueError, match="641x480 has an odd side"):
        _write_clip([_flat(480, 641, 0)], path)
    # Nothing was started for a clip that was refused.
    assert not path.exists()
    with pytest.raises(ValueError, match="the first is of shape"):
        _write_clip([_flat(480, 640, 0), _flat(240, 320, 0)], path)


def test_accumulate_joins_the_clips_it_is_asked_for(tmp_path):
    folders = []
    for name, value in (("a", 50), ("b", 150)):
        folder = tmp_path / name
        _write_clip([_flat(480, 640, value)] * 2, folder / "top.mp4")
        _write_clip([_flat(1008, 1920, value)] * 3, folder / "feeds.mp4")
        folders.append(folder)
    accumulate(folders, tmp_path, "overall", clips=("top", "feeds"))
    top = read_clip(tmp_path / "overall-top.mp4")
    feeds = read_clip(tmp_path / "overall-feeds.mp4")
    assert top.shape == (4, 480, 640, 3)
    assert feeds.shape == (6, 1008, 1920, 3)
    # The folders follow each other in the order given.
    assert abs(int(np.median(feeds[0])) - 50) <= 3
    assert abs(int(np.median(feeds[5])) - 150) <= 3
    # No clip was asked for that the folders do not hold.
    assert not (tmp_path / "overall-wrist.mp4").exists()
