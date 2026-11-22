import io
import json
from pathlib import Path

import numpy as np
import pytest

from so101_smolvla.policy_env import create_policy_environment, random_output_directory
from so101_smolvla.reachable_workspace import REACHABLE_CENTER, REACHABLE_HALF_EXTENT
from so101_smolvla.sapien_config import IMAGE_SHAPE, environment_arguments
from so101_smolvla.sapien_protocol import (
    OUTLINE_SHAPE,
    RESPONSE_HEADER,
    decode_payload,
    encode_payload,
    read_exact,
    read_response,
)


class _Fragmented:
    def __init__(self, payload: bytes, chunk_size: int) -> None:
        self._stream = io.BytesIO(payload)
        self._chunk_size = chunk_size

    def read(self, size: int) -> bytes:
        return self._stream.read(min(size, self._chunk_size))


def test_protocol_round_trips_metadata_and_two_images():
    top = np.full(IMAGE_SHAPE, 17, dtype=np.uint8)
    wrist = np.full(IMAGE_SHAPE, 29, dtype=np.uint8)
    payload = encode_payload({"state": [1, 2, 3, 4, 5, 6]}, top, wrist)

    response = decode_payload(7, payload)

    assert response.request_id == 7
    assert response.metadata["state"] == [1, 2, 3, 4, 5, 6]
    assert np.array_equal(response.top, top)
    assert np.array_equal(response.wrist, wrist)


def test_protocol_round_trips_outline_maps_behind_the_two_images():
    top = np.full(IMAGE_SHAPE, 17, dtype=np.uint8)
    wrist = np.full(IMAGE_SHAPE, 29, dtype=np.uint8)
    top_outline = np.zeros(OUTLINE_SHAPE, dtype=np.uint8)
    top_outline[10:20, 30:40] = 3
    wrist_outline = np.full(OUTLINE_SHAPE, 5, dtype=np.uint8)
    payload = encode_payload({"state": [1]}, top, wrist, top_outline, wrist_outline)

    response = decode_payload(9, payload)

    assert OUTLINE_SHAPE == IMAGE_SHAPE[:2]
    assert np.array_equal(response.top, top)
    assert np.array_equal(response.wrist, wrist)
    assert response.top_outline.dtype == np.uint8
    assert np.array_equal(response.top_outline, top_outline)
    assert np.array_equal(response.wrist_outline, wrist_outline)
    # A response without them carries none.
    plain = decode_payload(9, encode_payload({"state": [1]}, top, wrist))
    assert plain.top_outline is None and plain.wrist_outline is None


def test_protocol_rejects_outline_maps_that_do_not_belong_to_two_images():
    top = np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    outline = np.zeros(OUTLINE_SHAPE, dtype=np.uint8)
    with pytest.raises(ValueError, match="outline maps must be present together"):
        encode_payload({}, top, top, outline, None)
    with pytest.raises(ValueError, match="outline maps need the pictures"):
        encode_payload({}, None, None, outline, outline)
    with pytest.raises(ValueError, match="top outline map is"):
        encode_payload({}, top, top, outline[:-1], outline)
    with pytest.raises(ValueError, match="wrist outline map is"):
        encode_payload({}, top, top, outline, outline.astype(np.int16))
    # A payload cut off inside the outline maps is not read as pictures.
    payload = encode_payload({}, top, top, outline, outline)
    with pytest.raises(RuntimeError, match="image bytes, expected"):
        decode_payload(1, payload[:-1])


def test_read_exact_accepts_fragmented_pipe_reads():
    assert read_exact(_Fragmented(b"abcdefgh", 2), 8) == b"abcdefgh"


def test_read_exact_fails_when_worker_closes_early():
    with pytest.raises(RuntimeError, match="closed after 3 of 8"):
        read_exact(io.BytesIO(b"abc"), 8)


def test_response_id_must_match_request():
    payload = encode_payload({"ok": True})
    framed = RESPONSE_HEADER.pack(1, 8, len(payload)) + payload
    with pytest.raises(RuntimeError, match="does not match request 7"):
        read_response(io.BytesIO(framed), 7)


def test_worker_error_is_not_a_success_response():
    error = json.dumps({"error": "bad action"}).encode()
    framed = RESPONSE_HEADER.pack(0, 3, len(error)) + error
    with pytest.raises(RuntimeError, match="bad action"):
        read_response(io.BytesIO(framed), 3)


def test_backend_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown policy backend"):
        create_policy_environment("other")
    with pytest.raises(ValueError, match="unknown policy backend"):
        random_output_directory(Path("/tmp"), "other", "stamp")


def test_cube20_bin_task_differs_from_cube40_only_in_object_and_sentence():
    cube40 = environment_arguments("pick-place")
    cube20 = environment_arguments("pick-place-cube20")
    assert cube20["task"] == "SO101PickPlaceCube20-v1"
    assert cube20["task_description"] == "Pick up a small cube and place in the bin"
    assert cube20["spawn_box_pos"] == REACHABLE_CENTER.tolist()
    assert cube20["spawn_box_half_size"] == float(REACHABLE_HALF_EXTENT[0])
    differing = {key for key in cube40 if cube40[key] != cube20[key]}
    assert cube40.keys() == cube20.keys()
    assert differing == {"task", "task_description"}


def test_rollout_cli_routes_both_bin_tasks_to_their_own_directories(monkeypatch):
    from so101_smolvla import sapien_rollout

    calls = []

    def fake_run_batch(count, **kwargs):
        calls.append((count, kwargs))
        return [{"success": True}]

    monkeypatch.setattr(sapien_rollout, "run_batch", fake_run_batch)
    sapien_rollout.main(["--task", "pick-place-cube20", "--count", "2", "--steps", "3"])
    sapien_rollout.main(["--count", "1"])

    assert calls[0][0] == 2
    assert calls[0][1]["task_mode"] == "pick-place-cube20"
    assert calls[0][1]["steps"] == 3
    assert calls[0][1]["destination"].name.startswith("random-sapien-cube20-")
    assert calls[1][1]["task_mode"] == "pick-place"
    assert calls[1][1]["steps"] == sapien_rollout.DEFAULT_STEPS == 300
    assert calls[1][1]["destination"].name.startswith("random-sapien-2")


def test_rollout_cli_gives_a_stacking_command_its_own_step_limit(monkeypatch):
    from so101_smolvla import sapien_rollout

    calls = []

    def fake_run_stack_batch(count, **kwargs):
        calls.append((count, kwargs))
        return [{"success": True}]

    monkeypatch.setattr(sapien_rollout, "run_stack_batch", fake_run_stack_batch)
    sapien_rollout.main(["--task", "stack2-recording-rig", "--count", "1"])
    sapien_rollout.main(["--task", "stack2-recording-rig", "--count", "1", "--steps", "7"])

    # The latest stack in the random tests of seeds 0 to 19 is at step 604.
    assert sapien_rollout.DEFAULT_STACK_STEPS == 650
    assert calls[0][1]["task_mode"] == "stack2-recording-rig"
    assert calls[0][1]["steps_per_command"] == sapien_rollout.DEFAULT_STACK_STEPS
    assert calls[1][1]["steps_per_command"] == 7


def test_rollout_cli_routes_the_three_cube_task_with_pictures_and_order(monkeypatch):
    from so101_smolvla import sapien_rollout, three_cube_rollout

    calls = []

    def fake_run_three_cube_batch(count, **kwargs):
        calls.append((count, kwargs))
        return [{"success": True}]

    monkeypatch.setattr(
        three_cube_rollout, "run_three_cube_batch", fake_run_three_cube_batch
    )
    task = ["--task", "pick-place-three-cubes"]
    sapien_rollout.main(
        [*task, "--pictures", "others-painted", "--count", "2", "--order", "green,red,blue"]
    )
    sapien_rollout.main(
        [*task, "--pictures", "unedited", "--steps", "7", "--environment-seed-base", "5"]
    )
    sapien_rollout.main([*task, "--pictures", "others-painted", "--outlines", "sam3"])

    # Without the option the outlines are the simulator's, and the folder
    # keeps the name it had before there was a choice.
    assert calls[0][1]["outlines"] == "simulator"
    assert calls[1][1]["outlines"] == "simulator"
    assert calls[2][1]["outlines"] == "sam3"
    assert calls[2][1]["pictures"] == "others-painted"
    assert calls[2][1]["destination"].name.startswith(
        "random-three-cubes-sapien-others-painted-sam3-2"
    )
    assert calls[0][0] == 2
    assert calls[0][1]["pictures"] == "others-painted"
    assert calls[0][1]["order"] == ("green", "red", "blue")
    # A command runs the episode length of the simulator's bin scenes.
    assert calls[0][1]["steps_per_pick"] == three_cube_rollout.DEFAULT_STEPS_PER_PICK
    assert calls[0][1]["destination"].name.startswith(
        "random-three-cubes-sapien-others-painted-2"
    )
    assert calls[1][0] == 1
    assert calls[1][1]["pictures"] == "unedited"
    assert calls[1][1]["order"] is None
    assert calls[1][1]["steps_per_pick"] == 7
    assert calls[1][1]["environment_seed_base"] == 5
    assert calls[1][1]["destination"].name.startswith(
        "random-three-cubes-sapien-unedited-2"
    )


@pytest.mark.parametrize(
    "arguments",
    (
        # The three-cube task has to be told what SmolVLA is shown.
        ["--task", "pick-place-three-cubes"],
        ["--task", "pick-place-three-cubes", "--pictures", "blurred"],
        ["--task", "pick-place-three-cubes", "--pictures", "unedited", "--order", "red,red,blue"],
        ["--task", "pick-place-three-cubes", "--pictures", "unedited", "--outlines", "sam4"],
        # The other tasks have neither pictures to edit nor an order, nor
        # outlines to edit by.
        ["--task", "pick-place", "--pictures", "unedited"],
        ["--task", "stack3-color", "--order", "green,red,blue"],
        ["--task", "pick-place", "--outlines", "sam3"],
    ),
)
def test_rollout_cli_refuses_three_cube_options_that_do_not_fit(monkeypatch, arguments):
    from so101_smolvla import sapien_rollout, three_cube_rollout

    def must_not_run(*args, **kwargs):
        raise AssertionError("a refused command line started a batch")

    monkeypatch.setattr(three_cube_rollout, "run_three_cube_batch", must_not_run)
    monkeypatch.setattr(sapien_rollout, "run_batch", must_not_run)
    monkeypatch.setattr(sapien_rollout, "run_stack_batch", must_not_run)
    with pytest.raises(SystemExit) as refused:
        sapien_rollout.main(arguments)
    assert refused.value.code == 2


def test_bin_report_gives_the_finger_angle_of_every_test(tmp_path):
    from so101_smolvla.sapien_rollout import _write_report

    def row(seed: int, success: bool, contact: bool, angle: float, roll: float) -> dict:
        joints = {
            "shoulder_lift": {"reversal_rate": 0.05},
            "elbow_flex": {"reversal_rate": 0.10},
        }
        return {
            "environment_seed": seed,
            "policy_seed": 100 + seed,
            "success": success,
            "picked_up": success,
            "held": success,
            "grip": contact,
            "finger_face_deg": angle,
            "grip_roll_deg": roll,
            "raw_model_targets": joints,
            "achieved_joints": joints,
        }

    rows = [row(0, True, True, 5.0, -8.0), row(1, False, False, 35.0, 2.5)]
    text = _write_report(rows, tmp_path).read_text(encoding="utf-8")
    assert "Finger-to-face angle at grip: mean 20.0°, median 20.0°, max 35.0°. Ideal is 0°." in text
    assert "| 0 | 100 | pass | pass | pass | pass | 5.0 | -8.0 | 5.0% | 5.0% | 10.0% | 10.0% |" in text
    assert "| 1 | 101 | fail | fail | fail | fail | 35.0 | 2.5 | 5.0% | 5.0% | 10.0% | 10.0% |" in text


def test_bin_batch_rejects_a_stacking_task_before_creating_output(tmp_path):
    from so101_smolvla.sapien_rollout import run_batch

    destination = tmp_path / "batch"
    with pytest.raises(ValueError, match="unsupported pick-and-place task"):
        run_batch(1, destination=destination, task_mode="stack2-training")
    assert not destination.exists()
