import io
import json

import numpy as np
import pytest

from so101_smolvla.policy_env import create_policy_environment
from so101_smolvla.sapien_config import IMAGE_SHAPE
from so101_smolvla.sapien_protocol import (
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
