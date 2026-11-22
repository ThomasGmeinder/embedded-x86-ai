"""Versioned framed protocol shared by the SAPIEN parent and worker."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from so101_smolvla.sapien_config import IMAGE_SHAPE

RESPONSE_HEADER = struct.Struct(">BII")
METADATA_LENGTH = struct.Struct(">I")
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 4 * 1024 * 1024
IMAGE_BYTES = int(np.prod(IMAGE_SHAPE))
# An outline map says for every pixel of a picture what it shows.
OUTLINE_SHAPE = IMAGE_SHAPE[:2]
OUTLINE_BYTES = int(np.prod(OUTLINE_SHAPE))


@dataclass(frozen=True)
class Response:
    request_id: int
    metadata: dict
    top: np.ndarray | None = None
    wrist: np.ndarray | None = None
    top_outline: np.ndarray | None = None
    wrist_outline: np.ndarray | None = None


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise RuntimeError(
                f"SAPIEN worker closed after {len(chunks)} of {size} response bytes"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def encode_payload(
    metadata: dict,
    top: np.ndarray | None = None,
    wrist: np.ndarray | None = None,
    top_outline: np.ndarray | None = None,
    wrist_outline: np.ndarray | None = None,
) -> bytes:
    document = json.dumps(metadata, separators=(",", ":"), allow_nan=False).encode()
    payload = METADATA_LENGTH.pack(len(document)) + document
    if (top is None) != (wrist is None):
        raise ValueError("top and wrist images must be present together")
    if (top_outline is None) != (wrist_outline is None):
        raise ValueError("top and wrist outline maps must be present together")
    if top_outline is not None and top is None:
        raise ValueError("outline maps need the pictures they belong to")
    if top is not None and wrist is not None:
        frames = []
        for name, frame in (("top", top), ("wrist", wrist)):
            image = np.ascontiguousarray(frame, dtype=np.uint8)
            if image.shape != IMAGE_SHAPE:
                raise ValueError(f"{name} image shape {image.shape}, expected {IMAGE_SHAPE}")
            frames.append(image.tobytes())
        payload += b"".join(frames)
    if top_outline is not None and wrist_outline is not None:
        maps = []
        for name, outline in (("top", top_outline), ("wrist", wrist_outline)):
            codes = np.ascontiguousarray(outline)
            if codes.shape != OUTLINE_SHAPE or codes.dtype != np.uint8:
                raise ValueError(
                    f"{name} outline map is {codes.dtype} of shape {codes.shape}, "
                    f"expected uint8 of shape {OUTLINE_SHAPE}"
                )
            maps.append(codes.tobytes())
        payload += b"".join(maps)
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response payload is too large: {len(payload)} bytes")
    return payload


def decode_payload(request_id: int, payload: bytes) -> Response:
    if len(payload) < METADATA_LENGTH.size:
        raise RuntimeError("SAPIEN response is missing metadata length")
    metadata_size = METADATA_LENGTH.unpack(payload[: METADATA_LENGTH.size])[0]
    metadata_end = METADATA_LENGTH.size + metadata_size
    if metadata_end > len(payload):
        raise RuntimeError("SAPIEN response metadata length exceeds payload")
    metadata = json.loads(payload[METADATA_LENGTH.size : metadata_end])
    image_payload = payload[metadata_end:]
    if not image_payload:
        return Response(request_id=request_id, metadata=metadata)
    pictures_only = 2 * IMAGE_BYTES
    with_outlines = pictures_only + 2 * OUTLINE_BYTES
    if len(image_payload) not in (pictures_only, with_outlines):
        raise RuntimeError(
            f"SAPIEN response has {len(image_payload)} image bytes, expected "
            f"{pictures_only} or, with outline maps, {with_outlines}"
        )
    images = np.frombuffer(
        image_payload[:pictures_only], dtype=np.uint8
    ).reshape(2, *IMAGE_SHAPE)
    if len(image_payload) == pictures_only:
        return Response(
            request_id=request_id,
            metadata=metadata,
            top=images[0].copy(),
            wrist=images[1].copy(),
        )
    outlines = np.frombuffer(
        image_payload[pictures_only:], dtype=np.uint8
    ).reshape(2, *OUTLINE_SHAPE)
    return Response(
        request_id=request_id,
        metadata=metadata,
        top=images[0].copy(),
        wrist=images[1].copy(),
        top_outline=outlines[0].copy(),
        wrist_outline=outlines[1].copy(),
    )


def read_response(stream: BinaryIO, expected_request_id: int) -> Response:
    status, request_id, size = RESPONSE_HEADER.unpack(
        read_exact(stream, RESPONSE_HEADER.size)
    )
    if request_id != expected_request_id:
        raise RuntimeError(
            f"SAPIEN response id {request_id} does not match request {expected_request_id}"
        )
    if size > MAX_RESPONSE_BYTES:
        raise RuntimeError(f"SAPIEN response announces oversized payload: {size} bytes")
    payload = read_exact(stream, size)
    if status != 1:
        raise RuntimeError(payload.decode("utf-8", errors="replace"))
    return decode_payload(request_id, payload)
