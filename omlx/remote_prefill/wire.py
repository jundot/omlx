# SPDX-License-Identifier: Apache-2.0
"""MCDMA KV handoff protocol 1: message headers and the manifest a vLLM producer serves."""

from __future__ import annotations

import hashlib
import json
import struct
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

MAGIC = b"MKVH"
VERSION = 1
HEADER_BYTES = 128
# Requests the decoder sends, and the answers a producer gives.
OPEN, MANIFEST, WAIT, ERROR, PULL, DATA, CLOSE, ACK = range(1, 9)
_KINDS = frozenset(range(1, 9))
# Kinds whose payload follows the header.
_CARRIERS = frozenset({OPEN, MANIFEST, ERROR, DATA})
# Header flag: `crc` holds the zlib CRC-32 of the payload.
CHECKED = 1
# magic, version, kind, handoff, frame, frames, layer, flags, row_start, rows, nbytes, crc
_LAYOUT = struct.Struct("<4sHH16sIIIIQQQI")
# Element types a producer may export, by their numpy-compatible byte width.
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}
# What each dimension of an exported page array holds.
DIM_LABELS = frozenset(
    {"block", "token", "head", "head_dim", "kv", "kv_head_dim", "latent"}
)


class WireError(ValueError):
    """A handoff message is malformed or is not the answer the decoder asked for."""


@dataclass(frozen=True)
class Header:
    """One message header; DATA headers also say where the frame's pages belong."""

    kind: int
    handoff: bytes
    frame: int = 0
    frames: int = 0
    layer: int = 0
    flags: int = 0
    row_start: int = 0
    rows: int = 0
    nbytes: int = 0
    crc: int = 0


def new_handoff() -> bytes:
    """A fresh 16-byte handoff identifier."""
    return uuid.uuid4().bytes


def pack(header: Header) -> bytes:
    """Serialize a header into exactly HEADER_BYTES bytes."""
    if header.kind not in _KINDS or len(header.handoff) != 16:
        raise WireError("a handoff header needs a known kind and a 16-byte id")
    packed = _LAYOUT.pack(
        MAGIC,
        VERSION,
        header.kind,
        header.handoff,
        header.frame,
        header.frames,
        header.layer,
        header.flags,
        header.row_start,
        header.rows,
        header.nbytes,
        header.crc,
    )
    return packed.ljust(HEADER_BYTES, b"\0")


def unpack(payload: memoryview | bytes) -> Header:
    """Parse the header at the start of `payload`, or raise WireError."""
    if len(payload) < HEADER_BYTES:
        raise WireError("message is shorter than a handoff header")
    magic, version, kind, *fields = _LAYOUT.unpack_from(payload, 0)
    if magic != MAGIC or version != VERSION:
        raise WireError("message is not a KV handoff protocol 1 header")
    if kind not in _KINDS:
        raise WireError(f"unknown handoff message kind {kind}")
    header = Header(kind, *fields)
    if kind in _CARRIERS and len(payload) < HEADER_BYTES + header.nbytes:
        raise WireError("handoff payload is shorter than its header says")
    return header


def body(payload: memoryview | bytes, header: Header) -> memoryview:
    """The payload that follows `header`."""
    return memoryview(payload)[HEADER_BYTES : HEADER_BYTES + header.nbytes]


def token_sha256(tokens: list[int]) -> str:
    """The digest producers and decoders compare to prove they hold the same prompt."""
    return hashlib.sha256(np.asarray(tokens, dtype="<u4").tobytes()).hexdigest()


def _int(value: Any, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise WireError(f"manifest {name} must be an integer of at least {minimum}")
    return value


@dataclass(frozen=True)
class LayerExport:
    """One attention layer's exported pages: their array shape and what each dimension holds."""

    index: int
    kind: str
    shape: tuple[int, ...]
    dims: tuple[str, ...]
    dtype: str
    heads: int = 0
    total_heads: int = 0
    head_size: int = 0
    latent_size: int = 0
    rope_size: int = 0

    @property
    def row_bytes(self) -> int:
        """Bytes of one exported page row (one cache block of this layer)."""
        rows = self.shape[self.dims.index("block")]
        return int(np.prod(self.shape)) * DTYPE_BYTES[self.dtype] // max(1, rows)

    @classmethod
    def from_dict(cls, raw: Any) -> LayerExport:
        if not isinstance(raw, dict):
            raise WireError("manifest layer must be an object")
        shape = raw.get("shape")
        dims = raw.get("dims")
        if (
            not isinstance(shape, list)
            or not isinstance(dims, list)
            or len(shape) != len(dims)
            or not 2 <= len(shape) <= 6
        ):
            raise WireError("manifest layer shape and dims must be matching lists")
        layer = cls(
            index=_int(raw.get("index"), "layer index"),
            kind=str(raw.get("kind", "")),
            shape=tuple(_int(size, "layer shape", 1) for size in shape),
            dims=tuple(dims),
            dtype=str(raw.get("dtype", "")),
            heads=_int(raw.get("heads", 0), "heads"),
            total_heads=_int(raw.get("total_heads", 0), "total heads"),
            head_size=_int(raw.get("head_size", 0), "head size"),
            latent_size=_int(raw.get("latent_size", 0), "latent size"),
            rope_size=_int(raw.get("rope_size", 0), "rope size"),
        )
        if layer.kind not in {"attention", "mla"}:
            raise WireError(f"layer {layer.index} has unknown kind {layer.kind!r}")
        if layer.dtype not in DTYPE_BYTES:
            raise WireError(
                f"layer {layer.index} has unsupported dtype {layer.dtype!r}"
            )
        if not set(layer.dims) <= DIM_LABELS or len(set(layer.dims)) != len(layer.dims):
            raise WireError(f"layer {layer.index} has unknown or repeated dims")
        if "block" not in layer.dims or "token" not in layer.dims:
            raise WireError(f"layer {layer.index} pages need block and token dims")
        return layer


@dataclass(frozen=True)
class Manifest:
    """What one producer rank exported for a handoff, frame count included."""

    handoff: str
    model: str
    prompt_tokens: int
    first_token: int
    token_sha256: str
    block_size: int
    tp_rank: int
    tp_size: int
    layers: tuple[LayerExport, ...]
    frames: int

    @classmethod
    def from_json(cls, raw: bytes | memoryview) -> Manifest:
        try:
            payload = json.loads(bytes(raw))
        except ValueError as exc:
            raise WireError(f"manifest is not JSON: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("protocol") != VERSION:
            raise WireError("manifest does not speak KV handoff protocol 1")
        layers = payload.get("layers")
        if not isinstance(layers, list) or not layers:
            raise WireError("manifest lists no layers")
        manifest = cls(
            handoff=str(payload.get("handoff", "")),
            model=str(payload.get("model", "")),
            prompt_tokens=_int(payload.get("prompt_tokens"), "prompt_tokens", 1),
            first_token=_int(payload.get("first_token"), "first_token"),
            token_sha256=str(payload.get("token_sha256", "")),
            block_size=_int(payload.get("block_size"), "block_size", 1),
            tp_rank=_int(payload.get("tp_rank"), "tp_rank"),
            tp_size=_int(payload.get("tp_size"), "tp_size", 1),
            layers=tuple(LayerExport.from_dict(layer) for layer in layers),
            frames=_int(payload.get("frames"), "frames", 1),
        )
        if manifest.first_token >= manifest.prompt_tokens:
            raise WireError("manifest exports no tokens")
        if manifest.tp_rank >= manifest.tp_size:
            raise WireError("manifest tp_rank is outside tp_size")
        if len({layer.index for layer in manifest.layers}) != len(manifest.layers):
            raise WireError("manifest repeats a layer")
        return manifest
