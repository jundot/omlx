# SPDX-License-Identifier: Apache-2.0
"""Handoff headers round trip, page layouts decode to the same keys and values, manifests are checked."""

from __future__ import annotations

import json

import mlx.core as mx
import numpy as np
import pytest

from omlx.remote_prefill import pages, wire

HANDOFF = bytes(range(16))


def test_headers_round_trip():
    header = wire.Header(wire.DATA, HANDOFF, 3, 9, 2, wire.CHECKED, 16, 4, 8, 77)
    packed = wire.pack(header)
    assert len(packed) == wire.HEADER_BYTES
    assert wire.unpack(packed + b"\0" * 8) == header


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"\0" * 10, "shorter than a handoff header"),
        (b"XXXX" + b"\0" * 124, "not a KV handoff protocol 1 header"),
        (
            wire.pack(wire.Header(wire.DATA, HANDOFF, nbytes=8)),
            "shorter than its header",
        ),
    ],
)
def test_malformed_headers_are_refused(payload, reason):
    with pytest.raises(wire.WireError, match=reason):
        wire.unpack(payload)


def test_a_header_needs_a_sixteen_byte_handoff():
    with pytest.raises(wire.WireError):
        wire.pack(wire.Header(wire.OPEN, b"short"))


def _layer(**changes):
    base = {
        "index": 0,
        "kind": "attention",
        "shape": [2, 2, 16, 32],
        "dims": ["block", "head", "token", "kv_head_dim"],
        "dtype": "bfloat16",
        "heads": 2,
        "total_heads": 2,
        "head_size": 16,
    }
    return {**base, **changes}


def _mla_layer(**changes):
    base = {
        "index": 0,
        "kind": "mla",
        "shape": [2, 16, 40],
        "dims": ["block", "token", "latent"],
        "dtype": "bfloat16",
        "latent_size": 32,
        "rope_size": 8,
    }
    return {**base, **changes}


def _manifest(**changes):
    base = {
        "protocol": 1,
        "handoff": HANDOFF.hex(),
        "model": "m",
        "prompt_tokens": 30,
        "first_token": 0,
        "token_sha256": "0" * 64,
        "block_size": 16,
        "tp_rank": 0,
        "tp_size": 1,
        "layers": [_layer()],
        "frames": 1,
    }
    return json.dumps({**base, **changes}).encode()


def test_a_manifest_parses_with_its_page_geometry():
    manifest = wire.Manifest.from_json(_manifest())
    (layer,) = manifest.layers
    assert layer.row_bytes == 2 * 16 * 32 * 2
    assert manifest.frames == 1 and manifest.block_size == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"protocol": 2},
        {"layers": []},
        {"first_token": 30},
        {"tp_rank": 1},
        {"layers": [_layer(), _layer()]},
        {"layers": [_layer(dtype="float8_e4m3")]},
        {"layers": [_layer(kind="mamba")]},
        {"layers": [_layer(dims=["block", "head", "slot", "kv_head_dim"])]},
        {"layers": [_layer(dims=["kv", "head", "token", "head_dim"])]},
        {"layers": [_layer(shape=[2, 2, 16])]},
        {"frames": 0},
        {"layers": [_layer(heads=0)]},
        {"layers": [_layer(head_size=0)]},
        {"layers": [_layer(total_heads=1)]},
        {"layers": [_mla_layer(latent_size=0)]},
        {"layers": [_mla_layer(rope_size=0)]},
    ],
)
def test_bad_manifests_are_refused(changes):
    with pytest.raises(wire.WireError):
        wire.Manifest.from_json(_manifest(**changes))


def test_an_mla_manifest_parses():
    (layer,) = wire.Manifest.from_json(_manifest(layers=[_mla_layer()])).layers
    assert (layer.latent_size, layer.rope_size) == (32, 8)


def test_a_manifest_that_is_not_json_is_refused():
    with pytest.raises(wire.WireError, match="not JSON"):
        wire.Manifest.from_json(b"{")


def test_prompts_are_compared_by_their_uint32_digest():
    assert wire.token_sha256([1, 2, 3]) == wire.token_sha256(np.array([1, 2, 3]))
    assert wire.token_sha256([1, 2, 3]) != wire.token_sha256([1, 2, 4])


TOKENS, HEADS, DIM = 32, 2, 8


def _kv():
    mx.random.seed(4)
    keys = mx.random.normal((TOKENS, HEADS, DIM)).astype(mx.bfloat16)
    values = mx.random.normal((TOKENS, HEADS, DIM)).astype(mx.bfloat16)
    return keys, values


def _arrange(keys, values, dims):
    """Lay token-major keys and values out as pages with the given dims."""
    blocks = TOKENS // 16
    k = keys.reshape(blocks, 16, HEADS, DIM)
    v = values.reshape(blocks, 16, HEADS, DIM)
    if "kv_head_dim" in dims:
        stacked = mx.concatenate([k, v], axis=-1)
        natural = ["block", "token", "head", "kv_head_dim"]
    else:
        stacked = mx.stack([k, v])
        natural = ["kv", "block", "token", "head", "head_dim"]
    arranged = stacked.transpose([natural.index(dim) for dim in dims])
    layer = wire.LayerExport(
        0,
        "attention",
        tuple(arranged.shape),
        tuple(dims),
        "bfloat16",
        HEADS,
        HEADS,
        DIM,
    )
    return layer, arranged


@pytest.mark.parametrize(
    "dims",
    [
        ("block", "head", "token", "kv_head_dim"),
        ("block", "token", "head", "kv_head_dim"),
        ("kv", "block", "token", "head", "head_dim"),
        ("block", "kv", "token", "head", "head_dim"),
        ("kv", "block", "head", "token", "head_dim"),
        ("block", "kv", "head", "token", "head_dim"),
    ],
)
def test_every_known_page_layout_decodes_to_the_same_keys_and_values(dims):
    keys, values = _kv()
    layer, arranged = _arrange(keys, values, dims)
    raw = arranged.view(mx.uint16)
    got_keys, got_values = pages.split_pages(mx, layer, raw)
    assert mx.array_equal(got_keys, keys).item()
    assert mx.array_equal(got_values, values).item()


def test_mla_pages_split_into_latent_and_rope():
    mx.random.seed(5)
    rows = mx.random.normal((2, 16, 12)).astype(mx.bfloat16)
    layer = wire.LayerExport(
        0,
        "mla",
        (2, 16, 12),
        ("block", "token", "latent"),
        "bfloat16",
        latent_size=8,
        rope_size=4,
    )
    latent, rope = pages.split_pages(mx, layer, rows.view(mx.uint16))
    assert mx.array_equal(latent, rows.reshape(32, 12)[:, :8]).item()
    assert mx.array_equal(rope, rows.reshape(32, 12)[:, 8:]).item()


def test_pages_that_disagree_with_the_declared_heads_are_refused():
    keys, values = _kv()
    layer, arranged = _arrange(keys, values, ("block", "head", "token", "kv_head_dim"))
    wrong = wire.LayerExport(
        0, "attention", layer.shape, layer.dims, "bfloat16", 4, 4, DIM // 2
    )
    with pytest.raises(wire.WireError, match="heads by size"):
        pages.split_pages(mx, wrong, arranged.view(mx.uint16))


def test_frames_take_their_row_count_from_the_header():
    layer = wire.LayerExport(
        0,
        "attention",
        (9, 2, 16, 32),
        ("block", "head", "token", "kv_head_dim"),
        "bfloat16",
        2,
        2,
        16,
    )
    assert pages.frame_shape(layer, 3) == (3, 2, 16, 32)
