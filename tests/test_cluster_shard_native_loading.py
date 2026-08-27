# SPDX-License-Identifier: Apache-2.0
"""Exactness/safety gates for the shard-native safetensors prototype."""

from __future__ import annotations

import hashlib
import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from omlx.cluster.shard_native_loading import (
    LocalSafetensors,
    TensorPartition,
    deepseek_v4_partition,
    validate_quantized_partition,
)


def _write_model(tmp_path, tensors):
    config = b'{"model_type":"deepseek_v4"}'
    (tmp_path / "config.json").write_bytes(config)
    header = {"__metadata__": {"format": "mlx"}}
    payload = bytearray()
    weight_map = {}
    for name, (dtype, shape, raw) in tensors.items():
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
        weight_map[name] = "model.safetensors"
    encoded = json.dumps(header, separators=(",", ":")).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )


def _u32(value):
    return np.asarray(value, dtype="<u4").tobytes()


def _bf16_bits(value):
    return np.asarray(value, dtype="<u2").tobytes()


def test_segmented_weighted_slice_is_exact_and_manifest_pins_bytes(tmp_path):
    source = np.arange(8 * 8 * 3, dtype=np.uint32).reshape(64, 3)
    _write_model(
        tmp_path,
        {"layers.0.attn.wq_b.weight": ("U32", source.shape, source.tobytes())},
    )
    checkpoint = LocalSafetensors(tmp_path)
    partition = deepseek_v4_partition(
        "layers.0.attn.wq_b.weight",
        rank=0,
        shard_weights=(3, 5),
        world_size=2,
    )
    entry = checkpoint.manifest_entry("layers.0.attn.wq_b.weight", partition)
    expected = np.concatenate(
        [segment[:3] for segment in np.split(source, 8, axis=0)], axis=0
    )

    loaded = checkpoint.load_entry(entry, mx_module=mx)

    assert loaded.shape == (24, 3)
    assert np.array_equal(np.asarray(loaded), expected)
    assert entry.local_sha256 == hashlib.sha256(expected.tobytes()).hexdigest()

    path = tmp_path / "model.safetensors"
    payload = bytearray(path.read_bytes())
    header_length = struct.unpack("<Q", payload[:8])[0]
    payload[8 + header_length] ^= 1  # first byte is owned by rank zero
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        checkpoint.load_entry(entry, mx_module=mx)


def test_quant_boundary_and_metadata_corruption_fail_closed(tmp_path):
    source = np.arange(128, dtype=np.uint32).reshape(32, 4)
    _write_model(tmp_path, {"x": ("U32", source.shape, source.tobytes())})
    checkpoint = LocalSafetensors(tmp_path)
    with pytest.raises(ValueError, match="quantization group"):
        checkpoint.manifest_entry(
            "x",
            TensorPartition(
                axis=0,
                rank=0,
                weights=(3, 5),
                boundary_multiple=8,
            ),
        )

    path = tmp_path / "model.safetensors"
    raw = path.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + length])
    header["x"]["data_offsets"][1] -= 4
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + length :])
    with pytest.raises(ValueError, match="offsets do not match"):
        LocalSafetensors(tmp_path).descriptor("x")


def test_fixed_bf16_is_replicated_bit_exact_and_urls_are_rejected(tmp_path):
    bits = np.asarray([0x3F80, 0xBF80, 0x7FC1, 0x0001], dtype=np.uint16)
    _write_model(
        tmp_path,
        {"embed.weight": ("BF16", (2, 2), _bf16_bits(bits))},
    )
    checkpoint = LocalSafetensors(tmp_path)
    partition = deepseek_v4_partition(
        "embed.weight", rank=1, shard_weights=(3, 5), world_size=2
    )
    assert partition.axis is None
    entry = checkpoint.manifest_entry("embed.weight", partition)
    loaded = checkpoint.load_entry(entry, mx_module=mx)
    assert loaded.shape == (2, 2)
    assert np.array_equal(np.asarray(loaded.view(mx.uint16)), bits.reshape(2, 2))
    with pytest.raises(ValueError, match="complete local checkpoint"):
        LocalSafetensors("https://huggingface.co/model")


@pytest.mark.parametrize(
    "name,axis,weights",
    [
        ("head.weight", 0, (1, 1)),
        ("lm_head.scales", 0, (1, 1)),
        ("mtp.2.markov_head.markov_w2.weight", 0, (1, 1)),
        ("norm.weight", None, (1,)),
        ("mtp.0.norm.weight", None, (1,)),
    ],
)
def test_output_auxiliary_and_fixed_partition_contract(name, axis, weights):
    partition = deepseek_v4_partition(name, rank=1, shard_weights=(3, 5), world_size=2)
    assert partition.axis == axis
    assert partition.weights == weights


def test_model_identity_changes_with_checkpoint_manifest(tmp_path):
    _write_model(tmp_path, {"x": ("U32", (1,), _u32([7]))})
    original = LocalSafetensors(tmp_path)
    manifest = original.manifest({"x": TensorPartition()})
    original.verify_manifest(manifest)
    first = original.model_identity
    index = tmp_path / "model.safetensors.index.json"
    payload = json.loads(index.read_text())
    payload["metadata"] = {"revision": "different"}
    index.write_text(json.dumps(payload))
    second = LocalSafetensors(tmp_path).model_identity
    assert first != second
    with pytest.raises(ValueError, match="different model identity"):
        LocalSafetensors(tmp_path).verify_manifest(manifest)


def test_quantized_weight_and_scale_partitions_cover_same_input_groups(tmp_path):
    weight = np.arange(8 * 16, dtype=np.uint32).reshape(8, 16)
    scales = np.arange(8 * 4, dtype=np.uint8).reshape(8, 4)
    _write_model(
        tmp_path,
        {
            "w.weight": ("U32", weight.shape, weight.tobytes()),
            "w.scales": ("U8", scales.shape, scales.tobytes()),
        },
    )
    checkpoint = LocalSafetensors(tmp_path)
    partition = TensorPartition(axis=-1, rank=0, weights=(1, 1))
    validate_quantized_partition(
        checkpoint.descriptor("w.weight"),
        checkpoint.descriptor("w.scales"),
        partition,
        bits=8,
        group_size=16,
    )
    with pytest.raises(ValueError, match="cover different inputs"):
        validate_quantized_partition(
            checkpoint.descriptor("w.weight"),
            checkpoint.descriptor("w.scales"),
            partition,
            bits=4,
            group_size=16,
        )
