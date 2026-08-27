# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen4-Exp PLE resident/mmap size accounting."""

import json
import struct

from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
    prefault_qwen4_exp_checkpoint,
    qwen4_exp_residency_estimate,
)


def _write_safetensors(path, tensors: dict[str, int]) -> None:
    offset = 0
    header = {}
    for key, size in tensors.items():
        header[key] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_estimate_subtracts_only_mmap_backed_ngram_tensors(tmp_path):
    model = tmp_path / "qwen4"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    ple_key = (
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight"
    )
    compute_key = "model.language_model.layers.0.self_attn.q_proj.weight"
    _write_safetensors(model / "model.safetensors", {ple_key: 100, compute_key: 40})
    (model / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    ple_key: "model.safetensors",
                    compute_key: "model.safetensors",
                }
            }
        )
    )

    estimate = qwen4_exp_residency_estimate(model)

    assert estimate.supported is True
    assert estimate.ple_bytes == 100
    assert estimate.resident_bytes == int(estimate.checkpoint_bytes * 1.05)
    assert estimate.mmap_bytes == int((estimate.checkpoint_bytes - 100) * 1.05)


def test_offload_is_forced_only_when_it_makes_the_model_fit(tmp_path):
    model = tmp_path / "qwen4"
    model.mkdir()
    ple_key = "model.language_model.ngram_embedding.shard_0.weight"
    _write_safetensors(model / "model.safetensors", {ple_key: 100})
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {ple_key: "model.safetensors"}})
    )
    estimate = qwen4_exp_residency_estimate(model)

    between_modes = (estimate.resident_bytes + estimate.mmap_bytes) // 2
    assert estimate.force_ssd_offload(between_modes) is True
    assert estimate.force_ssd_offload(estimate.resident_bytes) is False
    assert estimate.force_ssd_offload(max(0, estimate.mmap_bytes - 1)) is False


def test_prefault_reads_checkpoint_once_and_honors_kill_switch(tmp_path, monkeypatch):
    model = tmp_path / "qwen4"
    model.mkdir()
    checkpoint = model / "model.safetensors"
    _write_safetensors(
        checkpoint,
        {
            "model.layer.weight": 4096,
            "model.ngram_embedding.shard_0.weight": 8192,
        },
    )

    read_bytes, elapsed = prefault_qwen4_exp_checkpoint(model, chunk_bytes=257)
    assert read_bytes == checkpoint.stat().st_size
    assert elapsed >= 0

    compute_bytes, _ = prefault_qwen4_exp_checkpoint(
        model, chunk_bytes=257, include_ple=False
    )
    assert 4096 < compute_bytes < read_bytes

    monkeypatch.setenv("OMLX_QWEN4_PREFAULT", "0")
    assert prefault_qwen4_exp_checkpoint(model) == (0, 0.0)
