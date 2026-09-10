# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen4-Exp PLE resident/mmap size accounting."""

import json
import struct

from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
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
        "model.language_model.layers.1.ple.ple_embedding."
        "ngram_embedding.shard_0.weight"
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


def test_residency_uses_the_stable_ceiling_not_the_instantaneous_one(tmp_path):
    """A model swap must not push a table that fits onto SSD.

    engine_pool used to ask the admission ceiling, which reads vm_stat. On a
    swap that reading lands right after the previous model unloaded, before
    the OS returned its pages, so the ceiling dips below the resident size and
    the table is forced to SSD for the rest of that engine's life. Measured
    cost of that mistake on this machine: 35.3 -> 14.1 tok/s.
    """
    from omlx.engine_pool import EngineEntry, EnginePool

    model = tmp_path / "qwen4"
    model.mkdir()
    ple_key = "model.language_model.ngram_embedding.shard_0.weight"
    _write_safetensors(model / "model.safetensors", {ple_key: 100})
    (model / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {ple_key: "model.safetensors"}})
    )
    estimate = qwen4_exp_residency_estimate(model)

    pool = EnginePool.__new__(EnginePool)
    pool._get_admission_ceiling = None
    pool._get_admission_soft_target = None
    pool._get_final_ceiling = None
    entry = EngineEntry.__new__(EngineEntry)
    entry.model_id = "qwen4"
    entry.model_path = str(model)
    entry.config_model_type = "qwen4_exp"

    # The instantaneous ceiling has dipped between the two modes — exactly the
    # window right after a swap. The stable ceiling still clears resident.
    deprimido = (estimate.resident_bytes + estimate.mmap_bytes) // 2
    pool._get_admission_ceiling = lambda: deprimido
    pool._get_residency_ceiling = lambda: estimate.resident_bytes

    enabled, forced, _ = pool._qwen4_ple_offload_status(entry, None)
    assert forced is False, "a dip in the instantaneous ceiling must not force SSD"
    assert enabled is False

    # With no stable ceiling wired up we still fall back to the old behaviour.
    pool._get_residency_ceiling = None
    _, forced_sem_cb, _ = pool._qwen4_ple_offload_status(entry, None)
    assert forced_sem_cb is True
