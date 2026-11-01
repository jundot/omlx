# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp cluster adapter: registration, capability and stage selection."""

from __future__ import annotations

import json
import struct

import mlx.core as mx

from omlx.patches.qwen4_exp_mlx_lm import apply_qwen4_exp_mlx_lm_patch


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def test_qwen4_registration_is_a_honored_pipeline_capability(tmp_path):
    from omlx.cluster.pipeline_compat import pipeline_support

    assert apply_qwen4_exp_mlx_lm_patch()
    import mlx_lm.models.qwen4_exp as module

    assert module.SUPPORTS_PIPELINE is True
    assert module.HONORS_PIPELINE_ASSIGNMENT is True
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    assert pipeline_support(tmp_path) == "honored"


def test_qwen4_stage_starts_keep_the_ple_table_on_the_head():
    import mlx_lm.models.qwen4_exp as module

    assert module.pipeline_stage_starts(
        {"text_config": {"num_hidden_layers": 6, "ple_layer_ids": [2]}}
    ) == (2, 3, 4, 5)
    assert module.pipeline_stage_starts({"text_config": {"num_hidden_layers": 6}}) == ()


def test_stage_keys_select_layers_heads_and_mtp_per_rank():
    from omlx.patches.qwen4_exp_mlx_lm.model import _stage_key_needed

    kwargs = dict(start=2, end=4, rank=0, mtp=True, mmap_ple=True)
    assert not _stage_key_needed("vision_tower.patch_embed.weight", **kwargs)
    assert _stage_key_needed(
        "language_model.model.layers.2.mlp.down_proj.weight", **kwargs
    )
    assert not _stage_key_needed(
        "language_model.model.layers.4.mlp.down_proj.weight", **kwargs
    )
    assert _stage_key_needed("language_model.lm_head.weight", **kwargs)
    assert not _stage_key_needed(
        "language_model.lm_head.weight", **{**kwargs, "rank": 1}
    )
    assert _stage_key_needed("mtp.layers.0.self_attn.q_proj.weight", **kwargs)
    assert not _stage_key_needed(
        "mtp.layers.0.self_attn.q_proj.weight", **{**kwargs, "rank": 1}
    )
    ple = (
        "language_model.model.layers.2.ple.ple_embedding."
        "ngram_embedding.shards.0.weight"
    )
    assert not _stage_key_needed(ple, **kwargs)
    assert _stage_key_needed(ple, **{**kwargs, "mmap_ple": False})


def test_stage_weights_only_read_shards_that_hold_the_stage(tmp_path):
    from omlx.patches.qwen4_exp_mlx_lm.model import _stage_weights

    _write_safetensors(
        tmp_path / "a.safetensors",
        [
            ("language_model.model.layers.0.mlp.down_proj.weight", 4),
            ("language_model.lm_head.weight", 4),
        ],
    )
    _write_safetensors(
        tmp_path / "b.safetensors",
        [
            ("language_model.model.layers.9.mlp.down_proj.weight", 4),
            ("vision_tower.patch_embed.weight", 4),
            ("mtp.layers.0.self_attn.q_proj.weight", 4),
        ],
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.0.mlp.down_proj.weight": "a.safetensors",
                    "language_model.lm_head.weight": "a.safetensors",
                    "language_model.model.layers.9.mlp.down_proj.weight": "b.safetensors",
                    "vision_tower.patch_embed.weight": "b.safetensors",
                    "mtp.layers.0.self_attn.q_proj.weight": "b.safetensors",
                }
            }
        )
    )

    weights = _stage_weights(
        tmp_path,
        start=0,
        end=1,
        rank=0,
        mtp=True,
        mmap_ple=False,
    )

    assert set(weights) == {
        "language_model.model.layers.0.mlp.down_proj.weight",
        "language_model.lm_head.weight",
        "mtp.layers.0.self_attn.q_proj.weight",
    }
    for value in weights.values():
        mx.eval(value)
