# SPDX-License-Identifier: Apache-2.0
"""Contract tests for oMLX's MiMo V2 extension to dflash-mlx."""

import json
import zipfile
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip("dflash_mlx")
pytest.importorskip("mlx_lm.models.mimo_v2_flash")


def _target_config(**overrides):
    config = dict(
        model_type="mimo_v2_flash",
        num_experts_per_tok=1,
        hybrid_layer_pattern=[0, 1],
        moe_layer_freq=[0, 0],
        add_swa_attention_sink_bias=True,
        add_full_attention_sink_bias=True,
        sliding_window_size=4,
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=2,
        routed_scaling_factor=1.0,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        max_position_embeddings=256,
        layernorm_epsilon=1e-6,
        rope_theta=10000.0,
        swa_rope_theta=10000.0,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        swa_head_dim=8,
        swa_v_head_dim=8,
        partial_rotary_factor=0.5,
    )
    config.update(overrides)
    return config


def _draft_config(**overrides):
    config = dict(
        architectures=["DFlashDraftModel"],
        model_type="qwen3",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        v_head_dim=8,
        partial_rotary_factor=0.5,
        block_size=4,
        layer_types=["sliding_attention", "sliding_attention"],
        sliding_window=16,
        is_causal=False,
        num_target_layers=2,
        vocab_size=128,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        attention_bias=False,
        dflash_config={
            "target_layer_ids": [0, 1],
            "mask_token_id": 127,
            "block_size": 4,
            "attention_value_scale": 0.612,
            "attention_sink_bias": True,
        },
    )
    config.update(overrides)
    return config


def _target_model():
    from mlx_lm.models import mimo_v2_flash

    return mimo_v2_flash.Model(mimo_v2_flash.ModelArgs.from_dict(_target_config()))


def _assert_close(actual, expected, atol=1e-5):
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual - expected)).item()) <= atol


def _write_mask_archive(path, values):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mask_embedding/byteorder", "little")
        archive.writestr("mask_embedding/data/0", values)


def test_installer_registers_target_backend_and_mimo_draft_classes():
    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    from omlx.patches.dflash_mimo_v2 import (
        MiMoDFlashDraftModel,
        MiMoDFlashDraftModelArgs,
        install_dflash_mimo_v2_backend,
    )

    install_dflash_mimo_v2_backend()

    assert "omlx.patches.dflash_mimo_v2:MiMoV2TargetOps" in target_ops.TARGET_BACKENDS
    assert loading._get_dflash_model_classes(_draft_config()) == (
        MiMoDFlashDraftModel,
        MiMoDFlashDraftModelArgs,
    )


def test_target_ops_matches_native_forward_and_captures_requested_layers():
    from omlx.patches.dflash_mimo_v2 import MiMoV2TargetOps

    model = _target_model()
    ops = MiMoV2TargetOps()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

    expected = model(inputs, cache=model.make_cache())
    actual, captured = ops.forward_with_hidden_capture(
        model,
        input_ids=inputs,
        cache=model.make_cache(),
        capture_layer_ids={1, 2},
    )

    _assert_close(actual, expected)
    assert set(captured) == {1, 2}
    assert ops.extract_context_feature(captured, [0, 1]).shape == (1, 3, 64)


def test_target_ops_rewinds_full_and_rotating_cache_after_rejection():
    from omlx.patches.dflash_mimo_v2 import MiMoV2TargetOps

    model = _target_model()
    ops = MiMoV2TargetOps()
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32),
        cache=cache,
    )
    ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[7, 8, 9]], dtype=mx.int32),
        target_cache=cache,
    )

    assert {int(entry.offset) for entry in cache} == {9}
    ops.restore_after_acceptance(
        cache,
        target_len=6,
        acceptance_length=0,
        drafted_tokens=3,
    )
    assert {int(entry.offset) for entry in cache} == {6}


def test_draft_args_keep_mimo_attention_contract():
    from omlx.patches.dflash_mimo_v2 import MiMoDFlashDraftModelArgs

    args = MiMoDFlashDraftModelArgs.from_dict(_draft_config())

    assert args.partial_rotary_factor == 0.5
    assert args.attention_value_scale == 0.612
    assert args.attention_sink_bias is True
    assert args.v_head_dim == 8


def test_prepare_draft_loads_trained_bfloat16_mask(tmp_path):
    from omlx.patches.dflash_mimo_v2 import (
        MiMoDFlashDraftModel,
        MiMoDFlashDraftModelArgs,
        prepare_mimo_draft,
    )

    draft = MiMoDFlashDraftModel(MiMoDFlashDraftModelArgs.from_dict(_draft_config()))
    mask_bits = np.arange(32, dtype=np.uint16)
    _write_mask_archive(tmp_path / "mask_embedding.pt", mask_bits.tobytes())
    meta = {}

    prepare_mimo_draft(draft, meta, tmp_path)

    assert draft.mask_embedding.shape == (32,)
    assert draft.mask_embedding.dtype == mx.bfloat16
    assert meta["mask_embedding"] == "mask_embedding.pt"


def test_mimo_backend_uses_trained_mask_vector(monkeypatch):
    import omlx.patches.dflash_mimo_v2 as mimo_patch

    captured = {}

    class Draft:
        mask_token_id = 9
        mask_embedding = mx.array([4.0, 5.0], dtype=mx.float32)

        def forward_projected_context(self, *, noise_embedding, draft_context, cache):
            captured["noise"] = noise_embedding
            return noise_embedding

    class Embedding:
        def __call__(self, ids):
            return mx.stack([ids.astype(mx.float32), ids.astype(mx.float32)], -1)

    target_ops = SimpleNamespace(
        embed_tokens=lambda _model: Embedding(),
        logits_from_hidden=lambda _model, hidden: hidden,
    )
    monkeypatch.setattr(mimo_patch, "_draft_compute_dtype", lambda _model: None)
    backend = mimo_patch.MiMoEagerDraftBackend()
    hidden, _ = backend._draft_block_hidden_logits(
        target_model=object(),
        target_ops=target_ops,
        draft_model=Draft(),
        draft_cache=[],
        staged_first=mx.array([3], dtype=mx.int32),
        draft_context=mx.zeros((1, 1, 2)),
        block_len=3,
        mask_token_tail=mx.array([9, 9], dtype=mx.int32),
    )
    mx.eval(hidden, captured["noise"])

    assert captured["noise"][0, 0].tolist() == [3.0, 3.0]
    assert captured["noise"][0, 1].tolist() == [4.0, 5.0]
    assert captured["noise"][0, 2].tolist() == [4.0, 5.0]


def test_bundled_draft_resolves_only_complete_mimo_payload(tmp_path):
    from omlx.patches.dflash_mimo_v2 import resolve_bundled_mimo_draft

    (tmp_path / "config.json").write_text(json.dumps(_target_config()))
    draft = tmp_path / "dflash"
    draft.mkdir()
    (draft / "config.json").write_text(json.dumps(_draft_config()))
    (draft / "model.safetensors").write_bytes(b"weights")
    _write_mask_archive(draft / "mask_embedding.pt", b"\0" * 64)

    assert resolve_bundled_mimo_draft(tmp_path, None) == str(draft)
    assert resolve_bundled_mimo_draft(tmp_path, "other/draft") == "other/draft"


@pytest.mark.parametrize("model_type", ["mimo_v2", "mimo_v2_flash"])
def test_dflash_compatibility_gate_accepts_mimo(tmp_path, model_type):
    from omlx.engine.dflash import is_dflash_compatible

    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))

    assert is_dflash_compatible(tmp_path) == (True, "")
