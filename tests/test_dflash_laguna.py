# SPDX-License-Identifier: Apache-2.0
"""Contract tests for oMLX's Laguna extension to dflash-mlx."""

import mlx.core as mx
import pytest

pytest.importorskip("dflash_mlx")


def _target_config(**overrides):
    config = dict(
        model_type="laguna",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        qkv_bias=False,
        attention_bias=False,
        gating="per-head",
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        partial_rotary_factor=1.0,
        sliding_window=4,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ],
        num_attention_heads_per_layer=[4, 4, 4, 4],
        num_experts=0,
        mlp_only_layers=[],
    )
    config.update(overrides)
    return config


def _draft_config(**overrides):
    config = dict(
        model_type="laguna",
        architectures=["DFlashLagunaForCausalLM"],
        vocab_size=128,
        draft_vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=500000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        partial_rotary_factor=0.5,
        sliding_window=4,
        layer_types=["sliding_attention", "sliding_attention"],
        gating="per-head",
        dflash_config={
            "block_size": 4,
            "mask_token_id": 12,
            "num_target_layers": 4,
            "target_layer_ids": [0, 3],
            "causal": True,
        },
    )
    config.update(overrides)
    return config


def _target_model():
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()
    from mlx_lm.models import laguna

    return laguna.Model(laguna.ModelArgs(**_target_config()))


def _assert_close(actual, expected, atol=1e-5):
    mx.eval(actual, expected)
    assert float(mx.max(mx.abs(actual - expected)).item()) <= atol


def test_installer_registers_target_backend_and_laguna_draft_classes():
    from dflash_mlx.engine import target_ops
    from dflash_mlx.runtime import loading

    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
        install_dflash_laguna_backend,
    )

    install_dflash_laguna_backend()

    assert "omlx.patches.dflash_laguna:LagunaTargetOps" in target_ops.TARGET_BACKENDS
    assert loading._get_dflash_model_classes(_draft_config()) == (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
    )


def test_target_ops_matches_native_forward_and_captures_requested_layers():
    from omlx.patches.dflash_laguna import LagunaTargetOps

    model = _target_model()
    ops = LagunaTargetOps()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

    expected = model(inputs, cache=model.make_cache())
    actual, captured = ops.forward_with_hidden_capture(
        model,
        input_ids=inputs,
        cache=model.make_cache(),
        capture_layer_ids={1, 4},
    )

    _assert_close(actual, expected)
    assert set(captured) == {1, 4}
    assert ops.extract_context_feature(captured, [0, 3]).shape == (1, 3, 64)


def test_target_ops_rewinds_full_and_rotating_cache_after_rejection():
    from omlx.patches.dflash_laguna import LagunaTargetOps

    model = _target_model()
    ops = LagunaTargetOps()
    cache = ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
    )
    ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32),
        cache=cache,
        capture_layer_ids={1},
    )
    ops.verify_block(
        target_model=model,
        verify_ids=mx.array([[6, 7, 8]], dtype=mx.int32),
        target_cache=cache,
        capture_layer_ids={1},
    )

    assert {int(entry.offset) for entry in cache} == {8}
    ops.restore_after_acceptance(
        cache,
        target_len=6,
        acceptance_length=1,
        drafted_tokens=3,
    )
    assert {int(entry.offset) for entry in cache} == {6}

    # Rewinding a wrapped ring must preserve the same usable history as a
    # clean prefill of the accepted prefix, not merely restore its offset.
    clean_cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32),
        cache=clean_cache,
        capture_layer_ids={1},
    )
    expected, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[9]], dtype=mx.int32),
        cache=clean_cache,
        capture_layer_ids={1},
    )
    actual, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[9]], dtype=mx.int32),
        cache=cache,
        capture_layer_ids={1},
    )
    _assert_close(actual, expected)


def test_target_ops_prefix_snapshot_round_trip_preserves_mixed_cache():
    from dflash_mlx.cache.codecs import build_snapshot, hydrate_target_cache
    from dflash_mlx.cache.fingerprints import DFlashPrefixKey
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.patches.dflash_laguna import LagunaTargetOps

    model = _target_model()
    ops = LagunaTargetOps()
    capabilities = ops.capabilities_for(model)
    assert capabilities.supports_prefix_snapshot is True
    assert capabilities.supports_rotating_cache_snapshot is True

    prefix_ids = [1, 2, 3, 4, 5, 6, 7]
    cache = ops.make_cache(model, enable_speculative_linear_cache=True)
    logits, captured = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([prefix_ids], dtype=mx.int32),
        cache=cache,
        capture_layer_ids={1, 4},
    )
    target_hidden = ops.extract_context_feature(captured, [0, 3])
    snapshot = build_snapshot(
        token_ids=prefix_ids,
        target_cache=cache,
        target_hidden=target_hidden,
        last_logits=logits[:, -1, :],
        key=DFlashPrefixKey(
            target_model_id="tiny-laguna-target",
            draft_model_id="tiny-laguna-draft",
            capture_layer_ids=(0, 3),
            draft_sink_size=2,
            draft_window_size=4,
            template_hash="template",
            prompt_policy_hash="policy",
        ),
        trim_target_hidden=False,
    )

    template = ops.make_cache(model, enable_speculative_linear_cache=True)
    hydrated = hydrate_target_cache(snapshot, template)

    assert isinstance(hydrated[0], KVCache)
    assert all(isinstance(entry, RotatingKVCache) for entry in hydrated[1:])
    assert [int(entry.offset) for entry in hydrated] == [len(prefix_ids)] * 4
    assert [len(state) for state in snapshot.fa_states] == [3, 4, 4, 4]
    assert [int(entry._idx) for entry in hydrated[1:]] == [
        int(state[3]) for state in snapshot.fa_states[1:]
    ]

    # A restored cache must produce the same continuation as the live cache,
    # including after the sliding-attention rings have wrapped.
    expected, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[8]], dtype=mx.int32),
        cache=cache,
        capture_layer_ids={1},
    )
    actual, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=mx.array([[8]], dtype=mx.int32),
        cache=hydrated,
        capture_layer_ids={1},
    )
    _assert_close(actual, expected)


def test_laguna_draft_decodes_trimmed_prefix_snapshot():
    from dflash_mlx.cache.codecs import build_snapshot
    from dflash_mlx.cache.fingerprints import DFlashPrefixKey
    from dflash_mlx.draft_backend import EagerDraftBackend
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.runtime import stream_dflash_generate
    from dflash_mlx.runtime.context import build_offline_runtime_context

    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
        LagunaTargetOps,
    )

    target = _target_model()
    ops = LagunaTargetOps()
    draft = LagunaDFlashDraftModel(
        LagunaDFlashDraftModelArgs.from_dict(_draft_config())
    )
    draft.bind_target_model(target, target_ops=ops)

    prefix_ids = [1, 2, 3, 4, 5, 6, 7]
    target_cache = ops.make_cache(target, enable_speculative_linear_cache=True)
    logits, captured = ops.forward_with_hidden_capture(
        target,
        input_ids=mx.array([prefix_ids], dtype=mx.int32),
        cache=target_cache,
        capture_layer_ids={1, 4},
    )
    target_hidden = ops.extract_context_feature(captured, [0, 3])
    projected = draft.project_target_hidden(target_hidden)
    snapshot = build_snapshot(
        token_ids=prefix_ids,
        target_cache=target_cache,
        target_hidden=projected,
        last_logits=logits[:, -1, :],
        key=DFlashPrefixKey(
            target_model_id="tiny-laguna-target",
            draft_model_id="tiny-laguna-draft",
            capture_layer_ids=(0, 3),
            draft_sink_size=2,
            draft_window_size=4,
            template_hash="template",
            prompt_policy_hash="policy",
        ),
        draft_model=draft,
        trim_target_hidden=True,
        draft_sink_size=2,
        draft_window_size=4,
    )

    assert snapshot.target_hidden_chunk_spans == ((0, 2), (3, 7))

    def generate(prefix_snapshot=None, *, hit_kind="miss"):
        return list(
            stream_dflash_generate(
                target_model=target,
                target_ops=ops,
                tokenizer=None,
                draft_model=draft,
                draft_backend=EagerDraftBackend(),
                prompt_tokens_override=prefix_ids,
                max_new_tokens=3,
                block_tokens=4,
                stop_token_ids=[],
                prefix_snapshot=prefix_snapshot,
                prefix_hit_kind=hit_kind,
                publish_generation_snapshot=False,
                runtime_context=build_offline_runtime_context(
                    draft_sink_size=2,
                    draft_window_size=4,
                ),
            )
        )

    cold_events = generate()
    events = generate(snapshot, hit_kind="l1")
    cold_summary = next(
        event for event in cold_events if isinstance(event, SummaryEvent)
    )
    summary = next(event for event in events if isinstance(event, SummaryEvent))
    assert summary.generated_token_ids == cold_summary.generated_token_ids
    assert summary.generation_tokens == 3
    assert summary.hit_kind == "l1"
    assert summary.fallback_ar is False


def test_laguna_draft_advances_trimmed_projected_context():
    from dflash_mlx.cache.snapshot import TargetHiddenChunks
    from dflash_mlx.draft_backend import EagerDraftBackend

    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
    )

    draft = LagunaDFlashDraftModel(
        LagunaDFlashDraftModelArgs.from_dict(_draft_config())
    )
    backend = EagerDraftBackend()
    dense = mx.arange(7 * 32, dtype=mx.float32).reshape(1, 7, 32)
    sparse = TargetHiddenChunks(
        total_len=7,
        chunks=(dense[:, :2, :], dense[:, 3:, :]),
        spans=((0, 2), (3, 7)),
    )
    dense_cache = backend.make_cache(
        draft_model=draft,
        sink_size=2,
        window_size=4,
    )
    sparse_cache = backend.make_cache(
        draft_model=draft,
        sink_size=2,
        window_size=4,
    )

    draft.advance_projected_context_cache(
        draft_context=dense,
        cache=dense_cache,
    )
    draft.advance_projected_context_cache(
        draft_context=sparse,
        cache=sparse_cache,
    )

    for dense_entry, sparse_entry in zip(dense_cache, sparse_cache, strict=True):
        dense_keys, dense_values = dense_entry.fetch()
        sparse_keys, sparse_values = sparse_entry.fetch()
        _assert_close(sparse_keys, dense_keys)
        _assert_close(sparse_values, dense_values)
        _assert_close(sparse_entry.position_indices(), dense_entry.position_indices())
        assert sparse_entry.offset == dense_entry.offset == 7


def test_laguna_draft_normalizes_nested_config_and_builds_gated_layers():
    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
    )

    args = LagunaDFlashDraftModelArgs.from_dict(_draft_config())
    draft = LagunaDFlashDraftModel(args)

    assert args.block_size == 4
    assert args.num_target_layers == 4
    assert args.tie_word_embeddings is True
    assert draft.target_layer_ids == [0, 3]
    assert len(draft.aux_hidden_norms) == 2
    assert draft.layers[0].self_attn.g_proj.weight.shape[0] == 4
    assert draft.layers[0].self_attn.rope.dims == 4


def test_laguna_draft_forward_uses_aux_norms_and_binds_matching_target():
    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
        LagunaTargetOps,
    )

    target = _target_model()
    draft = LagunaDFlashDraftModel(
        LagunaDFlashDraftModelArgs.from_dict(_draft_config())
    )
    draft.bind_target_model(target, target_ops=LagunaTargetOps())

    result = draft(
        noise_embedding=mx.zeros((1, 3, 32)),
        target_hidden=mx.zeros((1, 5, 64)),
    )
    mx.eval(result)
    assert result.shape == (1, 3, 32)


def test_laguna_draft_sanitize_splits_poolside_fused_qkv_layout():
    from omlx.patches.dflash_laguna import (
        LagunaDFlashDraftModel,
        LagunaDFlashDraftModelArgs,
    )

    draft = LagunaDFlashDraftModel(
        LagunaDFlashDraftModelArgs.from_dict(_draft_config())
    )
    # q=4*8, k=2*8, v=2*8: this is the layout used by Poolside's
    # model.safetensors, scaled down to the tiny test configuration.
    fused = mx.arange(64 * 32).reshape(64, 32)
    fused_scales = mx.arange(64 * 2).reshape(64, 2)
    weights = draft.sanitize(
        {
            "layers.0.self_attn.qkv_proj.weight": fused,
            "layers.0.self_attn.qkv_proj.scales": fused_scales,
            "norm.weight": mx.ones(32),
        }
    )

    assert "layers.0.self_attn.qkv_proj.weight" not in weights
    assert weights["layers.0.self_attn.q_proj.weight"].shape == (32, 32)
    assert weights["layers.0.self_attn.k_proj.weight"].shape == (16, 32)
    assert weights["layers.0.self_attn.v_proj.weight"].shape == (16, 32)
    assert weights["layers.0.self_attn.q_proj.scales"].shape == (32, 2)
    assert weights["layers.0.self_attn.k_proj.scales"].shape == (16, 2)
    assert weights["layers.0.self_attn.v_proj.scales"].shape == (16, 2)
    _assert_close(weights["layers.0.self_attn.q_proj.weight"], fused[:32])
    _assert_close(weights["layers.0.self_attn.k_proj.weight"], fused[32:48])
    _assert_close(weights["layers.0.self_attn.v_proj.weight"], fused[48:])


def test_laguna_draft_rejects_mixed_attention_flavors():
    from omlx.patches.dflash_laguna import LagunaDFlashDraftModelArgs

    with pytest.raises(ValueError, match="one attention type"):
        LagunaDFlashDraftModelArgs.from_dict(
            _draft_config(layer_types=["full_attention", "sliding_attention"])
        )


def test_target_ops_logits_last_only_slices_before_lm_head():
    """logits_last_only=True must equal full-logits[:, -1:, :] at tolerance.

    The DFlash target path slices the post-norm hidden states to the last
    position BEFORE the vocabulary head (Swift lagunaLastTokenHidden), so the
    prefill lm_head never computes the dead [L-1, vocab] slab. A [1,1,H] head
    matmul is ULP-divergent from the [B,L,H] full matmul (frame divergence,
    see docs/laguna-mlxfast-port-correctness.md C2); asserted at the repo
    tolerance, matching the DFlash reference layer's frame-divergence tolerance.
    """
    from omlx.patches.dflash_laguna import LagunaTargetOps

    model = _target_model()
    ops = LagunaTargetOps()
    inputs = mx.array([[1, 2, 3]], dtype=mx.int32)

    full, _ = ops.forward_with_hidden_capture(
        model,
        input_ids=inputs,
        cache=model.make_cache(),
    )
    last_only, captured = ops.forward_with_hidden_capture(
        model,
        input_ids=inputs,
        cache=model.make_cache(),
        capture_layer_ids={1},
        logits_last_only=True,
    )
    assert last_only.shape == (1, 1, full.shape[-1])
    _assert_close(last_only, full[:, -1:, :])
    assert set(captured) == {1, -1}
