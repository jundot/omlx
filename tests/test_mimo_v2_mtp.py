# SPDX-License-Identifier: Apache-2.0
"""MiMo-V2 native MTP head patch tests: attach gate, forward contract,
round routing, sanitize preservation and the model_loading whitelist."""

from __future__ import annotations

import mlx.core as mx

from omlx.utils.model_loading import _has_mtp_heads, _is_mtp_compatible


def _mimo_config(**overrides):
    cfg = {
        "model_type": "mimo_v2",
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 48,
        "moe_intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "v_head_dim": 8,
        "rope_theta": 10000.0,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 2,
        "swa_head_dim": 16,
        "swa_v_head_dim": 8,
        "swa_rope_theta": 1000.0,
        "sliding_window_size": 8,
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": True,
        "hybrid_layer_pattern": [1, 0],
        "moe_layer_freq": [0, 0],
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "noaux_tc",
        "partial_rotary_factor": 0.5,
        "attention_bias": False,
        "layernorm_epsilon": 1e-5,
        "max_position_embeddings": 128,
        "tie_word_embeddings": False,
        "num_nextn_predict_layers": 3,
    }
    cfg.update(overrides)
    return cfg


def _load_small_model(mtp_active: bool):
    from omlx.patches.mimo_v2 import apply_mimo_v2_patch, is_applied
    from omlx.patches.mlx_lm_mtp import (
        apply_mlx_lm_mtp_patch,
        set_mtp_active,
        set_mtp_depth,
    )

    apply_mimo_v2_patch()
    assert is_applied()
    assert apply_mlx_lm_mtp_patch()
    import mlx_lm.models.mimo_v2 as mimo

    set_mtp_active(mtp_active)
    set_mtp_depth(3)
    args = mimo.ModelArgs(**_mimo_config())
    try:
        return mimo.Model(args)
    finally:
        set_mtp_active(False)


class TestWhitelist:
    def test_mimo_v2_declared_heads_compatible(self):
        cfg = _mimo_config()
        assert _has_mtp_heads(cfg) is True
        assert _is_mtp_compatible(cfg, "mimo_v2") is True

    def test_no_heads_not_compatible(self):
        cfg = _mimo_config(num_nextn_predict_layers=0)
        assert _is_mtp_compatible(cfg, "mimo_v2") is False


class TestAttachGate:
    def test_active_attaches_head(self):
        model = _load_small_model(mtp_active=True)
        assert hasattr(model, "mtp")
        assert len(model.mtp.layers) == 3
        assert model._omlx_mtp_decode_enabled is True
        assert model._omlx_mtp_chain is True
        assert model._omlx_mtp_head_prenorm is True
        assert model._omlx_mtp_depth == 3

    def test_inactive_leaves_model_mtp_free(self):
        model = _load_small_model(mtp_active=False)
        assert not hasattr(model, "mtp")
        assert model._omlx_mtp_decode_enabled is False


class TestForwardContract:
    def test_backbone_return_hidden_split(self):
        model = _load_small_model(mtp_active=True)
        inputs = mx.arange(6).reshape(1, 6)
        logits, hidden = model(inputs, return_hidden=True)
        assert logits.shape == (1, 6, 64)
        assert hidden.shape == (1, 6, 32)
        # Pre-norm hidden must not be the trunk-normed tensor.
        normed = model.model.norm(hidden)
        assert not mx.allclose(normed, hidden, rtol=1e-3, atol=1e-3)

    def test_mtp_forward_shapes_and_logits_keep(self):
        model = _load_small_model(mtp_active=True)
        cache = model.make_mtp_cache()
        assert len(cache) == 3
        hidden = mx.random.normal((1, 5, 32), dtype=mx.float32)
        ids = mx.array([[1, 2, 3, 4, 5]])
        logits, head_hidden = model.mtp_forward(
            hidden, ids, cache, return_hidden=True, logits_keep=1
        )
        assert logits.shape == (1, 1, 64)
        assert head_hidden.shape == (1, 5, 32)

    def test_round_routing_walks_blocks(self):
        model = _load_small_model(mtp_active=True)
        cache = model.make_mtp_cache()
        hidden = mx.random.normal((1, 4, 32), dtype=mx.float32)
        ids = mx.array([[1, 2, 3, 4]])
        blocks = [id(layer) for layer in model.mtp.layers]
        seen_ids = []
        import mlx_lm.models.mimo_v2 as mimo

        original = mimo.MTPDecoderLayer.__call__

        def spy(self, *a, **kw):
            seen_ids.append(id(self))
            return original(self, *a, **kw)

        mimo.MTPDecoderLayer.__call__ = spy
        try:
            model.mtp_begin_cycle(cache, 3)
            model.mtp_forward(hidden, ids, cache)
            h = hidden[:, -1:]
            model.mtp_forward(h, ids[:, -1:], cache)
            model.mtp_forward(h, ids[:, -1:], cache)
        finally:
            mimo.MTPDecoderLayer.__call__ = original
        assert [id(layer) for layer in model.mtp.layers if any(id(layer) == s for s in seen_ids)] == [
            blocks[0],
            blocks[1],
            blocks[2],
        ]

    def test_round_clamps_at_last_block(self):
        model = _load_small_model(mtp_active=True)
        cache = model.make_mtp_cache()
        hidden = mx.random.normal((1, 2, 32), dtype=mx.float32)
        ids = mx.array([[7, 8]])
        for _ in range(6):
            out = model.mtp_forward(hidden, ids, cache)
            assert out.shape == (1, 2, 64)
        # Round counter keeps advancing but never escapes the block list.
        assert model.mtp._round == 6


class TestSanitize:
    def test_keeps_mtp_keys_when_enabled(self):
        model = _load_small_model(mtp_active=True)
        weights = {
            "model.mtp.layers.0.enorm.weight": mx.ones((32,), dtype=mx.bfloat16),
            "model.mtp.layers.0.eh_proj.weight": mx.ones(
                (32, 64), dtype=mx.bfloat16
            ),
            "visual.patch_bias": mx.ones((8,), dtype=mx.bfloat16),
            "model.embed_tokens.weight": mx.ones((64, 32), dtype=mx.bfloat16),
        }
        out = model.sanitize(dict(weights))
        assert "model.mtp.layers.0.enorm.weight" in out
        assert "model.mtp.layers.0.eh_proj.weight" in out
        assert "visual.patch_bias" not in out

    def test_strips_mtp_keys_when_disabled(self):
        model = _load_small_model(mtp_active=False)
        weights = {
            "model.mtp.layers.0.enorm.weight": mx.ones((32,), dtype=mx.bfloat16),
            "model.embed_tokens.weight": mx.ones((64, 32), dtype=mx.bfloat16),
        }
        out = model.sanitize(dict(weights))
        assert "model.mtp.layers.0.enorm.weight" not in out


class TestWeightBinding:
    def test_full_mtp_tree_binds(self):
        """Every checkpoint-style mtp key binds onto the attached head."""
        from mlx.utils import tree_flatten

        model = _load_small_model(mtp_active=True)
        keys = {k for k, _ in tree_flatten(model.parameters())}
        for i in range(3):
            p = f"model.mtp.layers.{i}"
            for key in (
                "enorm.weight",
                "hnorm.weight",
                "eh_proj.weight",
                "input_layernorm.weight",
                "pre_mlp_layernorm.weight",
                "final_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.attention_sink_bias",
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
                "mlp.down_proj.weight",
            ):
                assert f"{p}.{key}" in keys, f"{p}.{key}"

    def test_roundtrip_forward_after_bind(self):
        """Run the full draft cycle: backbone pre-norm hidden -> head."""
        model = _load_small_model(mtp_active=True)
        inputs = mx.arange(4).reshape(1, 4)
        cache = model.make_mtp_cache()
        logits, hidden = model(inputs, return_hidden=True)
        draft_logits = model.mtp_forward(hidden[:, -1:], inputs[:, -1:], cache)
        assert draft_logits.shape == (1, 1, 64)
        assert mx.isfinite(draft_logits).all()

    def test_backbone_accepts_n_confirmed_kwarg(self):
        """MTP chain verify always passes n_confirmed; MiMo accepts+ignores."""
        model = _load_small_model(mtp_active=True)
        inputs = mx.arange(4).reshape(1, 4)
        out, hidden = model(inputs, cache=None, return_hidden=True, n_confirmed=1)
        assert out.shape == (1, 4, 64)
        assert hidden.shape == (1, 4, 32)


class TestChainRollback:
    """Regression: MiMo lacked ``mtp_partial_rollback``, so every partial
    accept at depth>1 raised ``cache layer rejects chain rollback`` and each
    decode cycle fell back to a full re-prefill (0.5 tok/s)."""

    def test_head_clone_marker_set_on_chain_models(self):
        """Windowed head caches cannot be trimmed after rotation — the chain
        must draft on a per-cycle clone (head stays committed-only)."""
        model = _load_small_model(mtp_active=True)
        assert model._omlx_mtp_head_clone is True
        assert model.model._omlx_mtp_head_clone is True

    def test_rollback_trims_rejected_positions(self):
        """No rotation yet: every layer trims num_drafts - accepted."""
        model = _load_small_model(mtp_active=True)
        cache = model.make_cache()
        prompt = mx.arange(2).reshape(1, 2)
        model(prompt, cache=cache)
        verify = mx.array([[10, 11, 12, 13]])
        model(verify, cache=cache)
        assert [c.offset for c in cache] == [6, 6]
        assert model.mtp_partial_rollback(cache, accepted=1, num_drafts=3) is True
        assert [c.offset for c in cache] == [4, 4]

    def test_rotated_cache_rollback_matches_reference(self):
        """Rotated window layer (offset >= max_size) rolls back exactly via
        the undo stash armed around the verify forward, and continued decode
        after the rollback matches a reference run that never speculated."""
        from omlx.patches.mlx_lm_mtp import cache_rollback

        model = _load_small_model(mtp_active=True)
        prompt = mx.arange(6).reshape(1, 6)

        spec_cache = model.make_cache()
        model(prompt, cache=spec_cache)
        verify = mx.array([[10, 11, 12, 13]])  # [confirmed, d1, d2, d3]
        cache_rollback.set_undo_armed(True)
        try:
            model(verify, cache=spec_cache)
        finally:
            cache_rollback.set_undo_armed(False)
        assert spec_cache[0].offset >= spec_cache[0].max_size  # rotated
        # accepted=1 → keep [confirmed, d1], drop d2, d3.
        assert model.mtp_partial_rollback(spec_cache, accepted=1, num_drafts=3)
        assert spec_cache[0].offset == 8

        cont = mx.array([[20, 21]])
        out_spec = model(cont, cache=spec_cache)

        ref_cache = model.make_cache()
        model(prompt, cache=ref_cache)
        model(mx.array([[10]], dtype=verify.dtype), cache=ref_cache)
        model(mx.array([[11]], dtype=verify.dtype), cache=ref_cache)
        out_ref = model(cont, cache=ref_cache)

        assert [c.offset for c in spec_cache] == [c.offset for c in ref_cache]
        assert mx.allclose(out_spec, out_ref, rtol=1e-5, atol=1e-5).item()

    def test_rollback_refuses_untrimmable_layer_without_mutation(self):
        """Rotated layer with no armed undo → refuse (caller falls back),
        and no partial rollback may desync layer offsets."""
        model = _load_small_model(mtp_active=True)
        cache = model.make_cache()
        model(mx.arange(6).reshape(1, 6), cache=cache)
        # Verify forward WITHOUT arming the undo stash (stock semantics).
        model(mx.array([[10, 11, 12, 13]]), cache=cache)
        assert model.mtp_partial_rollback(cache, accepted=1, num_drafts=3) is False
        assert [c.offset for c in cache] == [10, 10]
