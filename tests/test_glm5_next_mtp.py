# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_vlm_mtp.glm5_next_vlm_runtime.

Covers nextn key matching, MTP block structure, the cache pair the head
needs, and two sanitize paths: a raw checkpoint whose head lives at
``layers.<num_hidden_layers>.*``, and a checkpoint this patch already
converted whose head is named ``mtp.*``. No weights are loaded; the config
is shrunk so the routed MoE never allocates.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

pytest.importorskip("mlx_vlm.models.deepseek_v4")

from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)

if not apply_mlx_vlm_glm5_next_compat_patch():
    pytest.importorskip("mlx_vlm.models.glm5_next")

from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime  # noqa: E402

N_MAIN = 45
N_MTP = 1

TINY_TEXT_CONFIG = {
    "model_type": "glm5_next_text",
    "hidden_size": 128,
    "num_hidden_layers": N_MAIN,
    "num_nextn_predict_layers": N_MTP,
    "intermediate_size": 256,
    "moe_intermediate_size": 64,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 0,
    "qk_nope_head_dim": 32,
    "qk_rope_head_dim": 0,
    "v_head_dim": 32,
    "kv_lora_rank": 64,
    "q_lora_rank": 64,
    "index_head_dim": 32,
    "index_n_heads": 4,
    "index_topk": 16,
    "index_kpool": 4,
    "index_kpool_compress": True,
    "index_kpool_always_select_tail": True,
    "index_share_for_mtp_iteration": True,
    "indexer_rope_interleave": True,
    "hc_mult": 4,
    "hc_eps": 1e-06,
    "hc_sinkhorn_iters": 20,
    "rms_norm_eps": 1e-05,
    "vocab_size": 512,
    "tie_word_embeddings": False,
    "swiglu_limit": 10.0,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "scoring_func": "sigmoid",
    "attention_bias": False,
    "max_position_embeddings": 4096,
    "layer_types": [
        "deepseek_sparse_attention" if (i % 4) == 3 else "linear_attention"
        for i in range(N_MAIN)
    ],
    "mlp_layer_types": ["dense" if i < 3 else "sparse" for i in range(N_MAIN)],
    "linear_attn_config": {
        "num_heads": 4,
        "head_dim": 32,
        "gate_lower_bound": 0.0,
        "short_conv_kernel_size": 4,
        "kda_layers": [i for i in range(N_MAIN) if (i % 4) != 3],
        "full_attn_layers": [i for i in range(N_MAIN) if (i % 4) == 3],
    },
}

PFX = f"model.language_model.layers.{N_MAIN}."


@pytest.fixture(scope="module")
def applied():
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import language

    return language


@pytest.fixture(scope="module")
def config(applied):
    from mlx_vlm.models.glm5_next.config import TextConfig

    return TextConfig.from_dict(dict(TINY_TEXT_CONFIG))


def _leaf_paths(value, prefix=""):
    if isinstance(value, dict):
        out = []
        for key, sub in value.items():
            out.extend(_leaf_paths(sub, f"{prefix}.{key}" if prefix else key))
        return out
    if isinstance(value, list):
        out = []
        for i, sub in enumerate(value):
            out.extend(_leaf_paths(sub, f"{prefix}.{i}"))
        return out
    return [prefix]


def _zeros(*shape):
    return mx.zeros(shape, dtype=mx.bfloat16)


def _raw_nextn_weights(config):
    """The tensor names a real glm5_next checkpoint ships for its nextn layer."""
    h = config.hidden_size
    heads = config.num_attention_heads
    nope, vhd = config.qk_nope_head_dim, config.v_head_dim
    kvl, ql = config.kv_lora_rank, config.q_lora_rank
    ihd, inh = config.index_head_dim, config.index_n_heads
    experts, moe = config.n_routed_experts, config.moe_intermediate_size

    weights = {
        PFX + "eh_proj.weight": _zeros(h, 2 * h),
        PFX + "enorm.weight": _zeros(h),
        PFX + "hnorm.weight": _zeros(h),
        PFX + "shared_head.norm.weight": _zeros(h),
        PFX + "shared_head.head.weight": _zeros(config.vocab_size, h),
        PFX + "input_layernorm.weight": _zeros(h),
        PFX + "post_attention_layernorm.weight": _zeros(h),
        PFX + "self_attn.q_a_proj.weight": _zeros(ql, h),
        PFX + "self_attn.q_a_layernorm.weight": _zeros(ql),
        PFX + "self_attn.q_b_proj.weight": _zeros(heads * nope, ql),
        PFX + "self_attn.kv_a_proj_with_mqa.weight": _zeros(kvl, h),
        PFX + "self_attn.kv_a_layernorm.weight": _zeros(kvl),
        PFX + "self_attn.kv_b_proj.weight": _zeros(heads * (nope + vhd), kvl),
        PFX + "self_attn.o_proj.weight": _zeros(h, heads * vhd),
        PFX + "self_attn.indexer.wk.weight": _zeros(ihd, h),
        PFX + "self_attn.indexer.wq_b.weight": _zeros(inh * ihd, ql),
        PFX + "self_attn.indexer.weights_proj.weight": _zeros(inh, h),
        PFX + "self_attn.indexer.k_norm.weight": _zeros(ihd),
        PFX + "self_attn.indexer.k_norm.bias": _zeros(ihd),
        PFX + "self_attn.indexer.index_kpool_compress_ape": _zeros(
            config.index_kpool, ihd
        ),
        PFX + "self_attn.indexer.index_kpool_compress_gate": _zeros(
            config.index_kpool, ihd
        ),
        PFX + "mlp.gate.weight": _zeros(experts, h),
        PFX + "mlp.gate.e_score_correction_bias": _zeros(experts),
        "model.language_model.layers.0.input_layernorm.weight": _zeros(h),
    }
    projections = {
        "gate_proj": (moe, h),
        "up_proj": (moe, h),
        "down_proj": (h, moe),
    }
    for expert in range(experts):
        for name, (out_f, in_f) in projections.items():
            weights[PFX + f"mlp.experts.{expert}.{name}.weight"] = _zeros(out_f, in_f)
    for name, (out_f, in_f) in projections.items():
        weights[PFX + f"mlp.shared_experts.{name}.weight"] = _zeros(out_f, in_f)
    return weights


class _Host:
    def __init__(self, config):
        self.args = config


@pytest.mark.parametrize(
    ("key", "expected_suffix"),
    [
        (PFX + "eh_proj.weight", "eh_proj.weight"),
        (PFX + "shared_head.norm.weight", "shared_head.norm.weight"),
        (PFX + "mlp.experts.3.up_proj.weight", "mlp.experts.3.up_proj.weight"),
    ],
)
def test_match_nextn_accepts_head_tensors(key, expected_suffix):
    index, suffix = glm5_next_vlm_runtime._match_nextn(key, N_MAIN, N_MTP)
    assert index == 0
    assert suffix == expected_suffix


@pytest.mark.parametrize(
    "key",
    [
        f"model.language_model.layers.{N_MAIN - 1}.self_attn.o_proj.weight",
        "model.visual.blocks.0.attn.qkv.weight",
    ],
)
def test_match_nextn_leaves_backbone_tensors(key):
    assert glm5_next_vlm_runtime._match_nextn(key, N_MAIN, N_MTP)[0] is None


def test_mtp_block_omits_hyper_connection(applied, config):
    """The nextn layer ships no hc_* tensors, so the block takes plain residuals."""
    block = applied.Glm5NextMTPBlock(config)
    paths = set(_leaf_paths(block.parameters()))
    assert {"enorm.weight", "hnorm.weight", "eh_proj.weight", "norm.weight"} <= paths
    assert any(p.startswith("block.self_attn.q_a_proj") for p in paths)
    assert any(".indexer." in p for p in paths)
    assert not [p for p in paths if "_hc." in p]


def test_make_mtp_cache_pairs_kv_and_pooling(applied, config):
    from mlx_lm.models.cache import KVCache, PoolingCache

    host = _Host(config)
    host.mtp = [applied.Glm5NextMTPBlock(config)]
    caches = applied.LanguageModel.make_mtp_cache(host)
    assert len(caches) == 2
    assert isinstance(caches[0], KVCache)
    assert isinstance(caches[1], PoolingCache)


def test_sanitize_binds_the_nextn_layer_exactly(applied, config):
    """Every tensor the block declares is produced, and nothing else is."""
    block = applied.Glm5NextMTPBlock(config)
    expected = {"mtp.0." + p for p in _leaf_paths(block.parameters())}

    out = applied.LanguageModel.sanitize(_Host(config), _raw_nextn_weights(config))
    produced = {k for k in out if k.startswith("mtp.")}

    assert not expected - produced
    assert not produced - expected
    assert any("embed_q" in k for k in produced)
    assert any("unembed_out" in k for k in produced)
    assert any("switch_mlp" in k for k in produced)
    assert not [k for k in produced if ".mlp.experts." in k]
    assert not [k for k in out if "shared_head.head" in k]
    assert "model.language_model.layers.0.input_layernorm.weight" in out


def test_sanitize_preserves_an_already_converted_head(applied, config):
    """Stock sanitize drops keys containing ``mtp.``; reloading must not."""
    h, experts = config.hidden_size, config.n_routed_experts
    weights = {
        "language_model.model.layers.0.input_layernorm.weight": _zeros(h),
        "language_model.mtp.0.enorm.weight": _zeros(h),
        "language_model.mtp.0.hnorm.weight": _zeros(h),
        "language_model.mtp.0.eh_proj.weight": _zeros(h, 2 * h),
        "language_model.mtp.0.norm.weight": _zeros(h),
        "language_model.mtp.0.block.input_layernorm.weight": _zeros(h),
        "language_model.mtp.0.block.mlp.gate.weight": _zeros(experts, h),
        "language_model.mtp.0.block.mlp.gate.e_score_correction_bias": _zeros(experts),
    }

    out = applied.LanguageModel.sanitize(_Host(config), weights)

    assert len([k for k in out if "mtp." in k]) == 7
    assert "language_model.model.layers.0.input_layernorm.weight" in out
    for key in out:
        if key.endswith(("mlp.gate.weight", "e_score_correction_bias")):
            assert out[key].dtype == mx.float32


def test_rollback_matches_the_recurrent_cache_family(applied, config, monkeypatch):
    """Linear layers use ArraysCache or SizedArraysCache depending on the path.

    Keying the rollback on one class leaves the other holding rejected tokens
    while the KV caches rewind, and the drift compounds every round.
    """
    import mlx_lm.models.cache as cache_mod

    seen = []

    class _Recurrent(cache_mod.ArraysCache):
        pass

    _Recurrent.__name__ = "SizedArraysCache"

    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None):
        seen.append("replayed")
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    c = _Recurrent(size=2)
    c.offset = 4
    K = 4
    gdn = [(mx.zeros((1, 4, 1, 1)),) * 5 + (mx.zeros((1, 1)), mx.zeros((1, 1)),
           None, mx.zeros((1, 4 + K - 1, 2)), K, 0.0)]

    host = _Host(config)
    applied.LanguageModel.rollback_speculative_cache(host, [c], gdn, 0, 4)

    assert seen == ["replayed"], "SizedArraysCache must take the recurrent path"
    assert c.offset == 1, f"offset must rewind by the rejected count, got {c.offset}"
