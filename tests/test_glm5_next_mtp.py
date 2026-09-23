# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.patches.mlx_vlm_mtp.glm5_next_vlm_runtime.

Covers nextn key matching, MTP block structure, the cache pair the head
needs, and two sanitize paths: a raw checkpoint whose head lives at
``layers.<num_hidden_layers>.*``, and a checkpoint this patch already
converted whose head is named ``mtp.*``. No model checkpoint is loaded.
Checkpoint-key tests keep the 45-layer layout; execution tests use eight layers.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest

pytest.importorskip("mlx_vlm.models.deepseek_v4")

from omlx.patches.mlx_vlm_glm5_next_compat import (
    apply_mlx_vlm_glm5_next_compat_patch,
)

if not apply_mlx_vlm_glm5_next_compat_patch():
    pytest.importorskip("mlx_vlm.models.glm5_next")

import copy
from types import MethodType, SimpleNamespace

from mlx.utils import tree_flatten
from mlx_vlm.models.glm5_next import language

from omlx.patches.mlx_lm_mtp import batch_generator as bg
from omlx.patches.mlx_lm_mtp import batched_head
from omlx.patches.mlx_lm_mtp import prompt_priming as pp
from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime  # noqa: E402
from omlx.patches.mlx_vlm_mtp.glm5_next_batch_rollback import rollback_rows

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
        PFX + "self_attn.indexer.index_kpool_compress_gate": _zeros(ihd, h),
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


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("prefix", glm5_next_vlm_runtime._NEXTN_PREFIXES)
def test_sanitize_loads_quantized_nextn_fusion(applied, config, bits, prefix):
    """Quantized nextn fusion loads and survives a converted-head reload."""
    source = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
    source = nn.QuantizedLinear.from_linear(source, group_size=64, bits=bits)
    weights = _raw_nextn_weights(config)
    weights.pop(PFX + "eh_proj.weight")
    for name, value in source.parameters().items():
        weights[PFX + "eh_proj." + name] = value
    source_prefix = prefix.format(i=N_MAIN)
    weights = {
        source_prefix + key[len(PFX):] if key.startswith(PFX) else key: value
        for key, value in weights.items()
    }
    sanitized = applied.LanguageModel.sanitize(_Host(config), weights)
    block = applied.Glm5NextMTPBlock(config)
    block.eh_proj = nn.QuantizedLinear.from_linear(
        block.eh_proj, group_size=64, bits=bits
    )
    inputs = mx.ones((1, 2, 2 * config.hidden_size))
    for checkpoint in (
        sanitized,
        applied.LanguageModel.sanitize(_Host(config), sanitized),
    ):
        block.load_weights(
            [(key[len("mtp.0."):], value)
             for key, value in checkpoint.items() if key.startswith("mtp.0.")],
            strict=True,
        )
        assert mx.array_equal(block.eh_proj(inputs), source(inputs)).item()


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

    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None,
                 mask=None):
        seen.append("replayed")
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    c = _Recurrent(size=2)

    host = _Host(config)
    applied.LanguageModel.rollback_speculative_cache(host, [c], _gdn_capture(), 0, 4)

    assert seen == ["replayed"], "SizedArraysCache must take the recurrent path"


def test_rollback_replays_with_the_gate_mask(applied, config, monkeypatch):
    """The replay must run the kernel under the mask the verify forward used.

    Without it the rebuilt recurrent state is not the state an n-token
    forward would have produced on a right-padded batch.
    """
    import mlx_lm.models.cache as cache_mod

    seen_masks = []

    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None,
                 mask=None):
        seen_masks.append(mask)
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    c = cache_mod.ArraysCache(size=2)
    gate_mask = mx.ones((1, 4), dtype=mx.bool_)

    host = _Host(config)
    applied.LanguageModel.rollback_speculative_cache(
        host, [c], _gdn_capture(gate_mask), 1, 4
    )

    assert len(seen_masks) == 1
    assert seen_masks[0] is not None, "the gate mask must reach the replay"
    assert seen_masks[0].shape == (1, 2), (
        f"mask must be sliced to the accepted prefix, got {seen_masks[0].shape}"
    )


def test_rollback_rewinds_the_recurrent_position(applied, config, monkeypatch):
    """These caches carry no offset: their position is lengths/left_padding,
    which the verify forward decremented with cache.advance(S). The rewind is
    the inverse advance, or the recurrent layers sit ahead of the KV layers by
    the rejected count on every partial accept.
    """
    import mlx_lm.models.cache as cache_mod

    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None,
                 mask=None):
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    c = cache_mod.ArraysCache(size=2)
    assert getattr(c, "offset", None) is None, (
        "a production recurrent cache has no offset attribute"
    )
    c.lengths = mx.array([0])  # advance(4) over the verify block

    host = _Host(config)
    applied.LanguageModel.rollback_speculative_cache(host, [c], _gdn_capture(), 0, 4)

    assert int(c.lengths[0]) == 3, (
        f"position must rewind by the rejected count, got lengths={c.lengths}"
    )


def test_rollback_refuses_before_trimming_when_a_layer_cannot_undo(
    applied, config, monkeypatch
):
    """A trim that fails after earlier layers trimmed leaves the layers at
    mixed lengths, so the whole rollback is declined before anything moves.
    """
    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None,
                 mask=None):
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    ok, blocked = _FakeSparse(can_undo=True), _FakeSparse(can_undo=False)

    host = _Host(config)
    with pytest.raises(RuntimeError):
        applied.LanguageModel.rollback_speculative_cache(
            host, [ok, blocked], _gdn_capture(), 0, 4
        )

    assert ok.trimmed == [], "no layer may be trimmed once one of them refuses"


def test_clamp_accept_lowers_the_accept_until_the_undo_fits(applied, config):
    """PoolingCache can only rebuild a confirmed prefix that stays inside its
    buffer, so accepting fewer drafts (a longer rejected tail) is what makes
    the rollback possible. The generator calls this before the rollback.
    """
    host = _Host(config)
    clamp = applied.LanguageModel.mtp_clamp_accept

    # Undo works from 2 rejected rows up, so accepted=2 of 3 drafts (1 row)
    # has to come down to 1 (2 rows).
    cache = [_FakeSparse(can_undo=lambda n: n >= 2)]
    assert clamp(host, cache, 2, 3) == 1

    # Nothing to undo at a full accept.
    assert clamp(host, cache, 3, 3) == 3


def test_clamp_accept_ignores_recurrent_caches(applied, config):
    """Recurrent layers are replayed, not trimmed, so they never constrain it."""
    import mlx_lm.models.cache as cache_mod

    host = _Host(config)
    cache = [cache_mod.ArraysCache(size=2)]
    assert applied.LanguageModel.mtp_clamp_accept(host, cache, 2, 3) == 2


def test_chain_depth_stays_inside_the_pooling_undo_window(applied, config):
    """PoolingCache stashes its undo log only for updates of 8 rows or fewer,
    and a depth-k chain verifies k+1 rows.
    """
    from omlx.patches import mlx_lm_mtp

    prev_depth, prev_active = mlx_lm_mtp.get_mtp_depth(), mlx_lm_mtp.is_mtp_active()
    try:
        mlx_lm_mtp.set_mtp_depth(8)
        mlx_lm_mtp.set_mtp_active(True)
        model = applied.LanguageModel(config)
    finally:
        mlx_lm_mtp.set_mtp_depth(prev_depth)
        mlx_lm_mtp.set_mtp_active(prev_active)

    assert model._omlx_mtp_depth == 7


def test_head_cache_is_committed_only(applied, config):
    """The chain must run its speculative head steps on a per-cycle clone.

    The head pairs each block with a PoolingCache, whose ``offset`` counts
    pooled rows and not tokens, so ``_mtp_head_trim_to`` cannot rewind it:
    with head_clone False the paired KVCache rewinds every cycle while the
    indexer pool keeps the rejected draft rows.
    """
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.mlx_lm_mtp.batch_generator import _resolve_mtp_chain_depth

    prev_active = mlx_lm_mtp.is_mtp_active()
    try:
        mlx_lm_mtp.set_mtp_active(True)
        model = applied.LanguageModel(config)
    finally:
        mlx_lm_mtp.set_mtp_active(prev_active)

    assert model._omlx_mtp_head_clone is True
    assert _resolve_mtp_chain_depth(model)[2] is True

    head_cache = model.make_mtp_cache()
    pools = [c for c in head_cache if getattr(c, "ratio", None) is not None]
    assert pools, "the head cache is expected to hold the indexer pool"
    for pool in pools:
        # `offset` is the pooled row count; _mtp_head_trim_to would compare it
        # against a token history offset. Guard the premise of the fix.
        assert pool.offset == 0
        assert hasattr(pool, "remainder")


def test_priming_captures_ordinary_text_but_not_verify_or_embedded_inputs(
    applied, config, monkeypatch
):
    from types import SimpleNamespace

    from omlx.patches.mlx_lm_mtp import prompt_priming

    hidden = mx.ones((1, 2, config.hidden_size))
    captures = []

    class Trunk:
        def __call__(self, inputs, cache=None, inputs_embeds=None, **kwargs):
            if kwargs.get("hidden_sink") is not None:
                kwargs["hidden_sink"].append(hidden)
            return hidden

    host = SimpleNamespace(
        args=config,
        model=Trunk(),
        lm_head=lambda value: value,
        mtp=object(),
        _omlx_mtp_decode_enabled=True,
        _omlx_mtp_chain=True,
    )
    monkeypatch.setattr(
        prompt_priming, "maybe_capture", lambda *args: captures.append(args)
    )
    inputs = mx.array([[1, 2]])
    cache = [object()]
    applied.LanguageModel.__call__(host, inputs, cache=cache)
    applied.LanguageModel.__call__(host, inputs, cache=cache, return_hidden=True)
    applied.LanguageModel.__call__(host, inputs, cache=cache, inputs_embeds=hidden)
    assert len(captures) == 1
    assert captures[0][0] is host and captures[0][1] is inputs
    assert captures[0][2] is hidden and captures[0][3] is cache


def _gdn_capture(gate_mask=None, block=4, K=4):
    """One layer's capture tuple, shaped as the verify forward records it."""
    return [(mx.zeros((1, block, 1, 1)),) * 5 + (
        mx.zeros((1, 1)), mx.zeros((1, 1)), None,
        mx.zeros((1, block + K - 1, 2)), K, 0.0, gate_mask,
    )]


class _FakeSparse:
    """Sparse-layer stand-in with PoolingCache's undo surface."""

    def __init__(self, can_undo=True):
        self._can_undo_fn = can_undo if callable(can_undo) else (lambda n: can_undo)
        self.remainder = 0
        self.trimmed = []

    def _can_undo(self, n):
        return self._can_undo_fn(n)

    def trim(self, n):
        self.trimmed.append(n)
        return n


_RUNTIME_MODULES = (
    "qwen35_vlm_runtime",
    "qwen35_moe_vlm_runtime",
    "gemma4_vlm_runtime",
    "inkling_vlm_runtime",
    "glm5_next_vlm_runtime",
)


def _record_runtime_applies(monkeypatch):
    import importlib

    applied = []
    for name in _RUNTIME_MODULES:
        module = importlib.import_module(f"omlx.patches.mlx_vlm_mtp.{name}")
        monkeypatch.setattr(
            module, "apply", lambda n=name: (applied.append(n), True)[1]
        )
    return applied


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("glm5_next", ["glm5_next_vlm_runtime"]),
        ("qwen3_5", ["qwen35_vlm_runtime"]),
        ("qwen3_5_moe", ["qwen35_moe_vlm_runtime", "qwen35_vlm_runtime"]),
        # qwen4_exp runs its own MTP and subclasses the qwen3_5 classes these
        # patches rebind, so the sweep must not reach it.
        ("qwen4_exp", []),
        # _is_mtp_compatible admits the family by prefix, so a variant has to
        # resolve to its family rather than falling back to the full sweep.
        ("qwen3_5_moe_variant", ["qwen35_moe_vlm_runtime", "qwen35_vlm_runtime"]),
    ],
)
def test_runtime_sweep_applies_only_the_loaded_architecture(
    monkeypatch, model_type, expected
):
    """Loading one MTP-capable model used to rewrite every other resident
    model's trunk, because these patches rebind methods on shared parent
    classes. A qwen4_exp model resident when a glm5_next model loaded then
    failed on its next request until the server restarted.
    """
    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_runtime_patch

    applied = _record_runtime_applies(monkeypatch)
    apply_mlx_vlm_mtp_runtime_patch(model_type)

    assert applied == expected


def test_runtime_sweep_applies_everything_for_an_unknown_model_type(monkeypatch):
    """The historical behaviour is kept for architectures not in the table."""
    from omlx.patches.mlx_vlm_mtp import apply_mlx_vlm_mtp_runtime_patch

    applied = _record_runtime_applies(monkeypatch)
    apply_mlx_vlm_mtp_runtime_patch(None)

    assert sorted(applied) == sorted(_RUNTIME_MODULES)


def test_rollback_refuses_when_the_capture_is_short(applied, config, monkeypatch):
    """A sink shorter than the recurrent layer count means the capture patch
    never ran. Rolling back part of the cache and reporting success is worse
    than declining, because the caller cannot tell the difference.
    """
    import mlx_lm.models.cache as cache_mod

    def fake_gdu(q, k, v, a, b, A_log, dt_bias, state=None, lower_bound=None,
                 mask=None):
        return None, mx.zeros((1, 1, 1, 1))

    monkeypatch.setattr(applied, "gated_delta_update", fake_gdu, raising=False)

    sparse = _FakeSparse(can_undo=True)
    recurrent = cache_mod.ArraysCache(size=2)

    host = _Host(config)
    with pytest.raises(RuntimeError):
        applied.LanguageModel.rollback_speculative_cache(
            host, [sparse, recurrent], [], 0, 4
        )

    assert sparse.trimmed == [], "no layer may be trimmed once the capture is short"


@pytest.mark.parametrize("layer_idx", [0, 3])
@pytest.mark.parametrize("width", [1, 2, 4, 8])
def test_verify_compiled_ffn_matches_eager(applied, config, layer_idx, width):
    layer = applied.Glm5NextDecoderLayer(config, layer_idx)
    layer.eval()
    x = mx.random.normal((1, width, config.hc_mult, config.hidden_size))
    layer.compile_ffn = False
    eager_sink = []
    expected = layer(x, gdn_sink=eager_sink)
    mx.eval(expected)

    layer.compile_ffn = True
    compiled_sink = []
    actual = layer(x, gdn_sink=compiled_sink)
    mx.eval(actual)

    assert layer._ffn_c is not None
    assert mx.allclose(actual, expected, atol=1e-5, rtol=1e-5).item()
    assert len(compiled_sink) == len(eager_sink) == int(layer.is_linear)


@pytest.mark.parametrize(
    "batch,width,verify", [(2, 4, True), (1, 9, True), (1, 4, False)]
)
def test_verify_ffn_compilation_stays_bounded(applied, config, batch, width, verify):
    layer = applied.Glm5NextDecoderLayer(config, 0)
    layer.eval()
    x = mx.zeros((batch, width, config.hc_mult, config.hidden_size))
    mx.eval(layer(x, gdn_sink=[] if verify else None))
    assert layer._ffn_c is None


def assert_row_states(actual, expected, size):
    for row in range(size):
        for left, right in zip(actual, expected):
            a, b = left.extract(row), right.extract(row)
            aa, bb = dict(tree_flatten(a.state)), dict(tree_flatten(b.state))
            assert aa.keys() == bb.keys()
            for key, value in aa.items():
                if isinstance(value, mx.array):
                    assert isinstance(bb[key], mx.array), (
                        row,
                        type(left).__name__,
                        key,
                        value.shape,
                        bb[key],
                    )
                    assert value.shape == bb[key].shape, (row, type(left).__name__, key)
                    assert mx.array_equal(value, bb[key]).item(), (
                        row,
                        type(left).__name__,
                        key,
                    )
                else:
                    assert not isinstance(bb[key], mx.array), (
                        row,
                        type(left).__name__,
                        key,
                        value,
                        bb[key].shape,
                    )
                    assert value == bb[key], (row, type(left).__name__, key)
            if hasattr(a, "caches"):
                for ac, bc in zip(a.caches, b.caches):
                    assert ac.offset == bc.offset
                    if hasattr(ac, "_processed"):
                        assert ac._processed == bc._processed


def make_host(dtype=mx.float32, *, mtp_layers=0):
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import language
    from mlx_vlm.models.glm5_next.config import TextConfig

    values = copy.deepcopy(TINY_TEXT_CONFIG)
    values["num_hidden_layers"] = 8
    values["num_nextn_predict_layers"] = mtp_layers
    values["layer_types"] = values["layer_types"][:8]
    values["mlp_layer_types"] = values["mlp_layer_types"][:8]
    values["linear_attn_config"]["kda_layers"] = [0, 1, 2, 4, 5, 6]
    values["linear_attn_config"]["full_attn_layers"] = [3, 7]
    host = language.LanguageModel(TextConfig.from_dict(values))
    host.set_dtype(dtype)
    return host


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("depth", [1, 2, 3, 7])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_vector_restore_and_continuation_match_scalar(size, depth, dtype):
    mx.random.seed(349)
    host = make_host(dtype)
    row_caches = []
    for row in range(size):
        cache = host.make_cache()
        tokens = mx.random.randint(0, 512, (1, 5 + row)).astype(mx.uint32)
        out = host(tokens, cache=cache)
        mx.eval(out.logits)
        row_caches.append(cache)
    cache = bg._merge_row_caches(row_caches)
    for cycle in range(3):
        block = mx.random.randint(0, 512, (size, depth + 1)).astype(mx.uint32)
        out = host(block, cache=cache, return_hidden=True)
        mx.eval(out.logits)
        accepted = (
            [depth] * size
            if cycle == 2
            else [(row + cycle * depth) % (depth + 1) for row in range(size)]
        )
        restored_rows = []
        for row, count in enumerate(accepted):
            scalar = copy.deepcopy(cache)
            host.rollback_speculative_cache(scalar, out.gdn_states, count, depth + 1)
            restored_rows.append([layer.extract(row) for layer in scalar])
        expected = bg._merge_row_caches(restored_rows)
        rollback_rows(language, cache, out.gdn_states, accepted, depth + 1)
        assert_row_states(cache, expected, size)
        # Retained KV batches may have more common left padding than a
        # freshly merged reference. Normalize both physical layouts before
        # requiring bitwise equality of subsequent attention reductions.
        cache = bg._merge_row_caches(
            [[layer.extract(row) for layer in cache] for row in range(size)]
        )
        for step in range(4):
            next_tokens = mx.random.randint(0, 512, (size, 1)).astype(mx.uint32)
            x = host(next_tokens, cache=cache).logits
            y = host(next_tokens, cache=expected).logits
            mx.eval(x, y)
            assert mx.array_equal(x, y).item(), (cycle, step, accepted)
        assert_row_states(cache, expected, size)


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_masked_replay_restores_metadata(size, dtype):
    from mlx_vlm.models.cache import ArraysCache
    from mlx_vlm.models.glm5_next import language

    mx.random.seed(884)
    host = make_host(dtype)
    width, heads, dim, kernel = 4, 2, 32, 4
    q, k, v, a = [
        mx.random.normal((size, width, heads, dim)).astype(dtype) for _ in range(4)
    ]
    b = mx.random.normal((size, width, heads)).astype(dtype)
    a_log, bias = mx.zeros((heads, 1)), mx.zeros((heads, dim))
    initial = mx.random.normal((size, heads, dim, dim))
    conv = mx.random.normal((size, width + kernel - 1, dim)).astype(dtype)
    mask = mx.array(
        [
            [True, False, True, True],
            [False, True, True, False],
            [True, True, False, True],
            [True, True, True, True],
        ][:size]
    )
    entry = (q, k, v, a, b, a_log, bias, initial, conv, kernel, None, mask)
    cache = ArraysCache(2)
    cache[0] = conv[:, -kernel + 1 :]
    _, cache[1] = language.gated_delta_update(
        q, k, v, a, b, a_log, bias, state=initial, mask=mask
    )
    cache.lengths = mx.array([8 + row for row in range(size)])
    cache.left_padding = mx.array([-4 - row for row in range(size)])
    counts = [0, 3, 1, 2][:size]
    refs = []
    for row, count in enumerate(counts):
        ref = copy.deepcopy(cache)
        host.rollback_speculative_cache([ref], [entry], count, width)
        refs.append(ref)
    rollback_rows(language, [cache], [entry], counts, width)
    for row, ref in enumerate(refs):
        for index in (0, 1):
            assert mx.array_equal(cache[index][row], ref[index][row]).item()
        assert cache.lengths[row].item() == ref.lengths[row].item()
        assert cache.left_padding[row].item() == ref.left_padding[row].item()


def test_invalid_vector_does_not_mutate_any_cache():
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, CacheList

    from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache

    cache = ArraysCache(2)
    cache[0] = mx.zeros((2, 3, 32))
    cache[1] = mx.zeros((2, 2, 32, 32))
    kv, pool = BatchKVCache([0, 0]), BatchPoolingCache(4, [0, 0])
    q = mx.zeros((2, 4, 2, 32))
    entry = (
        q,
        q,
        q,
        q,
        mx.zeros((2, 4, 2)),
        mx.zeros((2, 1)),
        mx.zeros((2, 32)),
        cache[1],
        mx.zeros((2, 7, 32)),
        4,
        None,
        None,
    )
    before = tuple(cache.cache)
    with pytest.raises(ValueError, match="Pooling cache cannot"):
        rollback_rows(language, [cache, CacheList(kv, pool)], [entry], [0, 3], 4)
    assert all(left is right for left, right in zip(before, cache.cache))
    assert kv.keys is None and kv._idx == 0


def test_pool_vector_preserves_scalar_cross_row_replay_decision():
    from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache

    mx.random.seed(173)
    pool = BatchPoolingCache(4, [0, 0])
    pool.buf_kv = mx.random.normal((2, 4, 32))
    pool.buf_gate = mx.random.normal((2, 4, 32))
    pool.remainder = [0, 2]
    pool._pool_lengths = [2, 2]
    pool._processed = [8, 10]
    pool.pooled = mx.random.normal((2, 2, 32))
    kv, gate = mx.random.normal((2, 4, 32)), mx.random.normal((2, 4, 32))
    pool.accumulate_windows(kv, gate, 0)
    pool.pooled = mx.random.normal((2, 3, 32))
    pool._pool_lengths = [3, 3]
    scalar = copy.deepcopy(pool)
    assert scalar.trim(2) == 2
    expected = scalar.extract(1).state
    pool.trim_rows([0, 2])
    actual = pool.extract(1).state
    assert actual[3] is not None and expected[3] is not None
    for value, other in zip(actual, expected):
        if value is None:
            assert other is None
        else:
            assert mx.array_equal(value, other).item()


def coupled_sampler(index):
    key = mx.random.key(190 + index)

    def sampler(lp):
        return mx.random.categorical(bg._accept_lp_for(sampler, lp), key=key)

    sampler.temp = 0.7
    sampler.top_p = 0.9
    return sampler


@pytest.mark.parametrize("size", [2, 3, 4])
@pytest.mark.parametrize("stochastic", [False, True])
@pytest.mark.parametrize("quantized", [False, True])
def test_history_and_clone_match_independent_heads(size, stochastic, quantized):
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import language
    from mlx_vlm.models.glm5_next.config import TextConfig

    mx.random.seed(918)
    config = TextConfig.from_dict(dict(TINY_TEXT_CONFIG))
    host = SimpleNamespace(
        args=config,
        mtp=[language.Glm5NextMTPBlock(config)],
        model=SimpleNamespace(
            embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size),
            norm=nn.RMSNorm(config.hidden_size),
        ),
        lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        _omlx_mtp_batch_rollback=True,
    )
    host.mtp_forward = MethodType(language.LanguageModel.mtp_forward, host)
    host.make_mtp_cache = MethodType(language.LanguageModel.make_mtp_cache, host)
    if quantized:
        # Tiny projection dimensions require g32; the real head probe uses g64.
        nn.quantize(
            host.mtp[0],
            bits=8,
            group_size=32,
            class_predicate=lambda path, module: hasattr(module, "to_quantized")
            and not path.endswith("mlp.gate"),
        )
    model = SimpleNamespace(_language_model=host, mtp_forward=host.mtp_forward)
    states = [
        bg._MtpState(uid=i, mtp_cache=host.make_mtp_cache(), head_clone=True)
        for i in range(size)
    ]
    refs = [
        bg._MtpState(uid=i, mtp_cache=host.make_mtp_cache(), head_clone=True)
        for i in range(size)
    ]
    rows = [
        SimpleNamespace(
            model=model, samplers=[lambda lp: mx.argmax(lp, -1)], logits_processors=None
        )
        for _ in states
    ]
    if stochastic:
        for index, (state, ref, row) in enumerate(zip(states, refs, rows)):
            sampler = coupled_sampler(index)
            state.draft_sampler = ref.draft_sampler = sampler
            row.samplers = [sampler]
    owner = bg._MtpBatchState(states=dict(enumerate(states)))
    batch = SimpleNamespace(model=model, _omlx_mtp_batch_state=owner)
    for cycle, depth in enumerate([2, 4, 1, 3]):
        jobs = []
        for index, (state, ref, row) in enumerate(zip(states, refs, rows)):
            state.depth = ref.depth = depth
            length = (cycle + index) % 3 + 1
            hidden = mx.random.normal((1, length, config.hidden_size))
            tokens = mx.random.randint(0, config.vocab_size, (length,)).astype(
                mx.uint32
            )
            bg._chain_next_drafts(row, ref, hidden, tokens, None)
            jobs.append((row, state, hidden, tokens, None))
        assert batched_head.eligible(
            batch, [(index, row, state) for index, (row, state, *_) in enumerate(jobs)]
        )
        batched_head.draft(batch, jobs)
        assert owner.head.speculative == 0
        for index, (state, ref) in enumerate(zip(states, refs)):
            assert state.hist_offset == ref.hist_offset
            assert mx.array_equal(state.drafts, ref.drafts).item()
            for actual, expected in zip(state.draft_accept_lps, ref.draft_accept_lps):
                assert mx.allclose(actual, expected, atol=3e-4, rtol=3e-4).item()
            for layer, reference in zip(owner.head.cache, ref.mtp_cache):
                actual = layer.extract(index)
                assert actual.offset == reference.offset
                from omlx.cache.type_registry import CacheTypeRegistry

                handler = CacheTypeRegistry.get_handler_for_object(actual)
                actual_state = dict(tree_flatten(handler.serialize_state(actual)))
                reference_state = dict(tree_flatten(handler.serialize_state(reference)))
                for key, value in actual_state.items():
                    expected = reference_state[key]
                    if isinstance(value, mx.array):
                        assert value.shape == expected.shape
                        assert mx.allclose(value, expected, atol=3e-4, rtol=3e-4).item()
        if cycle == 1:
            # Exercise extraction and re-merging after ownership changes.
            batched_head.flush(owner)
    batched_head.flush(owner)
    assert owner.head is None
    assert all(s.mtp_cache is not None for s in states)


@pytest.mark.parametrize("chunk", [1, 7, 512])
@pytest.mark.parametrize("history_length", [3, 17])
def test_deferred_glm_head_continuation(chunk, history_length):
    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import language
    from mlx_vlm.models.glm5_next.config import TextConfig

    mx.random.seed(12087)
    config = TextConfig.from_dict(dict(TINY_TEXT_CONFIG))
    host = SimpleNamespace(
        args=config,
        mtp=[language.Glm5NextMTPBlock(config)],
        model=SimpleNamespace(
            embed_tokens=nn.Embedding(config.vocab_size, config.hidden_size)
        ),
        lm_head=nn.Linear(config.hidden_size, config.vocab_size, bias=False),
    )
    host.mtp_forward = MethodType(language.LanguageModel.mtp_forward, host)
    host.make_mtp_cache = MethodType(language.LanguageModel.make_mtp_cache, host)
    cache = host.make_mtp_cache()
    hidden = mx.random.normal((1, history_length + 20, config.hidden_size))
    tokens = mx.random.randint(0, config.vocab_size, (1, history_length + 20))
    host.mtp_forward(hidden[:, :history_length], tokens[:, :history_length], cache)
    reference = copy.deepcopy(cache)
    ctx = pp._PrimeCtx(
        mtp_cache=cache, folded=history_length, expected_offset=history_length
    )
    ctx.deferred_pairs = []
    setattr(host, pp._CTX_ATTR, ctx)
    # The handoff token already ends the last committed head pair.
    # Capture its hidden without replaying that pair a second time.
    for position in range(history_length, history_length + 17):
        pp._capture_deferred_history(
            host,
            tokens[:, position : position + 1],
            hidden[:, position : position + 1],
            [SimpleNamespace(offset=position + 1)],
        )
        if position > history_length:
            host.mtp_forward(
                hidden[:, position - 1 : position],
                tokens[:, position : position + 1],
                reference,
            )
    pp._flush_deferred_history(host, ctx, chunk_size=chunk)
    assert ctx.folded == history_length + 16
    for layer, expected in zip(ctx.mtp_cache, reference):
        assert layer.offset == expected.offset
    for position in range(history_length + 17, history_length + 20):
        args = hidden[:, position - 1 : position], tokens[:, position : position + 1]
        actual = host.mtp_forward(*args, ctx.mtp_cache)
        expected = host.mtp_forward(*args, reference)
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=3e-4, rtol=3e-4).item()
