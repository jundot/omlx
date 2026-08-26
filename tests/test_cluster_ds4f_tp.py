# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash tensor parallelism: planner divisors, KV sizing, shard order.

DS4F keeps a single-head latent/pooled KV cache per layer that ``shard()``
never splits, so the (single) KV head count must not bound the TP degree, the
reservation must not be divided across ranks, and the per-head attention sinks
must follow wq_b's segment-interleaved sharding rather than a contiguous split.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    _deepseek_v4_compress_ratios,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_per_layer,
    _kv_cache_replicated_across_tp,
    _kv_fixed_bytes_per_layer,
    _max_context_for_stage,
    _tensor_parallel_divisors,
    inspect_safetensors_layout,
    plan_hybrid,
)

GIB = 1024**3

# DeepSeek-V4-Flash shaped: 64 attention heads over a single latent KV head,
# 8 output-projection groups, one compress ratio per layer.
DS4F_CONFIG = {
    "model_type": "deepseek_v4",
    "num_hidden_layers": 4,
    "compress_ratios": [0, 4, 128, 4],
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "hidden_size": 4096,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "sliding_window": 128,
    "o_groups": 8,
    "o_lora_rank": 1024,
    "q_lora_rank": 1536,
}


def _nodes(count, capacity_gib=64):
    return [
        NodeBudget(
            node_id=f"n{i}",
            capacity_bytes=capacity_gib * GIB,
            reserve_bytes=2 * GIB,
            rank=i,
        )
        for i in range(count)
    ]


# --- (a) TP eligibility: replicated KV, kv_heads excluded from divisors -----


def test_ds4f_is_kv_replicated_and_kv_heads_do_not_bound_tp():
    assert _kv_cache_replicated_across_tp(DS4F_CONFIG) is True
    divisors = _tensor_parallel_divisors(DS4F_CONFIG)
    # The single KV head must not kill every TP size.
    assert 1 not in divisors
    assert all(d % 2 == 0 for d in divisors)  # TP=2 valid
    # Heads, and heads per o_group (wq_b shards segment-interleaved).
    assert 64 in divisors
    assert 8 in divisors


def test_ds4f_variant_model_types_match_the_prefix():
    config = dict(DS4F_CONFIG, model_type="deepseek_v4_flash")
    assert _kv_cache_replicated_across_tp(config) is True


def test_ds4f_kv_rates_match_the_pooled_cache_shapes():
    # Ratio-4 layers: 512/4 latent + 128/4 indexer elements per token;
    # ratio-128 layers: 512/128; ratio-0 layers: window only (fixed term).
    elements = 2 * (512 // 4 + 128 // 4) + 512 // 128  # layers [0, 4, 128, 4]
    expected = -(-elements * 2 // 4)  # ceil, averaged per layer, fp16
    assert _kv_bytes_per_token_per_layer(DS4F_CONFIG) == expected == 162
    # The standard MHA formula would charge 1 * 512 * 2 * 2 — ~13x more.
    assert 2048 // expected >= 12
    # The sliding-window local cache is a fixed per-layer reservation.
    assert _kv_fixed_bytes_per_layer(DS4F_CONFIG) == 128 * 512 * 2


# --- (b) invalid compress_ratios -> not replicated --------------------------


@pytest.mark.parametrize(
    "ratios",
    (
        [0, 4, 7, 4],  # unsupported ratio
        [0, 4, 128],  # shorter than num_hidden_layers
        "0,4,128,4",  # not a sequence of ints
        None,  # absent entirely
    ),
    ids=("bad-ratio", "too-short", "string", "missing"),
)
def test_invalid_compress_ratios_are_not_treated_as_replicated(ratios):
    config = dict(DS4F_CONFIG)
    if ratios is None:
        del config["compress_ratios"]
    else:
        config["compress_ratios"] = ratios

    assert _deepseek_v4_compress_ratios(config) is None
    assert _kv_cache_replicated_across_tp(config) is False
    assert _kv_fixed_bytes_per_layer(config) is None
    # kv_heads=1 constrains again, so no TP degree above 1 is valid.
    divisors = _tensor_parallel_divisors(config)
    assert 1 in divisors
    assert any(d % 2 for d in divisors)


def test_non_ds4f_models_keep_the_kv_head_divisor():
    config = {"num_attention_heads": 24, "num_key_value_heads": 4}
    divisors = _tensor_parallel_divisors(config)
    assert set(divisors) == {24, 4}
    assert all(d % 2 == 0 for d in divisors)  # TP=2 still valid
    assert any(d % 8 for d in divisors)  # TP=8 still refused


# --- (c) attn_sink follows wq_b's segment-interleaved head order ------------


@pytest.fixture(scope="module")
def dsv4():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch

    apply_deepseek_v4_patch()
    import mlx_lm.models.deepseek_v4 as module

    return module


class _FakeGroup:
    """Just enough of mx.distributed.Group for shard() weight slicing."""

    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def _tiny_ds4f(dsv4):
    """4 heads / 2 o_groups, so TP=2 interleaves: rank r keeps head r of each
    group — [0, 2] and [1, 3] — not the contiguous halves [0, 1] / [2, 3]."""
    args = dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 4,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 4,
            "compress_ratios": [0, 0],
        }
    )
    model = dsv4.Model(args)
    for layer in model.model.pipeline_layers:
        # Stamp each sink with its head index and each wq_b row with its
        # global row index, so a shard reveals exactly what it kept.
        layer.attn.attn_sink = mx.arange(4, dtype=mx.float32)
        layer.attn.wq_b.weight = mx.arange(16 * 4, dtype=mx.float32).reshape(16, 4)
    return model


def test_attn_sink_sharding_matches_wq_b_head_group_order(dsv4):
    for rank in (0, 1):
        model = _tiny_ds4f(dsv4)
        model.shard(_FakeGroup(rank, 2))
        expected_heads = [rank, rank + 2]
        for layer in model.model.pipeline_layers:
            assert layer.attn.n_heads == 2
            # The sinks this rank kept, in head order.
            assert layer.attn.attn_sink.tolist() == [
                float(head) for head in expected_heads
            ]
            # The heads wq_b kept: head h owns rows h*head_dim.., and the
            # stamped first column carries the global row index.
            global_rows = [
                int(value) // 4 for value in layer.attn.wq_b.weight[:, 0].tolist()
            ]
            assert global_rows == [
                head * 4 + offset
                for head in expected_heads
                for offset in range(4)
            ]


# --- (d) kv_fixed_bytes_per_layer accounting --------------------------------


def _ds4f_layout(**overrides):
    fields = {
        "source": "test",
        "fixed_weight_bytes": GIB,
        "layer_weight_bytes": (GIB,) * 4,
        "tensor_parallel_heads": 64,
        "tensor_parallel_divisors": (64, 8),
        "supports_tensor_parallel": True,
        "kv_bytes_per_token_per_layer": 162,
        "kv_replicated_across_tp": True,
        "kv_fixed_bytes_per_layer": 131072,
    }
    fields.update(overrides)
    return ModelLayout(**fields)


def test_kv_fixed_term_is_reserved_on_top_of_per_token_growth():
    layout = _ds4f_layout()
    assert _kv_bytes_for_stage(layout, 2, 100) == 2 * 131072 + 162 * 2 * 100
    # Replicated: the TP degree divides nothing.
    assert _kv_bytes_for_stage(layout, 2, 100, 2) == 2 * 131072 + 162 * 2 * 100
    # No fixed reservation when planning for zero context.
    assert _kv_bytes_for_stage(layout, 2, 0) == 0


def test_kv_fixed_term_round_trips_and_defaults_to_none():
    layout = _ds4f_layout()
    rebuilt = ModelLayout.from_dict(layout.to_dict())
    assert rebuilt.kv_fixed_bytes_per_layer == 131072

    plain = ModelLayout(source="t", fixed_weight_bytes=0, layer_weight_bytes=(1,))
    assert plain.kv_fixed_bytes_per_layer is None
    assert ModelLayout.from_dict(plain.to_dict()).kv_fixed_bytes_per_layer is None
    # Other models' accounting is untouched.
    assert _kv_bytes_for_stage(plain, 1, 100) == 0


def test_kv_fixed_term_reduces_the_context_a_node_can_hold():
    layout = _ds4f_layout()
    node = _nodes(1)[0]
    layer_count = 4
    with_fixed = _max_context_for_stage(
        layout, node, layer_count=layer_count, weight_bytes=GIB
    )
    without_fixed = _max_context_for_stage(
        replace(layout, kv_fixed_bytes_per_layer=None),
        node,
        layer_count=layer_count,
        weight_bytes=GIB,
    )
    per_token = 162 * layer_count
    assert with_fixed == (node.usable_bytes - GIB - 4 * 131072) // per_token
    assert without_fixed == (node.usable_bytes - GIB) // per_token
    assert with_fixed < without_fixed


def test_ds4f_tp_plan_reserves_the_whole_cache_plus_window_on_every_member():
    layout = _ds4f_layout()
    plan = plan_hybrid(
        layout, _nodes(2, 90), tensor_parallel_size=2, context_tokens=8192
    )
    expected = 162 * 4 * 8192 + 4 * 131072
    assert plan.tensor_parallel_size == 2
    for assignment in plan.assignments:
        assert assignment.tensor_parallel_size == 2
        assert assignment.kv_cache_bytes == expected


# --- (e) catalogue: DS4F reports TP support, tensor strategy on fast links --


def _write_ds4f_model_dir(root):
    """A minimal on-disk DS4F checkpoint: config plus two layers of weights."""
    config = dict(
        DS4F_CONFIG,
        num_hidden_layers=2,
        compress_ratios=[0, 4],
        vocab_size=128,
    )
    (root / "config.json").write_text(json.dumps(config))
    weights = {"model.embed_tokens.weight": mx.zeros((128, 64), dtype=mx.float16)}
    for index in range(2):
        weights[f"model.layers.{index}.attn.wq_b.weight"] = mx.zeros(
            (64, 64), dtype=mx.float16
        )
    mx.save_safetensors(str(root / "model.safetensors"), weights)


def test_catalogue_reports_tensor_parallel_for_ds4f(tmp_path):
    from omlx.cluster.catalogue import assess_model_path

    _write_ds4f_model_dir(tmp_path)
    fit = assess_model_path(tmp_path, _nodes(2, 128))

    assert fit.fits
    assert fit.supports_tensor_parallel is True

    layout = inspect_safetensors_layout(tmp_path)
    assert layout.supports_tensor_parallel is True
    assert layout.kv_replicated_across_tp is True
    assert layout.kv_fixed_bytes_per_layer == 128 * 512 * 2
    assert 1 not in layout.tensor_parallel_divisors


def test_autoconfigure_picks_tensor_for_ds4f_on_a_fast_link(tmp_path):
    from omlx.cluster.autoconfigure import choose_parallelism

    _write_ds4f_model_dir(tmp_path)
    layout = inspect_safetensors_layout(tmp_path)
    transports = [SimpleNamespace(kind="rdma")]
    choice = choose_parallelism(
        layout, _nodes(2, 128), transports=transports, strategy="auto"
    )
    assert choice.tensor_parallel_size == 2

    slow = choose_parallelism(
        layout,
        _nodes(2, 128),
        transports=[SimpleNamespace(kind="ethernet")],
        strategy="auto",
    )
    assert slow.tensor_parallel_size == 1
    assert slow.pipeline_stages == 2


# --- prefill guard: replicated KV is not divided across TP ranks ------------


def test_prefill_guard_keeps_ds4f_kv_whole_under_tp(dsv4):
    from omlx.cluster.prefill_guard import rank_monitor

    args = dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 4,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 4,
            "sliding_window": 8,
            "compress_ratios": [0, 4, 128, 4],
        }
    )
    model = dsv4.Model(args)

    single = rank_monitor(model, layer_count=4, tensor_parallel_size=1)
    sharded = rank_monitor(model, layer_count=4, tensor_parallel_size=2)
    assert single is not None and sharded is not None

    # The pooled-cache growth is priced exactly (not by the 13x MHA formula)
    # and is NOT divided across TP members: every rank holds the whole cache.
    assert single._kv_bytes_per_token_override == 8.0
    assert sharded._kv_bytes_per_token_override == 8.0
    assert sharded._num_kv_heads == single._num_kv_heads == 1
    # The attention transient still shrinks with the shard.
    assert sharded._num_attention_heads == single._num_attention_heads // 2
    # Sliding-window layers stay window-capped rather than full linear caches.
    assert sharded._rotating_layer_specs == ((4, 8),)
    assert sharded.estimate_resident_kv_bytes(1000) == single.estimate_resident_kv_bytes(
        1000
    )
