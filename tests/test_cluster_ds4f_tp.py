# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash tensor parallelism: planner divisors, KV sizing, shard order.

DS4F keeps a single-head latent/pooled KV cache per layer that ``shard()``
never splits, so the (single) KV head count must not bound the TP degree, the
reservation must not be divided across ranks, and the per-head attention sinks
must follow wq_b's segment-interleaved sharding rather than a contiguous split.
"""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from omlx.cluster.performance import NodePerformanceProfile
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    _deepseek_v4_compress_ratios,
    _kv_bytes_for_stage,
    _kv_bytes_per_token_per_layer,
    _kv_cache_replicated_across_tp,
    _kv_fixed_bytes_per_layer,
    _max_context_for_stage,
    _tensor_parallel_divisors,
    _tensor_parallel_shard_units,
    _tensor_shard_weights,
    _supports_tensor_parallel,
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
    "moe_intermediate_size": 2048,
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
    assert _tensor_parallel_shard_units(DS4F_CONFIG) == 8
    assert _supports_tensor_parallel(DS4F_CONFIG) is True


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


def _tiny_ds4f(dsv4, *, heads=4, moe_intermediate=4):
    """4 heads / 2 o_groups, so TP=2 interleaves: rank r keeps head r of each
    group — [0, 2] and [1, 3] — not the contiguous halves [0, 1] / [2, 3]."""
    args = dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": moe_intermediate,
            "num_hidden_layers": 2,
            "num_attention_heads": heads,
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
            "compress_ratios": [0, 4],
        }
    )
    model = dsv4.Model(args)
    for layer in model.model.pipeline_layers:
        # Stamp each sink with its head index and each wq_b row with its
        # global row index, so a shard reveals exactly what it kept.
        layer.attn.attn_sink = mx.arange(heads, dtype=mx.float32)
        layer.attn.wq_b.weight = mx.arange(heads * 4 * 4, dtype=mx.float32).reshape(
            heads * 4, 4
        )
    return model


def test_attn_sink_sharding_matches_wq_b_head_group_order(dsv4):
    for rank in (0, 1):
        model = _tiny_ds4f(dsv4)
        group = _FakeGroup(rank, 2)
        model.shard(group)
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
        sparse = model.model.pipeline_layers[1].attn
        assert sparse.indexer.row_sharding_group is group


def test_ds4f_unequal_tp_keeps_paired_head_and_mlp_boundaries(dsv4, monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,1")
    expected = ([0, 1, 2, 4, 5, 6], [3, 7])
    for rank in (0, 1):
        model = _tiny_ds4f(dsv4, heads=8)
        model.shard(_FakeGroup(rank, 2))
        for layer in model.model.pipeline_layers:
            assert layer.attn.n_heads == len(expected[rank])
            assert layer.attn.attn_sink.tolist() == [
                float(head) for head in expected[rank]
            ]
            kept_heads = [
                int(value) // 4
                for value in layer.attn.wq_b.weight[:, 0].tolist()
            ]
            assert kept_heads == [
                head * 4 + offset
                for head in expected[rank]
                for offset in range(4)
            ]
            # Shared and routed MLP pairs use the same 3/1 intermediate split.
            assert layer.ffn.shared_experts.gate_proj.weight.shape[0] == (
                3 if rank == 0 else 1
            )
            assert layer.ffn.shared_experts.down_proj.weight.shape[1] == (
                3 if rank == 0 else 1
            )
            assert layer.ffn.switch_mlp.gate_proj.weight.shape[1] == (
                3 if rank == 0 else 1
            )
            assert layer.ffn.switch_mlp.down_proj.weight.shape[2] == (
                3 if rank == 0 else 1
            )


def test_ds4f_moe_override_keeps_outer_attention_and_shared_split(
    dsv4, monkeypatch
):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,1")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "2,2")
    for rank in (0, 1):
        model = _tiny_ds4f(dsv4, heads=8, moe_intermediate=128)
        model.shard(_FakeGroup(rank, 2))
        for layer in model.model.pipeline_layers:
            # Attention and dense/shared MLPs retain the signed outer 3:1.
            assert layer.attn.n_heads == (6 if rank == 0 else 2)
            assert layer.ffn.shared_experts.gate_proj.weight.shape[0] == (
                96 if rank == 0 else 32
            )
            assert layer.ffn.shared_experts.down_proj.weight.shape[1] == (
                96 if rank == 0 else 32
            )
            # Only routed expert banks move to the explicit equal split.
            assert layer.ffn.switch_mlp.gate_proj.weight.shape[1] == 64
            assert layer.ffn.switch_mlp.up_proj.weight.shape[1] == 64
            assert layer.ffn.switch_mlp.down_proj.weight.shape[2] == 64


def test_ds4f_non_moe_override_keeps_routed_banks_on_equal_outer_plan(
    dsv4, monkeypatch
):
    monkeypatch.delenv("OMLX_TP_SHARD_WEIGHTS", raising=False)
    monkeypatch.delenv("OMLX_TP_MOE_SHARD_WEIGHTS", raising=False)
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "3,1")
    for rank in (0, 1):
        model = _tiny_ds4f(dsv4, heads=8, moe_intermediate=128)
        model.shard(_FakeGroup(rank, 2))
        for layer in model.model.pipeline_layers:
            assert layer.attn.n_heads == (6 if rank == 0 else 2)
            assert layer.ffn.shared_experts.gate_proj.weight.shape[0] == (
                96 if rank == 0 else 32
            )
            assert layer.ffn.shared_experts.down_proj.weight.shape[1] == (
                96 if rank == 0 else 32
            )
            # The signed outer plan is equal, so routed banks remain 2:2.
            assert layer.ffn.switch_mlp.gate_proj.weight.shape[1] == 64
            assert layer.ffn.switch_mlp.up_proj.weight.shape[1] == 64
            assert layer.ffn.switch_mlp.down_proj.weight.shape[2] == 64


def test_ds4f_moe_override_reconstructs_routed_banks_exactly(dsv4, monkeypatch):
    monkeypatch.setenv("OMLX_TP_SHARD_WEIGHTS", "3,1")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "2,2")
    mx.random.seed(101)
    reference = _tiny_ds4f(dsv4, heads=8, moe_intermediate=128)
    ref = reference.model.pipeline_layers[0].ffn.switch_mlp
    gate_shards = []
    up_shards = []
    down_shards = []
    for rank in (0, 1):
        mx.random.seed(101)
        model = _tiny_ds4f(dsv4, heads=8, moe_intermediate=128)
        model.shard(_FakeGroup(rank, 2))
        routed = model.model.pipeline_layers[0].ffn.switch_mlp
        gate_shards.append(routed.gate_proj.weight)
        up_shards.append(routed.up_proj.weight)
        down_shards.append(routed.down_proj.weight)

    rebuilt_gate = mx.concatenate(gate_shards, axis=1)
    rebuilt_up = mx.concatenate(up_shards, axis=1)
    rebuilt_down = mx.concatenate(down_shards, axis=2)
    mx.eval(rebuilt_gate, rebuilt_up, rebuilt_down)
    assert mx.array_equal(rebuilt_gate, ref.gate_proj.weight).item()
    assert mx.array_equal(rebuilt_up, ref.up_proj.weight).item()
    assert mx.array_equal(rebuilt_down, ref.down_proj.weight).item()


@pytest.mark.parametrize(
    "outer, override, intermediate, message",
    (
        (None, "2,2", 128, "requires either a signed unequal outer"),
        ((3, 1), "4", 128, "one positive weight per TP rank"),
        ((3, 1), "2,0", 128, "one positive weight per TP rank"),
        ((3, 1), "3,3", 128, "must sum"),
        ((3, 1), "2,2", 130, "not divisible"),
        ((3, 1), "1,3", 96, "32-value MXFP4"),
    ),
)
def test_ds4f_moe_override_fails_before_sharding(
    dsv4, monkeypatch, outer, override, intermediate, message
):
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", override)
    args = SimpleNamespace(moe_intermediate_size=intermediate)
    with pytest.raises(ValueError, match=message):
        dsv4._validated_ds4_moe_tp_weights(args, _FakeGroup(0, 2), outer)


def test_ds4f_moe_override_default_reuses_outer_weights(dsv4, monkeypatch):
    monkeypatch.delenv("OMLX_TP_MOE_SHARD_WEIGHTS", raising=False)
    args = SimpleNamespace(moe_intermediate_size=2048)
    assert dsv4._validated_ds4_moe_tp_weights(
        args, _FakeGroup(0, 2), (3, 5)
    ) == (3, 5)


def test_real_ds4f_three_five_outer_accepts_equal_moe_banks(dsv4, monkeypatch):
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "4,4")
    args = SimpleNamespace(moe_intermediate_size=2048)
    assert dsv4._validated_ds4_moe_tp_weights(
        args, _FakeGroup(0, 2), (3, 5)
    ) == (4, 4)


def test_real_ds4f_equal_outer_accepts_three_five_non_moe_split(
    dsv4, monkeypatch
):
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "3,5")
    args = SimpleNamespace(
        num_attention_heads=64,
        o_groups=8,
        moe_intermediate_size=2048,
    )
    assert dsv4._validated_ds4_non_moe_tp_weights(
        args, _FakeGroup(0, 2), None
    ) == (3, 5)


def test_real_ds4f_equal_plan_accepts_explicit_four_four_moe_with_non_moe_split(
    dsv4, monkeypatch
):
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", "3,5")
    monkeypatch.setenv("OMLX_TP_MOE_SHARD_WEIGHTS", "4,4")
    args = SimpleNamespace(
        num_attention_heads=64,
        o_groups=8,
        moe_intermediate_size=2048,
    )
    assert dsv4._validated_ds4_moe_tp_weights(
        args, _FakeGroup(0, 2), None
    ) == (4, 4)


@pytest.mark.parametrize(
    "override, message",
    (
        ("4", "one positive weight per TP rank"),
        ("3,0", "one positive weight per TP rank"),
        ("3,4", "must sum"),
    ),
)
def test_ds4f_non_moe_override_fails_before_sharding(
    dsv4, monkeypatch, override, message
):
    monkeypatch.setenv("OMLX_TP_NON_MOE_SHARD_WEIGHTS", override)
    args = SimpleNamespace(
        num_attention_heads=64,
        o_groups=8,
        moe_intermediate_size=2048,
    )
    with pytest.raises(ValueError, match=message):
        dsv4._validated_ds4_non_moe_tp_weights(
            args, _FakeGroup(0, 2), None
        )


def test_unequal_tp_slices_quantized_values_on_exact_group_boundaries(dsv4):
    column = nn.QuantizedLinear(
        128, 128, bias=False, group_size=32, bits=4
    )
    row = nn.QuantizedLinear(128, 128, bias=False, group_size=32, bits=4)
    expected = ((96, 12, 3), (32, 4, 1))
    for rank in (0, 1):
        group = _FakeGroup(rank, 2)
        column_shard = dsv4._asymmetric_shard_parameters(
            column,
            "all-to-sharded",
            group=group,
            weights=(3, 1),
        )
        row_shard = dsv4._asymmetric_shard_parameters(
            row,
            "sharded-to-all",
            group=group,
            weights=(3, 1),
        )
        output_rows, packed_inputs, quant_groups = expected[rank]
        assert column_shard["weight"].shape[0] == output_rows
        assert column_shard["scales"].shape[0] == output_rows
        assert row_shard["weight"].shape[-1] == packed_inputs
        assert row_shard["scales"].shape[-1] == quant_groups


def test_unequal_tp_paired_mlp_sum_matches_the_unsharded_result(dsv4):
    mx.random.seed(7)
    gate = nn.Linear(8, 8, bias=False)
    up = nn.Linear(8, 8, bias=False)
    down = nn.Linear(8, 8, bias=False)
    x = mx.random.normal((3, 8))
    full_gate = gate(x)
    full = down((full_gate * mx.sigmoid(full_gate)) * up(x))
    partials = []
    for rank in (0, 1):
        group = _FakeGroup(rank, 2)
        gate_shard = dsv4._asymmetric_shard_parameters(
            gate, "all-to-sharded", group=group, weights=(3, 1)
        )["weight"]
        up_shard = dsv4._asymmetric_shard_parameters(
            up, "all-to-sharded", group=group, weights=(3, 1)
        )["weight"]
        down_shard = dsv4._asymmetric_shard_parameters(
            down, "sharded-to-all", group=group, weights=(3, 1)
        )["weight"]
        local_gate = x @ gate_shard.T
        local_hidden = (local_gate * mx.sigmoid(local_gate)) * (x @ up_shard.T)
        partials.append(local_hidden @ down_shard.T)
    mx.eval(full, *partials)
    assert mx.allclose(full, partials[0] + partials[1], rtol=1e-5, atol=1e-5).item()


def test_decode_non_owner_skips_duplicate_indexer_scoring(dsv4, monkeypatch):
    group = _FakeGroup(1, 2)
    expected = mx.arange(512, dtype=mx.uint32).reshape(1, 1, 512)
    fake = SimpleNamespace(
        compressor=lambda *_args: mx.zeros((1, 513, 4), dtype=mx.float16),
        index_topk=512,
        row_sharding_group=group,
    )
    monkeypatch.setenv("OMLX_DSV4_INDEXER_DECODE_OWNER_RANK", "0")

    def all_sum(local, *, group):
        assert group is fake.row_sharding_group
        assert not mx.any(local).item()
        return expected.astype(mx.int32)

    monkeypatch.setattr(mx.distributed, "all_sum", all_sum)
    result = dsv4.Indexer.__call__(
        fake,
        mx.zeros((1, 1, 8)),
        mx.zeros((1, 1, 4)),
        None,
        None,
        0,
    )

    assert mx.array_equal(result, expected).item()


@pytest.mark.parametrize("rank", (0, 1), ids=("owner", "receiver"))
def test_decode_indexer_decision_can_use_fixed_control_packet(
    dsv4, monkeypatch, rank
):
    group = _FakeGroup(rank, 2)
    expected = mx.arange(512, dtype=mx.uint32).reshape(1, 1, 512)
    packet = struct.pack("!512I", *range(512))
    calls = []

    class Control:
        world_size = 2

        def __init__(self, control_rank):
            self.rank = control_rank

        def broadcast_owned_bytes(
            self, payload, *, source_rank, expected_size
        ):
            calls.append((payload, source_rank, expected_size))
            assert source_rank == 0
            assert expected_size == len(packet)
            if self.rank == source_rank:
                assert payload == packet
                return payload
            assert payload is None
            return packet

    monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_INDEXER_DECISION_TRANSPORT", "control")
    from omlx.cluster import control_plane

    monkeypatch.setattr(
        control_plane,
        "active_rank_control_plane",
        lambda: Control(rank),
    )
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("control decisions must not enter JACCL all_sum")
        ),
    )

    result = dsv4._broadcast_indexer_indices(
        expected if rank == 0 else None,
        shape=tuple(expected.shape),
        group=group,
        owner=0,
    )

    assert mx.array_equal(result, expected).item()
    assert len(calls) == 1


def test_decode_indexer_control_transport_fails_closed_without_channel(
    dsv4, monkeypatch
):
    monkeypatch.setattr(dsv4, "_DEEPSEEK_V4_INDEXER_DECISION_TRANSPORT", "control")
    from omlx.cluster import control_plane

    monkeypatch.setattr(control_plane, "active_rank_control_plane", lambda: None)
    monkeypatch.setattr(
        mx.distributed,
        "all_sum",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a missing control plane must not fall back mid-schedule")
        ),
    )

    with pytest.raises(RuntimeError, match="active rank-control plane"):
        dsv4._broadcast_indexer_indices(
            None,
            shape=(1, 1, 512),
            group=_FakeGroup(1, 2),
            owner=0,
        )


# --- (d) kv_fixed_bytes_per_layer accounting --------------------------------


def _ds4f_layout(**overrides):
    fields = {
        "source": "test",
        "fixed_weight_bytes": GIB,
        "layer_weight_bytes": (GIB,) * 4,
        "tensor_parallel_heads": 64,
        "tensor_parallel_divisors": (64, 8),
        "tensor_parallel_shard_units": 8,
        "supports_tensor_parallel": True,
        "kv_bytes_per_token_per_layer": 162,
        "kv_replicated_across_tp": True,
        "kv_fixed_bytes_per_layer": 131072,
    }
    fields.update(overrides)
    return ModelLayout(**fields)


def _performance(node_id, rank, rate):
    return NodePerformanceProfile(
        node_id=node_id,
        rank=rank,
        decode_weight_bytes_per_second=rate,
        prefill_weight_bytes_per_second=rate,
        collective_latency_seconds=0.00003,
        collective_bandwidth_bytes_per_second=6.2e9,
        backend="jaccl",
        measured_at="2026-08-22T00:00:00+00:00",
        samples=5,
    )


def test_synthetic_ds4f_tp_candidate_does_not_self_promote():
    layout = _ds4f_layout(layer_weight_bytes=(20 * GIB,) * 4)
    nodes = [
        NodeBudget(
            node_id="m3-ultra",
            capacity_bytes=128 * GIB,
            reserve_bytes=2 * GIB,
            rank=0,
            performance=_performance("m3-ultra", 0, 100e9),
        ),
        NodeBudget(
            node_id="m5-max",
            capacity_bytes=128 * GIB,
            reserve_bytes=2 * GIB,
            rank=1,
            performance=_performance("m5-max", 1, 60e9),
        ),
    ]

    candidate = _tensor_shard_weights(
        layout,
        nodes,
        workload_profile="balanced",
    )
    plan = plan_hybrid(layout, nodes, tensor_parallel_size=2)

    assert candidate == (5, 3)
    assert [item.tensor_parallel_shard_weight for item in plan.assignments] == [4, 4]
    assert [item.layer_weight_bytes // GIB for item in plan.assignments] == [40, 40]
    assert max(item.predicted_stage_seconds for item in plan.assignments) > 0


def test_e2e_qualified_ds4f_tp_can_assign_five_eighths_to_faster_mac():
    layout = _ds4f_layout(layer_weight_bytes=(20 * GIB,) * 4)
    nodes = [
        NodeBudget(
            node_id="m3-ultra",
            capacity_bytes=128 * GIB,
            reserve_bytes=2 * GIB,
            rank=0,
            performance=_performance("m3-ultra", 0, 100e9),
        ),
        NodeBudget(
            node_id="m5-max",
            capacity_bytes=128 * GIB,
            reserve_bytes=2 * GIB,
            rank=1,
            performance=_performance("m5-max", 1, 60e9),
        ),
    ]

    plan = plan_hybrid(
        layout,
        nodes,
        tensor_parallel_size=2,
        qualified_tensor_shard_weights=((5, 3),),
    )

    assert [item.tensor_parallel_shard_weight for item in plan.assignments] == [5, 3]
    assert [item.layer_weight_bytes // GIB for item in plan.assignments] == [50, 30]


@pytest.mark.parametrize(
    "weights, message",
    (
        (((5, 2),), "sum"),
        (((5, 3, 1),), "positive integers"),
        (((5, 0),), "positive integers"),
    ),
)
def test_e2e_qualified_ds4f_tp_weights_fail_closed(weights, message):
    with pytest.raises(PlanningError, match=message):
        plan_hybrid(
            _ds4f_layout(),
            _nodes(2, capacity_gib=128),
            tensor_parallel_size=2,
            qualified_tensor_shard_weights=weights,
        )


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
    assert layout.supports_pipeline is True
    assert layout.kv_replicated_across_tp is True
    assert layout.kv_fixed_bytes_per_layer == 128 * 512 * 2
    assert layout.tensor_parallel_shard_units == 8
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


def test_prefill_guard_preserves_ds4f_exact_profile_under_asymmetric_tp(
    dsv4, monkeypatch
):
    from omlx.cluster.prefill_guard import rank_monitor

    monkeypatch.setattr(
        "omlx.memory_monitor.native_indexer_memory_safe_eligible",
        lambda **_kwargs: True,
    )
    monkeypatch.setattr(
        "omlx.patches.deepseek_v4.wsdpa_attention.wsdpa_prefill_route_active",
        lambda **_kwargs: True,
    )

    args = dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 4,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 512,
            "o_groups": 8,
            "o_lora_rank": 4,
            "index_n_heads": 64,
            "index_head_dim": 128,
            "index_topk": 512,
            "hc_mult": 4,
            "sliding_window": 8,
            "compress_ratios": [0, 4, 128, 4],
            "torch_dtype": "bfloat16",
        }
    )
    model = dsv4.Model(args)
    model.dtype = mx.bfloat16

    rank = rank_monitor(
        model,
        layer_count=4,
        tensor_parallel_size=2,
        tensor_parallel_shard_weight=5,
        tensor_parallel_shard_weight_total=8,
    )

    assert rank is not None
    assert rank._num_attention_heads == 40
    assert rank._prefill_memory_profile is not None
    assert rank._prefill_memory_profile.num_attention_heads == 40
    peak = int(rank.estimate_prefill_peak_bytes(250_000, 1024))
    assert peak < 4 * 1024**3

    from omlx.cluster.prefill_guard import RankPrefillGuard
    from omlx.exceptions import PrefillMemoryExceededError

    ceiling = 8 * 1024**3
    guard = RankPrefillGuard(
        rank,
        rank=1,
        ceiling_bytes=ceiling,
        prefill_step_size=1024,
    )
    guard.check(250_000, current_usage_bytes=ceiling - peak - 1)
    with pytest.raises(PrefillMemoryExceededError):
        guard.check(250_000, current_usage_bytes=ceiling - peak + 1)


def test_prefill_guard_slices_ds4f_profile_to_a_nonzero_pipeline_stage(dsv4):
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
    stage = rank_monitor(
        dsv4.Model(args),
        start_layer=1,
        layer_count=2,
    )

    assert stage is not None
    profile = stage._prefill_memory_profile
    assert profile is not None
    assert profile.local_layers == 2
    assert profile.ratio4_layers == 1
    assert profile.ratio128_layers == 1
