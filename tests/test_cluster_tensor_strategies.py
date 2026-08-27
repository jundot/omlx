"""Tensor-parallel sharding strategy regressions.

Focus: the Nemotron-H routed-expert MoE, whose quantized down-projection has a
prime number of quant groups (29 at group_size 64 over a 1856-wide
intermediate). An even ``mx.split`` cannot divide 29 across two ranks, so the
strategy slices explicit, possibly-unequal, group ranges. These tests pin the
range arithmetic and the numeric equivalence of the split against an unsharded
forward.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_lm.models.switch_layers import SwitchLinear

from omlx.cluster.tensor_strategies import (
    _gather_vocab_logits,
    _native_layerwise_shard,
    _shard_auxiliary_vocab_heads,
    _shard_output_head,
    _shard_switch_mlp_uneven,
    _uneven_group_ranges,
)


class _FakeGroup:
    def __init__(self, rank, size=2):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_ds4_native_sharding_materializes_only_the_lazy_local_slice(monkeypatch):
    class Layer:
        def __init__(self, name):
            self.value = name + "-full"

        def parameters(self):
            return [self.value]

    class Model:
        model_type = "deepseek_v4"

        def __init__(self):
            self.layers = [Layer("zero"), Layer("one")]

        def shard(self, _group):
            assert len(self.layers) == 1
            layer = self.layers[0]
            layer.value = layer.value.replace("-full", "-shard")

    class FakeMX:
        def __init__(self):
            self.evaluated = []
            self.syncs = 0
            self.clears = 0

        def eval(self, values):
            self.evaluated.extend(values)

        def synchronize(self):
            self.syncs += 1

        def clear_cache(self):
            self.clears += 1

    model = Model()
    fake_mx = FakeMX()
    monkeypatch.delenv("OMLX_TP_LAZY_NATIVE_SHARD", raising=False)
    monkeypatch.setattr(
        "omlx.cluster.tensor_strategies.native_shard_is_layer_local",
        lambda _shard: (True, "test"),
    )

    _native_layerwise_shard(model, _FakeGroup(0), fake_mx, None)

    assert fake_mx.evaluated == ["zero-shard", "one-shard"]
    assert fake_mx.syncs == fake_mx.clears == 2
    assert [layer.value for layer in model.layers] == ["zero-shard", "one-shard"]


def _linear_with_weight(weight):
    layer = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
    layer.weight = mx.array(weight)
    return layer


def test_vocab_parallel_head_slices_exact_output_rows(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    weight = mx.arange(24, dtype=mx.float32).reshape(6, 4)
    inputs = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4)
    expected = inputs @ weight.T
    local_logits = []

    for rank in (0, 1):
        events = []
        model = SimpleNamespace(lm_head=_linear_with_weight(weight))
        assert _shard_output_head(model, _FakeGroup(rank), mx, events.append)
        assert model.lm_head.weight.shape == (3, 4)
        assert model._omlx_vocab_parallel_head is True
        assert model._omlx_output_vocab_size == 6
        assert events == [
            {
                "phase": "tensor_output_head",
                "strategy": "vocab_parallel",
                "rank": rank,
                "ranks": 2,
                "head_bytes": 96,
                "local_head_bytes": 48,
                "vocab_size": 6,
                "local_vocab_size": 3,
            }
        ]
        local_logits.append(model.lm_head._local_logits(inputs))

    reconstructed = mx.concatenate(local_logits, axis=-1)
    assert mx.array_equal(reconstructed, expected).item()


def test_vocab_parallel_dense_head_preserves_linear_runtime_patches(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    original_call = nn.Linear.__call__
    calls = []

    def patched_call(module, value):
        calls.append(module)
        return original_call(module, value)

    monkeypatch.setattr(nn.Linear, "__call__", patched_call)
    model = SimpleNamespace(lm_head=_linear_with_weight(mx.zeros((6, 4))))
    _shard_output_head(model, _FakeGroup(0), mx)
    model.lm_head._local_logits(mx.zeros((1, 4)))

    assert calls == [model.lm_head]
    assert isinstance(model.lm_head, nn.Linear)


def test_vocab_parallel_head_can_expose_local_logits_for_rank_zero_sampling(
    monkeypatch,
):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    weight = mx.arange(24, dtype=mx.float32).reshape(6, 4)
    inputs = mx.arange(8, dtype=mx.float32).reshape(1, 2, 4)
    model = SimpleNamespace(lm_head=_linear_with_weight(weight))
    _shard_output_head(model, _FakeGroup(1), mx)
    model.lm_head._omlx_gather_vocab_logits = False

    logits = model.lm_head(inputs)

    assert logits.shape == (1, 2, 3)
    assert mx.array_equal(logits, inputs @ weight[3:].T).item()


def test_mtp_auxiliary_vocab_head_uses_matching_row_shard(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    main_weight = mx.arange(24, dtype=mx.float32).reshape(6, 4)
    markov_weight = mx.arange(18, dtype=mx.float32).reshape(6, 3)
    markov = SimpleNamespace(markov_w2=_linear_with_weight(markov_weight))
    model = SimpleNamespace(lm_head=_linear_with_weight(main_weight))
    model._omlx_tensor_vocab_modules = lambda: ((markov, "markov_w2"),)

    assert _shard_output_head(model, _FakeGroup(1), mx)
    assert _shard_auxiliary_vocab_heads(model, _FakeGroup(1), mx) == 1

    assert mx.array_equal(model.lm_head.weight, main_weight[3:]).item()
    assert mx.array_equal(markov.markov_w2.weight, markov_weight[3:]).item()
    assert model._omlx_vocab_parallel_aux_heads == (markov.markov_w2,)
    assert model._omlx_distributed_mtp_vocab_ready is True


def test_vocab_parallel_quantized_head_matches_full_projection(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    mx.random.seed(4)
    dense = nn.Linear(64, 6, bias=False)
    dense.weight = mx.random.normal((6, 64)) * 0.05
    head = nn.QuantizedLinear.from_linear(dense, group_size=32, bits=4)
    inputs = mx.random.normal((1, 3, 64))
    expected = head(inputs)
    parts = []

    for rank in (0, 1):
        model = SimpleNamespace(lm_head=copy.deepcopy(head))
        assert _shard_output_head(model, _FakeGroup(rank), mx)
        assert model.lm_head.weight.shape[0] == 3
        assert model.lm_head.scales.shape[0] == 3
        parts.append(model.lm_head._local_logits(inputs))

    reconstructed = mx.concatenate(parts, axis=-1)
    assert mx.allclose(reconstructed, expected, rtol=1e-5, atol=1e-5).item()


def test_vocab_logits_gather_restores_final_axis_order():
    local = mx.arange(12).reshape(2, 2, 3)

    class _Distributed:
        @staticmethod
        def all_gather(value, *, group):
            assert group == "group"
            return mx.concatenate([value, value + 100], axis=0)

    fake_mx = SimpleNamespace(
        contiguous=mx.contiguous,
        distributed=_Distributed(),
        swapaxes=mx.swapaxes,
    )
    gathered = _gather_vocab_logits(local, "group", fake_mx)
    expected = mx.concatenate([local, local + 100], axis=-1)
    assert mx.array_equal(gathered, expected).item()


def test_vocab_parallel_auto_keeps_small_head_replicated(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "auto")
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES", "97")
    head = _linear_with_weight(mx.zeros((6, 4)))
    model = SimpleNamespace(lm_head=head)

    assert not _shard_output_head(model, _FakeGroup(0), mx)
    assert model.lm_head is head
    assert not hasattr(model, "_omlx_vocab_parallel_head")


def test_qwen35_vocab_parallel_stays_fail_closed_without_parity(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "auto")
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES", "0")
    head = _linear_with_weight(mx.zeros((6, 4)))
    model = SimpleNamespace(model_type="qwen3_5", lm_head=head)

    assert not _shard_output_head(model, _FakeGroup(0), mx)
    assert model.lm_head is head
    assert "not parity-qualified" in model._omlx_vocab_parallel_disabled_reason

    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    with pytest.raises(RuntimeError, match="not parity-qualified"):
        _shard_output_head(model, _FakeGroup(0), mx)


@pytest.mark.parametrize("tie_by_config", [True, False])
def test_vocab_parallel_never_slices_a_tied_embedding(monkeypatch, tie_by_config):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "auto")
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL_MIN_BYTES", "0")
    head = _linear_with_weight(mx.zeros((6, 4)))
    embedding = nn.Embedding(6, 4)
    if not tie_by_config:
        embedding.weight = head.weight
    model = SimpleNamespace(
        args=SimpleNamespace(tie_word_embeddings=tie_by_config),
        model=SimpleNamespace(embed_tokens=embedding),
        lm_head=head,
    )

    assert not _shard_output_head(model, _FakeGroup(0), mx)
    assert model.lm_head is head


def test_vocab_parallel_finds_private_language_model_adapter(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    language_model = SimpleNamespace(
        lm_head=_linear_with_weight(mx.zeros((6, 4)))
    )
    model = SimpleNamespace(_language_model=language_model)

    assert _shard_output_head(model, _FakeGroup(0), mx)
    assert language_model.lm_head.weight.shape == (3, 4)


def test_forced_vocab_parallel_rejects_nondivisible_vocab(monkeypatch):
    monkeypatch.setenv("OMLX_CLUSTER_VOCAB_PARALLEL", "on")
    model = SimpleNamespace(lm_head=_linear_with_weight(mx.zeros((5, 4))))

    with pytest.raises(RuntimeError, match="not divisible"):
        _shard_output_head(model, _FakeGroup(0), mx)


def test_qwen3_moe_strategy_covers_text_and_vl_moe_layers(monkeypatch):
    from mlx_lm.models import qwen3_moe
    import mlx.nn.layers.distributed as distributed_layers
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe",
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[1],
        moe_intermediate_size=8,
        rms_norm_eps=1e-6,
        vocab_size=16,
        num_key_value_heads=2,
        head_dim=4,
        rope_theta=10_000.0,
        tie_word_embeddings=True,
        max_position_embeddings=128,
        norm_topk_prob=True,
    )
    model = qwen3_moe.Model(args)
    inplace = []
    monkeypatch.setattr(
        distributed_layers,
        "shard_linear",
        lambda module, _mode, *, group: module,
    )
    monkeypatch.setattr(
        distributed_layers,
        "shard_inplace",
        lambda module, mode, *, group: inplace.append((module, mode)),
    )
    monkeypatch.setattr(
        distributed_layers,
        "sum_gradients",
        lambda _group: lambda value: value,
    )

    assert apply_tensor_strategy(model, _FakeGroup(0), mx_module=mx) == "qwen3_moe"
    assert [layer.self_attn.n_heads for layer in model.layers] == [1, 1]
    assert [layer.self_attn.n_kv_heads for layer in model.layers] == [1, 1]
    assert len(inplace) == 3
    assert model.layers[0].mlp.inner.__class__ is qwen3_moe.Qwen3MoeSparseMoeBlock
    assert isinstance(model.layers[1].mlp, qwen3_moe.MLP)


def test_gemma4_strategy_handles_shared_kv_k_equals_v_and_moe(monkeypatch):
    from mlx_lm.models import gemma4_text
    import mlx.nn.layers.distributed as distributed_layers
    from omlx.cluster.tensor_strategies import apply_tensor_strategy

    args = gemma4_text.ModelArgs(
        model_type="gemma4_text",
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        head_dim=4,
        global_head_dim=4,
        vocab_size=16,
        vocab_size_per_layer_input=16,
        num_key_value_heads=2,
        num_global_key_value_heads=2,
        num_kv_shared_layers=1,
        hidden_size_per_layer_input=0,
        attention_k_eq_v=True,
        enable_moe_block=True,
        num_experts=2,
        top_k_experts=1,
        moe_intermediate_size=8,
        layer_types=["full_attention", "full_attention"],
        tie_word_embeddings=True,
    )
    model = gemma4_text.Model(args)
    linear_calls = []
    inplace = []
    monkeypatch.setattr(
        distributed_layers,
        "shard_linear",
        lambda module, mode, *, group: linear_calls.append((module, mode)) or module,
    )
    monkeypatch.setattr(
        distributed_layers,
        "shard_inplace",
        lambda module, mode, *, group: inplace.append((module, mode)),
    )
    monkeypatch.setattr(
        distributed_layers,
        "sum_gradients",
        lambda _group: lambda value: value,
    )

    assert apply_tensor_strategy(model, _FakeGroup(0), mx_module=mx) == "gemma4"
    assert [layer.self_attn.n_heads for layer in model.layers] == [1, 1]
    assert [layer.self_attn.n_kv_heads for layer in model.layers] == [1, 1]
    # Q/O on both layers, K only on the owning layer, no V for K==V, and
    # three dense MLP projections on both layers.
    assert len(linear_calls) == 11
    assert len(inplace) == 6
    assert all(hasattr(layer.experts, "inner") for layer in model.layers)


@pytest.mark.parametrize(
    "total, size, expected",
    [
        (29, 2, [(0, 15), (15, 29)]),  # the Nemotron-H case: 15 + 14
        (58, 2, [(0, 29), (29, 58)]),  # even divides
        (42, 3, [(0, 14), (14, 28), (28, 42)]),
        (29, 4, [(0, 8), (8, 15), (15, 22), (22, 29)]),
        (1, 1, [(0, 1)]),
    ],
)
def test_uneven_group_ranges(total, size, expected):
    ranges = _uneven_group_ranges(total, size)
    assert ranges == expected
    # Cover [0, total) with no gap or overlap, and skew at most one group.
    assert ranges[0][0] == 0 and ranges[-1][1] == total
    for a, b in zip(ranges, ranges[1:]):
        assert a[1] == b[0]
    widths = [hi - lo for lo, hi in ranges]
    assert max(widths) - min(widths) <= 1
    # Low ranks absorb the extra group (rank 0 is the coordinator).
    assert widths == sorted(widths, reverse=True)


class _SwitchMLP:
    def __init__(self, fc1, fc2):
        self.fc1 = fc1
        self.fc2 = fc2


def _make_quantized_switch_mlp(experts, hidden, intermediate, group_size, bits):
    fc1 = SwitchLinear(hidden, intermediate, experts, bias=False)
    fc2 = SwitchLinear(intermediate, hidden, experts, bias=False)
    fc1.weight = mx.random.normal(fc1.weight.shape) * 0.05
    fc2.weight = mx.random.normal(fc2.weight.shape) * 0.05
    fc1 = fc1.to_quantized(group_size=group_size, bits=bits)
    fc2 = fc2.to_quantized(group_size=group_size, bits=bits)
    return _SwitchMLP(fc1, fc2)


def test_uneven_switch_mlp_split_matches_unsharded():
    """rank0(15 groups) + rank1(14 groups) all_sum == unsharded MoE output."""

    mx.random.seed(0)
    experts, hidden, intermediate, gs, bits = 8, 2688, 1856, 64, 4
    tokens, top_k = 5, 3

    mlp = _make_quantized_switch_mlp(experts, hidden, intermediate, gs, bits)
    # The intermediate axis has a prime group count: this is the whole point.
    assert mlp.fc2.scales.shape[-1] == 29

    x = mx.random.normal((tokens, 1, 1, hidden))
    indices = mx.random.randint(0, experts, (tokens, 1, top_k))

    def forward(mod):
        h = mod.fc1(x, indices)
        h = mx.maximum(h, 0)
        h = h * h  # relu2, as in nemotron_h SwitchMLP
        return mod.fc2(h, indices)

    full = forward(mlp)

    parts = []
    for rank in (0, 1):
        shard = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
        _shard_switch_mlp_uneven(shard, group=None, mx=mx, rank=rank, size=2)
        parts.append(forward(shard))

    # rank0 owns 15 of 29 groups (960 dims), rank1 owns 14 (896).
    recombined = parts[0] + parts[1]  # the all_sum in _wrap_sharded_moe
    err = mx.abs(full - recombined).max().item()
    ref = mx.abs(full).max().item()
    assert err < 1e-4 * max(ref, 1.0), f"uneven split diverged: {err} vs {ref}"


def test_uneven_switch_mlp_shard_shapes():
    """Per-rank shard shapes land on group boundaries for weight and scales."""

    mx.random.seed(1)
    experts, hidden, intermediate, gs, bits = 8, 2688, 1856, 64, 4
    mlp = _make_quantized_switch_mlp(experts, hidden, intermediate, gs, bits)

    rank0 = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
    _shard_switch_mlp_uneven(rank0, group=None, mx=mx, rank=0, size=2)
    rank1 = _SwitchMLP(copy.deepcopy(mlp.fc1), copy.deepcopy(mlp.fc2))
    _shard_switch_mlp_uneven(rank1, group=None, mx=mx, rank=1, size=2)

    # fc1 column-parallel: output rows split 960 / 896 (= 15*64 / 14*64).
    assert rank0.fc1.weight.shape[1] == 960
    assert rank1.fc1.weight.shape[1] == 896
    # fc2 scales split 15 / 14 groups; packed weight cols split 120 / 112.
    assert rank0.fc2.scales.shape[-1] == 15
    assert rank1.fc2.scales.shape[-1] == 14
    assert rank0.fc2.weight.shape[-1] == 120  # 15 groups * (64/8) packed cols
    assert rank1.fc2.weight.shape[-1] == 112
    # No dropped groups.
    assert rank0.fc2.scales.shape[-1] + rank1.fc2.scales.shape[-1] == 29
