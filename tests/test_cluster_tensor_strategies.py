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

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import SwitchLinear

from omlx.cluster.tensor_strategies import (
    _shard_switch_mlp_uneven,
    _uneven_group_ranges,
)


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




def test_qwen4_exp_strategy_registration():
    from omlx.cluster.tensor_strategies import (
        QWEN4_EXP,
        registered_model_types,
        supports_model_type,
    )

    assert QWEN4_EXP.name == "qwen4_exp"
    assert "qwen4_exp" in QWEN4_EXP.model_types
    assert "qwen4_exp_text" in QWEN4_EXP.model_types
    assert "qwen4_exp" in registered_model_types()
    assert "qwen4_exp_text" in registered_model_types()
    assert supports_model_type("qwen4_exp")
    assert supports_model_type("qwen4_exp_text")


def test_qwen4_exp_planner_divisors_and_support():
    from omlx.cluster.planner import (
        _supports_tensor_parallel,
        _tensor_parallel_divisors,
    )

    config = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
        },
    }
    assert _supports_tensor_parallel(config)
    divisors = _tensor_parallel_divisors(config)
    assert 24 in divisors
    assert 2 in divisors
    assert 16 in divisors
    assert 48 in divisors


class _MockGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


def test_shard_qwen4_exp_linear_attention_dimensions():
    import mlx.nn as nn
    from omlx.cluster.tensor_strategies import _shard_qwen4_exp

    class MockGatedDeltaNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.dk = 128
            self.dv = 128
            self.n_k = 2
            self.n_v = 4
            self.key_dim = self.dk * self.n_k  # 256
            self.value_dim = self.dv * self.n_v  # 512
            self.conv_dim = self.key_dim * 2 + self.value_dim  # 1024
            d = 512
            self.conv1d = nn.Conv1d(
                self.conv_dim, self.conv_dim, kernel_size=4, groups=self.conv_dim, bias=False
            )
            self.in_proj_qkv = nn.Linear(d, self.conv_dim, bias=False)
            self.in_proj_z = nn.Linear(d, self.value_dim, bias=False)
            self.in_proj_b = nn.Linear(d, self.n_v, bias=False)
            self.in_proj_a = nn.Linear(d, self.n_v, bias=False)
            self.dt_bias = mx.ones(self.n_v)
            self.A_log = mx.zeros(self.n_v)
            self.out_proj = nn.Linear(self.value_dim, d, bias=False)

    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(512, 512, bias=False)
            self.up_proj = nn.Linear(512, 512, bias=False)
            self.down_proj = nn.Linear(512, 512, bias=False)

    class MockDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_type = "linear_attention"
            self.linear_attn = MockGatedDeltaNet()
            self.mlp = MockMLP()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [MockDecoderLayer()]

    model = MockModel()
    group = _MockGroup(rank=0, size=2)
    _shard_qwen4_exp(model, group, mx, None)

    attn = model.layers[0].linear_attn
    assert attn.n_k == 1
    assert attn.n_v == 2
    assert attn.key_dim == 128
    assert attn.value_dim == 256
    assert attn.conv_dim == 512
    assert attn.conv1d.groups == 512
    assert attn.conv1d.weight.shape[0] == 512
    assert attn.in_proj_qkv.weight.shape[0] == 512
    assert attn.in_proj_z.weight.shape[0] == 256
    assert attn.in_proj_b.weight.shape[0] == 2
    assert attn.in_proj_a.weight.shape[0] == 2
    assert attn.A_log.shape[0] == 2
    assert attn.dt_bias.shape[0] == 2


def test_shard_qwen4_exp_self_attention_dimensions():
    import mlx.nn as nn
    from omlx.cluster.tensor_strategies import _shard_qwen4_exp

    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_attention_heads = 24
            self.num_key_value_heads = 2
            d = 512
            self.q_proj = nn.Linear(d, d, bias=False)
            self.k_proj = nn.Linear(d, 128, bias=False)
            self.v_proj = nn.Linear(d, 128, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)

    class MockMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(512, 512, bias=False)
            self.up_proj = nn.Linear(512, 512, bias=False)
            self.down_proj = nn.Linear(512, 512, bias=False)

    class MockDecoderLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer_type = "full_attention"
            self.self_attn = MockAttention()
            self.mlp = MockMLP()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [MockDecoderLayer()]

    model = MockModel()
    group = _MockGroup(rank=1, size=2)
    _shard_qwen4_exp(model, group, mx, None)

    attn = model.layers[0].self_attn
    assert attn.num_attention_heads == 12
    assert attn.num_key_value_heads == 1
