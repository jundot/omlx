# SPDX-License-Identifier: Apache-2.0
"""Uneven tensor-parallel splits (DeepSeek V4.1 on Macs of different sizes)."""

import json
import struct
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.cluster.autoconfigure import candidate_tensor_parallel_sizes
from omlx.cluster.deployment import _assignment_from_dict
from omlx.cluster.inference_worker import _tensor_shard_options
from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PlanningError,
    inspect_safetensors_layout,
    plan_hybrid,
    tensor_split_from_dict,
)
from omlx.patches.deepseek_v41.config import ModelConfig, TensorSplit
from omlx.patches.deepseek_v41.loading import tensor_slice

GiB = 1024**3


def _layout(**overrides):
    """A V4.1-shaped layout: 8 output groups, 36 expert units, 40 layers."""

    values = dict(
        source="test",
        fixed_weight_bytes=2 * GiB,
        layer_weight_bytes=(7 * GiB,) * 40,
        tensor_parallel_heads=64,
        tensor_parallel_divisors=(64, 8),
        supports_tensor_parallel=True,
        tensor_parallel_split_units=(8, 36),
        # 40 layers x (0.25 GiB attention + 6.5 GiB experts + 0.25 replicated).
        tensor_parallel_split_bytes=(10 * GiB, 260 * GiB),
    )
    values.update(overrides)
    return ModelLayout(**values)


def _studio_and_macbook():
    return [
        NodeBudget("studio", 243 * GiB, reserve_bytes=8 * GiB, rank=0),
        NodeBudget("macbook", 121 * GiB, reserve_bytes=3 * GiB, rank=1),
    ]


def test_auto_split_is_equal_when_equal_fits():
    nodes = [
        NodeBudget("a", 243 * GiB, reserve_bytes=8 * GiB, rank=0),
        NodeBudget("b", 243 * GiB, reserve_bytes=8 * GiB, rank=1),
    ]

    plan = plan_hybrid(_layout(), nodes, tensor_parallel_size=2)

    assert all(not item.tensor_parallel_split for item in plan.assignments)
    # Equal plans serialise exactly as before, so their hashes do not move.
    assert all(
        "tensor_parallel_split" not in item.to_dict() for item in plan.assignments
    )


def test_auto_split_sizes_each_rank_when_equal_does_not_fit():
    model = _layout()
    with pytest.raises(PlanningError):
        plan_hybrid(
            model, _studio_and_macbook(), tensor_parallel_size=2, tensor_split="equal"
        )

    plan = plan_hybrid(model, _studio_and_macbook(), tensor_parallel_size=2)

    studio, macbook = plan.assignments
    split = studio.tensor_parallel_split
    assert split == macbook.tensor_parallel_split
    attention, experts = split
    assert sum(attention) == 8 and sum(experts) == 36
    assert attention[0] > attention[1] and experts[0] > experts[1]
    # Each rank holds the replicated part plus its own share of each axis.
    replicated = sum(model.tensor_parallel_layer_bytes) - 270 * GiB
    assert studio.layer_weight_bytes == (
        replicated + 10 * GiB * attention[0] // 8 + 260 * GiB * experts[0] // 36
    )
    for item, node in zip(plan.assignments, _studio_and_macbook()):
        assert item.planned_weight_bytes <= node.usable_bytes


def test_explicit_split_is_used_and_serialised():
    split = {"attention": [5, 3], "experts": [25, 11]}

    plan = plan_hybrid(
        _layout(), _studio_and_macbook(), tensor_parallel_size=2, tensor_split=split
    )

    for item in plan.assignments:
        assert item.to_dict()["tensor_parallel_split"] == split
        decoded = _assignment_from_dict(item.to_dict())
        assert decoded.tensor_parallel_split == ((5, 3), (25, 11))
    other = plan_hybrid(
        _layout(),
        _studio_and_macbook(),
        tensor_parallel_size=2,
        tensor_split={"attention": [5, 3], "experts": [24, 12]},
    )
    assert other.plan_hash != plan.plan_hash


@pytest.mark.parametrize(
    "split",
    [
        {"attention": [5, 2], "experts": [25, 11]},
        {"attention": [5, 3], "experts": [36, 0]},
        {"attention": [8], "experts": [36]},
    ],
)
def test_explicit_split_must_cover_every_unit_on_every_rank(split):
    with pytest.raises((PlanningError, ValueError)):
        plan_hybrid(
            _layout(), _studio_and_macbook(), tensor_parallel_size=2, tensor_split=split
        )


def test_split_that_overflows_a_rank_is_refused():
    with pytest.raises(PlanningError, match="does not fit"):
        plan_hybrid(
            _layout(),
            _studio_and_macbook(),
            tensor_parallel_size=2,
            tensor_split={"attention": [1, 7], "experts": [6, 30]},
        )


def test_uneven_split_needs_an_architecture_that_supports_it():
    model = _layout(tensor_parallel_split_units=(), tensor_parallel_split_bytes=())

    with pytest.raises(PlanningError, match="uneven tensor split"):
        plan_hybrid(
            model,
            _studio_and_macbook(),
            tensor_parallel_size=2,
            tensor_split="proportional",
        )


def test_layout_round_trip_keeps_split_metadata():
    model = _layout()

    assert ModelLayout.from_dict(model.to_dict()) == model
    plain = _layout(tensor_parallel_split_units=(), tensor_parallel_split_bytes=())
    assert "tensor_parallel_split_units" not in plain.to_dict()


def test_three_way_split_is_offered_when_units_allow_it():
    assert candidate_tensor_parallel_sizes(_layout(), 3)[0] == 3


def test_split_from_dict_rejects_malformed_input():
    assert tensor_split_from_dict(None) == ()
    for bad in ({"attention": [5, 3]}, {"attention": [5], "experts": [1, 2]}, [1, 2]):
        with pytest.raises(ValueError):
            tensor_split_from_dict(bad)


def test_worker_passes_its_plan_split_to_the_loader():
    execution = SimpleNamespace(ced_prefill=False, a8_prefill=False)
    assignment = SimpleNamespace(tensor_parallel_split=((5, 3), (25, 11)))

    assert _tensor_shard_options(execution, assignment) == {
        "tensor_split": {"attention": [5, 3], "experts": [25, 11]}
    }
    assert _tensor_shard_options(execution, SimpleNamespace()) == {}


def _write_safetensors(path, tensors):
    header, offset = {}, 0
    for name, size in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0" * offset)


def test_inspection_measures_attention_and_expert_bytes(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v41",
                "text_config": {
                    "num_attention_heads": 64,
                    "o_groups": 8,
                    "moe_intermediate_size": 2304,
                    "num_hidden_layers": 2,
                },
            }
        )
    )
    tensors = {"language_model.embed.weight": 100}
    for layer in range(2):
        prefix = f"language_model.layers.{layer}."
        tensors[prefix + "attn.wq_b.weight"] = 10
        tensors[prefix + "attn.wo_b.scales"] = 2
        tensors[prefix + "attn.attn_sink"] = 1
        tensors[prefix + "attn.wkv.weight"] = 5
        tensors[prefix + "ffn.experts.w1.weight"] = 40
        tensors[prefix + "ffn.shared_experts.w2.weight"] = 8
        tensors[prefix + "ffn.gate.weight"] = 3
    _write_safetensors(tmp_path / "model.safetensors", tensors)

    layout = inspect_safetensors_layout(tmp_path)

    assert layout.tensor_parallel_split_units == (8, 36)
    assert layout.tensor_parallel_split_bytes == (2 * 13, 2 * 48)


def _config():
    return ModelConfig(n_heads=64, o_groups=8, moe_inter_dim=2304)


def test_weighted_tensor_shard_shapes():
    split = TensorSplit((5, 3), (25, 11))

    first = _config().tensor_shard(2, 0, split)
    second = _config().tensor_shard(2, 1, split)

    assert (first.n_heads, first.o_groups, first.moe_inter_dim) == (40, 5, 1600)
    assert (second.n_heads, second.o_groups, second.moe_inter_dim) == (24, 3, 704)
    # The fused decode MoE kernel needs 64-neuron multiples on every rank.
    assert first.moe_inter_dim % 64 == 0 and second.moe_inter_dim % 64 == 0
    with pytest.raises(ValueError):
        _config().tensor_shard(2, 0, TensorSplit((4, 3), (25, 11)))
    with pytest.raises(ValueError):
        _config().tensor_shard(2, 0, TensorSplit((5, 3), (24, 11)))


@pytest.mark.parametrize(
    ("key", "shape", "axis"),
    [
        ("language_model.layers.0.attn.wq_b.weight", (64 * 4, 6), 0),
        ("language_model.layers.0.attn.wo_b.weight", (6, 8 * 16), 1),
        ("language_model.layers.0.attn.attn_sink", (64,), 0),
        ("language_model.layers.0.ffn.experts.w1.weight", (2, 2304, 4), 1),
        ("language_model.layers.0.ffn.shared_experts.w2.scales", (4, 2304 // 32), 1),
    ],
)
def test_weighted_tensor_slices_partition_each_axis(key, shape, axis):
    split = TensorSplit((5, 3), (25, 11))
    value = mx.arange(int(mx.prod(mx.array(shape)).item())).reshape(shape)

    parts = [tensor_slice(key, value, rank, 2, split) for rank in range(2)]

    expert = ".experts." in key or "shared_experts" in key
    first_share = 25 / 36 if expert else 5 / 8
    assert parts[0].shape[axis] == round(shape[axis] * first_share)
    assert mx.array_equal(mx.concatenate(parts, axis=axis), value)


def test_replicated_tensors_are_not_sliced():
    value = mx.ones((8, 8))
    split = TensorSplit((5, 3), (25, 11))

    assert (
        tensor_slice("language_model.layers.0.attn.wkv.weight", value, 1, 2, split)
        is value
    )


def test_routes_validate_the_split_and_approve_it():
    from omlx.cluster.routes import _placement_signature, _tensor_split_arg

    assert _tensor_split_arg(None) == "auto"
    with pytest.raises(PlanningError):
        _tensor_split_arg({"attention": [5]})
    first = plan_hybrid(
        _layout(),
        _studio_and_macbook(),
        tensor_parallel_size=2,
        tensor_split={"attention": [5, 3], "experts": [25, 11]},
    )
    second = plan_hybrid(
        _layout(),
        _studio_and_macbook(),
        tensor_parallel_size=2,
        tensor_split={"attention": [5, 3], "experts": [24, 12]},
    )
    # Approving one split must not approve another.
    assert _placement_signature(first.to_dict()) != _placement_signature(
        second.to_dict()
    )
