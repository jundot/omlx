# SPDX-License-Identifier: Apache-2.0
"""Tests for unequal-memory model layout and pipeline planning."""

import json
import struct

import pytest

from omlx.cluster.planner import (
    ModelLayout,
    NodeBudget,
    PipelineAssignment,
    PlanningError,
    apply_pipeline_assignment,
    inspect_safetensors_layout,
    plan_unequal_pipeline,
    synthetic_model_layout,
)

GIB = 1024**3


def _write_safetensors(path, tensors):
    offset = 0
    header = {}
    for name, size in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": [size],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * offset)


def test_inspect_safetensors_layout_reads_headers_without_loading_tensors(tmp_path):
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_proj.weight", 200),
            ("model.layers.0.mlp.down_proj.weight", 300),
            ("model.layers.1.self_attn.q_proj.weight", 400),
            ("model.norm.weight", 50),
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)

    assert layout.fixed_weight_bytes == 150
    assert layout.layer_weight_bytes == (500, 400)
    assert layout.total_weight_bytes == 1050
    assert layout.tensor_count == 5


def test_deepseek_v4_vision_layout_charges_sidecar_only_to_rank_zero(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "num_hidden_layers": 4,
                "vision_n_layers": 2,
                "vision_dim": 8,
                "vision_n_heads": 2,
                "vision_inter_dim": 16,
                "vision_patch_size": 2,
                "vision_rope_theta": 10000.0,
                "vision_downsample_ratio": 2,
                "vision_max_n_token": 16,
                "vision_min_pixels": 16,
                "vision_max_wh_ratio": 8,
            }
        )
    )
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("embed.weight", 20),
            ("vision.patch_embed.proj.weight", 50),
            ("vision.blocks.0.attn.wqkv.weight", 70),
            ("aligner.w1.weight", 30),
            ("image_start", 10),
            *((f"layers.{index}.weight", 100) for index in range(4)),
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)
    assert layout.fixed_weight_bytes == 20
    assert layout.coordinator_weight_bytes == 160
    assert layout.layer_weight_bytes == (100, 100, 100, 100)

    plan = plan_unequal_pipeline(
        layout,
        [
            NodeBudget(node_id="coordinator", capacity_bytes=480, rank=0),
            NodeBudget(node_id="worker", capacity_bytes=480, rank=1),
        ],
        context_tokens=0,
    )
    coordinator, worker = plan.assignments
    assert coordinator.coordinator_weight_bytes == 160
    assert worker.coordinator_weight_bytes == 0
    assert coordinator.layer_count < worker.layer_count
    assert plan.cluster_resident_weight_bytes == 600


def test_inspect_safetensors_layout_rejects_offset_past_file(tmp_path):
    header = {
        "model.layers.0.weight": {
            "dtype": "U8",
            "shape": [10],
            "data_offsets": [0, 10],
        }
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded
    )

    with pytest.raises(PlanningError, match="invalid data offsets"):
        inspect_safetensors_layout(tmp_path)


def test_inspect_safetensors_layout_rejects_overlapping_offsets(tmp_path):
    header = {
        "model.layers.0.first": {
            "dtype": "U8",
            "shape": [8],
            "data_offsets": [0, 8],
        },
        "model.layers.0.second": {
            "dtype": "U8",
            "shape": [6],
            "data_offsets": [4, 10],
        },
    }
    encoded = json.dumps(header).encode()
    (tmp_path / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"\0" * 10
    )

    with pytest.raises(PlanningError, match="overlapping"):
        inspect_safetensors_layout(tmp_path)


def test_mtp_layers_past_the_declared_depth_are_not_decoder_layers(tmp_path):
    """DeepSeek/GLM MTP heads live at index ``num_hidden_layers`` and up.

    The runtime model never instantiates them, so counting them as decoder
    layers put the last stage boundary past the model and activation failed
    with ``end_layer`` beyond the loaded layers.
    """

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 2}))
    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("model.embed_tokens.weight", 100),
            ("model.layers.0.self_attn.q_proj.weight", 500),
            ("model.layers.1.self_attn.q_proj.weight", 400),
            ("model.layers.2.eh_proj.weight", 900),  # the MTP head
        ],
    )

    layout = inspect_safetensors_layout(tmp_path)

    assert layout.layer_count == 2
    assert layout.layer_weight_bytes == (500, 400)


def test_layers_past_an_undeclared_depth_are_kept(tmp_path):
    """No config.json means no declared depth to trim against."""

    _write_safetensors(
        tmp_path / "model.safetensors",
        [
            ("model.layers.0.weight", 500),
            ("model.layers.1.weight", 400),
            ("model.layers.2.weight", 900),
        ],
    )

    assert inspect_safetensors_layout(tmp_path).layer_count == 3


def test_unequal_planner_gives_more_layers_to_larger_mac():
    model = synthetic_model_layout(
        total_weight_bytes=300 * GIB,
        layer_count=80,
    )
    nodes = [
        NodeBudget(node_id="studio", capacity_bytes=256 * GIB, rank=0),
        NodeBudget(node_id="mobile", capacity_bytes=128 * GIB, rank=1),
    ]

    plan = plan_unequal_pipeline(model, nodes)
    studio, mobile = plan.assignments

    assert studio.rank == 0
    assert studio.layer_count > mobile.layer_count
    assert mobile.start_layer == 0
    assert mobile.end_layer == studio.start_layer
    assert studio.end_layer == 80
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)
    assert len(plan.plan_hash) == 64


def test_unequal_planner_accounts_for_replicated_fixed_weights_and_reserve():
    model = ModelLayout(
        source="test",
        fixed_weight_bytes=4 * GIB,
        layer_weight_bytes=(8 * GIB,) * 6,
    )
    nodes = [
        NodeBudget(
            node_id="large",
            capacity_bytes=40 * GIB,
            reserve_bytes=4 * GIB,
            rank=0,
        ),
        NodeBudget(
            node_id="small",
            capacity_bytes=24 * GIB,
            reserve_bytes=4 * GIB,
            rank=1,
        ),
    ]

    plan = plan_unequal_pipeline(model, nodes)

    assert plan.cluster_resident_weight_bytes == model.total_weight_bytes + 4 * GIB
    assert all(item.headroom_bytes >= 0 for item in plan.assignments)


def test_unequal_planner_rejects_model_that_does_not_fit():
    model = synthetic_model_layout(
        total_weight_bytes=400 * GIB,
        layer_count=80,
    )
    nodes = [
        NodeBudget(node_id="large", capacity_bytes=256 * GIB, rank=0),
        NodeBudget(node_id="small", capacity_bytes=128 * GIB, rank=1),
    ]

    with pytest.raises(PlanningError, match="does not fit"):
        plan_unequal_pipeline(model, nodes)


def test_three_node_weight_targets_choose_the_nearest_contiguous_split():
    model = synthetic_model_layout(
        total_weight_bytes=120 * GIB,
        layer_count=12,
    )
    nodes = [
        NodeBudget(
            node_id="coordinator",
            capacity_bytes=80 * GIB,
            target_weight_bytes=20 * GIB,
            rank=0,
        ),
        NodeBudget(
            node_id="studio",
            capacity_bytes=100 * GIB,
            target_weight_bytes=70 * GIB,
            rank=1,
        ),
        NodeBudget(
            node_id="mini",
            capacity_bytes=60 * GIB,
            target_weight_bytes=30 * GIB,
            rank=2,
        ),
    ]

    plan = plan_unequal_pipeline(model, nodes)
    weights = {
        item.node_id: item.layer_weight_bytes + item.fixed_weight_bytes
        for item in plan.assignments
    }

    assert weights == {
        "coordinator": 20 * GIB,
        "studio": 70 * GIB,
        "mini": 30 * GIB,
    }
    assert [
        (item.start_layer, item.end_layer)
        for item in sorted(plan.assignments, key=lambda item: item.start_layer)
    ] == [(0, 3), (3, 10), (10, 12)]


def test_synthetic_layout_has_bounded_layer_count():
    with pytest.raises(ValueError, match="2048 layer limit"):
        synthetic_model_layout(total_weight_bytes=1, layer_count=2049)


def test_apply_pipeline_assignment_uses_explicit_range():
    class Group:
        @staticmethod
        def rank():
            return 1

        @staticmethod
        def size():
            return 2

    class PipelineModel:
        def __init__(self):
            self.layers = list(range(8))

    assignments = [
        PipelineAssignment(
            node_id="large",
            rank=0,
            start_layer=3,
            end_layer=8,
            layer_weight_bytes=5,
            fixed_weight_bytes=0,
            reserve_bytes=0,
            capacity_bytes=10,
        ),
        PipelineAssignment(
            node_id="small",
            rank=1,
            start_layer=0,
            end_layer=3,
            layer_weight_bytes=3,
            fixed_weight_bytes=0,
            reserve_bytes=0,
            capacity_bytes=10,
        ),
    ]
    model = PipelineModel()

    apply_pipeline_assignment(model, Group(), assignments)

    assert model.pipeline_rank == 1
    assert model.pipeline_size == 2
    assert model.start_idx == 0
    assert model.end_idx == 3
    assert model.layers == [0, 1, 2]


# --- The node role travels on the plan, because nothing else reaches a rank --


def test_the_planner_puts_each_nodes_role_on_its_own_assignment():
    """One plan, two Macs, two different roles.

    The launcher emits a single argv for the whole cluster, so this is the only
    place a per-Mac setting can be written down.
    """

    from omlx.cluster.planner import plan_hybrid

    model = synthetic_model_layout(total_weight_bytes=60 * GIB, layer_count=8)
    nodes = [
        NodeBudget(
            node_id="studio",
            capacity_bytes=256 * GIB,
            rank=0,
            role="headless",
        ),
        NodeBudget(
            node_id="macbook",
            capacity_bytes=107 * GIB,
            reserve_bytes=32 * GIB,
            rank=1,
            role="workstation",
        ),
    ]

    unequal = plan_unequal_pipeline(model, nodes)
    hybrid = plan_hybrid(model, nodes, tensor_parallel_size=1)

    for plan in (unequal, hybrid):
        by_id = {item.node_id: item.role for item in plan.assignments}
        assert by_id == {"studio": "headless", "macbook": "workstation"}


def test_tp_width_one_is_exactly_the_pipeline_plan_the_launcher_recomputes():
    """Autoconfigure and activation must sign the same unequal-memory cut."""

    from omlx.cluster.planner import ModelLayout, plan_hybrid

    model = ModelLayout(
        source="uneven",
        fixed_weight_bytes=0,
        layer_weight_bytes=tuple(
            size * GIB
            for size in (2, 4, 6, 1, 1, 5, 1, 3, 5, 1, 5, 2, 1, 1, 4, 4, 1, 2)
        ),
    )
    nodes = [
        NodeBudget(
            node_id="laptop",
            capacity_bytes=62 * GIB,
            reserve_bytes=11 * GIB,
            rank=0,
        ),
        NodeBudget(
            node_id="studio",
            capacity_bytes=162 * GIB,
            reserve_bytes=8 * GIB,
            rank=1,
        ),
    ]

    pipeline = plan_unequal_pipeline(model, nodes)
    automatic = plan_hybrid(model, nodes, tensor_parallel_size=1)

    assert automatic.assignments == pipeline.assignments
    assert automatic.plan_hash == pipeline.plan_hash
    assert automatic.pipeline_stages == 2


def test_the_planner_puts_each_nodes_memory_tier_on_its_own_assignment():
    from omlx.cluster.planner import plan_hybrid

    model = synthetic_model_layout(total_weight_bytes=60 * GIB, layer_count=8)
    nodes = [
        NodeBudget(
            node_id="studio",
            capacity_bytes=256 * GIB,
            rank=0,
            memory_guard_tier="aggressive",
        ),
        NodeBudget(
            node_id="macbook",
            capacity_bytes=107 * GIB,
            reserve_bytes=32 * GIB,
            rank=1,
            memory_guard_tier="safe",
        ),
    ]

    for plan in (
        plan_unequal_pipeline(model, nodes),
        plan_hybrid(model, nodes, tensor_parallel_size=1),
    ):
        assert {
            item.node_id: item.memory_guard_tier for item in plan.assignments
        } == {"studio": "aggressive", "macbook": "safe"}


def test_a_memory_tier_changes_the_plan_hash_and_typos_are_refused():
    model = synthetic_model_layout(total_weight_bytes=60 * GIB, layer_count=8)

    def planned(tier):
        return plan_unequal_pipeline(
            model,
            [
                NodeBudget(
                    node_id="studio",
                    capacity_bytes=256 * GIB,
                    rank=0,
                    memory_guard_tier=tier,
                ),
                NodeBudget(
                    node_id="macbook",
                    capacity_bytes=200 * GIB,
                    rank=1,
                ),
            ],
        )

    assert planned("safe").plan_hash != planned("aggressive").plan_hash
    with pytest.raises(ValueError, match="unknown memory guard tier"):
        NodeBudget(
            node_id="macbook",
            capacity_bytes=100 * GIB,
            memory_guard_tier="extreme",
        )


def test_a_node_role_changes_the_plan_hash():
    """Same layers, different admission fraction: a different plan.

    If the hash ignored the role, a headless plan the user previewed and a
    workstation plan that launched would be indistinguishable to every
    staleness check between here and the rank.
    """

    from omlx.cluster.planner import plan_hybrid

    model = synthetic_model_layout(total_weight_bytes=60 * GIB, layer_count=8)

    def _plan(role, planner):
        return planner(
            model,
            [
                NodeBudget(node_id="studio", capacity_bytes=256 * GIB, rank=0),
                NodeBudget(node_id="macbook", capacity_bytes=200 * GIB, rank=1, role=role),
            ],
        )

    for planner in (plan_unequal_pipeline, plan_hybrid):
        headless = _plan("headless", planner)
        workstation = _plan("workstation", planner)
        assert [item.layer_count for item in headless.assignments] == [
            item.layer_count for item in workstation.assignments
        ]
        assert headless.plan_hash != workstation.plan_hash


def test_an_unset_role_plans_exactly_as_before():
    model = synthetic_model_layout(total_weight_bytes=60 * GIB, layer_count=8)
    nodes = [
        NodeBudget(node_id="studio", capacity_bytes=256 * GIB, rank=0),
        NodeBudget(node_id="macbook", capacity_bytes=107 * GIB, rank=1),
    ]

    plan = plan_unequal_pipeline(model, nodes)

    assert [item.role for item in plan.assignments] == ["", ""]


def test_a_misspelled_role_is_refused_rather_than_quietly_made_headless():
    """The fallback direction is the dangerous one.

    ``role_for()`` maps anything unrecognised to headless — 0.90 of the Mac —
    which is right for a label in the UI and wrong for the number a rank
    admits against. A plan does not get to guess.
    """

    with pytest.raises(ValueError, match="unknown node role"):
        NodeBudget(node_id="macbook", capacity_bytes=100 * GIB, role="Workstaton")
    with pytest.raises(ValueError, match="unknown node role"):
        PipelineAssignment(
            node_id="macbook",
            rank=0,
            start_layer=0,
            end_layer=1,
            layer_weight_bytes=1,
            fixed_weight_bytes=0,
            reserve_bytes=0,
            capacity_bytes=10,
            role="laptop",
        )
    # Case and padding are the UI's business, not a reason to refuse a launch.
    assert NodeBudget(
            node_id="macbook", capacity_bytes=100 * GIB, role=" WorkStation "
    ).role == "workstation"


def test_nemotron_h_quant_group_divisors_cap_tp_degree():
    """Quantized even-split row-parallel dims contribute their group counts.

    The Nemotron-H MoE routed experts use a custom *uneven* split (29 groups),
    so 29 must NOT appear as a strict divisor. But the shared-expert down_proj
    (3712 / 64 = 58 groups) uses the even ``shard_inplace`` path, so 58 must be
    present — capping the model at TP=2 for the quantization reason as well as
    the ``num_key_value_heads=2`` reason.
    """

    from omlx.cluster.planner import _tensor_parallel_divisors

    config = {
        "model_type": "nemotron_h",
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "moe_shared_expert_intermediate_size": 3712,
        "quantization": {"group_size": 64, "bits": 4},
    }
    divisors = _tensor_parallel_divisors(config)

    assert 58 in divisors, divisors  # shared down_proj even-split group count
    assert 29 not in divisors, divisors  # routed fc2 handled by the uneven path
    assert all(v % 2 == 0 for v in divisors)  # TP=2 stays viable
    assert not all(v % 4 == 0 for v in divisors)  # TP=4 rejected (58 % 4 != 0)


def test_nemotron_h_divisors_omit_quant_groups_when_unquantized():
    """Without a quantization block there are no group-count constraints."""

    from omlx.cluster.planner import _tensor_parallel_divisors

    config = {
        "model_type": "nemotron_h",
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "moe_shared_expert_intermediate_size": 3712,
    }
    divisors = _tensor_parallel_divisors(config)
    assert 58 not in divisors
    assert set(divisors) == {32, 2, 64, 8}


def test_complete_model_layout_cache_invalidates_on_shard_overwrite(
    tmp_path, monkeypatch
):
    """An in-place shard rewrite bumps neither the directory mtime nor
    config.json's, so the shard stats themselves must be in the cache key."""

    from omlx.cluster import planner

    (tmp_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    _write_safetensors(
        tmp_path / "model.safetensors", [("model.layers.0.weight", 100)]
    )

    calls = []
    real_inspect = planner.inspect_safetensors_layout

    def counting_inspect(path):
        calls.append(str(path))
        return real_inspect(path)

    monkeypatch.setattr(planner, "inspect_safetensors_layout", counting_inspect)

    planner.complete_model_layout(tmp_path)
    planner.complete_model_layout(tmp_path)
    assert len(calls) == 1  # second read served from the cache

    _write_safetensors(
        tmp_path / "model.safetensors", [("model.layers.0.weight", 200)]
    )
    layout = planner.complete_model_layout(tmp_path)
    assert len(calls) == 2  # overwrite invalidated the cached entry
    assert layout.layer_weight_bytes == (200,)


def test_nemotron_h_per_module_quant_overrides_tighten_the_guard():
    """oQ mixed checkpoints override group_size per module inside the
    quantization dict; a coarser override can leave a prime group count the
    top-level size hides, so its group counts must constrain the degree too."""

    from omlx.cluster.planner import _tensor_parallel_divisors

    config = {
        "model_type": "nemotron_h",
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "mamba_num_heads": 64,
        "mamba_head_dim": 64,
        "n_groups": 8,
        "moe_shared_expert_intermediate_size": 3712,
        "quantization": {
            "group_size": 64,
            "bits": 4,
            "backbone.layers.0.mixer.shared_experts.down_proj": {
                "group_size": 128,
                "bits": 6,
            },
        },
    }
    divisors = _tensor_parallel_divisors(config)

    assert 58 in divisors  # 3712 / 64 under the top-level size
    assert 29 in divisors  # 3712 / 128 under the override


def test_nemotron_h_head_dim_falls_back_to_hidden_over_heads():
    """A config without head_dim must not silently drop the attention
    constraint; the runtime falls back to hidden_size // heads and so must
    the guard."""

    from omlx.cluster.planner import _tensor_parallel_divisors

    config = {
        "model_type": "nemotron_h",
        "num_attention_heads": 32,
        "num_key_value_heads": 2,
        "hidden_size": 4096,
        "n_groups": 8,
        "quantization": {"group_size": 64, "bits": 4},
    }
    divisors = _tensor_parallel_divisors(config)

    assert 64 in divisors  # (32 * 128) / 64 via the fallback head_dim


def test_supports_pipeline_false_for_vision_config_vlm(monkeypatch):
    # The text backbone declares a pipeline() in mlx-lm source, but the on-disk
    # checkpoint carries a vision sub-config, so it is served by mlx-vlm and has
    # no model.model.pipeline (progressive_loading gates on exactly that). The
    # static flag must mirror the runtime, i.e. report False.
    from omlx.cluster import planner

    monkeypatch.setattr(
        planner, "_model_source", lambda mt: "def pipeline(self, group): ..."
    )
    config = {"model_type": "qwen3_5_moe", "vision_config": {"depth": 24}}
    assert planner._supports_pipeline(config) is False


def test_supports_pipeline_true_for_text_model(monkeypatch):
    from omlx.cluster import planner

    monkeypatch.setattr(
        planner, "_model_source", lambda mt: "def pipeline(self, group): ..."
    )
    assert planner._supports_pipeline({"model_type": "qwen3_moe"}) is True


def test_explicit_support_declaration_wins_over_vision_guard(monkeypatch):
    # A VLM oMLX explicitly vouches for (ships its own pipeline()) stays True.
    import sys
    import types

    from omlx.cluster import planner

    monkeypatch.setitem(
        sys.modules,
        "mlx_lm.models.minimax_m3_vl",
        types.SimpleNamespace(SUPPORTS_PIPELINE=True),
    )
    config = {"model_type": "minimax_m3_vl", "vision_config": {"depth": 8}}
    assert planner._supports_pipeline(config) is True
