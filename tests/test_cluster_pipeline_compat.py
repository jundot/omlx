# SPDX-License-Identifier: Apache-2.0

import json

from omlx.cluster.pipeline_compat import (
    install_pipeline_compatibility,
    pipeline_assignment_is_honored,
)
from omlx.cluster.planner import PipelineAssignment


def _assignment():
    return (
        PipelineAssignment(
            node_id="local",
            rank=0,
            start_layer=0,
            end_layer=2,
            layer_weight_bytes=2,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=8,
        ),
    )


def _model_config(tmp_path, model_type):
    model = tmp_path / model_type
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"model_type": model_type}))
    return model


def test_standard_pipeline_mixin_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v3")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_thin_qwen_moe_wrapper_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "qwen3_5_moe")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_nemotron_compatibility_has_an_explicit_assignment_contract(tmp_path):
    model = _model_config(tmp_path, "nemotron_h")

    assert not pipeline_assignment_is_honored(model)
    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_deepseek_v32_inherits_the_pipeline_contract(tmp_path):
    model = _model_config(tmp_path, "deepseek_v32")

    with install_pipeline_compatibility(_assignment()):
        assert pipeline_assignment_is_honored(model)


def test_minimax_declares_its_wrapped_assigned_stage_contract(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "minimax_m3_vl")
    maybe_apply_pre_load_patches(str(model))

    assert pipeline_assignment_is_honored(model)


# -- GLM-5.2 / Deepseek-V3.2 vendored model: unequal plan must reach the loader


def _fake_group(rank: int, size: int):
    from types import SimpleNamespace

    return SimpleNamespace(rank=lambda: rank, size=lambda: size)


def _placeholder_v32_model(layer_count: int = 78):
    """DeepseekV32Model shell with fake layers — no weights, no MLX tensors."""

    from omlx.patches.glm_moe_dsa.deepseek_v32 import DeepseekV32Model

    model = DeepseekV32Model.__new__(DeepseekV32Model)
    model.layers = [object() for _ in range(layer_count)]
    model.start_idx = 0
    model.end_idx = layer_count
    model.num_layers = layer_count
    model.pipeline_rank = 0
    model.pipeline_size = 1
    return model


def _unequal_78():
    return (
        PipelineAssignment(
            node_id="rank0",
            rank=0,
            start_layer=26,
            end_layer=78,
            layer_weight_bytes=52,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=128,
        ),
        PipelineAssignment(
            node_id="rank1",
            rank=1,
            start_layer=0,
            end_layer=26,
            layer_weight_bytes=26,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=64,
        ),
    )


def test_glm_moe_dsa_vendored_model_honors_the_unequal_plan(tmp_path):
    """The reported defect: a 26/52 plan loads as an even 39/39 split.

    The vendored DeepseekV32Model.pipeline() computes its own even split
    from rank/world size alone, discarding the approved assignment; the
    post-load validator then refuses activation after a full 432 GB load.
    The compatibility hook must make the real loader honor the plan.
    """

    model = _placeholder_v32_model(78)
    with install_pipeline_compatibility(_unequal_78()):
        model.pipeline(_fake_group(rank=0, size=2))

    assert (model.start_idx, model.end_idx) == (26, 78)
    assert model.num_layers == 52
    # Prefix layers are placeholders, assigned layers keep their identity.
    assert all(layer is None for layer in model.layers[:26])
    assert all(layer is not None for layer in model.layers[26:])


def test_glm_moe_dsa_second_rank_gets_its_own_range(tmp_path):
    model = _placeholder_v32_model(78)
    with install_pipeline_compatibility(_unequal_78()):
        model.pipeline(_fake_group(rank=1, size=2))

    assert (model.start_idx, model.end_idx) == (0, 26)
    assert model.num_layers == 26


def test_glm_moe_dsa_declares_the_assignment_contract_once_patched(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "glm_moe_dsa")

    # The GLM pre-load patch wraps mlx_lm.generate's PromptProcessingBatch /
    # BatchGenerator / generate_step process-globally with no unapply. A
    # later test asserting install_runtime_optimizations' prompt contract
    # (test_cluster_performance) then fails when this file runs first.
    # Snapshot and restore so the test leaves the process as it found it.
    from mlx_lm import generate as _generate_fn  # noqa: F401 - name shadow note
    import importlib

    # ``mlx_lm.generate`` the attribute is the function; the module holds the
    # patched classes.
    mlx_generate = importlib.import_module("mlx_lm.generate")

    snapshot = {}
    for owner, attrs in (
        (mlx_generate.PromptProcessingBatch, ("__init__", "_copy", "split", "prompt")),
        (mlx_generate.BatchGenerator, ("__init__", "_next")),
    ):
        for attr in attrs:
            snapshot[(owner, attr)] = getattr(owner, attr)
    snapshot[(mlx_generate, "generate_step")] = mlx_generate.generate_step
    try:
        maybe_apply_pre_load_patches(str(model))

        assert not pipeline_assignment_is_honored(model)
        with install_pipeline_compatibility(_unequal_78()):
            assert pipeline_assignment_is_honored(model)
    finally:
        for (owner, attr), original in snapshot.items():
            setattr(owner, attr, original)


def test_glm_moe_dsa_without_compatibility_keeps_its_native_even_split():
    # Outside the worker's compatibility context the vendored method is
    # untouched: local single-node and upstream behavior are unchanged.
    model = _placeholder_v32_model(78)
    model.pipeline(_fake_group(rank=0, size=2))

    assert (model.start_idx, model.end_idx) == (39, 78)


def test_glm_moe_dsa_odd_counts_and_three_ranks_follow_the_plan():
    assignments = (
        PipelineAssignment(
            node_id="r0",
            rank=0,
            start_layer=40,
            end_layer=79,
            layer_weight_bytes=39,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=128,
        ),
        PipelineAssignment(
            node_id="r1",
            rank=1,
            start_layer=13,
            end_layer=40,
            layer_weight_bytes=27,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=96,
        ),
        PipelineAssignment(
            node_id="r2",
            rank=2,
            start_layer=0,
            end_layer=13,
            layer_weight_bytes=13,
            fixed_weight_bytes=1,
            reserve_bytes=1,
            capacity_bytes=64,
        ),
    )
    expected = {0: (40, 79), 1: (13, 40), 2: (0, 13)}
    with install_pipeline_compatibility(assignments):
        for rank, (start, end) in expected.items():
            model = _placeholder_v32_model(79)
            model.pipeline(_fake_group(rank=rank, size=3))
            assert (model.start_idx, model.end_idx) == (start, end)
            assert model.num_layers == end - start
