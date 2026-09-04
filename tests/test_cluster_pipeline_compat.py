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


def test_a_custom_unmarked_pipeline_remains_fail_closed(tmp_path):
    model = _model_config(tmp_path, "deepseek_v32")

    with install_pipeline_compatibility(_assignment()):
        assert not pipeline_assignment_is_honored(model)


def test_minimax_declares_its_wrapped_assigned_stage_contract(tmp_path):
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    model = _model_config(tmp_path, "minimax_m3_vl")
    maybe_apply_pre_load_patches(str(model))

    assert pipeline_assignment_is_honored(model)
