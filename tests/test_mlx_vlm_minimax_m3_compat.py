# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the vendored MiniMax M3 mlx-vlm compatibility layer."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


def test_minimax_m3_compat_installs_vendor_modules():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    import mlx_vlm.models.minimax_m3  # noqa: F401
    import mlx_vlm.models.minimax_m3_vl  # noqa: F401
    import mlx_vlm.models.minimax_m3_vl.language as language
    import mlx_vlm.models.minimax_m3_vl.msa as msa
    import mlx_vlm.tool_parsers.minimax_m3 as parser

    assert hasattr(language, "MiniMaxM3KVCache")
    assert hasattr(msa, "build_grouped_msa_topk")
    assert hasattr(parser, "parse_tool_call")


def test_minimax_architecture_fallback_selects_text_model():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import get_model_and_args

    module, model_type = get_model_and_args(
        {
            "model_type": "qwen3",
            "architectures": ["MiniMaxM3SparseForCausalLM"],
        }
    )

    assert model_type == "minimax_m3"
    assert module.__name__ == "mlx_vlm.models.minimax_m3"


def test_minimax_vl_model_type_is_not_downgraded_by_architecture():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import get_model_and_args

    module, model_type = get_model_and_args(
        {
            "model_type": "minimax_m3_vl",
            "architectures": ["MiniMaxM3SparseForCausalLM"],
        }
    )

    assert model_type == "minimax_m3_vl"
    assert module.__name__ == "mlx_vlm.models.minimax_m3_vl"


def test_process_inputs_forwards_kwargs_to_var_kwargs_processor():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import process_inputs

    seen = {}

    class Processor:
        def __call__(
            self,
            text,
            images=None,
            padding=True,
            return_tensors="mlx",
            **kwargs,
        ):
            seen.update(kwargs)
            return {
                "input_ids": [[1]],
                "attention_mask": [[1]],
            }

    process_inputs(
        Processor(),
        prompts=["hello"],
        max_long_side_pixel=1024,
        return_mm_token_type_ids=True,
    )

    assert seen["max_long_side_pixel"] == 1024
    assert seen["return_mm_token_type_ids"] is True


def test_minimax_prompt_utils_restore_image_placeholders():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.prompt_utils import apply_chat_template, get_message_json

    message = get_message_json("minimax_m3_vl", "describe", num_images=2)
    assert message == {
        "role": "user",
        "content": "]<]image[>[" * 2 + "describe",
    }

    rendered_messages = apply_chat_template(
        processor=None,
        config={"model_type": "minimax_m3_vl"},
        prompt=[{"role": "user", "content": "describe"}],
        num_images=1,
        return_messages=True,
        enable_thinking=False,
    )
    assert rendered_messages == [
        {"role": "user", "content": "]<]image[>[describe"}
    ]


def test_stopping_criteria_accepts_none_eos_ids():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import StoppingCriteria

    criteria = StoppingCriteria(None)
    assert criteria.eos_token_ids == []


def test_minimax_quantization_compat_restores_mxfp8_and_skip_module(tmp_path):
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import load_config, skip_multimodal_module

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "quantization_config": {
                    "quant_method": "mxfp8",
                    "ignored_layers": ["vision_tower"],
                },
            }
        )
    )

    config = load_config(tmp_path)

    assert config["quantization"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    assert skip_multimodal_module("patch_merge_mlp.layers.0")


def test_ignored_layer_matching_covers_children():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        _is_ignored_layer,
    )

    assert _is_ignored_layer("vision_tower", ("vision_tower",))
    assert _is_ignored_layer("vision_tower.block", ("vision_tower",))
    assert not _is_ignored_layer("language_model.block", ("vision_tower",))


def _tiny_text_config(*, pack_shared_expert):
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    return TextConfig(
        hidden_size=64,
        intermediate_size=32,
        shared_intermediate_size=32,
        dense_intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        num_local_experts=2,
        num_experts_per_tok=1,
        n_shared_experts=1,
        moe_layer_freq=[1],
        use_routing_bias=False,
        pack_shared_expert=pack_shared_expert,
    )


def test_minimax_unpack_sanitizer_keeps_shared_expert_separate():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import _sanitize_moe_weights

    args = _tiny_text_config(pack_shared_expert=False)
    prefix = "language_model.model.layers.0.block_sparse_moe"
    weights = {}
    for expert in range(args.num_local_experts):
        weights[f"{prefix}.experts.{expert}.w1.weight"] = mx.zeros((32, 64))
        weights[f"{prefix}.experts.{expert}.w2.weight"] = mx.zeros((64, 32))
        weights[f"{prefix}.experts.{expert}.w3.weight"] = mx.zeros((32, 64))
    for name, shape in (
        ("gate_proj", (32, 64)),
        ("down_proj", (64, 32)),
        ("up_proj", (32, 64)),
    ):
        weights[f"{prefix}.shared_experts.{name}.weight"] = mx.zeros(shape)

    _sanitize_moe_weights(weights, args)
    sanitized = weights

    assert sanitized[f"{prefix}.switch_mlp.gate_proj.weight"].shape == (2, 32, 64)
    assert sanitized[f"{prefix}.switch_mlp.down_proj.weight"].shape == (2, 64, 32)
    assert sanitized[f"{prefix}.switch_mlp.up_proj.weight"].shape == (2, 32, 64)
    for name in ("gate_proj", "down_proj", "up_proj"):
        assert f"{prefix}.shared_experts.{name}.weight" in sanitized
    assert f"{prefix}.switch_mlp.gate_up_proj.weight" not in sanitized


def test_minimax_unpacked_mixed_bit_moe_forward():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_lm.models.switch_layers import QuantizedSwitchLinear
    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxSparseMoeBlock

    block = MiniMaxSparseMoeBlock(_tiny_text_config(pack_shared_expert=False))

    def predicate(path, module):
        if not hasattr(module, "to_quantized"):
            return False
        if path.startswith("switch_mlp"):
            return {"bits": 4, "group_size": 32, "mode": "affine"}
        if path.startswith("shared_experts"):
            return {"bits": 8, "group_size": 32, "mode": "affine"}
        return False

    nn.quantize(
        block,
        group_size=32,
        bits=4,
        mode="affine",
        class_predicate=predicate,
    )

    assert block.pack_shared_expert is False
    assert isinstance(block.switch_mlp.gate_proj, QuantizedSwitchLinear)
    assert block.switch_mlp.gate_proj.bits == 4
    assert isinstance(block.shared_experts.gate_proj, nn.QuantizedLinear)
    assert block.shared_experts.gate_proj.bits == 8

    output = block(mx.random.normal((1, 1, 64)).astype(mx.bfloat16))
    mx.eval(output)
    assert output.shape == (1, 1, 64)
    assert bool(mx.all(mx.isfinite(output)).item())


def test_omlx_loader_respects_minimax_shared_expert_layout_override():
    from omlx.engine.vlm import _should_pack_minimax_m3_shared_expert

    base = {
        "n_shared_experts": 1,
        "shared_intermediate_size": 32,
        "intermediate_size": 32,
    }
    assert _should_pack_minimax_m3_shared_expert(SimpleNamespace(**base))
    assert not _should_pack_minimax_m3_shared_expert(
        SimpleNamespace(**base, pack_shared_expert=False)
    )
    assert _should_pack_minimax_m3_shared_expert(
        SimpleNamespace(**base, pack_shared_expert=True)
    )
