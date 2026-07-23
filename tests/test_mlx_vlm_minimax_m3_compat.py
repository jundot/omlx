# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the vendored MiniMax M3 mlx-vlm compatibility layer."""

from __future__ import annotations

import json


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
