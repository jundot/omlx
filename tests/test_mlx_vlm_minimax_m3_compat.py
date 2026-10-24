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
    import mlx_vlm.tools.parsers.minimax_m3 as parser

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

    # The generic VLM formatter can pass ordered parts instead of a string.
    message = get_message_json(
        "minimax_m3_vl",
        [{"type": "text", "text": "Describe this."}, {"type": "image"}],
        num_images=1,
    )
    assert message == {"role": "user", "content": "]<]image[>[Describe this."}

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
    tokenizer = SimpleNamespace(eos_token_id=7)
    criteria = StoppingCriteria(None, tokenizer, additional_eos_token_ids=[9])
    assert criteria.eos_token_ids == [9]


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


def test_minimax_nvfp4_experts_get_per_layer_quantization(tmp_path):
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import load_config

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "text_config": {"num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
                "quantization_config": {
                    "quant_method": "mxfp8",
                    "group_size": 16,
                    "moe_quant_algo": "NVFP4",
                    "moe_quant_format": "nvfp4-pack-quantized",
                },
            }
        )
    )

    config = load_config(tmp_path)

    nvfp4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    prefix = "language_model.model.layers.1.block_sparse_moe.switch_mlp"
    assert config["quantization"] == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
        f"{prefix}.gate_proj": nvfp4,
        f"{prefix}.up_proj": nvfp4,
        f"{prefix}.down_proj": nvfp4,
    }
    assert config["text_config"]["pack_shared_expert"] is False
    assert config["text_config"]["routed_expert_scales"] is True


def test_minimax_mlx_lm_global_scale_flags_select_scaled_experts(tmp_path):
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.utils import load_config

    switch = "language_model.model.layers.1.block_sparse_moe.switch_mlp"
    nvfp4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    affine = {"group_size": 64, "bits": 3, "mode": "affine"}
    quantization = {
        "group_size": 64,
        "bits": 3,
        "mode": "affine",
        f"{switch}.gate_proj": affine,
        f"{switch}.down_proj": {**nvfp4, "global_scale": True},
    }
    (tmp_path / "minimax").mkdir()
    (tmp_path / "minimax" / "config.json").write_text(
        json.dumps(
            {
                "model_type": "minimax_m3_vl",
                "text_config": {"num_hidden_layers": 2, "moe_layer_freq": [0, 1]},
                "quantization": quantization,
                "quantization_config": quantization,
            }
        )
    )
    (tmp_path / "other").mkdir()
    (tmp_path / "other" / "config.json").write_text(
        json.dumps({"model_type": "muse_spark", "quantization": quantization})
    )

    config = load_config(tmp_path / "minimax")

    for key in ("quantization", "quantization_config"):
        assert config[key][f"{switch}.down_proj"] == nvfp4
        assert config[key][f"{switch}.gate_proj"] == affine
    assert config["text_config"]["pack_shared_expert"] is False
    assert config["text_config"]["routed_expert_scales"] is True
    # Other models load their global_scale experts themselves.
    other = load_config(tmp_path / "other")
    assert other["quantization"][f"{switch}.down_proj"]["global_scale"] is True


def test_minimax_global_scale_vectors_become_inverse_scales():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import _sanitize_moe_weights

    args = _tiny_text_config(
        pack_shared_expert=False, routed_expert_scales=True, num_local_experts=4
    )
    switch = "language_model.model.layers.0.block_sparse_moe.switch_mlp"
    global_scale = mx.array([15000.0, 21000.0, 33333.0, 42001.5])
    weights = {f"{switch}.down_proj.global_scale": global_scale}
    for proj in ("gate_proj", "up_proj", "down_proj"):
        weights[f"{switch}.{proj}.weight"] = mx.zeros((4, 32, 8), dtype=mx.uint32)
        weights[f"{switch}.{proj}.scales"] = mx.zeros((4, 32, 4), dtype=mx.uint8)

    _sanitize_moe_weights(weights, args)

    assert f"{switch}.down_proj.global_scale" not in weights
    assert mx.array_equal(weights[f"{switch}.down_scale"], global_scale / 2688.0).item()
    for name in ("gate", "up"):
        assert mx.array_equal(weights[f"{switch}.{name}_scale"], mx.ones((4,))).item()

    # oQ sanitizes logical expert weights, which must not gain ones vectors.
    logical = {
        f"{switch}.{proj}.weight": mx.zeros((4, 32, 64))
        for proj in ("gate_proj", "up_proj", "down_proj")
    }
    _sanitize_moe_weights(logical, args)
    assert not any(key.endswith("_scale") for key in logical)


def test_minimax_release_template_renders_thinking_mode():
    jinja2 = pytest.importorskip("jinja2")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        minimax_thinking_mode_template,
    )

    # Like the release template: no thinking_mode, generation prompt last.
    release = (
        "{%- set think_begin_token = '<mm:think>' -%}"
        "{{- 'user\\n' }}"
        "{%- if add_generation_prompt -%}{{- 'ai\\n' }}{%- endif -%}"
    )
    template = jinja2.Environment().from_string(minimax_thinking_mode_template(release))

    def render(**kwargs):
        return template.render(add_generation_prompt=True, **kwargs)

    assert render() == "user\nai\n"
    assert render(thinking_mode="adaptive") == "user\nai\n"
    assert render(thinking_mode="disabled") == "user\nai\n</mm:think>"
    assert render(thinking_mode="enabled") == "user\nai\n<mm:think>"
    assert (
        template.render(add_generation_prompt=False, thinking_mode="disabled")
        == "user\n"
    )
    # A template that renders thinking_mode itself stays as it is.
    earlier = release + "{{ thinking_mode }}"
    assert minimax_thinking_mode_template(earlier) == earlier


def test_ignored_layer_matching_covers_children():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        _is_ignored_layer,
    )

    assert _is_ignored_layer("vision_tower", ("vision_tower",))
    assert _is_ignored_layer("vision_tower.block", ("vision_tower",))
    assert not _is_ignored_layer("language_model.block", ("vision_tower",))


def _tiny_text_config(*, pack_shared_expert, **overrides):
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    params = dict(
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
    params.update(overrides)
    return TextConfig(**params)


def test_minimax_nvfp4_experts_match_dequantized_reference():
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    from mlx.utils import tree_flatten

    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.models.minimax_m3_vl.language import (
        MiniMaxSparseMoeBlock,
        MiniMaxSwiGLUOAI,
    )
    from mlx_vlm.models.minimax_m3_vl.minimax_m3_vl import _sanitize_moe_weights

    swiglu_oai = MiniMaxSwiGLUOAI()
    mx.random.seed(0)
    args = _tiny_text_config(
        pack_shared_expert=False,
        routed_expert_scales=True,
        use_routing_bias=True,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    prefix = "language_model.model.layers.0.block_sparse_moe"
    shapes = {"w1": (32, 64), "w2": (64, 32), "w3": (32, 64)}

    # Store each expert the way compressed-tensors does: FP4 codes of
    # w * G plus E4M3 block scales, with the real weight being dequant / G.
    weights, real = {}, {}
    for expert in range(args.num_local_experts):
        for name, shape in shapes.items():
            w = mx.random.normal(shape) * 0.05
            global_scale = 2688.0 / mx.abs(w).max()
            wq, scales = mx.quantize(
                w * global_scale, group_size=16, bits=4, mode="nvfp4"
            )
            base = f"{prefix}.experts.{expert}.{name}"
            weights[f"{base}.weight_packed"] = wq.view(mx.uint8)
            weights[f"{base}.weight_scale"] = scales
            weights[f"{base}.weight_global_scale"] = global_scale.reshape(())
            weights[f"{base}.input_global_scale"] = mx.array(1.0)
            real[(expert, name)] = (
                mx.dequantize(wq, scales, group_size=16, bits=4, mode="nvfp4")
                / global_scale
            )
    shared = {
        "gate_proj": mx.random.normal((32, 64)) * 0.05,
        "up_proj": mx.random.normal((32, 64)) * 0.05,
        "down_proj": mx.random.normal((64, 32)) * 0.05,
    }
    for name, w in shared.items():
        weights[f"{prefix}.shared_experts.{name}.weight"] = w
    router = mx.random.normal((args.num_local_experts, 64))
    bias = mx.random.normal((args.num_local_experts,)) * 0.1
    weights[f"{prefix}.gate.weight"] = router
    weights[f"{prefix}.e_score_correction_bias"] = bias

    _sanitize_moe_weights(weights, args)
    assert f"{prefix}.experts.0.w1.weight_packed" not in weights
    assert weights[f"{prefix}.switch_mlp.down_scale"].dtype == mx.float32

    block = MiniMaxSparseMoeBlock(args)
    nvfp4 = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    nn.quantize(
        block,
        class_predicate=lambda path, m: (
            nvfp4
            if path.startswith("switch_mlp.") and hasattr(m, "to_quantized")
            else False
        ),
    )
    block.load_weights(
        [(k[len(prefix) + 1 :], v) for k, v in weights.items()], strict=True
    )
    assert {k for k, _ in tree_flatten(block.parameters())} >= {
        "switch_mlp.gate_scale",
        "switch_mlp.up_scale",
        "switch_mlp.down_scale",
    }

    x = mx.random.normal((1, 3, 64))
    output = block(x.astype(mx.bfloat16)).astype(mx.float32)

    # Float32 reference with the exact dequantized weights.
    scores = mx.sigmoid(x @ router.T)
    inds = mx.argpartition(-(scores + bias), kth=1, axis=-1)[..., :2]
    weights_k = mx.take_along_axis(scores, inds, axis=-1)
    weights_k = weights_k / weights_k.sum(-1, keepdims=True) * 2.0
    expected = x @ shared["up_proj"].T
    expected = swiglu_oai(expected, x @ shared["gate_proj"].T) @ shared["down_proj"].T
    for t in range(3):
        for slot in range(2):
            expert = int(inds[0, t, slot].item())
            xt = x[0, t]
            h = swiglu_oai(xt @ real[(expert, "w3")].T, xt @ real[(expert, "w1")].T)
            routed = weights_k[0, t, slot] * (h @ real[(expert, "w2")].T)
            expected[0, t] = expected[0, t] + routed

    error = mx.abs(output - expected).max().item()
    assert error < 2e-2 * max(mx.abs(expected).max().item(), 1.0)


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


def test_minimax_m3_kv_cache_state_and_to_batch_accept_both_kv_classes():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()

    from mlx_vlm.models.minimax_m3_vl.language import MiniMaxM3KVCache

    keys = mx.random.normal((1, 2, 5, 4))
    values = mx.random.normal((1, 2, 5, 4))
    fresh = MiniMaxM3KVCache()
    fresh.update_and_fetch(keys, values)
    # extract() wraps the mlx_lm KVCache instead of the mlx_vlm one.
    extracted = fresh.to_batch([0]).extract(0)
    assert type(extracted.kv_cache) is not type(fresh.kv_cache)

    for cache in (fresh, extracted):
        (state_keys, state_values), _ = cache.state
        assert mx.array_equal(state_keys, keys)
        assert mx.array_equal(state_values, values)

        batch_keys, batch_values = cache.to_batch([2]).kv_cache.keys_and_values()
        assert batch_keys.shape[2] == 7
        assert mx.array_equal(batch_keys[..., 2:, :], keys)
        assert mx.array_equal(batch_values[..., 2:, :], values)


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


def test_minimax_dspark_runtime_returns_captures_and_rolls_back():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.language import LanguageModel

    args = _tiny_text_config(
        pack_shared_expert=False,
        num_hidden_layers=2,
        moe_layer_freq=[0, 0],
        vocab_size=64,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1, 1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 8,
            "sparse_topk_blocks": 2,
        },
    )
    lm = LanguageModel(args)
    cache = lm.make_cache()
    tokens = mx.array([[1, 2, 3, 4, 5]])

    out = lm(tokens, cache=cache, return_hidden=True, capture_layer_ids=[0])
    plain = lm(tokens, cache=lm.make_cache(), capture_layer_ids=[0, 1])

    assert len(out.hidden_states) == 2
    assert mx.allclose(out.hidden_states[0], plain.hidden_states[0]).item()
    assert mx.allclose(out.hidden_states[1], plain.hidden_states[1]).item()
    assert lm.mtp_partial_rollback(cache, 1, 3)
    assert cache[0].offset == 3
    assert cache[0].index_offset == 3


def test_minimax_batch_verify_rollback_keeps_each_rows_accepted_prefix():
    mx = pytest.importorskip("mlx.core")
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    from mlx_vlm.models.minimax_m3_vl.language import (
        LanguageModel,
        MiniMaxM3BatchKVCache,
    )

    args = _tiny_text_config(
        pack_shared_expert=False,
        num_hidden_layers=2,
        moe_layer_freq=[0, 0],
        vocab_size=64,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1, 1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 1,
            "sparse_block_size": 8,
            "sparse_topk_blocks": 2,
        },
    )
    lm = LanguageModel(args)
    # Rows of different lengths, merged with left padding like a generation batch.
    rows = []
    for n in (5, 9):
        cache = lm.make_cache()
        lm(mx.arange(1, n + 1)[None], cache=cache)
        rows.append(cache)
    batch = [
        MiniMaxM3BatchKVCache.merge([row[layer] for row in rows])
        for layer in range(len(rows[0]))
    ]
    block, accepted = 4, [2, 0]
    out = lm(
        mx.array([[11, 12, 13, 14], [21, 22, 23, 24]]),
        cache=batch,
        return_hidden=True,
    )
    assert out.gdn_states == []
    # Reference: each row's cache after the verify, trimmed on its own.
    expected = []
    for row, count in enumerate(accepted):
        caches = [layer.extract(row) for layer in batch]
        for cache in caches:
            cache.trim(block - count - 1)
        expected.append(caches)

    assert lm.rollback_speculative_cache(batch, out.gdn_states, accepted, block) == 2
    for row, caches in enumerate(expected):
        for layer, want in zip(batch, caches):
            got = layer.extract(row)
            assert got.offset == want.offset
            n = want.offset
            for name in ("keys", "values"):
                a = getattr(got.kv_cache, name)[..., :n, :]
                b = getattr(want.kv_cache, name)[..., :n, :]
                assert mx.array_equal(a, b).item()
            assert mx.array_equal(
                got.index_keys[..., : got.index_offset, :],
                want.index_keys[..., : want.index_offset, :],
            ).item()
