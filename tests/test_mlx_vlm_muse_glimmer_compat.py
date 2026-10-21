# SPDX-License-Identifier: Apache-2.0
"""Muse Glimmer mlx-vlm compatibility patch tests.

Covers the vendor install/discovery surface (inkling test pattern), the
model behaviors oMLX depends on (mixed sliding/full cache, NoPE layers,
logit tail, vision-cache encode_image contract), the PR #1839 quantized
embedding-norm preservation, and the real checkpoint's chat template
contract (dict tool arguments, to=self reasoning, reasoning_strength).
"""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

_CHECKPOINT = Path("~/Workspace/models/meta-models/Muse-Glimmer-30B").expanduser()


@pytest.fixture(scope="module")
def applied():
    from omlx.patches.mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
        is_applied,
    )

    apply_mlx_vlm_muse_glimmer_compat_patch()
    assert is_applied()
    return True


def _tiny_config():
    from mlx_vlm.models.muse_glimmer.config import (
        ModelConfig,
        TextConfig,
        VisionConfig,
    )

    text = TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=128,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
    )
    vision = VisionConfig(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_hidden_layers=2,
        patch_size=2,
        patch_temporal=2,
        merge_size=2,
        pos_emb_height=4,
        pos_emb_width=4,
        max_position_embeddings=16,
        layer_types=["window_attention", "full_attention"],
    )
    return ModelConfig(
        text_config=text,
        vision_config=vision,
        image_token_id=7,
        video_token_id=6,
        out_hidden_size=32,
        projector_hidden_size=16,
    )


def test_vendor_module_resolves(applied):
    import importlib

    import mlx_vlm.utils as vlm_utils

    module = importlib.import_module("mlx_vlm.models.muse_glimmer")
    assert hasattr(module, "Model")
    assert hasattr(module, "LanguageModel")

    result = vlm_utils.get_model_and_args({"model_type": "muse_glimmer"})
    assert result[0] is module
    assert result[1] == "muse_glimmer"


def test_double_apply_is_noop(applied):
    from omlx.patches.mlx_vlm_muse_glimmer_compat import (
        apply_mlx_vlm_muse_glimmer_compat_patch,
    )

    assert apply_mlx_vlm_muse_glimmer_compat_patch() is False


def test_prompt_formatting_image_first(applied):
    from mlx_vlm.prompt_utils import MODEL_CONFIG, get_message_json

    assert "muse_glimmer" in MODEL_CONFIG

    message = get_message_json(
        "muse_glimmer", "describe this", role="user", num_images=2
    )
    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"
    assert content[1]["type"] == "image"
    assert content[2]["type"] == "text"


def test_other_models_untouched(applied):
    from mlx_vlm.prompt_utils import get_message_json

    message = get_message_json("qwen2_5_vl", "hello", role="user", num_images=1)
    assert message["role"] == "user"


def test_image_processor_patch_layout_and_grid(applied):
    import numpy as np
    from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
        MuseGlimmerImageProcessor,
        smart_resize,
    )
    from PIL import Image

    assert smart_resize(28, 56, patch_size=28, max_tokens=4096) == (28, 56)
    processor = MuseGlimmerImageProcessor(
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        max_image_tokens=4096,
    )
    output = processor(Image.new("RGB", (28, 28), (255, 0, 0)))
    assert output["pixel_values"].shape == (4, 1176)
    assert output["image_grid_thw"].tolist() == [[1, 2, 2]]
    # Temporal copies are adjacent within each flattened patch.
    first = output["pixel_values"][0].reshape(2, 3, 14, 14)
    np.testing.assert_array_equal(first[0], first[1])


def test_nope_layers_skip_rope(applied):
    from mlx_vlm.models.muse_glimmer import Model

    model = Model(_tiny_config())
    layers = model.language_model.model.layers
    assert layers[0].self_attn.use_rope is True
    assert layers[1].self_attn.use_rope is False


def test_cache_matches_layer_attention_type(applied):
    from mlx_vlm.models.cache import KVCache, RotatingKVCache
    from mlx_vlm.models.muse_glimmer import Model

    caches = Model(_tiny_config()).make_cache()
    assert isinstance(caches[0], RotatingKVCache)
    assert caches[0].max_size == 8
    assert isinstance(caches[1], KVCache)


def test_tiny_text_forward_and_logit_tail(applied):
    from mlx_vlm.models.muse_glimmer import Model

    mx.random.seed(0)
    model = Model(_tiny_config())
    ids = mx.array([[1, 2, 3]])

    output = model.language_model(ids)
    mx.eval(output.logits)
    assert output.logits.shape == (1, 3, 64)
    assert bool(mx.isfinite(output.logits).all().item())

    # The logit tail is lm_head -> output_multiplier -> tanh softcap.
    lm = model.language_model
    hidden = lm.model(ids)
    expected = lm.lm_head(hidden) * lm.output_multiplier
    expected = mx.tanh(expected / lm.final_logit_softcapping)
    expected = expected * lm.final_logit_softcapping
    assert bool(mx.allclose(output.logits, expected))
    assert float(mx.abs(output.logits).max()) <= lm.final_logit_softcapping


def test_sliding_window_decode_past_window(applied):
    from mlx_vlm.models.muse_glimmer import Model

    mx.random.seed(0)
    model = Model(_tiny_config())
    cache = model.make_cache()
    # Prefill past the window (8), then decode one token.
    prompt = mx.array([[i % 60 for i in range(24)]])
    out = model.language_model(prompt, cache=cache)
    mx.eval(out.logits)
    step = model.language_model(mx.array([[5]]), cache=cache)
    mx.eval(step.logits)
    assert step.logits.shape == (1, 1, 64)
    assert bool(mx.isfinite(step.logits).all().item())


def test_multimodal_forward_and_masked_scatter(applied):
    from mlx_vlm.models.muse_glimmer import Model
    from mlx_vlm.models.muse_glimmer.muse_glimmer import masked_scatter

    mx.random.seed(0)
    model = Model(_tiny_config())

    # A 2x2 raw patch grid is pixel-shuffled into one visual token.
    pixels = mx.zeros((4, 2 * 3 * 2 * 2), dtype=mx.float32)
    grid = mx.array([[1, 2, 2]])
    embeddings = model.get_input_embeddings(
        mx.array([[1, 7, 2]]), pixels, image_grid_thw=grid
    ).inputs_embeds
    mx.eval(embeddings)
    assert embeddings.shape == (1, 3, 16)
    assert bool(mx.isfinite(embeddings).all().item())

    inputs = mx.arange(12).reshape(1, 3, 4)
    mask = mx.broadcast_to(mx.array([[[False], [True], [False]]]), inputs.shape)
    source = mx.array([[20, 21, 22, 23]])
    output = masked_scatter(inputs, mask, source)
    mx.eval(output)
    assert output.tolist() == [[[0, 1, 2, 3], [20, 21, 22, 23], [8, 9, 10, 11]]]


def test_encode_image_matches_get_input_embeddings(applied):
    from mlx_vlm.models.muse_glimmer import Model

    mx.random.seed(0)
    model = Model(_tiny_config())
    pixels = mx.random.normal((4, 2 * 3 * 2 * 2)).astype(mx.float32)
    grid = mx.array([[1, 2, 2]])
    ids = mx.array([[1, 7, 2]])

    features = model.encode_image(pixels, image_grid_thw=grid)
    mx.eval(features)

    direct = model.get_input_embeddings(ids, pixels, image_grid_thw=grid)
    replayed = model.get_input_embeddings(
        ids, pixels, image_grid_thw=grid, cached_image_features=features
    )
    assert bool(mx.allclose(direct.inputs_embeds, replayed.inputs_embeds))


def test_centered_rms_norm_preserves_transformers_fp32_order(applied):
    # Ported from mlx-vlm PR #1838 (commit edfb0ef1): the centered scale is
    # applied in FP32 before the single downcast. Casting earlier (the old
    # mx.fast.rms_norm(x, 1+w) form) shifts ~39% of bf16 outputs by up to
    # 4 ulps on real weights, enough to flip near-tie decode choices.
    from mlx_vlm.models.muse_glimmer.language import CenteredRMSNorm

    norm = CenteredRMSNorm(4, eps=1e-6)
    norm.weight = (mx.arange(4, dtype=mx.float32) * 0.031 - 0.2).astype(mx.bfloat16)
    inputs = (
        (mx.arange(4, dtype=mx.float32) * 0.37 - 1.13).reshape(1, 4).astype(mx.bfloat16)
    )

    inputs32 = inputs.astype(mx.float32)
    variance = mx.mean(mx.square(inputs32), axis=-1, keepdims=True)
    expected = inputs32 * mx.rsqrt(variance + 1e-6)
    expected = expected * (1.0 + norm.weight.astype(mx.float32))
    expected = expected.astype(mx.bfloat16)

    assert bool(mx.array_equal(norm(inputs), expected))


def test_quantization_preserves_embedding_norm(applied):
    import mlx.nn as nn
    from mlx_vlm.models.muse_glimmer.config import TextConfig
    from mlx_vlm.models.muse_glimmer.language import TextModel

    # Upstream design (mlx-vlm #1848): embed_tokens is a plain nn.Embedding
    # and embed_norm is a separate weightless module, so quantizing the
    # embedding cannot drop the norm (unlike the PR #1839 NormedEmbedding
    # wrapper this test originally guarded). Verify the norm survives
    # quantization on the real module structure and the forward path
    # (embed_norm(embed_tokens(x))) still yields unit-RMS embeddings.
    model = TextModel(
        TextConfig(
            rms_norm_eps=1e-5,
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=32,
            sliding_window=8,
        )
    )
    # A deliberately large scale: only the norm brings this back to unit RMS.
    model.embed_tokens.weight = mx.random.normal((64, 64)) * 5.0

    ids = mx.array([[1, 2, 3]])
    reference = model.embed_norm(model.embed_tokens(ids))
    mx.eval(reference)

    nn.quantize(model, group_size=32, bits=8)

    quantized_embed = model.embed_tokens
    assert isinstance(quantized_embed, nn.QuantizedEmbedding)
    # The norm is a separate module, not swallowed by quantization.
    assert isinstance(model.embed_norm, nn.Module)

    quantized = model.embed_norm(model.embed_tokens(ids))
    mx.eval(quantized)
    assert abs(float(mx.sqrt(mx.mean(reference**2))) - 1.0) < 1e-2
    assert abs(float(mx.sqrt(mx.mean(quantized**2))) - 1.0) < 1e-2


def test_sanitize_maps_checkpoint_prefixes(applied):
    from mlx_vlm.models.muse_glimmer import Model

    model = Model(_tiny_config())
    weights = model.sanitize(
        {
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros((16, 16)),
            "model.vision_tower.ln_pre.weight": mx.ones((8,)),
            "lm_head.weight": mx.zeros((64, 16)),
        }
    )
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in weights
    assert "vision_tower.ln_pre.weight" in weights
    assert "language_model.lm_head.weight" in weights


# Muse Spark reuses the Glimmer vision tower and processor from this vendor
# package, and adds the oMLX mlx-lm text model (omlx.patches.muse_spark).

_SPARK_NVFP4_LAYER = 1


def _spark_text_kwargs(**overrides):
    kwargs = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        layer_rope_theta=[10000.0, 0],
        mlp_layer_types=["sparse", "sparse"],
        moe_hidden_size=32,
        moe_intermediate_size=32,
        num_local_experts=4,
        num_experts_per_tok=2,
    )
    kwargs.update(overrides)
    return kwargs


def _spark_modelopt_config():
    # Like the release: layer 0 keeps BF16 experts, the rest are NVFP4.
    return {
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
        "group_size": 16,
        "modules_to_not_convert": ["model.language_model.layers.0.mlp.experts"],
    }


def _spark_nvfp4_experts(num_experts, hidden, dims):
    """ModelOpt NVFP4 tensors for one fused gate_up and one down projection."""
    mx.random.seed(1)
    out = {}
    dense = {}
    for proj, rows, cols, n_scales in (
        ("gate_up_proj", 2 * hidden, dims, 2),
        ("down_proj", dims, hidden, 1),
    ):
        packed = mx.random.randint(0, 256, (num_experts, rows, cols // 2)).astype(
            mx.uint8
        )
        # E4M3 codes for 64 .. 448, the block scale range of the release.
        scales = mx.random.randint(0x68, 0x7F, (num_experts, rows, cols // 16)).astype(
            mx.uint8
        )
        # Release tensor scales are about 1e-5.
        scale2 = mx.random.uniform(1e-5, 4e-5, (num_experts, n_scales))
        weight = mx.dequantize(
            packed.view(mx.uint32), scales, group_size=16, bits=4, mode="nvfp4"
        )
        weight = weight * mx.repeat(scale2, rows // n_scales, axis=1)[..., None]
        out[proj] = packed
        out[f"{proj}_weight_scale"] = scales
        out[f"{proj}_weight_scale_2"] = scale2 if n_scales == 2 else scale2[:, 0]
        out[f"{proj}_input_scale"] = mx.ones(scale2.shape)
        dense[proj] = weight
    return out, dense


def _spark_hf_experts(weights):
    """Fold mlx-lm ``switch_mlp`` tensors back into the HF expert tensors."""
    out = {}
    for key, value in weights.items():
        if ".mlp.switch_mlp." in key:
            prefix, proj = key.split(".mlp.switch_mlp.")
            experts = f"{prefix}.mlp.experts."
            if proj == "down_proj.weight":
                out[experts + "down_proj"] = value
            elif proj == "gate_proj.weight":
                up = weights[f"{prefix}.mlp.switch_mlp.up_proj.weight"]
                out[experts + "gate_up_proj"] = mx.concatenate([value, up], axis=1)
            continue
        out[key.replace(".mlp.post_expert_norm.", ".mlp.experts.post_expert_norm.")] = (
            value
        )
    return out


def _spark_hf_weights(model):
    """Weights of a text ``model`` under the HF checkpoint names."""
    from mlx.utils import tree_flatten

    weights = {}
    for key, value in tree_flatten(model.parameters()):
        if key.startswith("model."):
            key = "model.language_model." + key[len("model.") :]
        weights[key] = value
    return _spark_hf_experts(weights)


@pytest.fixture(scope="module")
def spark_applied(applied):
    from omlx.patches.mlx_vlm_muse_spark_compat import (
        apply_mlx_vlm_muse_spark_compat_patch,
        is_applied,
    )
    from omlx.patches.muse_spark import apply_muse_spark_patch

    apply_muse_spark_patch()
    apply_mlx_vlm_muse_spark_compat_patch()
    assert is_applied()
    return True


def test_muse_spark_modules_resolve(spark_applied):
    import importlib

    import mlx_lm.utils as lm_utils
    import mlx_vlm.utils as vlm_utils
    from mlx_vlm.prompt_utils import MODEL_CONFIG

    from omlx.patches.muse_spark import muse_spark_model

    model_class, _ = lm_utils._get_classes({"model_type": "muse_spark"})
    assert model_class is muse_spark_model.Model
    module = importlib.import_module("mlx_vlm.models.muse_spark")
    assert vlm_utils.get_model_and_args({"model_type": "muse_spark"})[0] is module
    assert "muse_spark" in MODEL_CONFIG


@pytest.mark.parametrize("for_vlm", [False, True])
def test_preload_registers_muse_spark(spark_applied, tmp_path, for_vlm):
    import sys

    from omlx.patches.muse_spark import muse_spark_model
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text('{"model_type": "muse_spark"}')
    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=for_vlm)
    assert sys.modules["mlx_lm.models.muse_spark"] is muse_spark_model


def test_muse_spark_nvfp4_experts_match_dequantized_weights(spark_applied):
    """ModelOpt NVFP4 experts load as-is and apply the per-expert tensor scales.

    The NVFP4 model must match a float model that holds the dequantized
    weights, also with FP16 activations: unscaled NVFP4 matmul outputs are
    about 1e5 times too large and overflow FP16.
    """

    from mlx.utils import tree_flatten

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs

    mx.random.seed(0)
    dense = Model(ModelArgs(**_spark_text_kwargs()))
    quant = Model(
        ModelArgs(**_spark_text_kwargs(), quantization_config=_spark_modelopt_config())
    )

    nvfp4, dequantized = _spark_nvfp4_experts(4, 32, 32)
    prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.experts."
    hf = _spark_hf_weights(dense)
    for name, value in dequantized.items():
        hf[prefix + name] = value
    dense.load_weights(list(dense.sanitize(hf).items()), strict=True)
    # Without tensor scales in the checkpoint the layout is the mlx-lm one.
    assert not any("global_scale" in k for k, _ in tree_flatten(dense.parameters()))

    for name in ("gate_up_proj", "down_proj"):
        hf.pop(prefix + name)
    hf.update({prefix + name: value for name, value in nvfp4.items()})
    quant.load_weights(list(quant.sanitize(hf).items()), strict=True)
    switch = quant.model.layers[_SPARK_NVFP4_LAYER].mlp.switch_mlp
    assert switch.gate_proj.mode == "nvfp4" and "global_scale" in switch.gate_proj
    assert "global_scale" not in quant.model.layers[0].mlp.switch_mlp.gate_proj

    ids = mx.array([[1, 5, 9, 2, 33, 7, 11, 3, 60, 4, 8, 12]])
    expected, actual = dense(ids), quant(ids)
    mx.eval(expected, actual)
    assert float(mx.abs(expected - actual).max()) < 1e-3
    assert float(mx.abs(expected).max()) > 1e-2

    # oQ outputs list the passed-through NVFP4 experts in their quantization
    # map, so the loader asks to quantize those layers again after sanitize.
    import mlx.nn as nn

    oq = Model(ModelArgs(**_spark_text_kwargs()))
    weights = oq.sanitize(hf)
    entry = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    nn.quantize(
        oq,
        class_predicate=lambda p, _: entry if f"{p}.global_scale" in weights else False,
    )
    oq.load_weights(list(weights.items()), strict=True)
    assert mx.array_equal(oq(ids), actual).item()

    def to_fp16(path, value):
        if "global_scale" in path or value.dtype != mx.float32:
            return value
        return value.astype(mx.float16)

    from mlx.utils import tree_map_with_path

    quant.update(tree_map_with_path(to_fp16, quant.parameters()))
    half = quant(ids)
    assert half.dtype == mx.float16
    assert float(mx.abs(half.astype(mx.float32) - expected).max()) < 0.2


@pytest.mark.parametrize(
    "scaled",
    [("gate_proj", "up_proj", "down_proj"), ("down_proj",), ("up_proj",)],
)
def test_muse_spark_requantized_experts_keep_tensor_scale(spark_applied, scaled):
    """Affine experts may keep their NVFP4 units plus ``global_scale``.

    Older oQ outputs keep the tensor scale on every projection. oQ now folds it
    into the experts it requantizes, so only the projections it passes through
    as NVFP4 keep one. The loader quantizes them by config, and each tensor
    scale must survive that.
    """
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs

    mx.random.seed(0)
    dense = Model(ModelArgs(**_spark_text_kwargs()))
    weights = dict(tree_flatten(dense.parameters()))
    switch = f"model.layers.{_SPARK_NVFP4_LAYER}.mlp.switch_mlp."
    tensor_scale = mx.random.uniform(1e-5, 4e-5, (4,))
    for proj in ("gate_proj", "up_proj", "down_proj"):
        weight = weights.pop(f"{switch}{proj}.weight")
        if proj in scaled:
            weight = weight / tensor_scale[:, None, None]
            weights[f"{switch}{proj}.global_scale"] = tensor_scale * 6.0 * 448.0
        packed, scales, biases = mx.quantize(weight, group_size=32, bits=8)
        weights[f"{switch}{proj}.weight"] = packed
        weights[f"{switch}{proj}.scales"] = scales
        weights[f"{switch}{proj}.biases"] = biases

    # The mlx-lm loader order: sanitize, quantize by config, then load.
    model = Model(ModelArgs(**_spark_text_kwargs()))
    weights = model.sanitize(weights)
    nn.quantize(
        model,
        group_size=32,
        bits=8,
        class_predicate=lambda path, _: f"{path}.scales" in weights,
    )
    model.load_weights(list(weights.items()), strict=True)
    switch_mlp = model.model.layers[_SPARK_NVFP4_LAYER].mlp.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch_mlp, name)
        assert proj.mode == "affine"
        assert ("global_scale" in proj) == (name in scaled)

    ids = mx.array([[1, 5, 9, 2, 33, 7, 11, 3, 60, 4, 8, 12]])
    expected, actual = dense(ids), model(ids)
    mx.eval(expected, actual)
    assert float(mx.abs(expected - actual).max()) < 1e-2


def test_muse_spark_gate_up_fusion_is_bit_exact(spark_applied):
    import mlx.nn as nn

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs
    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(0)
    model = Model(ModelArgs(**_spark_text_kwargs()))
    nn.quantize(model, group_size=32, bits=4)
    # Decode rows run unsorted, the 40-token prompt takes the sorted path.
    prompts = [mx.array([[3]]), mx.array([[i % 60 for i in range(40)]])]
    before = [model(ids) for ids in prompts]
    mx.eval(before)

    assert apply_qwen35_moe_gate_up_fusion(model) == 2
    assert "gate_up_proj" in model.model.layers[0].mlp.switch_mlp
    after = [model(ids) for ids in prompts]
    mx.eval(after)
    for b, a in zip(before, after):
        assert mx.array_equal(b, a).item()


def test_muse_spark_norm_kernels_match_reference_ops(spark_applied):
    """The fused norm kernels round where the composed reference ops do.

    The reductions and the residual gate may still differ by one unit in the
    last place, so the check allows rare one-ulp differences.
    """
    from omlx.patches.muse_spark import norm_kernels as nk

    def reference_norm(x, w, eps):
        x32 = x.astype(mx.float32)
        x32 = x32 * mx.rsqrt(mx.mean(mx.square(x32), axis=-1, keepdims=True) + eps)
        return (x32 * w.astype(mx.float32)).astype(x.dtype)

    def input_scaled(x, w, eps):
        x = (x.astype(mx.float32) * (1.0 + w).astype(mx.float32)).astype(x.dtype)
        return mx.fast.rms_norm(x, None, eps)

    def close(actual, expected):
        diff = mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32))
        ulp = mx.abs(expected.astype(mx.float32)) * 2.0**-7 + 1e-6
        assert bool((diff <= ulp).all().item())
        assert (diff == 0).astype(mx.float32).mean().item() > 0.999

    mx.random.seed(0)
    x = (mx.random.normal((1, 3, 8192)) * 2).astype(mx.bfloat16)
    a = (mx.random.normal((1, 3, 8192)) * 3).astype(mx.bfloat16)
    w = (mx.random.normal((8192,)) * 0.1).astype(mx.bfloat16)
    gate = mx.random.normal((8192,)) / 0.3
    beta = mx.sigmoid(gate)
    alpha = mx.sqrt(mx.maximum(mx.sigmoid(-gate) * (1.0 + beta), 1e-3))

    close(nk.scaled_rms(x, w, 1e-5, centered=False), reference_norm(x, w, 1e-5))
    close(nk.scaled_rms(x, w, 1e-5, centered=True), reference_norm(x, 1.0 + w, 1e-5))

    gated = (alpha * x + beta * mx.fast.rms_norm(a, None, 1e-8)).astype(x.dtype)
    x_out, n_out = nk.post_attention(x, a, alpha, beta, w, 1e-8, 1e-5)
    close(x_out, gated)
    # The norm stage on the kernel's own gate output.
    close(n_out, reference_norm(x_out, 1.0 + w, 1e-5))

    y = (mx.random.normal((1, 3, 8, 4096)) * 4).astype(mx.bfloat16)
    w4 = (mx.random.normal((4096,)) * 0.1).astype(mx.bfloat16)
    scores = mx.softmax(mx.random.normal((1, 3, 8)), axis=-1)
    expected = (input_scaled(y, w4, 1e-8).astype(mx.float32) * scores[..., None]).sum(
        axis=-2
    )
    close(nk.moe_tail(y, scores, w4, 1e-8), expected.astype(mx.bfloat16))

    expected = (alpha * x + beta * input_scaled(a, w, 1e-8)).astype(x.dtype)
    close(nk.post_moe(x, a, alpha, beta, w, 1e-8), expected)


def test_muse_spark_sliding_layers_use_rotating_cache(spark_applied):
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs

    mx.random.seed(0)
    model = Model(ModelArgs(**_spark_text_kwargs()))
    cache = model.make_cache()
    assert isinstance(cache[0], RotatingKVCache) and cache[0].max_size == 8
    assert isinstance(cache[1], KVCache)

    # Chunked prefill past the window plus a decode step matches one full pass.
    prompt = mx.array([[i % 60 for i in range(21)]])
    full = model(prompt)
    for start in range(0, 20, 5):
        model(prompt[:, start : start + 5], cache=cache)
    step = model(prompt[:, 20:], cache=cache)
    mx.eval(full, step)
    assert float(mx.abs(full[:, -1] - step[:, -1]).max()) < 1e-4


def _spark_vlm_config(quantization_config=None):
    from mlx_vlm.models.muse_spark import ModelConfig

    return ModelConfig.from_dict(
        {
            "model_type": "muse_spark",
            "text_config": _spark_text_kwargs(model_type="muse_spark_text"),
            "vision_config": vars(_tiny_config().vision_config),
            "image_token_id": 7,
            "out_hidden_size": 32,
            "projector_hidden_size": 16,
            "quantization_config": quantization_config,
        }
    )


def test_muse_spark_vlm_loads_checkpoint_names(spark_applied):
    from mlx.utils import tree_flatten
    from mlx_vlm.models.muse_spark import Model

    model = Model(_spark_vlm_config(_spark_modelopt_config()))

    # HF-named weights from a float twin, with NVFP4 experts swapped in.
    mx.random.seed(0)
    twin = Model(_spark_vlm_config())
    hf = {}
    for key, value in tree_flatten(twin.parameters()):
        if key.startswith("language_model.model."):
            key = "model.language_model." + key[len("language_model.model.") :]
        elif key.startswith("language_model.lm_head."):
            key = key[len("language_model.") :]
        else:
            key = "model." + key
        hf[key] = value
    hf = _spark_hf_experts(hf)
    prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.experts."
    for name in ("gate_up_proj", "down_proj"):
        hf.pop(prefix + name)
    hf.update({prefix + k: v for k, v in _spark_nvfp4_experts(4, 32, 32)[0].items()})
    hf["model.audio_tower.proj.weight"] = mx.zeros((32, 8))

    model.load_weights(list(model.sanitize(hf).items()), strict=True)

    # A 2x2 raw patch grid is pixel-shuffled into one visual token.
    pixels = mx.random.normal((4, 2 * 3 * 2 * 2))
    grid = mx.array([[1, 2, 2]])
    out = model(mx.array([[1, 7, 2]]), pixel_values=pixels, image_grid_thw=grid)
    mx.eval(out.logits)
    assert out.logits.shape == (1, 3, 64)
    assert bool(mx.isfinite(out.logits).all().item())

    with pytest.raises(ValueError, match="pixel_values_videos"):
        model.get_input_embeddings(mx.array([[1]]), pixel_values_videos=pixels)


def _spark_randomize(model):
    """Random routers, residual gates and norm weights.

    Norm weights far from zero (gains below 0.5 included) make the layout's
    value transforms visible in the logits.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    mx.random.seed(2)
    updates = []
    for key, value in tree_flatten(model.parameters()):
        if "vision" in key:
            continue
        if key.endswith(("residual_gate.gate", "mlp.gate.weight")):
            updates.append((key, mx.random.normal(value.shape)))
        elif key.endswith(("norm.weight", "layernorm.weight")):
            weight = mx.random.normal(value.shape) * 0.5
            updates.append((key, weight.astype(mx.bfloat16)))
    model.update(tree_unflatten(updates))


def _spark_quantize(model, prefix):
    """Per-layer bits as oQ assigns them: one format per fused attention group."""
    import mlx.nn as nn

    def predicate(path, module):
        if not (hasattr(module, "to_quantized") and path.startswith(prefix)):
            return False
        if ".self_attn." in path and not path.endswith("o_proj"):
            bits = 4 if ".layers.0." in path else 8
            return {"group_size": 32, "bits": bits, "mode": "affine"}
        if path.endswith("lm_head"):
            return {"group_size": 32, "bits": 8, "mode": "affine"}
        return True

    nn.quantize(model, group_size=32, bits=3, class_predicate=predicate)


def _spark_save_upstream(model, path, config):
    """Save ``model`` in the upstream layout, one tensor at a time like oQ."""
    import json

    from mlx.utils import tree_flatten

    from omlx.patches.muse_spark.upstream_layout import UpstreamWriter

    writer = UpstreamWriter(config)
    weights = {}
    for key, value in tree_flatten(model.parameters()):
        weights.update(writer({key: value}))
    writer.finish()
    per_layer = {
        name: {"group_size": m.group_size, "bits": m.bits, "mode": m.mode}
        for name, m in model.named_modules()
        if hasattr(m, "bits") and m.bits != 3
    }
    quantization = {"group_size": 32, "bits": 3, "mode": "affine"}
    quantization.update(writer.quantization(per_layer))
    mx.save_safetensors(str(path / "model.safetensors"), weights)
    (path / "config.json").write_text(
        json.dumps({**config, "quantization": quantization})
    )
    return weights


def test_muse_spark_loads_upstream_layout(spark_applied, tmp_path):
    from mlx_lm.utils import load_model

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs

    config = {"model_type": "muse_spark", "text_config": _spark_text_kwargs()}
    model = Model(ModelArgs.from_dict(config))
    _spark_randomize(model)
    _spark_quantize(model, "")
    weights = _spark_save_upstream(model, tmp_path, config)
    assert "model.language_model.layers.1.self_attn.qkvg_proj.scales" in weights
    assert (
        "model.language_model.layers.0.mlp.experts.post_expert_norm.weight" in weights
    )
    assert "model.language_model.layers.0.add_ffn_beta" in weights
    assert not any("residual_gate" in key or "switch_mlp" in key for key in weights)

    # The per-module map uses upstream paths; the loader infers the bits.
    loaded, _ = load_model(tmp_path)
    assert loaded.model.layers[1].self_attn.q_proj.bits == 8
    ids = mx.array([[1, 5, 9, 2, 33, 60, 4]])
    expected, actual = model(ids), loaded(ids)
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


def test_muse_spark_vlm_loads_upstream_layout(spark_applied, tmp_path):
    from mlx_vlm.models.muse_spark import Model
    from mlx_vlm.utils import load_model

    config = {
        "model_type": "muse_spark",
        "text_config": _spark_text_kwargs(model_type="muse_spark_text"),
        "vision_config": vars(_tiny_config().vision_config),
        "image_token_id": 7,
        "out_hidden_size": 32,
        "projector_hidden_size": 16,
    }
    mx.random.seed(0)
    model = Model(_spark_vlm_config())
    _spark_randomize(model)
    _spark_quantize(model, "language_model.")
    weights = _spark_save_upstream(model, tmp_path, config)
    assert "model.vision_tower.layers.0.attn.qkv_proj.bias" in weights

    # mlx-vlm finds the per-module bits through the model's path aliases.
    loaded = load_model(tmp_path)
    assert loaded.language_model.model.layers[1].self_attn.gate_proj.bits == 8
    pixels = mx.random.normal((4, 2 * 3 * 2 * 2))
    grid = mx.array([[1, 2, 2]])
    ids = mx.array([[1, 7, 2, 9]])
    expected = model(ids, pixel_values=pixels, image_grid_thw=grid).logits
    actual = loaded(ids, pixel_values=pixels, image_grid_thw=grid).logits
    mx.eval(expected, actual)
    assert mx.array_equal(expected, actual).item()


def _spark_oq_source(path, nvfp4=False):
    """A small HF-named Muse Spark checkpoint for oQ.

    With ``nvfp4`` the experts of layer 1 are ModelOpt NVFP4, as in the
    release. Returns those NVFP4 tensors.
    """
    import json
    import struct

    from omlx.patches.muse_spark.muse_spark_model import Model, ModelArgs

    text = _spark_text_kwargs(
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        moe_hidden_size=64,
        moe_intermediate_size=64,
    )
    config = {"model_type": "muse_spark", "text_config": text}
    mx.random.seed(0)
    model = Model(ModelArgs.from_dict(config))
    _spark_randomize(model)
    weights = {k: v.astype(mx.bfloat16) for k, v in _spark_hf_weights(model).items()}
    experts = {}
    if nvfp4:
        config["quantization_config"] = _spark_modelopt_config()
        prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.experts."
        for name in ("gate_up_proj", "down_proj"):
            weights.pop(prefix + name)
        nvfp4_tensors = _spark_nvfp4_experts(4, 64, 64)[0]
        experts = {prefix + k: v for k, v in nvfp4_tensors.items()}
        weights.update(experts)
    path.mkdir()
    file = path / "model.safetensors"
    mx.save_safetensors(str(file), weights)
    if experts:
        # ModelOpt stores the block scales as F8_E4M3, which MLX cannot write.
        raw = file.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + size])
        for key in experts:
            if key.endswith("_weight_scale"):
                header[key]["dtype"] = "F8_E4M3"
        packed = json.dumps(header).encode()
        packed += b" " * (-len(packed) % 8)
        file.write_bytes(struct.pack("<Q", len(packed)) + packed + raw[8 + size :])
    (path / "config.json").write_text(json.dumps(config))
    return experts


def test_muse_spark_oq_writes_upstream_layout(spark_applied, tmp_path):
    import json

    import mlx.nn as nn
    from mlx_lm.utils import load_model

    from omlx.oq import quantize_oq_streaming

    src = tmp_path / "src"
    _spark_oq_source(src)

    out = tmp_path / "oq"
    quantize_oq_streaming(
        str(src), str(out), 4, sensitivity_map_override={0: 1.0, 1: 0.5}
    )

    saved = mx.load(str(out / "model.safetensors"))
    attn = "model.language_model.layers.0.self_attn."
    assert attn + "qkvg_proj.scales" in saved and attn + "q_proj.weight" not in saved
    assert "model.language_model.layers.1.add_attn_alpha" in saved
    # mlx-lm's Muse Spark needs a full-precision lm_head.
    assert "lm_head.scales" not in saved
    quantization = json.loads((out / "config.json").read_text())["quantization"]
    assert not any(".q_proj" in key or "switch_mlp" in key for key in quantization)

    loaded, _ = load_model(out)
    for layer in loaded.model.layers:
        attn = layer.self_attn
        bits = {p.bits for p in (attn.q_proj, attn.k_proj, attn.v_proj, attn.gate_proj)}
        assert len(bits) == 1
    assert isinstance(loaded.lm_head, nn.Linear)
    logits = loaded(mx.array([[1, 5, 9, 2, 33]]))
    assert bool(mx.isfinite(logits).all().item())


def test_muse_spark_oq_passes_nvfp4_experts_through(spark_applied, tmp_path):
    from mlx_lm.utils import load_model

    from omlx.oq import quantize_oq_streaming
    from omlx.patches.muse_spark.muse_spark_model import sanitize_experts

    src = tmp_path / "src"
    experts = _spark_oq_source(src, nvfp4=True)
    out = tmp_path / "oq"
    quantize_oq_streaming(
        str(src), str(out), 4, sensitivity_map_override={0: 1.0, 1: 0.5}
    )

    # Without a vision tower the output loads on the mlx-lm path, which does
    # not see the per-module map under the upstream names.
    loaded, _ = load_model(out)
    switch = loaded.model.layers[_SPARK_NVFP4_LAYER].mlp.switch_mlp
    expected = sanitize_experts(experts)
    prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.switch_mlp."
    for name in ("gate_proj", "up_proj", "down_proj"):
        proj = getattr(switch, name)
        assert proj.mode == "nvfp4"
        for param in ("weight", "scales", "global_scale"):
            assert mx.array_equal(proj[param], expected[f"{prefix}{name}.{param}"])
    logits = loaded(mx.array([[1, 5, 9, 2, 33]]))
    assert bool(mx.isfinite(logits).all().item())


@pytest.mark.skipif(
    not _CHECKPOINT.exists(), reason="Muse Glimmer checkpoint not available"
)
class TestRealChatTemplate:
    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(_CHECKPOINT))
        template = (_CHECKPOINT / "chat_template.jinja").read_text()
        tokenizer.chat_template = template
        return tokenizer

    def test_dict_tool_arguments_render_atem(self, tokenizer):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": {"city": "Seoul"},
                        },
                    }
                ],
            },
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        assert '<atem:invoke name="get_weather">' in rendered
        assert '<atem:parameter name="city">Seoul</atem:parameter>' in rendered
        assert "to=get_weather" in rendered

    def test_string_tool_arguments_raise(self, tokenizer):
        messages = [
            {"role": "user", "content": "weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Seoul"}',
                        },
                    }
                ],
            },
        ]
        with pytest.raises(Exception, match="mapping"):
            tokenizer.apply_chat_template(messages, tokenize=False)

    def test_reasoning_content_renders_to_self(self, tokenizer):
        messages = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "hello",
                "reasoning_content": "the user greets me",
            },
            {"role": "user", "content": "bye"},
        ]
        rendered = tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<|start|>assistant to=self<|message|>the user greets me" in rendered

    def test_reasoning_strength_kwarg(self, tokenizer):
        messages = [{"role": "user", "content": "hi"}]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            reasoning_strength="low",
        )
        assert "Reasoning strength: low." in rendered
        assert rendered.endswith("<|start|>assistant")
