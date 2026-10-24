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


# Muse Spark runs on mlx-lm's model (omlx.patches.muse_spark.model). The
# vendored mlx-vlm package adds its processor, built on the Glimmer one.

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
        # mlx-lm's Muse Spark requires these.
        rms_norm_eps=1e-5,
        post_norm_eps=1e-8,
        rope_parameters={"rope_theta": 10000.0},
        final_logit_softcapping=20.0,
        output_multiplier=0.17677669529663687,
    )
    kwargs.update(overrides)
    return kwargs


def _spark_config(*, vision=False, **text):
    config = {"model_type": "muse_spark", "text_config": _spark_text_kwargs(**text)}
    if vision:
        config.update(
            vision_config=vars(_tiny_config().vision_config),
            image_token_id=7,
            out_hidden_size=32,
            projector_hidden_size=16,
        )
    return config


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


def _spark_hf_weights(config, seed=0):
    """Random weights for ``config`` under the HF checkpoint names.

    The residual gates, the routers and the zero-centered norms are random
    too, so that their value transforms show in the logits.
    """
    import re

    from mlx.utils import tree_flatten
    from mlx_lm.models import muse_spark

    from omlx.patches.muse_spark.oq_layout import deinterleave_rope_rows

    mx.random.seed(seed)
    model = muse_spark.Model(muse_spark.ModelArgs.from_dict(config))
    text, vision = model.args.text, model.args.vision
    q_rows = text.num_attention_heads * text.head_dim
    kv_rows = text.num_key_value_heads * text.head_dim
    gates = {
        "add_attn_alpha": "post_attention_residual_gate.gate",
        "add_ffn_alpha": "post_feedforward_residual_gate.gate",
    }
    params = dict(tree_flatten(model.parameters()))
    out = {}
    for key, value in params.items():
        layer = re.match(r"(model\.language_model\.layers\.\d+\.)(.+)$", key)
        prefix, name = layer.groups() if layer else ("", key)
        if name in gates:
            out[prefix + gates[name]] = mx.random.normal(value.shape)
        elif name == "mlp.gate.weight" or name in muse_spark._ZERO_CENTERED_NORMS:
            out[key] = mx.random.normal(value.shape) * 0.5
        elif name == "self_attn.qkvg_proj.weight":
            splits = [q_rows, q_rows + kv_rows, q_rows + 2 * kv_rows]
            q, k, v, g = mx.split(value, splits)
            attn = prefix + "self_attn."
            out[attn + "q_proj.weight"] = deinterleave_rope_rows(q, text.head_dim)
            out[attn + "k_proj.weight"] = deinterleave_rope_rows(k, text.head_dim)
            out[attn + "v_proj.weight"] = v
            out[attn + "gate_proj.weight"] = g
        elif name == "mlp.experts.gate_proj.weight":
            up = params[prefix + "mlp.experts.up_proj.weight"]
            out[prefix + "mlp.experts.gate_up_proj"] = mx.concatenate([value, up], 1)
        elif name == "mlp.experts.down_proj.weight":
            out[prefix + "mlp.experts.down_proj"] = value
        elif name.endswith("_beta") or name == "mlp.experts.up_proj.weight":
            continue
        elif ".attn.qkv_proj." in key:
            attn, param = key.split(".attn.qkv_proj.")
            q, k, v = mx.split(value, 3)
            out[f"{attn}.attn.q_proj.{param}"] = deinterleave_rope_rows(
                q, vision.head_dim
            )
            out[f"{attn}.attn.k_proj.{param}"] = deinterleave_rope_rows(
                k, vision.head_dim
            )
            out[f"{attn}.attn.v_proj.{param}"] = v
        else:
            out[key] = value
    return out


def _spark_model(config, weights):
    """The served model with HF ``weights`` loaded through mlx-lm's sanitize."""
    from mlx_lm.models import muse_spark

    from omlx.patches.muse_spark.model import Model

    model = Model(muse_spark.ModelArgs.from_dict(config))
    model.load_weights(list(model.sanitize(dict(weights)).items()), strict=True)
    return model


def _spark_save(path, config, weights):
    """Save a one-file checkpoint with the NVFP4 block scales as F8_E4M3."""
    import json
    import struct

    path.mkdir(exist_ok=True)
    file = path / "model.safetensors"
    mx.save_safetensors(str(file), weights)
    f8 = [key for key in weights if key.endswith("_weight_scale")]
    if f8:
        # MLX cannot write F8_E4M3, so relabel the uint8 codes.
        raw = file.read_bytes()
        size = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + size])
        for key in f8:
            header[key]["dtype"] = "F8_E4M3"
        packed = json.dumps(header).encode()
        packed += b" " * (-len(packed) % 8)
        file.write_bytes(struct.pack("<Q", len(packed)) + packed + raw[8 + size :])
    (path / "config.json").write_text(json.dumps(config))


@pytest.fixture(scope="module")
def spark_applied(applied):
    from omlx.patches.mlx_vlm_muse_spark_compat import (
        apply_mlx_vlm_muse_spark_compat_patch,
        is_applied,
    )

    apply_mlx_vlm_muse_spark_compat_patch()
    assert is_applied()
    return True


@pytest.fixture
def spark_load(spark_applied, monkeypatch):
    """The VLM engine's Muse Spark loader, without a tokenizer to load."""
    from omlx.patches.muse_spark import model as spark_model

    monkeypatch.setattr(
        spark_model, "load_processor", lambda *args, **kwargs: "processor"
    )
    return lambda path: spark_model.load(path)[0]


def test_muse_spark_modules_resolve(spark_applied):
    import importlib

    import mlx_lm.utils as lm_utils
    from mlx_vlm.prompt_utils import MODEL_CONFIG

    model_class, _ = lm_utils._get_classes({"model_type": "muse_spark"})
    assert model_class.__module__ == "mlx_lm.models.muse_spark"
    # The vendored mlx-vlm package holds the processor only.
    module = importlib.import_module("mlx_vlm.models.muse_spark")
    assert hasattr(module, "MuseSparkProcessor") and not hasattr(module, "Model")
    assert "muse_spark" in MODEL_CONFIG


@pytest.mark.parametrize("for_vlm", [False, True])
def test_preload_keeps_mlx_lm_muse_spark(spark_applied, tmp_path, for_vlm):
    import importlib

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    (tmp_path / "config.json").write_text('{"model_type": "muse_spark"}')
    maybe_apply_pre_load_patches(str(tmp_path), for_vlm=for_vlm)
    module = importlib.import_module("mlx_lm.models.muse_spark")
    assert module.__name__ == "mlx_lm.models.muse_spark"


def test_muse_spark_loads_modelopt_nvfp4(spark_load, tmp_path):
    """ModelOpt NVFP4 experts stay packed and apply their tensor scales."""
    config = _spark_config()
    weights = _spark_hf_weights(config)
    nvfp4, dense = _spark_nvfp4_experts(4, 32, 32)
    prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.experts."
    weights.update({prefix + name: value for name, value in dense.items()})
    _spark_save(tmp_path / "dense", config, weights)
    for name in dense:
        weights.pop(prefix + name)
    weights.update({prefix + name: value for name, value in nvfp4.items()})
    config["quantization_config"] = _spark_modelopt_config()
    _spark_save(tmp_path / "nvfp4", config, weights)

    expected_model = spark_load(tmp_path / "dense")
    model = spark_load(tmp_path / "nvfp4")
    experts = model.layers[_SPARK_NVFP4_LAYER].mlp.experts
    assert experts.down_proj.mode == "nvfp4" and "global_scale" in experts.down_proj
    assert "scales" not in model.layers[0].mlp.experts.gate_proj

    ids = mx.array([[1, 5, 9, 2, 33, 7, 11, 3, 60, 4, 8, 12]])
    expected, actual = expected_model(ids), model(ids)
    mx.eval(expected, actual)
    assert float(mx.abs(expected - actual).max()) < 1e-3
    assert float(mx.abs(expected).max()) > 1e-2


def _spark_legacy_format(mixed):
    """(bits, group_size) of the HF tensors earlier conversions quantized.

    With ``mixed``, the q, k, v and gate of a layer differ in bits, as in
    some conversions on the Hub. mlx-lm fuses them, so they need one format.
    These bit widths requantize to the widest one almost without loss.
    """
    mixed_bits = {
        ".layers.0.self_attn.k_proj.weight": 8,
        ".layers.0.self_attn.v_proj.weight": 8,
        ".layers.1.self_attn.q_proj.weight": 2,
    }

    def fmt(key):
        if "vision" in key or not key.endswith(
            ("_proj", "_proj.weight", "embed_tokens.weight", "lm_head.weight")
        ):
            return None
        if ".mlp.experts." in key:
            return 3, 32
        if key == "lm_head.weight":
            return 6, 32
        bits = next((b for k, b in mixed_bits.items() if key.endswith(k)), 4)
        return (bits if mixed else 4), 32

    return fmt


def _spark_round(weights, fmt):
    """``weights`` with each tensor that ``fmt`` names rounded to its format."""
    out = dict(weights)
    for key, value in weights.items():
        if spec := fmt(key):
            bits, group_size = spec
            packed = mx.quantize(value, group_size=group_size, bits=bits)
            out[key] = mx.dequantize(*packed, group_size=group_size, bits=bits)
    return out


def _spark_legacy(weights, fmt, *, oq_layout):
    """HF ``weights`` in the layout of earlier oMLX-based conversions.

    The mlx-lm text layout has the text weights under ``model.``, the earlier
    oQ layout under ``language_model.model.`` with the vision tower. Both
    keep the experts at ``switch_mlp`` and quantize the tensors ``fmt`` names.
    """
    if oq_layout:
        prefixes = (
            ("model.language_model.", "language_model.model."),
            ("lm_head.", "language_model.lm_head."),
            ("model.", ""),
        )
    else:
        prefixes = (("model.language_model.", "model."), ("lm_head.", "lm_head."))
    out = {}
    for key, value in weights.items():
        old = next((o for o, _ in prefixes if key.startswith(o)), None)
        if old is None:
            continue
        name = dict(prefixes)[old] + key[len(old) :]
        tensors = {name: value}
        if ".mlp.experts." in name:
            layer, part = name.split(".mlp.experts.")
            switch = f"{layer}.mlp.switch_mlp."
            if part == "gate_up_proj":
                gate, up = mx.split(value, 2, axis=1)
                tensors = {switch + "gate_proj.weight": gate}
                tensors[switch + "up_proj.weight"] = up
            elif part == "down_proj":
                tensors = {switch + "down_proj.weight": value}
            else:
                tensors = {f"{layer}.mlp.{part}": value}
        for name, tensor in tensors.items():
            if (spec := fmt(key)) is None:
                out[name] = tensor
                continue
            bits, group_size = spec
            module = name.removesuffix(".weight")
            packed = mx.quantize(tensor, group_size=group_size, bits=bits)
            for param, array in zip(
                ("weight", "scales", "biases"), packed, strict=True
            ):
                out[f"{module}.{param}"] = array
    return out


@pytest.mark.parametrize(
    "oq_layout, quantized", [(False, True), (False, False), (True, True)]
)
def test_muse_spark_loads_earlier_layouts(spark_load, tmp_path, oq_layout, quantized):
    """Checkpoints that oMLX-based conversions wrote before load unchanged.

    The mlx-lm text layout cases have a vision_config without vision weights.
    Quantized, they mix q, k, v and gate formats and have a per-module map that
    names their own modules. The earlier oQ layout case has the vision tower.
    """
    import json

    import mlx.nn as nn

    mixed = quantized and not oq_layout
    config = _spark_config(vision=True)
    weights = _spark_hf_weights(config)
    fmt = _spark_legacy_format(mixed) if quantized else lambda key: None
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        _spark_legacy(weights, fmt, oq_layout=oq_layout),
    )
    saved = dict(config)
    if quantized:
        saved["quantization"] = {"group_size": 32, "bits": 4, "mode": "affine"}
    if mixed:
        saved["quantization"]["lm_head"] = {"group_size": 32, "bits": 6}
        saved["quantization"]["model.layers.0.mlp.gate"] = False
    (tmp_path / "config.json").write_text(json.dumps(saved))

    model = spark_load(tmp_path)
    if not oq_layout:
        config["vision_config"] = None
    reference = _spark_model(config, _spark_round(weights, fmt))
    assert (model.args.vision is None) == (not oq_layout)
    # mlx-lm needs a float lm_head.
    assert type(model.lm_head) is nn.Linear
    if quantized:
        attention = [layer.self_attn.qkvg_proj for layer in model.layers]
        expected = [(8, 32), (4, 32)] if mixed else [(4, 32), (4, 32)]
        assert [(a.bits, a.group_size) for a in attention] == expected

    ids = mx.array([[1, 7, 2, 9, 33, 60, 4]])
    images = {}
    if oq_layout:
        # A 2x2 patch grid merges into the one image token.
        images["pixel_values"] = mx.random.normal((4, 2 * 3 * 2 * 2))
        images["image_grid_thw"] = mx.array([[1, 2, 2]])
    embeds = reference.get_input_embeddings(ids, **images).inputs_embeds
    actual = model.get_input_embeddings(ids, **images).inputs_embeds
    assert mx.array_equal(actual, embeds).item()
    expected = reference.language_model(ids, inputs_embeds=embeds)
    actual = model.language_model(ids, inputs_embeds=embeds)
    error = mx.abs(expected - actual).max() / mx.abs(expected).max()
    assert error.item() < (0.01 if mixed else 1e-4)


def test_muse_spark_gate_up_fusion_is_bit_exact(spark_applied):
    import mlx.nn as nn
    from mlx_lm.models import muse_spark

    from omlx.patches.muse_spark.model import Model
    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(0)
    model = Model(muse_spark.ModelArgs.from_dict(_spark_config()))
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda p, m: hasattr(m, "to_quantized") and ".experts." in p,
    )
    # Decode rows run unsorted, the 40-token prompt takes the sorted path.
    prompts = [mx.array([[3]]), mx.array([[i % 60 for i in range(40)]])]
    before = [model(ids) for ids in prompts]
    mx.eval(before)

    assert apply_qwen35_moe_gate_up_fusion(model) == 2
    assert "gate_up_proj" in model.model.layers[0].mlp.experts
    after = [model(ids) for ids in prompts]
    mx.eval(after)
    for b, a in zip(before, after, strict=True):
        assert mx.array_equal(b, a).item()


def test_muse_spark_serving_uses_mlx_lm_model(spark_load, tmp_path):
    """The VLM engine serves mlx-lm's model through the mlx-vlm interface.

    The Muse Glimmer image processor's patch rows give the same embeddings as
    mlx-lm's own processor and ``embed_inputs``.
    """
    import numpy as np
    from mlx.utils import tree_flatten
    from mlx_lm.models import muse_spark
    from mlx_lm.processors.muse_spark import Processor
    from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
        MuseGlimmerImageProcessor,
    )

    from omlx.models.vlm import VLMModelAdapter

    # A checkpoint in mlx-lm's own layout.
    config = _spark_config(vision=True)
    mx.random.seed(3)
    source = muse_spark.Model(muse_spark.ModelArgs.from_dict(config))
    _spark_save(tmp_path, config, dict(tree_flatten(source.parameters())))
    model = spark_load(tmp_path)
    assert isinstance(model, muse_spark.Model)
    assert (model.config.model_type, model.config.image_token_id) == ("muse_spark", 7)

    # A non-square image whose sides are whole merged patches (no resize).
    image = np.random.default_rng(0).integers(0, 256, (12, 20, 3), dtype=np.uint8)
    pixels, tokens = Processor.from_config(config).preprocess_image(image)
    rows = MuseGlimmerImageProcessor(patch_size=2, merge_size=2)(image)
    patches = mx.array(rows["pixel_values"])
    grid = mx.array(rows["image_grid_thw"])
    ids = mx.array([[1, 2] + [7] * tokens + [3]])

    expected = model.embed_inputs(ids, pixel_values=[pixels])
    embeds = model.get_input_embeddings(ids, patches, image_grid_thw=grid)
    assert mx.allclose(embeds.inputs_embeds, expected, atol=1e-5).item()
    features = model.encode_image(patches, image_grid_thw=grid)
    cached = model.get_input_embeddings(ids, patches, cached_image_features=features)
    assert mx.array_equal(cached.inputs_embeds, embeds.inputs_embeds).item()
    with pytest.raises(ValueError, match="pixel_values_videos"):
        model.get_input_embeddings(ids, pixel_values_videos=patches)

    adapter = VLMModelAdapter(model)
    assert adapter.args is model.args.text
    assert len(adapter.make_cache()) == len(adapter.layers)
    logits = model.language_model(ids, inputs_embeds=expected)
    assert mx.array_equal(logits, model(ids, pixel_values=[pixels])).item()


def _spark_oq_source(path, nvfp4=False, mlx_lm_layout=False):
    """A small BF16 Muse Spark checkpoint with the HF names, for oQ.

    With ``nvfp4`` the experts of layer 1 are ModelOpt NVFP4, as in the
    release. Returns those NVFP4 tensors. With ``mlx_lm_layout`` the same
    weights are saved as mlx-lm's model holds them.
    """
    config = _spark_config(
        hidden_size=64,
        intermediate_size=128,
        head_dim=16,
        moe_hidden_size=64,
        moe_intermediate_size=64,
    )
    weights = _spark_hf_weights(config)
    weights = {key: value.astype(mx.bfloat16) for key, value in weights.items()}
    experts = {}
    if nvfp4:
        config["quantization_config"] = _spark_modelopt_config()
        prefix = f"model.language_model.layers.{_SPARK_NVFP4_LAYER}.mlp.experts."
        for name in ("gate_up_proj", "down_proj"):
            weights.pop(prefix + name)
        nvfp4_tensors = _spark_nvfp4_experts(4, 64, 64)[0]
        experts = {prefix + k: v for k, v in nvfp4_tensors.items()}
        weights.update(experts)
    if mlx_lm_layout:
        from mlx.utils import tree_flatten

        model = _spark_model(config, weights)
        weights = dict(tree_flatten(model.parameters()))
    _spark_save(path, config, weights)
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

    # Strict loading proves one format per fused qkvg_proj.
    loaded, _ = load_model(out)
    assert type(loaded).__module__ == "mlx_lm.models.muse_spark"
    assert isinstance(loaded.lm_head, nn.Linear)
    logits = loaded(mx.array([[1, 5, 9, 2, 33]]))
    assert bool(mx.isfinite(logits).all().item())

    # The same weights in mlx-lm's layout give the same output.
    _spark_oq_source(tmp_path / "mlx_lm_src", mlx_lm_layout=True)
    quantize_oq_streaming(
        str(tmp_path / "mlx_lm_src"),
        str(tmp_path / "mlx_lm_oq"),
        4,
        sensitivity_map_override={0: 1.0, 1: 0.5},
    )
    other = mx.load(str(tmp_path / "mlx_lm_oq" / "model.safetensors"))
    assert other.keys() == saved.keys()
    for key, value in saved.items():
        assert value.dtype == other[key].dtype, key
        assert mx.array_equal(value, other[key]).item(), key


def test_muse_spark_oq_passes_nvfp4_experts_through(spark_applied, tmp_path):
    from mlx_lm.utils import load_model

    from omlx.oq import quantize_oq_streaming
    from omlx.patches.muse_spark.oq_layout import sanitize_experts

    src = tmp_path / "src"
    experts = _spark_oq_source(src, nvfp4=True)
    out = tmp_path / "oq"
    quantize_oq_streaming(
        str(src), str(out), 4, sensitivity_map_override={0: 1.0, 1: 0.5}
    )

    loaded, _ = load_model(out)
    switch = loaded.model.layers[_SPARK_NVFP4_LAYER].mlp.experts
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
