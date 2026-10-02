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

_CHECKPOINT = Path(
    "~/Workspace/models/meta-models/Muse-Glimmer-30B"
).expanduser()


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
    from PIL import Image

    from mlx_vlm.models.muse_glimmer.processing_muse_glimmer import (
        MuseGlimmerImageProcessor,
        smart_resize,
    )

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

    from mlx_vlm.models.muse_glimmer.language import TextModel
    from mlx_vlm.models.muse_glimmer.config import TextConfig

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
            "model.language_model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (16, 16)
            ),
            "model.vision_tower.ln_pre.weight": mx.ones((8,)),
            "lm_head.weight": mx.zeros((64, 16)),
        }
    )
    assert "language_model.model.layers.0.self_attn.q_proj.weight" in weights
    assert "vision_tower.ln_pre.weight" in weights
    assert "language_model.lm_head.weight" in weights


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
