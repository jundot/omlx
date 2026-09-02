# SPDX-License-Identifier: Apache-2.0
"""Deterministic tests for the DeepSeek-V4 image token contract."""

import base64
import io
from dataclasses import dataclass
from types import SimpleNamespace

import mlx.core as mx
from PIL import Image

from omlx.cluster.deepseek_v4_vision_runtime import (
    install_deepseek_v4_vision_runtime,
)
from omlx.deepseek_v4_vision import (
    IMAGE,
    IMAGE_END,
    IMAGE_PLACEHOLDER,
    IMAGE_START,
    is_deepseek_v4_vision_config,
)
from omlx.patches.deepseek_v4.vision_inputs import (
    build_image_block,
    prepare_token_ids,
)
from omlx.patches.deepseek_v4.vision_model import Aligner, Attention, ViT


def _config():
    return {
        "model_type": "deepseek_v4",
        "vocab_size": 100,
        "vision_n_layers": 2,
        "vision_dim": 8,
        "vision_n_heads": 2,
        "vision_inter_dim": 16,
        "vision_patch_size": 2,
        "vision_rope_theta": 10_000.0,
        "vision_downsample_ratio": 2,
        "vision_max_n_token": 32,
        "vision_min_pixels": 16,
        "vision_max_wh_ratio": 8,
    }


def _data_url(color):
    image = Image.new("RGB", (4, 4), color=color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()


def test_capability_requires_complete_flat_vision_contract():
    config = _config()
    assert is_deepseek_v4_vision_config(config)
    assert not is_deepseek_v4_vision_config(
        {key: value for key, value in config.items() if key != "vision_n_layers"}
    )
    assert not is_deepseek_v4_vision_config({"model_type": "deepseek_v4"})


def test_image_block_is_deterministic_and_aligned():
    first = build_image_block(3, 2, 5)
    second = build_image_block(3, 2, 5)
    types, permutation = first

    assert first == second
    assert types.count(IMAGE_START) == 1
    assert types.count(IMAGE_END) == 1
    assert types.count(IMAGE) == len(permutation) == 6
    assert (5 + types.index(IMAGE_START)) % 4 == 3


def test_text_only_prefill_preserves_tokens():
    tokens, images = prepare_token_ids(
        [1, 2, 3], [], image_token_id=99, config=_config()
    )
    assert tokens == [1, 2, 3]
    assert images == ()


def test_one_and_multiple_images_expand_in_prompt_order():
    tokens, images = prepare_token_ids(
        [1, 99, 2, 99, 3],
        [{"url": _data_url("red")}, {"url": _data_url("blue")}],
        image_token_id=99,
        config=_config(),
    )

    assert len(images) == 2
    assert images[0].start < images[1].start
    assert len(tokens) == 3 + sum(len(image.types) for image in images)
    assert all(token >= 100 for token in tokens if token not in {1, 2, 3})
    assert IMAGE_PLACEHOLDER.startswith("<")


def test_tiny_vision_encoder_and_aligner_produce_language_embeddings():
    config = SimpleNamespace(
        vision_patch_size=2,
        vision_dim=8,
        vision_n_heads=2,
        vision_inter_dim=16,
        vision_n_layers=1,
        vision_rope_theta=10_000.0,
        vision_downsample_ratio=2,
        hidden_size=12,
    )
    vision = ViT(config)
    aligner = Aligner(config)

    features = vision(mx.ones((16, 3, 2, 2)), 4, 4)
    embeddings = aligner(features, 4, 4)
    mx.eval(embeddings)

    assert features.shape == (16, 8)
    assert embeddings.shape == (4, 12)
    assert embeddings.dtype == mx.float32


def test_vision_attention_splits_fused_qkv_before_head_reshape():
    config = SimpleNamespace(vision_dim=4, vision_n_heads=2)
    attention = Attention(config)
    projection = mx.arange(12).reshape(1, 12)
    attention.wqkv = lambda _x: projection
    seen = {}

    def capture(q, k, v, *, scale):
        seen.update(q=q, k=k, v=v, scale=scale)
        return v

    original = mx.fast.scaled_dot_product_attention
    mx.fast.scaled_dot_product_attention = capture
    try:
        attention.wo = lambda value: value
        attention(
            mx.zeros((1, 4)),
            mx.ones((1, 1, 1)),
            mx.zeros((1, 1, 1)),
        )
    finally:
        mx.fast.scaled_dot_product_attention = original

    assert seen["q"].reshape(-1).tolist() == [0, 1, 2, 3]
    assert seen["k"].reshape(-1).tolist() == [4, 5, 6, 7]
    assert seen["v"].reshape(-1).tolist() == [8, 9, 10, 11]


def test_rank_zero_runtime_dispatches_image_prefill_and_streaming():
    @dataclass
    class Request:
        messages: list

    class Model:
        def set_vision_inputs(self, images):
            self.images = images

    class ResponseGenerator:
        def __init__(self, provider):
            self.model_provider = provider

        def _share_object(self, payload):
            return payload

        def _tokenize(self, _tokenizer, _request, _args):
            return [1, 99, 2], [[1, 99, 2]], ["assistant"], "normal"

    seen = {}

    def stream_generate(*_args, **kwargs):
        seen.update(kwargs)
        yield "first"
        yield "second"

    server = SimpleNamespace(
        ResponseGenerator=ResponseGenerator, stream_generate=stream_generate
    )
    provider = SimpleNamespace(
        model=Model(), model_key=("model", None, None), is_batchable=True
    )
    tokenizer = SimpleNamespace(
        convert_tokens_to_ids=lambda token: 99 if token == IMAGE_PLACEHOLDER else None,
        unk_token_id=-1,
    )
    request = Request(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _data_url("red")}},
                    {"type": "text", "text": "describe"},
                ],
            }
        ]
    )

    with install_deepseek_v4_vision_runtime(server, provider, config=_config(), rank=0):
        prompt, segments, _types, state = ResponseGenerator(provider)._tokenize(
            tokenizer, request, SimpleNamespace()
        )
        assert len(provider.model.images) == 1
        streamed = list(server.stream_generate(prompt=prompt))

    assert provider.is_batchable is False
    assert provider.model.images is None
    assert segments == [prompt]
    assert state == "normal"
    assert streamed == ["first", "second"]
    assert seen["prefill_step_size"] == len(prompt)
