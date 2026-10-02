# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GLM-5.3-Flash mlx-vlm compatibility overlay."""

from __future__ import annotations

import base64
import copy
import gc
import importlib
import io
import json
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from PIL import Image

from omlx.memory_monitor import estimate_mla_kv_bytes_per_token
from omlx.model_discovery import detect_model_type
from omlx.oq import (
    _build_model_sanitizer,
    _is_vlm_load,
    universal_quant_predicate,
)
from omlx.patches import mlx_vlm_glm5_next_compat as compat
from omlx.patches.glm_moe_dsa import indexer_nax, sparse_mla, sparse_mla_nax
from omlx.patches import qwen35_verify_qmm
from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk
from omlx.utils.layer_pipeline import LayerPipeline


@pytest.fixture(autouse=True)
def _apply_glm5_next_compat():
    compat.apply_mlx_vlm_glm5_next_compat_patch()


def _tiny_config(*, with_vision: bool = False):
    from mlx_vlm.models import glm5_next

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "dense"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    vision = None
    if with_vision:
        vision = glm5_next.VisionConfig(
            model_type="glm5_next_vision",
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            patch_size=2,
            out_hidden_size=32,
            projection_intermediate_size=64,
            image_size=4,
            spatial_merge_size=2,
            temporal_patch_size=2,
        )
    return glm5_next.ModelConfig(
        text_config=text,
        model_type="glm5_next",
        vision_config=vision,
        image_token_id=120,
        video_token_id=121,
    )


def _tiny_config_dict(*, with_vision: bool = False) -> dict:
    config = _tiny_config(with_vision=with_vision)
    text = dict(vars(config.text_config))
    text["linear_attn_config"] = {
        "num_heads": config.text_config.linear_num_heads,
        "head_dim": config.text_config.linear_head_dim,
        "short_conv_kernel_size": config.text_config.linear_conv_kernel_dim,
        "gate_lower_bound": config.text_config.linear_lower_bound,
    }
    payload = {
        "model_type": "glm5_next",
        "architectures": [
            "Glm5NextForConditionalGeneration" if with_vision else "Glm5NextForCausalLM"
        ],
        "text_config": text,
    }
    if with_vision:
        payload["vision_config"] = dict(vars(config.vision_config))
    return payload


def _feed_pool(cache, token_count: int) -> None:
    width = 4
    values = mx.arange(token_count * width, dtype=mx.float32).reshape(
        1, token_count, width
    )
    gates = mx.zeros_like(values)
    ready, _, _ = cache.accumulate_windows(values, gates, 0)
    pooled = ready.reshape(1, -1, cache.ratio, width).mean(axis=2)
    cache.update_and_fetch(pooled)


def test_glm5_next_registers_pinned_upstream_model():
    assert compat.apply_mlx_vlm_glm5_next_compat_patch() in {True, False}
    from mlx_vlm.models import glm5_next
    from mlx_vlm.utils import get_model_and_args, update_module_configs

    module, model_type = get_model_and_args(_tiny_config_dict(with_vision=True))
    config_dict = _tiny_config_dict(with_vision=True)
    model_config = module.ModelConfig.from_dict(config_dict)
    model_config = update_module_configs(
        model_config, module, config_dict, ["text", "vision"]
    )

    assert model_type == "glm5_next"
    assert module is glm5_next
    assert model_config.text_config.model_type == "glm5_next_text"
    assert model_config.vision_config.model_type == "glm5_next_vision"
    assert compat.PR_URL.endswith("/2030")


@pytest.mark.parametrize("with_vision", [False, True])
def test_glm5_next_discovery_uses_vlm_loader(tmp_path, with_vision):
    (tmp_path / "config.json").write_text(
        json.dumps(_tiny_config_dict(with_vision=with_vision))
    )
    assert detect_model_type(tmp_path) == "vlm"


def test_text_only_config_does_not_construct_a_vision_tower():
    from mlx_vlm.models import glm5_next
    from mlx_vlm.utils import update_module_configs

    config_dict = _tiny_config_dict()
    config_dict["vision_config"] = {}
    model_config = glm5_next.ModelConfig.from_dict(config_dict)
    model_config = update_module_configs(
        model_config, glm5_next, config_dict, ["text", "vision"]
    )
    model = glm5_next.Model(model_config)

    assert model.vision_model is None
    with pytest.raises(ValueError, match="vision_config is None"):
        model.get_input_embeddings(
            input_ids=mx.array([[1]], dtype=mx.int32),
            pixel_values=mx.zeros((1, 1)),
        )


def test_torch_free_processor_expands_image_tokens_and_runs_vision_path():
    from mlx_vlm.models import glm5_next

    class TokenizerStub:
        model_input_names = ["input_ids", "attention_mask"]

        @staticmethod
        def convert_tokens_to_ids(token):
            return {"<|image|>": 120, "<|video|>": 121}[token]

        @staticmethod
        def __call__(texts, **kwargs):
            del kwargs
            rows = []
            for text in texts:
                rows.append([1] + [120] * text.count("<|image|>") + [2])
            return {
                "input_ids": rows,
                "attention_mask": [[1] * len(row) for row in rows],
            }

    image_processor = glm5_next.Glm5NextImageProcessor(
        patch_size=2,
        temporal_patch_size=2,
        merge_size=2,
        min_image_tokens=1,
        max_image_tokens=4,
    )
    processor = glm5_next.Glm5NextProcessor(
        image_processor=image_processor,
        tokenizer=TokenizerStub(),
    )
    inputs = processor(
        images=[Image.new("RGB", (8, 4), "blue")],
        text=["<|begin_of_image|><|image|><|end_of_image|>"],
    )

    image_tokens = int(mx.sum(inputs["input_ids"] == 120).item())
    expected_tokens = int(inputs["image_grid_thw"][0].prod().item()) // 4
    assert image_tokens == expected_tokens == 2
    assert inputs["pixel_values"].shape == (8, 24)

    model = glm5_next.Model(_tiny_config(with_vision=True))
    features = model.encode_image(
        inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    )
    embeddings = model.get_input_embeddings(
        inputs["input_ids"],
        inputs["pixel_values"],
        image_grid_thw=inputs["image_grid_thw"],
    ).inputs_embeds
    mx.eval(features, embeddings)

    assert features.shape == (2, 32)
    assert embeddings.shape == (1, 4, 32)
    assert mx.all(mx.isfinite(features)).item()


def test_glm_image_budget_uses_8k_limit_and_exact_resize_count():
    from mlx_vlm.models.glm5_next import Glm5NextImageProcessor

    from omlx.engine.vlm import (
        _count_image_tokens_real,
        _derive_image_token_upper_bound,
    )

    processor = Glm5NextImageProcessor()
    wrapper = type("Processor", (), {"image_processor": processor})()
    buffer = io.BytesIO()
    Image.new("RGB", (56, 42)).save(buffer, format="PNG")
    data_uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": data_uri}}],
        }
    ]

    assert _derive_image_token_upper_bound(wrapper) == 8000
    assert _count_image_tokens_real(messages, wrapper, upper_bound=8000) == 20


def test_tiny_text_prefill_decode_and_batch_match():
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    single_cache = model.make_cache()
    prompt = mx.array([[2, 3, 4, 5, 6, 7]], dtype=mx.int32)
    prefill = model(prompt, cache=single_cache).logits
    decoded = model(mx.array([[8]], dtype=mx.int32), cache=single_cache).logits
    mx.eval(prefill, decoded)

    assert prefill.shape == (1, 6, 128)
    assert decoded.shape == (1, 1, 128)
    assert mx.all(mx.isfinite(prefill)).item()
    sparse_cache = single_cache[1]
    assert type(sparse_cache).__name__ == "CacheList"
    assert sparse_cache[0].values.shape[-1] == 0
    assert type(sparse_cache[1]).__name__ == "PoolingCache"

    generate = importlib.import_module("mlx_lm.generate")
    batch_cache = generate._merge_caches([model.make_cache(), model.make_cache()])
    batch_tokens = mx.concatenate([prompt, prompt], axis=0)
    batch_logits = model(batch_tokens, cache=batch_cache).logits
    left_logits = model(prompt, cache=model.make_cache()).logits
    right_logits = model(prompt, cache=model.make_cache()).logits
    mx.eval(batch_logits, left_logits, right_logits)

    assert type(batch_cache[1][1]).__name__ == "BatchPoolingCache"
    assert mx.allclose(batch_logits[:1], left_logits, atol=3e-4).item()
    assert mx.allclose(batch_logits[1:], right_logits, atol=3e-4).item()


@pytest.mark.parametrize("batch_size,block_size", [(1, 2), (2, 4), (3, 8), (4, 2)])
def test_short_verify_keeps_latent_kv_and_matches_decode(
    batch_size, block_size, monkeypatch
):
    from mlx_lm.models.mla import MultiLinear
    from mlx_vlm.models.glm5_next.language import LanguageModel

    mx.random.seed(937)
    config = _tiny_config()
    config.text_config.index_topk = 64
    model = LanguageModel(config.text_config, config)
    prompt = mx.arange(batch_size * 12).reshape(batch_size, 12) % 100
    row_caches = []
    for row in range(batch_size):
        row_cache = model.make_cache()
        mx.eval(model(prompt[row : row + 1, row:], cache=row_cache).logits)
        row_caches.append(row_cache)
    cache = [type(rows[0]).merge(rows) for rows in zip(*row_caches)]
    reference_cache = copy.deepcopy(cache)
    block = mx.arange(batch_size * block_size).reshape(batch_size, block_size) + 32
    attention = model.model.layers[1].self_attn
    projections = []
    original = MultiLinear.__call__

    def traced(self, x, *args, **kwargs):
        if self is attention.embed_q or self is attention.unembed_out:
            projections.append(x.shape[-2])
        return original(self, x, *args, **kwargs)

    monkeypatch.setattr(MultiLinear, "__call__", traced)
    verified = model(block, cache=cache).logits
    mx.eval(verified)
    # Project the short query/output block, never all cached keys and values.
    assert projections == [block_size, block_size]
    sequential = mx.concatenate(
        [
            model(block[:, i : i + 1], cache=reference_cache).logits
            for i in range(block_size)
        ],
        axis=1,
    )
    mx.eval(sequential)
    assert mx.allclose(verified, sequential, atol=3e-4, rtol=3e-4).item()


def test_variable_length_batch_matches_single_request_greedy_tokens():
    from mlx_lm.generate import BatchGenerator
    from mlx_vlm.models.glm5_next import Model

    from omlx.models.vlm import VLMModelAdapter

    mx.random.seed(17)
    config = _tiny_config()
    model = VLMModelAdapter(Model(config))

    def generate(prompts, max_tokens=4):
        generator = BatchGenerator(
            model,
            max_tokens=max_tokens,
            prefill_batch_size=len(prompts),
            completion_batch_size=len(prompts),
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        uids = generator.insert(prompts, max_tokens=[max_tokens] * len(prompts))
        outputs = {uid: [] for uid in uids}
        for _ in range(max_tokens + 4):
            _, responses = generator.next()
            for response in responses:
                outputs[response.uid].append(response.token)
            if all(len(tokens) == max_tokens for tokens in outputs.values()):
                break
        return [outputs[uid] for uid in uids]

    short_prompt = [2, 3, 4, 5, 6, 7]
    long_prompt = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    single = generate([short_prompt])[0]
    batched = generate([short_prompt, long_prompt])[0]

    assert batched == single


def test_variable_length_batch_logits_match_single_requests():
    from mlx_lm.generate import BatchGenerator
    from mlx_vlm.models.glm5_next import Model

    from omlx.models.vlm import VLMModelAdapter

    mx.random.seed(3184)
    model = VLMModelAdapter(Model(_tiny_config()))

    def first_logits(prompts):
        captured = []

        def sampler(logits):
            mx.eval(logits)
            captured.append(logits)
            return mx.argmax(logits, axis=-1)

        generator = BatchGenerator(
            model,
            max_tokens=3,
            prefill_batch_size=len(prompts),
            completion_batch_size=len(prompts),
            sampler=sampler,
        )
        generator.insert(prompts, max_tokens=[3] * len(prompts))
        for _ in range(4):
            generator.next()
            if captured:
                break
        assert len(captured) == 1
        return captured[0]

    short_prompt = [2, 3, 4]
    long_prompt = [2, 3, 4, 5]
    short_logits = first_logits([short_prompt])[0]
    long_logits = first_logits([long_prompt])[0]
    batch_logits = first_logits([short_prompt, long_prompt])

    assert mx.allclose(batch_logits[0], short_logits, atol=3e-4, rtol=3e-4).item()
    assert mx.allclose(batch_logits[1], long_logits, atol=3e-4, rtol=3e-4).item()


def test_late_join_batch_matches_single_request_greedy_tokens():
    from mlx_lm.generate import BatchGenerator
    from mlx_vlm.models.glm5_next import Model

    from omlx.models.vlm import VLMModelAdapter

    mx.random.seed(31)
    model = VLMModelAdapter(Model(_tiny_config()))
    prompts = [
        [2, 3, 4, 5, 6, 7],
        [8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    ]
    max_tokens = 4

    def generate_single(prompt):
        generator = BatchGenerator(
            model,
            max_tokens=max_tokens,
            prefill_batch_size=1,
            completion_batch_size=2,
            sampler=lambda logits: mx.argmax(logits, axis=-1),
        )
        uid = generator.insert([prompt], max_tokens=[max_tokens])[0]
        output = []
        while len(output) < max_tokens:
            _, responses = generator.next()
            output.extend(r.token for r in responses if r.uid == uid)
        return output

    expected = [generate_single(prompt) for prompt in prompts]
    generator = BatchGenerator(
        model,
        max_tokens=max_tokens,
        prefill_batch_size=1,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    first_uid = generator.insert([prompts[0]], max_tokens=[max_tokens])[0]
    outputs = {first_uid: []}
    _, responses = generator.next()
    outputs[first_uid].extend(r.token for r in responses if r.uid == first_uid)

    second_uid = generator.insert([prompts[1]], max_tokens=[max_tokens])[0]
    outputs[second_uid] = []
    for _ in range(max_tokens + 6):
        _, responses = generator.next()
        for response in responses:
            outputs[response.uid].append(response.token)
        if all(len(tokens) == max_tokens for tokens in outputs.values()):
            break

    assert outputs[first_uid] == expected[0]
    assert outputs[second_uid] == expected[1]


def test_pooling_cache_filter_extend_and_reorder_preserve_row_state():
    from mlx_lm.models.cache import BatchPoolingCache, PoolingCache

    first = PoolingCache(2)
    second = PoolingCache(2)
    third = PoolingCache(2)
    _feed_pool(first, 5)
    _feed_pool(second, 3)
    _feed_pool(third, 7)

    batch = BatchPoolingCache.merge([first, second])
    assert batch._processed == [5, 3]
    batch.filter(mx.array([1], dtype=mx.int32))
    batch.extend(BatchPoolingCache.merge([third]))
    assert batch._processed == [3, 7]

    batch.filter(mx.array([1, 0], dtype=mx.int32))
    assert batch._processed == [7, 3]
    assert batch._pool_lengths == [3, 1]
    assert batch.extract(0).remainder == 1
    assert batch.extract(1).remainder == 1


def test_nope_mla_memory_estimate_accounts_for_pooled_indexer():
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    # One sparse layer: 8 latent elements/token plus 8/2 pooled-index elements.
    assert (
        estimate_mla_kv_bytes_per_token(
            config.text_config, model.make_cache(), dtype_size=2
        )
        == 24
    )


def test_sanitize_and_oq_keep_sensitive_parameters_in_fp32():
    config_dict = _tiny_config_dict()
    assert _is_vlm_load(config_dict) is True
    sanitizer = _build_model_sanitizer(config_dict)
    assert sanitizer is not None

    weights = {
        "model.language_model.layers.0.self_attn.A_log": mx.ones(
            (2,), dtype=mx.bfloat16
        ),
        "model.language_model.layers.0.hc_attn_alpha": mx.ones((2,), dtype=mx.bfloat16),
        "model.language_model.mtp.fc.weight": mx.ones((2, 2)),
        "model.language_model.layers.1.self_attn.kv_b_proj.weight": mx.ones(
            (32, 8), dtype=mx.bfloat16
        ),
    }
    sanitized = sanitizer(weights)

    a_log = "language_model.model.layers.0.self_attn.forget_gate.A_log"
    hc = "language_model.model.layers.0.attn_hc.alpha"
    assert sanitized[a_log].dtype == mx.float32
    assert sanitized[hc].dtype == mx.float32
    assert sanitized[
        "language_model.model.layers.1.self_attn.embed_q.weight"
    ].shape == (2, 8, 8)
    assert sanitized[
        "language_model.model.layers.1.self_attn.unembed_out.weight"
    ].shape == (2, 8, 8)
    assert not any("mtp" in key for key in sanitized)
    assert sanitizer._omlx_cast_predicate(a_log) is False
    assert universal_quant_predicate(
        "model.layers.1.self_attn.indexer.wk",
        None,
        config_dict,
        oq_level=4,
    ) == {"bits": 8, "group_size": 64, "mode": "affine"}


def test_sanitize_remaps_quantized_forget_gate_sidecars():
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    prefix = "language_model.model.layers.0.self_attn."
    weights = {
        prefix + "f_a_proj.weight": mx.ones((32, 32)),
        prefix + "f_a_proj.scales": mx.ones((2,), dtype=mx.bfloat16),
        prefix + "f_a_proj.biases": mx.ones((2,), dtype=mx.bfloat16),
        prefix + "f_b_proj.weight": mx.ones((32, 32)),
        prefix + "f_b_proj.scales": mx.ones((2,), dtype=mx.bfloat16),
        prefix + "f_b_proj.biases": mx.ones((2,), dtype=mx.bfloat16),
    }
    sanitized = model.sanitize(dict(weights))
    gate = prefix + "forget_gate."
    for proj in ("f_a_proj", "f_b_proj"):
        for part in ("weight", "scales", "biases"):
            assert gate + proj + "." + part in sanitized
    assert not any(
        key.startswith(prefix + "f_") and ".forget_gate." not in key
        for key in sanitized
    )


@pytest.mark.parametrize("text_only", [False, True])
@pytest.mark.parametrize("preserve_mtp", [False, True])
def test_oq_roundtrip_with_nextn_weights(tmp_path, monkeypatch, text_only, preserve_mtp):
    from mlx.utils import tree_flatten
    from mlx_vlm.models import glm5_next
    from mlx_vlm.utils import load_model

    from omlx.oq import quantize_oq_streaming
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
    from tests.test_glm5_next_mtp import TINY_TEXT_CONFIG

    glm5_next_vlm_runtime.apply()
    config = _tiny_config(with_vision=True)
    text = copy.deepcopy(TINY_TEXT_CONFIG)
    text.update(
        num_hidden_layers=2,
        qk_nope_head_dim=64,
        v_head_dim=64,
        index_head_dim=64,
        layer_types=["linear_attention", "deepseek_sparse_attention"],
        mlp_layer_types=["dense", "dense"],
    )
    text["linear_attn_config"].update(kda_layers=[0], full_attn_layers=[1])
    config.text_config = glm5_next.TextConfig.from_dict(text)
    model = glm5_next.Model(config)
    weights = dict(tree_flatten(model.parameters()))
    for prefix in (
        "language_model.model.layers.1.self_attn.",
        "language_model.mtp.0.block.self_attn.",
    ):
        wk = weights.pop(prefix + "embed_q.weight").swapaxes(-1, -2)
        wv = weights.pop(prefix + "unembed_out.weight")
        weights[prefix + "kv_b_proj.weight"] = mx.concatenate([wk, wv], axis=1).reshape(
            -1, text["kv_lora_rank"]
        )
    raw = {}
    for key, value in weights.items():
        key = key.replace("language_model.model.", "model.language_model.")
        key = key.replace("language_model.lm_head.", "lm_head.")
        key = key.replace("vision_model.", "model.visual.")
        key = key.replace(
            "language_model.mtp.0.block.", "model.language_model.layers.2."
        )
        key = key.replace(
            "language_model.mtp.0.norm.",
            "model.language_model.layers.2.shared_head.norm.",
        )
        key = key.replace("language_model.mtp.0.", "model.language_model.layers.2.")
        key = key.replace(".forget_gate.", ".")
        raw[key] = value.astype(mx.bfloat16)
    source = tmp_path / "source"
    source.mkdir()
    payload = _tiny_config_dict(with_vision=True)
    payload.update(text_config=text, eos_token_id=[2])
    (source / "config.json").write_text(json.dumps(payload))
    mx.save_safetensors(str(source / "model.safetensors"), raw)
    tokens = mx.array([[1, 3, 4, 5, 6, 7, 8, 9]], dtype=mx.int32)
    monkeypatch.setattr("omlx.oq._load_calibration_data", lambda *a, **kw: tokens)
    monkeypatch.setattr("mlx_lm.tokenizer_utils.load", lambda *a, **kw: object())
    output = tmp_path / "output"
    quantize_oq_streaming(
        str(source),
        str(output),
        4,
        text_only=text_only,
        preserve_mtp=preserve_mtp,
        sensitivity_map_override={0: 1, 1: 1},
        enhanced=True,
        imatrix_num_samples=1,
        imatrix_seq_length=8,
    )
    loaded = load_model(output, lazy=True, strict=True)
    params = dict(tree_flatten(loaded.parameters()))
    assert any(".mtp." in key for key in params) == preserve_mtp
    assert (loaded.vision_model is None) == text_only
    assert mx.isfinite(loaded(mx.array([[1, 3, 4]])).logits).all().item()
    if preserve_mtp:
        lm = loaded.language_model
        result = lm(tokens, return_hidden=True)
        draft = lm.mtp_forward(result.hidden_states[-1], tokens)
        assert mx.isfinite(draft).all().item()


def test_vector_gate_kernel_matches_reference_with_padding_mask():
    from mlx_vlm.models.glm5_next.gated_delta import gated_delta_update

    mx.random.seed(19)
    shape = (1, 4, 2, 32)
    q = mx.random.normal(shape, dtype=mx.float16)
    k = mx.random.normal(shape, dtype=mx.float16)
    v = mx.random.normal(shape, dtype=mx.float16)
    a = mx.random.normal(shape, dtype=mx.float16)
    beta = mx.random.normal((1, 4, 2), dtype=mx.float16)
    a_log = mx.zeros((2, 1), dtype=mx.float32)
    dt_bias = mx.zeros((2, 32), dtype=mx.float32)
    mask = mx.array([[True, True, False, True]])

    expected, expected_state = gated_delta_update(
        q,
        k,
        v,
        a,
        beta,
        a_log,
        dt_bias,
        mask=mask,
        use_kernel=False,
        lower_bound=-5.0,
    )
    actual, actual_state = gated_delta_update(
        q,
        k,
        v,
        a,
        beta,
        a_log,
        dt_bias,
        mask=mask,
        use_kernel=True,
        lower_bound=-5.0,
    )
    mx.eval(expected, expected_state, actual, actual_state)

    assert mx.allclose(actual, expected, atol=2e-3, rtol=2e-3).item()
    assert mx.allclose(actual_state, expected_state, atol=2e-3, rtol=2e-3).item()


def test_native_glm_indexer_scores_match_mlx_reference_when_available():
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_indexer_scores"):
        pytest.skip("GLM DSA native indexer extension is not built")

    config = _tiny_config().text_config
    config.index_n_heads = 32
    config.index_head_dim = 128
    indexer = Glm5NextIndexer(config)
    mx.random.seed(23)
    q = mx.random.normal((1, 5, 32, 128), dtype=mx.float16)
    keys = mx.random.normal((1, 7, 128), dtype=mx.float16)
    weights = mx.random.normal((1, 5, 32), dtype=mx.float16)
    actual = indexer._native_scores(q, keys, weights)
    if actual is None:
        pytest.skip("GLM DSA indexer kernel rejected the installed ABI")
    reference = mx.sum(
        weights[..., None] * mx.maximum(q @ keys[:, None].swapaxes(-1, -2), 0),
        axis=2,
    )
    mx.eval(actual, reference)
    assert mx.allclose(actual, reference, atol=0.08, rtol=0.02).item()


def test_glm5_next_switch_moe_uses_opt_in_native_weighted_sum():
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.switch_layers import SwitchGLU

    if not fast.has_symbol("glm_moe_weighted_sum"):
        pytest.skip("GLM native MoE weighted-sum extension is not built")

    mx.random.seed(29)
    layer = SwitchGLU(16, 8, 8)
    layer.set_dtype(mx.float16)
    x = mx.random.normal((1, 8, 16), dtype=mx.float16)
    indices = mx.array(
        [[[(token + expert) % 8 for expert in range(8)] for token in range(8)]],
        dtype=mx.int32,
    )
    scores = mx.softmax(mx.random.normal(indices.shape, dtype=mx.float32), axis=-1)

    native = layer(x, indices, scores=scores, weighted_sum=True)
    experts = layer(x, indices, scores=scores, weighted_sum=False)
    reference = (experts * scores[..., None]).sum(axis=-2).astype(native.dtype)
    mx.eval(native, reference)

    assert native.shape == (1, 8, 16)
    assert mx.allclose(native, reference, atol=2e-3, rtol=2e-3).item()


def test_glm5_next_affine_prefill_uses_shared_qmm_kernel(monkeypatch):
    import mlx.nn as nn
    from mlx_vlm.models.glm5_next.linear import linear_forward

    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_q4_affine_qmm_t"):
        pytest.skip("Qwen affine prefill QMM extension is not built")

    base = nn.Linear(64, 64, bias=False)
    base.set_dtype(mx.float16)
    linear = base.to_quantized(group_size=64, bits=4, mode="affine")
    x = mx.random.normal((1, 128, 64), dtype=mx.float16)
    reference = linear(x)

    original = fast.qwen35_q4_affine_qmm_t
    calls = 0

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q4_affine_qmm_t", spy)
    actual = linear_forward(linear, x)
    mx.eval(actual, reference)

    assert calls == 1
    assert mx.allclose(actual, reference, atol=2e-3, rtol=2e-3).item()


def test_glm5_next_q8_indexer_prefill_uses_shared_qmm_kernel(monkeypatch):
    import mlx.nn as nn
    from mlx_vlm.models.glm5_next.linear import linear_forward

    from omlx.custom_kernels.qwen35_prefill import fast

    if not fast.has_symbol("qwen35_q8_affine_qmm_t"):
        pytest.skip("Qwen Q8 affine prefill QMM extension is not built")

    mx.random.seed(37)
    base = nn.Linear(1536, 4096, bias=False)
    base.set_dtype(mx.float16)
    linear = base.to_quantized(group_size=64, bits=8, mode="affine")
    x = mx.random.normal((1, 1024, 1536), dtype=mx.float16)
    reference = linear(x)

    original = fast.qwen35_q8_affine_qmm_t
    calls = 0

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(fast, "qwen35_q8_affine_qmm_t", spy)
    actual = linear_forward(linear, x)
    mx.eval(actual, reference)

    assert calls == 1
    assert mx.allclose(actual, reference, atol=2e-3, rtol=2e-3).item()


@pytest.mark.parametrize(("bits", "tokens"), [(5, 128), (8, 1024)])
def test_glm5_next_prefill_qmm_handles_strided_input(bits, tokens):
    import mlx.nn as nn
    from mlx_vlm.models.glm5_next.linear import linear_forward

    from omlx.custom_kernels.qwen35_prefill import fast

    name = f"qwen35_q{bits}_affine_qmm_t"
    if not fast.has_symbol(name):
        pytest.skip(f"{name} native kernel is not built")

    mx.random.seed(11)
    dims = 128
    base = nn.Linear(dims, dims, bias=False)
    base.set_dtype(mx.float16)
    linear = base.to_quantized(group_size=64, bits=bits, mode="affine")

    wide = mx.random.normal((1, tokens, 2 * dims), dtype=mx.float16)
    mx.eval(wide)
    strided = mx.split(wide, [dims], axis=-1)[1]

    reference = linear(strided)
    actual = linear_forward(linear, strided)
    mx.eval(actual, reference)

    assert mx.allclose(actual, reference, atol=2e-3, rtol=2e-3).item()


@pytest.mark.parametrize(("bits", "tokens"), [(5, 128), (8, 1024)])
def test_glm5_next_fused_qmm_handles_strided_input(bits, tokens):
    import mlx.nn as nn
    from mlx_vlm.models.glm5_next.linear import fused_quantized_matmul

    from omlx.custom_kernels.qwen35_prefill import fast

    name = f"qwen35_q{bits}_affine_qmm_t"
    if not fast.has_symbol(name):
        pytest.skip(f"{name} native kernel is not built")

    mx.random.seed(11)
    dims = 128
    base = nn.Linear(dims, dims, bias=False)
    base.set_dtype(mx.float16)
    linear = base.to_quantized(group_size=64, bits=bits, mode="affine")

    wide = mx.random.normal((1, tokens, 2 * dims), dtype=mx.float16)
    mx.eval(wide)
    strided = mx.split(wide, [dims], axis=-1)[1]

    reference = linear(strided)
    actual = fused_quantized_matmul(
        strided,
        linear.weight,
        linear.scales,
        linear.biases,
        bits=bits,
        group_size=64,
    )
    mx.eval(actual, reference)

    assert mx.allclose(actual, reference, atol=2e-3, rtol=2e-3).item()


def test_sparse_attention_native_routes_get_fp16_despite_fp32_activations(monkeypatch):
    """FP32 projections must produce FP16 inputs at native attention boundaries."""
    import mlx_vlm.models.glm5_next.language as lang

    text = lang.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=4096,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=64,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=0,
        v_head_dim=256,
        qk_nope_head_dim=256,
        mla_use_nope=True,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    attn = lang.Glm5NextSparseAttention(text)

    seen = []

    def spy_sma(q_latent, q_pe, kv_latent, k_pe, topk_indices, scale, **kw):
        seen.append(
            ("sparse_mla", *(t.dtype for t in (q_latent, q_pe, kv_latent, k_pe)))
        )
        # Native sparse MLA returns latent-width output [B, H, L, 512].
        return mx.zeros(
            q_latent.shape[:2] + (q_latent.shape[2], 512), dtype=q_latent.dtype
        )

    def spy_eba(q, k, v, topk_indices, scale, **kw):
        seen.append(("exact_block", *(t.dtype for t in (q, k, v))))
        return mx.zeros(q.shape, dtype=q.dtype)

    def spy_nax(q_latent, kv_latent, topk_indices, scale):
        # The tensor-unit kernel is tried first on M5 hosts; decline here so
        # the native kernel route below is exercised on every host.
        seen.append(("sparse_mla_nax", q_latent.dtype, kv_latent.dtype))
        return None

    monkeypatch.setattr(lang, "sparse_mla_attention", spy_sma)
    monkeypatch.setattr(lang, "exact_block_token_attention", spy_eba)
    monkeypatch.setattr(lang, "sparse_mla_attention_nax", spy_nax)
    monkeypatch.setattr(lang, "q8_vup_flat", lambda *a, **k: None)
    # Keep every row on the mocked native routes; the dense prefix would add
    # a real FP32 attention pass over 2051 rows.
    monkeypatch.setattr(attn, "_dense_prefix_rows", lambda *args: (0, 0))

    x = mx.random.normal((1, 4096, 4096), dtype=mx.float32)
    out = attn(x, mask=None, cache=None)
    mx.eval(out)
    nax = [s for s in seen if s[0] == "sparse_mla_nax"]
    assert nax, "the sparse chunk must try the tensor-unit route first"
    assert all(dt == mx.float16 for dt in nax[0][1:]), (
        f"tensor-unit sparse MLA received {nax[0][1:]}, expected fp16"
    )
    sma = [s for s in seen if s[0] == "sparse_mla"]
    assert sma, "Kv>=4096 must attempt the native sparse MLA route"
    assert all(dt == mx.float16 for dt in sma[0][1:]), (
        f"native sparse MLA received {sma[0][1:]}, expected fp16"
    )

    seen.clear()
    x = mx.random.normal((1, 2500, 4096), dtype=mx.float32)
    out = attn(x, mask=None, cache=None)
    mx.eval(out)
    eba = [s for s in seen if s[0] == "exact_block"]
    assert eba, "2048<Kv<4096 must attempt the native exact-block route"
    assert all(dt == mx.float16 for dt in eba[0][1:]), (
        f"native exact-block received {eba[0][1:]}, expected fp16"
    )


def test_q8_vup_flat_gates_dtype_mismatch_and_preserves_projection_contract():
    """Use fused v-up only for matching dtypes and preserve FP32 scales otherwise."""
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.glm_moe_dsa.sparse_mla import q8_vup_flat

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")

    from mlx_lm.models.mla import QuantizedMultiLinear

    x = mx.random.normal((1, 64, 32, 512), dtype=mx.float16)
    mx.eval(x)

    # Use 8-bit affine weights with FP32 scales and biases.
    proj = QuantizedMultiLinear(512, 256, 64, group_size=64, bits=8, mode="affine")
    assert proj.scales.dtype == mx.float32
    # Must NOT raise the native dtype-mismatch; returns None to fall back.
    assert q8_vup_flat(x, proj, key_length=32768) is None
    # The fallback projection preserves the fp32 contract (promotes to fp32).
    out = proj(x)
    mx.eval(out)
    assert out.dtype == mx.float32

    # Matching FP16 scales must still use the fused kernel.
    proj16 = QuantizedMultiLinear(
        512, 256, 64, group_size=64, bits=8, mode="affine"
    )
    proj16.scales = proj16.scales.astype(mx.float16)
    proj16.biases = proj16.biases.astype(mx.float16)
    mx.eval(proj16.scales, proj16.biases)
    fused = q8_vup_flat(x, proj16, key_length=32768)
    mx.eval(fused)
    assert fused is not None and fused.dtype == mx.float16
    # Fused result matches the tolerant quantized-matmul reference layout.
    ref = proj16(x).transpose(0, 2, 1, 3).reshape(1, 32, -1)
    mx.eval(ref)
    assert float(mx.max(mx.abs(fused - ref.astype(mx.float16))).item()) <= 0.125


def test_sparse_attention_completes_at_32k_with_fp32_scale_projection(monkeypatch):
    """Verify native sparse MLA output can feed an FP32-scale projection at 32K."""
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.glm_moe_dsa.sparse_mla import q8_vup_flat, sparse_mla_attention

    if not fast.is_native_available():
        pytest.skip("GLM MoE DSA native extension is unavailable")

    from mlx_lm.models.mla import QuantizedMultiLinear

    B, H, L, Kv, topk = 1, 64, 32, 32768, 2048
    mx.random.seed(0)
    q = mx.random.normal((B, H, L, 512), dtype=mx.float16)
    q_pe = mx.zeros((B, H, L, 64), dtype=mx.float16)
    kv = mx.random.normal((B, 1, Kv, 512), dtype=mx.float16)
    k_pe = mx.zeros((B, 1, Kv, 64), dtype=mx.float16)
    idx = mx.broadcast_to(
        mx.arange(topk, dtype=mx.uint32)[None, None, None, :], (B, 1, L, topk)
    )
    mx.eval(q, q_pe, kv, k_pe, idx)

    out = sparse_mla_attention(q, q_pe, kv, k_pe, idx, 1.0 / (256**0.5))
    mx.eval(out)
    assert out.dtype == mx.float16, "native sparse-MLA must return fp16"

    proj = QuantizedMultiLinear(512, 256, 64, group_size=64, bits=8, mode="affine")
    assert proj.scales.dtype == mx.float32
    # The exact call that used to raise must now fall back, not crash.
    assert q8_vup_flat(out, proj, key_length=Kv) is None
    residual = proj(out)
    mx.eval(residual)
    assert residual.shape == (B, H, L, 256) and residual.dtype == mx.float32, (
        "v-up fallback must preserve the fp32 residual contract"
    )


def test_prefill_evals_stream_per_layer_to_bound_transient(monkeypatch):
    """Prefill releases layer intermediates and cached buffers; decode stays lazy."""
    import mlx_vlm.models.glm5_next.language as lang

    text = _tiny_config().text_config
    model = lang.Glm5NextModel(text)

    calls = []
    clears = []
    real_eval = mx.eval
    real_clear = mx.clear_cache

    def spy(*args, **kw):
        calls.append(sum(len(a) if isinstance(a, (tuple, list)) else 1 for a in args))
        return real_eval(*args, **kw)

    def clear_spy(**kw):
        clears.append(1)
        return real_clear(**kw)

    monkeypatch.setattr(lang.mx, "eval", spy)
    monkeypatch.setattr(lang.mx, "clear_cache", clear_spy)

    ids = mx.zeros((1, 256), dtype=mx.int32)
    out = model(ids)
    # The last layer's output stays lazy (a prefill chunk only needs its
    # cache update); every earlier layer is evaluated and released.
    assert len(calls) >= text.num_hidden_layers - 1, (
        f"prefill width must eval the stream per layer, got {len(calls)} eval calls"
        f" for {text.num_hidden_layers} layers"
    )
    # Layer-specific buffer sizes can accumulate in the allocator pool.
    assert len(clears) >= text.num_hidden_layers - 1, (
        f"prefill must clear the allocator pool per layer, got {len(clears)}"
        f" clears for {text.num_hidden_layers} layers"
    )
    real_eval(out)

    calls.clear()
    clears.clear()
    decode = mx.zeros((1, 1), dtype=mx.int32)
    out = model(decode)
    real_eval(out)
    assert len(calls) < text.num_hidden_layers, (
        "decode width must stay lazy (no per-layer eval)"
    )
    assert not clears, "decode width must not clear the pool per layer"


def test_prefill_leaves_the_last_layer_output_lazy():
    """A prefill chunk evaluates every layer but the last through the pipeline
    (the chunk only needs the last layer's cache update); the caches and the
    output are identical to a forward whose output is read right away."""
    import mlx_vlm.models.glm5_next.language as lang

    text = _tiny_config().text_config
    lm = lang.LanguageModel(text)
    mx.eval(lm.parameters())
    ids = mx.array([[(7 * i + 3) % text.vocab_size for i in range(256)]], dtype=mx.int32)

    def flat_state(cache):
        arrays = []
        for c in cache:
            for part in getattr(c, "caches", None) or (c,):
                state = part.state
                for a in state if isinstance(state, (list, tuple)) else (state,):
                    if isinstance(a, mx.array):
                        arrays.append(a)
        return arrays

    queued = []
    real_async_eval = mx.async_eval

    def spy(*arrays):
        queued.append(len(arrays))
        return real_async_eval(*arrays)

    lang.mx.async_eval = spy
    try:
        cache_a = lm.make_cache()
        out_a = lm.model(ids, cache=cache_a)
        mx.eval(flat_state(cache_a))
    finally:
        lang.mx.async_eval = real_async_eval
    assert len(queued) == text.num_hidden_layers - 1
    mx.eval(out_a)

    cache_b = lm.make_cache()
    out_b = lm.model(ids, cache=cache_b)
    mx.eval(out_b, flat_state(cache_b))
    assert mx.array_equal(out_a, out_b).item()
    state_a, state_b = flat_state(cache_a), flat_state(cache_b)
    assert len(state_a) == len(state_b) > 0
    for a, b in zip(state_a, state_b):
        assert a.shape == b.shape and mx.array_equal(a, b).item()


def test_patch_overrides_site_packages_glm5_next_copy():
    """The vendor module must replace an already imported upstream module."""
    import sys
    from pathlib import Path

    import mlx_vlm.models

    pkg = "mlx_vlm.models.glm5_next"
    vendor_str = str(compat._VENDOR_MLX_VLM)

    saved_modules = {
        n: sys.modules.pop(n)
        for n in list(sys.modules)
        if n == pkg or n.startswith(pkg + ".")
    }
    saved_path = list(mlx_vlm.models.__path__)
    for p in [p for p in list(mlx_vlm.models.__path__) if vendor_str in p]:
        mlx_vlm.models.__path__.remove(p)
    applied = compat._APPLIED
    compat._APPLIED = False
    try:
        # Server state: discovery imported the site-packages copy BEFORE the
        # patch ran, so the package is cached in sys.modules already.
        import mlx_vlm.models.glm5_next.language as early

        assert vendor_str not in str(early.__file__)
        assert compat.apply_mlx_vlm_glm5_next_compat_patch() is True
        import mlx_vlm.models.glm5_next.language as lang

        assert str(Path(lang.__file__).resolve()).startswith(
            str(Path(vendor_str).resolve())
        ), f"patch did not override: {lang.__file__}"
        src = Path(lang.__file__).read_text()
        assert "native_dtype" in src, "vendor language.py fix missing"
        assert "clear_cache" in src, "vendor eval backpressure missing"
    finally:
        for n in [n for n in list(sys.modules) if n == pkg or n.startswith(pkg + ".")]:
            del sys.modules[n]
        sys.modules.update(saved_modules)
        mlx_vlm.models.__path__[:] = saved_path
        compat._APPLIED = applied


def test_dense_prefix_bypass_matches_reference(monkeypatch):
    from mlx_vlm.models.glm5_next import language

    mx.random.seed(313)
    config = _tiny_config()
    model = language.LanguageModel(config.text_config, config)
    prompt = mx.arange(13, dtype=mx.int32)[None] + 1
    attention = model.model.layers[1].self_attn

    dense_prefix_rows = attention._dense_prefix_rows
    monkeypatch.setattr(attention, "_dense_prefix_rows", lambda *args: (0, 0))
    reference_cache = model.make_cache()
    reference = model(prompt, cache=reference_cache).logits

    monkeypatch.setattr(attention, "_dense_prefix_rows", dense_prefix_rows)
    engaged = []
    original_dense = attention._dense_flat

    def spy(rows):
        engaged.append(rows)
        return original_dense

    monkeypatch.setattr(
        attention,
        "_dense_flat",
        lambda q, kv, mask, rows, past: spy(rows)(q, kv, mask, rows, past),
    )
    bypass_cache = model.make_cache()
    bypassed = model(prompt, cache=bypass_cache).logits
    decoded = model(mx.array([[13]], dtype=mx.int32), cache=bypass_cache).logits
    reference_decoded = model(
        mx.array([[13]], dtype=mx.int32), cache=reference_cache
    ).logits
    mx.eval(reference, bypassed, decoded, reference_decoded)

    # index_topk=4, index_kpool=2 -> the first (4//2+1)*2-1 = 5 rows bypass.
    assert engaged == [5]
    assert mx.allclose(bypassed, reference, atol=3e-4, rtol=3e-4).item()
    # Dense rows still feed the pool, so following decode selects identically.
    assert mx.allclose(decoded, reference_decoded, atol=1e-5, rtol=1e-5).item()


def test_chunked_prefill_with_dense_bypass_matches_single_pass():
    from mlx_vlm.models.glm5_next.language import LanguageModel

    mx.random.seed(126)
    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    tokens = mx.arange(13, dtype=mx.int32)[None] + 1

    single_cache = model.make_cache()
    single = model(tokens, cache=single_cache).logits

    # Chunk 1 (9 rows) exercises the dense prefix + scored tail; chunk 2 (4
    # rows) verifies the pool advanced over the bypassed rows too.
    chunked_cache = model.make_cache()
    first = model(tokens[:, :9], cache=chunked_cache).logits
    second = model(tokens[:, 9:], cache=chunked_cache).logits
    mx.eval(single, first, second)

    assert mx.allclose(first, single[:, :9], atol=3e-4, rtol=3e-4).item()
    assert mx.allclose(second, single[:, 9:], atol=3e-4, rtol=3e-4).item()


def test_indexer_score_from_matches_suffix():
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    mx.random.seed(7)
    config = _tiny_config().text_config
    indexer = Glm5NextIndexer(config)
    x = mx.random.normal((1, 13, config.hidden_size), dtype=mx.float32)
    qr = mx.random.normal((1, 13, config.q_lora_rank), dtype=mx.float32)

    full = indexer(x, qr, None)
    tail = indexer(x, qr, None, score_from=5)
    mx.eval(full, tail)

    assert full.shape[:3] == (1, 1, 13)
    assert tail.shape[:3] == (1, 1, 8)
    assert mx.array_equal(tail, full[:, :, 5:]).item()
    assert indexer(x, qr, None, score_from=13) is None


def test_dense_prefix_rows_gating():
    from mlx_vlm.models.cache import KVCache
    from mlx_vlm.models.glm5_next.language import LanguageModel

    config = _tiny_config()
    model = LanguageModel(config.text_config, config)
    attention = model.model.layers[1].self_attn
    indexer = attention.indexer
    # index_topk=4, index_kpool=2 -> boundary (4//2+1)*2-1 = 5 rows.
    assert indexer.index_topk == 4 and indexer.index_kpool == 2
    assert attention._dense_prefix_rows(13, None, None) == (5, 0)

    populated = KVCache()
    populated.offset = 3
    assert attention._dense_prefix_rows(13, None, [populated, None]) == (2, 3)
    populated.offset = 5
    assert attention._dense_prefix_rows(13, None, [populated, None]) == (0, 5)
    assert attention._dense_prefix_rows(4, None, None) == (4, 0)

    without_tail = indexer.index_kpool_always_select_tail
    indexer.index_kpool_always_select_tail = False
    assert attention._dense_prefix_rows(13, None, None) == (0, 0)
    indexer.index_kpool_always_select_tail = without_tail

    padded = KVCache()
    padded.left_padding = mx.zeros((1,), dtype=mx.int32)
    assert attention._dense_prefix_rows(13, None, [padded, None]) == (0, 0)

    merged = KVCache()
    merged.offset = mx.array([3, 3], dtype=mx.int32)
    assert attention._dense_prefix_rows(13, None, [merged, None]) == (0, 0)

    class _BatchPool:
        _processed = [1, 2]

    assert attention._dense_prefix_rows(13, None, [None, _BatchPool()]) == (0, 0)

    assert attention._dense_prefix_rows(13, "causal", None) == (0, 0)
    short_keys = mx.ones((1, 1, 13, 12), dtype=mx.bool_)
    assert attention._dense_prefix_rows(13, short_keys, None) == (0, 0)
    per_key = mx.ones((1, 1, 1, 13), dtype=mx.bool_)
    assert attention._dense_prefix_rows(13, per_key, None) == (0, 0)
    aligned = mx.ones((1, 1, 13, 13), dtype=mx.bool_)
    assert attention._dense_prefix_rows(13, aligned, None) == (5, 0)


# ---------------------------------------------------------------------------
# Fused KDA (linear attention) prefill prework / norm-gate.
# ---------------------------------------------------------------------------


def _kda_reference_prework(mixed, conv_state, conv_w, heads, dim, q_scale):
    """Stock Glm5NextLinearAttention prework chain in MX ops."""
    import mlx.nn as nn

    length = mixed.shape[1]
    c_dim = 3 * heads * dim
    conv = nn.Conv1d(c_dim, c_dim, kernel_size=4, groups=c_dim, bias=False)
    conv.weight = conv_w
    conv_input = mx.concatenate([conv_state, mixed], axis=1)
    activated = nn.silu(conv(conv_input))
    q, k, v = mx.split(activated, [heads * dim, 2 * heads * dim], axis=-1)
    shape = (1, length, heads, dim)
    q, k, v = q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def l2norm(x):
        return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + 1e-6)

    q = (l2norm(q.astype(mx.float32)) * q_scale).astype(mixed.dtype)
    k = l2norm(k.astype(mx.float32)).astype(mixed.dtype)
    return q, k, v, conv_input[:, -3:, :]


@pytest.mark.parametrize("seq", [1, 2, 3, 5, 130])
def test_kda_prework_kernel_matches_stock(seq):
    from omlx.patches.glm53_kda_prework import kda_prework_fused

    mx.random.seed(41)
    heads, dim = 2, 128
    c_dim = 3 * heads * dim
    mixed = (mx.random.normal((1, seq, c_dim)) * 0.5).astype(mx.bfloat16)
    conv_state = (mx.random.normal((1, 3, c_dim)) * 0.5).astype(mx.bfloat16)
    conv_w = (mx.random.normal((c_dim, 4, 1)) * 0.2).astype(mx.bfloat16)
    q_scale = dim**-0.5

    q, k, v, next_state = kda_prework_fused(
        mixed, conv_state, conv_w, mx.array(q_scale, dtype=mx.float32), seq, heads, dim
    )
    rq, rk, rv, r_state = _kda_reference_prework(
        mixed, conv_state, conv_w, heads, dim, q_scale
    )
    mx.eval(q, k, v, next_state, rq, rk, rv, r_state)
    assert mx.array_equal(q, rq)
    assert mx.array_equal(k, rk)
    assert mx.array_equal(v, rv)
    assert mx.array_equal(next_state, r_state)


def test_kda_norm_gate_kernel_matches_o_norm():
    from mlx_vlm.models.glm5_next.language import Glm5NextRMSNormGated
    from omlx.patches.glm53_kda_prework import kda_norm_gate_fused

    mx.random.seed(42)
    heads, dim, seq = 2, 128, 40
    y = (mx.random.normal((1, seq, heads, dim)) * 0.8).astype(mx.bfloat16)
    gate = (mx.random.normal((1, seq, heads, dim)) * 0.8).astype(mx.bfloat16)
    norm = Glm5NextRMSNormGated(dim, eps=1e-5)
    norm.weight = (mx.random.normal((dim,)) * 0.5 + 1.0).astype(mx.float32)

    fused = kda_norm_gate_fused(y, gate, norm.weight, norm.eps, heads, dim)
    reference = norm(y, gate).reshape(1, seq, heads * dim)
    mx.eval(fused, reference)
    assert mx.array_equal(fused, reference)


def _kda_tiny_config():
    from mlx_vlm.models import glm5_next

    config = _tiny_config()
    text = config.text_config
    text.linear_num_heads = 2
    text.linear_head_dim = 128
    return config


def _kda_model(seed: int):
    from mlx_vlm.models.glm5_next import language

    mx.random.seed(seed)
    config = _kda_tiny_config()
    model = language.LanguageModel(config.text_config, config)
    model.set_dtype(mx.bfloat16)
    return model, language


def test_kda_fused_prefill_matches_stock(monkeypatch):
    from omlx.patches import glm53_kda_prework as kda

    model = _kda_model(77)[0]
    prompt = mx.arange(70, dtype=mx.int32)[None] + 1

    monkeypatch.setattr(kda, "_GLM53_KDA_PREFILL_ENABLED", False)
    reference_cache = model.make_cache()
    reference = model(prompt, cache=reference_cache).logits
    mx.eval(reference)
    ref_conv = [cache[0] for cache in reference_cache]
    ref_state = [cache[1] for cache in reference_cache]

    monkeypatch.setattr(kda, "_GLM53_KDA_PREFILL_ENABLED", True)
    engaged = []
    original = kda.glm53_kda_prefill

    def spy(module, inputs, cache):
        engaged.append(int(inputs.shape[1]))
        return original(module, inputs, cache)

    monkeypatch.setattr(kda, "glm53_kda_prefill", spy)
    fused_cache = model.make_cache()
    fused = model(prompt, cache=fused_cache).logits
    fused_conv = [cache[0] for cache in fused_cache]
    fused_state = [cache[1] for cache in fused_cache]
    decoded = model(mx.array([[71]], dtype=mx.int32), cache=fused_cache).logits
    reference_decoded = model(
        mx.array([[71]], dtype=mx.int32), cache=reference_cache
    ).logits
    mx.eval(fused, decoded, reference_decoded)

    # The single linear-attention layer takes the fused route for this chunk.
    assert engaged == [70]
    assert mx.allclose(fused, reference, atol=3e-4, rtol=3e-4).item()
    for layer, (conv, ref_conv_row) in enumerate(zip(fused_conv, ref_conv)):
        state, ref_state_row = fused_state[layer], ref_state[layer]
        if not isinstance(conv, mx.array) or not isinstance(ref_conv_row, mx.array):
            continue
        assert mx.array_equal(conv, ref_conv_row), f"layer {layer} conv state"
        assert mx.allclose(state, ref_state_row, atol=1e-6, rtol=1e-6).item(), (
            f"layer {layer} recurrent state"
        )
    assert mx.allclose(decoded, reference_decoded, atol=1e-5, rtol=1e-5).item()


def test_kda_fused_chunked_prefill_matches_single_pass(monkeypatch):
    model, language = _kda_model(78)
    prompt = mx.arange(80, dtype=mx.int32)[None] + 1

    single_cache = model.make_cache()
    single = model(prompt, cache=single_cache).logits

    chunked_cache = model.make_cache()
    first = model(prompt[:, :64], cache=chunked_cache).logits
    # The narrow tail chunk runs the stock path with the fused chunk's state.
    second = model(prompt[:, 64:], cache=chunked_cache).logits
    mx.eval(single, first, second)

    assert mx.allclose(second, single[:, 64:], atol=3e-4, rtol=3e-4).item()


def test_kda_prefill_eligibility_gating(monkeypatch):
    from mlx_vlm.models.glm5_next import language
    from omlx.patches.glm53_kda_prework import (
        glm53_kda_prefill,
        glm53_kda_prefill_eligible,
    )

    model = _kda_model(79)[0]
    layer = model.model.layers[0].self_attn
    assert layer.__class__ is language.Glm5NextLinearAttention
    cache = model.make_cache()[0]
    inputs = (mx.random.normal((1, 70, model.args.hidden_size))).astype(mx.bfloat16)
    assert glm53_kda_prefill_eligible(layer, inputs, None, cache)

    assert not glm53_kda_prefill_eligible(
        layer, inputs, mx.ones((1, 70), dtype=mx.bool_), cache
    )
    assert not glm53_kda_prefill_eligible(layer, inputs[:, :63], None, cache)
    assert not glm53_kda_prefill_eligible(
        layer, inputs.astype(mx.float32), None, cache
    )
    padded = model.make_cache()[0]
    padded.lengths = mx.array([70])
    assert not glm53_kda_prefill_eligible(layer, inputs, None, padded)

    monkeypatch.setattr(
        "omlx.patches.glm53_kda_prework._GLM53_KDA_PREFILL_ENABLED", False
    )
    assert not glm53_kda_prefill_eligible(layer, inputs, None, cache)
    monkeypatch.undo()

    # The fused driver runs end to end and matches the module's own route.
    fused_cache = model.make_cache()[0]
    out_fused = glm53_kda_prefill(layer, inputs, fused_cache)
    stock_cache = model.make_cache()[0]
    monkeypatch.setattr(
        "omlx.patches.glm53_kda_prework._GLM53_KDA_PREFILL_ENABLED", False
    )
    out_stock = layer(inputs, None, stock_cache)
    mx.eval(out_fused, out_stock)
    assert mx.allclose(out_fused, out_stock, atol=3e-4, rtol=3e-4).item()
    assert mx.array_equal(fused_cache[0], stock_cache[0])


def test_kda_fused_prefill_survives_mtp_runtime_patch(monkeypatch):
    """The MTP runtime replaces the whole __call__; it must keep fusing.

    ``glm5_next_vlm_runtime.apply()`` is process-wide and sticky, so any
    test ordering that runs it first used to silently disable fused KDA
    prefill for every later forward.
    """
    from mlx_vlm.models.glm5_next import language
    from omlx.patches import glm53_kda_prework as kda
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime

    assert glm5_next_vlm_runtime.apply()
    assert getattr(language.Glm5NextLinearAttention, "_omlx_mtp_capture_patched", False)

    model = _kda_model(80)[0]
    prompt = mx.arange(70, dtype=mx.int32)[None] + 1

    monkeypatch.setattr(kda, "_GLM53_KDA_PREFILL_ENABLED", False)
    reference_cache = model.make_cache()
    reference = model(prompt, cache=reference_cache).logits

    monkeypatch.setattr(kda, "_GLM53_KDA_PREFILL_ENABLED", True)
    engaged = []
    original = kda.glm53_kda_prefill

    def spy(module, inputs, cache):
        engaged.append(int(inputs.shape[1]))
        return original(module, inputs, cache)

    monkeypatch.setattr(kda, "glm53_kda_prefill", spy)
    fused_cache = model.make_cache()
    fused = model(prompt, cache=fused_cache).logits
    mx.eval(fused, reference)

    assert engaged == [70]
    assert mx.allclose(fused, reference, atol=3e-4, rtol=3e-4).item()


@pytest.mark.parametrize("past_len", [0, 37])
def test_dense_prefix_row_blocks_bitwise(monkeypatch, past_len):
    """Causal row blocks of the dense-prefix attention reproduce the one-call
    result bitwise (keys past a block's last row are masked for all of its
    rows, so they only ever contribute exact zeros)."""
    from mlx_vlm.models.glm5_next import language

    mx.random.seed(7)
    config = _tiny_config()
    model = language.LanguageModel(config.text_config, config)
    attention = model.model.layers[1].self_attn
    heads = attention.num_heads
    rows = 2100  # rows // 256 = 8: exercises 2, 3, 4 and 8 blocks
    q = mx.random.normal((1, heads, rows, attention.q_head_dim)).astype(mx.bfloat16)
    kv = mx.random.normal((1, 1, past_len + rows, attention.kv_lora_rank)).astype(mx.bfloat16)
    total = past_len + rows
    # The engine passes a boolean causal mask sliced out of a larger one.
    full = mx.tril(mx.ones((total + 64, total + 64), dtype=mx.bool_), k=0)
    mask = full[past_len:total, :total][None, None]

    monkeypatch.setattr(language, "_DENSE_ROW_BLOCKS", 1)
    ref = attention._dense_flat(q, kv, mask, rows, past_len)
    outs = []
    for blocks in (2, 3, 4, 8):
        monkeypatch.setattr(language, "_DENSE_ROW_BLOCKS", blocks)
        outs.append(attention._dense_flat(q, kv, mask, rows, past_len))
    mx.eval(ref, outs)
    for out in outs:
        assert out.shape == ref.shape and out.dtype == ref.dtype
        assert mx.array_equal(out, ref).item()
    # Without an explicit mask the one-call "causal" path is kept.
    monkeypatch.setattr(language, "_DENSE_ROW_BLOCKS", 4)
    unmasked = attention._dense_flat(q, kv, None, rows, past_len)
    monkeypatch.setattr(language, "_DENSE_ROW_BLOCKS", 1)
    unmasked_ref = attention._dense_flat(q, kv, None, rows, past_len)
    mx.eval(unmasked, unmasked_ref)
    assert mx.array_equal(unmasked, unmasked_ref).item()


# ---------------------------------------------------------------------------
# LayerPipeline (pipelined per-layer prefill evaluation)


def _pipeline_layers(n, width=256):
    mx.random.seed(0)
    return [mx.random.normal((width, width)) * 0.05 for _ in range(n)]


@pytest.mark.parametrize("lazy_last", [False, True])
def test_layer_pipeline_matches_eager_evaluation(lazy_last):
    ws = _pipeline_layers(6)
    x0 = mx.random.normal((4, 256))
    eager = x0
    for w in ws:
        eager = mx.tanh(eager @ w)
        mx.eval(eager)
    pipe = LayerPipeline(depth=1, lazy_last=lazy_last)
    h = x0
    for w in ws:
        h = mx.tanh(h @ w)
        pipe.push(h)
    pipe.drain()
    assert mx.array_equal(h, eager).item()


@pytest.mark.parametrize("lazy_last", [False, True])
def test_layer_pipeline_bounds_in_flight_work(monkeypatch, lazy_last):
    evaluated, waited, hooks = [], [], []
    monkeypatch.setattr(mx, "async_eval", lambda *a: evaluated.append(a))
    monkeypatch.setattr(mx, "eval", lambda *a: waited.append(a))
    pipe = LayerPipeline(
        depth=1, on_evaluated=lambda: hooks.append(1), lazy_last=lazy_last
    )
    arrays = [object() for _ in range(4)]
    for a in arrays:
        pipe.push(a)
    # At most two layers in flight; with lazy_last the newest push is held.
    queued = arrays[:3] if lazy_last else arrays
    assert [e[0] for e in evaluated] == queued
    assert [w[0] for w in waited] == queued[:-1]
    pipe.drain()
    assert [w[0] for w in waited] == queued
    assert len(hooks) == len(queued)


# ---------------------------------------------------------------------------
# Fused hyper-connection prefill kernels


def _hc_modules():
    from mlx_vlm.models.deepseek_v4 import hyper_connection as dsv4_hc
    from mlx_vlm.models.glm5_next import hc_prefill, language

    return dsv4_hc, hc_prefill, language


def _hc_connection(hidden=4096, seed=0, cls=None):
    dsv4_hc, _, language = _hc_modules()
    cls = cls or language.HyperConnection
    config = SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        hidden_size=hidden,
    )
    connection = cls(config)
    key = mx.random.key(seed)
    k1, k2, k3 = mx.random.split(key, 3)
    connection.fn = mx.random.normal((24, 4 * hidden), key=k1) * 0.02
    connection.base = mx.random.normal((24,), key=k2) * 0.5
    connection.scale = mx.random.uniform(0.5, 2.0, (3,), key=k3)
    connection.eval()
    mx.eval(connection.parameters())
    return connection


def _hc_stream(length, hidden=4096, seed=1, batch=1):
    x = mx.random.normal((batch, length, 4, hidden), key=mx.random.key(seed))
    return x.astype(mx.bfloat16)


def _hc_fp32_reference_pre(connection, x):
    """Canonical math with the mixes in full fp32 (CPU matmul)."""
    dsv4_hc, _, _ = _hc_modules()
    y = x.astype(mx.float32)
    z = mx.fast.rms_norm(y.flatten(-2), None, connection.norm_eps)
    mixes = mx.matmul(z, connection.fn.T, stream=mx.cpu)
    return dsv4_hc._hc_kernel(
        x,
        y,
        mixes,
        connection.scale,
        connection.base,
        connection.hc_mult,
        connection.sinkhorn_iters,
        connection.hc_eps,
    )


def _hc_bf16_ulps(a, b):
    a = np.array(a.astype(mx.float32)).view(np.int32) >> 16
    b = np.array(b.astype(mx.float32)).view(np.int32) >> 16
    a = np.where(a < 0, -(a & 0x7FFF), a).astype(np.int64)
    b = np.where(b < 0, -(b & 0x7FFF), b).astype(np.int64)
    return np.abs(a - b)


def test_fused_pre_matches_fp32_reference_up_to_summation_order():
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(96)
    fused = hc_prefill.hc_pre(connection, x)
    assert fused is not None
    reference = _hc_fp32_reference_pre(connection, x)
    mx.eval(fused, reference)
    for got, want in zip(fused[1:], reference[1:]):
        assert got.shape == want.shape and got.dtype == want.dtype
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-5)
    assert fused[0].shape == reference[0].shape
    assert fused[0].dtype == mx.bfloat16
    # Only rounding flips (and near-cancellation) from the fp32 mix order.
    ulps = _hc_bf16_ulps(fused[0], reference[0])
    assert (ulps > 0).mean() < 0.01
    np.testing.assert_allclose(
        np.array(fused[0].astype(mx.float32)),
        np.array(reference[0].astype(mx.float32)),
        rtol=2**-7,
        atol=1e-3,
    )


@pytest.mark.parametrize("length", [2048, 777])
def test_fused_pre_is_batch_invariant(length):
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(length)
    full = hc_prefill.hc_pre(connection, x)
    pieces = [
        hc_prefill.hc_pre(connection, x[:, s : s + 256]) for s in range(0, length, 256)
    ]
    tiled = [mx.concatenate([p[i] for p in pieces], axis=1) for i in range(3)]
    shifted = hc_prefill.hc_pre(connection, x[:, 3 : length - 5])
    mx.eval(full, tiled, shifted)
    for a, b, c in zip(full, tiled, shifted):
        assert mx.array_equal(a, b)
        assert mx.array_equal(a[:, 3 : length - 5], c)


def test_fused_pre_batch_rows_match_single_requests():
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(40, batch=3)
    batched = hc_prefill.hc_pre(connection, x)
    singles = [hc_prefill.hc_pre(connection, x[i : i + 1]) for i in range(3)]
    mx.eval(batched, singles)
    for i, single in enumerate(singles):
        for a, b in zip(batched, single):
            assert mx.array_equal(a[i : i + 1], b)


@pytest.mark.parametrize("rows, threads", [(8, 1024), (32, 1024), (16, 512), (8, 256)])
def test_fused_pre_does_not_depend_on_tile_shape(monkeypatch, rows, threads):
    """The reduction order is fixed, so any tile height / thread count gives
    the same bits as the default configuration."""
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(203)
    default = hc_prefill.hc_pre(connection, x)
    mx.eval(default)
    monkeypatch.setattr(hc_prefill, "_ROWS", rows)
    monkeypatch.setattr(hc_prefill, "_THREADS", threads)
    # A launch failure disables the kernels module-wide; restore it afterwards.
    monkeypatch.setattr(hc_prefill, "_DISABLED", hc_prefill._DISABLED)
    other = hc_prefill.hc_pre(connection, x)
    if other is None:
        # The kernel fails closed where the device cannot launch this
        # threadgroup (e.g. virtual GPUs capped below 1024 threads).
        pytest.skip(
            f"this GPU cannot run {rows} rows x {threads} threads per threadgroup"
        )
    mx.eval(other)
    for a, b in zip(default, other):
        assert mx.array_equal(a, b)


@pytest.mark.parametrize("length", [64, 37])
def test_fused_expand_matches_exact_short_block_kernel(length):
    from mlx_vlm.models.fast_ops import exact_hc_expand

    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(length)
    branch = (mx.random.normal((1, length, 4096), key=mx.random.key(7)) * 0.5).astype(
        mx.bfloat16
    )
    _, post, comb = hc_prefill.hc_pre(connection, x)
    fused = hc_prefill.hc_expand(branch, x, post, comb)
    exact = exact_hc_expand(branch, x, post, comb)
    assert fused is not None and exact is not None
    mx.eval(fused, exact)
    assert fused.shape == x.shape and fused.dtype == x.dtype
    assert mx.array_equal(fused, exact)


def test_fused_expand_is_batch_invariant():
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection()
    x = _hc_stream(600)
    branch = (mx.random.normal((1, 600, 4096), key=mx.random.key(9))).astype(
        mx.bfloat16
    )
    _, post, comb = hc_prefill.hc_pre(connection, x)
    full = hc_prefill.hc_expand(branch, x, post, comb)
    pieces = [
        hc_prefill.hc_expand(
            branch[:, s : s + 256],
            x[:, s : s + 256],
            post[:, s : s + 256],
            comb[:, s : s + 256],
        )
        for s in range(0, 600, 256)
    ]
    assert mx.array_equal(full, mx.concatenate(pieces, axis=1))


def test_bitwise_equal_to_canonical_path_without_tf32():
    """With MLX_ENABLE_TF32=0 the canonical prefill path runs its matmuls in
    fp32; the fused kernels then reproduce it bit for bit on this data."""
    script = textwrap.dedent("""
        import mlx.core as mx
        from omlx.patches import mlx_vlm_glm5_next_compat as compat
        compat.apply_mlx_vlm_glm5_next_compat_patch()
        from mlx_vlm.models.deepseek_v4 import hyper_connection as dsv4_hc
        from mlx_vlm.models.glm5_next import hc_prefill
        from tests.test_mlx_vlm_glm5_next_compat import _hc_connection, _hc_stream
        c = _hc_connection(cls=dsv4_hc.HyperConnection)
        x = _hc_stream(300)
        branch = (mx.random.normal((1, 300, 4096), key=mx.random.key(3))).astype(mx.bfloat16)
        ref = c(x)
        fused = hc_prefill.hc_pre(c, x)
        ref_e = dsv4_hc.hc_expand(branch, x, ref[1], ref[2])
        fused_e = hc_prefill.hc_expand(branch, x, fused[1], fused[2])
        mx.eval(ref, fused, ref_e, fused_e)
        ok = all(bool(mx.array_equal(a, b)) for a, b in zip(ref, fused))
        ok = ok and bool(mx.array_equal(ref_e, fused_e))
        print("BITWISE", ok)
        """)
    env = dict(os.environ, MLX_ENABLE_TF32="0")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert "BITWISE True" in result.stdout, result.stdout + result.stderr[-2000:]


def test_unsupported_inputs_fall_back():
    _, hc_prefill, _ = _hc_modules()
    connection = _hc_connection(hidden=512)
    x = _hc_stream(32, hidden=512)
    assert hc_prefill.hc_pre(connection, x) is None  # 4 * 512 is not a 4096 multiple
    connection = _hc_connection()
    assert hc_prefill.hc_pre(connection, _hc_stream(32).astype(mx.float32)) is None
    connection.train()
    assert hc_prefill.hc_pre(connection, _hc_stream(32)) is None
    connection.eval()
    connection.fn = connection.fn.astype(mx.bfloat16)
    assert hc_prefill.hc_pre(connection, _hc_stream(32)) is None
    branch = mx.zeros((1, 32, 4096), dtype=mx.bfloat16)
    post = mx.zeros((1, 32, 4), dtype=mx.float32)
    comb = mx.zeros((1, 32, 4, 4), dtype=mx.float32)
    assert (
        hc_prefill.hc_expand(branch, _hc_stream(32), post.astype(mx.bfloat16), comb)
        is None
    )
    assert (
        hc_prefill.hc_expand(branch[..., :4088], _hc_stream(32)[..., :4088], post, comb)
        is None
    )


def test_layer_routes_only_prefill_blocks_to_fused_kernels(monkeypatch):
    _, hc_prefill, language = _hc_modules()
    connection = _hc_connection()
    calls = {"pre": 0, "expand": 0}
    real_pre, real_expand = hc_prefill.hc_pre, hc_prefill.hc_expand

    def count_pre(*args):
        calls["pre"] += 1
        return real_pre(*args)

    def count_expand(*args):
        calls["expand"] += 1
        return real_expand(*args)

    monkeypatch.setattr(hc_prefill, "hc_pre", count_pre)
    monkeypatch.setattr(hc_prefill, "hc_expand", count_expand)
    for length, expected in ((1, 0), (8, 0), (9, 1)):
        x = _hc_stream(length)
        before = dict(calls)
        collapsed, post, comb = connection(x)
        out = language.hc_expand(collapsed, x, post, comb)
        mx.eval(out)
        assert out.shape == x.shape
        assert calls["pre"] - before["pre"] == expected
        assert calls["expand"] - before["expand"] == expected


def test_disabled_env_keeps_canonical_path(monkeypatch):
    _, hc_prefill, _ = _hc_modules()
    monkeypatch.setattr(hc_prefill, "_DISABLED", True)
    connection = _hc_connection()
    assert hc_prefill.hc_pre(connection, _hc_stream(32)) is None
    x = _hc_stream(32)
    post = mx.zeros((1, 32, 4), dtype=mx.float32)
    comb = mx.zeros((1, 32, 4, 4), dtype=mx.float32)
    assert hc_prefill.hc_expand(x[:, :, 0], x, post, comb) is None


def test_hc_prefill_failure_latches_the_canonical_path(monkeypatch):
    """A kernel failure disables the fused path instead of retrying per call."""
    _, hc_prefill, _ = _hc_modules()
    monkeypatch.setattr(hc_prefill, "_DISABLED", False)
    attempts = []

    def broken(*args):
        attempts.append(1)
        raise RuntimeError("kernel build failed")

    monkeypatch.setattr(hc_prefill, "_kernel", broken)
    connection = _hc_connection()
    x = _hc_stream(32)
    assert hc_prefill.hc_pre(connection, x) is None
    assert hc_prefill.hc_pre(connection, x) is None
    assert attempts == [1]
    assert not hc_prefill.enabled()


# ---------------------------------------------------------------------------
# KDA prefill recurrence kernels


def _kda_inputs(B, T, H, seed):
    mx.random.seed(seed)
    D = 128

    def l2(x):
        return x * mx.rsqrt((x * x).sum(-1, keepdims=True) + 1e-6)

    q = (l2(mx.random.normal((B, T, H, D))) * D**-0.5).astype(mx.bfloat16)
    k = l2(mx.random.normal((B, T, H, D))).astype(mx.bfloat16)
    v = mx.random.normal((B, T, H, D)).astype(mx.bfloat16)
    a = (2 * mx.random.normal((B, T, H, D))).astype(mx.bfloat16)
    beta = mx.sigmoid(mx.random.normal((B, T, H)).astype(mx.bfloat16))
    a_log = mx.random.uniform(0.3, 2.0, (H,))
    dt_bias = 0.5 * mx.random.normal((H * D,))
    state = 0.1 * mx.random.normal((B, H, D, D))
    return q, k, v, a, beta, a_log, dt_bias, state


def _kda_stock(q, k, v, a, beta, a_log, dt_bias, state, mask=None, ops=False):
    from mlx_vlm.models.glm5_next import gated_delta as G

    H = q.shape[2]
    g = G.compute_g_safe(a_log.reshape(H, 1), a, dt_bias.reshape(H, 128), -5.0)
    if ops:
        return G.gated_delta_ops(q, k, v, g, beta, state, mask)
    return G.gated_delta_kernel(q, k, v, g, beta, state, mask)


@pytest.mark.parametrize(
    "cfg", [(16, 64, 2, True), (16, 32, 1, False), (8, 16, 2, True), (8, 128, 2, False)]
)
def test_recurrence_matches_stock_kernel_and_fp32_reference(cfg):
    from omlx.patches.glm53_kda_recurrence import RecurrenceConfig, kda_recurrence

    B, T, H = 2, 45, 2
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(B, T, H, 5)
    y_ker, s_ker = _kda_stock(q, k, v, a, beta, a_log, dt_bias, state)
    y_ref, s_ref = _kda_stock(q, k, v, a, beta, a_log, dt_bias, state, ops=True)
    y, s = kda_recurrence(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=RecurrenceConfig(*cfg)
    )
    mx.eval(y_ker, s_ker, y_ref, s_ref, y, s)
    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32
    # Same recurrence; only the fp32 summation order of the dots differs.
    assert mx.allclose(s, s_ker, rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(s, s_ref, rtol=1e-4, atol=1e-5).item()
    assert mx.allclose(y, y_ker, rtol=1e-2, atol=1e-3).item()
    assert mx.allclose(y, y_ref, rtol=1e-2, atol=1e-3).item()


@pytest.mark.parametrize("threadgroups", [None, 7])
def test_recurrence_chunked_equals_one_shot(threadgroups):
    from omlx.patches.glm53_kda_recurrence import PerCoreConfig, kda_recurrence

    cfg = PerCoreConfig(threadgroups=threadgroups)
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(1, 100, 2, 7)
    y_full, s_full = kda_recurrence(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=cfg
    )
    ys, s = [], state
    for lo, hi in [(0, 13), (13, 64), (64, 100)]:
        y, s = kda_recurrence(
            q[:, lo:hi],
            k[:, lo:hi],
            v[:, lo:hi],
            a[:, lo:hi],
            beta[:, lo:hi],
            a_log,
            dt_bias,
            -5.0,
            s,
            config=cfg,
        )
        ys.append(y)
    y_chunks = mx.concatenate(ys, axis=1)
    mx.eval(y_full, s_full, y_chunks, s)
    assert mx.array_equal(y_full, y_chunks).item()
    assert mx.array_equal(s_full, s).item()


@pytest.mark.parametrize(
    "B,T,H,threadgroups",
    [
        (1, 45, 4, 5),  # 102/104-row ranges: most threadgroups span two heads
        (2, 37, 3, 4),  # batch > 1, 96-row ranges
        (1, 25, 2, 2),  # one head per threadgroup, tail-only blocks
        (1, 61, 5, 6),  # ragged ranges (106/108 rows), full + tail blocks
        (1, 1, 4, 5),  # single token
        (1, 30, 64, 80),  # GLM-5.3 head count, one range per M5 Ultra core
        (1, 30, 64, None),  # default: one range per GPU core
    ],
)
def test_percore_recurrence_is_bitwise_the_blocked_kernel(B, T, H, threadgroups):
    """Same per-row arithmetic and summation order -> identical bits."""
    from omlx.patches.glm53_kda_recurrence import (
        PerCoreConfig,
        RecurrenceConfig,
        _percore_threadgroups,
        kda_recurrence,
    )

    cfg = PerCoreConfig(threadgroups=threadgroups)
    ntg = _percore_threadgroups(H * 128, cfg)
    if threadgroups is None and ntg is None:
        pytest.skip("per-core ranges do not cover the rows on this GPU")
    assert ntg is not None
    _kda_skip_unless_launchable(cfg, H, ntg)
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(B, T, H, 11 + T)
    y, s = kda_recurrence(q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=cfg)
    y_b, s_b = kda_recurrence(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=RecurrenceConfig()
    )
    mx.eval(y, s, y_b, s_b)
    assert y.dtype == mx.bfloat16 and s.dtype == mx.float32
    assert mx.array_equal(y, y_b).item()
    assert mx.array_equal(s, s_b).item()


def _kda_skip_unless_launchable(cfg, H, ntg):
    from omlx.patches.glm53_kda_recurrence import _percore_launchable

    max_rows = 2 * -(-(H * 128 // 2) // ntg)
    if not _percore_launchable(cfg.tb, max_rows, mx.bfloat16, 128, 128, H, ntg):
        pytest.skip(
            f"this GPU cannot launch the per-core kernel with {max_rows * 8} threads"
            " per threadgroup; the blocked kernel runs instead"
        )


def test_percore_recurrence_matches_stock_kernel_and_fp32_reference():
    from omlx.patches.glm53_kda_recurrence import (
        PerCoreConfig,
        _percore_threadgroups,
        kda_recurrence,
    )

    B, T, H = 2, 45, 2
    cfg = PerCoreConfig(threadgroups=3)
    _kda_skip_unless_launchable(cfg, H, _percore_threadgroups(H * 128, cfg))
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(B, T, H, 5)
    y_ker, s_ker = _kda_stock(q, k, v, a, beta, a_log, dt_bias, state)
    y_ref, s_ref = _kda_stock(q, k, v, a, beta, a_log, dt_bias, state, ops=True)
    y, s = kda_recurrence(q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=cfg)
    mx.eval(y_ker, s_ker, y_ref, s_ref, y, s)
    assert mx.allclose(s, s_ker, rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(s, s_ref, rtol=1e-4, atol=1e-5).item()
    assert mx.allclose(y, y_ker, rtol=1e-2, atol=1e-3).item()
    assert mx.allclose(y, y_ref, rtol=1e-2, atol=1e-3).item()


def test_percore_recurrence_falls_back_to_blocked(monkeypatch):
    from omlx.patches import glm53_kda_recurrence as R

    # More than 128 rows per threadgroup (would span three heads): blocked kernel.
    assert R._percore_threadgroups(5 * 128, R.PerCoreConfig(threadgroups=3)) is None
    # One range per core only while a core gets at most 128 rows.
    monkeypatch.setattr(R, "gpu_core_count", lambda: 80)
    assert R._percore_threadgroups(64 * 128, R.PerCoreConfig()) == 80
    monkeypatch.setattr(R, "gpu_core_count", lambda: 40)
    assert R._percore_threadgroups(64 * 128, R.PerCoreConfig()) is None
    # Unknown core count, fp32 activations: blocked kernel, same bits.
    monkeypatch.setattr(R, "gpu_core_count", lambda: None)
    assert R._percore_threadgroups(64 * 128, R.PerCoreConfig()) is None
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(1, 20, 3, 2)
    real = R._blocked
    for args in [(q, k, v), tuple(x.astype(mx.float32) for x in (q, k, v))]:
        launched = []
        monkeypatch.setattr(R, "_blocked", lambda *xs: launched.append(1) or real(*xs))
        y, s = R.kda_recurrence(*args, a, beta, a_log, dt_bias, -5.0, state)
        y_b, s_b = real(*args, a, beta, a_log, dt_bias, -5.0, state, R.DEFAULT_CONFIG)
        mx.eval(y, s, y_b, s_b)
        assert launched == [1]
        assert mx.array_equal(y, y_b).item() and mx.array_equal(s, s_b).item()


def _kda_attention(num_heads=4, hidden=256, seed=0, bits=None):
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextLinearAttention

    config = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=hidden,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=128,
        rms_norm_eps=1e-5,
        index_topk=4,
        index_head_dim=8,
        index_n_heads=2,
        layer_types=["linear_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": num_heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    mx.random.seed(seed)
    attn = Glm5NextLinearAttention(config)
    params = []
    for name, p in nn.utils.tree_flatten(attn.parameters()):
        if name.endswith("A_log"):
            params.append((name, mx.random.uniform(0.3, 2.0, p.shape)))
        elif name.endswith("dt_bias"):
            params.append((name, mx.random.normal(p.shape) * 0.5))
        elif name.endswith("o_norm.weight"):
            params.append((name, 1 + 0.1 * mx.random.normal(p.shape)))
        else:
            params.append((name, mx.random.normal(p.shape) * p.shape[-1] ** -0.5))
    attn.load_weights(params)
    if bits:
        nn.quantize(
            attn,
            group_size=64,
            bits=bits,
            class_predicate=lambda _, m: isinstance(m, nn.Linear),
        )
    # Checkpoint dtype policy: bf16 except the fp32 gate parameters.
    attn.set_dtype(mx.bfloat16)
    fg = attn.forget_gate
    fg.A_log = fg.A_log.astype(mx.float32)
    fg.dt_bias = fg.dt_bias.astype(mx.float32)
    mx.eval(attn.parameters())
    return config, attn


def _kda_run(attn, x, conv0, state0, fused, chunks=None):
    from mlx_vlm.models.cache import ArraysCache

    from omlx.patches import glm53_kda_prework as kda

    enabled = kda._GLM53_KDA_PREFILL_ENABLED
    kda._GLM53_KDA_PREFILL_ENABLED = fused
    try:
        cache = ArraysCache(size=2)
        cache[0] = conv0
        cache[1] = state0
        outs, t = [], 0
        for n in chunks or [x.shape[1]]:
            outs.append(attn(x[:, t : t + n], None, cache))
            t += n
        out = mx.concatenate(outs, axis=1)
        mx.eval(out, cache[0], cache[1])
        return out, cache[0], cache[1]
    finally:
        kda._GLM53_KDA_PREFILL_ENABLED = enabled


def test_fused_prefill_runs_blocked_recurrence(monkeypatch):
    from mlx_vlm.models.glm5_next import gated_delta as G

    from omlx.patches import glm53_kda_prework as kda

    config, attn = _kda_attention()
    H, D = attn.num_heads, attn.head_dim
    mx.random.seed(11)
    x = mx.random.normal((1, 150, config.hidden_size)).astype(mx.bfloat16)
    conv0 = mx.random.normal((1, 3, attn.conv_dim)).astype(mx.bfloat16)
    state0 = 0.05 * mx.random.normal((1, H, D, D))

    calls = []
    real = kda.kda_recurrence

    def counted(*args, **kwargs):
        calls.append(args[0].shape)
        return real(*args, **kwargs)

    monkeypatch.setattr(kda, "kda_recurrence", counted)
    stock = _kda_run(attn, x, conv0, state0, fused=False)
    assert not calls
    fused = _kda_run(attn, x, conv0, state0, fused=True)
    assert calls == [(1, 150, H, D)]
    assert mx.array_equal(stock[1], fused[1]).item()  # conv state
    assert mx.allclose(stock[2], fused[2], rtol=1e-5, atol=1e-6).item()
    assert mx.allclose(stock[0], fused[0], rtol=2e-2, atol=2e-3).item()

    # With the stock recurrence swapped in, the fused route is bit-identical.
    def stock_recurrence(q, k, v, a, beta, a_log, dt_bias, lb, state, config=None):
        g = G.compute_g_safe(a_log.reshape(H, 1), a, dt_bias.reshape(H, D), lb)
        return G.gated_delta_kernel(q, k, v, g, beta, state)

    monkeypatch.setattr(kda, "kda_recurrence", stock_recurrence)
    exact = _kda_run(attn, x, conv0, state0, fused=True)
    for s, e in zip(stock, exact):
        assert mx.array_equal(s, e).item()

    # Chunked prefill carries conv and recurrent state exactly.
    monkeypatch.setattr(kda, "kda_recurrence", real)
    chunked = _kda_run(attn, x, conv0, state0, fused=True, chunks=[64, 86])
    assert mx.array_equal(chunked[1], fused[1]).item()
    assert mx.array_equal(chunked[2], fused[2]).item()


def test_fused_prefill_keeps_stock_recurrence_for_non_fp32_gate(monkeypatch):
    from omlx.patches import glm53_kda_prework as kda

    config, attn = _kda_attention(seed=4)
    attn.forget_gate.dt_bias = attn.forget_gate.dt_bias.astype(mx.bfloat16)
    calls = []
    monkeypatch.setattr(kda, "kda_recurrence", lambda *a, **k: calls.append(1))
    x = mx.random.normal((1, 80, config.hidden_size)).astype(mx.bfloat16)
    out = _kda_run(attn, x, None, None, fused=True)[0]
    assert out.shape == (1, 80, config.hidden_size) and not calls


def test_fused_prefill_leaves_stock_shaped_caches():
    """Decode paths consume cache[0]/cache[1]; they must look like stock's."""
    config, attn = _kda_attention(seed=3)
    H, D = attn.num_heads, attn.head_dim
    x = mx.random.normal((1, 96, config.hidden_size)).astype(mx.bfloat16)
    stock = _kda_run(attn, x, None, None, fused=False)
    fused = _kda_run(attn, x, None, None, fused=True)
    assert fused[1].shape == stock[1].shape == (1, 3, attn.conv_dim)
    assert fused[1].dtype == stock[1].dtype == mx.bfloat16
    assert fused[2].shape == stock[2].shape == (1, H, D, D)
    assert fused[2].dtype == stock[2].dtype == mx.float32
    assert mx.array_equal(fused[1], stock[1]).item()
    assert mx.allclose(fused[2], stock[2], rtol=1e-5, atol=1e-6).item()


def test_prework_reads_qkv_in_place_from_the_fused_projection():
    """A wider input (the whole fused projection) gives the concat's bits."""
    from omlx.patches.glm53_kda_prework import kda_prework_fused

    mx.random.seed(11)
    heads, dim, length = 4, 128, 37
    c_dim = 3 * heads * dim
    fused = (mx.random.normal((1, length, c_dim + 320)) * 0.5).astype(mx.bfloat16)
    conv_state = (mx.random.normal((1, 3, c_dim)) * 0.5).astype(mx.bfloat16)
    conv_w = (mx.random.normal((c_dim, 1, 4)) * 0.3).astype(mx.bfloat16)
    scale = mx.array(dim**-0.5, dtype=mx.float32)
    wide = kda_prework_fused(fused, conv_state, conv_w, scale, length, heads, dim)
    packed = kda_prework_fused(
        mx.contiguous(fused[..., :c_dim]), conv_state, conv_w, scale, length, heads, dim
    )
    for a, b in zip(wide, packed):
        assert a.shape == b.shape
        assert mx.array_equal(a, b).item()


def test_percore_recurrence_falls_back_when_the_launch_is_rejected(monkeypatch):
    """A GPU that rejects the per-core threadgroup size keeps the blocked kernel."""
    import omlx.patches.glm53_kda_recurrence as rec

    monkeypatch.setattr(rec, "_percore_launchable", lambda *a: False)
    B, T, H = 1, 30, 2
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(B, T, H, 3)
    y, s = rec.kda_recurrence(
        q,
        k,
        v,
        a,
        beta,
        a_log,
        dt_bias,
        -5.0,
        state,
        config=rec.PerCoreConfig(threadgroups=3),
    )
    y_b, s_b = rec.kda_recurrence(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, config=rec.RecurrenceConfig()
    )
    mx.eval(y, s, y_b, s_b)
    assert mx.array_equal(y, y_b).item() and mx.array_equal(s, s_b).item()


@pytest.mark.parametrize("percore_default", [False, True])
def test_kda_default_recurrence_uses_percore_only_on_nax_hosts(
    monkeypatch, percore_default
):
    """Without NAX the default is the blocked kernel (per-core is slower on an
    80-core M3 Ultra); on NAX hosts the per-core kernel runs when it covers."""
    from omlx.patches import glm53_kda_recurrence as R

    monkeypatch.setattr(R, "_PERCORE_DEFAULT", percore_default)
    monkeypatch.setattr(R, "gpu_core_count", lambda: 80)
    launched = []
    real_percore = R._percore
    monkeypatch.setattr(
        R, "_percore", lambda *xs: launched.append(1) or real_percore(*xs)
    )
    q, k, v, a, beta, a_log, dt_bias, state = _kda_inputs(1, 20, 3, 2)
    y, s = R.kda_recurrence(q, k, v, a, beta, a_log, dt_bias, -5.0, state)
    y_b, s_b = R._blocked(
        q, k, v, a, beta, a_log, dt_bias, -5.0, state, R.DEFAULT_CONFIG
    )
    mx.eval(y, s, y_b, s_b)
    assert launched == ([1] if percore_default else [])
    assert mx.array_equal(y, y_b).item() and mx.array_equal(s, s_b).item()


# ---------------------------------------------------------------------------
# Tensor-unit (NAX) DSA indexer scores

_needs_nax_indexer = pytest.mark.skipif(
    not indexer_nax.nax_indexer_available(), reason="needs an M5 (NAX) GPU"
)


def _bf16_ulp_distance(a: mx.array, b: mx.array) -> mx.array:
    ai = a.view(mx.int16).astype(mx.int32)
    bi = b.view(mx.int16).astype(mx.int32)
    ai = mx.where(ai < 0, -32768 - ai, ai)
    bi = mx.where(bi < 0, -32768 - bi, bi)
    return mx.abs(ai - bi)


def _ixn_reference(q, k, w, before, pool_len, ratio):
    """fp32 scores, heads accumulated in order, masked like the call site."""
    S, H, _ = q.shape
    P = k.shape[0]
    qf, kf, wf = (a.astype(mx.float32) for a in (q, k, w))
    acc = mx.zeros((S, P), mx.float32)
    for h in range(H):
        acc = acc + mx.maximum(qf[:, h] @ kf.T, 0.0) * wf[:, h : h + 1]
    s = mx.arange(S)[:, None]
    p = mx.arange(P)[None]
    valid = (p < pool_len) & ((p + 1) * ratio - 1 <= before + s)
    return acc, valid


def _ixn_inputs(S, P, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((S, 32, 128)) * 0.5).astype(mx.bfloat16)
    k = (mx.random.normal((P, 128)) * 0.5).astype(mx.bfloat16)
    w = (mx.random.normal((S, 32)) * 0.1).astype(mx.bfloat16)
    return q, k, w


@_needs_nax_indexer
@pytest.mark.parametrize(
    "S,P,before",
    [
        (64, 64, 192),
        (100, 300, 1000),
        (511, 1024, 3584),
        (257, 777, 3000),
        (33, 600, 2400),
    ],
)
def test_nax_indexer_scores_match_fp32_reference_and_mask(S, P, before):
    q, k, w = _ixn_inputs(S, P)
    pool_len = min(P, (before + S) // 4)
    out = indexer_nax.indexer_scores_nax(q, k, w, before, pool_len, 4)
    acc, valid = _ixn_reference(q, k, w, before, pool_len, 4)
    mx.eval(out, acc, valid)
    assert out.shape == (S, P) and out.dtype == mx.bfloat16
    # Masked entries carry exactly the sentinel the old mx.where wrote.
    sentinel = mx.where(valid, out, mx.array(-1e30, mx.bfloat16))
    assert mx.array_equal(out.view(mx.int16), sentinel.view(mx.int16)).item()
    # Live entries: the fp32 sum rounded to bf16, up to summation order.
    ref = acc.astype(mx.bfloat16)
    ulp = mx.where(valid, _bf16_ulp_distance(out, ref), 0)
    scale = mx.abs(mx.where(valid, acc, 0)).max(axis=-1, keepdims=True)
    err = mx.where(valid, mx.abs(out.astype(mx.float32) - acc), 0)
    # Within one bf16 ulp of the value, or tiny relative to the row scale
    # (cancellation near zero).
    ok = (ulp <= 1) | (err <= 1e-4 * scale)
    assert mx.all(ok).item()


@_needs_nax_indexer
def test_nax_indexer_scores_match_native_kernel_within_rounding():
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_indexer_scores"):
        pytest.skip("native DSA indexer kernel unavailable")
    S, P, before = 512, 1024, 3584
    q, k, w = _ixn_inputs(S, P, seed=1)
    pool_len = P
    out = indexer_nax.indexer_scores_nax(q, k, w, before, pool_len, 4)
    native = fast.dsa_indexer_scores(
        q[None].transpose(0, 2, 1, 3), k[None, None], w[None], causal=False
    )[0, 0]
    acc, valid = _ixn_reference(q, k, w, before, pool_len, 4)
    native = mx.where(valid, native, -1e30)
    mx.eval(out, native, acc)
    scale = mx.abs(acc).max(axis=-1, keepdims=True)
    diff = mx.abs(out.astype(mx.float32) - native.astype(mx.float32))
    ulp = _bf16_ulp_distance(out, native)
    assert mx.all((ulp <= 2) | (diff <= 1e-4 * scale)).item()


def test_nax_indexer_unsupported_inputs_return_none():
    q, k, w = _ixn_inputs(8, 16)
    assert indexer_nax.indexer_scores_nax(q.astype(mx.float16), k, w, 0, 16, 4) is None
    assert indexer_nax.indexer_scores_nax(q[..., :64], k, w, 0, 16, 4) is None
    assert indexer_nax.indexer_scores_nax(q, k, w[:4], 0, 16, 4) is None


def test_nax_indexer_row_cap_is_a_multiple_of_64():
    assert indexer_nax.max_rows_per_call(1) % 64 == 0
    assert indexer_nax.max_rows_per_call(1 << 30) == 64
    assert indexer_nax.max_rows_per_call(16384) == (1 << 27) // 16384


def _make_indexer():
    from mlx.utils import tree_map
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextIndexer

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=8,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=8,
        qk_nope_head_dim=8,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    mx.random.seed(3)
    indexer = Glm5NextIndexer(text)
    indexer.index_kpool_compress_gate = mx.random.normal((128, 64)) * 0.1
    indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    indexer.update(tree_map(lambda a: a.astype(mx.bfloat16), indexer.parameters()))
    return indexer


def _run_indexer(indexer, chunks, seed=5):
    from mlx_lm.models.cache import KVCache, PoolingCache

    mx.random.seed(seed)
    pool = PoolingCache(4)
    kv = KVCache()
    outs = []
    for n in chunks:
        x = mx.random.normal((1, n, 64)).astype(mx.bfloat16)
        qr = mx.random.normal((1, n, 32)).astype(mx.bfloat16)
        kv.update_and_fetch(
            mx.zeros((1, 1, n, 8), mx.bfloat16), mx.zeros((1, 1, n, 0), mx.bfloat16)
        )
        out = indexer(x, qr, None, cache=pool, kv_cache=kv)
        if out is not None:
            mx.eval(out)
        outs.append(out)
    return outs


def _require_native_scores():
    """The non-NAX indexer path these tests compare against scores with the
    native kernel; without it that path falls back to MLX ops, which round
    the scores differently, so the exactness and near-tie bounds do not
    apply."""
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_indexer_scores"):
        pytest.skip("native DSA indexer kernel unavailable")


def _native_masked_scores(q, pool_keys, weights, before, pool_len, ratio):
    """The previous call-site computation: native kernel + mx.where mask."""
    from mlx_vlm.models.glm5_next import language

    idx = language.Glm5NextIndexer.__new__(language.Glm5NextIndexer)
    idx.n_heads = q.shape[1]
    idx.head_dim = q.shape[2]
    scores = language.Glm5NextIndexer._native_scores(
        idx, q[None], pool_keys[None], weights[None]
    )[0]
    S, P = scores.shape
    s = mx.arange(S)[:, None]
    p = mx.arange(P)[None]
    valid = (p < pool_len) & ((p + 1) * ratio - 1 <= before + s)
    return mx.where(valid, scores, -1e30)


def test_indexer_fast_path_plumbing_is_exact(monkeypatch):
    """With the old score computation plugged in, the all-rows fast path
    returns bit-identical top-k indices to the 512-row loop."""
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700, 1500]

    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = _run_indexer(indexer, chunks)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", _native_masked_scores)
    got = _run_indexer(indexer, chunks)

    assert expected[0] is not None
    for e, g in zip(expected, got):
        assert e.shape == g.shape and e.dtype == g.dtype
        assert mx.array_equal(e, g).item()


@_needs_nax_indexer
def test_indexer_nax_selection_matches_up_to_near_ties(monkeypatch):
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700]

    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = _run_indexer(indexer, chunks)
    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    got = _run_indexer(indexer, chunks)

    for e, g in zip(expected, got):
        e = mx.sort(e[0, 0], axis=-1)
        g = mx.sort(g[0, 0], axis=-1)
        rows_equal = mx.all(e == g, axis=-1)
        # bf16 scores tie often; the kernels differ only in fp32 summation
        # order, so at most a few rows may pick a different tie member.
        assert mx.mean(rows_equal.astype(mx.float32)).item() > 0.97


@_needs_nax_indexer
def test_indexer_row_chunking_is_exact(monkeypatch):
    """A score-buffer cap that splits the rows gives identical indices."""
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    chunks = [2600, 700]
    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    whole = _run_indexer(indexer, chunks)
    monkeypatch.setattr(indexer_nax, "_MAX_SCORE_ELEMENTS", 64 * 700)
    split = _run_indexer(indexer, chunks)
    for a, b in zip(whole, split):
        assert mx.array_equal(a, b).item()


@_needs_nax_indexer
def test_topk_differences_are_threshold_near_ties():
    """Rows where the NAX scores select a different pool set than the native
    scores differ only by pools whose native scores sit within one bf16 ulp
    of that row's top-k threshold (fp32 summation order at a near-tie)."""
    from mlx_vlm.models.glm5_next import language
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dsa_topk_indices"):
        pytest.skip("native top-k kernel unavailable")
    S, P, before = 512, 4096, 16384 - 512
    q, k, w = _ixn_inputs(S, P, seed=11)
    native = _native_masked_scores(q, k, w, before, P, 4)
    nax = indexer_nax.indexer_scores_nax(q, k, w, before, P, 4)
    sel_n = language.Glm5NextIndexer._native_topk(native[None], 512)[0]
    sel_x = language.Glm5NextIndexer._native_topk(nax[None], 512)[0]
    vals_n = mx.sort(mx.take_along_axis(native, sel_n, axis=-1), axis=-1)
    vals_x = mx.sort(mx.take_along_axis(native, sel_x, axis=-1), axis=-1)
    mx.eval(vals_n, vals_x)
    # Same multiset of native scores up to one ulp at the threshold.
    ulp = _bf16_ulp_distance(vals_n, vals_x)
    assert mx.max(ulp).item() <= 1
    same = mx.all(mx.sort(sel_n, axis=-1) == mx.sort(sel_x, axis=-1), axis=-1)
    assert mx.mean(same.astype(mx.float32)).item() > 0.9


@_needs_nax_indexer
def test_indexer_without_cache(monkeypatch):
    """Cache-less prefill (positions from 0) takes the NAX path and selects
    the same pools as the native path (up to near ties)."""
    _require_native_scores()
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    mx.random.seed(9)
    x = mx.random.normal((1, 2600, 64)).astype(mx.bfloat16)
    qr = mx.random.normal((1, 2600, 32)).astype(mx.bfloat16)
    monkeypatch.setattr(language, "nax_indexer_available", lambda: False)
    expected = indexer(x, qr, None, cache=None, kv_cache=None)
    calls = []
    orig = indexer_nax.indexer_scores_nax

    def spy(*args, **kwargs):
        calls.append(args[3])  # before
        return orig(*args, **kwargs)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", spy)
    got = indexer(x, qr, None, cache=None, kv_cache=None)
    assert calls == [0]
    assert got.shape == expected.shape and got.dtype == expected.dtype
    e = mx.sort(expected[0, 0], axis=-1)
    g = mx.sort(got[0, 0], axis=-1)
    rows_equal = mx.all(e == g, axis=-1)
    assert mx.mean(rows_equal.astype(mx.float32)).item() > 0.97


@_needs_nax_indexer
def test_nax_indexer_score_from_matches_suffix(monkeypatch):
    """With the dense-prefix bypass the attention layer scores only the rows
    from ``score_from`` on; the NAX path must return exactly the suffix of
    the all-rows selection (rows are scored independently)."""
    from mlx_vlm.models.glm5_next import language

    indexer = _make_indexer()
    mx.random.seed(21)
    x = mx.random.normal((1, 2600, 64)).astype(mx.bfloat16)
    qr = mx.random.normal((1, 2600, 32)).astype(mx.bfloat16)
    calls = []
    orig = indexer_nax.indexer_scores_nax

    def spy(*args, **kwargs):
        calls.append((args[0].shape[0], args[3]))  # (rows, first row position)
        return orig(*args, **kwargs)

    monkeypatch.setattr(language, "nax_indexer_available", lambda: True)
    monkeypatch.setattr(language, "indexer_scores_nax", spy)
    full = indexer(x, qr, None)
    tail = indexer(x, qr, None, score_from=2051)
    mx.eval(full, tail)
    assert calls == [(2600, 0), (549, 2051)]
    assert tail.shape[:3] == (1, 1, 549)
    assert mx.array_equal(tail, full[:, :, 2051:]).item()
    assert indexer(x, qr, None, score_from=2600) is None


# ---------------------------------------------------------------------------
# Tensor-unit (NAX) sparse MLA prefill attention

_needs_nax_sparse_mla = pytest.mark.skipif(
    not sparse_mla_nax.nax_sparse_mla_available(), reason="needs an M5 (NAX) GPU"
)


def _smla_reference(q_latent, kv_latent, topk, scale):
    """fp32 causal attention of every head over its query's selected rows."""
    _, H, L, D = q_latent.shape
    K = kv_latent.shape[2]
    idx = topk[0, 0].astype(mx.int32)
    valid = (idx >= 0) & (idx < K) & (idx <= (mx.arange(L) + (K - L))[:, None])
    keys = kv_latent[0, 0].astype(mx.float32)[mx.where(valid, idx, 0)]
    q = q_latent[0].swapaxes(0, 1).astype(mx.float32)
    scores = (q @ keys.swapaxes(-1, -2)) * scale
    scores = mx.where(valid[:, None, :], scores, -mx.inf)
    out = mx.softmax(scores, axis=-1) @ keys
    return out.swapaxes(0, 1)[None]


def _smla_inputs(L, K, topk, H=64, q_scale=1.0, dtype=mx.bfloat16, seed=0):
    mx.random.seed(seed)
    q = (mx.random.normal((1, H, L, 512)) * q_scale).astype(dtype)
    kv = mx.random.normal((1, 1, K, 512)).astype(dtype)
    pos = (K - L) + mx.arange(L)[:, None]
    idx = (mx.random.uniform(shape=(L, topk)) * (pos + 1)).astype(mx.int32)
    # Unused slots of every kind: negative, past the query, beyond the cache.
    u = mx.random.uniform(shape=(L, topk))
    idx = mx.where(u < 0.02, -1, idx)
    idx = mx.where((u >= 0.02) & (u < 0.03), pos + 1 + (idx % 7), idx)
    idx = mx.where((u >= 0.03) & (u < 0.035), K + 5, idx)
    return q, kv, idx[None, None]


def _smla_native(q, kv, idx, scale):
    zq = mx.zeros(q.shape[:-1] + (64,), q.dtype)
    zk = mx.zeros(kv.shape[:-1] + (64,), kv.dtype)
    return sparse_mla.sparse_mla_attention(q, zq, kv, zk, idx, scale)


@_needs_nax_sparse_mla
@pytest.mark.parametrize(
    "L,K,topk,q_scale",
    [
        (64, 256, 128, 1.0),
        (37, 700, 200, 1.0),
        (128, 4096, 2051, 1.0),
        (96, 8192, 2051, 3.0),
    ],
)
def test_nax_sparse_mla_matches_fp32_reference_like_native_kernel(L, K, topk, q_scale):
    q, kv, idx = _smla_inputs(L, K, topk, q_scale=q_scale)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _smla_reference(q, kv, idx, scale)
    mx.eval(out, ref)
    assert out.shape == q.shape and out.dtype == q.dtype
    err = mx.abs(out.astype(mx.float32) - ref)
    # Output rounding: half a bf16 ulp of each value, plus fp32 slack.
    bound = mx.abs(ref) * 2.0**-8 + 1e-3
    assert mx.all(err <= bound).item()
    native = _smla_native(q, kv, idx, scale)
    if native is not None:
        n_err = mx.abs(native.astype(mx.float32) - ref)
        # Same arithmetic as the native kernel, different summation order.
        assert mx.mean(err).item() <= 1.05 * mx.mean(n_err).item() + 1e-6
        assert mx.max(err).item() <= 1.05 * mx.max(n_err).item() + 1e-3


@_needs_nax_sparse_mla
def test_nax_sparse_mla_fp16_inputs():
    q, kv, idx = _smla_inputs(40, 1024, 300, dtype=mx.float16)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _smla_reference(q, kv, idx, scale)
    assert out.dtype == mx.float16
    err = mx.abs(out.astype(mx.float32) - ref)
    assert mx.all(err <= mx.abs(ref) * 2.0**-10 + 1e-3).item()


@_needs_nax_sparse_mla
def test_nax_sparse_mla_uint32_indices_match_int32():
    q, kv, idx = _smla_inputs(32, 512, 128)
    scale = 256**-0.5
    a = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    b = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx.astype(mx.uint32), scale)
    assert mx.array_equal(a, b).item()


@_needs_nax_sparse_mla
def test_nax_sparse_mla_unsupported_shapes_return_none():
    q, kv, idx = _smla_inputs(16, 64, 32)
    scale = 1.0
    f = sparse_mla_nax.sparse_mla_attention_nax
    assert f(q[..., :256], kv[..., :256], idx, scale) is None  # latent width
    assert f(q[:, :16], kv, idx, scale) is None  # heads not a multiple of 32
    assert f(mx.concatenate([q, q]), kv, idx, scale) is None  # batch
    assert f(q, kv, idx[:, :, :8], scale) is None  # rows mismatch
    assert f(q.astype(mx.float32), kv.astype(mx.float32), idx, scale) is None
    assert f(q, kv[:, :, :8], idx, scale) is None  # fewer keys than queries


@_needs_nax_sparse_mla
def test_nax_sparse_mla_disabled_by_env(monkeypatch):
    monkeypatch.setattr(sparse_mla_nax, "_ENABLED", False)
    sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    try:
        q, kv, idx = _smla_inputs(16, 64, 32)
        assert sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, 1.0) is None
    finally:
        sparse_mla_nax.nax_sparse_mla_available.cache_clear()


@_needs_nax_sparse_mla
def test_nax_sparse_mla_deterministic_across_runs():
    # Large enough to keep many threadgroups in flight; every run must be
    # bit-identical (guards against threadgroup-memory races).
    q, kv, idx = _smla_inputs(256, 8192, 2051, q_scale=3.0, seed=4)
    scale = 256**-0.5
    outs = [
        sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale) for _ in range(6)
    ]
    mx.eval(outs)
    for o in outs[1:]:
        assert mx.array_equal(outs[0], o).item()
    ref = _smla_reference(q, kv, idx, scale)
    err = mx.abs(outs[0].astype(mx.float32) - ref)
    assert mx.all(err <= mx.abs(ref) * 2.0**-8 + 1e-3).item()


@_needs_nax_sparse_mla
def test_nax_sparse_mla_probabilities_keep_fp32_precision():
    """The PV product must use the fp32 probabilities (like the native
    kernel), not a reduced-precision copy: a value column of large
    alternating +-1000 entries cancels in the weighted sum and exposes any
    rounding of the probabilities (~11-bit rounding shows up as >> 1 ulp)."""
    L, K, topk = 32, 4096, 2051
    q, kv, idx = _smla_inputs(L, K, topk, seed=7)
    q = q.astype(mx.float32)
    q[..., 0] = 0.0  # the big column must not influence the scores
    q = q.astype(mx.bfloat16)
    kv = kv.astype(mx.float32)
    sign = mx.where(mx.arange(K) % 2 == 0, 1.0, -1.0)
    kv[0, 0, :, 0] = 1000.0 * sign
    kv = kv.astype(mx.bfloat16)
    scale = 256**-0.5
    out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    ref = _smla_reference(q, kv, idx, scale)
    mx.eval(out, ref)
    col = ref[..., 0]
    err = mx.abs(out[..., 0].astype(mx.float32) - col)
    # fp32 probabilities: the error is the bf16 rounding of the result
    # (half an ulp) plus fp32 summation noise of 1000 * sqrt(topk) * 2^-24.
    ulp = mx.power(2.0, mx.floor(mx.log2(mx.maximum(mx.abs(col), 1e-3))) - 7)
    assert mx.all(err <= 0.5 * ulp + 0.02).item()


def _make_sparse_attention():
    from mlx.utils import tree_map
    from mlx_vlm.models import glm5_next
    from mlx_vlm.models.glm5_next.language import Glm5NextSparseAttention

    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=32,
        n_shared_experts=None,
        n_routed_experts=None,
        routed_scaling_factor=1.0,
        kv_lora_rank=512,
        q_lora_rank=32,
        qk_rope_head_dim=0,
        v_head_dim=16,
        qk_nope_head_dim=16,
        num_experts_per_tok=2,
        first_k_dense_replace=99,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        layer_types=["deepseek_sparse_attention"],
        mlp_layer_types=["dense"],
        linear_attn_config={
            "num_heads": 2,
            "head_dim": 32,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4,
        hc_mult=2,
        hc_sinkhorn_iters=2,
    )
    mx.random.seed(3)
    attn = Glm5NextSparseAttention(text)
    attn.indexer.index_kpool_compress_gate = mx.random.normal((128, 64)) * 0.1
    attn.indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    attn.update(tree_map(lambda a: a.astype(mx.bfloat16), attn.parameters()))
    return attn


def _run_attention(attn, chunks, seed=5):
    from mlx_lm.models.cache import KVCache, PoolingCache

    mx.random.seed(seed)
    cache = [KVCache(), PoolingCache(4)]
    outs = []
    for n in chunks:
        x = mx.random.normal((1, n, 64)).astype(mx.bfloat16)
        out = attn(x, None, cache)
        mx.eval(out)
        outs.append(out)
    return outs


@_needs_nax_sparse_mla
def test_nax_sparse_mla_call_site_matches_fallback_paths(monkeypatch):
    """The GLM-5.3 call site with the NAX kernel matches the previous paths:
    the native kernel at >= 4096 keys (same latent-space math) and the
    expanded exact-block attention below (a reassociation of the same
    products, so bf16 intermediate rounding differs)."""
    attn = _make_sparse_attention()
    chunks = [2600, 1600]  # sparse at 2600 keys, then at 4200 keys
    monkeypatch.setattr(sparse_mla_nax, "_ENABLED", False)
    sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    try:
        expected = _run_attention(attn, chunks)
    finally:
        monkeypatch.setattr(sparse_mla_nax, "_ENABLED", True)
        sparse_mla_nax.nax_sparse_mla_available.cache_clear()
    calls = []
    orig = sparse_mla_nax.sparse_mla_attention_nax

    def spy(*args, **kwargs):
        out = orig(*args, **kwargs)
        calls.append(out is not None)
        return out

    from mlx_vlm.models.glm5_next import language

    monkeypatch.setattr(language, "sparse_mla_attention_nax", spy)
    got = _run_attention(attn, chunks)
    assert calls == [True, True]
    for e, g, tol in zip(expected, got, (2e-2, 1e-2)):
        assert e.shape == g.shape and e.dtype == g.dtype
        e32, g32 = e.astype(mx.float32), g.astype(mx.float32)
        scale = mx.abs(e32).max().item()
        assert mx.abs(e32 - g32).max().item() <= tol * scale


def _smla_realistic_inputs(L, K, H=64, dtype=mx.bfloat16, seed=0, topk_blocks=512):
    """Indexer-like top-k rows: 4-token blocks (sinks, a drifting scattered
    set, the most recent blocks) in ascending order, then the 3 causal tail
    slots; unused slots (-1) sort last."""
    rng = np.random.default_rng(seed)
    mx.random.seed(seed)
    q = mx.random.normal((1, H, L, 512)).astype(dtype)
    kv = mx.random.normal((1, 1, K, 512)).astype(dtype)
    topk = 4 * topk_blocks + 3
    idx = np.full((L, topk), -1, dtype=np.int32)
    cur = None
    for i in range(L):
        p = K - L + i
        nb = (p + 1) // 4
        if nb <= topk_blocks:
            blocks = np.arange(nb)
        else:
            hi = nb - topk_blocks // 8
            n_rand = topk_blocks - topk_blocks // 8 - 4
            if cur is None:
                cur = rng.choice(np.arange(4, hi), n_rand, replace=False)
            cur = cur[(cur < hi) & (rng.random(cur.size) >= 0.05)]
            if cur.size < n_rand:
                pool = np.setdiff1d(np.arange(4, hi), cur)
                cur = np.concatenate(
                    [cur, rng.choice(pool, n_rand - cur.size, replace=False)]
                )
            blocks = np.sort(np.concatenate([np.arange(4), cur, np.arange(hi, nb)]))
        rows = (blocks[:, None] * 4 + np.arange(4)[None]).reshape(-1)
        idx[i, : rows.size] = rows
        tc = (p + 1) % 4
        for j in range(3):
            idx[i, topk - 3 + j] = p + 1 - tc + j if j < tc else -1
    return q, kv, mx.array(idx)[None, None]


@_needs_nax_sparse_mla
def test_nax_sparse_mla_dead_tiles_and_ragged_topk():
    """Whole unused 128-slot tiles in the middle and at the end, and top-k
    widths that are not multiples of the tile or fragment sizes."""
    scale = 256**-0.5
    for topk in (17, 100, 129, 400, 2051):
        q, kv, idx = _smla_inputs(48, 3000, topk, seed=topk)
        a = np.array(idx)
        if topk > 256:
            a[..., 128:256] = -1  # a dead tile in the middle
        a[..., ::5, max(0, topk - 70) :] = -1  # dead tail tiles for some rows
        a[0, 0, :, 0] = 3000 - 48 + np.arange(48)  # keep one usable slot per row
        idx = mx.array(a)
        out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
        ref = _smla_reference(q, kv, idx, scale)
        err = mx.abs(out.astype(mx.float32) - ref)
        assert mx.all(err <= mx.abs(ref) * 2.0**-8 + 1e-3).item(), topk


@_needs_nax_sparse_mla
def test_nax_sparse_mla_deterministic_many_threadgroups():
    """Many threadgroups in flight, realistic index rows: every run must be
    bit-identical (a software-pipelined variant with in-flight loads into
    dead registers was not)."""
    q, kv, idx = _smla_realistic_inputs(1024, 8192, seed=11)
    scale = 256**-0.5
    first = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
    mx.eval(first)
    for _ in range(8):
        out = sparse_mla_nax.sparse_mla_attention_nax(q, kv, idx, scale)
        mx.eval(out)
        assert mx.array_equal(out, first).item()


# ---------------------------------------------------------------------------
# Fused decode/verify kernels (bitwise against the reference op graphs)


def _language():
    from mlx_vlm.models.glm5_next import language

    return language


def _stats():
    return dk.STATS


def _bits(a: mx.array) -> mx.array:
    view = {2: mx.uint16, 4: mx.uint32}[a.dtype.size]
    return a.view(view)


def _mismatches(a: mx.array, b: mx.array) -> int:
    assert a.shape == b.shape and a.dtype == b.dtype
    return int(mx.sum(_bits(a) != _bits(b)).item())


def _rand_affine(lead, out_dims, in_dims, bits, group_size, scale=0.004):
    """Random packed affine weights + bf16 scales/biases (valid for any bits)."""
    words = in_dims * bits // 32
    w = mx.random.randint(0, 2**31 - 1, (*lead, out_dims, words)).astype(mx.uint32)
    w = w * 2 + mx.random.randint(0, 2, w.shape).astype(mx.uint32)
    groups = in_dims // group_size
    s = (mx.random.uniform(0.5, 1.5, (*lead, out_dims, groups)) * scale).astype(
        mx.bfloat16
    )
    b = (
        -mx.random.uniform(0.5, 1.5, (*lead, out_dims, groups)) * scale * 2**bits / 2
    ).astype(mx.bfloat16)
    return w, s, b


def _quantized_linear(out_dims, in_dims, bits, group_size=64):
    layer = nn.QuantizedLinear(64, 64, bias=False, group_size=group_size, bits=bits)
    layer.weight, layer.scales, layer.biases = _rand_affine(
        (), out_dims, in_dims, bits, group_size
    )
    return layer


def _switch_linear(experts, out_dims, in_dims, bits, group_size=64):
    from omlx.patches.deepseek_v4.switch_layers import QuantizedSwitchLinear

    layer = QuantizedSwitchLinear(64, 64, 2, False, group_size, bits)
    layer.weight, layer.scales, layer.biases = _rand_affine(
        (experts,), out_dims, in_dims, bits, group_size
    )
    return layer


# ---------------------------------------------------------------------------
# Hyper-connection collapse + branch RMSNorm
# ---------------------------------------------------------------------------


def _hyper_connection(hidden=4096, seed=0):
    from mlx_vlm.models.deepseek_v4.hyper_connection import HyperConnection

    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        rms_norm_eps=1e-5,
        hidden_size=hidden,
    )
    hc = HyperConnection(cfg)
    hc.fn = mx.random.normal(hc.fn.shape) * 0.01
    hc.base = mx.random.normal(hc.base.shape) * 0.5
    hc.scale = mx.random.uniform(0.5, 1.5, (3,))
    norm = nn.RMSNorm(hidden, eps=1e-5)
    norm.weight = mx.random.uniform(0.5, 1.5, (hidden,)).astype(mx.bfloat16)
    hc.eval()
    norm.eval()
    mx.eval(hc.parameters(), norm.parameters())
    return hc, norm


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("hidden", [4096, 1024])
@pytest.mark.parametrize("length", [1, 2, 4, 8])
def test_decode_hc_pre_is_bitwise_reference(hidden, length):
    language = _language()
    hc, norm = _hyper_connection(hidden, seed=length)
    for trial in range(3):
        x = (mx.random.normal((1, length, 4, hidden)) * (1 + 2 * trial)).astype(
            mx.bfloat16
        )
        collapsed, post, comb = hc(x)
        reference = norm(collapsed)
        fused = language._decode_hc_pre(hc, norm, x)
        assert fused is not None
        normalized, fused_post, fused_comb = fused
        assert _mismatches(normalized, reference) == 0
        assert _mismatches(fused_post, post) == 0
        assert _mismatches(fused_comb, comb) == 0
        # Each verify row equals the one-token decode of that row.
        for row in range(length):
            single = language._decode_hc_pre(hc, norm, x[:, row : row + 1])
            assert _mismatches(single[0], normalized[:, row : row + 1]) == 0
            assert _mismatches(single[2], fused_comb[:, row : row + 1]) == 0


def _check_one_token_hc_expand(hidden):
    """Bitwise check of the fused one-token expand; returns engaged calls."""
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    language = _language()
    hc, norm = _hyper_connection(hidden, seed=hidden)
    engaged = 0
    for trial in range(8):
        residual = (mx.random.normal((1, 1, 4, hidden)) * (1 + 2 * trial)).astype(
            mx.bfloat16
        )
        _, post, comb = hc(residual)
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + trial)).astype(mx.bfloat16)
        before = _stats()["hc_expand"]
        fused = language._decode_hc_expand(x, residual, post, comb)
        engaged += _stats()["hc_expand"] - before
        reference = hc_expand(x, residual, post, comb)
        assert _mismatches(fused, reference) == 0
        compiled = mx.compile(lambda *a: language._decode_hc_expand(*a))(
            x, residual, post, comb
        )
        assert _mismatches(compiled, reference) == 0
    return engaged


def _run_with_tf32(snippet: str) -> str:
    """Run ``snippet`` with this module imported as ``t`` and MLX TF32 on.

    The test session disables TF32 (conftest), which moves MLX's fp32 GEMMs
    off the NAX units; the production default keeps them there.
    """
    here = Path(__file__).resolve().parent
    code = (
        "import sys; sys.path[:0] = [%r, %r]\n"
        "from omlx.patches import mlx_vlm_glm5_next_compat as compat\n"
        "compat.apply_mlx_vlm_glm5_next_compat_patch()\n"
        "import test_mlx_vlm_glm5_next_compat as t\n" % (str(here), str(here.parent))
    ) + snippet
    env = dict(os.environ, MLX_ENABLE_TF32="1")
    done = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert done.returncode == 0, done.stderr[-4000:]
    return done.stdout


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("hidden", [4096, 1024])
def test_one_token_hc_expand_declines_without_nax_tf32(hidden):
    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    assert _check_one_token_hc_expand(hidden) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_one_token_hc_expand_is_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for hidden in (4096, 1024):\n"
        "        assert t._check_one_token_hc_expand(hidden) == 8\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


def _check_hc_deferred_chain(hidden):
    """Chained one-token HC pres with each expand folded into the next
    (``_decode_hc_pre_deferred``) against the reference HyperConnection,
    RMSNorm and hc_expand; returns the number of checked half-layers."""
    from mlx_vlm.models.deepseek_v4.hyper_connection import hc_expand

    language = _language()
    checked = 0
    for seed in range(6):
        layers = [_hyper_connection(hidden, seed=100 * seed + i) for i in range(4)]
        mx.random.seed(seed)
        h_ref = (mx.random.normal((1, 1, 4, hidden)) * (1 + seed)).astype(mx.bfloat16)
        x = h_ref
        for step, (hc, norm) in enumerate(layers):
            if seed % 2:
                hc.base = hc.base * 10  # sharper sinkhorn / sigmoid inputs
            assert dk.hc_defer_supported(hc, norm, mx.bfloat16, hidden)
            collapsed, post, comb = hc(h_ref)
            reference = norm(collapsed)
            before = _stats()["hc_pre_fused"]
            xn, h, f_post, f_comb, mm = language._decode_hc_pre_deferred(hc, norm, x)
            assert _stats()["hc_pre_fused"] == before + 1
            assert _mismatches(h, h_ref) == 0, (seed, step)
            assert _mismatches(xn, reference) == 0, (seed, step)
            assert _mismatches(f_post, post) == 0, (seed, step)
            assert _mismatches(f_comb, comb) == 0, (seed, step)
            y = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            h_ref = hc_expand(y, h_ref, post, comb)
            x = language._HCDeferred(y, h, f_post, f_comb, mm)
            checked += 1
        assert _mismatches(x.materialize(), h_ref) == 0
    return checked


@pytest.mark.usefixtures("glm5_fused_decode")
def test_hc_deferred_chain_declines_without_nax_tf32():
    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    hc, norm = _hyper_connection(1024)
    assert not dk.hc_defer_supported(hc, norm, mx.bfloat16, 1024)


@pytest.mark.usefixtures("glm5_fused_decode")
def test_hc_deferred_chain_is_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for hidden in (4096, 1024, 2048):\n"
        "        assert t._check_hc_deferred_chain(hidden) == 24\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


@pytest.mark.usefixtures("glm5_fused_decode")
def test_hc_defer_declines_uncovered_connections():
    hc, norm = _hyper_connection(1024)
    assert not dk.hc_defer_supported(hc, norm, mx.float32, 1024)
    hc3, norm3 = _hyper_connection(768)
    assert not dk.hc_defer_supported(hc3, norm3, mx.bfloat16, 768)


@pytest.mark.usefixtures("glm5_fused_decode")
def test_decode_hc_pre_declines_uncovered_inputs():
    language = _language()
    hc, norm = _hyper_connection(1024)
    assert (
        language._decode_hc_pre(hc, norm, mx.zeros((2, 1, 4, 1024), mx.bfloat16))
        is None
    )
    assert (
        language._decode_hc_pre(hc, norm, mx.zeros((1, 9, 4, 1024), mx.bfloat16))
        is None
    )
    hc.train()
    assert (
        language._decode_hc_pre(hc, norm, mx.zeros((1, 1, 4, 1024), mx.bfloat16))
        is None
    )


# ---------------------------------------------------------------------------
# MoE experts
# ---------------------------------------------------------------------------


def _moe(experts=16, hidden=1024, inter=512, top_k=8, shared_bits=8, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        n_routed_experts=experts,
        swiglu_limit=10.0,
        num_experts_per_tok=top_k,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        n_shared_experts=1 if shared_bits else None,
        intermediate_size=4 * hidden,
    )
    moe = language.Glm5NextMoE(cfg)
    sw = moe.switch_mlp
    sw.gate_proj = _switch_linear(experts, inter, hidden, 4)
    sw.up_proj = _switch_linear(experts, inter, hidden, 4)
    sw.down_proj = _switch_linear(experts, hidden, inter, 4)
    if shared_bits:
        sh = moe.shared_experts
        sh.gate_proj = _quantized_linear(inter, hidden, shared_bits)
        sh.up_proj = _quantized_linear(inter, hidden, shared_bits)
        sh.down_proj = _quantized_linear(hidden, inter, shared_bits)
    moe.gate.weight = mx.random.normal((experts, hidden)) * 0.02
    moe.gate.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    moe.eval()
    mx.eval(moe.parameters())
    return moe


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("shared_bits", [8, 4, 0])
@pytest.mark.parametrize("length", [1, 2, 4, 7])
def test_decode_experts_are_bitwise_reference(length, shared_bits, monkeypatch):
    language = _language()
    moe = _moe(shared_bits=shared_bits, seed=length)
    for trial in range(2):
        x = (mx.random.normal((1, length, 1024)) * (0.5 + trial)).astype(mx.bfloat16)
        indices, scores = moe.gate(x)
        wide_before = _stats()["moe_shared_wide"]
        down_before = _stats()["moe_down_shared_wide"]
        fused = moe._decode_experts(x, indices, scores)
        assert fused is not None
        wide_used = _stats()["moe_shared_wide"] - wide_before
        assert wide_used == (1 if shared_bits and length > 1 else 0)
        assert _stats()["moe_down_shared_wide"] - down_before == wide_used
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = moe(x)
        compiled = mx.compile(moe)(x) if length == 1 else reference
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, reference) == 0
        assert _mismatches(fused, compiled) == 0
        assert _mismatches(moe(x), reference) == 0


class _OffloadedExperts(nn.Module):
    """Stands in for expert offload's SwitchGLU wrapper: routed experts
    behind the module, no gate/up/down projections on it."""

    def __init__(self, glu):
        super().__init__()
        self.glu = glu

    def __call__(self, x, indices, **kwargs):
        return self.glu(x, indices, **kwargs)


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("length", [1, 3])
def test_decode_moe_declines_offloaded_experts(length, monkeypatch):
    language = _language()
    moe = _moe(seed=3)
    x = (mx.random.normal((1, length, 1024)) * 0.5).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = moe(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", True)
    moe.switch_mlp = _OffloadedExperts(moe.switch_mlp)
    assert moe._decode_select(x) is None
    indices, scores = moe.gate(x)
    assert moe._decode_experts(x, indices, scores) is None
    assert _mismatches(moe(x), reference) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_multi_linear_declines_armed_verify_routes():
    """Armed MTP verify routes replace the reference multi-row qmm, which the
    fused projections replay, so multi-row blocks keep the reference call."""
    assert qwen35_verify_qmm.apply_verify_qmm_patch()
    language = _language()
    x = (mx.random.normal((1, 3, 1024)) * 0.7).astype(mx.bfloat16)
    layers = [_quantized_linear(256, 1024, 8), _quantized_linear(128, 1024, 8)]
    assert language._multi_linear(x, layers) is not None
    for row_exact in (False, True):
        qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=row_exact)
        try:
            assert language._multi_linear(x, layers) is None
            assert language._multi_linear(x[:, :1], layers) is not None
        finally:
            qwen35_verify_qmm.set_verify_qmm_armed(False)


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_one_token_moe_selects_routes_inside_gate_up(seed, monkeypatch):
    """One token: the gate/up kernel replays the router's top-k selection
    (router logits -> gate/up -> down), bitwise like the reference MoE,
    including exact score ties."""
    language = _language()
    moe = _moe(experts=288, hidden=4096, inter=2048, shared_bits=8, seed=seed)
    if seed == 2:
        weight = moe.gate.weight
        bias = moe.gate.e_score_correction_bias
        for e in (7, 70, 140, 280):  # exact duplicates of expert 200
            weight[e] = weight[200]
            bias[e] = bias[200]
        moe.gate.weight, moe.gate.e_score_correction_bias = weight, bias
    for trial in range(4):
        x = (mx.random.normal((1, 1, 4096)) * (0.3 + trial)).astype(mx.bfloat16)
        before = _stats()["router_select_fused"]
        fused = moe(x)
        assert _stats()["router_select_fused"] == before + 1
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = moe(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, reference) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("bits", [8, 4])
def test_one_token_dense_mlp_gate_up_is_bitwise_reference(bits, monkeypatch):
    """GLM-5.3's dense MLP layers: gate/up + clamped SwiGLU in one dispatch
    for one token, bitwise like the eager and the compiled reference."""
    language = _language()
    cfg = SimpleNamespace(hidden_size=1024, intermediate_size=2048, swiglu_limit=10.0)
    mlp = language.Glm5NextMLP(cfg)
    mlp.gate_proj = _quantized_linear(2048, 1024, bits)
    mlp.up_proj = _quantized_linear(2048, 1024, bits)
    mlp.down_proj = _quantized_linear(1024, 2048, bits)
    mlp.eval()
    mx.eval(mlp.parameters())
    for trial in range(4):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + 3 * trial)).astype(mx.bfloat16)
        before = _stats()["mlp_gate_up"]
        fused = mlp(x)
        assert _stats()["mlp_gate_up"] == before + 1
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = mlp(x)
        compiled = mx.compile(mlp)(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused, reference) == 0
        assert _mismatches(fused, compiled) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("bits", [8, 5, 4, 6])
@pytest.mark.parametrize("n_k", [(512, 256), (256, 512), (128, 1024)])
def test_one_token_mla_head_qmv_is_bitwise_reference(bits, n_k):
    """embed_q / unembed_out (QuantizedMultiLinear, 64 heads) for one token."""
    from mlx_lm.models.mla import MultiLinear

    N, K = n_k
    for gs in (64, 32):
        mx.random.seed(bits * 31 + N + gs)
        layer = MultiLinear(K, N, 64)
        layer.weight = (mx.random.normal(layer.weight.shape) * 0.05).astype(mx.bfloat16)
        layer = layer.to_quantized(gs, bits)
        for trial in range(3):
            x = (mx.random.normal((1, 64, 1, K)) * (1 + 2 * trial)).astype(mx.bfloat16)
            reference = layer(x)
            for nsg in (2, 8):
                fused = dk.mla_head_qmv(x, layer, nsg=nsg)
                assert fused is not None
                assert _mismatches(fused, reference) == 0, (gs, trial, nsg)


@pytest.mark.usefixtures("glm5_fused_decode")
def test_mla_head_qmv_declines_qmv_quad_shapes():
    from mlx_lm.models.mla import MultiLinear

    layer = MultiLinear(128, 512, 8).to_quantized(64, 8)
    assert dk.mla_head_qmv(mx.zeros((1, 8, 1, 128), mx.bfloat16), layer) is None
    layer = MultiLinear(256, 512, 8).to_quantized(64, 8)
    assert dk.mla_head_qmv(mx.zeros((1, 8, 2, 256), mx.bfloat16), layer) is None
    assert (
        dk.mla_head_qmv(mx.zeros((1, 8, 1, 256), mx.bfloat16), MultiLinear(256, 512, 8))
        is None
    )


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("kv_len", [1, 300, 4099])
@pytest.mark.parametrize("width", [1, 7, 2051])
def test_dsa_gather_selected_is_bitwise_reference(kv_len, width):
    """The one-token sparse attention's clipped take_along_axis and mask."""
    mx.random.seed(kv_len + width)
    cache = (mx.random.normal((1, 1, kv_len + 64, 512)) * 3).astype(mx.bfloat16)
    kv = cache[:, :, :kv_len, :]
    idx = mx.random.randint(-3, kv_len + 3, (1, 1, 1, width)).astype(mx.int32)
    idx = mx.where(idx >= kv_len, -1, idx)
    clamped = mx.clip(idx[:, :, 0, :], 0, kv_len - 1)[..., None]
    reference = mx.take_along_axis(
        kv, mx.broadcast_to(clamped, clamped.shape[:-1] + (512,)), axis=2
    )
    reference_mask = (idx >= 0)[:, :, 0, :][:, :, None, :]
    out, valid = dk.dsa_gather_selected(kv, idx[:, :, 0, :])
    assert _mismatches(out, reference) == 0
    assert valid.dtype == mx.bool_ and valid.shape == reference_mask.shape
    assert mx.array_equal(valid, reference_mask).item()


@pytest.mark.usefixtures("glm5_fused_decode")
def test_decode_experts_leave_sorted_route_counts_to_switch_glu():
    moe = _moe()
    x = mx.random.normal((1, 8, 1024)).astype(mx.bfloat16)  # 64 routes -> sorted
    indices, scores = moe.gate(x)
    assert moe._decode_experts(x, indices, scores) is None


@pytest.mark.usefixtures("glm5_fused_decode")
def test_decode_experts_compile_inside_ffn_graph(monkeypatch):
    language = _language()
    moe = _moe(seed=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    fused = mx.compile(moe)(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = moe(x)
    assert _mismatches(fused, reference) == 0


# ---------------------------------------------------------------------------
# DSA indexer decode scores and selection
# ---------------------------------------------------------------------------


def _indexer(hidden=256, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        hidden_size=hidden,
        index_n_heads=32,
        index_head_dim=128,
        index_topk=2048,
        index_kpool=4,
        index_kpool_always_select_tail=True,
        q_lora_rank=128,
    )
    indexer = language.Glm5NextIndexer(cfg)
    indexer.index_kpool_compress_ape = mx.random.normal((4, 128)) * 0.1
    indexer.index_kpool_compress_gate = mx.random.normal((128, hidden)) * 0.05
    indexer.set_dtype(mx.bfloat16)
    indexer.eval()
    mx.eval(indexer.parameters())
    return indexer


def _native_indexer_available():
    from omlx.custom_kernels.glm_moe_dsa import fast

    return fast.has_symbol("dsa_indexer_scores") and fast.has_symbol("dsa_topk_indices")


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("length", [1, 2, 4, 5, 8])
def test_dsa_decode_scores_match_native_steel_tile(length):
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    indexer = _indexer(seed=length)
    for pool in (513, 1024, 1090):
        q = mx.random.normal((1, length, 32, 128)).astype(mx.bfloat16)
        keys = (mx.random.normal((1, pool, 128)) * 0.7).astype(mx.bfloat16)
        weights = (mx.random.normal((1, length, 32)) * 0.1).astype(mx.bfloat16)
        pool_len = pool - 2
        first = pool_len * 4 - length + 1
        reference = indexer._native_scores(q, keys, weights)
        idx = mx.arange(pool)
        query_pos = first + mx.arange(length)
        valid = (idx[None, None] < pool_len) & (
            ((idx + 1) * 4 - 1)[None, None] <= query_pos[None, :, None]
        )
        reference = mx.where(valid, reference, -1e30)
        scores = dk.dsa_decode_scores(q, keys, weights, first, pool_len, 4)
        assert _mismatches(scores, reference) == 0


def _make_pool_caches():
    from omlx.patches.deepseek_v4 import apply_pooling_cache_support

    apply_pooling_cache_support()
    from mlx_lm.models.cache import KVCache, PoolingCache

    return PoolingCache(4), KVCache()


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("pool", [512, 1025, 1026, 1500, 2048])
def test_dsa_topk_rows_matches_native_topk(pool):
    """Decode/verify indexer top-k (bitonic sort) against the native
    radix-select kernel: same indices in the same order, with exact score
    ties, -1e30 masked blocks and a NaN."""
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    from omlx.custom_kernels.glm_moe_dsa import fast

    for rows in (1, 4, 8):
        for trial in range(4):
            mx.random.seed(pool + 10 * rows + trial)
            s = mx.random.normal((1, rows, pool))
            if trial == 1:
                s = mx.round(s * 4) / 4
            if trial == 2:
                s = mx.where(mx.random.uniform(shape=s.shape) < 0.3, -1e30, s)
            s = s.astype(mx.bfloat16)
            if trial == 3:
                s = s.at[0, 0, 7].add(float("nan"))
            expected = fast.dsa_topk_indices(s[:, None], 512)[:, 0]
            got = dk.dsa_topk_rows(s, 512)
            assert got is not None and got.dtype == expected.dtype
            assert mx.array_equal(got, expected).item(), (rows, trial)


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("native_topk", [True, False])
def test_indexer_fast_selection_matches_general_path(monkeypatch, native_topk):
    """Where the general path scores decode rows with the NAX indexer, the
    fast selection takes the same scores."""
    if native_topk and not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    if not native_topk:
        from omlx.custom_kernels.glm_moe_dsa import fast

        has_symbol = fast.has_symbol
        monkeypatch.setattr(
            fast,
            "has_symbol",
            lambda name: name != "dsa_topk_indices" and has_symbol(name),
        )

    language = _language()
    nax_calls = []
    real_nax = language.indexer_scores_nax
    monkeypatch.setattr(
        language,
        "indexer_scores_nax",
        lambda *args: nax_calls.append(args[0].shape[0]) or real_nax(*args),
    )
    indexer = _indexer(seed=11)
    hidden = 256
    mx.random.seed(12)
    prompt = (mx.random.normal((1, 2105, hidden)) * 0.5).astype(mx.bfloat16)
    qr_prompt = (mx.random.normal((1, 2105, 128)) * 0.5).astype(mx.bfloat16)
    pools = []
    for _ in range(2):
        pool, kv = _make_pool_caches()
        for start in range(0, 2105, 512):
            indexer.fast_decode = False
            out = indexer(
                prompt[:, start : start + 512],
                qr_prompt[:, start : start + 512],
                None,
                cache=pool,
                kv_cache=kv,
            )
            if out is not None:
                mx.eval(out)
        pools.append((pool, kv))
    (fast_pool, fast_kv), (ref_pool, ref_kv) = pools
    for step, width in enumerate([1, 1, 3, 1, 4, 8, 1, 2]):
        x = (mx.random.normal((1, width, hidden)) * 0.5).astype(mx.bfloat16)
        qr = (mx.random.normal((1, width, 128)) * 0.5).astype(mx.bfloat16)
        indexer.fast_decode = True
        calls = len(nax_calls)
        fast = indexer(x, qr, None, cache=fast_pool, kv_cache=fast_kv)
        assert len(nax_calls) - calls == int(indexer_nax.nax_indexer_available())
        indexer.fast_decode = False
        reference = indexer(x, qr, None, cache=ref_pool, kv_cache=ref_kv)
        assert fast is not None and reference is not None
        assert fast.shape == reference.shape
        assert fast.dtype == reference.dtype
        assert int(mx.sum(fast != reference).item()) == 0, f"step {step} width {width}"


# ---------------------------------------------------------------------------
# End to end: a small quantized GLM-5.3 whose shapes engage every fused path
# ---------------------------------------------------------------------------


def _fused_shape_model(seed, heads=16, quantize_mla=False):
    from mlx_lm.models.mla import MultiLinear
    from mlx_vlm.models import glm5_next

    from omlx.patches.deepseek_v4.switch_layers import SwitchLinear

    quantized = (nn.Linear, SwitchLinear) + ((MultiLinear,) if quantize_mla else ())

    language = _language()
    text = glm5_next.TextConfig(
        model_type="glm5_next_text",
        vocab_size=256,
        hidden_size=1024,
        intermediate_size=2048,
        moe_intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        n_shared_experts=1,
        n_routed_experts=128,
        routed_scaling_factor=2.5,
        kv_lora_rank=512,
        q_lora_rank=256,
        qk_rope_head_dim=0,
        v_head_dim=64,
        qk_nope_head_dim=64,
        num_experts_per_tok=8,
        first_k_dense_replace=1,
        max_position_embeddings=8192,
        rms_norm_eps=1e-5,
        index_topk=2048,
        index_head_dim=128,
        index_n_heads=32,
        layer_types=[
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
            "deepseek_sparse_attention",
        ],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        linear_attn_config={
            "num_heads": 8,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
        },
        index_kpool=4,
        hc_mult=4,
        hc_sinkhorn_iters=20,
    )
    mx.random.seed(seed)
    model = language.LanguageModel(text)
    nn.quantize(
        model,
        group_size=64,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, quantized),
    )
    params = []
    for name, value in nn.utils.tree_flatten(model.parameters()):
        if value.dtype == mx.uint32:
            continue
        if language.glm5_next_cast_predicate(name):
            value = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
            if name.endswith("scales"):
                value = mx.abs(value) * 0.1 + 0.002
            if "norm" in name and name.endswith("weight"):
                value = (1.0 + value).astype(mx.bfloat16)
        else:
            value = mx.random.normal(value.shape) * 0.05
            if name.endswith("hc.scale") or name.endswith("hc.base"):
                value = value * 10
        params.append((name, value))
    model.load_weights(params, strict=False)
    model.eval()
    mx.eval(model.parameters())
    return model


def _check_small_model(seed=41, prompt_len=2101, heads=16, quantize_mla=False):
    """Fused vs reference logits of a small model, bitwise; returns families used.

    Prompts beyond index_topk (2048) run the sparse DSA paths, shorter ones
    the dense latent attention.
    """
    language = _language()
    fused_model = _fused_shape_model(seed, heads, quantize_mla)
    reference_model = _fused_shape_model(seed, heads, quantize_mla)
    prompt = mx.random.randint(0, 256, (1, prompt_len)).astype(mx.int32)
    caches = []
    for model in (fused_model, reference_model):
        cache = model.make_cache()
        for start in range(0, prompt.shape[1], 512):
            logits = model(prompt[:, start : start + 512], cache=cache).logits
            mx.eval(logits)
        caches.append(cache)
    fused_cache, reference_cache = caches
    next_ids = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
    before = dict(_stats())
    saved = language._DECODE_FUSION
    try:
        for step, width in enumerate([1, 1, 4, 1, 7, 2, 1, 8, 3]):
            block = mx.concatenate(
                [next_ids, (next_ids + mx.arange(1, width)[None]) % 256], axis=1
            )[:, :width]
            language._DECODE_FUSION = True
            fused = fused_model(block, cache=fused_cache).logits
            mx.eval(fused)
            language._DECODE_FUSION = False
            reference = reference_model(block, cache=reference_cache).logits
            mx.eval(reference)
            assert mx.all(mx.isfinite(reference)).item()
            assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
            next_ids = mx.argmax(reference[:, -1:], axis=-1).astype(mx.int32)
    finally:
        language._DECODE_FUSION = saved
    return {k for k, v in _stats().items() if v > before.get(k, 0)}


_CORE_FUSED = {"hc_mix", "kda", "moe_gate_up", "moe_down"}


@pytest.mark.usefixtures("glm5_fused_decode")
def test_small_model_decode_and_verify_logits_are_bitwise_reference():
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    used = _check_small_model()
    assert _CORE_FUSED | {"dsa_gather"} <= used, used


@pytest.mark.usefixtures("glm5_fused_decode")
def test_small_model_dense_attention_is_bitwise_reference():
    used = _check_small_model(seed=43, prompt_len=300)
    assert _CORE_FUSED <= used, used


@pytest.mark.usefixtures("glm5_fused_decode")
def test_small_model_quantized_mla_is_bitwise_reference():
    """Quantized MLA projections (as in the checkpoint): unembed_out (K =
    kv_lora_rank) runs mla_head_qmv for one token; logits stay bitwise."""
    used = _check_small_model(seed=47, prompt_len=300, quantize_mla=True)
    assert "mla_head_qmv" in used, used


@pytest.mark.usefixtures("glm5_fused_decode")
def test_small_model_bitwise_reference_with_nax_tf32():
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    out = _run_with_tf32(
        "print(sorted(t._check_small_model() | t._check_small_model(43, 300)))\n"
    )
    used = set(eval(out.strip().splitlines()[-1]))
    assert _CORE_FUSED <= used, used
    for family in ("hc_expand", "router_rows", "hc_pre_fused", "hc_post_mm"):
        if family in used:
            continue
        # Only acceptable where MLX itself would not use NAX relaxed fp32.
        assert (
            not _run_with_tf32(
                "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
                "print(dk.nax_relaxed_fp32_matmul())\n"
            )
            .strip()
            .endswith("True")
        )


@pytest.mark.usefixtures("glm5_fused_decode")
def test_small_model_mtp_runtime_loop_defers_hc_bitwise():
    """The MTP runtime's replacement model loop and KDA layer call (plain
    decode) keep the fused paths, HC expands folded into the next layer
    included, bit for bit."""
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime as rt\n"
        "assert rt.apply()\n"
        "from mlx_vlm.models.glm5_next import language as g5\n"
        "assert g5.Glm5NextModel._omlx_mtp_call_patched\n"
        "assert g5.Glm5NextLinearAttention._omlx_mtp_capture_patched\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "used = t._check_small_model(43, 300) | t._check_small_model()\n"
        "print(dk.nax_relaxed_fp32_matmul(), sorted(used))\n"
    )
    last = out.strip().splitlines()[-1]
    assert "'kda'" in last and "'hc_mix'" in last, last
    if last.startswith("True"):
        assert "'hc_pre_fused'" in last and "'hc_post_mm'" in last, last


# ---------------------------------------------------------------------------
# KDA linear attention layer body
# ---------------------------------------------------------------------------


def _kda_layer(heads=8, hidden=1024, gate_bits=8, seed=0, v_bits=8):
    from mlx_vlm.models import glm5_next

    language = _language()
    mx.random.seed(seed)
    config = SimpleNamespace(
        hidden_size=hidden,
        linear_num_heads=heads,
        linear_head_dim=128,
        linear_conv_kernel_dim=4,
        linear_lower_bound=-5.0,
        rms_norm_eps=1e-5,
    )
    layer = language.Glm5NextLinearAttention(config)
    qkv = heads * 128
    for name, (out_dims, in_dims) in {
        "q_proj": (qkv, hidden),
        "k_proj": (qkv, hidden),
        "v_proj": (qkv, hidden),
        "g_a_proj": (128, hidden),
        "b_proj": (heads, hidden),
    }.items():
        bits = v_bits if name == "v_proj" else 8
        setattr(layer, name, _quantized_linear(out_dims, in_dims, bits))
    fg = layer.forget_gate
    fg.f_a_proj = _quantized_linear(128, hidden, 8)
    fg.f_b_proj = _quantized_linear(qkv, 128, gate_bits)
    layer.g_b_proj = _quantized_linear(qkv, 128, gate_bits)
    fg.A_log = mx.random.normal((heads,)) * 0.5
    fg.dt_bias = mx.random.normal((qkv,)) * 0.5
    layer.conv1d.weight = (mx.random.normal((3 * qkv, 4, 1)) * 0.5).astype(mx.bfloat16)
    layer.o_norm.weight = mx.random.uniform(0.5, 1.5, (128,)).astype(mx.bfloat16)
    layer.o_proj = nn.Identity()  # compare the o_proj input directly
    layer.eval()
    mx.eval(layer.parameters())
    del glm5_next
    return layer


def _arrays_cache():
    from mlx_vlm.models.cache import ArraysCache

    return ArraysCache(size=2)


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("gate_bits", [8, 5])
def test_kda_decode_step_is_bitwise_reference(gate_bits, monkeypatch):
    language = _language()
    layer = _kda_layer(gate_bits=gate_bits, seed=gate_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 3, 8, 2, 1]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        before = _stats()["kda"]
        fused = layer(x, cache=fused_cache)
        assert _stats()["kda"] == before + 1
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[0], reference_cache[0]) == 0
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_eager_sigmoid_probe_reproduces_mx_sigmoid():
    for dtype in (mx.bfloat16, mx.float32):
        precise = dk.eager_sigmoid_precise(dtype)
        assert precise in (True, False), dtype
        x = (mx.random.normal((4096,)) * 6).astype(dtype)
        kernel = mx.fast.metal_kernel(
            name="glm5_sigmoid_probe",
            input_names=["x"],
            output_names=["default_out", "precise_out"],
            header=dk._QMV_HEADER,
            source=dk._SIGMOID_PROBE_SOURCE,
        )
        default, exact = kernel(
            inputs=[x],
            template=[("T", dtype)],
            grid=(x.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[x.shape] * 2,
            output_dtypes=[dtype] * 2,
        )
        assert _mismatches(exact if precise else default, mx.sigmoid(x)) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("gate_bits", [8, 5])
@pytest.mark.parametrize("seed", [20, 24, 28])
def test_kda_decode_step_seed_sweep_is_bitwise_reference(seed, gate_bits, monkeypatch):
    """The reference's beta and output-gate sigmoids are eager mx.sigmoid
    kernels, whose exp differs between MLX builds (precise in the release
    wheel's precompiled kernels); several of these seeds differed in a few
    outputs (and then in the recurrent state) when the kernel always used
    the runtime-compiled exp."""
    language = _language()
    layer = _kda_layer(seed=seed, gate_bits=gate_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 4, 1, 8, 1, 1, 2, 1, 1]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        gate5 = _stats()["kda_gate5"]
        fused = layer(x, cache=fused_cache)
        # 5-bit gate rows are replayed in the kernel for one token only.
        assert _stats()["kda_gate5"] - gate5 == int(gate_bits == 5 and width == 1)
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("v_bits", [5, 4])
def test_kda_decode_step_with_mixed_projection_bits(v_bits, monkeypatch):
    """GLM-5.3 layer 40 quantizes v_proj to 5 bits and q/k/gates to 8: the
    decode path runs one projection matmul per quantization instead of the
    reference layer body."""
    language = _language()
    layer = _kda_layer(seed=20 + v_bits, v_bits=v_bits)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    prompt = (mx.random.normal((1, 12, 1024)) * 0.8).astype(mx.bfloat16)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    for cache in (fused_cache, reference_cache):
        mx.eval(layer(prompt, cache=cache))
    for step, width in enumerate([1, 1, 4, 1, 8]):
        x = (mx.random.normal((1, width, 1024)) * (0.5 + step % 3)).astype(mx.bfloat16)
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        reference = layer(x, cache=reference_cache)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        before = _stats()["kda"]
        fused = layer(x, cache=fused_cache)
        assert _stats()["kda"] == before + 1
        mx.eval(reference, fused, fused_cache.cache, reference_cache.cache)
        assert _mismatches(fused, reference) == 0, f"step {step} width {width}"
        assert _mismatches(fused_cache[0], reference_cache[0]) == 0
        assert _mismatches(fused_cache[1], reference_cache[1]) == 0
    assert not layer._fused_ready and layer._decode_groups


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("width", [1, 4])
def test_kda_decode_step_from_empty_cache(width, monkeypatch):
    language = _language()
    layer = _kda_layer(seed=30 + width)
    x = (mx.random.normal((1, width, 1024)) * 0.7).astype(mx.bfloat16)
    fused_cache, reference_cache = _arrays_cache(), _arrays_cache()
    before = _stats()["kda"]
    fused = layer(x, cache=fused_cache)
    assert _stats()["kda"] == before + 1
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = layer(x, cache=reference_cache)
    mx.eval(fused, reference)
    assert _mismatches(fused, reference) == 0
    assert _mismatches(fused_cache[0], reference_cache[0]) == 0
    assert _mismatches(fused_cache[1], reference_cache[1]) == 0


# ---------------------------------------------------------------------------
# MoE router (one token)
# ---------------------------------------------------------------------------


def _router(experts=288, hidden=4096, seed=0):
    language = _language()
    mx.random.seed(seed)
    cfg = SimpleNamespace(
        num_experts_per_tok=8,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=2.5,
        n_routed_experts=experts,
        hidden_size=hidden,
    )
    gate = language.Glm5NextMoEGate(cfg)
    gate.weight = mx.random.normal((experts, hidden)) * 0.02
    gate.e_score_correction_bias = mx.random.normal((experts,)) * 0.01
    return gate


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize(
    "experts,hidden,bias", [(288, 4096, 0.0), (288, 4096, 14.0), (64, 512, 0.0)]
)
def test_router_is_bitwise_reference(experts, hidden, bias, monkeypatch):
    """One-token router vs group_expert_select, bitwise over many draws.

    The reference takes the sigmoid with MLX's eager kernel (precise exp on
    release wheels, where a runtime-compiled exp differs in the last bit for
    a few percent of logits and so, after normalization, in ~5% of routes);
    ``bias`` 14 is the checkpoint's e_score_correction_bias level.
    """
    language = _language()
    gate = _router(experts, hidden, seed=experts)
    gate.e_score_correction_bias = gate.e_score_correction_bias + bias
    for trial in range(96):
        x = (mx.random.normal((1, 1, hidden)) * (0.3 + trial % 6)).astype(mx.bfloat16)
        before = _stats()["router"]
        indices, scores = gate(x)
        assert _stats()["router"] == before + 1
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        ref_indices, ref_scores = gate(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert indices.dtype == ref_indices.dtype and indices.shape == ref_indices.shape
        assert mx.array_equal(indices, ref_indices).item()
        assert _mismatches(scores, ref_scores) == 0, trial


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_breaks_exact_ties_like_argpartition(monkeypatch):
    language = _language()
    gate = _router(seed=3)
    weight = gate.weight
    bias = gate.e_score_correction_bias
    # Experts 7, 70, 140 and 280 become exact duplicates of expert 200.
    for e in (7, 70, 140, 280):
        weight[e] = weight[200]
        bias[e] = bias[200]
    bias[200] = bias[200] + 1.0  # push the tied group into the top-k
    for e in (7, 70, 140, 280):
        bias[e] = bias[200]
    gate.weight, gate.e_score_correction_bias = weight, bias
    x = mx.random.normal((1, 1, 4096)).astype(mx.bfloat16)
    indices, scores = gate(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_indices, ref_scores = gate(x)
    tied = [int(i) for i in ref_indices[0, 0].tolist() if i in (7, 70, 140, 200, 280)]
    assert tied == sorted(tied) and len(tied) == 5
    assert mx.array_equal(indices, ref_indices).item()
    assert _mismatches(scores, ref_scores) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_orders_nan_scores_like_argpartition(monkeypatch):
    language = _language()
    gate = _router(64, 512, seed=9)
    bias = gate.e_score_correction_bias
    for e in range(64):
        if e not in (5, 33, 60):
            bias[e] = float("nan")
    gate.e_score_correction_bias = bias
    x = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    before = _stats()["router"]
    indices, scores = gate(x)
    assert _stats()["router"] == before + 1
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    ref_indices, ref_scores = gate(x)
    assert sorted(ref_indices[0, 0].tolist()[:3]) == [5, 33, 60]
    assert mx.array_equal(indices, ref_indices).item()
    assert _mismatches(scores, ref_scores) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_declines_other_gemv_configurations():
    # K >= 16 * E selects MLX's split-K (bn=8) gemv, which is not replicated.
    x = mx.zeros((1, 1024), mx.bfloat16)
    assert dk.moe_router(x, mx.zeros((16, 1024)), mx.zeros((16,)), 8, 2.5, True) is None


def _check_router_rows(experts=288, hidden=4096):
    """Verify-block routers (2..8 rows), bitwise; returns engaged calls."""
    language = _language()
    gate = _router(experts, hidden, seed=7)
    engaged = 0
    for trial, rows in enumerate([2, 3, 4, 5, 8, 4]):
        x = (mx.random.normal((1, rows, hidden)) * (0.3 + trial)).astype(mx.bfloat16)
        before = _stats()["router_rows"]
        indices, scores = gate(x)
        engaged += _stats()["router_rows"] - before
        language._DECODE_FUSION = False
        try:
            ref_indices, ref_scores = gate(x)
        finally:
            language._DECODE_FUSION = True
        assert mx.array_equal(indices, ref_indices).item(), rows
        assert _mismatches(scores, ref_scores) == 0, rows
    return engaged


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_rows_declines_without_nax_tf32():
    if dk.nax_relaxed_fp32_matmul():
        pytest.skip("TF32 NAX matmuls are enabled in this session")
    assert _check_router_rows() == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_rows_bitwise_reference_with_nax_tf32():
    out = _run_with_tf32(
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if dk.nax_relaxed_fp32_matmul():\n"
        "    for args in ((), (128, 1024)):\n"
        "        n = t._check_router_rows(*args)\n"
        "        declined = [k for k, ok in dk._ROUTER_ROWS_CHECKED.items() if not ok]\n"
        "        # Every call is bitwise the reference (checked above); a call may\n"
        "        # skip the kernel only because its configuration was declined by\n"
        "        # the first-use check (a build whose TF32 GEMM rounds that shape\n"
        "        # differently, e.g. the stock wheel for some row counts).\n"
        "        assert n == 6 or (0 < len(declined) and n >= 1), (n, declined)\n"
        "    print('checked')\n"
        "else:\n"
        "    print('no-nax')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


# ---------------------------------------------------------------------------
# Latent MLA attention (64 heads, as in GLM-5.3)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Projections sharing one input (MLA q_a / kv_a / indexer wk, weights_proj)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("tokens", [1, 2, 3, 5, 8])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_multi_qmv_is_bitwise_separate_projections(tokens, bits):
    language = _language()
    mx.random.seed(tokens * 10 + bits)
    k = 1024
    layers = [_quantized_linear(n, k, bits) for n in (512, 136, 128, 32)]
    x = (mx.random.normal((1, tokens, k)) * 0.7).astype(mx.bfloat16)
    for count in (1, 2, 4):
        group = layers[:count]
        fused = dk.multi_qmv(x.reshape(tokens, k), group)
        assert fused is not None
        for layer, out in zip(group, fused):
            reference = language.linear_forward(layer, x).reshape(tokens, -1)
            assert _mismatches(out, reference) == 0, (count, layer.weight.shape)


@pytest.mark.usefixtures("glm5_fused_decode")
def test_multi_qmv_declines_uncovered_projections():
    mx.random.seed(3)
    x = (mx.random.normal((1, 1024)) * 0.7).astype(mx.bfloat16)
    eight, six = _quantized_linear(256, 1024, 8), _quantized_linear(256, 1024, 6)
    assert dk.multi_qmv(x, [eight, six]) is None  # mixed bits
    assert dk.multi_qmv(x, [eight, _quantized_linear(12, 1024, 8)]) is None
    biased = nn.QuantizedLinear(1024, 256, bias=True, bits=8)
    assert dk.multi_qmv(x, [eight, biased]) is None
    assert dk.multi_qmv(x, [eight, nn.Linear(1024, 256, bias=False)]) is None
    assert dk.multi_qmv(mx.zeros((9, 1024), mx.bfloat16), [eight]) is None
    # one token: qmv_fast shapes only (4-bit needs K % 512 == 0)
    four = _quantized_linear(256, 1280, 4)
    assert dk.multi_qmv(mx.zeros((1, 1280), mx.bfloat16), [four, four]) is None


@pytest.mark.usefixtures("glm5_fused_decode")
def test_multi_linear_groups_projections_by_quantization():
    language = _language()
    mx.random.seed(5)
    x = (mx.random.normal((1, 3, 1024)) * 0.7).astype(mx.bfloat16)
    layers = [
        _quantized_linear(512, 1024, 6),
        _quantized_linear(256, 1024, 8),
        _quantized_linear(128, 1024, 6),
        _quantized_linear(32, 1024, 8),
    ]
    before = _stats()["multi_qmv"]
    outs = language._multi_linear(x, layers)
    assert _stats()["multi_qmv"] == before + 2
    for layer, out in zip(layers, outs):
        assert _mismatches(out, language.linear_forward(layer, x)) == 0
    assert language._multi_linear(x, layers[:2]) is None


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_rows_first_use_check_rejects_wrong_kernels():
    out = _run_with_tf32(
        "import mlx.core as mx\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if not dk.nax_relaxed_fp32_matmul():\n"
        "    print('no-nax')\n"
        "else:\n"
        "    real = dk._router_select_kernel()\n"
        "    def wrong(**kw):\n"
        "        idx, sc = real(**kw)\n"
        "        return idx, sc * 1.5\n"
        "    dk._router_select_kernel = lambda: wrong\n"
        "    gate = t._router(64, 512, seed=4)\n"
        "    x = mx.random.normal((4, 512)).astype(mx.bfloat16)\n"
        "    args = (gate.weight, gate.e_score_correction_bias, 8, 2.5, True)\n"
        "    assert dk.moe_router_rows(x, *args) is None\n"
        "    assert dk.moe_router_rows(x, *args) is None\n"
        "    assert list(dk._ROUTER_ROWS_CHECKED.values()) == [False]\n"
        "    print('checked')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("tokens", [2, 5, 8])
def test_down_combine_folds_glm_shared_down_bitwise(tokens):
    """GLM-5.3 shapes: routed down 4-bit [E, 4096, 2048], shared down 8-bit."""
    language = _language()
    mx.random.seed(tokens)
    routed = _switch_linear(10, 4096, 2048, 4)
    shared = _quantized_linear(4096, 2048, 8)
    act = (mx.random.normal((tokens, 8, 2048)) * 0.3).astype(mx.bfloat16)
    shared_act = (mx.random.normal((tokens, 2048)) * 0.3).astype(mx.bfloat16)
    idx = mx.stack([mx.random.permutation(10)[:8] for _ in range(tokens)]).astype(
        mx.uint32
    )
    scores = mx.random.uniform(0.05, 0.4, (tokens, 8))
    shared_y = language.linear_forward(
        shared, shared_act.reshape(1, tokens, -1)
    ).reshape(tokens, -1)
    reference = dk.moe_down_combine(act, idx, scores, routed, shared_y=shared_y)
    fused = dk.moe_down_combine(act, idx, scores, routed, shared, shared_act=shared_act)
    assert fused is not None and reference is not None
    assert _mismatches(fused, reference) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("seed", [44, 48])
@pytest.mark.parametrize("prompt_len", [300, 2101])
def test_small_model_seed_sweep_is_bitwise_reference(seed, prompt_len):
    heads = 8
    if prompt_len > 2048 and not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")
    _check_small_model(seed, prompt_len, heads)


def _fuse_gate_up(switch_mlp):
    """The MoE gate/up fusion's layout: gate_up_proj = [gate; up] rows per
    expert (omlx.patches.moe_gate_up_fusion._fuse_one)."""
    gate, up = switch_mlp.gate_proj, switch_mlp.up_proj
    fused = {
        f: mx.concatenate([gate[f], up[f]], axis=1)
        for f in ("weight", "scales", "biases")
    }
    mx.eval(list(fused.values()))
    for field, value in fused.items():
        setattr(gate, field, value)
    switch_mlp.gate_up_proj = gate
    del switch_mlp.gate_proj
    del switch_mlp.up_proj


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("length", [1, 2, 5, 8])
def test_decode_experts_read_fused_gate_up_layout(length, monkeypatch):
    language = _language()
    moe = _moe(seed=10 + length)
    x = (mx.random.normal((1, length, 1024)) * 0.7).astype(mx.bfloat16)
    indices, scores = moe.gate(x)
    split = moe._decode_experts(x, indices, scores)
    monkeypatch.setattr(language, "_DECODE_FUSION", False)
    reference = moe(x)
    monkeypatch.setattr(language, "_DECODE_FUSION", True)
    if split is None:  # 64 routes: sorted, left to SwitchGLU
        assert length == 8
    else:
        assert _mismatches(split, reference) == 0
    _fuse_gate_up(moe.switch_mlp)
    fused = moe._decode_experts(x, indices, scores)
    assert (fused is None) == (split is None)
    if fused is not None:
        assert _mismatches(fused, split) == 0
    if hasattr(type(moe.switch_mlp), "projections"):
        # This build's SwitchGLU runs the fused layout (MoE gate/up fusion).
        monkeypatch.setattr(language, "_DECODE_FUSION", False)
        fused_reference = moe(x)
        monkeypatch.setattr(language, "_DECODE_FUSION", True)
        assert _mismatches(fused_reference, reference) == 0
        assert _mismatches(moe(x), reference) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_router_rows_first_use_check_inside_compile_uses_reference():
    out = _run_with_tf32(
        "import mlx.core as mx\n"
        "from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as dk\n"
        "if not dk.nax_relaxed_fp32_matmul():\n"
        "    print('no-nax')\n"
        "else:\n"
        "    language = t._language()\n"
        "    gate = t._router(64, 512, seed=6)\n"
        "    x = mx.random.normal((1, 4, 512)).astype(mx.bfloat16)\n"
        "    traced = mx.compile(lambda v: gate(v))(x)\n"
        "    assert dk._ROUTER_ROWS_CHECKED == {}  # no check inside compile\n"
        "    eager = gate(x)\n"
        "    assert list(dk._ROUTER_ROWS_CHECKED.values()) == [True]\n"
        "    mx.eval(traced)\n"
        "    language._DECODE_FUSION = False\n"
        "    ref_eager = gate(x)\n"
        "    assert mx.array_equal(eager[0], ref_eager[0]).item()\n"
        "    assert mx.array_equal(eager[1].view(mx.uint32), ref_eager[1].view(mx.uint32)).item()\n"
        "    print('checked')\n"
    )
    if "no-nax" in out:
        pytest.skip("this GPU runs fp32 GEMMs without NAX")
    assert "checked" in out


@pytest.mark.usefixtures("glm5_fused_decode")
def test_upstream_kda_prefill_then_fused_decode_is_bitwise_reference(monkeypatch):
    """Caches written by upstream's fused KDA prefill (glm53_kda_prework,
    >= 64-row chunks; the test prompts' 512-token chunks) feed the fused
    decode and verify paths exactly like the stock prefill's."""
    try:
        from omlx.patches import glm53_kda_prework as prework
    except ImportError:
        pytest.skip("this build has no glm53 fused KDA prefill")
    _language()
    if not getattr(prework, "_GLM53_KDA_PREFILL_ENABLED", False):
        pytest.skip("glm53 fused KDA prefill disabled")
    if not _native_indexer_available():
        pytest.skip("GLM DSA native indexer extension is not built")

    monkeypatch.setattr(prework, "_GLM53_KDA_ENGAGED_LOGGED", False)
    used = _check_small_model() | _check_small_model(43, 300)
    assert prework._GLM53_KDA_ENGAGED_LOGGED
    assert _CORE_FUSED <= used, used


@pytest.mark.usefixtures("glm5_fused_decode")
@pytest.mark.parametrize("every", [1, 3])
def test_decode_early_eval_only_schedules(every, monkeypatch):
    """One-token decode forwards evaluate every few layers while the graph is
    still being built; the logits and caches are those of the lazy forward."""
    language = _language()
    model = _fused_shape_model(seed=45)
    prompt = mx.random.randint(0, 256, (1, 300)).astype(mx.int32)
    caches = []
    for _ in range(2):
        cache = model.make_cache()
        mx.eval(model(prompt, cache=cache).logits)
        caches.append(cache)
    calls = []
    real_async_eval = mx.async_eval

    def counting_async_eval(*args):
        calls.append(len(args))
        return real_async_eval(*args)

    token = mx.array([[17]], dtype=mx.int32)
    for step in range(3):
        monkeypatch.setattr(language, "_DECODE_EVAL_EVERY", 0)
        lazy = model(token, cache=caches[0]).logits
        mx.eval(lazy)
        monkeypatch.setattr(language, "_DECODE_EVAL_EVERY", every)
        monkeypatch.setattr(mx, "async_eval", counting_async_eval)
        early = model(token, cache=caches[1]).logits
        monkeypatch.setattr(mx, "async_eval", real_async_eval)
        mx.eval(early)
        assert _mismatches(early, lazy) == 0, f"step {step}"
        token = mx.argmax(lazy[:, -1:], axis=-1).astype(mx.int32)
    # 4 layers: evaluations after layers `every`, 2 * every, ... (not the last).
    assert len(calls) == 3 * len(range(every, 4, every))
    for a, b in zip(caches[0], caches[1]):
        for x, y in zip(a.state, b.state):
            if isinstance(x, mx.array):
                assert _mismatches(x, y) == 0


@pytest.mark.usefixtures("glm5_fused_decode")
def test_compiled_decode_releases_the_weights_when_the_model_is_dropped():
    """One-token steps compile each layer's FFN half around multi-output fused
    kernels (router logits, route-selecting gate/up, HC pre/post). MLX 0.32.2
    leaks such intermediates of a compiled trace with everything they reference
    (ml-explore/mlx#4453): with the weights as trace constants, dropping the
    model left MoE layers' routed gate/up experts allocated (~73 MB per layer
    here, ~2.5 GB on GLM-5.3). Runs on a worker thread whose final
    ``mx.clear_streams()`` drops its compile cache, like an engine thread."""
    import gc
    import threading

    result = {}

    def work():
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        # Pausing the collector makes the groups MLX 0.32.2 leaks deterministic
        # here (one to three MoE layers' gate/up experts without the fix).
        gc.disable()
        try:
            run()
        finally:
            gc.enable()
        gc.collect()
        mx.synchronize()
        mx.clear_streams()
        gc.collect()
        mx.clear_cache()
        result["leak"] = mx.get_active_memory() - base

    def run():
        model = _fused_shape_model(7)
        cache = model.make_cache()
        prompt = mx.random.randint(0, 256, (1, 64)).astype(mx.int32)
        logits = model(prompt, cache=cache).logits
        token = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
        before = dict(_stats())
        for _ in range(3):
            logits = model(token, cache=cache).logits
            token = mx.argmax(logits[:, -1:], axis=-1).astype(mx.int32)
            mx.eval(token)
        result["router_select_fused"] = _stats()["router_select_fused"] - before.get(
            "router_select_fused", 0
        )

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    assert result["router_select_fused"] > 0  # the compiled fused MoE path ran
    assert result["leak"] < (1 << 20), f"{result['leak']} bytes still active"


# ---------------------------------------------------------------------------
# Compiled decode FFN (weights traced as inputs)


_CFFN_WORDS = 1 << 21  # 8 MB of float32 per weight: a pinned weight is unmistakable


_CFFN_KERNELS = {}


def _cffn_kernel(n_in: int, n_out: int):
    """out_j[i] = sum_k in_k[i] + j (a stand-in for the fused decode kernels)."""
    key = (n_in, n_out)
    if key not in _CFFN_KERNELS:
        total = " + ".join(f"in{k}[i]" for k in range(n_in))
        _CFFN_KERNELS[key] = mx.fast.metal_kernel(
            name=f"compile_ffn_probe_{n_in}_{n_out}",
            input_names=[f"in{k}" for k in range(n_in)],
            output_names=[f"out{j}" for j in range(n_out)],
            source="uint i = thread_position_in_grid.x;\n"
            + "".join(f"out{j}[i] = {total} + {j};\n" for j in range(n_out)),
        )
    return _CFFN_KERNELS[key]


def _cffn_run(inputs, n_out):
    return _cffn_kernel(len(inputs), n_out)(
        inputs=inputs,
        grid=(8, 1, 1),
        threadgroup=(8, 1, 1),
        output_shapes=[(8,)] * n_out,
        output_dtypes=[mx.float32] * n_out,
    )


class _CffnWeights(nn.Module):
    def __init__(self, offset: float):
        super().__init__()
        self.weight = mx.arange(_CFFN_WORDS, dtype=mx.float32) * 1e-6 + offset


class _CffnLayer(nn.Module):
    """The FFN half of a decoder layer, with the leaking topology: multi-output
    kernels fed by the weights whose outputs feed further kernels."""

    def __init__(self):
        super().__init__()
        self.ffn_hc = _CffnWeights(1.0)
        self.post_attention_layernorm = _CffnWeights(2.0)
        self.mlp = _CffnWeights(3.0)
        self.mlp.experts = [_CffnWeights(4.0)]

    def _ffn_block(self, x):
        a, b = _cffn_run([x, self.ffn_hc.weight], 2)  # like the router logits kernel
        act, routes, scores = _cffn_run(
            [a, b, self.mlp.weight, self.mlp.experts[0].weight], 3
        )  # like the route-selecting gate/up kernel
        y = _cffn_run([act, routes, scores, self.post_attention_layernorm.weight], 1)[0]
        return y * 2


def _leaked_bytes(compile_fn) -> int:
    """Build, run and drop a layer on a worker thread (like an engine thread,
    whose final reclaim clears its compile cache); bytes left behind."""
    result = {}

    def work():
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        layer = _CffnLayer()
        mx.eval(layer.parameters())
        x = mx.ones((8,), dtype=mx.float32)
        mx.eval(compile_fn(layer)(x))
        del layer, x
        gc.collect()
        mx.synchronize()
        mx.clear_streams()
        gc.collect()
        mx.clear_cache()
        result["leak"] = mx.get_active_memory() - base

    worker = threading.Thread(target=work)
    worker.start()
    worker.join()
    return result["leak"]


def test_compile_ffn_block_releases_the_layer_weights():
    language = _language()
    leak = _leaked_bytes(
        lambda layer: language.compile_ffn_block(layer, layer._ffn_block)
    )
    # A pinned weight would leave 8 MB+; only tiny trace constants may remain.
    assert leak < (64 << 10), f"{leak} bytes still active after the layer was dropped"


def test_compile_ffn_block_matches_eager_and_plain_compile():
    language = _language()
    layer = _CffnLayer()
    x = mx.arange(8, dtype=mx.float32)
    eager = layer._ffn_block(x)
    plain = mx.compile(layer._ffn_block)(x)
    fixed = language.compile_ffn_block(layer, layer._ffn_block)
    first, second = fixed(x), fixed(x + 1)
    mx.eval(eager, plain, first, second)
    assert mx.array_equal(first, eager).item()
    assert mx.array_equal(first, plain).item()
    assert mx.array_equal(second, layer._ffn_block(x + 1)).item()


def test_compile_ffn_block_keeps_module_arrays_when_the_trace_raises():
    language = _language()
    layer = _CffnLayer()
    before = [layer.ffn_hc.weight, layer.mlp.weight, layer.mlp.experts[0].weight]

    def broken(x):
        raise RuntimeError("trace failed")

    with pytest.raises(RuntimeError, match="trace failed"):
        language.compile_ffn_block(layer, broken)(mx.ones((8,)))
    after = [layer.ffn_hc.weight, layer.mlp.weight, layer.mlp.experts[0].weight]
    assert all(a is b for a, b in zip(before, after))
    mx.eval(layer._ffn_block(mx.ones((8,))))
