# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GLM-5.3-Flash mlx-vlm compatibility overlay."""

from __future__ import annotations

import base64
import importlib
import io
import json

import mlx.core as mx
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
    batch_cache = generate._make_cache(model, [0, 0], None)
    batch_tokens = mx.concatenate([prompt, prompt], axis=0)
    batch_logits = model(batch_tokens, cache=batch_cache).logits
    left_logits = model(prompt, cache=model.make_cache()).logits
    right_logits = model(prompt, cache=model.make_cache()).logits
    mx.eval(batch_logits, left_logits, right_logits)

    assert type(batch_cache[1][1]).__name__ == "BatchPoolingCache"
    assert mx.allclose(batch_logits[:1], left_logits, atol=3e-4).item()
    assert mx.allclose(batch_logits[1:], right_logits, atol=3e-4).item()


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
