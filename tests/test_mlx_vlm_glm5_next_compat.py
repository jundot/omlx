# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GLM-5.3-Flash mlx-vlm compatibility overlay."""

from __future__ import annotations

import base64
import copy
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

    monkeypatch.setattr(lang, "sparse_mla_attention", spy_sma)
    monkeypatch.setattr(lang, "exact_block_token_attention", spy_eba)
    monkeypatch.setattr(lang, "q8_vup_flat", lambda *a, **k: None)

    x = mx.random.normal((1, 4096, 4096), dtype=mx.float32)
    out = attn(x, mask=None, cache=None)
    mx.eval(out)
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
    real_eval(out)
    assert len(calls) >= text.num_hidden_layers, (
        f"prefill width must eval the stream per layer, got {len(calls)} eval calls"
        f" for {text.num_hidden_layers} layers"
    )
    # Layer-specific buffer sizes can accumulate in the allocator pool.
    assert len(clears) >= text.num_hidden_layers, (
        f"prefill must clear the allocator pool per layer, got {len(clears)}"
        f" clears for {text.num_hidden_layers} layers"
    )

    calls.clear()
    clears.clear()
    decode = mx.zeros((1, 1), dtype=mx.int32)
    out = model(decode)
    real_eval(out)
    assert len(calls) < text.num_hidden_layers, (
        "decode width must stay lazy (no per-layer eval)"
    )
    assert not clears, "decode width must not clear the pool per layer"


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
