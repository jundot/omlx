# SPDX-License-Identifier: Apache-2.0
"""Tests for the Ling 3.0 Flash ``bailing_hybrid`` mlx-lm patch."""

import importlib
import json
import sys
from types import SimpleNamespace

import mlx.core as mx
import pytest


def _minimal_config(**overrides):
    config = {
        "model_type": "bailing_hybrid",
        "architectures": ["BailingHybridForCausalLM"],
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "num_shared_experts": 0,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "layer_group_size": 2,
        "group_norm_size": 1,
        "vocab_size": 128,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 256,
        "routed_scaling_factor": 1.0,
        "head_dim": 8,
        "kv_lora_rank": 8,
        "qk_rope_head_dim": 4,
        "qk_nope_head_dim": 4,
        "v_head_dim": 4,
        "short_conv_kernel_size": 3,
    }
    config.update(overrides)
    return config


def _load_patch_module():
    from omlx.patches.bailing_hybrid import apply_bailing_hybrid_patch

    apply_bailing_hybrid_patch()
    return importlib.import_module("mlx_lm.models.bailing_hybrid")


def test_apply_registers_bailing_hybrid_module():
    module = _load_patch_module()

    assert module.__package__ == "mlx_lm.models"
    assert sys.modules["mlx_lm.models.bailing_hybrid"] is module

    import mlx_lm.models as models_pkg

    assert models_pkg.bailing_hybrid is module


def test_apply_is_idempotent():
    from omlx.patches.bailing_hybrid import (
        apply_bailing_hybrid_patch,
        is_applied,
    )

    first = apply_bailing_hybrid_patch()
    second = apply_bailing_hybrid_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_apply_prefers_upstream_module(monkeypatch):
    from omlx.patches import bailing_hybrid

    upstream = SimpleNamespace(_omlx_swiglu_clamp_native=True)
    models_pkg = SimpleNamespace()

    def fake_import(name):
        if name == "mlx_lm.models.bailing_hybrid":
            return upstream
        if name == "mlx_lm.models":
            return models_pkg
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(bailing_hybrid, "_APPLIED", False)
    monkeypatch.setattr(bailing_hybrid.importlib, "import_module", fake_import)
    monkeypatch.setattr(
        bailing_hybrid,
        "_register_module",
        lambda: (_ for _ in ()).throw(AssertionError("vendored module used")),
    )

    assert bailing_hybrid.apply_bailing_hybrid_patch() is False
    assert models_pkg.bailing_hybrid is upstream


def test_apply_propagates_clamp_install_failure(monkeypatch):
    from omlx.patches import bailing_hybrid

    upstream = SimpleNamespace()
    models_pkg = SimpleNamespace()

    def fake_import(name):
        if name == "mlx_lm.models.bailing_hybrid":
            return upstream
        if name == "mlx_lm.models":
            return models_pkg
        raise AssertionError(f"unexpected import: {name}")

    def fail_install(_module):
        raise RuntimeError("clamp install failed")

    monkeypatch.setattr(bailing_hybrid, "_APPLIED", False)
    monkeypatch.setattr(bailing_hybrid.importlib, "import_module", fake_import)
    monkeypatch.setattr(bailing_hybrid, "ensure_swiglu_clamp", fail_install)

    with pytest.raises(RuntimeError, match="clamp install failed"):
        bailing_hybrid.apply_bailing_hybrid_patch()

    assert bailing_hybrid.is_applied() is False


def test_get_classes_resolves_bailing_hybrid():
    _load_patch_module()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_config())

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_mixed_global_and_linear_attention_cache_forward():
    bailing_hybrid = _load_patch_module()
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.models.cache import ArraysCache, KVCache

    model = bailing_hybrid.Model(
        bailing_hybrid.ModelArgs.from_dict(_minimal_config())
    )
    cache = model.make_cache()

    assert type(cache[0]) is ArraysCache
    assert type(cache[1]) is KVCache

    prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(prefill, decode)

    assert prefill.shape == (1, 3, 128)
    assert decode.shape == (1, 1, 128)
    assert cache[0][0] is not None
    assert cache[1].offset == 4

    generator = BatchGenerator(
        model,
        max_tokens=2,
        prefill_batch_size=2,
        completion_batch_size=2,
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert([[1, 2, 3], [4, 5]], max_tokens=[2, 2])
    finished = []
    for _ in range(8):
        _, responses = generator.next()
        finished.extend(r for r in responses if r.finish_reason is not None)
        if len(finished) == 2:
            break

    assert uids == [0, 1]
    assert {response.uid for response in finished} == {0, 1}
    assert all(response.finish_reason == "length" for response in finished)


def _batch_greedy_tokens(model, prompts, max_tokens=6):
    from mlx_lm.generate import BatchGenerator

    generator = BatchGenerator(
        model,
        max_tokens=max_tokens,
        prefill_batch_size=len(prompts),
        completion_batch_size=len(prompts),
        sampler=lambda logits: mx.argmax(logits, axis=-1),
    )
    uids = generator.insert(prompts, max_tokens=[max_tokens] * len(prompts))
    tokens = {uid: [] for uid in uids}
    for _ in range(max_tokens + 4):
        _, responses = generator.next()
        for response in responses:
            tokens[response.uid].append(response.token)
        if all(len(output) == max_tokens for output in tokens.values()):
            break
    return [tokens[uid] for uid in uids]


def test_variable_length_batch_matches_single_request_greedy_tokens():
    bailing_hybrid = _load_patch_module()
    mx.random.seed(7)
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(_minimal_config()))

    short_prompt = [4, 5]
    long_prompt = [7, 8, 9, 10, 11, 12]
    single = _batch_greedy_tokens(model, [short_prompt])[0]
    batched = _batch_greedy_tokens(model, [short_prompt, long_prompt])[0]

    assert batched == single


def test_depthwise_conv_matches_token_loop_reference():
    bailing_hybrid = _load_patch_module()
    conv = bailing_hybrid.DepthwiseConv1d(channels=4, kernel_size=3)
    conv.weight = mx.arange(12, dtype=mx.float32).reshape(4, 1, 3) / 12
    x = mx.arange(32, dtype=mx.float32).reshape(2, 4, 4) / 32
    initial_cache = mx.arange(24, dtype=mx.float32).reshape(2, 4, 3) / 24

    expected_cache = initial_cache
    expected_outputs = []
    weight = conv.weight[:, 0, :]
    for token_idx in range(x.shape[1]):
        current = x[:, token_idx : token_idx + 1, :].transpose(0, 2, 1)
        expected_cache = mx.concatenate(
            [expected_cache[:, :, 1:], current],
            axis=2,
        )
        value = (expected_cache * weight[None, :, :]).sum(axis=2)
        expected_outputs.append(mx.sigmoid(value) * value)
    expected = mx.stack(expected_outputs, axis=1)

    actual, actual_cache = conv(x, initial_cache)
    mx.eval(expected, expected_cache, actual, actual_cache)

    assert mx.allclose(actual, expected, rtol=1e-5, atol=1e-6)
    assert mx.allclose(actual_cache, expected_cache)


def test_depthwise_conv_uses_lengths_for_right_padded_cache_state():
    bailing_hybrid = _load_patch_module()
    conv = bailing_hybrid.DepthwiseConv1d(channels=4, kernel_size=3)
    conv.weight = mx.arange(12, dtype=mx.float32).reshape(4, 1, 3) / 12
    x = mx.arange(32, dtype=mx.float32).reshape(2, 4, 4) / 32
    initial_cache = mx.arange(24, dtype=mx.float32).reshape(2, 4, 3) / 24
    mask = mx.array(
        [[True, True, False, False], [True, True, True, True]],
        dtype=mx.bool_,
    )

    batch_output, batch_cache = conv(
        x,
        initial_cache,
        mask=mask,
        lengths=mx.array([2, 4]),
    )
    single_output, single_cache = conv(x[:1, :2], initial_cache[:1])
    mx.eval(batch_output, batch_cache, single_output, single_cache)

    assert mx.allclose(batch_output[0, :2], single_output[0])
    assert mx.allclose(batch_cache[0], single_cache[0])


@pytest.mark.parametrize("safe_gate", [False, True])
def test_fused_kda_matches_reference(safe_gate):
    bailing_hybrid = _load_patch_module()
    batch, length, heads, head_dim = 1, 5, 2, 8
    q = mx.arange(batch * length * heads * head_dim, dtype=mx.float32).reshape(
        batch, length, heads, head_dim
    )
    q = q / 100
    k = q + 0.1
    v = q + 0.2
    g = q + 0.3
    beta = mx.arange(batch * length * heads, dtype=mx.float32).reshape(
        batch, length, heads
    )
    beta = beta / 10
    a_log = mx.array([-0.2, 0.3], dtype=mx.float32)
    dt_bias = mx.arange(heads * head_dim, dtype=mx.float32) / 50
    initial_state = mx.arange(
        batch * heads * head_dim * head_dim,
        dtype=mx.float32,
    ).reshape(batch, heads, head_dim, head_dim)
    initial_state = initial_state / 1000

    reference_state = initial_state
    reference_outputs = []
    for token_idx in range(length):
        q_t = q[:, token_idx]
        k_t = k[:, token_idx]
        v_t = v[:, token_idx]
        q_t = q_t / mx.sqrt(mx.sum(q_t * q_t, axis=-1, keepdims=True) + 1e-6)
        k_t = k_t / mx.sqrt(mx.sum(k_t * k_t, axis=-1, keepdims=True) + 1e-6)
        gate_input = g[:, token_idx] + dt_bias.reshape(heads, head_dim)
        if safe_gate:
            log_decay = -5.0 * mx.sigmoid(
                mx.exp(a_log)[None, :, None] * gate_input
            )
        else:
            log_decay = -mx.exp(a_log)[None, :, None] * mx.logaddexp(
                gate_input,
                mx.array(0.0),
            )
        reference_state = reference_state * mx.exp(log_decay)[..., None]
        delta = v_t - mx.sum(reference_state * k_t[..., None], axis=2)
        delta = delta * mx.sigmoid(beta[:, token_idx])[..., None]
        reference_state = reference_state + k_t[..., None] * delta[..., None, :]
        reference_outputs.append(
            mx.sum(reference_state * q_t[..., None], axis=2) * (head_dim**-0.5)
        )
    expected = mx.stack(reference_outputs, axis=1)

    actual, actual_state = bailing_hybrid.recurrent_kda(
        q,
        k,
        v,
        g,
        beta,
        a_log,
        dt_bias,
        initial_state,
        safe_gate=safe_gate,
        lower_bound=-5.0,
    )
    mx.eval(expected, reference_state, actual, actual_state)

    assert mx.allclose(actual, expected, rtol=2e-4, atol=2e-5)
    assert mx.allclose(actual_state, reference_state, rtol=2e-4, atol=2e-5)


def test_external_prefill_upgrades_legacy_one_slot_cache():
    bailing_hybrid = _load_patch_module()
    from mlx_lm.models.cache import ArraysCache

    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler

    model = bailing_hybrid.Model(
        bailing_hybrid.ModelArgs.from_dict(_minimal_config())
    )
    source_cache = model.make_cache()
    prefix_logits = model(
        mx.array([[1, 2]], dtype=mx.int32),
        cache=source_cache,
    )
    mx.eval(prefix_logits)

    legacy_cache = ArraysCache(size=1)
    legacy_cache[0] = tuple(source_cache[0].state)
    cache = [legacy_cache, source_cache[1]]
    request = Request(
        request_id="ling-legacy-prefill",
        prompt=[3, 4],
        sampling_params=SamplingParams(max_tokens=1),
    )
    request.prompt_token_ids = [3, 4]
    request.num_prompt_tokens = 2

    tokenizer = SimpleNamespace(
        encode=lambda _text: [0],
        eos_token_id=127,
        all_special_ids=[127],
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer)
    prefilled_cache, last_token = scheduler._do_external_prefill(
        request,
        request.prompt_token_ids,
        cache,
    )

    assert prefilled_cache is cache
    assert last_token == [4]
    assert len(legacy_cache.state) == 4
    assert all(state is not None for state in legacy_cache.state)


def test_scheduler_rejects_legacy_zero_slot_cache():
    bailing_hybrid = _load_patch_module()
    from mlx_lm.models.cache import ArraysCache

    from omlx.scheduler import Scheduler

    model = bailing_hybrid.Model(
        bailing_hybrid.ModelArgs.from_dict(_minimal_config())
    )
    source_cache = model.make_cache()
    logits = model(mx.array([[1, 2]], dtype=mx.int32), cache=source_cache)
    mx.eval(logits)

    tokenizer = SimpleNamespace(
        encode=lambda _text: [0],
        eos_token_id=127,
        all_special_ids=[127],
    )
    scheduler = Scheduler(model=model, tokenizer=tokenizer)

    assert scheduler._validate_cache([ArraysCache(size=0), source_cache[1]]) is False
    assert scheduler._validate_cache(source_cache) is True


def test_sanitize_remaps_moe_and_mla_weights():
    bailing_hybrid = _load_patch_module()
    model = bailing_hybrid.Model(
        bailing_hybrid.ModelArgs.from_dict(_minimal_config())
    )

    weights = {
        "model.layers.1.mlp.gate.weight": mx.ones((2, 32)),
        "model.layers.1.mlp.gate.bias": mx.ones((2,)),
        "model.layers.1.attention.kv_b_proj.weight": mx.arange(128).reshape(16, 8),
        "model.layers.2.mtp.weight": mx.ones((1,)),
    }
    for projection, shape in (
        ("gate_proj", (16, 32)),
        ("up_proj", (16, 32)),
        ("down_proj", (32, 16)),
    ):
        for expert in range(2):
            weights[f"model.layers.1.mlp.experts.{expert}.{projection}.weight"] = (
                mx.full(shape, expert + 1)
            )

    sanitized = model.sanitize(weights)

    assert "model.layers.1.mlp.gate.weight" not in sanitized
    assert "model.layers.1.mlp.gate.bias" not in sanitized
    assert sanitized["model.layers.1.mlp.gate.gate_proj.weight"].shape == (2, 32)
    assert sanitized["model.layers.1.mlp.gate.gate_proj.bias"].shape == (2,)
    assert sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"].shape == (
        2,
        16,
        32,
    )
    assert sanitized["model.layers.1.mlp.switch_mlp.up_proj.weight"].shape == (
        2,
        16,
        32,
    )
    assert sanitized["model.layers.1.mlp.switch_mlp.down_proj.weight"].shape == (
        2,
        32,
        16,
    )
    assert sanitized["model.layers.1.attention.embed_q.weight"].shape == (2, 8, 4)
    assert sanitized["model.layers.1.attention.unembed_out.weight"].shape == (
        2,
        4,
        8,
    )
    assert "model.layers.1.attention.kv_b_proj.weight" not in sanitized
    assert "model.layers.2.mtp.weight" not in sanitized


def test_sanitize_converts_block_fp8_weights_to_affine_runtime_layout():
    bailing_hybrid = _load_patch_module()
    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    source = mx.linspace(-1.0, 1.0, 16 * 64).reshape(16, 64)
    fp8 = mx.to_fp8(source)
    weight_key = "model.layers.0.attention.q_proj.weight"
    scale_key = f"{weight_key}_scale_inv"

    sanitized = model.sanitize(
        {
            weight_key: fp8,
            scale_key: mx.array([[0.5]], dtype=mx.float32),
        }
    )
    restored = mx.dequantize(
        sanitized[weight_key],
        sanitized[weight_key.replace("weight", "scales")],
        sanitized[weight_key.replace("weight", "biases")],
        group_size=64,
        bits=8,
    )
    expected = mx.from_fp8(fp8, dtype=mx.bfloat16) * 0.5
    mx.eval(restored, expected)

    assert scale_key not in sanitized
    assert sanitized[weight_key].dtype == mx.uint32
    assert mx.allclose(restored, expected, rtol=2e-2, atol=5e-3)


def test_sanitize_stacks_fp8_expert_weights_and_sidecars():
    bailing_hybrid = _load_patch_module()
    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    weights = {}
    for expert in range(2):
        prefix = f"model.layers.1.mlp.experts.{expert}.gate_proj"
        source = mx.full((16, 64), 0.25 * (expert + 1), dtype=mx.float32)
        weights[f"{prefix}.weight"] = mx.to_fp8(source)
        weights[f"{prefix}.weight_scale_inv"] = mx.ones((1, 1))

    sanitized = model.sanitize(weights)
    prefix = "model.layers.1.mlp.switch_mlp.gate_proj"

    assert sanitized[f"{prefix}.weight"].shape == (2, 16, 16)
    assert sanitized[f"{prefix}.scales"].shape == (2, 16, 1)
    assert sanitized[f"{prefix}.biases"].shape == (2, 16, 1)
    assert not any(key.endswith("weight_scale_inv") for key in sanitized)


def test_sanitize_preserves_packed_mxfp4_expert_weights():
    bailing_hybrid = _load_patch_module()
    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "routed_experts_quant_method": "mxfp4",
            "routed_experts_group_size": 32,
        },
    )
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    weights = {}
    expected = []
    for expert in range(2):
        prefix = f"model.layers.1.mlp.experts.{expert}.gate_proj"
        source = mx.linspace(-1.0, 1.0, 16 * 64).reshape(16, 64) * (expert + 1)
        packed, scales = mx.quantize(
            source,
            group_size=32,
            bits=4,
            mode="mxfp4",
        )
        weights[f"{prefix}.weight"] = packed.view(mx.int8)
        weights[f"{prefix}.weight_scale_inv"] = scales
        expected.append(
            mx.dequantize(
                packed,
                scales,
                None,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
        )

    sanitized = model.sanitize(weights)
    prefix = "model.layers.1.mlp.switch_mlp.gate_proj"
    restored = mx.dequantize(
        sanitized[f"{prefix}.weight"],
        sanitized[f"{prefix}.scales"],
        None,
        group_size=32,
        bits=4,
        mode="mxfp4",
    )
    expected = mx.stack(expected)
    mx.eval(restored, expected)

    assert sanitized[f"{prefix}.weight"].shape == (2, 16, 8)
    assert sanitized[f"{prefix}.weight"].dtype == mx.uint32
    assert sanitized[f"{prefix}.scales"].shape == (2, 16, 2)
    assert sanitized[f"{prefix}.scales"].dtype == mx.uint8
    assert not any(key.endswith("weight_scale_inv") for key in sanitized)
    assert mx.array_equal(restored, expected)


def test_sanitize_decodes_e8m0_fp8_block_scales():
    bailing_hybrid = _load_patch_module()
    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    )
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    source = mx.linspace(-1.0, 1.0, 16 * 64).reshape(16, 64)
    fp8 = mx.to_fp8(source)
    weight_key = "model.layers.0.attention.q_proj.weight"

    sanitized = model.sanitize(
        {
            weight_key: fp8,
            f"{weight_key}_scale_inv": mx.array([[126]], dtype=mx.uint8),
        }
    )
    restored = mx.dequantize(
        sanitized[weight_key],
        sanitized[weight_key.replace("weight", "scales")],
        sanitized[weight_key.replace("weight", "biases")],
        group_size=64,
        bits=8,
    )
    expected = mx.from_fp8(fp8, dtype=mx.bfloat16) * 0.5
    mx.eval(restored, expected)

    assert mx.allclose(restored, expected, rtol=2e-2, atol=5e-3)


def test_bailing_fp8_config_normalizes_to_affine_runtime_quantization():
    from omlx.utils.model_loading import normalize_bailing_hybrid_fp8_quant

    config = _minimal_config(
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        }
    )

    assert normalize_bailing_hybrid_fp8_quant(config) is config
    assert config["quantization"] == {"group_size": 64, "bits": 8}


def test_bailing_mixed_fp4_config_adds_routed_expert_overrides():
    from omlx.utils.model_loading import normalize_bailing_hybrid_fp8_quant

    config = _minimal_config(
        num_hidden_layers=3,
        first_k_dense_replace=1,
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "routed_experts_quant_method": "mxfp4",
            "routed_experts_group_size": 32,
        },
    )

    assert normalize_bailing_hybrid_fp8_quant(config) is config
    quantization = config["quantization"]
    assert quantization["group_size"] == 64
    assert quantization["bits"] == 8
    assert "model.layers.0.mlp.switch_mlp.gate_proj" not in quantization
    expected = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    for layer_idx in (1, 2):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            assert (
                quantization[
                    f"model.layers.{layer_idx}.mlp.switch_mlp.{projection}"
                ]
                == expected
            )


def test_fp8_checkpoint_loads_strictly_as_quantized_model(tmp_path):
    bailing_hybrid = _load_patch_module()
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
        },
    )
    source_model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(source_model.parameters()))
    weight_key = "model.layers.0.attention.q_proj.weight"
    source_weight = weights[weight_key]
    weights[weight_key] = mx.to_fp8(source_weight.astype(mx.float32))
    weights[f"{weight_key}_scale_inv"] = mx.ones((1, 1), dtype=mx.float32)
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))

    from mlx_lm.utils import load_model

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))
    loaded, loaded_config = load_model(tmp_path, strict=True)
    logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(logits)

    assert loaded_config["quantization"] == {"group_size": 64, "bits": 8}
    assert isinstance(loaded.model.layers[0].attention.q_proj, nn.QuantizedLinear)
    assert logits.shape == (1, 3, config["vocab_size"])


def test_mixed_fp4_checkpoint_loads_strictly(tmp_path):
    bailing_hybrid = _load_patch_module()
    from mlx.utils import tree_flatten
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    config = _minimal_config(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "routed_experts_quant_method": "mxfp4",
            "routed_experts_group_size": 32,
        },
    )
    source_model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    weights = dict(tree_flatten(source_model.parameters()))
    for projection in ("gate_proj", "up_proj", "down_proj"):
        runtime_key = f"model.layers.1.mlp.switch_mlp.{projection}.weight"
        expert_weights = weights.pop(runtime_key)
        for expert, expert_weight in enumerate(expert_weights):
            packed, scales = mx.quantize(
                expert_weight,
                group_size=32,
                bits=4,
                mode="mxfp4",
            )
            checkpoint_prefix = (
                f"model.layers.1.mlp.experts.{expert}.{projection}"
            )
            weights[f"{checkpoint_prefix}.weight"] = packed.view(mx.int8)
            weights[f"{checkpoint_prefix}.weight_scale_inv"] = scales

    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    (tmp_path / "config.json").write_text(json.dumps(config))

    from mlx_lm.utils import load_model

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))
    loaded, loaded_config = load_model(tmp_path, strict=True)
    logits = loaded(mx.array([[1, 2, 3]], dtype=mx.int32))
    mx.eval(logits)

    quantization = loaded_config["quantization"]
    expected = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    assert (
        quantization["model.layers.1.mlp.switch_mlp.gate_proj"] == expected
    )
    assert isinstance(
        loaded.model.layers[1].mlp.switch_mlp.gate_proj,
        QuantizedSwitchLinear,
    )
    assert loaded.model.layers[1].mlp.switch_mlp.gate_proj.mode == "mxfp4"
    assert logits.shape == (1, 3, config["vocab_size"])


def test_oq_discovers_ling_embeddings_and_hybrid_layer_masks():
    bailing_hybrid = _load_patch_module()
    from omlx.oq import (
        _find_model_layers,
        _layer_masks_for_model,
        _uses_quantized_source_sensitivity,
    )

    config = _minimal_config(
        quantization_config={"quant_method": "fp8"},
    )
    model = bailing_hybrid.Model(bailing_hybrid.ModelArgs.from_dict(config))
    embed_fn, layers = _find_model_layers(model)
    hidden = embed_fn(mx.array([[1, 2, 3]], dtype=mx.int32))
    masks = _layer_masks_for_model(model, layers, hidden)

    assert embed_fn is model.model.word_embeddings
    assert layers is model.model.layers
    assert masks[0] is None
    assert masks[1] is not None
    assert _uses_quantized_source_sensitivity(config) is True


def test_pre_load_dispatch_calls_bailing_hybrid_patch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "omlx.patches.bailing_hybrid.apply_bailing_hybrid_patch",
        lambda: calls.append(True) or True,
    )
    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert calls == [True]


def test_bailing_hybrid_is_discovered_as_llm(tmp_path):
    from omlx.model_discovery import detect_model_type

    (tmp_path / "config.json").write_text(json.dumps(_minimal_config()))

    assert detect_model_type(tmp_path) == "llm"
