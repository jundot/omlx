# SPDX-License-Identifier: Apache-2.0
"""ParoQuant eligibility and numerical DFlash capture/rollback regression tests."""

import json
from dataclasses import asdict

import pytest


@pytest.fixture(
    params=[
        (family, group, rotations)
        for family in ("qwen3", "qwen3_moe", "qwen3_5", "qwen3_5_moe")
        for group, rotations in ((32, 2), (64, 4), (128, 8))
    ]
)
def rotated_qwen(request):
    return make_rotated_model(*request.param)


def make_rotated_model(architecture, group_size, krot, *, for_vlm=False):
    import importlib

    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    paro = pytest.importorskip("paroquant.inference.backends.mlx.modules")
    from mlx.utils import tree_unflatten
    from mlx_lm.models.switch_layers import SwitchGLU, SwitchLinear

    from omlx.patches.mlx_lm_mtp import cache_rollback, qwen35_model

    cache_rollback.apply()
    qwen35_model.apply()
    mx.random.seed(7)
    text = dict(
        model_type=architecture,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=128,
        rms_norm_eps=1e-6,
        rope_theta=100000.0,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=32,
        linear_value_head_dim=32,
    )
    if "moe" in architecture:
        text.update(
            num_experts=4,
            num_experts_per_tok=2,
            decoder_sparse_step=1,
            mlp_only_layers=[],
            moe_intermediate_size=256,
            shared_expert_intermediate_size=256,
            norm_topk_prob=True,
        )
    cfg = (
        dict(model_type=architecture, text_config=text)
        if architecture.startswith("qwen3_5")
        else text
    )
    if for_vlm:
        from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig
        from mlx_vlm.models.qwen3_5.language import LanguageModel

        from omlx.patches.mlx_vlm_mtp import qwen35_vlm_runtime

        qwen35_vlm_runtime.apply()
        args = TextConfig.from_dict(dict(text, linear_conv_kernel_dim=4))
        model = LanguageModel(args, ModelConfig(args, VisionConfig(), architecture))
    else:
        module = importlib.import_module("mlx_lm.models." + architecture)
        model = module.Model(module.ModelArgs.from_dict(cfg))
    replacements = []
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Linear) and not name.endswith("lm_head"):
            out_dim, in_dim = layer.weight.shape
            rotated = paro.RotateQuantizedLinear(
                in_dim, out_dim, bias="bias" in layer, group_size=group_size, krot=krot
            )
            rotated.weight, rotated.scales, rotated.biases = mx.quantize(
                layer.weight, group_size=group_size, bits=4
            )
            rotated.theta = mx.random.normal((krot, in_dim // 2)) * 0.2
            rotated.pairs = mx.broadcast_to(
                mx.arange(in_dim, dtype=mx.int16) % group_size, (krot, in_dim)
            )
            rotated.channel_scales = mx.ones((1, in_dim)) * 0.9
            if "bias" in layer:
                rotated.bias = layer.bias
            replacements.append((name, rotated))
    model.update_modules(tree_unflatten(replacements))
    nn.quantize(
        model,
        group_size=group_size,
        bits=4,
        class_predicate=lambda _, m: isinstance(m, SwitchLinear),
    )
    for name, layer in list(model.named_modules()):
        if isinstance(layer, SwitchGLU):
            rotated = paro.RotateSwitchGLU(layer, group_size, krot)
            for prefix in ("gate_up_rot", "down_rot"):
                dim = getattr(rotated, prefix + "_theta").shape[1] * 2
                setattr(
                    rotated, prefix + "_theta", mx.random.normal((krot, dim // 2)) * 0.2
                )
                setattr(
                    rotated,
                    prefix + "_pairs",
                    mx.broadcast_to(
                        mx.arange(dim, dtype=mx.int16) % group_size, (krot, dim)
                    ),
                )
            model.update_modules(tree_unflatten([(name, rotated)]))
    model.eval()
    mx.eval(model.parameters())
    return model


def test_rotated_target_load_capture_and_rollback(rotated_qwen, tmp_path, monkeypatch):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models.cache import make_prompt_cache
    from paroquant.inference.backends.mlx import load as paro_loader
    from paroquant.inference.backends.mlx.modules import RotateQuantizedLinear

    from omlx.admin.routes import _dflash_compat_for_model
    from omlx.engine.batched import BatchedEngine
    from omlx.utils.model_loading import maybe_load_custom_quantization

    rotation = next(
        m
        for _, m in rotated_qwen.named_modules()
        if isinstance(m, RotateQuantizedLinear)
    )
    config = asdict(rotated_qwen.args)
    config["quantization_config"] = {
        "quant_method": "paroquant",
        "bits": 4,
        "group_size": rotation.group_size,
        "krot": rotation.theta.shape[0],
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        dict(tree_flatten(rotated_qwen.parameters())),
    )
    # Only tokenizer construction is replaced; model loading and rotations are real.
    monkeypatch.setattr(paro_loader, "_load_processor", lambda *_: object())
    assert _dflash_compat_for_model(
        {
            "model_path": str(tmp_path),
            "config_model_type": config["model_type"],
            "engine_type": "batched",
        }
    ) == (True, "")
    model, _ = maybe_load_custom_quantization(str(tmp_path), is_vlm=False)
    ids = mx.array([[1, 2, 3]])
    expected = model(ids, cache=make_prompt_cache(model))
    engine = BatchedEngine(str(tmp_path))
    engine._model = model
    engine.prepare_dflash_target()
    host = engine._dflash_host()
    actual = host(
        ids,
        cache=make_prompt_cache(model),
        return_hidden=True,
        capture_layer_ids=[0, 3],
    )
    mx.eval(expected, actual.logits)
    assert float(mx.max(mx.abs(actual.logits - expected))) < 1e-4
    assert len(actual.hidden_states) == 3
    for accepted in range(4):
        prefix, proposed, next_token = [1, 2, 3], [4, 5, 6, 7], [8]
        reference = model(
            mx.array([prefix + proposed[: accepted + 1] + next_token]),
            cache=make_prompt_cache(model),
        )[:, -1:, :]
        cache = make_prompt_cache(model)
        host(mx.array([prefix]), cache=cache)
        out = host(
            mx.array([proposed]),
            cache=cache,
            return_hidden=True,
            capture_layer_ids=[0, 3],
            n_confirmed=1,
        )
        mx.eval(out.logits)
        host.rollback_speculative_cache(
            cache, getattr(out, "gdn_states", None), accepted, 4
        )
        result = host(mx.array([next_token]), cache=cache)[:, -1:, :]
        mx.eval(reference, result)
        assert float(mx.max(mx.abs(result - reference))) < 1e-4
        assert mx.array_equal(mx.argmax(result, axis=-1), mx.argmax(reference, axis=-1))


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_type", "gemma4"),
        ("model_type", "qwen2"),
        ("hidden_size", 0),
        ("bits", 8),
        ("group_size", 16),
        ("krot", 0),
    ],
)
def test_unvalidated_paroquant_layout_remains_rejected(tmp_path, field, value):
    from omlx.admin.routes import _dflash_compat_for_model

    config = {
        "model_type": "qwen3",
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "vocab_size": 128,
        "quantization_config": {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 32,
            "krot": 2,
        },
    }
    if field in ("bits", "group_size", "krot"):
        config["quantization_config"][field] = value
    else:
        config[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert not _dflash_compat_for_model(
        {
            "model_path": str(tmp_path),
            "config_model_type": config["model_type"],
            "engine_type": "batched",
        }
    )[0]


@pytest.mark.parametrize("family", ["qwen3_5", "qwen3_5_moe"])
def test_vlm_rotated_target_capture_and_rollback(family):
    from types import SimpleNamespace

    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    from omlx.engine.vlm import VLMBatchedEngine

    model = make_rotated_model(family, 32, 2, for_vlm=True)
    engine = VLMBatchedEngine("tiny-paroquant-vlm")
    engine._vlm_model = SimpleNamespace(language_model=model)
    assert engine.prepare_dflash_target().language_model is model
    prefix, proposed, next_token = [1, 2, 3], [4, 5, 6, 7], [8]
    reference = model(mx.array([prefix]), cache=make_prompt_cache(model)).logits
    actual = model(
        mx.array([prefix]),
        cache=make_prompt_cache(model),
        return_hidden=True,
        capture_layer_ids=[0, 3],
    )
    mx.eval(reference, actual.logits)
    assert float(mx.max(mx.abs(actual.logits - reference))) < 1e-4
    for accepted in range(4):
        reference = model(
            mx.array([prefix + proposed[: accepted + 1] + next_token]),
            cache=make_prompt_cache(model),
        ).logits[:, -1:, :]
        cache = make_prompt_cache(model)
        model(mx.array([prefix]), cache=cache)
        out = model(
            mx.array([proposed]),
            cache=cache,
            return_hidden=True,
            capture_layer_ids=[0, 3],
            n_confirmed=1,
        )
        mx.eval(out.logits)
        model.rollback_speculative_cache(
            cache, getattr(out, "gdn_states", None), accepted, 4
        )
        actual = model(mx.array([next_token]), cache=cache).logits[:, -1:, :]
        mx.eval(reference, actual)
        assert float(mx.max(mx.abs(actual - reference))) < 1e-4


@pytest.mark.parametrize(
    "family,engine_type",
    [
        ("qwen3", "batched"),
        ("qwen3_moe", "batched"),
        ("qwen3_5", "batched"),
        ("qwen3_5_moe", "batched"),
        ("qwen3_5", "vlm"),
        ("qwen3_5_moe", "vlm"),
    ],
)
def test_valid_paroquant_gate_does_not_require_importing_optional_loader(
    tmp_path, family, engine_type
):
    from omlx.admin.routes import _dflash_compat_for_model

    config = {
        "model_type": family,
        "hidden_size": 128,
        "num_hidden_layers": 4,
        "vocab_size": 128,
        "quantization_config": {
            "quant_method": "paroquant",
            "bits": 4,
            "group_size": 32,
            "krot": 2,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    assert _dflash_compat_for_model(
        {
            "model_path": str(tmp_path),
            "config_model_type": family,
            "engine_type": engine_type,
        }
    ) == (True, "")
