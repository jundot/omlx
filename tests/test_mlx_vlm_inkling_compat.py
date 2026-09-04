# SPDX-License-Identifier: Apache-2.0
"""Inkling mlx-vlm compatibility patch tests.

Covers the vendor install/discovery surface (unlimited-ocr test pattern),
the torch-free processor pieces, the NVFP4 config translation, and the
batched right-padded prefill parity that the vendored conv_mask wiring
(G2) exists for.
"""

from __future__ import annotations

import json

import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")


@pytest.fixture()
def strict_math_device():
    """Use deterministic CPU reductions for compact/padded KV parity."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture(scope="module")
def applied():
    from omlx.patches.mlx_vlm_inkling_compat import (
        apply_mlx_vlm_inkling_compat_patch,
        is_applied,
    )

    apply_mlx_vlm_inkling_compat_patch()
    assert is_applied()
    return True


def test_vendor_module_resolves(applied):
    import mlx_vlm.utils as vlm_utils

    assert vlm_utils.MODEL_REMAPPING.get("inkling_mm_model") == "inkling"

    import importlib

    module = importlib.import_module("mlx_vlm.models.inkling")
    assert hasattr(module, "Model")
    assert hasattr(module, "LanguageModel")

    # get_model_and_args resolves the checkpoint model_type.
    arch, model_type = _get_model_and_args(vlm_utils, "inkling_mm_model")
    assert model_type == "inkling"
    assert arch is module


def _get_model_and_args(vlm_utils, model_type):
    config = {"model_type": model_type}
    result = vlm_utils.get_model_and_args(config)
    # Signature drift guard: pinned mlx-vlm returns (arch_module, model_type)
    # or (arch, model_type, quant) depending on version.
    return result[0], result[1]


def test_prompt_formatting_image_first(applied):
    from mlx_vlm.prompt_utils import get_message_json

    message = get_message_json(
        "inkling_mm_model", "describe this", role="user", num_images=2
    )
    assert message["role"] == "user"
    content = message["content"]
    assert isinstance(content, list)
    assert content[0] == {"type": "image"}
    assert content[1] == {"type": "image"}
    assert content[2]["type"] == "text"
    assert content[2]["text"] == "describe this"

    # Assistant/no-image turns stay plain strings.
    assistant = get_message_json("inkling", "hello", role="assistant")
    assert assistant["content"] == "hello"


def test_other_models_untouched(applied):
    from mlx_vlm.prompt_utils import get_message_json

    message = get_message_json("qwen2_vl", "hi", role="user", num_images=1)
    assert message["role"] == "user"
    assert message["content"] != [{"type": "image"}, {"type": "text", "text": "hi"}]


def test_load_config_translates_nvfp4(applied, tmp_path):
    import mlx_vlm.utils as vlm_utils

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling_mm_model", "vocab_size": 128})
    )
    (tmp_path / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}})
    )
    config = vlm_utils.load_config(tmp_path)
    assert config["quantization"] == {"group_size": 16, "bits": 4, "mode": "nvfp4"}

    # Non-inkling checkpoints are not touched.
    other = tmp_path / "other"
    other.mkdir()
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (other / "hf_quant_config.json").write_text(
        json.dumps({"quantization": {"quant_algo": "NVFP4"}})
    )
    config = vlm_utils.load_config(other)
    assert "quantization" not in config


def test_raw_inkling_layout_detection_uses_weight_index(applied, tmp_path):
    from omlx.patches.mlx_vlm_inkling_compat import _has_raw_inkling_weights

    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "inkling_mm_model"})
    )
    index_path = tmp_path / "model.safetensors.index.json"
    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.llm.layers.0.attn.wq_du.weight": "model-1.safetensors"
                }
            }
        )
    )
    assert _has_raw_inkling_weights(tmp_path)

    index_path.write_text(
        json.dumps(
            {
                "weight_map": {
                    "language_model.model.layers.0.self_attn.qkvr_proj.weight": (
                        "model-1.safetensors"
                    )
                }
            }
        )
    )
    assert not _has_raw_inkling_weights(tmp_path)


@pytest.mark.parametrize(("raw_layout", "sanitize_calls"), [(True, 1), (False, 0)])
def test_load_model_forces_sanitize_only_for_raw_inkling(
    applied, tmp_path, monkeypatch, raw_layout, sanitize_calls
):
    from types import SimpleNamespace

    import mlx.nn as nn
    import mlx_vlm.utils as vlm_utils
    import numpy as np
    from safetensors.numpy import save_file

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "inkling_mm_model",
                "text_config": {},
                "quantization": {"group_size": 64, "bits": 4},
            }
        )
    )
    prefix = "model.llm." if raw_layout else ""
    save_file(
        {
            prefix + "linear.weight": np.zeros((64, 8), dtype=np.uint32),
            prefix + "linear.scales": np.ones((64, 1), dtype=np.float16),
            prefix + "linear.biases": np.zeros((64, 1), dtype=np.float16),
        },
        tmp_path / "model.safetensors",
        metadata={"format": "mlx"},
    )

    class FakeModelConfig:
        @classmethod
        def from_dict(cls, _config):
            return SimpleNamespace()

    class FakeModel(nn.Module):
        calls = 0

        def __init__(self, _config):
            super().__init__()
            self.linear = nn.Linear(64, 64, bias=False)

        def sanitize(self, weights):
            type(self).calls += 1
            return {
                key.removeprefix("model.llm."): value for key, value in weights.items()
            }

    arch = SimpleNamespace(ModelConfig=FakeModelConfig, Model=FakeModel)
    monkeypatch.setattr(
        vlm_utils, "get_model_and_args", lambda config: (arch, "inkling")
    )
    monkeypatch.setattr(
        vlm_utils,
        "update_module_configs",
        lambda model_config, *_args: model_config,
    )
    monkeypatch.setattr(
        vlm_utils,
        "apply_generation_config_defaults",
        lambda model_config, _config: model_config,
    )

    FakeModel.calls = 0
    model = vlm_utils.load_model(tmp_path, lazy=True)

    assert FakeModel.calls == sanitize_calls
    assert isinstance(model.linear, nn.QuantizedLinear)


def test_model_load_weights_remaps_legacy_mlx_layouts(applied, monkeypatch):
    from types import SimpleNamespace

    import mlx.nn as nn
    from mlx_vlm.models.inkling.inkling import Model

    prefix = "language_model.model.layers.0.self_attn."
    weights = {
        **{
            f"{prefix}{name}_proj.weight": mx.full((2, 4), index + 1)
            for index, name in enumerate("qkvr")
        },
        "language_model.model.layers.0.mlp.shared_experts.gate_proj.weight": (
            mx.zeros((2, 4, 8))
        ),
        "language_model.model.layers.0.mlp.shared_experts.down_proj.weight": (
            mx.zeros((2, 8, 4))
        ),
    }
    loaded = {}

    def capture_load_weights(_self, transformed, strict=True):
        assert strict
        loaded.update(dict(transformed))
        return _self

    monkeypatch.setattr(nn.Module, "load_weights", capture_load_weights)
    model = Model.__new__(Model)
    model.config = SimpleNamespace(text_config=_tiny_text_config())
    model.load_weights(list(weights.items()))

    assert loaded[prefix + "qkvr_proj.weight"].shape == (8, 4)
    assert not any(f"{prefix}{name}_proj.weight" in loaded for name in "qkvr")
    assert loaded[
        "language_model.model.layers.0.mlp.shared_experts.gate_proj.weight"
    ].shape == (8, 8)
    assert loaded[
        "language_model.model.layers.0.mlp.shared_experts.down_proj.weight"
    ].shape == (8, 8)


def test_image_processor_patch_grid(applied):
    import importlib

    import numpy as np
    from PIL import Image

    processing_inkling = importlib.import_module(
        "mlx_vlm.models.inkling.processing_inkling"
    )

    proc = processing_inkling.InklingImageProcessor()
    image = Image.fromarray(
        np.full((100, 50, 3), 128, dtype=np.uint8)
    )  # H=100, W=50
    out = proc.preprocess([image])
    # rows = ceil(100/40) = 3, cols = 50//40 + 1 = 2 (reference grid).
    assert out["num_patches"].tolist() == [6]
    assert out["pixel_values"].shape == (6, 2, 40, 40, 3)
    # Padded region carries -1.0 pre-rescale: (-1 * 1/255 - mean) / std.
    # Patch 1 covers x = [40, 80); the image ends at x = 50, so patch-local
    # x >= 10 is padding.
    padded_pixel = out["pixel_values"][1, 0, 0, 20, 0]
    expected = (-1.0 / 255.0 - proc.image_mean[0]) / proc.image_std[0]
    assert abs(float(padded_pixel) - float(expected)) < 1e-5
    # Temporal duplication is exact.
    assert np.array_equal(
        out["pixel_values"][:, 0], out["pixel_values"][:, 1]
    )


def _tiny_text_config():
    from mlx_vlm.models.inkling.config import TextConfig

    return TextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        vocab_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=8,
        sliding_window_size=8,
        layer_types=["hybrid_sliding", "full"],
        d_rel=4,
        rel_extent=4,
        log_scaling_n_floor=4,
        sconv_kernel_size=4,
        mlp_layer_types=["dense", "sparse"],
        intermediate_size=16,
        dense_intermediate_size=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        tie_word_embeddings=True,
    )


def _tiny_language_model():
    from mlx_vlm.models.inkling.language import LanguageModel

    mx.random.seed(7)
    model = LanguageModel(_tiny_text_config())
    # Give routing and rel-bias non-degenerate weights.
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.rel_proj = (
            mx.random.normal(attn.rel_proj.shape).astype(attn.rel_proj.dtype) * 0.05
        )
        if hasattr(layer.mlp, "gate_weight"):
            layer.mlp.gate_weight = (
                mx.random.normal(layer.mlp.gate_weight.shape) * 0.05
            )
    mx.eval(model.parameters())
    return model


def test_tiny_model_single_forward(applied):
    model = _tiny_language_model()
    cache = model.make_cache()
    tokens = mx.array([[1, 5, 9, 13, 17]])
    out = model(tokens, cache=cache)
    assert out.logits.shape == (1, 5, 128)
    step = model(mx.array([[21]]), cache=cache)
    assert step.logits.shape == (1, 1, 128)
    kv_state = cache[0][0].state
    assert kv_state[0].shape[2] == 6
    conv_slots = list(cache[0][1].state)
    assert len(conv_slots) == 4
    assert all(s is not None for s in conv_slots)


def test_dense_intermediate_size_required(applied):
    from mlx_vlm.models.inkling.language import LanguageModel

    config = _tiny_text_config()
    config.dense_intermediate_size = None
    with pytest.raises(ValueError, match="dense_intermediate_size"):
        LanguageModel(config)


def test_batched_right_padded_prefill_parity(applied, strict_math_device):
    """G2: a short request prefILLED inside a right-padded batch must end
    with the same conv states and next-token logits as the same request
    run alone. Without the vendored conv_mask / lengths-aware state /
    key-masking wiring, the pad rows pollute the short-conv states and
    the attention keys."""
    from mlx_lm.models.cache import CacheList

    model = _tiny_language_model()

    tokens_a = [3, 17, 44, 91, 12, 7, 63]  # length 7
    tokens_b = [8, 22, 5, 99, 41, 33, 27, 54, 76, 11, 90, 2]  # length 12
    la, lb = len(tokens_a), len(tokens_b)

    # Single-request reference for A.
    cache_a = model.make_cache()
    logits_a = model(mx.array([tokens_a]), cache=cache_a).logits
    mx.eval(logits_a)

    # Batched: merge fresh per-request caches (the BatchGenerator path),
    # right-pad, chunked prefill, finalize.
    cache_1 = model.make_cache()
    cache_2 = model.make_cache()
    merged = [
        CacheList.merge([c1, c2]) for c1, c2 in zip(cache_1, cache_2)
    ]
    padded = [tokens_a + [0] * (lb - la), tokens_b]
    for c in merged:
        c.prepare(lengths=[la, lb], right_padding=[lb - la, 0])

    chunk = 5
    batch_tokens = mx.array(padded)
    logits_chunks = []
    for start in range(0, lb, chunk):
        out = model(batch_tokens[:, start : start + chunk], cache=merged)
        logits_chunks.append(out.logits)
    logits_batch = mx.concatenate(logits_chunks, axis=1)
    for c in merged:
        c.finalize()
    mx.eval(logits_batch)

    # Conv states of A inside the batch == single-run states.
    for layer_idx in range(2):
        batch_conv = merged[layer_idx][1]
        single_conv = cache_a[layer_idx][1]
        for slot in range(4):
            got = batch_conv[slot][0:1]
            want = single_conv[slot]
            assert mx.max(mx.abs(got - want)).item() < 1e-4, (
                f"layer {layer_idx} conv slot {slot} diverged in batch "
                "(pad pollution)"
            )

    # Last valid-token logits of A == single-run logits.
    diff = mx.max(
        mx.abs(logits_batch[0, la - 1] - logits_a[0, -1])
    ).item()
    assert diff < 1e-3, f"prefill logits diverged: {diff}"

    # One decode step: exercises left_padding key masking + per-seq tau.
    step_a = model(mx.array([[100]]), cache=cache_a).logits
    step_batch = model(mx.array([[100], [101]]), cache=merged).logits
    mx.eval(step_a, step_batch)
    diff = mx.max(mx.abs(step_batch[0, 0] - step_a[0, 0])).item()
    assert diff < 1e-3, f"decode logits diverged: {diff}"


def test_sanitize_maps_bf16_checkpoint_keys(applied):
    """The vendored sanitize must cover the bf16 original repo's key
    layout: attn projections, sconv transpose, router bias, and the
    interleaved w13 expert de-interleave."""
    import importlib

    inkling_mod = importlib.import_module("mlx_vlm.models.inkling.inkling")

    model = inkling_mod.Model.__new__(inkling_mod.Model)  # sanitize is pure

    hidden, inter, n_experts = 8, 4, 2
    w13 = mx.arange(n_experts * 2 * inter * hidden, dtype=mx.float32).reshape(
        n_experts, 2 * inter, hidden
    )
    w2 = mx.ones((n_experts, hidden, inter))
    sconv = mx.arange(hidden * 4, dtype=mx.float32).reshape(hidden, 1, 4)
    weights = {
        "model.llm.layers.1.attn.wq_du.weight": mx.zeros((hidden, hidden)),
        "model.llm.layers.1.attn.wk_dv.weight": mx.zeros((hidden, hidden)),
        "model.llm.layers.1.attn.wv_dv.weight": mx.zeros((hidden, hidden)),
        "model.llm.layers.1.attn.wr_du.weight": mx.zeros((hidden, hidden)),
        "model.llm.layers.1.attn.rel_logits_proj.proj": mx.zeros((4, 8)),
        "model.llm.layers.1.attn.k_sconv.weight": sconv,
        "model.llm.layers.1.attn_sconv.weight": sconv,
        "model.llm.layers.1.mlp.gate.weight": mx.zeros((n_experts + 1, hidden)),
        "model.llm.layers.1.mlp.gate.bias": mx.zeros((n_experts,)),
        "model.llm.layers.1.mlp.gate.global_scale": mx.ones((1,)),
        "model.llm.layers.1.mlp.experts.w13_weight": w13,
        "model.llm.layers.1.mlp.experts.w2_weight": w2,
        "model.llm.embed.weight": mx.zeros((16, hidden)),
        "model.llm.unembed.weight": mx.zeros((16, hidden)),
        "model.mtp.layers.0.input_proj.weight": mx.zeros((4, 4)),
    }
    out = inkling_mod.Model.sanitize(model, weights)

    prefix = "language_model.model.layers.1."
    qkvr = out[prefix + "self_attn.qkvr_proj.weight"]
    assert qkvr.shape == (4 * hidden, hidden)
    assert prefix + "self_attn.q_proj.weight" not in out
    assert prefix + "self_attn.rel_proj" in out
    assert out[prefix + "self_attn.k_sconv.conv.weight"].shape == (hidden, 4, 1)
    assert out[prefix + "attn_sconv.conv.weight"].shape == (hidden, 4, 1)
    assert prefix + "mlp.gate_weight" in out
    assert prefix + "mlp.e_score_correction_bias" in out
    assert prefix + "mlp.global_scale" in out
    assert "language_model.model.embed_tokens.weight" in out
    assert "language_model.lm_head.weight" in out
    # Raw mtp keys never leak; with the Lightning MTP hook installed
    # (process-wide once any MTP-aware sanitize ran) they map to
    # language_model.mtp.*, otherwise they are dropped.
    assert not any(k.startswith("model.mtp") for k in out)

    gate = out[prefix + "mlp.switch_mlp.gate_proj.weight"]
    up = out[prefix + "mlp.switch_mlp.up_proj.weight"]
    assert gate.shape == (n_experts, inter, hidden)
    # w13 rows interleave gate/up: gate = rows 0,2,4..., up = rows 1,3,5...
    ref = w13.reshape(n_experts, inter, 2, hidden)
    assert mx.array_equal(gate, ref[:, :, 0, :])
    assert mx.array_equal(up, ref[:, :, 1, :])
    assert mx.array_equal(out[prefix + "mlp.switch_mlp.down_proj.weight"], w2)
    # bf16 path synthesizes identity per-expert scales.
    assert mx.array_equal(
        out[prefix + "mlp.switch_mlp.gate_scale"], mx.ones((n_experts,))
    )


def test_sanitize_maps_community_experts_only_layout(applied):
    from types import SimpleNamespace

    from mlx_vlm.models.inkling.inkling import Model

    model = Model.__new__(Model)
    model.config = SimpleNamespace(text_config=_tiny_text_config())
    hidden, inter, n_experts = 8, 4, 2
    sconv = mx.arange(hidden * 4, dtype=mx.float32).reshape(hidden, 4, 1)
    weights = {
        **{
            f"model.llm.layers.1.attn.{name}.weight": mx.full(
                (hidden, hidden), index + 1
            )
            for index, name in enumerate(("wq_du", "wk_dv", "wv_dv", "wr_du"))
        },
        "model.llm.layers.0.mlp.gate_proj.weight": mx.zeros((inter, hidden)),
        "model.llm.layers.0.mlp.gate_proj.scales": mx.ones((inter, 1)),
        "model.llm.layers.0.mlp.gate_proj.biases": mx.zeros((inter, 1)),
        "model.llm.layers.1.mlp.experts.gate_proj.weight": mx.zeros(
            (n_experts, inter, 2), dtype=mx.uint32
        ),
        "model.llm.layers.1.mlp.experts.gate_proj.scales": mx.ones(
            (n_experts, inter, 1)
        ),
        "model.llm.layers.1.mlp.experts.gate_proj.biases": mx.zeros(
            (n_experts, inter, 1)
        ),
        "model.llm.layers.1.mlp.experts.up_proj.weight": mx.zeros(
            (n_experts, inter, 2), dtype=mx.uint32
        ),
        "model.llm.layers.1.mlp.experts.down_proj.weight": mx.zeros(
            (n_experts, hidden, 1), dtype=mx.uint32
        ),
        "model.llm.layers.1.attn.k_sconv.weight": sconv,
    }

    out = Model.sanitize(model, weights)

    dense = "language_model.model.layers.0.mlp.gate_proj."
    assert all(dense + leaf in out for leaf in ("weight", "scales", "biases"))
    prefix = "language_model.model.layers.1."
    assert out[prefix + "self_attn.qkvr_proj.weight"].shape == (
        4 * hidden,
        hidden,
    )
    assert prefix + "self_attn.qkvr_proj.scales" not in out
    assert mx.array_equal(out[prefix + "self_attn.k_sconv.conv.weight"], sconv)
    switch = prefix + "mlp.switch_mlp."
    assert all(
        switch + "gate_proj." + leaf in out for leaf in ("weight", "scales", "biases")
    )
    assert mx.array_equal(out[switch + "gate_scale"], mx.ones((n_experts,)))
    assert mx.array_equal(out[switch + "out_scale"], mx.ones((n_experts,)))


def test_sanitize_maps_community_uniform_affine_qkvr_sidecars(applied):
    from types import SimpleNamespace

    from mlx_vlm.models.inkling.inkling import Model

    model = Model.__new__(Model)
    model.config = SimpleNamespace(text_config=_tiny_text_config())
    rows = {"wq_du": 4, "wk_dv": 2, "wv_dv": 2, "wr_du": 4}
    weights = {}
    expected = {leaf: [] for leaf in ("weight", "scales", "biases")}
    for index, (name, out_rows) in enumerate(rows.items(), start=1):
        parts = {
            "weight": mx.full((out_rows, 2), index, dtype=mx.uint32),
            "scales": mx.full((out_rows, 1), index, dtype=mx.float16),
            "biases": mx.full((out_rows, 1), -index, dtype=mx.float16),
        }
        for leaf, value in parts.items():
            weights[f"model.llm.layers.0.attn.{name}.{leaf}"] = value
            expected[leaf].append(value)

    for leaf, value in {
        "weight": mx.zeros((8, 2), dtype=mx.uint32),
        "scales": mx.ones((8, 1)),
        "biases": mx.zeros((8, 1)),
    }.items():
        weights[f"model.llm.layers.0.attn.wo_ud.{leaf}"] = value
        weights[f"model.visual.layers.linear_1.{leaf}"] = value
        weights[f"model.llm.embed.{leaf}"] = value
        weights[f"model.llm.unembed.{leaf}"] = value
        weights[f"model.audio.encoder.{leaf}"] = value

    out = Model.sanitize(model, weights)

    attn = "language_model.model.layers.0.self_attn."
    for leaf, parts in expected.items():
        key = attn + "qkvr_proj." + leaf
        assert mx.array_equal(out[key], mx.concatenate(parts, axis=0))
        assert all(attn + name + "_proj." + leaf not in out for name in "qkvr")
    for leaf in ("weight", "scales", "biases"):
        assert attn + "o_proj." + leaf in out
        assert f"vision_tower.encoder_layers.1.projection.{leaf}" in out
        assert "language_model.model.embed_tokens." + leaf in out
        assert "language_model.lm_head." + leaf in out
        assert "audio_tower.embed_audio_tokens." + leaf in out


def test_qkvr_fusion_policy_preserves_mixed_quant_layers(applied):
    from mlx_vlm.models.inkling.config import ModelConfig
    from mlx_vlm.models.inkling.language import InklingAttention

    base = {"bits": 4, "group_size": 64, "mode": "affine"}
    quantization = {
        **base,
        "language_model.model.layers.0.self_attn.v_proj": {
            "bits": 6,
            "group_size": 64,
            "mode": "affine",
        },
    }
    config = ModelConfig.from_dict(
        {
            "text_config": {
                "hidden_size": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "swa_num_attention_heads": 4,
                "swa_num_key_value_heads": 2,
                "swa_head_dim": 16,
            },
            "quantization": quantization,
            "quantization_config": quantization,
        }
    )

    assert config.text_config.qkvr_fused_layers == [False, True]
    assert not any(key.endswith("qkvr_proj") for key in config.quantization)
    split = InklingAttention(config.text_config, 0)
    fused = InklingAttention(config.text_config, 1)
    assert hasattr(split, "q_proj") and not hasattr(split, "qkvr_proj")
    assert hasattr(fused, "qkvr_proj") and not hasattr(fused, "q_proj")


def test_fuse_qkvr_only_stacks_compatible_layers(applied):
    from mlx_vlm.models.inkling.language import fuse_qkvr

    config = _tiny_text_config()
    config.qkvr_fused_layers = [False, True]
    weights = {}
    for layer_idx in range(2):
        prefix = f"language_model.model.layers.{layer_idx}.self_attn."
        for proj_idx, name in enumerate("qkvr"):
            weights[f"{prefix}{name}_proj.weight"] = mx.full(
                (2, 4), proj_idx + 1
            )

    out = fuse_qkvr(weights, config)
    split_prefix = "language_model.model.layers.0.self_attn."
    fused_prefix = "language_model.model.layers.1.self_attn."
    assert all(f"{split_prefix}{name}_proj.weight" in out for name in "qkvr")
    assert f"{split_prefix}qkvr_proj.weight" not in out
    fused = out[f"{fused_prefix}qkvr_proj.weight"]
    assert fused.shape == (8, 4)
    assert not any(f"{fused_prefix}{name}_proj.weight" in out for name in "qkvr")


def test_shared_experts_dense_weight_remap(applied):
    from mlx_vlm.models.inkling.language import shared_experts_to_dense

    weights = {
        "layer.mlp.shared_experts.gate_proj.weight": mx.zeros((2, 4, 8)),
        "layer.mlp.shared_experts.up_proj.scales": mx.zeros((2, 4, 1)),
        "layer.mlp.shared_experts.down_proj.weight": mx.zeros((2, 8, 4)),
    }
    out = shared_experts_to_dense(weights)
    assert out["layer.mlp.shared_experts.gate_proj.weight"].shape == (8, 8)
    assert out["layer.mlp.shared_experts.up_proj.scales"].shape == (8, 1)
    assert out["layer.mlp.shared_experts.down_proj.weight"].shape == (8, 8)


def test_moe_route_kernel_matches_reference(applied):
    from mlx_vlm.models.inkling.language import InklingSparseMoE

    moe = InklingSparseMoE(_tiny_text_config())
    logits = mx.array(
        [[0.4, -0.2, 1.1, 0.7, -0.3], [-0.6, 0.8, 0.2, 1.3, 0.1]],
        dtype=mx.float32,
    )
    moe.e_score_correction_bias = mx.array([0.03, -0.01, 0.02, 0.0])
    idx, topk_w, gamma = moe._route(logits)

    scores = mx.sigmoid(logits[:, :4]) + moe.e_score_correction_bias
    expected_idx = mx.argsort(-scores, axis=-1)[:, :2]
    selected = mx.take_along_axis(logits[:, :4], expected_idx, axis=-1)
    combined = mx.concatenate([selected, logits[:, 4:]], axis=-1)
    log_weights = -mx.logaddexp(mx.zeros_like(combined), -combined)
    weights = mx.exp(
        log_weights - mx.logsumexp(log_weights, axis=-1, keepdims=True)
    ) * moe.route_scale
    expected_gamma = mx.repeat(weights[:, 2:], moe.intermediate_size, axis=-1)
    mx.eval(idx, topk_w, gamma, expected_idx, weights, expected_gamma)

    assert mx.array_equal(idx, expected_idx.astype(mx.uint32))
    assert mx.max(mx.abs(topk_w - weights[:, :2])).item() < 1e-5
    assert mx.max(mx.abs(gamma - expected_gamma)).item() < 1e-5


def test_sconv_decode_kernel_matches_masked_fallback(applied):
    from mlx_lm.models.cache import ArraysCache
    from mlx_vlm.models.inkling.language import InklingShortConvolution

    mx.random.seed(13)
    conv = InklingShortConvolution(32, 4, 0)
    x = mx.random.normal((2, 3, 32)).astype(mx.bfloat16)
    residual = mx.random.normal((2, 3, 32)).astype(mx.bfloat16)
    fused_cache = ArraysCache(1)
    fallback_cache = ArraysCache(1)

    fused = conv(x, cache=fused_cache, residual=residual)
    fallback = conv(
        x,
        cache=fallback_cache,
        mask=mx.ones((2, 3), dtype=mx.bool_),
        residual=residual,
    )
    mx.eval(fused, fallback, fused_cache[0], fallback_cache[0])
    # The fused accumulation can move by one bfloat16 ULP versus Conv1d.
    assert mx.max(mx.abs(fused - fallback)).item() <= 0.0078125
    assert mx.max(mx.abs(fused_cache[0] - fallback_cache[0])).item() == 0


def test_quantized_down_combine_kernel_matches_dequantized_reference(applied):
    from mlx_vlm.models.inkling.language import _down_combine_kernel

    mx.random.seed(17)
    n_tokens, top_k, n_experts = 2, 6, 8
    input_dims, output_dims = 2048, 64
    weights = mx.random.normal((n_experts, output_dims, input_dims)).astype(
        mx.bfloat16
    )
    packed, scales, biases = mx.quantize(
        weights, group_size=64, bits=4, mode="affine"
    )
    inputs = (
        mx.random.normal((n_tokens, top_k, input_dims)) * 0.01
    ).astype(mx.bfloat16)
    indices = mx.array(
        [[0, 2, 3, 5, 6, 7], [1, 2, 4, 5, 6, 7]], dtype=mx.uint32
    )
    route_weights = mx.softmax(
        mx.random.normal((n_tokens, top_k)).astype(mx.float32), axis=-1
    ).astype(mx.bfloat16)

    fused = _down_combine_kernel(
        inputs=[inputs, packed, scales, biases, indices, route_weights],
        template=[
            ("T", mx.bfloat16),
            ("OUT", output_dims),
            ("IN", input_dims),
            ("GROUPS", input_dims // 64),
            ("K", top_k),
        ],
        grid=(256, output_dims, n_tokens),
        threadgroup=(256, 1, 1),
        output_shapes=[(n_tokens, output_dims)],
        output_dtypes=[mx.bfloat16],
    )[0]

    reference_rows = []
    for token_idx in range(n_tokens):
        expert_rows = []
        for route_idx in range(top_k):
            expert_idx = int(indices[token_idx, route_idx].item())
            weight = mx.dequantize(
                packed[expert_idx],
                scales[expert_idx],
                biases[expert_idx],
                group_size=64,
                bits=4,
                mode="affine",
            )
            expert_rows.append(inputs[token_idx, route_idx] @ weight.T)
        expert_rows = mx.stack(expert_rows).astype(mx.bfloat16)
        reference_rows.append(
            (expert_rows * route_weights[token_idx, :, None])
            .astype(mx.bfloat16)
            .astype(mx.float32)
            .sum(axis=0)
            .astype(mx.bfloat16)
        )
    reference = mx.stack(reference_rows)
    mx.eval(fused, reference)

    assert mx.max(mx.abs(fused - reference)).item() <= 0.03125


def test_cache_snapshot_restores_empty_composite_cache(applied):
    from mlx_vlm.models.inkling.language import (
        _restore_cache_state,
        _snapshot_cache_state,
    )

    model = _tiny_language_model()
    cache = model.make_cache()
    snapshot = _snapshot_cache_state(cache)
    model(mx.array([[1, 2, 3]]), cache=cache)
    assert cache[0][0].keys is not None
    assert cache[0][1][0] is not None

    _restore_cache_state(cache, snapshot)
    assert cache[0][0].keys is None
    assert all(cache[0][1][slot] is None for slot in range(4))


def test_sliding_window_slice_parity(applied, monkeypatch):
    """Slicing sliding-layer K/V to the window must match full-sequence
    SDPA (masked keys contribute exactly zero after softmax)."""
    import importlib

    language = importlib.import_module("mlx_vlm.models.inkling.language")
    model = _tiny_language_model()
    # window (sliding_window_size=8) well exceeded by prompt + decode.
    tokens = [(i * 37 + 11) % 128 for i in range(24)]

    def run():
        cache = model.make_cache()
        logits = [model(mx.array([tokens]), cache=cache).logits[:, -1]]
        for step in range(4):
            logits.append(model(mx.array([[step + 1]]), cache=cache).logits[:, -1])
        out = mx.concatenate(logits, axis=0)
        mx.eval(out)
        return out

    monkeypatch.setattr(language, "_SLIDING_WINDOW_SLICE", False)
    reference = run()
    monkeypatch.setattr(language, "_SLIDING_WINDOW_SLICE", True)
    sliced = run()

    diff = mx.max(mx.abs(reference - sliced)).item()
    assert diff < 2e-5, f"sliding-window slice diverged: {diff}"


def test_attention_bias_transient_registration():
    """The banded-mask transient must be priced into the SDPA estimate
    when registered, and cleared registrations must restore the base
    estimate (process-wide registry across model swaps)."""
    from omlx.memory_monitor import (
        MemoryMonitor,
        register_attention_bias_transient,
    )

    monitor = MemoryMonitor.__new__(MemoryMonitor)
    monitor._head_dim = 128
    monitor._num_attention_heads = 32
    monitor._num_kv_heads = 8
    monitor._score_dtype_size = 2

    try:
        register_attention_bias_transient(None)
        base = monitor._estimate_sdpa_activation_bytes(2048, 65536)
        register_attention_bias_transient(2)
        with_bias = monitor._estimate_sdpa_activation_bytes(2048, 65536)
        assert with_bias - base == 32 * 2048 * 65536 * 2
    finally:
        register_attention_bias_transient(None)
    assert monitor._estimate_sdpa_activation_bytes(2048, 65536) == base
