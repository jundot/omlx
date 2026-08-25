# SPDX-License-Identifier: Apache-2.0
"""Tests for the Laguna MLX-LM monkey-patch (issue #2073).

These tests protect the upstream-first Laguna compatibility contract, including
dynamic module registration and the loader/parser boundaries it enables.
"""

import importlib
import importlib.machinery
import json
import sys

import mlx.core as mx
import pytest


def _minimal_laguna_config(**overrides):
    """Flat minimal Laguna text-model config for ModelArgs construction.

    Native (non-wrapper) config: all fields live at the top level, not
    nested under ``text_config``. Only the fields required for a fast
    CPU/MLX model construction are included.
    """
    cfg = dict(
        model_type="laguna",
        vocab_size=1024,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        qkv_bias=False,
        attention_bias=False,
        gating="per-head",
        tie_word_embeddings=False,
        rope_theta=500000.0,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        partial_rotary_factor=1.0,
        rope_style="rotate-half",
        sliding_window=None,
        layer_types=["full_attention", "full_attention"],
        num_attention_heads_per_layer=[4, 4],
        swa_rope_parameters=None,
        swa_attention_sink_enabled=False,
        num_experts=0,
        num_experts_per_tok=0,
        moe_intermediate_size=0,
        shared_expert_intermediate_size=0,
        norm_topk_prob=True,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        moe_routed_scaling_factor=1.0,
        moe_apply_router_weight_on_input=False,
        moe_router_logit_softcapping=0.0,
        moe_router_use_sigmoid=True,
    )
    cfg.update(overrides)
    return cfg


def test_apply_registers_laguna_module():
    """``apply_laguna_patch()`` makes ``mlx_lm.models.laguna`` importable."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    assert "mlx_lm.models.laguna" in sys.modules
    mod = importlib.import_module("mlx_lm.models.laguna")
    assert mod.__package__ == "mlx_lm.models"

    import mlx_lm.models as models_pkg

    assert models_pkg.laguna is mod


def test_apply_is_idempotent():
    """Calling ``apply_laguna_patch()`` twice is a no-op after the first."""
    from omlx.patches.laguna import apply_laguna_patch, is_applied

    first = apply_laguna_patch()
    second = apply_laguna_patch()

    assert is_applied() is True
    assert second is False
    assert first in (True, False)


def test_module_registration_cleans_up_after_execution_failure(monkeypatch):
    """A failed vendored import must not leave a poisoned sys.modules entry."""
    from omlx.patches import laguna

    module_name = "mlx_lm.models.laguna_broken_test"

    class FailingLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            raise RuntimeError("simulated vendored module failure")

    failing_spec = importlib.machinery.ModuleSpec(module_name, FailingLoader())
    monkeypatch.setattr(
        laguna.importlib.util,
        "spec_from_file_location",
        lambda *_: failing_spec,
    )
    sys.modules.pop(module_name, None)

    with pytest.raises(RuntimeError, match="simulated vendored module failure"):
        laguna._register_module(module_name, "not-used.py", "mlx_lm.models")

    assert module_name not in sys.modules


def test_get_classes_resolves_laguna():
    """After patching, ``_get_classes()`` resolves a Laguna config."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.utils import _get_classes

    model_cls, args_cls = _get_classes(_minimal_laguna_config())

    assert model_cls.__name__ == "Model"
    assert args_cls.__name__ == "ModelArgs"


def test_laguna_model_instantiates_with_flat_args():
    """``Model`` holds ``args``, ``model_type``, and ``model`` (native)."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(**_minimal_laguna_config())
    model = laguna.Model(args)

    assert model.args is args
    assert model.model_type == "laguna"
    assert model.model is not None
    assert hasattr(model, "layers")


def test_laguna_uses_bounded_cache_for_sliding_attention():
    """Mixed attention uses full KV only where the model can attend globally."""
    from mlx_lm.models.cache import KVCache, RotatingKVCache

    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            layer_types=["full_attention", "sliding_attention"],
            sliding_window=8,
        )
    )
    model = laguna.Model(args)

    cache = model.make_cache()

    assert type(cache[0]) is KVCache
    assert type(cache[1]) is RotatingKVCache
    assert cache[1].max_size == 8

    prefill_logits = model(mx.array([[1, 2]], dtype=mx.int32), cache=cache)
    decode_logits = model(mx.array([[3]], dtype=mx.int32), cache=cache)
    mx.eval(prefill_logits, decode_logits)

    assert prefill_logits.shape == (1, 2, 1024)
    assert decode_logits.shape == (1, 1, 1024)


def _s21_shaped_config():
    """Scaled-down Laguna S-2.1 config: per-layer lists + dual yarn RoPE."""
    return _minimal_laguna_config(
        num_hidden_layers=8,
        layer_types=[
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
        ]
        * 2,
        sliding_window=8,
        num_attention_heads_per_layer=[4, 6, 6, 6, 4, 6, 6, 6],
        mlp_layer_types=["dense"] + ["sparse"] * 7,
        gating_types=["per_head"] * 8,
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        moe_routed_scaling_factor=2.5,
        partial_rotary_factor=None,
        rope_parameters={
            "full_attention": {
                "rope_type": "yarn",
                "rope_theta": 500000.0,
                "factor": 32.0,
                "original_max_position_embeddings": 64,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "attention_factor": 1.3465735902799727,
                "partial_rotary_factor": 0.5,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
        },
    )


def test_laguna_s21_shaped_model_forward():
    """S-2.1 config surface: per-layer MLP/gating lists, variable query heads,
    yarn on full-attention layers, and mixed bounded caches."""
    import math

    from mlx_lm.models.cache import KVCache, RotatingKVCache
    from mlx_lm.models.rope_utils import YarnRoPE

    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(**_s21_shaped_config())
    model = laguna.Model(args)

    cache = model.make_cache()
    for layer_idx, layer_cache in enumerate(cache):
        if layer_idx % 4 == 0:
            assert type(layer_cache) is KVCache
        else:
            assert type(layer_cache) is RotatingKVCache
            assert layer_cache.max_size == 8

    layers = model.model.layers
    assert type(layers[0].mlp).__name__ == "MLP"
    assert all(
        type(layers[i].mlp).__name__ == "LagunaSparseMoeBlock" for i in range(1, 8)
    )
    assert layers[0].self_attn.n_heads == 4
    assert layers[1].self_attn.n_heads == 6
    assert layers[1].self_attn.gate_per_head is True

    # Full-attention layers use yarn over the rotary half of head_dim, and the
    # default mscale must equal the published attention_factor formula.
    full_rope = layers[0].self_attn.rope
    assert isinstance(full_rope, YarnRoPE)
    assert full_rope.dims == args.head_dim // 2
    assert abs(full_rope.mscale - (0.1 * math.log(32.0) + 1.0)) < 1e-9
    assert not isinstance(layers[1].self_attn.rope, YarnRoPE)

    prefill_logits = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
    decode_logits = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(prefill_logits, decode_logits)

    assert prefill_logits.shape == (1, 3, 1024)
    assert decode_logits.shape == (1, 1, 1024)


def test_mlp_layer_types_overrides_legacy_cadence():
    """An explicit mlp_layer_types list wins over mlp_only_layers cadence."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=32,
            shared_expert_intermediate_size=32,
            # Legacy cadence alone would make every layer sparse.
            mlp_only_layers=[],
            mlp_layer_types=["dense", "sparse"],
        )
    )
    model = laguna.Model(args)

    assert type(model.model.layers[0].mlp).__name__ == "MLP"
    assert type(model.model.layers[1].mlp).__name__ == "LagunaSparseMoeBlock"


def test_gating_types_normalized_per_layer():
    """gating_types entries are normalized and applied per layer."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(gating_types=["per_head", "per_element"])
    )
    model = laguna.Model(args)

    per_head_attn = model.model.layers[0].self_attn
    per_element_attn = model.model.layers[1].self_attn
    assert per_head_attn.gate_per_head is True
    assert per_head_attn.g_proj.weight.shape[0] == per_head_attn.n_heads
    assert per_element_attn.gate_per_head is False
    assert (
        per_element_attn.g_proj.weight.shape[0]
        == per_element_attn.n_heads * per_element_attn.head_dim
    )


def test_per_layer_list_length_mismatch_raises():
    """Per-layer lists that disagree with num_hidden_layers are rejected."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    with pytest.raises(ValueError, match="mlp_layer_types"):
        laguna.ModelArgs(**_minimal_laguna_config(mlp_layer_types=["dense"]))
    with pytest.raises(ValueError, match="gating_types"):
        laguna.ModelArgs(**_minimal_laguna_config(gating_types=["per_head"]))


def test_laguna_sanitize_remaps_gate_and_stacks_experts():
    """``Model.sanitize`` remaps ``mlp.gate.weight`` and stacks expert proj weights."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    # Add MoE expert weights for layer 0 to test stacking behavior
    weights = {
        "model.embed_tokens.weight": mx.zeros((1024, 64)),
        "lm_head.weight": mx.zeros((1024, 64)),
        "model.norm.weight": mx.ones((64,)),
        "model.layers.0.self_attn.q_proj.weight": mx.zeros((64, 64)),
        # Legacy gate weight (remapped to gate.proj)
        "model.layers.0.mlp.gate.weight": mx.zeros((64,)),
        # Indexed expert projection weights (stacked into switch_mlp)
        "model.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((64, 128)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((64, 128)),
    }

    out = model.sanitize(weights)

    # Normal model keys are kept as-is (no language_model. prefix)
    assert "model.embed_tokens.weight" in out
    assert "lm_head.weight" in out
    assert "model.norm.weight" in out
    assert "model.layers.0.self_attn.q_proj.weight" in out

    # Legacy gate.weight is remapped to gate.proj.weight
    assert "model.layers.0.mlp.gate.proj.weight" in out
    assert "model.layers.0.mlp.gate.weight" not in out

    # Indexed expert weights are stacked into switch_mlp.* tensors
    assert "model.layers.0.mlp.switch_mlp.gate_proj.weight" in out
    assert "model.layers.0.mlp.switch_mlp.up_proj.weight" in out
    assert "model.layers.0.mlp.switch_mlp.down_proj.weight" in out

    # Stacked tensors should have shape (num_experts, ...)
    stacked_gate = out["model.layers.0.mlp.switch_mlp.gate_proj.weight"]
    assert stacked_gate.shape == (2, 128, 64)


def test_sanitize_remaps_quant_router_sidecars():
    """``_remap_router_weights`` remaps ``gate.scales`` and ``gate.biases`` too.

    Quantized checkpoints produced by oQ / mlx-vlm carry quantization
    sidecars alongside ``gate.weight``.  Remapping only ``.weight`` leaves
    ``gate.scales`` and ``gate.biases`` orphaned, triggering
    ``ValueError: Received N parameters not in model``.
    """
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {
        "model.embed_tokens.weight": mx.zeros((1024, 64)),
        "lm_head.weight": mx.zeros((1024, 64)),
        "model.norm.weight": mx.ones((64,)),
        # Router weight + quant sidecars keyed under ``gate`` (not ``gate.proj``)
        "model.layers.0.mlp.gate.weight": mx.zeros((2, 64)),
        "model.layers.0.mlp.gate.scales": mx.zeros((2, 64)),
        "model.layers.0.mlp.gate.biases": mx.zeros((2, 64)),
        # Expert weights (pre-stacked layout)
        "model.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((64, 128)),
        "model.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((64, 128)),
    }

    out = model.sanitize(weights)

    # All three router tensors are remapped from gate.<suffix> to
    # gate.proj.<suffix>
    assert "model.layers.0.mlp.gate.proj.weight" in out
    assert "model.layers.0.mlp.gate.proj.scales" in out
    assert "model.layers.0.mlp.gate.proj.biases" in out

    # No orphaned keys remain under the old ``gate.`` prefix
    assert not any(
        k.startswith("model.layers.0.mlp.gate.") and ".proj." not in k
        for k in out
        if "e_score_correction_bias" not in k
    )


def test_sanitize_remaps_bare_score_correction_bias():
    """Bare ``mlp.e_score_correction_bias`` maps to ``gate.e_score_correction_bias``.

    The published pipenetwork/Laguna-S-2.1 MLX conversions store the
    router correction bias at ``mlp.e_score_correction_bias`` (without the
    ``experts.`` prefix the legacy sanitizer branch checked for).
    """
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {
        "model.embed_tokens.weight": mx.zeros((1024, 64)),
        "lm_head.weight": mx.zeros((1024, 64)),
        "model.norm.weight": mx.ones((64,)),
        "model.layers.0.mlp.gate.weight": mx.zeros((2, 64)),
        "model.layers.0.mlp.e_score_correction_bias": mx.zeros((2,)),
        "model.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((64, 128)),
        "model.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((64, 128)),
    }

    out = model.sanitize(weights)

    assert "model.layers.0.mlp.gate.e_score_correction_bias" in out
    assert "model.layers.0.mlp.e_score_correction_bias" not in out


def test_sanitize_remaps_experts_prefixed_score_correction_bias():
    """Legacy ``experts.e_score_correction_bias`` still maps correctly."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {
        "model.embed_tokens.weight": mx.zeros((1024, 64)),
        "lm_head.weight": mx.zeros((1024, 64)),
        "model.norm.weight": mx.ones((64,)),
        "model.layers.0.mlp.gate.weight": mx.zeros((2, 64)),
        "model.layers.0.mlp.experts.e_score_correction_bias": mx.zeros((2,)),
        "model.layers.0.mlp.experts.0.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.gate_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.1.up_proj.weight": mx.zeros((128, 64)),
        "model.layers.0.mlp.experts.0.down_proj.weight": mx.zeros((64, 128)),
        "model.layers.0.mlp.experts.1.down_proj.weight": mx.zeros((64, 128)),
    }

    out = model.sanitize(weights)

    assert "model.layers.0.mlp.gate.e_score_correction_bias" in out
    assert "model.layers.0.mlp.experts.e_score_correction_bias" not in out


def test_sanitize_dequantizes_fp8_block_weights():
    """FP8 e4m3 weight + f32 block scales convert to 8-bit affine triples."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(**_minimal_laguna_config())
    model = laguna.Model(args)

    out_dim, in_dim = 128, 256
    w_true = (
        (mx.arange(out_dim * in_dim).reshape(out_dim, in_dim) % 37) - 18
    ).astype(mx.float32) / 5.0
    scale = mx.array([[0.5, 2.0]], dtype=mx.float32)  # blocks [1, 2]
    scale_expand = mx.repeat(mx.repeat(scale, out_dim, axis=0), 128, axis=1)
    codes = mx.to_fp8(w_true / scale_expand)
    assert codes.dtype == mx.uint8

    key = "model.layers.0.mlp.shared_expert.gate_proj"
    out = model.sanitize(
        {
            f"{key}.weight": codes,
            f"{key}.weight_scale": scale,
            "model.layers.0.self_attn.q_proj.weight": mx.zeros(
                (128, 64), dtype=mx.bfloat16
            ),
            "model.layers.0.self_attn.k_scale": mx.array([1.0]),
            "model.layers.0.self_attn.v_scale": mx.array([1.0]),
        }
    )

    assert out[f"{key}.weight"].dtype == mx.uint32
    assert f"{key}.scales" in out and f"{key}.biases" in out
    assert f"{key}.weight_scale" not in out
    assert "model.layers.0.self_attn.k_scale" not in out
    assert "model.layers.0.self_attn.v_scale" not in out
    # Untouched bf16 module stays bf16
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype == mx.bfloat16

    ref = mx.from_fp8(codes, dtype=mx.float32) * scale_expand
    deq = mx.dequantize(
        out[f"{key}.weight"],
        out[f"{key}.scales"],
        out[f"{key}.biases"],
        group_size=64,
        bits=8,
    ).astype(mx.float32)
    max_err = mx.abs(deq - ref).max().item()
    assert max_err < 0.1, f"affine8 round-trip error too large: {max_err}"


def test_sanitize_stacks_and_dequantizes_fp8_experts():
    """Per-expert FP8 tensors stack first, then convert as one batched tensor."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {}
    for e in range(2):
        for proj, (o, i) in {
            "gate_proj": (128, 64),
            "up_proj": (128, 64),
            "down_proj": (64, 128),
        }.items():
            base = f"model.layers.0.mlp.experts.{e}.{proj}"
            weights[f"{base}.weight"] = mx.to_fp8(
                mx.ones((o, i), dtype=mx.float32) * (e + 1)
            )
            weights[f"{base}.weight_scale"] = mx.ones((1, 1), dtype=mx.float32)

    out = model.sanitize(weights)

    stacked = "model.layers.0.mlp.switch_mlp.gate_proj"
    assert out[f"{stacked}.weight"].dtype == mx.uint32
    assert out[f"{stacked}.weight"].shape == (2, 128, 16)  # 4 int8 per uint32
    assert out[f"{stacked}.scales"].shape == (2, 128, 1)
    assert not any(k.endswith(".weight_scale") for k in out)
    assert not any(".experts." in k for k in out)

    deq = mx.dequantize(
        out[f"{stacked}.weight"],
        out[f"{stacked}.scales"],
        out[f"{stacked}.biases"],
        group_size=64,
        bits=8,
    ).astype(mx.float32)
    assert abs(deq[0].mean().item() - 1.0) < 0.05
    assert abs(deq[1].mean().item() - 2.0) < 0.05


def test_sanitize_unpacks_int4_stacked_experts():
    """Pack-quantized int4 expert tensors unpack after stacking."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {}
    for e in range(2):
        base = f"model.layers.0.mlp.experts.{e}.gate_proj"
        weights[f"{base}.weight_packed"] = mx.full((128, 32), e + 1, dtype=mx.uint8)
        weights[f"{base}.weight_scale"] = mx.full((128, 2), 0.25, dtype=mx.float16)
        weights[f"{base}.weight_shape"] = mx.array([128, 64])

    out = model.sanitize(weights)

    stacked = "model.layers.0.mlp.switch_mlp.gate_proj"
    assert out[f"{stacked}.weight"].dtype == mx.uint32
    assert out[f"{stacked}.weight"].shape == (2, 128, 8)
    assert out[f"{stacked}.scales"].shape == (2, 128, 2)
    biases = out[f"{stacked}.biases"]
    assert mx.allclose(biases, -8 * out[f"{stacked}.scales"]).item()
    assert not any(k.endswith(".weight_shape") for k in out)
    assert not any(k.endswith(".weight_packed") for k in out)


def test_sanitize_strips_language_model_prefix():
    """VLM-tree checkpoints (language_model.*) load on the flat text tree.

    mlx-community oQ outputs of Laguna S-2.1 were produced through the
    mlx-vlm route, so every key is nested under language_model. including
    already-sanitized names like gate.proj and stacked switch_mlp triples.
    """
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    out = model.sanitize(
        {
            "language_model.lm_head.weight": mx.zeros((1024, 64)),
            "language_model.model.embed_tokens.weight": mx.zeros((1024, 64)),
            "language_model.model.norm.weight": mx.ones((64,)),
            "language_model.model.layers.0.mlp.gate.proj.weight": mx.zeros((2, 64)),
            "language_model.model.layers.0.mlp.gate.e_score_correction_bias": (
                mx.zeros((2,))
            ),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": (
                mx.zeros((2, 128, 8), dtype=mx.uint32)
            ),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.scales": (
                mx.zeros((2, 128, 1), dtype=mx.float16)
            ),
            "language_model.model.layers.0.mlp.switch_mlp.gate_proj.biases": (
                mx.zeros((2, 128, 1), dtype=mx.float16)
            ),
        }
    )

    assert "lm_head.weight" in out
    assert "model.embed_tokens.weight" in out
    assert "model.layers.0.mlp.gate.proj.weight" in out
    assert "model.layers.0.mlp.switch_mlp.gate_proj.scales" in out
    assert not any(k.startswith("language_model.") for k in out)


def test_sanitize_repacks_compressed_nvfp4_experts():
    """nvfp4-pack tensors reinterpret bit-exactly into mlx nvfp4 layout with
    the per-tensor global scale folded into the e4m3 group scales."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()

    from mlx_lm.models import laguna

    args = laguna.ModelArgs(
        **_minimal_laguna_config(
            num_experts=2,
            num_experts_per_tok=1,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=128,
        )
    )
    model = laguna.Model(args)

    weights = {}
    expected = {}
    for e in range(2):
        w_true = (
            (mx.arange(128 * 64).reshape(128, 64) % 23) - 11
        ).astype(mx.float32) / (3.0 + e)
        packed, scales = mx.quantize(w_true, group_size=16, bits=4, mode="nvfp4")
        expected[e] = (packed, scales)
        global_scale = 2.0
        base = f"model.layers.0.mlp.experts.{e}.gate_proj"
        weights[f"{base}.weight_packed"] = packed.view(mx.uint8)
        weights[f"{base}.weight_scale"] = mx.to_fp8(
            mx.from_fp8(scales, dtype=mx.float32) * global_scale
        )
        weights[f"{base}.weight_global_scale"] = mx.array(
            [global_scale], dtype=mx.float32
        )
        weights[f"{base}.input_global_scale"] = mx.array([1.0], dtype=mx.float32)

    out = model.sanitize(weights)

    stacked = "model.layers.0.mlp.switch_mlp.gate_proj"
    assert out[f"{stacked}.weight"].dtype == mx.uint32
    assert f"{stacked}.biases" not in out
    assert not any(k.endswith(".weight_global_scale") for k in out)
    assert not any(k.endswith(".input_global_scale") for k in out)
    for e in range(2):
        packed, scales = expected[e]
        assert mx.array_equal(out[f"{stacked}.weight"][e], packed).item()
        assert mx.array_equal(out[f"{stacked}.scales"][e], scales).item()


def test_normalize_laguna_compressed_quant_formats():
    """Each compressed-tensors format maps to its mlx quantization target."""
    from omlx.utils.model_loading import normalize_laguna_compressed_quant

    def cfg(fmt, weights):
        return {
            "model_type": "laguna",
            "quantization_config": {
                "quant_method": "compressed-tensors",
                "format": fmt,
                "config_groups": {"group_0": {"format": fmt, "weights": weights}},
            },
        }

    fp8 = normalize_laguna_compressed_quant(
        cfg("float-quantized", {"num_bits": 8, "type": "float"})
    )
    assert fp8["quantization"] == {"group_size": 64, "bits": 8}

    nvfp4 = normalize_laguna_compressed_quant(
        cfg("nvfp4-pack-quantized", {"num_bits": 4, "group_size": 16})
    )
    assert nvfp4["quantization"] == {"group_size": 16, "bits": 4, "mode": "nvfp4"}

    int4 = normalize_laguna_compressed_quant(
        cfg("pack-quantized", {"num_bits": 4, "group_size": 32})
    )
    assert int4["quantization"] == {"group_size": 32, "bits": 4}

    # Non-laguna and already-quantized configs are untouched
    other = {"model_type": "llama", "quantization_config": {"quant_method": "compressed-tensors"}}
    assert "quantization" not in normalize_laguna_compressed_quant(other)
    pre = cfg("pack-quantized", {})
    pre["quantization"] = {"group_size": 16, "bits": 4, "mode": "nvfp4"}
    assert normalize_laguna_compressed_quant(pre)["quantization"]["mode"] == "nvfp4"


def test_pre_load_dispatch_applies_laguna_patch(tmp_path):
    """``maybe_apply_pre_load_patches`` dispatches for ``model_type: laguna``."""
    from omlx.patches import laguna

    laguna._APPLIED = False
    sys.modules.pop("mlx_lm.models.laguna", None)
    import mlx_lm.models as models_pkg

    if hasattr(models_pkg, "laguna"):
        delattr(models_pkg, "laguna")

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "laguna"}))

    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    maybe_apply_pre_load_patches(str(tmp_path))

    assert laguna.is_applied() is True
    assert "mlx_lm.models.laguna" in sys.modules


def test_pre_load_dispatch_skips_laguna_patch_for_other_model_types(
    tmp_path, monkeypatch
):
    """A non-Laguna config must leave the compatibility patch untouched."""
    from omlx.patches import laguna
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    patch_invocations: list[None] = []
    monkeypatch.setattr(
        laguna,
        "apply_laguna_patch",
        lambda: patch_invocations.append(None) or True,
    )
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    maybe_apply_pre_load_patches(str(tmp_path))

    assert patch_invocations == []


def _laguna_tool_parser():
    """Return the parser registered by the Laguna compatibility patch."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()
    return importlib.import_module("mlx_lm.tool_parsers.laguna")


def test_apply_registers_laguna_tool_parser():
    """The compatibility patch registers Laguna's native tool parser."""
    tool_parser = _laguna_tool_parser()

    assert tool_parser.tool_call_start == "<tool_call>"
    assert tool_parser.tool_call_end == "</tool_call>"


def test_tool_parser_registration_does_not_mask_upstream_dependency_failure(
    monkeypatch,
):
    """A broken upstream parser must surface instead of being overwritten."""
    from omlx.patches import laguna

    registered_modules: list[tuple[str, str, str]] = []
    original_import_module = importlib.import_module

    def import_module_with_broken_laguna_parser(module_name: str):
        if module_name == "mlx_lm.tool_parsers.laguna":
            raise ModuleNotFoundError(
                "No module named 'missing_laguna_dependency'",
                name="missing_laguna_dependency",
            )
        return original_import_module(module_name)

    monkeypatch.setattr(
        laguna.importlib,
        "import_module",
        import_module_with_broken_laguna_parser,
    )
    monkeypatch.setattr(
        laguna,
        "_register_module",
        lambda qualname, filename, package: registered_modules.append(
            (qualname, filename, package)
        ),
    )

    with pytest.raises(ModuleNotFoundError, match="missing_laguna_dependency"):
        laguna._register_tool_parser()

    assert registered_modules == []


def test_laguna_tool_parser_parses_xml_call():
    """The parser extracts an XML-style Laguna function call."""
    tool_parser = _laguna_tool_parser()

    tool_call = (
        "<tool_call>get_weather\n"
        "<arg_key>city</arg_key>\n"
        "<arg_value>San Francisco</arg_value></tool_call>"
    )

    assert tool_parser.parse_tool_call(tool_call) == {
        "name": "get_weather",
        "arguments": {"city": "San Francisco"},
    }


def test_laguna_tool_parser_parses_json_call():
    """The parser preserves typed JSON arguments inside a Laguna tool call."""
    tool_parser = _laguna_tool_parser()

    tool_call = (
        '<tool_call>{"name":"get_weather","arguments":'
        '{"city":"Paris","days":3}}</tool_call>'
    )

    assert tool_parser.parse_tool_call(tool_call) == {
        "name": "get_weather",
        "arguments": {"city": "Paris", "days": 3},
    }


def test_laguna_tool_parser_preserves_schema_declared_string_arguments():
    """Schema-declared strings must not be coerced into JSON scalar types."""
    tool_parser = _laguna_tool_parser()

    tool_call = (
        "<tool_call>set_feature\n"
        "<arg_key>enabled</arg_key>\n"
        "<arg_value>true</arg_value></tool_call>"
    )
    tools = [
        {
            "function": {
                "name": "set_feature",
                "parameters": {"properties": {"enabled": {"type": "string"}}},
            }
        }
    ]

    assert tool_parser.parse_tool_call(tool_call, tools) == {
        "name": "set_feature",
        "arguments": {"enabled": "true"},
    }


def test_laguna_attention_resolves_sdpa_through_module():
    """The vendored model must not bind SDPA at import time (issue #2372).

    This module is imported from maybe_apply_pre_load_patches, before the engine
    installs the TurboQuant dispatcher, and the dispatcher's rebinding sweep only
    covers mlx_lm/mlx_vlm model modules, so an import-time binding here would
    never see TurboQuant at all.
    """
    from omlx.patches.laguna import laguna_model

    assert not hasattr(laguna_model, "scaled_dot_product_attention")
    code = laguna_model.Attention.__call__.__code__
    assert "mlx_lm_base" in code.co_names
    assert "scaled_dot_product_attention" in code.co_names


# --- mlxfast-challenge port: compiled fusions (Validate submission 8b4de42b) ---


def _nvfp4_sparse_config(**overrides):
    """Sparse-MoE NVFP4-shaped config exercising the fused decode path."""
    cfg = _minimal_laguna_config(
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        mlp_only_layers=[],
        mlp_layer_types=["sparse", "sparse"],
        moe_routed_scaling_factor=2.5,
    )
    cfg.update(overrides)
    return cfg


def _registered_laguna_module():
    """The exec'd model module the loader registers (patch must precede it)."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()
    import mlx_lm.models.laguna as lm

    return lm


def _quantized_sparse_model():
    """Small 2-layer sparse model with NVFP4 group-16 4-bit switch banks."""
    from omlx.patches.laguna import apply_laguna_patch

    apply_laguna_patch()
    from mlx_lm.models import laguna

    args = laguna.ModelArgs(**_nvfp4_sparse_config())
    model = laguna.Model(args)
    for layer in model.model.layers:
        sp = layer.mlp
        if type(sp).__name__ == "LagunaSparseMoeBlock":
            sw = sp.switch_mlp
            sw.gate_proj = sw.gate_proj.to_quantized(16, 4, mode="nvfp4")
            sw.up_proj = sw.up_proj.to_quantized(16, 4, mode="nvfp4")
            sw.down_proj = sw.down_proj.to_quantized(16, 4, mode="nvfp4")
    return model


def test_compiled_softplus_gate_matches_eager():
    """Compiled softplus gate is bit-identical to the eager float32 path."""
    import mlx.nn as nn

    lm = _registered_laguna_module()
    gate = mx.random.normal((1, 1, 4), dtype=mx.float32)
    out = lm._compiled_softplus_gate(gate)
    ref = nn.softplus(gate.astype(mx.float32)).astype(gate.dtype)
    mx.eval(out, ref)
    assert mx.array_equal(out, ref)


def test_compiled_swiglu_matches_eager():
    """Compiled SiLU product is bit-identical to mlx_lm's swiglu."""
    from mlx_lm.models.activations import swiglu

    lm = _registered_laguna_module()
    gate = mx.random.normal((1, 1, 8, 32), dtype=mx.float32)
    up = mx.random.normal((1, 1, 8, 32), dtype=mx.float32)
    out = lm._swiglu(gate, up)
    ref = swiglu(gate, up)
    mx.eval(out, ref)
    assert mx.array_equal(out, ref)


def test_compiled_fusions_bit_exact(monkeypatch):
    """Compiled fusions reproduce eager output exactly on one model instance."""
    lm = _registered_laguna_module()
    model = _quantized_sparse_model()

    def run(compiled):
        monkeypatch.setattr(lm, "_COMPILED_FUSIONS", compiled)
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
        mx.eval(prefill, decode)
        return prefill, decode

    pre_on, dec_on = run(True)
    pre_off, dec_off = run(False)
    assert mx.array_equal(pre_on, pre_off)
    assert mx.array_equal(dec_on, dec_off)
    assert int(mx.max(mx.abs(dec_on - dec_off)).item()) == 0


# --- mlxfast-challenge port: fused gate/up banks (Validate submission 613aaf69) ---


def test_fused_routed_gate_up_parity_is_bit_exact(monkeypatch):
    """Fused [gate; up] decode bank must be bit-identical to the stock path.

    Toggles the actual registered model module (``mlx_lm.models.laguna``): the
    loader exec's ``laguna_model.py`` into that module and the model reads its
    ``_FUSED_ROUTED_GATE_UP`` global from there.
    """
    lm = _registered_laguna_module()
    model = _quantized_sparse_model()

    def run(fusion_on):
        monkeypatch.setattr(lm, "_FUSED_ROUTED_GATE_UP", fusion_on)
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
        mx.eval(prefill, decode)
        return prefill, decode

    pre_on, dec_on = run(True)
    pre_off, dec_off = run(False)

    block = model.model.layers[0].mlp
    assert block._fusion_ready is True
    assert block._fused_gateup_split == 32
    assert block._fused_gateup_weight.shape == (4, 64, 8)

    assert mx.array_equal(pre_on, pre_off)
    assert mx.array_equal(dec_on, dec_off)


def test_fused_shared_gate_up_parity_is_bit_exact(monkeypatch):
    """Fused shared-expert [gate; up] NVFP4 bank must be bit-identical."""
    lm = _registered_laguna_module()
    model = _quantized_sparse_model()
    for layer in model.model.layers:
        sp = layer.mlp
        if type(sp).__name__ == "LagunaSparseMoeBlock":
            se = sp.shared_expert
            se.gate_proj = se.gate_proj.to_quantized(16, 4, mode="nvfp4")
            se.up_proj = se.up_proj.to_quantized(16, 4, mode="nvfp4")
            se.down_proj = se.down_proj.to_quantized(16, 4, mode="nvfp4")

    def run(fused):
        monkeypatch.setattr(lm, "_FUSED_SHARED_GATE_UP", fused)
        cache = model.make_cache()
        prefill = model(mx.array([[1, 2, 3]], dtype=mx.int32), cache=cache)
        decode = model(mx.array([[4]], dtype=mx.int32), cache=cache)
        mx.eval(prefill, decode)
        return prefill, decode

    pre_on, dec_on = run(True)
    pre_off, dec_off = run(False)
    se = model.model.layers[0].mlp.shared_expert
    assert se._fusion_ready is True
    assert mx.array_equal(pre_on, pre_off)
    assert mx.array_equal(dec_on, dec_off)


def test_fused_banks_default_off_and_guard_unquantized(monkeypatch):
    """Fusion defaults OFF (neutral on current MLX); unquantized banks refuse."""
    lm = _registered_laguna_module()
    assert lm._FUSED_ROUTED_GATE_UP is False
    assert lm._FUSED_SHARED_GATE_UP is False
    monkeypatch.setattr(lm, "_FUSED_ROUTED_GATE_UP", True)
    monkeypatch.setattr(lm, "_FUSED_SHARED_GATE_UP", True)
    model = _quantized_sparse_model()
    for layer in model.model.layers:
        sp = layer.mlp
        if type(sp).__name__ == "LagunaSparseMoeBlock":
            from mlx_lm.models.switch_layers import SwitchLinear

            sp.switch_mlp.gate_proj = SwitchLinear(64, 32, 4)
            sp.switch_mlp.up_proj = SwitchLinear(64, 32, 4)
            sp.switch_mlp.down_proj = SwitchLinear(32, 64, 4)
    cache = model.make_cache()
    out = model(mx.array([[4]], dtype=mx.int32), cache=cache)
    mx.eval(out)
    assert model.model.layers[0].mlp._fusion_ready is False


def test_two_output_compiled_tail_is_numerically_close():
    """C1 marker: two-output mx.compile parity depends on the Apple GPU.

    The Swift challenge compiles the router tail
    ``(sigmoid(logits), -(sigmoid(logits)+bias))`` into one kernel (challenge
    commit f8848e0 / submission 8adb56be). In Python MLX 0.32.0 that two-output
    compiled function is bit-exact on the macos-14-arm64 CI runner but is
    deterministically ULP-divergent on an M3 Ultra. The router tail therefore
    stays eager in the port because it feeds argpartition expert selection.
    A zero difference on one GPU does not resolve C1 for every supported Mac.
    """
    key = mx.random.normal((2, 256), dtype=mx.float32)
    bias = mx.random.normal((256,), dtype=mx.float32)

    def tail(a, b):
        s = mx.sigmoid(a)
        return s, -(s + b.astype(s.dtype))

    compiled = mx.compile(tail, shapeless=True)
    scores, neg = compiled(key, bias)
    ref_scores, ref_neg = tail(key, bias)
    mx.eval(scores, neg, ref_scores, ref_neg)
    sig_diff = float(mx.max(mx.abs(scores - ref_scores)).item())
    neg_diff = float(mx.max(mx.abs(neg - ref_neg)).item())
    assert sig_diff <= 1e-4, f"compiled score tail changed: max-abs {sig_diff}"
    assert neg_diff <= 1e-4, f"compiled corrected tail changed: max-abs {neg_diff}"


def test_compiled_combine_matches_eager():
    """The compiled weighted-expert combine reproduces the eager reduction."""
    lm = _registered_laguna_module()
    combine = lm._compiled_combine_for(2.5)
    y = mx.random.normal((1, 1, 2, 64), dtype=mx.float32)
    weights = mx.random.uniform(shape=(1, 1, 2), dtype=mx.float32)
    shared = mx.random.normal((1, 1, 64), dtype=mx.float32)
    out = combine(y, weights, shared)
    ref = mx.sum(y * weights[..., None], axis=-2) * 2.5 + shared
    mx.eval(out, ref)
    assert mx.array_equal(out, ref)


def test_normalize_then_combine_equals_folded():
    """eb76e2b8 equivalence gate: router-side normalize + combine is bit-identical
    to the Swift's folded lagunaCompiledNormalizedExpertCombine.

    The submission folds top-k renormalization into the expert combine
    (deferred). oMLX keeps the normalize in the router (8adb56be) and the
    combine separate (9a37e4dc); this pins that the two compositions are
    bit-identical, so the folded variant adds nothing and is not re-ported.
    """
    scale = 2.5
    outputs = mx.random.normal((1, 1, 2, 64), dtype=mx.float32)
    weights = mx.random.uniform(shape=(1, 1, 2), dtype=mx.float32)
    shared = mx.random.normal((1, 1, 64), dtype=mx.float32)

    # oMLX path: normalize in the router, then the compiled combine.
    normalized = weights / mx.sum(weights, axis=-1, keepdims=True)
    typed = normalized.astype(outputs.dtype)
    routed = mx.sum(outputs * typed[..., None], axis=-2)
    separate = routed * scale + shared

    # Swift folded path (lagunaCompiledNormalizedExpertCombine body).
    folded = (
        mx.sum(outputs * (weights / mx.sum(weights, axis=-1, keepdims=True)).astype(
            outputs.dtype
        )[..., None], axis=-2)
        * scale
        + shared
    )
    mx.eval(separate, folded)
    assert mx.array_equal(separate, folded)


def test_compiled_combine_residual_matches_eager():
    """The compiled residual combine reproduces the eager h + moe bit-exactly."""
    lm = _registered_laguna_module()
    combine = lm._compiled_combine_residual_for(2.5)
    y = mx.random.normal((1, 1, 2, 64), dtype=mx.float32)
    weights = mx.random.uniform(shape=(1, 1, 2), dtype=mx.float32)
    shared = mx.random.normal((1, 1, 64), dtype=mx.float32)
    residual = mx.random.normal((1, 1, 64), dtype=mx.float32)
    out = combine(y, weights, shared, residual)
    moe = mx.sum(y * weights[..., None], axis=-2) * 2.5 + shared
    ref = residual + moe
    mx.eval(out, ref)
    assert mx.array_equal(out, ref)
