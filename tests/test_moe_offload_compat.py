# SPDX-License-Identifier: Apache-2.0
"""Offload visibility and validation share checkpoint eligibility."""

import json
from types import SimpleNamespace

import mlx.core as mx
import pytest
from fastapi import HTTPException

from omlx.patches.moe_offload_compat import moe_offload_compatibility


def _checkpoint(path, kind="qwen4_exp", per_expert=False):
    text = dict(
        num_hidden_layers=2,
        num_experts=16,
        num_experts_per_tok=2,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=64,
        enable_moe_block=True,
    )
    raw = {"model_type": kind, "quantization": {"bits": 4, "group_size": 32}}
    if kind == "olmoe":
        raw.update(text)
    elif kind == "glm_moe_dsa":
        # the flagship layout: n_routed_experts, and the first layer dense
        raw.update(text, n_routed_experts=16, first_k_dense_replace=1)
    elif kind == "deepseek_v4":
        # text-only flat config, experts under ffn.switch_mlp
        raw.update(text, n_routed_experts=16)
    elif kind == "glm5_next":
        # VLM text tower with the per-layer sparse selection
        raw["text_config"] = dict(
            text,
            n_routed_experts=16,
            first_k_dense_replace=1,
            mlp_layer_types=["dense", "sparse"],
        )
    else:
        raw["text_config"] = text
    (path / "config.json").write_text(json.dumps(raw))
    tensors = {}
    for layer in range(2):
        if kind in ("glm_moe_dsa", "glm5_next") and layer == 0:
            continue  # dense layer: no experts to cover
        if kind in ("olmoe", "glm_moe_dsa"):
            prefix = f"model.layers.{layer}.mlp.switch_mlp"
        elif kind == "gemma4":
            prefix = f"language_model.model.layers.{layer}.experts.switch_glu"
        elif kind == "deepseek_v4":
            prefix = f"model.layers.{layer}.ffn.switch_mlp"
        else:
            prefix = f"language_model.model.layers.{layer}.mlp.switch_mlp"
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for expert in range(16) if per_expert else (None,):
                key = (
                    f'{prefix.rsplit(".", 1)[0]}.experts.{expert}.{proj}'
                    if per_expert
                    else f"{prefix}.{proj}"
                )
                shape = (64, 64) if per_expert else (16, 64, 64)
                for field, value in zip(
                    ("weight", "scales", "biases"),
                    mx.quantize(mx.ones(shape), group_size=32),
                ):
                    tensors[f"{key}.{field}"] = value
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    return tensors


@pytest.mark.parametrize(
    "kind,per_expert",
    [
        ("qwen4_exp", False),
        ("qwen3_5_moe", False),
        ("gemma4", False),
        ("olmoe", False),
        ("olmoe", True),
        ("glm_moe_dsa", False),
        ("deepseek_v4", False),
        ("glm5_next", False),
    ],
)
def test_supported_layouts_use_headers_only(tmp_path, monkeypatch, kind, per_expert):
    _checkpoint(tmp_path, kind, per_expert)
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("Loaded tensor data"))
    assert moe_offload_compatibility(tmp_path) == (True, "")


def _mimo_checkpoint(path):
    """mimo_v2_flash: n_routed_experts (no num_experts), mxfp4 experts, and
    a dense layer 0 via moe_layer_freq — whose scales are deliberately
    written F16 (affine dtype) to prove the gate never inspects it."""
    text = dict(
        num_hidden_layers=2,
        n_routed_experts=16,
        num_experts_per_tok=2,
        hidden_size=64,
        moe_intermediate_size=64,
        moe_layer_freq=[0, 1],
    )
    raw = {
        "model_type": "mimo_v2_flash",
        "text_config": text,
        "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"},
    }
    (path / "config.json").write_text(json.dumps(raw))
    tensors = {}
    for layer, scales_dtype in ((0, mx.float16), (1, mx.uint8)):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.mlp.switch_mlp.{proj}"
            tensors[key + ".weight"] = mx.zeros((16, 64, 8), mx.uint32)
            tensors[key + ".scales"] = mx.zeros((16, 64, 2), scales_dtype)
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    return tensors


def test_mimo_v2_flash_supported_and_dense_layer_skipped(tmp_path, monkeypatch):
    _mimo_checkpoint(tmp_path)
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("Loaded tensor data"))
    assert moe_offload_compatibility(tmp_path) == (True, "")


def test_mimo_v2_flash_dense_layer_flag_matters(tmp_path):
    """With moe_layer_freq all-1 the gate inspects layer 0 — and its F16
    scales fail the mxfp4 U8 requirement, so eligibility must flip."""
    _mimo_checkpoint(tmp_path)
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"]["moe_layer_freq"] = [1, 1]
    path.write_text(json.dumps(raw))
    ok, reason = moe_offload_compatibility(tmp_path)
    assert ok is False
    assert "dtype" in reason or "shape" in reason


@pytest.mark.parametrize(
    "change", ["missing_expert", "wrong_shape", "wrong_dtype", "bias"]
)
def test_incompatible_checkpoint_is_hidden_and_api_rejected(tmp_path, change):
    from omlx.admin.routes import _validate_model_settings

    tensors = _checkpoint(tmp_path, "olmoe", per_expert=True)
    key = "model.layers.1.mlp.experts.15.down_proj.weight"
    if change == "missing_expert":
        del tensors[key]
    elif change == "wrong_shape":
        tensors[key] = tensors[key][:1]
    elif change == "wrong_dtype":
        tensors[key] = tensors[key].astype(mx.int32)
    else:
        tensors[key.removesuffix("weight") + "bias"] = mx.zeros((64,))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    assert moe_offload_compatibility(tmp_path)[0] is False
    with pytest.raises(HTTPException) as error:
        _validate_model_settings(
            SimpleNamespace(model_path=str(tmp_path)),
            {"moe_expert_offload_enabled": True},
        )
    assert error.value.status_code == 400


@pytest.mark.parametrize("kind", ["mixtral", "llama"])
def test_unverified_type_is_hidden_even_with_matching_experts(tmp_path, kind):
    _checkpoint(tmp_path, kind)
    assert moe_offload_compatibility(tmp_path)[0] is False


def test_deepseek_v4_requires_the_stacked_slabs(tmp_path):
    """The adapter reads ffn.switch_mlp positionally; a gap hides it."""
    tensors = _checkpoint(tmp_path, "deepseek_v4")
    assert moe_offload_compatibility(tmp_path) == (True, "")
    del tensors["model.layers.1.ffn.switch_mlp.gate_proj.weight"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    ok, reason = moe_offload_compatibility(tmp_path)
    assert ok is False and "layers.1" in reason


def test_glm5_next_dense_layers_skipped_and_routed_required(tmp_path):
    """mlp_layer_types decides which layers must carry routed experts."""
    tensors = _checkpoint(tmp_path, "glm5_next")
    assert moe_offload_compatibility(tmp_path) == (True, "")
    del tensors["language_model.model.layers.1.mlp.switch_mlp.up_proj.scales"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    ok, reason = moe_offload_compatibility(tmp_path)
    assert ok is False and "layers.1" in reason
    _checkpoint(tmp_path, "glm5_next")
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"]["mlp_layer_types"] = ["dense", "dense"]
    path.write_text(json.dumps(raw))
    assert moe_offload_compatibility(tmp_path)[0] is False


def test_glm_moe_dsa_requires_every_moe_layer(tmp_path):
    """The dense prefix is skipped; a missing routed layer still hides it."""
    tensors = _checkpoint(tmp_path, "glm_moe_dsa")
    assert moe_offload_compatibility(tmp_path) == (True, "")
    del tensors["model.layers.1.mlp.switch_mlp.up_proj.scales"]
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    ok, reason = moe_offload_compatibility(tmp_path)
    assert ok is False and "layers.1" in reason


def test_dense_gemma_is_hidden(tmp_path):
    _checkpoint(tmp_path, "gemma4")
    path = tmp_path / "config.json"
    raw = json.loads(path.read_text())
    raw["text_config"]["enable_moe_block"] = False
    path.write_text(json.dumps(raw))
    assert moe_offload_compatibility(tmp_path)[0] is False


def test_checkpoint_replacement_invalidates_eligibility(tmp_path):
    tensors = _checkpoint(tmp_path)
    assert moe_offload_compatibility(tmp_path)[0] is True
    tensors.pop(next(iter(tensors)))
    mx.save_safetensors(str(tmp_path / "model.safetensors"), tensors)
    assert moe_offload_compatibility(tmp_path)[0] is False


def test_unsupported_saved_setting_rejected_before_load(tmp_path):
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import maybe_apply_pre_load_patches

    _checkpoint(tmp_path, "mixtral")
    with pytest.raises(ValueError, match="not supported for this model type"):
        maybe_apply_pre_load_patches(
            str(tmp_path), ModelSettings(moe_expert_offload_enabled=True)
        )


@pytest.mark.parametrize("kind", ["qwen4_exp", "deepseek_v41"])
def test_admin_uses_adjusted_residency_for_offload_models(tmp_path, kind):
    import asyncio
    from dataclasses import replace
    from unittest.mock import MagicMock, patch

    from omlx.admin import routes
    from omlx.model_settings import ModelSettings
    from omlx.patches.deepseek_v41.residency import EngramResidencyEstimate
    from omlx.patches.mlx_vlm_qwen4_exp_compat.residency import (
        Qwen4ExpResidencyEstimate,
    )

    qwen = kind == "qwen4_exp"
    estimate = (
        Qwen4ExpResidencyEstimate(True, 950, 550, 1000, 400)
        if qwen
        else EngramResidencyEstimate(True, 550, 1000, 400)
    )
    adjusted = replace(estimate, resident_bytes=450, mmap_bytes=100)
    pool = MagicMock()
    pool.get_status.return_value = {
        "models": [
            {"id": "model", "model_path": str(tmp_path), "config_model_type": kind}
        ]
    }
    pool._fallback_admission_ceiling.return_value = 500
    status_method = (
        "_qwen4_ple_offload_status" if qwen else "_deepseek_v41_engram_offload_status"
    )
    getattr(pool, status_method).return_value = (False, False, adjusted)
    manager = MagicMock()
    manager.get_all_settings.return_value = {
        "model": ModelSettings(moe_expert_offload_enabled=True)
    }
    estimator = (
        "mlx_vlm_qwen4_exp_compat.residency.qwen4_exp_residency_estimate"
        if qwen
        else "deepseek_v41.residency.deepseek_v41_residency_estimate"
    )
    with (
        patch.object(routes, "_get_engine_pool", return_value=pool),
        patch.object(routes, "_get_settings_manager", return_value=manager),
        patch.object(
            routes, "_get_server_state", return_value=MagicMock(default_model=None)
        ),
        patch.object(routes, "_get_global_settings", return_value=None),
        patch.object(routes, "_dflash_compat_for_model", return_value=(False, "")),
        patch.object(routes, "_mtp_compat_for_model", return_value=(False, "")),
        patch.object(routes, "_paroquant_compat_for_model", return_value=(False, "")),
        patch("omlx.patches." + estimator, return_value=estimate),
        patch(
            "omlx.patches.moe_offload_compat.moe_offload_compatibility",
            return_value=(True, ""),
        ),
    ):
        model = asyncio.run(routes.list_models(is_admin=True))["models"][0]
    prefix = "qwen4_ple" if qwen else "deepseek_v41_engram"
    assert model["moe_expert_offload_supported"] is True
    assert model[prefix + "_ssd_offload_forced"] is False
    assert model[prefix + "_resident_bytes"] == 450


@pytest.mark.parametrize("layout", ["stacked", "per_expert"])
def test_glm_loader_paths_match_offload_admission(tmp_path, layout):
    """Through the real GLM loader, eligibility, the adapter's reads and the
    admission estimate agree for both checkpoint layouts: every MoE layer
    is wrapped, the output is unchanged, and the estimate discounts exactly
    the experts the wrapper leaves on disk."""
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.utils import load_model

    from omlx.patches.glm_moe_dsa import apply_glm_moe_dsa_patch
    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        estimate_offload_admission_bytes,
    )

    apply_glm_moe_dsa_patch()
    from mlx_lm.models import glm_moe_dsa

    config = dict(
        model_type="glm_moe_dsa",
        vocab_size=1024,
        hidden_size=128,
        index_head_dim=16,
        index_n_heads=4,
        index_topk=4,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_shared_experts=1,
        n_routed_experts=16,
        routed_scaling_factor=2.5,
        kv_lora_rank=16,
        q_lora_rank=24,
        qk_rope_head_dim=16,
        v_head_dim=32,
        qk_nope_head_dim=16,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        norm_topk_prob=True,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        moe_layer_freq=1,
        first_k_dense_replace=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_theta": 10000.0},
        attention_bias=False,
        index_topk_pattern="FSF",
    )
    model = glm_moe_dsa.Model(glm_moe_dsa.ModelArgs.from_dict(config))
    # The tiny MLA ranks are narrower than a group; the loader likewise
    # quantizes only the layers whose scales the checkpoint carries.
    nn.quantize(
        model,
        group_size=32,
        bits=4,
        class_predicate=lambda _, m: hasattr(m, "to_quantized")
        and m.weight.shape[-1] % 32 == 0,
    )
    weights = dict(tree_flatten(model.parameters()))
    expert_bytes = sum(v.nbytes for k, v in weights.items() if ".switch_mlp." in k)
    # As shipped: split gate/up projections, stacked or one tensor per expert.
    shipped = {}
    for key, value in weights.items():
        if ".switch_mlp.gate_up_proj." in key:
            gate, up = mx.split(value, 2, axis=value.ndim - 2)
            shipped[key.replace("gate_up_proj", "gate_proj")] = gate
            shipped[key.replace("gate_up_proj", "up_proj")] = up
        else:
            shipped[key] = value
    if layout == "per_expert":
        stacked = {k: v for k, v in shipped.items() if ".switch_mlp." in k}
        for key, value in stacked.items():
            del shipped[key]
            parent, rest = key.split(".switch_mlp.")
            for e in range(value.shape[0]):
                shipped[f"{parent}.experts.{e}.{rest}"] = value[e]
    config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = tmp_path / "model.safetensors"
    mx.save_safetensors(str(checkpoint), shipped)

    loaded, _ = load_model(tmp_path, lazy=True)
    inputs = mx.array([[1, 2, 3]])
    expected = loaded(inputs)
    mx.eval(expected)
    assert moe_offload_compatibility(tmp_path) == (True, "")
    # Two MoE layers behind the dense first one; 8 of 16 experts stay resident.
    assert apply_moe_expert_offload(loaded, tmp_path, 0.5) == 2
    actual = loaded(inputs)
    mx.eval(actual)
    assert mx.array_equal(expected, actual).item()
    full = checkpoint.stat().st_size
    assert (
        estimate_offload_admission_bytes(tmp_path, full, 0.5)
        == full - expert_bytes // 2
    )


@pytest.mark.parametrize("flat", [False, True])
def test_qwen35_loader_paths_match_offload_admission(tmp_path, flat):
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
    from mlx_lm.utils import load_model

    from omlx.patches.moe_expert_offload import (
        apply_moe_expert_offload,
        estimate_offload_admission_bytes,
    )

    config = dict(
        model_type="qwen3_5_moe",
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=128,
        num_experts=16,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        full_attention_interval=1,
    )
    model = Model(ModelArgs.from_dict(config))
    nn.quantize(model, group_size=32, bits=4)
    weights = dict(tree_flatten(model.parameters()))
    expert_bytes = sum(v.nbytes for k, v in weights.items() if ".switch_mlp." in k)
    if flat:
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
    config["quantization"] = {"group_size": 32, "bits": 4}
    (tmp_path / "config.json").write_text(json.dumps(config))
    checkpoint = tmp_path / "model.safetensors"
    mx.save_safetensors(str(checkpoint), weights)

    loaded, _ = load_model(tmp_path, lazy=True)
    inputs = mx.array([[1, 2, 3]])
    expected = loaded(inputs)
    mx.eval(expected)
    assert moe_offload_compatibility(tmp_path) == (True, "")
    assert apply_moe_expert_offload(loaded, tmp_path, 0.25) == 2
    actual = loaded(inputs)
    mx.eval(actual)
    assert mx.array_equal(expected, actual).item()
    # The minimum capacity keeps eight of sixteen experts per layer.
    full = checkpoint.stat().st_size
    assert (
        estimate_offload_admission_bytes(tmp_path, full, 0.25)
        == full - expert_bytes // 2
    )

    weights.pop(
        next(k for k in weights if "layers.1.mlp.switch_mlp.gate_proj.weight" in k)
    )
    mx.save_safetensors(str(checkpoint), weights)
    assert moe_offload_compatibility(tmp_path)[0] is False
    full = checkpoint.stat().st_size
    assert estimate_offload_admission_bytes(tmp_path, full, 0.25) == full
