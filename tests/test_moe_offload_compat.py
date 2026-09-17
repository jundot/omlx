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
    else:
        raw["text_config"] = text
    (path / "config.json").write_text(json.dumps(raw))
    tensors = {}
    for layer in range(2):
        if kind == "olmoe":
            prefix = f"model.layers.{layer}.mlp.switch_mlp"
        elif kind == "gemma4":
            prefix = f"language_model.model.layers.{layer}.experts.switch_glu"
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
        ("gemma4", False),
        ("olmoe", False),
        ("olmoe", True),
    ],
)
def test_supported_layouts_use_headers_only(tmp_path, monkeypatch, kind, per_expert):
    _checkpoint(tmp_path, kind, per_expert)
    monkeypatch.setattr(mx, "load", lambda *a, **kw: pytest.fail("Loaded tensor data"))
    assert moe_offload_compatibility(tmp_path) == (True, "")


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


@pytest.mark.parametrize(
    "kind", ["glm5_next", "glm_moe_dsa", "deepseek_v4", "qwen3_5_moe"]
)
def test_unverified_type_is_hidden_even_with_matching_experts(tmp_path, kind):
    _checkpoint(tmp_path, kind)
    assert moe_offload_compatibility(tmp_path)[0] is False


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

    _checkpoint(tmp_path, "glm5_next")
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
