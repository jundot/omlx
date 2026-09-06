# SPDX-License-Identifier: Apache-2.0
"""CPU-only contract tests for the optional DS4 kernel-provider seam."""

from __future__ import annotations

from importlib import resources

import pytest

from omlx.custom_kernels import NATIVE_KERNEL_PACKAGES
from omlx.custom_kernels.ds4 import (
    DS4_FLASH_FINGERPRINT,
    DS4CheckpointLayout,
    DS4ExecutionMode,
    DS4KernelRequest,
    DS4ProviderCapability,
    DS4ProviderDecisionCode,
    DS4ProviderSupport,
    dispatch_ds4_kernel,
    ds4_flash_fingerprint_mismatches,
    is_exact_ds4_flash_config,
    resolve_ds4_provider,
)


def _official_config(**changes):
    # Independent, literal fixture from the converted 0731 MXFP4 MLX config.
    # Do not derive this from DS4_FLASH_FINGERPRINT: the test must catch
    # omitted architecture discriminators as well as incorrect expected values.
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129_280,
        "hidden_size": 4_096,
        "moe_intermediate_size": 2_048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "qk_rope_head_dim": 64,
        "q_lora_rank": 1_024,
        "o_lora_rank": 1_024,
        "n_shared_experts": 1,
        "n_routed_experts": 256,
        "num_experts_per_tok": 6,
        "num_hash_layers": 3,
        "num_nextn_predict_layers": 1,
        "hc_mult": 4,
        "hc_eps": 1e-6,
        "hc_sinkhorn_iters": 20,
        "o_groups": 8,
        "sliding_window": 128,
        "max_position_embeddings": 1_048_576,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "swiglu_limit": 10.0,
        "routed_scaling_factor": 1.5,
        "scoring_func": "sqrtsoftplus",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "compress_ratios": [
            0,
            0,
            *[value for _ in range(20) for value in (4, 128)],
            4,
            0,
            0,
            0,
        ],
        "dspark_block_size": 5,
        "dspark_noise_token_id": 128_799,
        "dspark_target_layer_ids": [40, 41, 42],
        "dspark_markov_rank": 256,
    }
    config.update(changes)
    return config


def _request(**changes):
    values = {
        "operation": "mxfp4_pair_swiglu_prefill",
        "model_config": _official_config(),
        "checkpoint_layout": DS4CheckpointLayout.MLX_SAFETENSORS,
        "execution_mode": DS4ExecutionMode.LOCAL,
        "experimental_enabled": True,
        "metadata": {"routes": 6144},
    }
    values.update(changes)
    return DS4KernelRequest(**values)


class _Provider:
    name = "test-ds4-provider"

    def __init__(self, support=True, detail=""):
        self.support = support
        self.detail = detail
        self.calls = 0
        self.requests = []

    def probe(self, request):
        self.calls += 1
        self.requests.append(request)
        return DS4ProviderSupport(self.support, self.detail)


def test_flash_fingerprint_is_strict_but_accepts_json_numeric_spelling():
    official = _official_config()
    assert tuple(official) == tuple(key for key, _value in DS4_FLASH_FINGERPRINT)
    assert is_exact_ds4_flash_config(official)
    assert is_exact_ds4_flash_config(_official_config(swiglu_limit=10))

    mismatches = ds4_flash_fingerprint_mismatches(
        _official_config(n_shared_experts=True, hidden_size=8192)
    )
    assert mismatches == ("hidden_size", "n_shared_experts")


def test_flash_fingerprint_reports_missing_fields_in_contract_order():
    config = _official_config()
    del config["model_type"]
    del config["index_topk"]

    assert ds4_flash_fingerprint_mismatches(config) == (
        "model_type",
        "index_topk",
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("operation", "", ValueError),
        ("checkpoint_layout", "mlx_safetensors", TypeError),
        ("execution_mode", "local", TypeError),
        ("experimental_enabled", 1, TypeError),
        ("metadata", (), TypeError),
    ],
)
def test_request_rejects_ambiguous_contract_values(field, value, error):
    with pytest.raises(error):
        _request(**{field: value})


def test_experiment_is_default_off_and_does_not_probe_provider():
    provider = _Provider()
    request = _request(experimental_enabled=False)

    capability = resolve_ds4_provider(provider, request)

    assert capability.code is DS4ProviderDecisionCode.EXPERIMENT_DISABLED
    assert capability.selected is False
    assert provider.calls == 0


def test_common_model_and_checkpoint_gates_precede_provider_probe():
    provider = _Provider()

    model_capability = resolve_ds4_provider(
        provider,
        _request(model_config=_official_config(model_type="deepseek_v4_pro")),
    )
    gguf_capability = resolve_ds4_provider(
        provider,
        _request(checkpoint_layout=DS4CheckpointLayout.DS4_GGUF),
    )

    assert model_capability.code is DS4ProviderDecisionCode.MODEL_MISMATCH
    assert model_capability.detail == "model_type"
    assert gguf_capability.code is (
        DS4ProviderDecisionCode.CHECKPOINT_LAYOUT_UNSUPPORTED
    )
    assert gguf_capability.detail == "ds4_gguf"
    assert provider.calls == 0


def test_missing_or_rejecting_provider_selects_fallback_once():
    fallback_calls = 0
    provider_calls = 0

    def fallback():
        nonlocal fallback_calls
        fallback_calls += 1
        return "mlx"

    def provider_call():
        nonlocal provider_calls
        provider_calls += 1
        return "native"

    missing = dispatch_ds4_kernel(
        None,
        _request(),
        provider_call=provider_call,
        fallback=fallback,
    )
    rejected_provider = _Provider(False, "shape unsupported")
    rejected = dispatch_ds4_kernel(
        rejected_provider,
        _request(),
        provider_call=provider_call,
        fallback=fallback,
    )

    assert missing.value == rejected.value == "mlx"
    assert missing.capability.code is DS4ProviderDecisionCode.PROVIDER_UNAVAILABLE
    assert rejected.capability.code is DS4ProviderDecisionCode.PROVIDER_REJECTED
    assert rejected.capability.detail == "shape unsupported"
    assert fallback_calls == 2
    assert provider_calls == 0
    assert rejected_provider.calls == 1


@pytest.mark.parametrize(
    "bad_probe",
    [
        lambda _request: (_ for _ in ()).throw(RuntimeError("probe failed")),
        lambda _request: True,
    ],
    ids=("raises", "invalid-result"),
)
def test_broken_optional_probe_fails_soft_to_reference(bad_probe):
    provider = _Provider()
    provider.probe = bad_probe

    result = dispatch_ds4_kernel(
        provider,
        _request(),
        provider_call=lambda: "native",
        fallback=lambda: "mlx",
    )

    assert result.value == "mlx"
    assert result.capability.code is DS4ProviderDecisionCode.PROVIDER_PROBE_ERROR
    assert result.used_provider is False


@pytest.mark.parametrize(
    "execution_mode",
    (DS4ExecutionMode.LOCAL, DS4ExecutionMode.JACCL_TENSOR_PARALLEL),
)
def test_supported_provider_is_selected_for_explicit_execution_mode(execution_mode):
    provider = _Provider(True, "exact shape")
    fallback_calls = 0

    def fallback():
        nonlocal fallback_calls
        fallback_calls += 1
        return "mlx"

    result = dispatch_ds4_kernel(
        provider,
        _request(execution_mode=execution_mode),
        provider_call=lambda: "native",
        fallback=fallback,
    )

    assert result.value == "native"
    assert result.used_provider is True
    assert result.capability.code is DS4ProviderDecisionCode.SELECTED
    assert result.capability.provider_name == provider.name
    assert provider.requests[0].execution_mode is execution_mode
    assert fallback_calls == 0


def test_provider_call_failure_propagates_without_unsafe_retry():
    provider = _Provider()
    fallback_calls = 0

    def provider_call():
        raise RuntimeError("Metal submission failed")

    def fallback():
        nonlocal fallback_calls
        fallback_calls += 1
        return "mlx"

    with pytest.raises(RuntimeError, match="Metal submission failed"):
        dispatch_ds4_kernel(
            provider,
            _request(),
            provider_call=provider_call,
            fallback=fallback,
        )

    assert provider.calls == 1
    assert fallback_calls == 0


def test_capability_rejects_inconsistent_selected_state():
    with pytest.raises(ValueError, match="must agree"):
        DS4ProviderCapability(
            selected=True,
            code=DS4ProviderDecisionCode.PROVIDER_REJECTED,
            operation="test",
        )


def test_provenance_resources_ship_with_the_provider_contract():
    package = resources.files("omlx.custom_kernels.ds4")
    license_text = package.joinpath("LICENSE.ds4-metal").read_text()
    vendor_text = package.joinpath("VENDOR.md").read_text()
    normalized_vendor = " ".join(vendor_text.split())

    assert "MIT License" in license_text
    assert "Copyright (c) 2026 The ds4.c authors" in license_text
    assert "Copyright (c) 2023-2026 The ggml authors" in license_text
    assert "78269ce7ca0f8fd4deff15b803ea4bc87fc6b99e" in vendor_text
    assert "kernel_mul_mm_id_mxfp4_pair_swiglu_f16" in vendor_text
    assert "dsa_indexer_nax.metal" in vendor_text
    assert "kernel_dsv4_indexer_scores_nax" in vendor_text
    assert "No DwarfStar Metal source is copied" in normalized_vendor


def test_seam_is_not_registered_as_a_production_native_package():
    assert "ds4" not in NATIVE_KERNEL_PACKAGES
