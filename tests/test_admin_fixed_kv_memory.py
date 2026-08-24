# SPDX-License-Identifier: Apache-2.0
"""Admin API coverage for fixed KV-cache memory estimates and launch errors."""

import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import omlx.server  # noqa: F401 - initialize admin route getters
from omlx.admin import routes as admin_routes
from omlx.exceptions import (
    InsufficientMemoryError,
    ModelBusyError,
    ModelLoadingError,
    ModelTooLargeError,
    ModelUnavailableError,
)


class _Plan:
    def __init__(self, payload):
        self.payload = payload

    def to_dict(self):
        return self.payload


def _install_fake_planner(monkeypatch, estimate):
    module = ModuleType("omlx.fixed_kv_memory")
    module.estimate_model_memory = estimate
    module.validate_fixed_kv_runtime_features = lambda settings, **kwargs: None
    monkeypatch.setitem(sys.modules, "omlx.fixed_kv_memory", module)


def _entry(*, native_context=262_144, memory=None, model_path="/models/deepseek"):
    return SimpleNamespace(
        model_path=model_path,
        estimated_size=12_000,
        model_context_length=native_context,
        engine=None,
        is_loading=False,
        memory=memory,
    )


def _patch_estimate_dependencies(
    monkeypatch,
    *,
    entry,
    model_context=None,
    global_context=32_768,
    policy=None,
    available=50_000,
    admission_ceiling=45_000,
    metal_cap=40_000,
    fixed_kv_enabled=None,
):
    pool = SimpleNamespace(
        get_entry=lambda model_id: entry if model_id == "deepseek" else None,
        _scheduler_config=SimpleNamespace(max_num_seqs=6),
    )
    manager = SimpleNamespace(
        get_settings=lambda model_id: SimpleNamespace(
            max_context_window=model_context,
            fixed_kv_cache_enabled=fixed_kv_enabled,
        )
    )
    enforcer = SimpleNamespace(
        get_admission_ceiling=lambda: admission_ceiling,
        get_ceiling_breakdown=lambda: {"metal_cap": metal_cap},
    )
    state = SimpleNamespace(process_memory_enforcer=enforcer)
    global_settings = SimpleNamespace(
        sampling=SimpleNamespace(
            max_context_window=global_context,
            max_context_window_policy=policy,
        )
    )
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(admin_routes, "_get_server_state", lambda: state)
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: global_settings)
    monkeypatch.setattr(
        admin_routes,
        "get_system_memory_info",
        lambda: {"available_bytes": available},
    )
    return pool


@pytest.mark.asyncio
async def test_memory_estimate_passes_exact_request_and_memory_snapshot(monkeypatch):
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return _Plan({"context_window": args[1], "fixed_kv_cache_bytes": 9_000})

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(monkeypatch, entry=_entry())

    result = await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=200_000, is_admin=True
    )

    assert result == {
        "context_window": 200_000,
        "fixed_kv_cache_bytes": 9_000,
        "fixed_kv_cache_enabled": True,
        "model_context_limit": 262_144,
    }
    assert calls == [
        (
            ("/models/deepseek", 200_000),
            {
                "weights_bytes": 12_000,
                "other_fixed_bytes": 0,
                "requested_session_slots": 6,
                "available_memory_bytes": 50_000,
                "memory_ceiling_bytes": 40_000,
                "prefill_step_size": 2_048,
                "native_mtp_enabled": False,
            },
        )
    ]


@pytest.mark.asyncio
async def test_disabled_fixed_kv_returns_weight_only_estimate(monkeypatch):
    def estimate(*args, **kwargs):
        raise AssertionError("disabled models must not run the fixed-cache planner")

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(),
        fixed_kv_enabled=False,
    )

    result = await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=32_768, is_admin=True
    )

    assert result["fixed_kv_cache_enabled"] is False
    assert result["lifecycle"] == "disabled"
    assert result["fixed_kv_cache_bytes"] == 0
    assert result["reserved_session_slots"] == 0
    assert result["estimated_total_bytes"] == 12_000
    assert result["projected_remaining_bytes"] == 28_000
    assert "grow during inference" in result["fit_reason"]


@pytest.mark.asyncio
async def test_memory_estimate_uses_launch_weight_and_other_fixed_charges(
    monkeypatch,
):
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return _Plan({"context_window": args[1]})

    _install_fake_planner(monkeypatch, estimate)
    pool = _patch_estimate_dependencies(monkeypatch, entry=_entry())
    pool._fixed_kv_allocation_sizes = lambda entry, settings: (10_000, 7_000)

    await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=32_768, is_admin=True
    )

    assert calls[0][1]["weights_bytes"] == 10_000
    assert calls[0][1]["other_fixed_bytes"] == 7_000


@pytest.mark.asyncio
async def test_memory_estimate_uses_same_live_budget_as_engine_admission(monkeypatch):
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return _Plan({"context_window": args[1]})

    _install_fake_planner(monkeypatch, estimate)
    pool = _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(),
        available=90_000,
        metal_cap=80_000,
    )
    pool._fixed_kv_launch_budget = lambda entry: (31_000, 27_000)

    await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=32_768, is_admin=True
    )

    assert calls[0][1]["available_memory_bytes"] == 31_000
    assert calls[0][1]["memory_ceiling_bytes"] == 27_000


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (SimpleNamespace(max_context_window=32_768, dflash_enabled=True), "DFlash"),
        (
            SimpleNamespace(
                max_context_window=32_768,
                turboquant_kv_enabled=True,
            ),
            "TurboQuant",
        ),
    ],
)
async def test_memory_estimate_rejects_launch_incompatible_modes(
    monkeypatch, settings, message
):
    _patch_estimate_dependencies(monkeypatch, entry=_entry())
    monkeypatch.setattr(
        admin_routes,
        "_get_settings_manager",
        lambda: SimpleNamespace(get_settings=lambda model_id: settings),
    )

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=32_768, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert message in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_rejects_distributed_fixed_pool(monkeypatch):
    pool = _patch_estimate_dependencies(monkeypatch, entry=_entry())
    pool._distributed_deployment_for_entry = lambda entry: object()

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=32_768, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert "distributed" in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_integrates_with_real_planner(monkeypatch, tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "num_hidden_layers": 2,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "hidden_size": 512,
                "torch_dtype": "float16",
                "max_position_embeddings": 8_192,
            }
        ),
        encoding="utf-8",
    )
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(native_context=8_192, model_path=str(tmp_path)),
        available=100_000_000,
        admission_ceiling=100_000_000,
        metal_cap=100_000_000,
    )

    result = await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=4_096, is_admin=True
    )

    assert result["context_window"] == 4_096
    assert result["weights_bytes"] == 12_000
    assert result["requested_session_slots"] == 6
    assert result["fixed_kv_cache_bytes"] > 0
    assert result["estimated_total_bytes"] > result["weights_bytes"]


@pytest.mark.asyncio
async def test_memory_estimate_uses_model_setting_before_native(monkeypatch):
    contexts = []

    def estimate(model_path, context_window, **kwargs):
        contexts.append(context_window)
        return _Plan({"context_window": context_window})

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(),
        model_context=131_072,
        policy=16_384,
    )

    result = await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=None, is_admin=True
    )

    assert result["context_window"] == 131_072
    assert contexts == [131_072]


@pytest.mark.asyncio
async def test_memory_estimate_policy_caps_native_default(monkeypatch):
    contexts = []

    def estimate(model_path, context_window, **kwargs):
        contexts.append(context_window)
        return _Plan({"context_window": context_window})

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(native_context=262_144),
        policy=65_536,
    )

    await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=None, is_admin=True
    )

    assert contexts == [65_536]


@pytest.mark.asyncio
async def test_memory_estimate_uses_global_fallback_and_unknown_available(
    monkeypatch,
):
    calls = []

    def estimate(*args, **kwargs):
        calls.append((args, kwargs))
        return _Plan({"context_window": args[1]})

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(native_context=None),
        global_context=16_384,
        available=0,
    )

    await admin_routes.get_model_memory_estimate(
        "deepseek", max_context_window=None, is_admin=True
    )

    assert calls[0][0][1] == 16_384
    assert calls[0][1]["available_memory_bytes"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("context_window", [0, -1])
async def test_memory_estimate_rejects_non_positive_context(
    monkeypatch, context_window
):
    _patch_estimate_dependencies(monkeypatch, entry=_entry())

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=context_window, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert "positive" in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_rejects_invalid_persisted_context(monkeypatch):
    _patch_estimate_dependencies(
        monkeypatch,
        entry=_entry(native_context=131_072),
        model_context=0,
    )

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=None, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert "positive" in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_rejects_context_above_native_without_clamping(
    monkeypatch,
):
    _patch_estimate_dependencies(monkeypatch, entry=_entry(native_context=131_072))

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=200_000, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert "200000" in exc_info.value.detail
    assert "131072" in exc_info.value.detail
    assert "or fewer" in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_maps_planner_validation_to_actionable_400(monkeypatch):
    def estimate(*args, **kwargs):
        raise ValueError("unsupported cache layout: mystery_mla")

    _install_fake_planner(monkeypatch, estimate)
    _patch_estimate_dependencies(monkeypatch, entry=_entry())

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=32_768, is_admin=True
        )

    assert exc_info.value.status_code == 400
    assert "unsupported cache layout" in exc_info.value.detail


@pytest.mark.asyncio
async def test_memory_estimate_refuses_unplanned_auxiliary_cache(monkeypatch):
    def estimate(*args, **kwargs):
        return _Plan({"context_window": args[1]})

    _install_fake_planner(monkeypatch, estimate)
    module = sys.modules["omlx.fixed_kv_memory"]

    def validate(settings, **kwargs):
        if settings.specprefill_enabled:
            raise ValueError("SpecPrefill draft cache is outside the fixed plan")

    module.validate_fixed_kv_runtime_features = validate
    pool = _patch_estimate_dependencies(monkeypatch, entry=_entry())
    manager = SimpleNamespace(
        get_settings=lambda model_id: SimpleNamespace(
            max_context_window=32_768,
            specprefill_enabled=True,
        )
    )
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.get_model_memory_estimate(
            "deepseek", max_context_window=32_768, is_admin=True
        )

    assert pool is not None
    assert exc_info.value.status_code == 400
    assert "SpecPrefill draft cache" in exc_info.value.detail


@pytest.mark.asyncio
async def test_admin_model_list_adds_committed_memory_when_present(monkeypatch):
    memory = _Plan({"fixed_kv_cache_bytes": 8_192, "committed": True})
    entry = _entry(memory=memory)
    status_row = {
        "id": "deepseek",
        "model_path": entry.model_path,
        "loaded": True,
        "estimated_size": entry.estimated_size,
        "pinned": False,
        "engine_type": "batched",
        "model_type": "llm",
    }
    pool = SimpleNamespace(
        get_status=lambda: {"models": [status_row]},
        get_entry=lambda model_id: entry,
        _scheduler_config=SimpleNamespace(paged_ssd_cache_dir=None),
    )
    manager = SimpleNamespace(
        get_all_settings=lambda: {},
        list_profiles=lambda model_id: [],
    )
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(admin_routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(
        admin_routes, "_get_server_state", lambda: SimpleNamespace(default_model=None)
    )
    monkeypatch.setattr(admin_routes, "_get_global_settings", lambda: None)

    result = await admin_routes.list_models(is_admin=True)

    assert result["models"][0]["memory"] == {
        "fixed_kv_cache_bytes": 8_192,
        "committed": True,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_status"),
    [
        (ModelTooLargeError("deepseek", 20_000, 10_000), 507),
        (InsufficientMemoryError(20_000, 5_000, "reduce context length"), 507),
        (ModelBusyError("deepseek", "reload"), 409),
        (ModelLoadingError("deepseek"), 409),
        (ModelUnavailableError("deepseek", "cached load failure"), 409),
    ],
)
async def test_admin_load_maps_typed_pool_errors(monkeypatch, error, expected_status):
    entry = _entry()
    pool = SimpleNamespace(
        get_entry=lambda model_id: entry,
        get_engine=AsyncMock(side_effect=error),
    )
    monkeypatch.setattr(admin_routes, "_get_engine_pool", lambda: pool)

    with pytest.raises(admin_routes.HTTPException) as exc_info:
        await admin_routes.load_model("deepseek", is_admin=True)

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == str(error)
