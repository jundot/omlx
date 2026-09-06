# SPDX-License-Identifier: Apache-2.0
"""Uno helpers, settings validation and engine selection."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from omlx.engine_pool import EnginePool
from omlx.model_discovery import discover_models
from omlx.model_settings import ModelSettings, ModelSettingsManager
from omlx.uno_bundle import resolve_uno_bundle


@pytest.fixture
def models(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "variant-Q4-Uno"
    base.mkdir()
    adapter.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            dict(
                model_type="k2_horizon",
                hidden_size=1536,
                num_hidden_layers=28,
                intermediate_size=5120,
                vocab_size=64256,
                max_position_embeddings=131072,
            )
        )
    )
    (base / "model.safetensors").write_bytes(b"fixture")
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            dict(peft_type="LORA", base_model_name_or_path="IFM/K2-Horizon-0.9B")
        )
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
    return base, adapter


def test_helper_discovery_and_settings_roundtrip(models, tmp_path):
    base, adapter = models
    found = discover_models(tmp_path)
    assert found[adapter.name].is_helper
    assert found[adapter.name].config_model_type == "k2_horizon_uno"
    assert not ModelSettings().uno_enabled
    ordinary = tmp_path / "ordinary-lora"
    ordinary.mkdir()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        (ordinary / name).write_bytes((adapter / name).read_bytes())
    assert ordinary.name not in discover_models(tmp_path)
    settings = ModelSettings(uno_enabled=True, uno_adapter_model=adapter.name)
    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(base.name, settings)
    assert (
        ModelSettingsManager(tmp_path / "settings").get_settings(base.name).uno_enabled
    )
    bundle = resolve_uno_bundle(base, adapter)
    assert bundle.base_model_id == "IFM/K2-Horizon-0.9B"
    config = json.loads((adapter / "adapter_config.json").read_text())
    config["base_model_name_or_path"] = "IFM/K2-Horizon-7B"
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="matches"):
        resolve_uno_bundle(base, adapter)


@pytest.mark.parametrize(
    "flag",
    [
        "mtp_enabled",
        "vlm_mtp_enabled",
        "dflash_enabled",
        "specprefill_enabled",
        "turboquant_kv_enabled",
        "guided_grammar_enabled",
        "thinking_budget_enabled",
    ],
)
def test_uno_rejects_conflicting_settings(flag):
    with pytest.raises(ValueError, match="Uno"):
        ModelSettings(uno_enabled=True, uno_adapter_model="adapter", **{flag: True})


@pytest.mark.asyncio
async def test_pool_selects_uno_and_charges_adapter(models, tmp_path):
    base, adapter = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    settings = ModelSettings(uno_enabled=True, uno_adapter_model=adapter.name)
    entry = pool.get_entry(base.name)
    before = pool._engine_runtime_signature(base.name, ModelSettings())
    assert before != pool._engine_runtime_signature(base.name, settings)
    assert pool._entry_runtime_resident_size(entry, settings) == (
        entry.estimated_size + pool.get_entry(adapter.name).estimated_size
    )
    with patch("omlx.engine.uno.UnoEngine") as constructor:
        constructor.return_value.start = AsyncMock()
        with patch.object(pool, "_validate_llm_engine_ready"):
            await pool._load_engine(base.name, runtime_settings=settings)
        assert constructor.call_args.kwargs["adapter_path"] == str(adapter)
    assert (
        pool._entry_runtime_resident_size(entry, settings)
        == entry.runtime_estimated_size
    )


@pytest.mark.asyncio
async def test_admin_validates_pair_before_saving(models, tmp_path, monkeypatch):
    from fastapi import HTTPException

    from omlx.admin import routes

    base, adapter = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    manager = ModelSettingsManager(tmp_path / "settings")
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    request = routes.ModelSettingsRequest(uno_enabled=True, uno_adapter_model=base.name)
    with pytest.raises(HTTPException) as error:
        await routes.update_model_settings(base.name, request, is_admin=True)
    assert error.value.status_code == 400
    assert not manager.get_settings(base.name).uno_enabled
    from omlx.exceptions import InvalidRequestError

    for effort in ("off", "xhigh"):
        with pytest.raises(InvalidRequestError, match="reasoning_effort"):
            await routes.create_model_profile(
                base.name,
                routes.CreateProfileRequest(
                    name="invalid",
                    display_name="Invalid",
                    settings={"chat_template_kwargs": {"reasoning_effort": effort}},
                ),
                is_admin=True,
            )
    assert not manager.list_profiles(base.name)

    manager.set_settings(base.name, ModelSettings(enable_thinking=False))
    request = routes.ModelSettingsRequest(
        uno_enabled=True, uno_adapter_model=adapter.name, enable_thinking=None
    )
    await routes.update_model_settings(base.name, request, is_admin=True)
    assert manager.get_settings(base.name).uno_adapter_model == adapter.name
    assert manager.get_settings(base.name).enable_thinking is None
    await routes.update_model_settings(
        base.name, routes.ModelSettingsRequest(uno_enabled=False), is_admin=True
    )
    assert not manager.get_settings(base.name).uno_enabled


@pytest.mark.asyncio
async def test_adapter_is_not_a_standalone_api_model(models, tmp_path, monkeypatch):
    import omlx.server as server
    from omlx.exceptions import ModelUnavailableError

    base, adapter = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    monkeypatch.setattr(server, "_server_state", server.ServerState(engine_pool=pool))
    assert [model.id for model in (await server.list_models(True)).data] == [base.name]
    with pytest.raises(ModelUnavailableError, match="compatible K2 base"):
        await pool._load_engine(adapter.name)


@pytest.mark.asyncio
@pytest.mark.parametrize("memory_abort", [False, True])
async def test_closing_stream_waits_for_worker_and_releases_request(
    monkeypatch, memory_abort
):
    import asyncio
    import threading
    from unittest.mock import MagicMock

    from omlx.engine.base import GenerationOutput
    from omlx.engine.uno import UnoEngine
    from omlx.exceptions import PrefillMemoryAbortedError

    engine = UnoEngine("base", adapter_path="adapter")
    engine._prefill_guard = MagicMock()
    monkeypatch.setattr(engine, "_prompt_ids", lambda prompt: [3, 4])
    monkeypatch.setattr(engine, "_preflight", lambda *args, **kwargs: None)
    ended = threading.Event()

    def run(ids, options, stops, cancelled, publish):
        try:
            publish(GenerationOutput(text="a", new_text="a", finished=False))
            cancelled.wait(5)
        finally:
            ended.set()

    monkeypatch.setattr(engine, "_run", run)
    stream = engine.stream_generate("prompt")
    await anext(stream)
    assert engine.has_active_requests()
    if memory_abort:
        assert await engine.abort_all_requests() == 1
        assert await engine.abort_all_requests() == 0
        with pytest.raises(PrefillMemoryAbortedError, match="process memory limit"):
            await asyncio.wait_for(anext(stream), timeout=2)
    else:
        await asyncio.wait_for(stream.aclose(), timeout=2)
    assert ended.is_set() and not engine._lock.locked()
    assert not engine.has_active_requests()


@pytest.mark.parametrize(
    "kwargs",
    [{"reasoning_effort": value} for value in ("off", "xhigh", "max", 1, None)]
    + [{"enable_thinking": False}],
)
def test_k2_rejects_unsupported_kwargs_before_template_fallback(kwargs):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from omlx.engine.batched import BatchedEngine
    from omlx.engine.uno import UnoEngine
    from omlx.exceptions import InvalidRequestError

    for engine in (BatchedEngine("base"), UnoEngine("base", adapter_path="adapter")):
        engine._tokenizer = MagicMock()
        engine._model = SimpleNamespace(args=SimpleNamespace(model_type="k2_horizon"))
        render = (
            engine._apply_chat_template
            if isinstance(engine, BatchedEngine)
            else engine._chat_prompt
        )
        with pytest.raises(InvalidRequestError, match="K2"):
            render([{"role": "user", "content": "Hello"}], chat_template_kwargs=kwargs)
        engine._tokenizer.apply_chat_template.assert_not_called()
