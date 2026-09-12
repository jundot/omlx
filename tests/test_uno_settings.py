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
async def test_admin_supplies_uno_adapters_constraints_and_thinking_modes(
    models, tmp_path, monkeypatch
):
    from omlx.admin import routes
    from omlx.model_settings import UNO_REQUIRED_SETTINGS

    base, adapter = models
    other = tmp_path / "other-Uno"
    other.mkdir()
    (other / "adapter_config.json").write_text(
        json.dumps(
            {"peft_type": "LORA", "base_model_name_or_path": "IFM/K2-Horizon-7B"}
        )
    )
    (other / "adapter_model.safetensors").write_bytes(b"fixture")
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    manager = ModelSettingsManager(tmp_path / "settings")
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(routes, "_get_global_settings", lambda: None)
    for enabled in (False, True, False):
        manager.set_settings(
            base.name,
            ModelSettings(uno_enabled=enabled, uno_adapter_model=adapter.name),
        )
        response = (await routes.list_models(is_admin=True))["models"]
        model = next(item for item in response if item["id"] == base.name)
        assert model["uno_compatible"] is True
        assert model["uno_adapters"] == [adapter.name]
        assert model["uno_required_settings"] == UNO_REQUIRED_SETTINGS
        assert model["thinking_modes"] == (
            ["auto"] if enabled else ["auto", "on_limit"]
        )
        for helper in (item for item in response if item["is_helper"]):
            assert helper["uno_compatible"] is False
            assert helper["uno_adapters"] == []


def test_dashboard_uno_selection_inheritance_and_repair():
    import re
    import shutil
    import subprocess
    from pathlib import Path

    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required to exercise dashboard JavaScript")
    source = (
        Path(__file__).parents[1] / "omlx/admin/static/js/dashboard.js"
    ).read_text()
    methods = "\n".join(
        re.search(
            r"^            " + name + r"\([^\n]*\) \{.*?^            \},",
            source,
            re.M | re.S,
        ).group()
        for name in (
            "unoAdapterCandidates",
            "get unoSelection",
            "set unoSelection",
            "unoConflicts",
            "applyUnoSettings",
            "unoLocks",
            "thinkingBudgetAvailable",
        )
    )
    template = (
        Path(__file__).parents[1]
        / "omlx/admin/templates/dashboard/_modal_model_settings.html"
    ).read_text()
    save_guard = re.search(
        r':disabled="(savingModelSettings[^"\n]*unoConflicts[^"\n]*)"', template
    ).group(1)
    script = (
        (
            "const app = {"
            + methods
            + "};const saveDisabled = new Function('app',"
            + json.dumps("with (app) { return (" + save_guard + "); }")
            + ");"
        )
        + """
app.selectedModel = {uno_adapters: ['variant', 'variant2'], uno_required_settings:
  {repetition_penalty: 1, min_p: 0, thinking_budget_enabled: 0, guided_grammar_enabled: 0, turboquant_kv_enabled: 0}};
app.models = [{id: 'variant'}, {id: 'variant2'}, {id: 'unlisted'}];
app.globalSettings = {sampling: {repetition_penalty: 1.1}};
app.modelSettings = {repetition_penalty: '', min_p: '', enableThinkingBudget: false, guided_grammar_enabled: true, uno_enabled: false};
app.unoSelection = 'variant2';
app.savingModelSettings = false;
const blocked = saveDisabled(app) === true;
const conflicts = app.unoConflicts();
const canDisableGrammar = !app.unoLocks('guided_grammar_enabled');
app.applyUnoSettings();
const repaired = app.unoConflicts();
const saveable = saveDisabled(app) === false;
const locks = app.unoLocks('turboquant_kv_enabled') && !app.thinkingBudgetAvailable();
const selection = app.unoSelection;
app.unoSelection = '';
console.log(JSON.stringify({blocked, saveable, conflicts, canDisableGrammar, repaired, locks, selection,
  off: !app.modelSettings.uno_enabled && app.thinkingBudgetAvailable() && !app.unoLocks('turboquant_kv_enabled'),
  repairedPenalty: app.modelSettings.repetition_penalty, global: app.globalSettings.sampling.repetition_penalty, adapters: app.unoAdapterCandidates()}));
"""
    )
    result = json.loads(subprocess.check_output([node, "-e", script], text=True))
    assert result == {
        "conflicts": [
            {
                "key": "repetition_penalty",
                "neutral": 1,
                "value": 1.1,
                "inherited": True,
            },
            {
                "key": "guided_grammar_enabled",
                "neutral": 0,
                "value": True,
                "inherited": False,
            },
        ],
        "blocked": True,
        "saveable": True,
        "canDisableGrammar": True,
        "repaired": [],
        "locks": True,
        "selection": "variant2",
        "off": True,
        "repairedPenalty": 1,
        "global": 1.1,
        "adapters": [{"id": "variant"}, {"id": "variant2"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("ane_enabled", [False, True])
async def test_pool_selects_uno_and_charges_adapter(models, tmp_path, ane_enabled):
    base, adapter = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    settings = ModelSettings(
        uno_enabled=True,
        uno_adapter_model=adapter.name,
        qwen35_ane_prefill_enabled=ane_enabled,
    )
    entry = pool.get_entry(base.name)
    before = pool._engine_runtime_signature(base.name, ModelSettings())
    assert before != pool._engine_runtime_signature(base.name, settings)
    weight_size = entry.estimated_size + pool.get_entry(adapter.name).estimated_size
    assert (
        pool._entry_runtime_resident_size(
            entry, settings, include_ane_reservation=False
        )
        == weight_size
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
    assert entry.runtime_settle_size == weight_size
    assert (entry.runtime_estimated_size > weight_size) == ane_enabled


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
async def test_uno_stream_cleanup_uses_shared_engine(memory_abort):
    from types import SimpleNamespace

    from omlx.engine.uno import UnoEngine
    from omlx.exceptions import PrefillMemoryAbortedError
    from omlx.request import RequestOutput

    collectors = {"request": object()}

    async def outputs(_):
        yield RequestOutput(request_id="request", new_text="a", output_text="a")
        raise PrefillMemoryAbortedError("process memory limit")

    async def abort(_):
        collectors.clear()

    engine = UnoEngine("base", adapter_path="adapter")
    engine._loaded = True
    engine._tokenizer = SimpleNamespace(encode=lambda _: [3, 4])
    engine._bundle = SimpleNamespace(config={"vocab_size": 128}, context_length=1024)
    engine._engine = SimpleNamespace(
        add_request=AsyncMock(return_value="request"),
        stream_outputs=outputs,
        abort_request=AsyncMock(side_effect=abort),
        engine=SimpleNamespace(_output_collectors=collectors),
    )
    stream = engine.stream_generate("prompt")
    await anext(stream)
    assert engine.has_active_requests()
    if memory_abort:
        with pytest.raises(PrefillMemoryAbortedError, match="process memory limit"):
            await anext(stream)
    else:
        await stream.aclose()
    engine._engine.abort_request.assert_awaited_once_with("request")
    assert not engine.has_active_requests()


@pytest.mark.asyncio
@pytest.mark.parametrize("uno", [False, True])
@pytest.mark.parametrize(
    "entry", ["generate", "stream_generate", "preflight_chat", "preflight_completion"]
)
async def test_uno_admission_hooks_preserve_ordinary_requests(uno, entry):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from omlx.engine.batched import BatchedEngine
    from omlx.engine.uno import UnoEngine
    from omlx.exceptions import InvalidRequestError

    class AdmittedError(Exception):
        pass

    engine = UnoEngine("base", adapter_path="adapter") if uno else BatchedEngine("base")
    engine._loaded = True
    engine._tokenizer = SimpleNamespace(encode=lambda _: [3, 4])
    engine._bundle = SimpleNamespace(config={"vocab_size": 128}, context_length=1024)
    engine._apply_chat_template = Mock(return_value="prompt")
    admitted = AsyncMock(side_effect=AdmittedError)
    engine._engine = SimpleNamespace(
        generate=admitted,
        add_request=admitted,
        engine=SimpleNamespace(scheduler=object()),
    )
    engine._preflight_or_raise_with_eviction = admitted
    prompt = (
        [{"role": "user", "content": "prompt"}]
        if entry == "preflight_chat"
        else "prompt"
    )
    with pytest.raises(InvalidRequestError if uno else AdmittedError):
        result = getattr(engine, entry)(prompt, min_p=0.1)
        if entry == "stream_generate":
            await anext(result)
        else:
            await result
    assert admitted.await_count == (0 if uno else 1)


@pytest.mark.asyncio
async def test_uno_preparation_uses_loader_executor_before_transforms(monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from omlx.engine.uno import UnoEngine

    class SetupCheckedError(Exception):
        pass

    order = []
    model, tokenizer = object(), object()

    def load(*args, **kwargs):
        order.append(("load", threading.get_ident()))
        return model, tokenizer

    def prepare():
        assert engine._model is model and engine._tokenizer is tokenizer
        order.append(("prepare", threading.get_ident()))

    def transforms(*args):
        order.append(("transforms", threading.get_ident()))
        raise SetupCheckedError

    engine = UnoEngine("base", adapter_path="adapter")
    monkeypatch.setattr(engine, "_prepare_uno_model", prepare)
    monkeypatch.setattr("omlx.engine.batched.get_tokenizer_config", lambda *a, **k: {})
    monkeypatch.setattr(
        "omlx.utils.model_loading.maybe_apply_pre_load_patches", lambda *a, **k: None
    )
    monkeypatch.setattr("omlx.utils.model_loading.maybe_load_custom_quantization", load)
    monkeypatch.setattr(
        "omlx.utils.model_loading.apply_post_load_transforms", transforms
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        monkeypatch.setattr("omlx.engine_core.get_mlx_executor", lambda: executor)
        with pytest.raises(SetupCheckedError):
            await engine.start()
    assert [name for name, _ in order] == ["load", "prepare", "transforms"]
    assert order[0][1] == order[1][1] != threading.get_ident()
    assert order[2][1] == threading.get_ident()


@pytest.mark.parametrize(
    "kwargs",
    [{"reasoning_effort": value} for value in ("off", "xhigh", "max", 1, None)]
    + [{"enable_thinking": False}],
)
def test_uno_rejects_unsupported_kwargs_before_template_fallback(kwargs):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from omlx.engine.uno import UnoEngine
    from omlx.exceptions import InvalidRequestError

    for engine in (UnoEngine("base", adapter_path="adapter"),):
        engine._tokenizer = MagicMock()
        engine._model = SimpleNamespace(args=SimpleNamespace(model_type="k2_horizon"))
        render = engine._apply_chat_template
        with pytest.raises(InvalidRequestError, match="K2"):
            render([{"role": "user", "content": "Hello"}], chat_template_kwargs=kwargs)
        engine._tokenizer.apply_chat_template.assert_not_called()


def test_uno_ane_setting_roundtrip_and_reservation(models, tmp_path):
    from omlx.patches.k2_horizon.ane_prefill import prefill_memory_reservation

    base, adapter = models
    assert ModelSettings(qwen35_ane_prefill_enabled=True).qwen35_ane_prefill_enabled
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    ordinary = ModelSettings(uno_enabled=True, uno_adapter_model=adapter.name)
    ane = ModelSettings(
        uno_enabled=True,
        uno_adapter_model=adapter.name,
        qwen35_ane_prefill_enabled=True,
    )
    manager = ModelSettingsManager(tmp_path / "settings")
    manager.set_settings(base.name, ane)
    assert (
        ModelSettingsManager(tmp_path / "settings")
        .get_settings(base.name)
        .qwen35_ane_prefill_enabled
    )
    assert pool._engine_runtime_signature(
        base.name, ordinary
    ) != pool._engine_runtime_signature(base.name, ane)
    entry = pool.get_entry(base.name)
    config = json.loads((base / "config.json").read_text())
    assert pool._entry_runtime_resident_size(
        entry, ane
    ) - pool._entry_runtime_resident_size(
        entry, ordinary
    ) == prefill_memory_reservation(
        config
    )


def test_future_dense_uno_identity_is_not_a_size_whitelist(models):
    from omlx.uno_bundle import uno_base_id

    base, adapter = models
    config = json.loads((base / "config.json").read_text())
    config.update(
        _name_or_path="IFM/K2-Horizon-32B",
        hidden_size=5120,
        intermediate_size=26624,
        num_hidden_layers=64,
        max_position_embeddings=524288,
    )
    (base / "config.json").write_text(json.dumps(config))
    cfg = json.loads((adapter / "adapter_config.json").read_text())
    cfg["base_model_name_or_path"] = "IFM/K2-Horizon-32B"
    (adapter / "adapter_config.json").write_text(json.dumps(cfg))
    assert uno_base_id(base) == uno_base_id(adapter) == "IFM/K2-Horizon-32B"
    assert resolve_uno_bundle(base, adapter).context_length == 524288
    config["mova_num_experts"] = 64
    (base / "config.json").write_text(json.dumps(config))
    assert uno_base_id(base) is None


@pytest.mark.parametrize("uno", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "endpoint", ["/v1/chat/completions", "/v1/responses", "/v1/messages"]
)
def test_real_route_reaches_memory_preflight(monkeypatch, uno, stream, endpoint):
    from types import SimpleNamespace
    from unittest.mock import MagicMock, Mock

    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    import omlx.server as srv
    from omlx.engine.batched import BatchedEngine
    from omlx.engine.uno import UnoEngine

    engine = UnoEngine("base", adapter_path="adapter") if uno else BatchedEngine("base")
    engine._loaded = True
    engine._model = SimpleNamespace(args=SimpleNamespace(model_type="k2_horizon"))
    engine._tokenizer = SimpleNamespace(encode=lambda _: [3, 4])
    engine._bundle = SimpleNamespace(config={"vocab_size": 128}, context_length=131072)
    engine._apply_chat_template = Mock(return_value="prompt")
    engine.count_chat_tokens = Mock(return_value=2)
    engine._engine = SimpleNamespace(engine=SimpleNamespace(scheduler=object()))
    engine._preflight_or_raise_with_eviction = AsyncMock(
        side_effect=HTTPException(status_code=418, detail="Reached memory preflight")
    )
    pool = MagicMock()
    pool.get_entry.return_value = SimpleNamespace(
        config_model_type="k2_horizon", preserve_thinking_default=None
    )
    monkeypatch.setattr(srv._server_state, "engine_pool", pool)
    monkeypatch.setattr(srv, "get_engine_for_model", AsyncMock(return_value=engine))
    monkeypatch.setattr(srv, "resolve_model_id", lambda name: name)
    monkeypatch.setattr(srv, "validate_context_window", lambda *a, **k: None)
    monkeypatch.setattr(
        srv, "get_model_settings_for_request", lambda _: ModelSettings()
    )
    monkeypatch.setitem(srv.app.dependency_overrides, srv.verify_api_key, lambda: True)
    body = {"model": "base", "stream": stream, "max_tokens": 16}
    if endpoint == "/v1/responses":
        body["input"] = "Hello"
    else:
        body["messages"] = [{"role": "user", "content": "Hello"}]
    response = TestClient(srv.app, raise_server_exceptions=False).post(
        endpoint, json=body
    )
    assert response.status_code == 418, response.text


@pytest.mark.asyncio
async def test_profiles_and_load_revalidate_adapters_and_inherited_defaults(
    models, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from fastapi import HTTPException

    from omlx.admin import routes
    from omlx.exceptions import ModelUnavailableError

    base, adapter = models
    pool = EnginePool()
    pool.discover_models(str(tmp_path))
    manager = ModelSettingsManager(tmp_path / "settings")
    defaults = {"repetition_penalty": 1.1}
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)
    monkeypatch.setattr(routes, "_get_settings_manager", lambda: manager)
    monkeypatch.setattr(routes, "_get_server_state", lambda: None)
    monkeypatch.setattr(
        routes,
        "_get_global_settings",
        lambda: SimpleNamespace(sampling=SimpleNamespace(to_dict=lambda: defaults)),
    )
    inherited = dict(uno_enabled=True, uno_adapter_model=adapter.name)
    valid = dict(inherited, repetition_penalty=1.0)
    invalid = dict(valid, uno_adapter_model="missing-Uno")
    for settings in (inherited, invalid):
        with pytest.raises(HTTPException) as error:
            await routes.update_model_settings(
                base.name, routes.ModelSettingsRequest(**settings), is_admin=True
            )
        assert error.value.status_code == 400
        with pytest.raises(HTTPException) as error:
            await routes.create_model_profile(
                base.name,
                routes.CreateProfileRequest(
                    name="uno", display_name="Uno", settings=settings
                ),
                is_admin=True,
            )
        assert error.value.status_code == 400
    assert manager.list_profiles(base.name) == []
    await routes.create_model_profile(
        base.name,
        routes.CreateProfileRequest(name="uno", display_name="Uno", settings=valid),
        is_admin=True,
    )
    with pytest.raises(HTTPException):
        await routes.update_model_profile(
            base.name,
            "uno",
            routes.UpdateProfileRequest(settings=invalid),
            is_admin=True,
        )
    await routes.apply_model_profile(base.name, "uno", is_admin=True)
    assert manager.get_settings(base.name).repetition_penalty == 1.0
    manager.set_settings(base.name, ModelSettings(temperature=0.3))
    # A previously valid profile must fail atomically if its local helper disappears.
    pool._entries.pop(adapter.name)
    with pytest.raises(HTTPException) as error:
        await routes.apply_model_profile(base.name, "uno", is_admin=True)
    assert error.value.status_code == 400
    assert manager.get_settings(base.name) == ModelSettings(temperature=0.3)
    with patch("omlx.engine.uno.UnoEngine") as constructor:
        with pytest.raises(ModelUnavailableError, match="available Uno adapter"):
            await pool._load_engine(base.name, runtime_settings=ModelSettings(**valid))
        constructor.assert_not_called()
    assert defaults == {"repetition_penalty": 1.1}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_id,supported", [("IFM/K2-Horizon-0.9B", True), ("other/model", False)]
)
async def test_model_card_recognizes_supported_uno_adapter(
    tmp_path, base_id, supported
):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from omlx.admin.hf_downloader import HFDownloader

    config = tmp_path / "adapter_config.json"
    config.write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": base_id})
    )
    info = MagicMock(
        id="user/K2-Uno",
        downloads=0,
        likes=0,
        tags=[],
        pipeline_tag="text-generation",
        created_at=None,
        last_modified=None,
        safetensors=None,
        card_data=None,
    )
    info.siblings = [
        SimpleNamespace(rfilename=name, size=1000)
        for name in ("adapter_config.json", "adapter_model.safetensors")
    ]

    def download(*args, filename, **kwargs):
        if filename == "adapter_config.json":
            return str(config)
        raise FileNotFoundError(filename)

    with (
        patch("omlx.admin.hf_downloader.HfApi") as api,
        patch("omlx.admin.hf_downloader.hf_hub_download", side_effect=download),
    ):
        api.return_value.model_info.return_value = info
        result = await HFDownloader.get_model_info("user/K2-Uno")
    assert result["is_adapter"] is True
    assert result["is_uno_adapter"] is supported


@pytest.mark.parametrize("base_uno", [False, True])
def test_exposed_profile_sampling_view_matches_its_uno_engine(tmp_path, base_uno):
    manager = ModelSettingsManager(tmp_path)
    manager.set_settings(
        "base", ModelSettings(uno_enabled=base_uno, uno_adapter_model="adapter")
    )
    manager.save_profile(
        "base",
        "mode",
        "Mode",
        description=None,
        settings={
            "uno_enabled": not base_uno,
            "uno_adapter_model": "adapter",
            "repetition_penalty": 1.2 if base_uno else 1.0,
        },
        expose_as_model=True,
        api_name="mode",
    )
    sampling = manager.get_settings_for_request("base:mode")
    _, runtime = manager.get_exposed_profile_runtime_settings_for_request("base:mode")
    assert sampling.uno_enabled == runtime.uno_enabled == (not base_uno)
    assert sampling.repetition_penalty == runtime.repetition_penalty
    assert manager.get_settings("base").uno_enabled == base_uno


def test_benchmark_reports_uno_and_sanitizes_adapter_paths():
    from omlx.admin.benchmark import (
        _detect_experimental_features,
        _filter_uploaded_settings,
    )

    settings = ModelSettings(
        uno_enabled=True, uno_adapter_model="/private/models/K2-Uno"
    )
    assert "uno" in _detect_experimental_features(settings)
    snapshot = _filter_uploaded_settings(settings)
    assert snapshot["uno_enabled"] is True
    assert snapshot["uno_adapter_model"] == "K2-Uno"
