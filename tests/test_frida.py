# SPDX-License-Identifier: Apache-2.0
"""FRIDA adapter contracts with tiny local original-format checkpoints."""

import asyncio
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import numpy as np
import pytest
from frida_decisions.base import BaseJudge
from frida_decisions.config import DecisionsConfig
from frida_decisions.mlx_backend import MlxJudge
from frida_decisions.mlx_modeling import FridaMlxDecisionModel, StateCache
from frida_decisions.protocol import aggregate
from mlx.utils import tree_flatten
from tokenizers import Tokenizer, models, pre_tokenizers

from omlx.api.systemone_models import SystemOneRequest
from omlx.engine.decision import _MODEL_CLASSES, DecisionEngine
from omlx.model_discovery import (
    _is_hf_cache_mlx_compatible,
    decision_kind,
    detect_model_type,
    estimate_model_size,
)
from omlx.model_settings import ModelSettings
from omlx.models.decision import DecisionContextLengthError, DecisionRequestError
from omlx.models.frida import FridaModel
from omlx.models.frida_memory import estimate_frida_memory


@pytest.fixture
def checkpoint(tmp_path):
    cfg = dict(
        model_type="t5",
        feed_forward_proj="gated-gelu",
        vocab_size=16,
        d_model=8,
        d_ff=16,
        d_kv=4,
        num_heads=2,
        num_layers=2,
        layer_norm_epsilon=1e-6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
    )
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    DecisionsConfig().save(tmp_path)
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 3, "текст": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    model = FridaMlxDecisionModel(SimpleNamespace(**cfg))
    parameters = dict(tree_flatten(model.parameters()))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"),
        {
            name: parameters[target].astype(mx.bfloat16)
            for name, target in model.weight_mapping().items()
        },
    )
    mx.save_safetensors(
        str(tmp_path / "head.safetensors"),
        {
            name: parameters[f"head.{name}"].astype(mx.float32)
            for name in ("weight", "bias")
        },
    )
    return tmp_path


def request(options=2):
    return {
        "state": {"message": "текст"},
        "questions": {
            "choice": {
                "type": "choice",
                "instructions": "Выбери",
                "criteria": {"b": None, "a": {"x": 1}},
            },
            "score": {
                "type": "score",
                "instructions": "Оцени",
                "criteria": ["низкий", "высокий"],
            },
            "yes": {"type": "noul", "instructions": "Верно?"},
            "ranking": {
                "type": "ranking",
                "instructions": "Порядок",
                "criteria": {str(i): "текст" for i in range(options)},
            },
        },
    }


def drain(steps):
    yields = []
    while True:
        try:
            yields.append(next(steps))
        except StopIteration as done:
            return done.value, yields


def test_discovery_registration_and_hf_snapshot(checkpoint, tmp_path):
    assert decision_kind(checkpoint) == "frida"
    assert detect_model_type(checkpoint) == "decision"
    assert _is_hf_cache_mlx_compatible(checkpoint, "ai-forever/FRIDA-Decisions")
    assert _MODEL_CLASSES["frida"] is FridaModel
    snapshot = tmp_path / "models--someone--renamed" / "snapshots" / ("a" * 40)
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "model.safetensors",
        "head.safetensors",
        "decisions_config.json",
        "tokenizer.json",
    ):
        (snapshot / name).symlink_to(checkpoint / name)
    assert decision_kind(snapshot) == "frida"
    assert _is_hf_cache_mlx_compatible(snapshot, "someone/renamed")


@pytest.mark.parametrize(
    "missing",
    [
        "model.safetensors",
        "head.safetensors",
        "decisions_config.json",
        "tokenizer.json",
    ],
)
def test_incomplete_checkpoint(checkpoint, missing):
    (checkpoint / missing).unlink()
    assert decision_kind(checkpoint) is None
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        FridaModel(str(checkpoint)).load()


def test_requires_t5(checkpoint):
    assert decision_kind(checkpoint, {"model_type": "qwen3_5"}) is None


def test_precision_and_memory(checkpoint):
    fp32 = estimate_frida_memory(checkpoint, "fp32")
    bf16 = estimate_frida_memory(checkpoint, "bf16")
    with patch("mlx.core.load", side_effect=AssertionError("must not allocate")):
        assert estimate_frida_memory(checkpoint) == fp32
    assert fp32.resident_bytes > bf16.resident_bytes
    assert fp32.source_bytes == bf16.source_bytes
    assert fp32.loading_bytes > fp32.resident_bytes + fp32.source_bytes
    assert estimate_model_size(checkpoint) == fp32.resident_bytes
    assert ModelSettings().frida_precision == "fp32"
    assert (
        ModelSettings.from_dict({"frida_precision": "bf16"}).to_dict()[
            "frida_precision"
        ]
        == "bf16"
    )
    with pytest.raises(ValueError):
        ModelSettings(frida_precision="fp16")


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_load_original_weights_and_close(checkpoint, precision):
    adapter = FridaModel(str(checkpoint), precision=precision)
    adapter.load()
    judge = adapter.judge
    assert judge.dtype == (mx.float32 if precision == "fp32" else mx.bfloat16)
    assert judge.model.head.weight.dtype == mx.float32
    assert judge.state_max == 384 and judge.rows_per_forward == 1
    assert judge.compile_encoder and judge.state_cache is None
    adapter.close()
    assert adapter.judge is None
    assert (
        judge.model is judge._forward_encoder is judge._forward_cached_encoder is None
    )


def test_validation_caps_and_cpu_plan(checkpoint):
    adapter = FridaModel(str(checkpoint))
    # CPU compiler needs no loaded encoder.
    adapter.judge = BaseJudge(checkpoint, state_max=384)
    body = request()
    plan = adapter.encode(body)
    assert plan.parsed.state == body["state"]
    assert not any(
        isinstance(v, (mx.array, FridaModel, MlxJudge)) for v in vars(plan).values()
    )
    assert (
        SystemOneRequest(model="frida", **body).questions["ranking"].type == "ranking"
    )
    for field in ("state", "instructions", "option"):
        body = request()
        if field == "state":
            body["state"] = "текст " * 385
        elif field == "instructions":
            body["questions"]["yes"]["instructions"] = "текст " * 97
        else:
            body["questions"]["ranking"]["criteria"]["0"] = "текст " * 256
        with pytest.raises(DecisionContextLengthError):
            adapter.encode(body, truncate=False)
        truncated = adapter.encode(body)
        assert len(truncated.tokenized.state) <= 384
        assert all(len(q) <= 96 for q in truncated.tokenized.questions)
        assert all(len(o) <= 256 and o[-1] == 2 for _, o in truncated.tokenized.options)
        assert truncated.state_truncated == (field == "state")
    with pytest.raises(DecisionRequestError, match="images"):
        adapter.encode({**request(), "images": ["broken"]})
    assert adapter.encode({**request(), "images": []})
    bad = request()
    bad["questions"]["ranking"]["criteria"] = ["only"]
    with pytest.raises(DecisionRequestError, match="at least two"):
        adapter.encode(bad)
    body = request()
    body["questions"]["ranking"]["criteria"] = [{"x": 1}, {"x": 2}]
    assert adapter.encode(body).parsed.questions["ranking"].criteria == {
        "0": {"x": 1},
        "1": {"x": 2},
    }


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
@pytest.mark.parametrize("options", [2, 20])
def test_row_execution_parity_usage_and_request_cache(checkpoint, precision, options):
    adapter = FridaModel(str(checkpoint), precision=precision)
    adapter.load()
    judge = adapter.judge
    plan = adapter.encode(request(options))
    captured = []
    original = StateCache.clear

    def clear(cache):
        original(cache)
        captured.append(cache)

    with patch.object(StateCache, "clear", clear):
        result, yields = drain(adapter.run(plan, lambda: 1))
    assert sum(yields) == result["input_tokens"]
    assert judge.state_cache is None
    if options == 20:
        assert len(yields) > 2  # complete state forward, then query rows
        assert captured and captured[0].bytes == 0
        assert captured[0].max_bytes == 512 * 2**20
    else:
        assert len(yields) == 1 and not captured
    expected_margins = judge._score_packed([plan.tokenized])[0][0]
    if options == 20:
        judge.state_cache = StateCache(512 * 2**20)
        cached, _ = judge._score_cached(plan.tokenized)
        np.testing.assert_allclose(cached, expected_margins, atol=1e-3, rtol=1e-3)
        expected_margins = cached
        judge.state_cache.clear()
        judge.state_cache = None
    expected = aggregate(plan.parsed, plan.candidates, expected_margins)
    assert result["answers"]["choice"]["choice"] == expected["choice"]["choice"]
    assert result["answers"]["ranking"]["ranking"] == expected["ranking"]["ranking"]
    np.testing.assert_allclose(
        list(result["answers"]["ranking"]["scores"].values()),
        list(expected["ranking"]["scores"].values()),
        atol=1e-3,
        rtol=1e-3,
    )
    assert set(result["answers"]["ranking"]) == {
        "type",
        "ranking",
        "scores",
        "probabilities",
        "confidence",
    }
    assert result["usage"] == {
        "state_tokens": len(plan.tokenized.state),
        "state_truncated": False,
    }
    assert sum(result["answers"]["ranking"]["probabilities"].values()) == pytest.approx(
        1
    )
    # Equal texts yield equal scores: upstream sorts IDs lexicographically.
    tied = aggregate(plan.parsed, plan.candidates, [0.0] * len(plan.candidates))
    assert tied["ranking"]["ranking"] == sorted(plan.questions["ranking"]["criteria"])
    adapter.close()


@pytest.mark.parametrize("failure", [False, True])
def test_cache_cleanup_on_close_and_failure(checkpoint, failure):
    adapter = FridaModel(str(checkpoint))
    adapter.load()
    captured = []
    original = StateCache.clear

    def clear(cache):
        original(cache)
        captured.append(cache)

    steps = adapter.run(adapter.encode(request(20)), lambda: 1)
    with patch.object(StateCache, "clear", clear):
        next(steps)  # state forward; cached tensors exist
        if failure:
            with (
                patch.object(
                    adapter.judge,
                    "_forward_cached_encoder",
                    side_effect=RuntimeError("failure"),
                ),
                pytest.raises(RuntimeError),
            ):
                next(steps)
        else:
            steps.close()
    assert captured and captured[0].bytes == 0 and len(captured[0]) == 0
    assert adapter.judge.state_cache is None
    adapter.close()


@pytest.mark.asyncio
async def test_engine_scheduling_and_reload(checkpoint):
    engine = DecisionEngine(str(checkpoint))
    await engine.start()
    assert engine.get_model_info()["vision"] is False
    plan = await engine.encode(request(20))
    with (
        patch.object(engine._fairness, "wait_turn", return_value=False) as wait,
        patch.object(engine._fairness, "settle") as settle,
    ):
        result = await engine.systemone(plan)
    assert wait.call_count == settle.call_count >= 4
    assert result["input_tokens"] > 0
    await engine.stop()
    await engine.start()
    reloaded = (await engine.systemone(plan))["answers"]
    np.testing.assert_allclose(
        reloaded["yes"]["noul"], result["answers"]["yes"]["noul"], atol=1e-6
    )
    np.testing.assert_allclose(
        list(reloaded["ranking"]["scores"].values()),
        list(result["answers"]["ranking"]["scores"].values()),
        atol=1e-6,
    )
    await engine.stop()


@pytest.mark.asyncio
async def test_cancel_waits_for_gpu_before_cleanup_and_stop(checkpoint):
    engine = DecisionEngine(str(checkpoint))
    await engine.start()
    plan = await engine.encode(request(20))
    entered, release = threading.Event(), threading.Event()
    original = engine._step

    def blocked(steps):
        entered.set()
        release.wait(10)
        return original(steps)

    with patch.object(engine, "_step", blocked):
        task = asyncio.create_task(engine.systemone(plan))
        await asyncio.to_thread(entered.wait, 10)
        task.cancel()
        stop = asyncio.create_task(engine.stop())
        await asyncio.sleep(0)
        assert not stop.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        await stop
    assert engine._model is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["clef", "openjev"])
async def test_legacy_rejects_ranking_before_encode(kind):
    engine = DecisionEngine("unused")
    engine._kind = kind
    engine._model = MagicMock()
    with pytest.raises(DecisionRequestError, match="ranking"):
        await engine.encode(request())
    engine._model.encode.assert_not_called()


@pytest.fixture
def frida_pool(checkpoint):
    from omlx.engine_pool import EngineEntry, EnginePool

    pool = EnginePool()
    pool._get_final_ceiling = lambda: 10 * 2**30
    pool._entries["frida"] = EngineEntry(
        model_id="frida",
        model_path=str(checkpoint),
        model_type="decision",
        engine_type="decision",
        estimated_size=estimate_model_size(checkpoint),
    )
    settings = ModelSettings(frida_precision="fp32", ttl_seconds=1)
    pool._settings_manager = SimpleNamespace(get_settings=lambda _: settings)
    return pool, settings


@pytest.mark.asyncio
async def test_pool_first_load_lease_pinning_idle_eviction_and_precision_reload(
    frida_pool,
):
    from omlx.exceptions import ModelBusyError

    pool, settings = frida_pool
    entry = pool._entries["frida"]
    with patch("omlx.engine_pool.get_phys_footprint", return_value=0):
        assert entry.engine is None
        async with pool.acquire("frida") as engine:
            assert entry.in_use == 1
            assert engine.kind == "frida"
            assert (
                pool.current_model_memory
                == estimate_frida_memory(entry.model_path).resident_bytes
            )
            entry.last_access = 0
            assert await pool.check_ttl_expirations(pool._settings_manager) == []
            settings.frida_precision = "bf16"
            with pytest.raises(ModelBusyError):
                await pool.get_engine("frida")
        bf16 = await pool.get_engine("frida")
        assert bf16 is not engine and bf16._model.judge.dtype == mx.bfloat16
        assert (
            pool.current_model_memory
            == estimate_frida_memory(entry.model_path, "bf16").resident_bytes
        )
        entry.is_pinned = True
        entry.last_access = 0
        assert await pool.check_ttl_expirations(pool._settings_manager) == []
        entry.is_pinned = False
        entry.engine._begin_activity("decision", detail="active")
        assert not await pool.unload_if_idle_unpinned("frida")
        entry.engine._reset_activity_tracking()
        entry.last_access = 0
        assert await pool.check_ttl_expirations(pool._settings_manager) == ["frida"]
        assert pool.current_model_memory == 0 and entry.runtime_estimated_size is None
        again = await pool.get_engine("frida")
        assert again is not bf16
        assert await pool.unload_if_idle_unpinned("frida")
        assert pool.current_model_memory == 0


@pytest.mark.asyncio
async def test_pool_admits_loading_peak_before_allocation(frida_pool):
    from omlx.exceptions import ModelTooLargeError

    pool, settings = frida_pool
    estimate = estimate_frida_memory(pool._entries["frida"].model_path)
    pool._get_final_ceiling = lambda: estimate.loading_bytes - 1
    with (
        patch("omlx.engine_pool.mx.get_active_memory", return_value=0),
        patch("omlx.engine_pool._settled_phys_footprint", return_value=0),
        patch("omlx.engine_pool.DecisionEngine") as constructor,
    ):
        with pytest.raises(ModelTooLargeError):
            await pool.get_engine("frida")
        constructor.assert_not_called()
    assert pool.current_model_memory == 0


def test_precision_signature_and_admin_schema(frida_pool):
    from omlx.admin.routes import ModelSettingsRequest

    pool, settings = frida_pool
    before = pool._engine_runtime_signature("frida", settings)
    settings.frida_precision = "bf16"
    assert pool._engine_runtime_signature("frida", settings) != before
    assert ModelSettingsRequest(frida_precision="bf16").frida_precision == "bf16"
    with pytest.raises(ValueError):
        ModelSettingsRequest(frida_precision="fp16")


def test_bundle_generated_dependency(tmp_path, monkeypatch):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "frida_bundle_build", root / "packaging/build.py"
    )
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    (tmp_path / "venvstacks.toml").write_text(
        (root / "packaging/venvstacks.toml").read_text()
    )
    monkeypatch.setattr(build, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(build, "_read_pyproject_requirements", lambda: original)
    import tomllib

    original = {
        "project": tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "dependencies"
        ]
    }
    original.update(
        tomllib.loads((root / "pyproject.toml").read_text())["project"][
            "optional-dependencies"
        ]
    )
    generated = build._generate_venvstacks_toml()
    assert any(
        "frida-decisions" in req and "00c8b5f0a88312e969d65b01a076bd9e447de2db" in url
        for req, url in build._parse_git_requirements(generated)
    )


@pytest.mark.asyncio
async def test_admin_precision_persists_and_defers_busy_reload(frida_pool):
    from omlx.admin import routes

    pool, settings = frida_pool
    manager = MagicMock()
    manager.get_settings.return_value = settings
    manager.set_settings.side_effect = lambda _, updated: None
    state = MagicMock()
    with patch("omlx.engine_pool.get_phys_footprint", return_value=0):
        async with pool.acquire("frida"):
            with (
                patch.object(routes, "_get_engine_pool", return_value=pool),
                patch.object(routes, "_get_settings_manager", return_value=manager),
                patch.object(routes, "_get_server_state", return_value=state),
            ):
                result = await routes.update_model_settings(
                    "frida",
                    routes.ModelSettingsRequest(frida_precision="bf16"),
                    is_admin=True,
                )
            assert result["requires_reload"] and result["reload_deferred"]
            assert settings.frida_precision == "bf16"
        # The pool completes the deferred unload on lease release.
        engine = await pool.get_engine("frida")
        assert engine._model.judge.dtype == mx.bfloat16
        assert await pool.unload_if_idle_unpinned("frida")


def test_precision_persists_to_disk(tmp_path):
    from omlx.model_settings import ModelSettingsManager

    manager = ModelSettingsManager(tmp_path)
    manager.set_settings("frida", ModelSettings(frida_precision="bf16"))
    assert (
        ModelSettingsManager(tmp_path).get_settings("frida").frida_precision == "bf16"
    )


@pytest.mark.asyncio
async def test_insufficient_memory_preserves_pinned_model(frida_pool):
    from omlx.engine_pool import EngineEntry
    from omlx.exceptions import InsufficientMemoryError

    pool, settings = frida_pool
    estimate = estimate_frida_memory(pool._entries["frida"].model_path)
    pool._get_final_ceiling = lambda: estimate.loading_bytes + 1024
    active = MagicMock()
    active.has_active_requests.return_value = False
    pool._entries["pinned"] = EngineEntry(
        model_id="pinned",
        model_path="unused",
        model_type="decision",
        engine_type="decision",
        estimated_size=2048,
        engine=active,
        is_pinned=True,
    )
    pool._current_model_memory = 2048
    with (
        patch("omlx.engine_pool.mx.get_active_memory", return_value=0),
        patch("omlx.engine_pool._settled_phys_footprint", return_value=0),
        patch("omlx.engine_pool.DecisionEngine") as constructor,
    ):
        with pytest.raises(InsufficientMemoryError):
            await pool.get_engine("frida")
        constructor.assert_not_called()
    assert pool._entries["pinned"].engine is active
    assert pool.current_model_memory == 2048
