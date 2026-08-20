# SPDX-License-Identifier: Apache-2.0
"""Tests for DS4 EnginePool lifecycle integration."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from omlx.ds4_process import DS4LogLine
from omlx.engine.ds4 import (
    DS4ProcessEngine,
    DS4ProxyError,
    DS4ProxyResponse,
    parse_ds4_progress_log_line,
)
from omlx.engine_pool import EnginePool
from omlx.exceptions import InsufficientMemoryError, ModelLoadingError
from omlx.server import ServerState, app
from omlx.settings import DS4_THINK_MAX_CONTEXT_TOKENS, DS4Settings


class FakeManagedProcess:
    """Small stand-in for DS4ManagedProcess that avoids spawning subprocesses."""

    instances: list[FakeManagedProcess] = []

    def __init__(self, config, *, max_log_lines: int = 500):
        self.config = config
        self.max_log_lines = max_log_lines
        self.process = None
        self.port = None
        self.command = None
        self.log_path = None
        self.logs = []
        self.started = False
        self.stopped = False
        FakeManagedProcess.instances.append(self)

    @property
    def is_running(self) -> bool:
        return (
            self.started
            and self.process is not None
            and self.process.returncode is None
        )

    async def start(self) -> None:
        self.started = True
        self.port = self.config.port or 49152
        self.command = self.config.build_command(self.port)
        self.log_path = self.config.log_path
        self.process = SimpleNamespace(pid=12345, returncode=None)

    async def stop(self) -> None:
        self.stopped = True
        if self.process is not None:
            self.process.returncode = 0

    def recent_log_text(self) -> str:
        return "fake ds4 logs"

    def crash(self, returncode: int = 9) -> None:
        if self.process is not None:
            self.process.returncode = returncode


def _patch_fake_process(monkeypatch):
    FakeManagedProcess.instances = []
    monkeypatch.setattr("omlx.engine.ds4.DS4ManagedProcess", FakeManagedProcess)


def _pool_with_ds4(tmp_path, *, ds4_enabled: bool = True) -> EnginePool:
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
            debug_dir=str(tmp_path / "debug"),
            enabled=ds4_enabled,
        ),
    )
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path))
    return pool


def test_parse_ds4_progress_log_lines():
    """DS4 server progress logs expose phase and token-rate details."""
    prefill = parse_ds4_progress_log_line(
        "stderr: ds4-server: chat ctx=0..100 RESPPROTO,TOOLS prefill "
        "chunk 250/1000 (25.0%) chunk=125.50 t/s avg=83.33 t/s 3.000s"
    )
    decode = parse_ds4_progress_log_line(
        "stderr: ds4-server: completion ctx=100..104 gen=4 RESPPROTO "
        "decoding chunk=10.00 t/s avg=8.00 t/s 0.500s"
    )

    assert prefill == {
        "kind": "chat",
        "phase": "prefill",
        "phase_type": "prefill",
        "current_tokens": 250,
        "total_tokens": 1000,
        "percent": 25.0,
        "chunk_tokens_per_second": 125.5,
        "average_tokens_per_second": 83.33,
        "elapsed_seconds": 3.0,
    }
    assert decode == {
        "kind": "completion",
        "phase": "decoding",
        "phase_type": "generation",
        "generated_tokens": 4,
        "chunk_tokens_per_second": 10.0,
        "average_tokens_per_second": 8.0,
        "elapsed_seconds": 0.5,
    }


@pytest.mark.asyncio
async def test_ds4_process_engine_activity_snapshot_surfaces_progress(
    monkeypatch, tmp_path
):
    """Active Models can show DS4 phase/TPS from captured ds4-server logs."""
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    FakeManagedProcess.instances[0].logs.append(
        DS4LogLine(
            "stderr",
            "ds4-server: chat ctx=0..100 prefill chunk 250/1000 "
            "(25.0%) chunk=125.50 t/s avg=83.33 t/s 3.000s",
            1.0,
        )
    )
    await engine.begin_proxy_request_window()
    try:
        with patch("omlx.engine.ds4.time.monotonic", return_value=4.5):
            snapshot = engine.get_activity_snapshot()
            stats = engine.get_stats()
    finally:
        engine.end_proxy_request_window()
        await engine.stop()

    assert snapshot["active_requests"] == 1
    assert snapshot["activities"] == [
        {
            "request_id": "ds4-foo",
            "kind": "ds4_proxy",
            "detail": "DS4 prefill 25.0%",
            "active_requests": 1,
            "elapsed_seconds": 3.0,
            "last_activity_age_seconds": 3.5,
            "current_tokens": 250,
            "total_tokens": 1000,
            "token_count": 250,
            "tokens_per_second": 83.33,
            "chunk_tokens_per_second": 125.5,
            "ds4_phase": "prefill",
            "ds4_phase_type": "prefill",
        }
    ]
    assert stats["active_requests"] == 1
    assert stats["progress"]["average_tokens_per_second"] == 83.33


@pytest.mark.asyncio
async def test_ds4_process_engine_starts_and_stops_fake_process(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        assert engine.is_running is True
        assert engine.port == 49152
        assert engine.pid == 12345
        stats = engine.get_stats()
        assert stats["backend"] == "ds4"
        assert stats["host"] == "127.0.0.1"
        assert stats["port"] == 49152
        assert stats["running"] is True
        assert stats["log_path"] == str(
            tmp_path / "logs" / "ds4-debug" / "foo" / "ds4.log"
        )
        assert stats["recent_logs"] == "fake ds4 logs"
    finally:
        await engine.stop()

    assert engine.is_running is False
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_ds4_process_engine_raises_context_for_think_max(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    settings = DS4Settings(
        context_default_tokens=32_768,
        support_dir=str(tmp_path / "support" / "ds4"),
        kv_root=str(tmp_path / "kv"),
    )
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=settings,
        base_path=tmp_path,
    )

    await engine.start()
    raised = await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)

    try:
        assert raised is True
        assert engine.context_tokens == DS4_THINK_MAX_CONTEXT_TOKENS
        assert settings.context_default_tokens == 32_768
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
        assert FakeManagedProcess.instances[1].config.context_tokens == (
            DS4_THINK_MAX_CONTEXT_TOKENS
        )
        assert "--ctx" in FakeManagedProcess.instances[1].command
        assert str(DS4_THINK_MAX_CONTEXT_TOKENS) in FakeManagedProcess.instances[1].command
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_restarts_with_configured_context(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    restarted = await engine.restart_with_context(100_000)

    try:
        assert restarted is True
        assert engine.context_tokens == 100_000
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].config.context_tokens == 100_000
        assert "100000" in FakeManagedProcess.instances[1].command
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_rejects_context_restart_if_request_starts(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    stop_started = asyncio.Event()
    continue_stop = asyncio.Event()

    async def slow_stop(self):
        stop_started.set()
        await continue_stop.wait()
        self.stopped = True
        if self.process is not None:
            self.process.returncode = 0

    monkeypatch.setattr(FakeManagedProcess, "stop", slow_stop)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    restart_task = asyncio.create_task(engine.restart_with_context(100_000))
    await stop_started.wait()
    begin_task = asyncio.create_task(engine.begin_proxy_request_window())
    await asyncio.sleep(0)
    assert engine.has_active_requests() is True
    continue_stop.set()

    with pytest.raises(DS4ProxyError, match="retry when idle"):
        await restart_task
    await begin_task
    try:
        assert engine.context_tokens is None
        assert engine.has_active_requests() is True
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[1].config.context_tokens is None
    finally:
        engine.end_proxy_request_window()
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_serializes_concurrent_context_raises(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)

    async def slow_stop(self):
        await asyncio.sleep(0.01)
        self.stopped = True
        self.running = False

    monkeypatch.setattr(FakeManagedProcess, "stop", slow_stop)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        results = await asyncio.gather(
            engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS),
            engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS),
        )
        assert results == [True, False]
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_think_max_context_noops_when_already_high(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=DS4_THINK_MAX_CONTEXT_TOKENS,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    try:
        raised = await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)
        assert raised is False
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_rejects_context_raise_while_active(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            context_default_tokens=32_768,
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    engine._increment_active_requests()
    try:
        with pytest.raises(DS4ProxyError, match="retry when idle"):
            await engine.ensure_min_context(DS4_THINK_MAX_CONTEXT_TOKENS)
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        engine._decrement_active_requests()
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_restarts_crashed_backend_before_request(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    def fake_proxy_response(self, path, body):
        try:
            assert self.is_running is True
            assert path == "/v1/chat/completions"
            assert body == {"model": "foo"}
            return DS4ProxyResponse(
                status_code=200,
                headers={"content-type": "application/json"},
                body=b'{"ok":true}',
            )
        finally:
            self._decrement_active_requests()

    monkeypatch.setattr(
        DS4ProcessEngine,
        "_proxy_json_response_blocking",
        fake_proxy_response,
    )

    await engine.start()
    FakeManagedProcess.instances[0].crash(returncode=9)

    response = await engine.proxy_chat_completion({"model": "foo"})

    try:
        assert response.body == b'{"ok":true}'
        assert len(FakeManagedProcess.instances) == 2
        assert FakeManagedProcess.instances[0].stopped is True
        assert FakeManagedProcess.instances[1].started is True
        stats = engine.get_stats()
        assert stats["running"] is True
        assert stats["crashed"] is False
        assert stats["crash_count"] == 1
        assert stats["restart_count"] == 1
        assert stats["last_crash_exit_code"] == 9
        assert stats["last_crash_logs"] == "fake ds4 logs"
    finally:
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_rejects_crash_restart_while_active(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(
        model_id="foo",
        model_path=gguf,
        settings=DS4Settings(
            support_dir=str(tmp_path / "support" / "ds4"),
            kv_root=str(tmp_path / "kv"),
        ),
        base_path=tmp_path,
    )

    await engine.start()
    FakeManagedProcess.instances[0].crash(returncode=9)
    engine._increment_active_requests()
    try:
        with pytest.raises(DS4ProxyError, match="retry when idle"):
            await engine.restart_if_crashed()
        assert len(FakeManagedProcess.instances) == 1
        assert FakeManagedProcess.instances[0].stopped is False
    finally:
        engine._decrement_active_requests()
        await engine.stop()


@pytest.mark.asyncio
async def test_ds4_process_engine_protocol_methods_are_explicitly_deferred(tmp_path):
    gguf = tmp_path / "Foo.gguf"
    gguf.write_bytes(b"0" * 1000)
    engine = DS4ProcessEngine(model_id="foo", model_path=gguf, base_path=tmp_path)

    with pytest.raises(RuntimeError, match="protocol forwarding"):
        await engine.chat([])
    with pytest.raises(RuntimeError, match="protocol forwarding"):
        await engine.generate("hello")


@pytest.mark.asyncio
async def test_engine_pool_loads_and_unloads_ds4_entries(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)

    engine = await pool.get_engine("foo")

    assert isinstance(engine, DS4ProcessEngine)
    assert engine.is_running is True
    assert pool.get_loaded_model_ids() == ["foo"]
    entry = pool.get_entry("foo")
    assert entry is not None
    assert entry.engine is engine
    assert entry.actual_size is not None
    status = pool.get_status()["models"][0]
    assert status["id"] == "foo"
    assert status["loaded"] is True
    assert status["engine_type"] == "ds4"
    assert status["ds4"]["running"] is True
    assert status["ds4"]["port"] == 49152

    await pool._unload_engine("foo")

    assert entry.engine is None
    assert pool.get_loaded_model_ids() == []
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_engine_pool_loads_ds4_with_per_model_context(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(max_context_window=100_000)

    pool._settings_manager = SettingsManager()

    await pool.get_engine("foo")

    assert FakeManagedProcess.instances[0].config.context_tokens == 100_000
    assert "--ctx" in FakeManagedProcess.instances[0].command
    assert "100000" in FakeManagedProcess.instances[0].command


@pytest.mark.asyncio
async def test_engine_pool_loads_ds4_with_per_model_mtp_args(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    mtp = tmp_path / "MTP.gguf"
    mtp.write_bytes(b"0" * 100)

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(
                ds4_mtp_enabled=True,
                ds4_mtp_path=str(mtp),
                ds4_mtp_draft=2,
                ds4_mtp_margin=3.0,
            )

    pool._settings_manager = SettingsManager()

    await pool.get_engine("foo")

    launch = FakeManagedProcess.instances[0]
    assert launch.config.mtp_path == mtp.resolve()
    assert "--mtp" in launch.command
    assert launch.command[launch.command.index("--mtp") + 1] == str(mtp.resolve())
    assert launch.command[launch.command.index("--mtp-draft") + 1] == "2"
    assert launch.command[launch.command.index("--mtp-margin") + 1] == "3.0"
    assert "--ssd-streaming" not in launch.command


@pytest.mark.asyncio
async def test_engine_pool_auto_enables_ds4_ssd_streaming_when_budget_is_tight(
    monkeypatch, tmp_path
):
    """DS4 auto mode uses --ssd-streaming when a GGUF exceeds memory budget."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))

    await pool.get_engine("foo")

    launch = FakeManagedProcess.instances[0]
    assert launch.config.auto_enable_ssd_streaming is True
    assert "--ssd-streaming" in launch.command
    assert "--ssd-streaming-cache-experts" not in launch.command


@pytest.mark.asyncio
async def test_engine_pool_ds4_mtp_disables_auto_ssd_streaming_admission(
    monkeypatch, tmp_path
):
    """MTP must not load under an auto-streaming memory assumption."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    mtp = tmp_path / "MTP.gguf"
    mtp.write_bytes(b"0" * 100)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ds4_mtp_enabled=True, ds4_mtp_path=str(mtp))

    pool._settings_manager = SettingsManager()

    with pytest.raises(InsufficientMemoryError):
        await pool.get_engine("foo")

    assert pool.get_entry("foo").ds4_auto_enable_ssd_streaming is False
    assert FakeManagedProcess.instances == []


@pytest.mark.asyncio
async def test_engine_pool_dspark_allows_auto_ssd_streaming_admission(
    monkeypatch, tmp_path
):
    """DSpark can combine its support GGUF with main-model SSD streaming."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.ds4_gguf.detect_ds4_mtp_sidecar_kind", lambda _path: "dspark"
    )
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    dspark = tmp_path / "DSpark-support.gguf"
    dspark.write_bytes(b"0" * 100)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ds4_mtp_enabled=True, ds4_mtp_path=str(dspark))

    pool._settings_manager = SettingsManager()

    await pool.get_engine("foo")

    launch = FakeManagedProcess.instances[0]
    assert launch.config.mtp_kind == "dspark"
    assert launch.config.auto_enable_ssd_streaming is True
    assert "--mtp" in launch.command
    assert "--dspark" in launch.command
    assert "--ssd-streaming" in launch.command


@pytest.mark.asyncio
async def test_engine_pool_waits_for_ds4_ceiling_recovery_after_eviction(
    monkeypatch, tmp_path
):
    """A stale low ceiling after DS4 eviction should not force streaming."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_AUTO_ADMISSION_SETTLE_TIMEOUT_SECONDS",
        2.0,
    )
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_AUTO_ADMISSION_SETTLE_INTERVAL_SECONDS",
        0.1,
    )

    (tmp_path / "Pro.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Flash.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    ceiling = {"value": 20_000}
    pool._get_final_ceiling = lambda: ceiling["value"]
    pool.discover_models(str(tmp_path))
    pool._entries["pro"].estimated_size = 10_000
    pool._entries["flash"].estimated_size = 4_000

    await pool.get_engine("pro")

    ceiling["value"] = 1_500
    sleep_calls = 0

    async def recover_during_settle(_duration):
        nonlocal sleep_calls
        sleep_calls += 1
        ceiling["value"] = 20_000

    monkeypatch.setattr("omlx.engine_pool.asyncio.sleep", recover_during_settle)

    await pool.get_engine("flash")

    launch = FakeManagedProcess.instances[-1]
    assert sleep_calls >= 1
    assert launch.config.auto_enable_ssd_streaming is False
    assert "--ssd-streaming" not in launch.command


@pytest.mark.asyncio
async def test_engine_pool_preserves_zero_ceiling_during_ds4_settle(
    monkeypatch, tmp_path
):
    """A 0 ceiling sampled during DS4 settle still means guard disabled."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.engine_pool.EnginePool._system_available_memory_bytes",
        lambda self: 20_000,
    )
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_AUTO_ADMISSION_SETTLE_TIMEOUT_SECONDS",
        2.0,
    )
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_AUTO_ADMISSION_SETTLE_INTERVAL_SECONDS",
        0.1,
    )

    (tmp_path / "Pro.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Flash.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    ceiling = {"value": 20_000}
    pool._get_final_ceiling = lambda: ceiling["value"]
    pool.discover_models(str(tmp_path))
    pool._entries["pro"].estimated_size = 10_000
    pool._entries["flash"].estimated_size = 4_000

    await pool.get_engine("pro")

    ceiling["value"] = 1_500

    async def disable_guard_during_settle(_duration):
        ceiling["value"] = 0

    monkeypatch.setattr("omlx.engine_pool.asyncio.sleep", disable_guard_during_settle)

    await pool.get_engine("flash")

    launch = FakeManagedProcess.instances[-1]
    assert launch.config.auto_enable_ssd_streaming is False
    assert "--ssd-streaming" not in launch.command


@pytest.mark.asyncio
async def test_engine_pool_unloads_idle_ds4_before_switch(monkeypatch, tmp_path):
    """DS4 model switches must stop the old singleton process first."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Pro.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Flash.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 20_000
    pool.discover_models(str(tmp_path))
    pool._entries["pro"].estimated_size = 4_000
    pool._entries["flash"].estimated_size = 4_000

    await pool.get_engine("pro")
    await pool.get_engine("flash")

    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True
    assert pool._entries["pro"].engine is None
    assert pool._entries["flash"].engine is not None


@pytest.mark.asyncio
async def test_engine_pool_rejects_ds4_switch_while_current_request_active(
    monkeypatch, tmp_path
):
    """A busy DS4 process should block switching before spawning a second one."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_SINGLETON_SWITCH_IDLE_WAIT_SECONDS",
        0.2,
    )
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_SINGLETON_SWITCH_IDLE_POLL_SECONDS",
        0.01,
    )
    (tmp_path / "Pro.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Flash.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))
    pool._entries["pro"].estimated_size = 10_000
    pool._entries["flash"].estimated_size = 4_000

    await pool.get_engine("pro")
    monkeypatch.setattr(
        pool._entries["pro"].engine,
        "has_active_requests",
        lambda: True,
    )

    with pytest.raises(ModelLoadingError, match="Timed out waiting to switch"):
        await pool.get_engine("flash")

    assert len(FakeManagedProcess.instances) == 1
    assert FakeManagedProcess.instances[0].stopped is False


@pytest.mark.asyncio
async def test_engine_pool_waits_for_ds4_cancel_before_switch(
    monkeypatch, tmp_path
):
    """A just-cancelled DS4 stream may go idle during the switch grace."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_SINGLETON_SWITCH_IDLE_WAIT_SECONDS",
        1.0,
    )
    monkeypatch.setattr(
        "omlx.engine_pool._DS4_SINGLETON_SWITCH_IDLE_POLL_SECONDS",
        0.1,
    )
    (tmp_path / "Pro.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Flash.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="auto"),
    )
    pool._get_final_ceiling = lambda: 20_000
    pool.discover_models(str(tmp_path))
    pool._entries["pro"].estimated_size = 4_000
    pool._entries["flash"].estimated_size = 4_000

    await pool.get_engine("pro")
    active = {"value": True}
    monkeypatch.setattr(
        pool._entries["pro"].engine,
        "has_active_requests",
        lambda: active["value"],
    )

    async def finish_cancel(_duration):
        active["value"] = False

    monkeypatch.setattr("omlx.engine_pool.asyncio.sleep", finish_cancel)

    await pool.get_engine("flash")

    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True


@pytest.mark.asyncio
async def test_engine_pool_ds4_ssd_streaming_off_preserves_full_admission(
    monkeypatch, tmp_path
):
    """User-forced off mode does not bypass normal memory admission."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="off"),
    )
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))

    with pytest.raises(InsufficientMemoryError):
        await pool.get_engine("foo")

    assert FakeManagedProcess.instances == []


@pytest.mark.asyncio
async def test_engine_pool_ds4_ssd_streaming_on_forces_launch_flag(
    monkeypatch, tmp_path
):
    """User-forced on mode passes --ssd-streaming even when auto would not."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(
        base_path=tmp_path,
        ds4_settings=DS4Settings(ssd_streaming="on"),
    )
    pool._get_final_ceiling = lambda: 5_000
    pool.discover_models(str(tmp_path))

    await pool.get_engine("foo")

    launch = FakeManagedProcess.instances[0]
    assert launch.config.auto_enable_ssd_streaming is False
    assert "--ssd-streaming" in launch.command


@pytest.mark.asyncio
async def test_engine_pool_preloads_pinned_ds4_models(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])

    await pool.preload_pinned_models()

    assert pool.get_entry("foo").engine is not None
    assert FakeManagedProcess.instances[0].started is True


@pytest.mark.asyncio
async def test_engine_pool_restarts_crashed_pinned_ds4_models(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])
    await pool.preload_pinned_models()

    FakeManagedProcess.instances[0].crash(returncode=7)
    restarted = await pool.restart_crashed_pinned_ds4()

    assert restarted == ["foo"]
    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True
    status = pool.get_status()["models"][0]
    assert status["ds4"]["running"] is True
    assert status["ds4"]["restart_count"] == 1
    assert status["ds4"]["last_crash_exit_code"] == 7


@pytest.mark.asyncio
async def test_engine_pool_leaves_unpinned_crashed_ds4_stopped_until_request(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    await pool.get_engine("foo")

    FakeManagedProcess.instances[0].crash(returncode=8)
    restarted = await pool.restart_crashed_pinned_ds4()

    assert restarted == []
    assert len(FakeManagedProcess.instances) == 1
    status = pool.get_status()["models"][0]
    assert status["ds4"]["running"] is False
    assert status["ds4"]["crashed"] is True
    assert status["ds4"]["exit_code"] == 8
    assert status["ds4"]["crash_count"] == 1
    assert status["ds4"]["last_crash_exit_code"] == 8
    assert status["ds4"]["last_crash_logs"] == "fake ds4 logs"

    status_again = pool.get_status()["models"][0]
    assert status_again["ds4"]["crash_count"] == 1


@pytest.mark.asyncio
async def test_ttl_check_restarts_crashed_pinned_ds4_models(
    monkeypatch, tmp_path
):
    _patch_fake_process(monkeypatch)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 0
    pool.discover_models(str(tmp_path), pinned_models=["foo"])
    await pool.preload_pinned_models()

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ttl_seconds=0)

    FakeManagedProcess.instances[0].crash(returncode=7)
    expired = await pool.check_ttl_expirations(SettingsManager())

    assert expired == []
    assert len(FakeManagedProcess.instances) == 2
    assert FakeManagedProcess.instances[1].started is True


@pytest.mark.asyncio
async def test_unpinned_ds4_process_can_be_evicted_under_memory_ceiling(
    monkeypatch, tmp_path
):
    """DS4 estimated memory participates in pre-load LRU admission."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 1_000)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Bar.gguf").write_bytes(b"0" * 1000)
    pool = EnginePool(base_path=tmp_path, ds4_settings=DS4Settings())
    pool._get_final_ceiling = lambda: 2_500
    pool.discover_models(str(tmp_path))

    await pool.get_engine("foo")
    await pool.get_engine("bar")

    assert pool.get_entry("foo").engine is None
    assert pool.get_entry("bar").engine is not None
    assert FakeManagedProcess.instances[0].stopped is True
    assert FakeManagedProcess.instances[1].started is True
    assert pool.current_model_memory + 1_000 <= 2_500


@pytest.mark.asyncio
async def test_ttl_expiration_unloads_idle_unpinned_ds4_process(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    await pool.get_engine("foo")

    class SettingsManager:
        def get_settings(self, model_id):
            from omlx.model_settings import ModelSettings

            return ModelSettings(ttl_seconds=0)

    expired = await pool.check_ttl_expirations(SettingsManager())

    assert expired == ["foo"]
    assert pool.get_entry("foo").engine is None
    assert FakeManagedProcess.instances[0].stopped is True


@pytest.mark.asyncio
async def test_engine_pool_rejects_ds4_load_when_backend_disabled(monkeypatch, tmp_path):
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path, ds4_enabled=False)

    with pytest.raises(RuntimeError, match="DS4 backend is disabled"):
        await pool.get_engine("foo")

    assert FakeManagedProcess.instances == []


@pytest.mark.asyncio
async def test_disabled_ds4_load_does_not_evict_existing_victim(monkeypatch, tmp_path):
    """Disabled backend errors before admission evicts loaded models."""
    _patch_fake_process(monkeypatch)
    monkeypatch.setattr("omlx.engine_pool.get_phys_footprint", lambda: 0)
    monkeypatch.setattr("omlx.engine_pool.mx.get_active_memory", lambda: 0)
    (tmp_path / "Foo.gguf").write_bytes(b"0" * 1000)
    (tmp_path / "Victim.gguf").write_bytes(b"0" * 1000)
    ds4_settings = DS4Settings(enabled=True)
    pool = EnginePool(base_path=tmp_path, ds4_settings=ds4_settings)
    pool._get_final_ceiling = lambda: 1_500
    pool.discover_models(str(tmp_path))
    await pool.get_engine("victim")

    ds4_settings.enabled = False
    with pytest.raises(RuntimeError, match="DS4 backend is disabled"):
        await pool.get_engine("foo")

    assert pool.get_entry("victim").engine is not None
    assert pool.get_entry("foo").engine is None
    assert FakeManagedProcess.instances[0].stopped is False


def test_public_load_unload_endpoints_manage_ds4_process(monkeypatch, tmp_path):
    """Existing manual model lifecycle endpoints work for DS4 entries."""
    _patch_fake_process(monkeypatch)
    pool = _pool_with_ds4(tmp_path)
    state = ServerState()
    state.engine_pool = pool
    state.api_key = None

    with patch("omlx.server._server_state", state):
        with TestClient(app, raise_server_exceptions=False) as client:
            load_response = client.post("/v1/models/foo/load")
            unload_response = client.post("/v1/models/foo/unload")

    assert load_response.status_code == 200
    assert load_response.json()["status"] == "ok"
    assert unload_response.status_code == 200
    assert unload_response.json()["status"] == "ok"
    assert FakeManagedProcess.instances[0].started is True
    assert FakeManagedProcess.instances[0].stopped is True
