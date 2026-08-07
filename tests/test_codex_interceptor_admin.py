"""Admin control-plane tests for the native Codex interceptor."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from omlx.admin.routes import (
    CodexInterceptorStartRequest,
    CodexInterceptorSwitchRequest,
    get_codex_interceptor_doctor,
    get_codex_interceptor_status,
    start_codex_interceptor,
    stop_codex_interceptor,
    switch_codex_interceptor_model,
)


def test_start_builds_loopback_process_scoped_configuration(tmp_path):
    manager = MagicMock()
    manager.start.return_value = {
        "phase": "running",
        "running": True,
        "session_id": "session-1",
    }
    pool = MagicMock()
    pool.get_status.return_value = {
        "models": [
            {
                "id": "local/qwen",
                "model_type": "llm",
                "model_context_length": 131072,
                "loaded": False,
                "is_loading": False,
            }
        ]
    }
    pool.get_engine = AsyncMock(return_value=object())
    settings = SimpleNamespace(
        server=SimpleNamespace(port=8123),
        auth=SimpleNamespace(api_key="local-key"),
    )
    request = CodexInterceptorStartRequest(
        model="local/qwen",
        project=str(tmp_path),
        context_window=131072,
    )

    async def run():
        with (
            patch(
                "omlx.codex_interceptor.get_codex_interceptor_manager",
                return_value=manager,
            ),
            patch("omlx.admin.routes._get_global_settings", return_value=settings),
            patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        ):
            result = await start_codex_interceptor(request, is_admin=True)
            await asyncio.sleep(0)
            return result

    result = asyncio.run(run())
    interceptor_config = manager.start.call_args.args[0]

    assert result["running"] is True
    assert interceptor_config.upstream_url == "http://127.0.0.1:8123/v1/responses"
    assert interceptor_config.api_key == "local-key"
    assert interceptor_config.project == tmp_path
    assert interceptor_config.context_window == 131072
    assert interceptor_config.replace_existing is False
    pool.get_engine.assert_awaited_once_with("local/qwen")
    manager.set_warmup_status.assert_called_once_with(
        "session-1", "ready", "local/qwen"
    )


def test_switch_returns_immediately_then_publishes_after_warmup():
    manager = MagicMock()
    manager.begin_model_switch.return_value = (
        7,
        {
            "phase": "running",
            "running": True,
            "active_model": "local/old",
            "model_switch_loading": True,
        },
    )
    pool = MagicMock()
    pool.get_status.return_value = {
        "models": [
            {
                "id": "local/new",
                "model_type": "llm",
                "model_context_length": 65536,
                "loaded": False,
                "is_loading": False,
            }
        ]
    }
    load_gate = asyncio.Event()

    async def load_model(_model):
        await load_gate.wait()
        return object()

    pool.get_engine = AsyncMock(side_effect=load_model)
    request = CodexInterceptorSwitchRequest(model="local/new", context_window=65536)

    async def run():
        with (
            patch(
                "omlx.codex_interceptor.get_codex_interceptor_manager",
                return_value=manager,
            ),
            patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        ):
            result = await switch_codex_interceptor_model(request, is_admin=True)
            manager.complete_model_switch.assert_not_called()
            load_gate.set()
            for _ in range(20):
                if manager.complete_model_switch.called:
                    break
                await asyncio.sleep(0.01)
            return result

    result = asyncio.run(run())

    assert result["model_switch_loading"] is True
    manager.begin_model_switch.assert_called_once_with(
        "local/new",
        context_window=65536,
        local_label="Local · oMLX · new",
    )
    pool.get_engine.assert_awaited_once_with("local/new")
    manager.complete_model_switch.assert_called_once_with(7)


def test_switch_rejects_before_loading_when_context_is_unsafe():
    manager = MagicMock()
    manager.begin_model_switch.side_effect = RuntimeError(
        "stop the interceptor and start a fresh Codex session"
    )
    pool = MagicMock()
    pool.get_status.return_value = {
        "models": [
            {
                "id": "local/small",
                "model_type": "llm",
                "model_context_length": 8192,
            }
        ]
    }
    pool.get_engine = AsyncMock()

    async def run():
        with (
            patch(
                "omlx.codex_interceptor.get_codex_interceptor_manager",
                return_value=manager,
            ),
            patch("omlx.admin.routes._get_engine_pool", return_value=pool),
        ):
            with pytest.raises(HTTPException) as caught:
                await switch_codex_interceptor_model(
                    CodexInterceptorSwitchRequest(model="local/small"),
                    is_admin=True,
                )
            return caught.value

    error = asyncio.run(run())
    assert error.status_code == 409
    pool.get_engine.assert_not_awaited()


def test_status_doctor_and_stop_delegate_to_single_manager():
    manager = MagicMock()
    manager.status.return_value = {"phase": "running"}
    manager.doctor.return_value = {"ready": True}
    manager.stop.return_value = {"phase": "stopped"}

    async def run():
        with patch(
            "omlx.codex_interceptor.get_codex_interceptor_manager",
            return_value=manager,
        ):
            return (
                await get_codex_interceptor_status(is_admin=True),
                await get_codex_interceptor_doctor(is_admin=True),
                await stop_codex_interceptor(is_admin=True),
            )

    status, doctor, stopped = asyncio.run(run())
    assert status["phase"] == "running"
    assert doctor["ready"] is True
    assert stopped["phase"] == "stopped"
