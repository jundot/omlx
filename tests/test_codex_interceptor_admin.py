"""Admin control-plane tests for the native Codex interceptor."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from omlx.admin.routes import (
    CodexInterceptorStartRequest,
    get_codex_interceptor_doctor,
    get_codex_interceptor_status,
    start_codex_interceptor,
    stop_codex_interceptor,
)


def test_start_builds_loopback_process_scoped_configuration(tmp_path):
    manager = MagicMock()
    manager.start.return_value = {
        "phase": "running",
        "running": True,
        "session_id": "session-1",
    }
    pool = MagicMock()
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
    manager.set_warmup_status.assert_called_once_with("session-1", "ready")


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
