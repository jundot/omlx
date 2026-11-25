"""Regression tests for the ``python -m omlx.server`` entry point.

This path has broken twice: #2241 (main() passed kwargs init_server() no
longer accepts, TypeError at startup) and #2282 (the admin initial
API-key setup form 500s on a missing GlobalSettings). #2282 has two
layers: main() never loaded settings, and, deeper, running the file as
``__main__`` means the admin routes' request-time ``from ..server
import`` executes the module a second time and repoints the admin state
getters at a server state init_server() never touched. The in-process
tests cover the first layer against the real ``init_server``; only the
subprocess test can catch the second, because a normal import has a
single module instance by construction.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture
def module_entry(monkeypatch, tmp_path):
    """Run server.main() as ``python -m omlx.server --model-dir <tmp>``."""
    from omlx import server
    from omlx.settings import reset_settings

    # main() loads GlobalSettings. OMLX_BASE_PATH is first in its base
    # path resolution order, so setting it (plus HOME for any other ~
    # expansion) keeps the test away from the real user configuration;
    # HOME alone is not enough when a macOS app bootstrap file exists.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    reset_settings()

    # Reset the middleware stack so init_server's add_middleware works
    # even if another test in this process already started the app.
    server.app.middleware_stack = None

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = ["omlx.server", "--model-dir", str(model_dir)]
    with (
        patch.object(sys, "argv", argv),
        # Keep the process-wide allocator setting out of the test run.
        patch("mlx.core.set_cache_limit"),
        patch("uvicorn.run") as uvicorn_run,
    ):
        server.main()

    # Other suites patch admin_routes._get_global_settings and can leak
    # a mock into this process; re-run the canonical wiring so the
    # in-process tests see main()'s state exactly as a fresh interpreter
    # would (the subprocess test below covers the fresh process for
    # real, wiring included).
    from omlx.admin.routes import set_admin_getters

    set_admin_getters(
        server.get_server_state,
        server.get_engine_pool,
        lambda: server._server_state.settings_manager,
        lambda: server._server_state.global_settings,
    )

    yield server, uvicorn_run
    reset_settings()


def test_main_reaches_uvicorn_with_model_dir(module_entry):
    _, uvicorn_run = module_entry
    uvicorn_run.assert_called_once()


def test_main_defaults_to_loopback(module_entry):
    _, uvicorn_run = module_entry
    assert uvicorn_run.call_args.kwargs["host"] == "127.0.0.1"


def test_main_rejects_network_bind_without_api_key(monkeypatch, tmp_path, capsys):
    from omlx import server
    from omlx.settings import reset_settings

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    reset_settings()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = [
        "omlx.server",
        "--model-dir",
        str(model_dir),
        "--host",
        "0.0.0.0",
    ]

    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(server, "init_server") as init_server,
            patch("uvicorn.run") as uvicorn_run,
            pytest.raises(SystemExit) as exc_info,
        ):
            server.main()

        assert exc_info.value.code == 1
        assert "API key is required" in capsys.readouterr().out
        init_server.assert_not_called()
        uvicorn_run.assert_not_called()
    finally:
        reset_settings()


def test_main_accepts_network_bind_with_api_key(monkeypatch, tmp_path):
    from omlx import server
    from omlx.settings import reset_settings

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    monkeypatch.delenv("OMLX_API_KEY", raising=False)
    reset_settings()
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = [
        "omlx.server",
        "--model-dir",
        str(model_dir),
        "--host",
        "0.0.0.0",
        "--api-key",
        "test-key",
    ]

    try:
        with (
            patch.object(sys, "argv", argv),
            patch.object(server, "init_server") as init_server,
            patch("mlx.core.set_cache_limit"),
            patch("uvicorn.run") as uvicorn_run,
        ):
            server.main()

        init_server.assert_called_once()
        assert init_server.call_args.kwargs["api_key"] == "test-key"
        assert uvicorn_run.call_args.kwargs["host"] == "0.0.0.0"
    finally:
        reset_settings()


def test_init_server_rejects_network_bind_without_api_key(tmp_path):
    from omlx import server
    from omlx.settings import GlobalSettings

    settings = GlobalSettings(base_path=tmp_path)
    settings.server.host = "0.0.0.0"

    with pytest.raises(ValueError, match="API key is required"):
        server.init_server(
            model_dirs=str(tmp_path),
            api_key=None,
            global_settings=settings,
        )


def test_main_wires_global_settings(module_entry):
    # The admin routes resolve settings via _server_state.global_settings;
    # None here is what turned the API-key setup form into a 500 (#2282).
    server, _ = module_entry
    assert server._server_state.global_settings is not None


def test_admin_api_key_setup_succeeds(module_entry):
    # Reporter's repro for #2282: boot via python -m, submit the initial
    # API-key setup form. Must not 500 on a missing GlobalSettings.
    from fastapi.testclient import TestClient

    server, _ = module_entry
    client = TestClient(server.app, client=("127.0.0.1", 50000))
    resp = client.post(
        "/admin/api/setup-api-key",
        json={"api_key": "test-key-1234", "api_key_confirm": "test-key-1234"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("success") is True
    assert server._server_state.api_key == "test-key-1234"


MODULE_ENTRY_PROBE = textwrap.dedent("""
    import inspect
    import json
    import runpy
    import sys

    import uvicorn
    from fastapi.testclient import TestClient

    KEY = "e2e-key-1234"
    run_signature = inspect.signature(uvicorn.run)


    def probe(app, **kwargs):
        run_signature.bind(app, **kwargs)
        setup = TestClient(app, client=("127.0.0.1", 50000)).post(
            "/admin/api/setup-api-key",
            json={"api_key": KEY, "api_key_confirm": KEY},
        )
        client = TestClient(app, follow_redirects=False)
        html = {"Accept": "text/html"}
        status = {
            "setup": [setup.status_code, setup.json().get("success")],
            "login_page": client.get("/admin", headers=html).status_code,
            "dashboard": client.get("/admin/dashboard", headers=html).status_code,
            "static": client.get("/admin/static/favicon.svg").status_code,
            "auto_login": client.get(
                "/admin/auto-login", params={"key": KEY}
            ).status_code,
            "api_html_no_session": client.get(
                "/admin/api/global-settings", headers=html
            ).status_code,
        }
        client.post("/admin/api/login", json={"api_key": KEY})
        status["api_with_session"] = client.get(
            "/admin/api/global-settings"
        ).status_code
        print(json.dumps(status))


    uvicorn.run = probe
    runpy.run_module("omlx.server", run_name="__main__", alter_sys=True)
    """)


def test_headless_module_entry_end_to_end(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    env = {**os.environ, "OMLX_HEADLESS": "1", "HOME": str(tmp_path)}
    env["OMLX_BASE_PATH"] = str(tmp_path / "base")
    result = subprocess.run(
        [sys.executable, "-c", MODULE_ENTRY_PROBE, "--model-dir", str(model_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    status = json.loads(result.stdout.strip().splitlines()[-1])
    assert status == {
        "setup": [200, True],
        "login_page": 404,
        "dashboard": 404,
        "static": 404,
        "auto_login": 404,
        "api_html_no_session": 401,
        "api_with_session": 200,
    }


@pytest.fixture
def wired_downloaders(monkeypatch, tmp_path):
    """Boot the real init_server() wiring with both downloaders present."""
    from omlx import server
    from omlx.settings import reset_settings

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "omlx-base"))
    reset_settings()
    server.app.middleware_stack = None

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    argv = ["omlx.server", "--model-dir", str(model_dir)]
    with (
        patch.object(sys, "argv", argv),
        patch("mlx.core.set_cache_limit"),
        # The MS downloader is optional and its SDK is not installed here;
        # the wiring under test only needs the constructor to run.
        patch("omlx.admin.ms_downloader.MS_SDK_AVAILABLE", True),
        patch("uvicorn.run"),
    ):
        server.main()

    from omlx.admin.routes import set_admin_getters

    set_admin_getters(
        server.get_server_state,
        server.get_engine_pool,
        lambda: server._server_state.settings_manager,
        lambda: server._server_state.global_settings,
    )

    yield server
    reset_settings()


def test_download_queues_are_wired_for_restart_resume(wired_downloaders):
    """Pin the wiring that makes restart-resume reachable end to end.

    init_server() chooses the two queue file names and the lifespan awaits
    restore_tasks() on each downloader. Either half missing silently turns a
    restart into an empty queue, and nothing else in the suite covers the
    lifespan's startup half.
    """
    from fastapi.testclient import TestClient

    server = wired_downloaders
    base = Path(server._server_state.global_settings.base_path)
    hf = server._server_state.hf_downloader
    ms = server._server_state.ms_downloader
    assert hf is not None and ms is not None
    assert hf._tasks_file == base / "hf_download_tasks.json"
    assert ms._tasks_file == base / "ms_download_tasks.json"

    hf.restore_tasks = AsyncMock()
    ms.restore_tasks = AsyncMock()
    with (
        patch("omlx.utils.network.detect_server_aliases", return_value=[]),
        TestClient(server.app),
    ):
        pass

    hf.restore_tasks.assert_awaited_once()
    ms.restore_tasks.assert_awaited_once()
