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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from unittest.mock import patch

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
    client = TestClient(server.app)
    resp = client.post(
        "/admin/api/setup-api-key",
        json={"api_key": "test-key-1234", "api_key_confirm": "test-key-1234"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json().get("success") is True
    assert server._server_state.api_key == "test-key-1234"


def test_module_entry_api_key_setup_end_to_end(tmp_path):
    # The double-import layer of #2282 only exists when the module runs
    # as ``__main__``, so this must be a real ``python -m omlx.server``
    # subprocess; every in-process test is structurally blind to it.
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["OMLX_BASE_PATH"] = str(tmp_path / "base")

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "omlx.server",
            "--model-dir",
            str(model_dir),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                pytest.fail(f"server exited early:\n{proc.stdout.read()}")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=1)
                break
            except (urllib.error.URLError, OSError):
                time.sleep(0.25)
        else:
            pytest.fail("server did not become healthy within 60s")

        req = urllib.request.Request(
            f"{base}/admin/api/setup-api-key",
            data=json.dumps(
                {"api_key": "e2e-key-1234", "api_key_confirm": "e2e-key-1234"}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as e:
            pytest.fail(f"setup-api-key returned {e.code}: {e.read().decode()}")
        assert resp.status == 200
        assert json.load(resp).get("success") is True
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)
