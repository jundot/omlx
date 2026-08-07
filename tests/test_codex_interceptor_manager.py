"""Lifecycle, privacy, and configuration invariants for native Codex routing."""

import json
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from omlx.codex_interceptor.manager import (
    CodexInterceptorConfig,
    CodexInterceptorManager,
)


class FakeProcess:
    pid = 424242
    returncode = None

    def __init__(self):
        self.exited = threading.Event()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.exited.wait(timeout)
        return self.returncode


def config(tmp_path: Path) -> CodexInterceptorConfig:
    return CodexInterceptorConfig(
        model="local/model",
        upstream_url="http://127.0.0.1:8000/v1/responses",
        project=tmp_path,
        launch_app=False,
    )


def test_managed_start_and_stop_never_modify_codex_config(tmp_path):
    codex_config = tmp_path / ".codex" / "config.toml"
    codex_config.parent.mkdir()
    original = b"[features]\nmulti_agent = true\n"
    codex_config.write_bytes(original)
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")
    proxy = FakeProcess()

    with (
        patch("omlx.codex_interceptor.manager._CODEX_CONFIG", codex_config),
        patch.object(manager, "_check_upstream"),
        patch.object(manager, "_resolve_proxy_command", return_value=["mitmdump"]),
        patch.object(manager, "_start_proxy", return_value=proxy),
        patch("omlx.codex_interceptor.manager._terminate"),
    ):
        status = manager.start(config(tmp_path))
        assert status["running"] is True
        assert status["config_modified"] is False
        stopped = manager.stop()

    assert stopped["phase"] == "stopped"
    assert (tmp_path / "runtime").stat().st_mode & 0o777 == 0o700
    assert stopped["config_modified"] is False
    assert codex_config.read_bytes() == original
    assert list(codex_config.parent.glob("*.bak")) == []


def test_initial_status_does_not_report_existing_config_as_modified(tmp_path):
    codex_config = tmp_path / "config.toml"
    codex_config.write_text("[features]\n")
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")

    with patch("omlx.codex_interceptor.manager._CODEX_CONFIG", codex_config):
        assert manager.status()["config_modified"] is False


def test_status_aggregates_only_safe_metadata(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path)
    status_path = tmp_path / "status.jsonl"
    status_path.write_text(
        '{"event":"local_slot_recovered","time":0,"local_slot":"gpt-5-codex"}\n'
        '{"event":"inference_routed","time":1,"local_model":"qwen",'
        '"request_bytes":123,"input":"secret prompt"}\n'
        '{"event":"inference_first_visible_event","time":2,"first_visible_ms":42}\n'
        '{"event":"inference_completed","time":3,"duration_ms":88,'
        '"output_tokens_per_second":9.5,"output":"secret answer"}\n'
        '{"event":"remote_inference_routed","time":4,"requested_model":"gpt-cloud",'
        '"authorization":"secret token"}\n'
    )
    manager._status_path = status_path
    manager._config = config(tmp_path)

    status = manager.status()

    assert status["local_requests"] == 1
    assert status["local_slot"] == "gpt-5-codex"
    assert status["cloud_requests"] == 1
    assert status["completed_requests"] == 1
    assert status["active_local_requests"] == 0
    assert status["latest_metrics"]["first_visible_ms"] == 42
    assert status["latest_metrics"]["tokens_per_second"] == 9.5
    serialized = repr(status["recent_events"])
    assert "secret prompt" not in serialized
    assert "secret answer" not in serialized
    assert "secret token" not in serialized


def test_codex_child_environment_is_process_scoped(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path)
    session = tmp_path / "session"
    cert_dir = session / "mitmproxy"
    cert_dir.mkdir(parents=True)
    (cert_dir / "mitmproxy-ca-cert.pem").write_text("LOCAL CA\n")
    manager._session_dir = session
    manager._listen_port = 9146

    with patch.dict(
        "os.environ", {"CODEX_HOME": "/custom", "PATH": "/usr/bin"}, clear=True
    ):
        env = manager._codex_environment()

    assert env["HTTPS_PROXY"] == "http://127.0.0.1:9146"
    assert env["NODE_EXTRA_CA_CERTS"].endswith("mitmproxy-ca-cert.pem")
    assert "127.0.0.1" in env["NO_PROXY"]
    assert "CODEX_HOME" not in env
    assert not (tmp_path / ".codex" / "config.toml").exists()


def test_proxy_child_drops_stale_interceptor_and_audit_environment(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")
    session = tmp_path / "session"
    session.mkdir()
    status_path = session / "status.jsonl"
    status_path.touch()
    route_path = session / "route.json"
    route_path.write_text("{}")
    captured = {}

    def fake_popen(_argv, **kwargs):
        captured["env"] = kwargs["env"]
        cert_dir = session / "mitmproxy"
        (cert_dir / "mitmproxy-ca-cert.pem").write_text("LOCAL CA\n")
        return FakeProcess()

    with (
        patch.dict(
            "os.environ",
            {
                "PATH": "/usr/bin",
                "HARNESS_INTERCEPTOR_MODEL": "stale",
                "OMLX_CODEX_INTERCEPTOR_GPT56_PRO": "1",
                "OMLX_CODEX_INTERCEPTOR_CLOUD_AUDIT_DIR": "/tmp/unsafe",
            },
            clear=True,
        ),
        patch(
            "omlx.codex_interceptor.manager.subprocess.Popen", side_effect=fake_popen
        ),
        patch("omlx.codex_interceptor.manager._port_is_open", return_value=True),
    ):
        manager._start_proxy(
            config=config(tmp_path),
            command=["mitmdump"],
            listen_port=9146,
            session_id="session-1",
            session_dir=session,
            status_path=status_path,
            route_path=route_path,
        )

    env = captured["env"]
    assert env["OMLX_CODEX_INTERCEPTOR_MODEL"] == "local/model"
    assert "HARNESS_INTERCEPTOR_MODEL" not in env
    assert "OMLX_CODEX_INTERCEPTOR_GPT56_PRO" not in env
    assert "OMLX_CODEX_INTERCEPTOR_CLOUD_AUDIT_DIR" not in env
    assert env["OMLX_CODEX_INTERCEPTOR_ROUTE_PATH"] == str(route_path)


def test_model_switch_is_warmed_before_atomic_next_turn_publish(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")
    proxy = FakeProcess()
    initial = replace(config(tmp_path), context_window=32768)

    with (
        patch.object(manager, "_check_upstream"),
        patch.object(manager, "_resolve_proxy_command", return_value=["mitmdump"]),
        patch.object(manager, "_start_proxy", return_value=proxy),
        patch("omlx.codex_interceptor.manager._terminate"),
    ):
        started = manager.start(initial)
        with pytest.raises(RuntimeError, match="finish loading"):
            manager.begin_model_switch("local/larger", context_window=65536)
        manager.set_warmup_status(started["session_id"], "ready", "local/model")
        route_path = manager._route_path
        assert route_path is not None
        assert route_path.stat().st_mode & 0o777 == 0o600
        before = json.loads(route_path.read_text())

        generation, loading = manager.begin_model_switch(
            "local/larger",
            context_window=65536,
        )
        manager.set_warmup_status(
            started["session_id"],
            "ready",
            "local/model",
        )
        assert loading["active_model"] == "local/model"
        assert loading["model_switch_loading"] is True
        assert manager.status()["warmup_model"] == "local/larger"
        assert manager.status()["warmup_status"] == "loading"
        assert json.loads(route_path.read_text()) == before

        queued = manager.complete_model_switch(generation)
        published = json.loads(route_path.read_text())
        assert published["model"] == "local/larger"
        assert published["revision"] == before["revision"] + 1
        assert queued["active_model"] == "local/model"
        assert queued["pending_model"] == "local/larger"

        manager._status_path.write_text(
            json.dumps(
                {
                    "event": "local_model_changed",
                    "local_model": "local/larger",
                    "advertised_context_window": 65536,
                    "route_revision": published["revision"],
                }
            )
            + "\n"
        )
        active = manager.status()

    assert active["active_model"] == "local/larger"
    assert active["active_context_window"] == 65536
    assert active["pending_model"] is None
    assert active["model_switching"] is False


@pytest.mark.parametrize("target_context", [None, 16384])
def test_model_switch_refuses_unknown_or_smaller_context(tmp_path, target_context):
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")
    proxy = FakeProcess()
    initial = replace(config(tmp_path), context_window=32768)

    with (
        patch.object(manager, "_check_upstream"),
        patch.object(manager, "_resolve_proxy_command", return_value=["mitmdump"]),
        patch.object(manager, "_start_proxy", return_value=proxy),
        patch("omlx.codex_interceptor.manager._terminate"),
    ):
        started = manager.start(initial)
        manager.set_warmup_status(started["session_id"], "ready", "local/model")
        with pytest.raises(RuntimeError, match="fresh Codex session"):
            manager.begin_model_switch(
                "local/unsafe",
                context_window=target_context,
            )


def test_stopping_invalidates_an_in_progress_model_load(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path / "runtime")
    proxy = FakeProcess()
    initial = replace(config(tmp_path), context_window=32768)

    with (
        patch.object(manager, "_check_upstream"),
        patch.object(manager, "_resolve_proxy_command", return_value=["mitmdump"]),
        patch.object(manager, "_start_proxy", return_value=proxy),
        patch("omlx.codex_interceptor.manager._terminate"),
    ):
        started = manager.start(initial)
        manager.set_warmup_status(started["session_id"], "ready", "local/model")
        generation, _ = manager.begin_model_switch(
            "local/larger",
            context_window=65536,
        )
        manager.stop()
        with pytest.raises(RuntimeError, match="no longer current"):
            manager.complete_model_switch(generation)


def test_proxy_crash_closes_managed_codex_and_surfaces_error(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path)
    proxy = FakeProcess()
    opener = FakeProcess()
    proxy.returncode = 9
    proxy.exited.set()
    manager._proxy = proxy
    manager._opener = opener
    manager._app_pid = 1234
    manager._listen_port = 9146

    with (
        patch.object(manager, "_quit_pids") as quit_pids,
        patch("omlx.codex_interceptor.manager._terminate") as terminate,
    ):
        manager._watch_proxy_exit(proxy)

    assert manager.status()["phase"] == "error"
    assert "status 9" in manager.status()["error"]
    quit_pids.assert_called_once_with([1234])
    terminate.assert_called_once_with(opener, timeout=2)


def test_codex_exit_reaps_proxy(tmp_path):
    manager = CodexInterceptorManager(runtime_root=tmp_path)
    proxy = FakeProcess()
    opener = FakeProcess()
    opener.returncode = 0
    opener.exited.set()
    manager._proxy = proxy
    manager._opener = opener
    manager._app_pid = 1234
    manager._listen_port = 9146

    with patch("omlx.codex_interceptor.manager._terminate") as terminate:
        manager._watch_codex_exit(opener)

    status = manager.status()
    assert status["phase"] == "stopped"
    assert status["running"] is False
    terminate.assert_called_once_with(proxy)
