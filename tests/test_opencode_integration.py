import json
import threading
from pathlib import Path
from types import SimpleNamespace

from omlx.integrations.base import IntegrationContext
from omlx.integrations.opencode import OpenCodeIntegration


def make_context(model: str) -> IntegrationContext:
    return IntegrationContext(
        host="127.0.0.1",
        port=8000,
        api_key="test-key",
        model=model,
        context_window=32768,
        max_tokens=4096,
    )


def test_launch_does_not_modify_persistent_config(monkeypatch, tmp_path):
    config_path = tmp_path / "opencode.json"
    original = {
        "model": "anthropic/claude-sonnet",
        "mcp": {"example": {"type": "remote"}},
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")

    integration = OpenCodeIntegration()
    monkeypatch.setattr(integration, "CONFIG_PATH", config_path)

    captured = {}

    def fake_run(args, env, check):
        captured["args"] = args
        captured["config"] = json.loads(
            Path(env["OPENCODE_CONFIG"]).read_text(encoding="utf-8")
        )
        captured["config_path"] = env["OPENCODE_CONFIG"]
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("omlx.integrations.opencode.subprocess.run", fake_run)

    try:
        integration.launch(make_context("model-a"))
    except SystemExit as exc:
        assert exc.code == 0

    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert captured["config"]["model"] == "omlx/model-a"
    assert captured["config"]["mcp"] == original["mcp"]
    assert captured["config_path"] != str(config_path)
    assert not list(tmp_path.glob("opencode.*.bak"))


def test_concurrent_launches_use_independent_configs(monkeypatch, tmp_path):
    config_path = tmp_path / "opencode.json"
    original = {
        "model": "anthropic/claude-sonnet",
        "mcp": {"example": {"type": "remote"}},
    }
    config_path.write_text(json.dumps(original), encoding="utf-8")

    integration = OpenCodeIntegration()
    monkeypatch.setattr(integration, "CONFIG_PATH", config_path)

    barrier = threading.Barrier(2)
    launches = []
    errors = []

    def fake_run(args, env, check):
        config = json.loads(
            Path(env["OPENCODE_CONFIG"]).read_text(encoding="utf-8")
        )
        launches.append((env["OPENCODE_CONFIG"], config))
        barrier.wait(timeout=5)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("omlx.integrations.opencode.subprocess.run", fake_run)

    def worker(model: str):
        try:
            integration.launch(make_context(model))
        except SystemExit as exc:
            if exc.code != 0:
                errors.append(exc)
        except Exception as exc:  # pragma: no cover - defensive test reporting
            errors.append(exc)

    threads = [
        threading.Thread(target=worker, args=("model-a",)),
        threading.Thread(target=worker, args=("model-b",)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert len(launches) == 2
    assert launches[0][0] != launches[1][0]
    assert {item[1]["model"] for item in launches} == {
        "omlx/model-a",
        "omlx/model-b",
    }
    assert all(item[1]["mcp"] == original["mcp"] for item in launches)
    assert json.loads(config_path.read_text(encoding="utf-8")) == original
    assert not list(tmp_path.glob("opencode.*.bak"))
