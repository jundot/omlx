# SPDX-License-Identifier: Apache-2.0
"""The settings-file fallback (review B4): never wipe a Mac's model settings.

``write_model_settings_on_mac`` falls back to writing the per-model JSON file
when the settings route answers 404 (a staged model the pool does not list yet).
That fallback used to hardcode ``~/.omlx``, reset the whole document on *any*
read error, and share one fixed ``.tmp`` name. These tests pin the fix:

* the path comes from the real oMLX configuration (a moved data root is honoured),
  both in this process and in the worker snippet;
* a missing file may start a fresh document, but a corrupt or unreadable file
  aborts the write and is left byte-for-byte unchanged;
* the atomic swap uses a unique temp name, and concurrent local writes keep
  every model entry.

Everything is fake-driven: the local branch runs against a tmp path and the
remote snippet is executed in-process with a sandboxed base path, so no test
reaches a real Mac, the real ``~/.omlx`` or the network.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from omlx.cluster.assistant import provisioning
from omlx.cluster.assistant.provisioning import (
    NodeTarget,
    ProvisioningError,
    _write_model_settings_file,
)


def _local_target() -> NodeTarget:
    return NodeTarget(
        node_id="m5", address="127.0.0.1", ssh="", local=True, port=8099
    )


def _remote_target() -> NodeTarget:
    return NodeTarget(
        node_id="m1",
        address="192.0.2.14",
        ssh="user@192.0.2.14",
        local=False,
        port=8099,
    )


@pytest.fixture()
def local_file(tmp_path, monkeypatch):
    """Point the canonical resolver at a tmp settings file (local branch)."""

    path = tmp_path / "model_settings.json"
    monkeypatch.setitem(provisioning.__dict__, "_local_model_settings_path", lambda: path)
    return path


def _run_snippet(monkeypatch, tmp_path, payload: dict) -> str:
    """Execute the worker snippet in-process against a sandboxed base path.

    ``HOME`` is moved so the *old* hardcoded ``~/.omlx`` could never touch the
    real home either; ``OMLX_BASE_PATH`` is what the canonical resolver reads.
    The snippet's canonical file is therefore ``<tmp>/base/model_settings.json``.
    """

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "base"))
    monkeypatch.setitem(sys.__dict__, "argv", ["model-settings", json.dumps(payload)])
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(  # noqa: S102 - the shipped snippet itself, sandboxed above
            compile(provisioning._SETTINGS_FILE_SNIPPET, "<settings-snippet>", "exec"),
            {},
        )
    return out.getvalue()


def _snippet_file(tmp_path) -> Path:
    return tmp_path / "base" / "model_settings.json"


# ---------------------------------------------------------------------------
# Local branch: missing / merge / corrupt / unreadable
# ---------------------------------------------------------------------------


def test_local_missing_file_is_created_with_the_patch(local_file) -> None:
    assert not local_file.exists()

    _write_model_settings_file(_local_target(), "qwen", {"mtp_enabled": True})

    document = json.loads(local_file.read_text())
    assert document["version"] == 1
    assert document["models"]["qwen"]["mtp_enabled"] is True


def test_local_merge_preserves_other_models_and_keys(local_file) -> None:
    local_file.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "other": {"temperature": 0.2},
                    "qwen": {"top_p": 0.9},
                },
            }
        )
    )

    _write_model_settings_file(_local_target(), "qwen", {"mtp_enabled": True})

    document = json.loads(local_file.read_text())
    assert document["models"]["other"] == {"temperature": 0.2}
    # entry.update(patch): the existing key survives, the new one is added.
    assert document["models"]["qwen"] == {"top_p": 0.9, "mtp_enabled": True}


def test_local_corrupt_json_aborts_and_leaves_bytes_unchanged(local_file) -> None:
    corrupt = b'{"version": 1, "models": {'
    local_file.write_bytes(corrupt)

    with pytest.raises(ProvisioningError):
        _write_model_settings_file(_local_target(), "qwen", {"mtp_enabled": True})

    assert local_file.read_bytes() == corrupt


def test_local_permission_denied_aborts_and_leaves_file_unchanged(local_file) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses file permissions")
    local_file.write_text(
        json.dumps({"version": 1, "models": {"other": {"temperature": 0.2}}})
    )
    original = local_file.read_bytes()
    local_file.chmod(0o000)
    try:
        # Prove the denial actually occurs in this environment before asserting
        # the fallback respects it.
        try:
            local_file.read_text()
        except PermissionError:
            pass
        else:
            pytest.skip("cannot simulate a permission-denied read here")
        with pytest.raises(ProvisioningError):
            _write_model_settings_file(
                _local_target(), "qwen", {"mtp_enabled": True}
            )
    finally:
        local_file.chmod(0o600)
    assert local_file.read_bytes() == original


def test_local_resolver_honors_a_moved_base_path(tmp_path, monkeypatch) -> None:
    import omlx.settings as settings_module

    # An uninitialized process resolves through GlobalSettings.load(), which
    # reads OMLX_BASE_PATH ahead of the ~/.omlx default.
    monkeypatch.setitem(settings_module.__dict__, "_global_settings", None)
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path / "moved"))

    path = provisioning._local_model_settings_path()
    assert path == tmp_path / "moved" / "model_settings.json"

    _write_model_settings_file(_local_target(), "qwen", {"mtp_enabled": True})
    assert path.exists()
    assert json.loads(path.read_text())["models"]["qwen"]["mtp_enabled"] is True


def test_concurrent_local_writes_keep_every_model_entry(local_file) -> None:
    count = 8
    barrier = threading.Barrier(count)

    def write(index: int) -> None:
        barrier.wait()
        _write_model_settings_file(
            _local_target(), f"model-{index}", {"temperature": index / 10}
        )

    with ThreadPoolExecutor(max_workers=count) as pool:
        list(pool.map(write, range(count)))

    document = json.loads(local_file.read_text())
    assert set(document["models"]) == {f"model-{i}" for i in range(count)}
    for index in range(count):
        assert document["models"][f"model-{index}"]["temperature"] == index / 10


# ---------------------------------------------------------------------------
# Remote snippet: canonical path, abort-not-wipe, unique temp
# ---------------------------------------------------------------------------


def test_snippet_resolves_the_canonical_path_not_the_hardcoded_default(
    tmp_path, monkeypatch
) -> None:
    stdout = _run_snippet(
        monkeypatch, tmp_path, {"model_id": "qwen", "patch": {"mtp_enabled": True}}
    )

    assert json.loads(stdout.strip().splitlines()[-1]) == {"ok": True}
    document = json.loads(_snippet_file(tmp_path).read_text())
    assert document["models"]["qwen"]["mtp_enabled"] is True
    # The hardcoded ~/.omlx (under the sandboxed HOME) must not be used.
    assert not (tmp_path / "home" / ".omlx" / "model_settings.json").exists()


def test_snippet_merges_and_preserves_other_models(tmp_path, monkeypatch) -> None:
    path = _snippet_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"version": 1, "models": {"other": {"temperature": 0.2}}})
    )

    _run_snippet(
        monkeypatch, tmp_path, {"model_id": "qwen", "patch": {"mtp_enabled": True}}
    )

    document = json.loads(path.read_text())
    assert document["models"]["other"] == {"temperature": 0.2}
    assert document["models"]["qwen"]["mtp_enabled"] is True


def test_snippet_aborts_on_corrupt_json_without_wiping(tmp_path, monkeypatch) -> None:
    path = _snippet_file(tmp_path)
    path.parent.mkdir(parents=True)
    corrupt = b'{"version": 1, "models": {'
    path.write_bytes(corrupt)

    with pytest.raises(Exception):  # noqa: B017 - any loud failure is the point
        _run_snippet(
            monkeypatch, tmp_path, {"model_id": "qwen", "patch": {"mtp_enabled": True}}
        )

    assert path.read_bytes() == corrupt


def test_snippet_aborts_on_permission_denied(tmp_path, monkeypatch) -> None:
    if os.geteuid() == 0:
        pytest.skip("root bypasses file permissions")
    path = _snippet_file(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"version": 1, "models": {"other": {}}}))
    original = path.read_bytes()
    path.chmod(0o000)
    try:
        try:
            path.read_text()
        except PermissionError:
            pass
        else:
            pytest.skip("cannot simulate a permission-denied read here")
        with pytest.raises(Exception):  # noqa: B017 - any loud failure is the point
            _run_snippet(
                monkeypatch,
                tmp_path,
                {"model_id": "qwen", "patch": {"mtp_enabled": True}},
            )
    finally:
        path.chmod(0o600)
    assert path.read_bytes() == original


def test_snippet_uses_canonical_settings_and_a_unique_temp_name() -> None:
    snippet = provisioning._SETTINGS_FILE_SNIPPET
    assert "GlobalSettings.load()" in snippet
    assert "~/.omlx" not in snippet
    assert "uuid.uuid4().hex" in snippet
    assert "os.replace" in snippet
    # Only a missing file may start a fresh document; a blanket reset is the bug.
    assert "except FileNotFoundError" in snippet
    assert "except Exception" not in snippet


def test_remote_write_still_goes_through_run_remote_python(monkeypatch) -> None:
    import omlx.cluster.staging as staging

    captured: dict = {}

    def fake_run(ssh, snippet, argument, *, description, python_executable, timeout):
        captured.update(ssh=ssh, snippet=snippet, argument=argument)
        return {"ok": True}

    monkeypatch.setitem(staging.__dict__, "run_remote_python", fake_run)

    _write_model_settings_file(_remote_target(), "qwen", {"mtp_enabled": True})

    assert captured["ssh"] == "user@192.0.2.14"
    assert captured["snippet"] is provisioning._SETTINGS_FILE_SNIPPET
    payload = json.loads(captured["argument"])
    assert payload["model_id"] == "qwen"
    assert payload["patch"] == {"mtp_enabled": True}


def test_the_mac_wide_tier_write_uses_the_global_settings_route(monkeypatch) -> None:
    """The tier is Mac-wide, so it goes to ``/admin/api/global-settings``.

    It must also work without a model id: that is what lets apply write it
    before staging, where it changes the ceiling the launch preflight reads.
    """

    from omlx.cluster.assistant import start

    calls: list[tuple[str, str, object, str]] = []

    def fake_local(port, path, body=None, *, method="GET", timeout=30.0):
        calls.append(("local", path, body, method))
        return {"status": "success"}

    def fake_remote(ssh, port, path, body, *, method="POST", timeout=120.0):
        calls.append((ssh, path, body, method))
        return {"status": "success"}

    monkeypatch.setitem(start.__dict__, "_local_admin", fake_local)
    monkeypatch.setitem(start.__dict__, "remote_admin", fake_remote)

    provisioning.write_global_settings_on_mac(
        _local_target(), {"memory_guard_tier": "aggressive"}
    )
    provisioning.write_global_settings_on_mac(
        _remote_target(), {"memory_guard_tier": "aggressive"}
    )

    assert calls == [
        ("local", start.GLOBAL_SETTINGS_PATH, {"memory_guard_tier": "aggressive"}, "POST"),
        (
            "user@192.0.2.14",
            start.GLOBAL_SETTINGS_PATH,
            {"memory_guard_tier": "aggressive"},
            "POST",
        ),
    ]


def test_a_mac_wide_write_without_a_port_is_one_action() -> None:
    target = NodeTarget(node_id="m1", address="192.0.2.14", ssh="u@h", local=False, port=0)

    with pytest.raises(ProvisioningError) as raised:
        provisioning.write_global_settings_on_mac(target, {"memory_guard_tier": "safe"})

    assert raised.value.code == "configure_endpoint"
    assert raised.value.action == "configure_endpoint"
