# SPDX-License-Identifier: Apache-2.0
"""Eviction of competing standalone models after a memory-attributed failure.

A cluster activation that dies of ``InsufficientMemoryError`` on a rank may
have been competing with a model that node's own oMLX server loaded through
its normal path. These tests cover the SSH-side report (`launch`), the
peer-side loopback script itself, and the routes-layer recovery that frames
the retry message and incidents.
"""

import asyncio
import contextlib
import io
import json
import subprocess

import pytest

from omlx.cluster import launch, routes
from omlx.cluster.deployment import ClusterHost
from omlx.cluster.incidents import IncidentStore
from omlx.cluster.launch import DistributedLaunchError, evict_remote_local_models
from types import SimpleNamespace


# ---------------------------------------------------------------------------
# evict_remote_local_models (coordinator -> peer SSH boundary)
# ---------------------------------------------------------------------------


def _completed(argv, returncode, stdout, stderr=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, stderr)


def test_remote_eviction_returns_peer_report():
    report = {
        "server_reachable": True,
        "evicted": ["qwen3-8b"],
        "draining": [],
        "skipped_pinned": ["gemma-fast"],
        "errors": [],
    }
    captured = {}

    def runner(argv, **kwargs):
        captured["argv"] = argv
        return _completed(argv, 0, json.dumps(report) + "\n")

    payload = evict_remote_local_models(
        "peer.local",
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert payload == report
    remote_command = captured["argv"][-1]
    assert remote_command.startswith("/opt/omlx/bin/python")
    assert "/admin/api/models" in remote_command
    assert captured["argv"][-2] == "peer.local"


def test_remote_eviction_falls_back_to_discovered_interpreter():
    calls = []

    def runner(argv, **kwargs):
        remote_command = argv[-1]
        calls.append(remote_command)
        if remote_command.startswith("/stale/python"):
            return _completed(argv, 1, "", "No such file or directory")
        if "/admin/api/models" in remote_command:
            return _completed(argv, 0, json.dumps({"evicted": [], "errors": []}))
        # Interpreter discovery probes answer with the resolved path.
        return _completed(argv, 0, "/opt/omlx/bin/python\n")

    payload = evict_remote_local_models(
        "peer.local",
        python_executable="/stale/python",
        runner=runner,
    )

    assert payload["evicted"] == []
    assert calls[-1].startswith("/opt/omlx/bin/python")


def test_remote_eviction_rejects_non_json_report():
    def runner(argv, **kwargs):
        remote_command = argv[-1]
        if "/admin/api/models" in remote_command:
            return _completed(argv, 0, "Segmentation fault")
        return _completed(argv, 0, "/opt/omlx/bin/python\n")

    with pytest.raises(DistributedLaunchError, match="eviction report"):
        evict_remote_local_models(
            "peer.local",
            python_executable="/opt/omlx/bin/python",
            runner=runner,
        )


# ---------------------------------------------------------------------------
# The peer-side loopback script (runs against the peer's own admin API)
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()

    def read(self, *args):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _run_peer_script(monkeypatch, tmp_path, settings, responder):
    import urllib.request

    (tmp_path / "settings.json").write_text(json.dumps(settings))
    monkeypatch.setenv("OMLX_BASE_PATH", str(tmp_path))
    requests = []

    def fake_urlopen(request, timeout=None):
        requests.append(request)
        return _FakeResponse(responder(request))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        with pytest.raises(SystemExit):
            exec(compile(launch._REMOTE_LOCAL_EVICTION, "<eviction>", "exec"), {})
    return json.loads(stdout.getvalue()), requests


def test_peer_script_unloads_only_unpinned_standalone_models(
    monkeypatch, tmp_path
):
    models = [
        {"id": "qwen3-8b", "loaded": True, "pinned": False},
        {"id": "gemma-fast", "loaded": True, "pinned": True},
        {"id": "cluster-shard", "loaded": True, "source_type": "cluster"},
        {"id": "markitdown", "loaded": True, "virtual": True},
        {"id": "cold-model", "loaded": False},
        {"id": "half-loaded", "loaded": True, "is_loading": True},
    ]

    def responder(request):
        if request.get_method() == "GET":
            return {"models": models}
        assert "/admin/api/models/qwen3-8b/unload" in request.full_url
        return {"status": "ok", "model_id": "qwen3-8b"}

    report, requests = _run_peer_script(
        monkeypatch,
        tmp_path,
        {"server": {"port": 8123}, "auth": {"secret_key": "s" * 32}},
        responder,
    )

    assert report["evicted"] == ["qwen3-8b"]
    assert report["skipped_pinned"] == ["gemma-fast"]
    assert report["draining"] == []
    assert report["errors"] == []
    assert report["server_reachable"] is True
    # Every call targets the configured port on loopback with a session cookie
    # signed by the persisted secret.
    from itsdangerous import URLSafeTimedSerializer

    for request in requests:
        assert request.full_url.startswith("http://127.0.0.1:8123/")
        token = request.get_header("Cookie").split("=", 1)[1]
        data = URLSafeTimedSerializer("s" * 32).loads(token, max_age=60)
        assert data == {"admin": True, "remember": False}


def test_peer_script_reports_draining_and_unreachable_server(
    monkeypatch, tmp_path
):
    def responder(request):
        if request.get_method() == "GET":
            return {"models": [{"id": "busy-model", "loaded": True}]}
        return {"status": "unloading", "model_id": "busy-model"}

    report, _ = _run_peer_script(
        monkeypatch,
        tmp_path,
        {"server": {"port": 8000}, "auth": {"secret_key": "s" * 32}},
        responder,
    )
    assert report["draining"] == ["busy-model"]
    assert report["evicted"] == []

    def refused(request):
        raise OSError("connection refused")

    report, _ = _run_peer_script(
        monkeypatch,
        tmp_path,
        {"server": {"port": 8000}, "auth": {"secret_key": "s" * 32}},
        refused,
    )
    assert report["server_reachable"] is False
    assert report["evicted"] == []


# ---------------------------------------------------------------------------
# Routes-layer recovery
# ---------------------------------------------------------------------------


LOCAL_HOST = ClusterHost(node_id="mini", ssh="127.0.0.1", ips=("172.16.99.1",))
PEER_HOST = ClusterHost(
    node_id="M-FJX1D769D0.local",
    ssh="aphoenix@mbp.local",
    ips=("172.16.99.2",),
    python_executable="/opt/omlx/bin/python",
)

FAILURE_DETAIL = (
    "Cluster readiness check failed: distributed launcher exited with code 0: "
    "rank 1 (M-FJX1D769D0.local): InsufficientMemoryError: rank 1 reached "
    "20.4 GiB while loading, above the 18.8 GiB it was admitted at"
)


def _deployment(hosts=(LOCAL_HOST, PEER_HOST)):
    return SimpleNamespace(deployment_id="dep-1", hosts=tuple(hosts))


def _incident_store(tmp_path, monkeypatch):
    from omlx.cluster import incidents as incidents_module

    store = IncidentStore(tmp_path)
    monkeypatch.setattr(incidents_module, "_configured_incidents", store)
    return store


class FakeEntry:
    def __init__(self, *, loaded=True, pinned=False, loading=False, source="local"):
        self.engine = object() if loaded else None
        self.is_pinned = pinned
        self.is_loading = loading
        self.source_type = source


class FakePool:
    def __init__(self, entries):
        self.entries = entries
        self.unloaded = []

    def get_loaded_model_ids(self):
        return [mid for mid, e in self.entries.items() if e.engine is not None]

    def get_entry(self, model_id):
        return self.entries.get(model_id)

    async def request_unload(self, model_id, *, reason):
        self.unloaded.append((model_id, reason))
        return True


def test_memory_squeezed_hosts_names_only_the_failing_rank():
    deployment = _deployment()
    hosts = routes._memory_squeezed_hosts(deployment, FAILURE_DETAIL)
    assert [host.node_id for host in hosts] == ["M-FJX1D769D0.local"]

    every = routes._memory_squeezed_hosts(
        deployment, "InsufficientMemoryError: budget exceeded"
    )
    assert [host.node_id for host in every] == ["mini", "M-FJX1D769D0.local"]

    assert routes._memory_squeezed_hosts(deployment, "rank 1 died of SIGHUP") == []


def test_recovery_evicts_peer_and_records_incident(tmp_path, monkeypatch):
    store = _incident_store(tmp_path, monkeypatch)
    remote_calls = []

    def fake_remote(ssh, *, python_executable=None):
        remote_calls.append((ssh, python_executable))
        return {
            "evicted": ["qwen3-8b"],
            "draining": [],
            "skipped_pinned": ["gemma-fast"],
            "errors": [],
        }

    monkeypatch.setattr(routes, "evict_remote_local_models", fake_remote)

    detail = asyncio.run(
        routes._evict_competing_local_models(_deployment(), FAILURE_DETAIL)
    )

    assert remote_calls == [("aphoenix@mbp.local", "/opt/omlx/bin/python")]
    assert "qwen3-8b on M-FJX1D769D0.local" in detail
    assert "retry the activation" in detail
    assert "Pinned model(s) gemma-fast on M-FJX1D769D0.local" in detail
    recorded = store.list()
    assert [incident.state_code for incident in recorded] == [
        "activation_memory_recovery"
    ]
    assert recorded[0].deployment_id == "dep-1"


def test_recovery_evicts_coordinator_models_through_the_pool(
    tmp_path, monkeypatch
):
    _incident_store(tmp_path, monkeypatch)
    pool = FakePool(
        {
            "local-a": FakeEntry(),
            "pinned-b": FakeEntry(pinned=True),
            "cluster-c": FakeEntry(source="cluster"),
            "loading-d": FakeEntry(loading=True),
        }
    )
    monkeypatch.setattr(routes, "_get_engine_pool", lambda: pool)

    detail = asyncio.run(
        routes._evict_competing_local_models(
            _deployment((LOCAL_HOST,)),
            "rank 0 (mini): InsufficientMemoryError: over budget",
        )
    )

    assert [model_id for model_id, _ in pool.unloaded] == ["local-a"]
    assert "failed for lack of memory" in pool.unloaded[0][1]
    assert "local-a on mini" in detail
    assert "pinned-b on mini" in detail


def test_recovery_failure_never_masks_the_original_error(tmp_path, monkeypatch):
    store = _incident_store(tmp_path, monkeypatch)

    def fake_remote(ssh, *, python_executable=None):
        raise DistributedLaunchError("SSH connection failed for mbp.local")

    monkeypatch.setattr(routes, "evict_remote_local_models", fake_remote)

    detail = asyncio.run(
        routes._evict_competing_local_models(
            _deployment((PEER_HOST,)), FAILURE_DETAIL
        )
    )

    assert detail.startswith(FAILURE_DETAIL)
    assert "could not complete everywhere" in detail
    assert [incident.state_code for incident in store.list()] == [
        "activation_memory_recovery_failed"
    ]


def test_recovery_ignores_failures_that_are_not_memory_shaped(
    tmp_path, monkeypatch
):
    store = _incident_store(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("no eviction may run for a non-memory failure")

    monkeypatch.setattr(routes, "evict_remote_local_models", explode)
    monkeypatch.setattr(routes, "_get_engine_pool", explode)

    detail = asyncio.run(
        routes._evict_competing_local_models(
            _deployment(), "distributed launcher exited with code 1: SIGHUP"
        )
    )

    assert detail == "distributed launcher exited with code 1: SIGHUP"
    assert not store.list()


def test_recovery_reports_when_nothing_was_loaded(tmp_path, monkeypatch):
    store = _incident_store(tmp_path, monkeypatch)

    def fake_remote(ssh, *, python_executable=None):
        return {
            "server_reachable": False,
            "evicted": [],
            "draining": [],
            "skipped_pinned": [],
            "errors": [],
        }

    monkeypatch.setattr(routes, "evict_remote_local_models", fake_remote)

    detail = asyncio.run(
        routes._evict_competing_local_models(_deployment((PEER_HOST,)), FAILURE_DETAIL)
    )

    assert "No competing local models were loaded" in detail
    # Nothing was freed and nothing failed, so no incident is warranted.
    assert not store.list()
