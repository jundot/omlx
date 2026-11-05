# SPDX-License-Identifier: Apache-2.0
"""Eviction of competing standalone models after a memory-attributed failure.

Reworked from a closed PR (jundot/omlx#2870) per maintainer feedback:
opt-in (off by default), and keyed on structured failure attribution
(RankFailure fields) instead of a regex over free-form error text. See
docs/cluster-competing-model-eviction-redesign.md for the full design.

These tests cover: the opt-in setting itself (default-off regression guard,
round-trip, and the gate-check's own fail-closed behavior), the structured
marker fields a rank writes on failure, DistributedJobSupervisor's
structured extraction of those fields, the SSH-side report (`launch`), the
peer-side loopback script itself, and the routes-layer recovery that frames
the retry message and incidents.
"""

import asyncio
import contextlib
import io
import json
import subprocess
from types import SimpleNamespace

import pytest

from omlx.cluster import launch, routes
from omlx.cluster.deployment import ClusterHost
from omlx.cluster.incidents import IncidentStore
from omlx.cluster.launch import DistributedLaunchError, RankFailure, evict_remote_local_models
from omlx.settings import ClusterSettings, GlobalSettings


# ---------------------------------------------------------------------------
# ClusterSettings — the opt-in flag
# ---------------------------------------------------------------------------


def test_cluster_settings_defaults_off():
    assert ClusterSettings().auto_evict_competing_local_models is False


def test_cluster_settings_round_trips_through_dict():
    settings = ClusterSettings(auto_evict_competing_local_models=True)
    restored = ClusterSettings.from_dict(settings.to_dict())
    assert restored.auto_evict_competing_local_models is True

    # Missing key (e.g. a settings.json written before this field existed)
    # must default to off, not raise.
    assert ClusterSettings.from_dict({}).auto_evict_competing_local_models is False


def test_global_settings_persists_cluster_section(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    settings.cluster.auto_evict_competing_local_models = True
    settings.save()

    reloaded = GlobalSettings(base_path=tmp_path)
    reloaded._load_from_file(tmp_path / "settings.json")
    assert reloaded.cluster.auto_evict_competing_local_models is True


def test_global_settings_to_dict_includes_cluster_section(tmp_path):
    settings = GlobalSettings(base_path=tmp_path)
    assert settings.to_dict()["cluster"] == {
        "auto_evict_competing_local_models": False
    }


# ---------------------------------------------------------------------------
# The gate check itself must fail closed, never mask the original failure
# ---------------------------------------------------------------------------


def test_auto_evict_enabled_reflects_the_setting(monkeypatch):
    from omlx import settings as settings_module

    off = GlobalSettings(base_path="/tmp")
    monkeypatch.setattr(settings_module, "_global_settings", off)
    assert routes._auto_evict_enabled() is False

    on = GlobalSettings(base_path="/tmp")
    on.cluster.auto_evict_competing_local_models = True
    monkeypatch.setattr(settings_module, "_global_settings", on)
    assert routes._auto_evict_enabled() is True


def test_auto_evict_enabled_fails_closed_when_settings_unavailable(monkeypatch):
    """get_settings() raises when init_settings() was never called (worker-
    only installs, some test apps). _auto_evict_enabled() must swallow that
    and return False — not let it propagate out of the DistributedLaunchError
    handler and replace the intended 503 with an unrelated 500. Regression
    test for a real bug caught while implementing this — see
    docs/cluster-competing-model-eviction-redesign.md §4 addendum.
    """
    from omlx import settings as settings_module

    monkeypatch.setattr(settings_module, "_global_settings", None)
    assert routes._auto_evict_enabled() is False


# ---------------------------------------------------------------------------
# Structured failure fields: rank marker -> RankFailure
# ---------------------------------------------------------------------------


def test_rank_marker_records_structured_error_type(tmp_path):
    from omlx.cluster.inference_worker import RuntimeMarker
    from omlx.cluster.liveness import read_marker
    from omlx.exceptions import InsufficientMemoryError

    marker = RuntimeMarker(
        state_dir=str(tmp_path),
        deployment_id="dep-1",
        rank=1,
        world_size=2,
        model="qwen3-35b",
        backend="ring",
        plan_hash="hash-1",
    )
    exc = InsufficientMemoryError(
        required=20_400_000_000,
        current=18_800_000_000,
        message="rank 1 reached 20.4 GiB while loading, above the 18.8 GiB ceiling",
    )
    marker.update(
        "failed",
        error=f"{type(exc).__name__}: {exc}"[:1000],
        error_type=type(exc).__name__,
        required_bytes=exc.required,
        current_bytes=exc.current,
    )

    data = read_marker(tmp_path / "dep-1-rank-1.json")
    assert data["rank"] == 1
    assert data["phase"] == "failed"
    assert data["error_type"] == "InsufficientMemoryError"
    assert data["required_bytes"] == 20_400_000_000
    assert data["current_bytes"] == 18_800_000_000
    assert "InsufficientMemoryError" in data["error"]


def test_runtime_failures_extracts_structured_fields(tmp_path):
    """DistributedJobSupervisor._runtime_failures() reads the same markers
    _runtime_failure_reason() does, but returns real fields instead of one
    joined string.
    """
    from omlx.cluster.inference_worker import RuntimeMarker

    for rank, error_type in ((0, None), (1, "InsufficientMemoryError")):
        marker = RuntimeMarker(
            state_dir=str(tmp_path),
            deployment_id="dep-1",
            rank=rank,
            world_size=2,
            model="qwen3-35b",
            backend="ring",
            plan_hash="hash-1",
        )
        extra = {"error": f"boom on rank {rank}"}
        if error_type:
            extra["error_type"] = error_type
            extra["required_bytes"] = 20_400_000_000
            extra["current_bytes"] = 18_800_000_000
        marker.update("failed", **extra)

    # Both hosts use a loopback ssh target so _read_rank_marker takes the
    # local read_marker() path for both — this test is about field
    # extraction, not the SSH-vs-local branch (covered elsewhere), so a real
    # SSH call to a nonexistent "peer.local" would just hang/error here.
    hosts = (
        ClusterHost(node_id="mini", ssh="127.0.0.1", ips=("172.16.99.1",)),
        ClusterHost(node_id="peer", ssh="localhost", ips=("172.16.99.2",)),
    )
    deployment = SimpleNamespace(
        deployment_id="dep-1", plan_hash="hash-1", hosts=hosts
    )
    supervisor = SimpleNamespace(deployment=deployment, state_dir=str(tmp_path))
    supervisor._read_rank_marker = launch.DistributedJobSupervisor._read_rank_marker.__get__(
        supervisor
    )
    failures = launch.DistributedJobSupervisor._runtime_failures(supervisor)

    assert len(failures) == 2
    assert failures[0] == RankFailure(
        rank=0, node_id="mini", error_type=None, error="boom on rank 0"
    )
    assert failures[1] == RankFailure(
        rank=1,
        node_id="peer",
        error_type="InsufficientMemoryError",
        error="boom on rank 1",
        required_bytes=20_400_000_000,
        current_bytes=18_800_000_000,
    )

    # _runtime_failure_reason() must still produce the same human string as
    # before this refactor — it now derives from _runtime_failures().
    supervisor._runtime_failures = launch.DistributedJobSupervisor._runtime_failures.__get__(
        supervisor
    )
    reason = launch.DistributedJobSupervisor._runtime_failure_reason(supervisor)
    assert reason == "rank 0 (mini): boom on rank 0; rank 1 (peer): boom on rank 1"


def test_runtime_failures_absent_error_type_is_none(tmp_path):
    """An old-format marker (written before this rework, or a rank binary
    that hasn't been upgraded) has no error_type field — must decode as
    None, not crash or default to a memory-shaped type.
    """
    from omlx.cluster.inference_worker import RuntimeMarker

    marker = RuntimeMarker(
        state_dir=str(tmp_path),
        deployment_id="dep-1",
        rank=0,
        world_size=1,
        model="m",
        backend="ring",
        plan_hash="h",
    )
    marker.update("failed", error="ConnectionError: peer unreachable")

    hosts = (ClusterHost(node_id="mini", ssh="127.0.0.1", ips=("172.16.99.1",)),)
    deployment = SimpleNamespace(deployment_id="dep-1", plan_hash="h", hosts=hosts)
    supervisor = SimpleNamespace(deployment=deployment, state_dir=str(tmp_path))
    supervisor._read_rank_marker = launch.DistributedJobSupervisor._read_rank_marker.__get__(
        supervisor
    )
    failures = launch.DistributedJobSupervisor._runtime_failures(supervisor)
    assert failures[0].error_type is None


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


def test_peer_script_unloads_only_unpinned_standalone_models(monkeypatch, tmp_path):
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


def test_peer_script_reports_draining_and_unreachable_server(monkeypatch, tmp_path):
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


def test_peer_script_does_not_guess_a_port(monkeypatch, tmp_path):
    """Redesign change vs. the original (#2870): no fallback to guessed
    conventional ports (8000, 9000). A missing/invalid configured port must
    report the failure and stop — never send a signed admin request to a
    guessed destination.
    """
    calls = []

    def responder(request):
        calls.append(request)
        return {"models": []}

    report, requests = _run_peer_script(
        monkeypatch,
        tmp_path,
        {"auth": {"secret_key": "s" * 32}},  # no "server" section at all
        responder,
    )
    assert requests == []  # never even tried to reach a server
    assert report["server_reachable"] is False
    assert any("no configured server port" in e for e in report["errors"])


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


def _deployment(hosts=(LOCAL_HOST, PEER_HOST)):
    return SimpleNamespace(deployment_id="dep-1", hosts=tuple(hosts))


def _memory_error(rank_failures):
    return DistributedLaunchError("activation failed", rank_failures=rank_failures)


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
    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="InsufficientMemoryError",
                error="over budget",
            ),
        )
    )
    hosts = routes._memory_squeezed_hosts(deployment, exc)
    assert [host.node_id for host in hosts] == ["M-FJX1D769D0.local"]


def test_memory_squeezed_hosts_no_fallback_to_every_host():
    """Redesign change vs. the original: a memory-shaped failure with no
    rank_failures (or none matching the allow-list) evicts nothing — it does
    NOT fall back to every host in the deployment, unlike #2870.
    """
    deployment = _deployment()
    assert routes._memory_squeezed_hosts(deployment, _memory_error(())) == []
    assert (
        routes._memory_squeezed_hosts(
            deployment,
            _memory_error(
                (RankFailure(rank=0, node_id="mini", error_type=None, error="x"),)
            ),
        )
        == []
    )


def test_memory_squeezed_hosts_ignores_non_memory_error_types():
    deployment = _deployment()
    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="ConnectionError",
                error="peer unreachable",
            ),
        )
    )
    assert routes._memory_squeezed_hosts(deployment, exc) == []


def test_memory_failure_types_is_an_allow_list(monkeypatch):
    """§6 decision: a set, not a single comparison — adding a second entry
    must be matched the same way as the first, guarding against a future
    edit reverting this to a single `==` compare.
    """
    monkeypatch.setattr(
        routes, "_MEMORY_FAILURE_TYPES", frozenset({"InsufficientMemoryError", "SomeOtherMemoryError"})
    )
    deployment = _deployment()
    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="SomeOtherMemoryError",
                error="also over budget",
            ),
        )
    )
    assert [h.node_id for h in routes._memory_squeezed_hosts(deployment, exc)] == [
        "M-FJX1D769D0.local"
    ]


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

    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="InsufficientMemoryError",
                error="rank 1 reached 20.4 GiB, above the 18.8 GiB ceiling",
            ),
        )
    )
    detail = asyncio.run(routes._evict_competing_local_models(_deployment(), exc))

    assert remote_calls == [("aphoenix@mbp.local", "/opt/omlx/bin/python")]
    assert "qwen3-8b on M-FJX1D769D0.local" in detail
    assert "retry the activation" in detail
    assert "Pinned model(s) gemma-fast on M-FJX1D769D0.local" in detail
    recorded = store.list()
    assert [incident.state_code for incident in recorded] == [
        "activation_memory_recovery"
    ]
    assert recorded[0].deployment_id == "dep-1"


def test_recovery_evicts_coordinator_models_through_the_pool(tmp_path, monkeypatch):
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

    exc = _memory_error(
        (
            RankFailure(
                rank=0, node_id="mini", error_type="InsufficientMemoryError", error="over budget"
            ),
        )
    )
    detail = asyncio.run(
        routes._evict_competing_local_models(_deployment((LOCAL_HOST,)), exc)
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

    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="InsufficientMemoryError",
                error="over budget",
            ),
        )
    )
    detail = asyncio.run(
        routes._evict_competing_local_models(_deployment((PEER_HOST,)), exc)
    )

    assert detail.startswith(str(exc))
    assert "could not complete everywhere" in detail
    assert [incident.state_code for incident in store.list()] == [
        "activation_memory_recovery_failed"
    ]


def test_recovery_ignores_failures_that_are_not_memory_shaped(tmp_path, monkeypatch):
    store = _incident_store(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise AssertionError("no eviction may run for a non-memory failure")

    monkeypatch.setattr(routes, "evict_remote_local_models", explode)
    monkeypatch.setattr(routes, "_get_engine_pool", explode)

    exc = _memory_error(
        (RankFailure(rank=0, node_id="mini", error_type=None, error="SIGHUP"),)
    )
    detail = asyncio.run(routes._evict_competing_local_models(_deployment(), exc))

    assert detail == str(exc)
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

    exc = _memory_error(
        (
            RankFailure(
                rank=1,
                node_id="M-FJX1D769D0.local",
                error_type="InsufficientMemoryError",
                error="over budget",
            ),
        )
    )
    detail = asyncio.run(
        routes._evict_competing_local_models(_deployment((PEER_HOST,)), exc)
    )

    assert "No competing local models were loaded" in detail
    # Nothing was freed and nothing failed, so no incident is warranted.
    assert not store.list()


def test_recovery_no_op_when_deployment_is_none():
    exc = _memory_error(
        (RankFailure(rank=0, node_id="mini", error_type="InsufficientMemoryError", error="x"),)
    )
    detail = asyncio.run(routes._evict_competing_local_models(None, exc))
    assert detail == str(exc)
