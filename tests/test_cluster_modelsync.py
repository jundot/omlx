# SPDX-License-Identifier: Apache-2.0
"""Cluster v2 model sync: manifests, path_map resolution, migration, decisions.

All transports are mocked — no test here opens a socket, runs rsync, or
touches the Hugging Face hub.
"""

import base64
import json
import platform
import stat
import struct
import subprocess
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import modelsync
from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    decode_worker_contract,
    decode_worker_path_map,
    validate_model_path_map,
)
from omlx.cluster.modelsync import (
    AUTO_RSYNC_THRESHOLD_BYTES,
    ModelSyncError,
    ModelSyncManager,
    allow_patterns_for_shard,
    build_manifest,
    build_rsync_argv,
    compare_manifests,
    parse_rsync_progress,
)
from omlx.cluster.planner import PipelineAssignment, synthetic_model_layout, ShardPlan
from omlx.cluster.registry import ClusterRegistry


def _write_shard(directory, name, tensors, payload=b"\x00" * 32):
    header = {t: {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]} for t in tensors}
    blob = json.dumps(header).encode()
    (directory / name).write_bytes(struct.pack("<Q", len(blob)) + blob + payload)


def _model(root, layers=4, per_file=2, with_index=True):
    root.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for start in range(0, layers, per_file):
        names = [
            f"model.layers.{i}.self_attn.q_proj.weight"
            for i in range(start, min(start + per_file, layers))
        ]
        fname = f"model-{start:05d}.safetensors"
        _write_shard(root, fname, names)
        for n in names:
            mapping[n] = fname
    _write_shard(root, "model-shared.safetensors", ["model.embed_tokens.weight"])
    mapping["model.embed_tokens.weight"] = "model-shared.safetensors"
    if with_index:
        (root / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping})
        )
    (root / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (root / "tokenizer.json").write_text("{}")
    return root


def _deployment(model="/models/shared", path_map=None) -> ClusterDeployment:
    return ClusterDeployment(
        deployment_id="sync-test",
        model=model,
        backend="ring",
        hosts=(
            ClusterHost("local", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("peer", "user@studio.local", ("10.0.0.2",)),
        ),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 2, 4, 64),
            PipelineAssignment("peer", 1, 0, 2, 10, 2, 4, 32),
        ),
        plan_hash="e" * 64,
        path_map=path_map or {},
    )


# -- manifest -----------------------------------------------------------------


def test_build_manifest_on_fixture_model(tmp_path):
    root = _model(tmp_path / "model-a")

    manifest = build_manifest(root, model_id="org/model-a")

    assert manifest.model_id == "org/model-a"
    assert manifest.index_sha256 is not None
    assert len(manifest.index_sha256) == 64
    assert len(manifest.identity_sha256) == 64
    names = [item.name for item in manifest.files]
    assert "model-00000.safetensors" in names
    assert "model-shared.safetensors" in names
    assert "config.json" in names  # sidecars travel with the manifest
    assert manifest.total_bytes == sum(item.size_bytes for item in manifest.files)
    assert manifest.total_bytes > 0
    # Round-trip through the wire format the endpoint serves.
    restored = modelsync.ModelManifest.from_dict(manifest.to_dict())
    assert restored == manifest


def test_build_manifest_without_index_still_identifies(tmp_path):
    root = _model(tmp_path / "model-b", with_index=False)

    manifest = build_manifest(root)

    assert manifest.index_sha256 is None
    assert manifest.identity_sha256
    assert manifest.total_bytes > 0


def test_build_manifest_rejects_non_model_dir(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()

    with pytest.raises(ValueError, match="no safetensors"):
        build_manifest(root)


@pytest.fixture
def manifest_client(tmp_path, monkeypatch):
    """The manifest router mounted bare, backed by fixture model dirs."""

    models_root = tmp_path / "models"
    _model(models_root / "alpha")
    settings = SimpleNamespace(
        get_effective_model_dirs=lambda: [str(models_root)]
    )
    monkeypatch.setattr(modelsync, "_pool_getter", None)
    monkeypatch.setattr(modelsync, "_settings_loader", lambda: settings)
    app = FastAPI()
    app.include_router(modelsync.manifest_router)
    return TestClient(app), models_root


def test_manifest_endpoint_serves_known_model(manifest_client):
    client, models_root = manifest_client

    response = client.get(f"/api/cluster/models/{models_root}/alpha/manifest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 1
    assert payload["index_sha256"]
    assert payload["total_bytes"] > 0
    names = {item["name"] for item in payload["files"]}
    assert "model-shared.safetensors" in names
    assert "config.json" in names


def test_manifest_endpoint_404_for_unknown_model(manifest_client):
    client, _ = manifest_client

    response = client.get("/api/cluster/models/no/such-model/manifest")

    assert response.status_code == 404


def test_manifest_endpoint_refuses_paths_outside_model_dirs(
    manifest_client, tmp_path
):
    client, _ = manifest_client
    outside = _model(tmp_path / "elsewhere")

    response = client.get(f"/api/cluster/models/{outside}/manifest")

    assert response.status_code == 403


# -- manifest comparison / status ----------------------------------------------


def test_compare_manifests_states(tmp_path):
    root = _model(tmp_path / "source")
    local = build_manifest(root)

    assert compare_manifests(local, None)["state"] == "missing"

    present = build_manifest(root)
    assert compare_manifests(local, present)["state"] == "present"
    assert compare_manifests(local, present)["bytes"] == local.total_bytes

    peer_root = _model(tmp_path / "peer")
    (peer_root / "model-00002.safetensors").unlink()
    partial = build_manifest(peer_root)
    result = compare_manifests(local, partial)
    assert result["state"] == "partial"
    assert result["missing"] == ["model-00002.safetensors"]
    assert result["bytes"] == local.total_bytes - sum(
        item.size_bytes for item in local.files if item.name == "model-00002.safetensors"
    )

    (peer_root / "config.json").write_text(json.dumps({"model_type": "other"}))
    mismatch = build_manifest(peer_root)
    assert compare_manifests(local, mismatch)["state"] == "mismatch"


def test_status_uses_peer_manifest_over_http(tmp_path):
    root = _model(tmp_path / "source")
    manager = ModelSyncManager(
        http_fetch=lambda url, timeout: build_manifest(root, model_id="m").to_dict(),
        settings_loader=lambda: SimpleNamespace(get_effective_model_dirs=lambda: []),
    )

    result = manager.status("peer.local:8080", str(root))

    assert result["state"] == "present"
    assert result["bytes"] == result["total_bytes"]


def test_status_missing_when_peer_lacks_model(tmp_path):
    root = _model(tmp_path / "source")
    manager = ModelSyncManager(http_fetch=lambda url, timeout: None)

    result = manager.status("peer.local:8080", str(root))

    assert result["state"] == "missing"
    assert result["bytes"] == 0


def test_peer_manifest_requires_a_port(tmp_path):
    manager = ModelSyncManager()

    with pytest.raises(ValueError, match="no port"):
        manager.peer_manifest("peer.local", "model")


# -- path_map resolution --------------------------------------------------------


def test_path_map_resolution_with_fallback():
    deployment = _deployment(
        path_map={"peer": "/Volumes/models/studio-copy"},
    )

    assert deployment.model_path_for("peer") == "/Volumes/models/studio-copy"
    # Unlisted nodes keep the shared coordinator path — pre-v2 behavior.
    assert deployment.model_path_for("local") == "/models/shared"
    # A deployment with no path_map resolves everywhere to the shared path.
    legacy = _deployment()
    assert legacy.model_path_for("peer") == "/models/shared"


def test_path_map_validation():
    with pytest.raises(ValueError, match="absolute"):
        _deployment(path_map={"peer": "relative/dir"})
    with pytest.raises(ValueError, match="outside the deployment"):
        _deployment(path_map={"stranger": "/models/x"})
    assert validate_model_path_map(None) == {}
    assert validate_model_path_map({"a": "/m"}, ("a",)) == {"a": "/m"}


def test_current_deployment_round_trip_preserves_path_map():
    deployment = _deployment(path_map={"peer": "/Volumes/models/copy"})

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored == deployment
    assert restored.path_map == {"peer": "/Volumes/models/copy"}
    assert deployment.to_dict()["schema_version"] == 3


def test_legacy_v1_deployment_decodes_without_path_map():
    payload = _deployment().to_dict()
    payload["schema_version"] = 1
    del payload["path_map"]

    restored = ClusterDeployment.from_dict(payload)

    assert restored.path_map == {}
    assert restored.model_path_for("peer") == restored.model


def test_path_map_rides_the_worker_contract():
    deployment = _deployment(path_map={"peer": "/Volumes/models/copy"})

    encoded = deployment.encode_worker_plan()
    plan_hash, assignments, profiles, tp = decode_worker_contract(encoded)

    assert plan_hash == deployment.plan_hash
    assert assignments == deployment.assignments
    assert decode_worker_path_map(encoded) == {"peer": "/Volumes/models/copy"}


def test_v1_worker_contract_decodes_to_empty_path_map():
    deployment = _deployment()
    raw = json.dumps(
        {
            "schema_version": 1,
            "plan_hash": deployment.plan_hash,
            "assignments": [
                assignment.to_dict() for assignment in deployment.assignments
            ],
            "performance_profiles": [],
            "tensor_parallel_size": 1,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    encoded = base64.urlsafe_b64encode(zlib.compress(raw, level=9)).decode()

    assert decode_worker_path_map(encoded) == {}
    plan_hash, assignments = decode_worker_contract(encoded)[:2]
    assert plan_hash == deployment.plan_hash
    assert len(assignments) == 2


def test_plan_to_dict_carries_path_map_without_changing_hash():
    plan = ShardPlan(
        model=synthetic_model_layout(total_weight_bytes=1024, layer_count=4),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 2, 4, 64),
            PipelineAssignment("peer", 1, 0, 2, 10, 2, 4, 32),
        ),
        plan_hash="f" * 64,
    )
    mapped = ShardPlan(
        model=plan.model,
        assignments=plan.assignments,
        plan_hash=plan.plan_hash,
        path_map={"peer": "/Volumes/models/copy"},
    )

    assert "path_map" not in plan.to_dict()
    assert mapped.to_dict()["path_map"] == {"peer": "/Volumes/models/copy"}
    # The layer split is path-independent: the map is display/staging
    # metadata and must not mint a new plan identity.
    assert mapped.plan_hash == plan.plan_hash


def test_placement_signature_covers_path_map_only_when_present():
    from omlx.cluster import routes

    plan = ShardPlan(
        model=synthetic_model_layout(total_weight_bytes=1024, layer_count=4),
        assignments=(
            PipelineAssignment("local", 0, 2, 4, 20, 2, 4, 64),
            PipelineAssignment("peer", 1, 0, 2, 10, 2, 4, 32),
        ),
        plan_hash="f" * 64,
    ).to_dict()

    legacy = routes._placement_signature(plan)
    assert routes._placement_signature(dict(plan)) == legacy
    mapped = plan | {"path_map": {"peer": "/Volumes/models/copy"}}
    assert routes._placement_signature(mapped) != legacy


# -- legacy deployments.json migration ------------------------------------------


def _write_legacy_registry(base: Path, deployment: ClusterDeployment) -> Path:
    entry = deployment.to_dict()
    entry["schema_version"] = 1
    del entry["path_map"]
    path = base / "cluster" / "deployments.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"schema_version": 1, "deployments": [entry]}, indent=2)
    )
    return path


def test_legacy_deployments_json_migrates_on_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    deployment = _deployment(str(model))
    path = _write_legacy_registry(tmp_path, deployment)

    registry = ClusterRegistry(tmp_path)

    loaded = registry.get_for_model(str(model))
    assert loaded is not None
    assert loaded.path_map == {}
    assert loaded.model_path_for("peer") == str(model)
    assert registry.migrated_from == 1
    # The registry remains schema v2 while each upgraded deployment now carries
    # schema v3 (signed phase ownership) and an explicit empty path_map.
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 2
    assert on_disk["deployments"][0]["schema_version"] == 3
    assert on_disk["deployments"][0]["path_map"] == {}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # A v2 file loads cleanly on the next start, with no migration flagged.
    assert ClusterRegistry(tmp_path).migrated_from is None


def test_registry_round_trip_preserves_path_map(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    deployment = _deployment(
        str(model), path_map={"peer": "/Volumes/models/copy"}
    )
    registry = ClusterRegistry(tmp_path)

    registry.upsert(deployment)
    restored = ClusterRegistry(tmp_path).get_for_model(str(model))

    assert restored is not None
    assert restored.path_map == {"peer": "/Volumes/models/copy"}
    assert restored.model_path_for("peer") == "/Volumes/models/copy"


# -- sync decisions --------------------------------------------------------------


def test_auto_method_prefers_rsync_for_large_models_with_ssh_trust():
    manager = ModelSyncManager(ssh_trust=lambda target: True)

    assert (
        manager.decide_method("user@peer", AUTO_RSYNC_THRESHOLD_BYTES + 1)
        == "rsync"
    )
    # At or below 20 GiB each node downloads its own shard files instead.
    assert (
        manager.decide_method("user@peer", AUTO_RSYNC_THRESHOLD_BYTES)
        == "download"
    )
    # No enrolled SSH trust: download regardless of size.
    untrusted = ModelSyncManager(ssh_trust=lambda target: False)
    assert (
        untrusted.decide_method("user@peer", AUTO_RSYNC_THRESHOLD_BYTES * 4)
        == "download"
    )
    # No SSH target at all: download.
    assert manager.decide_method(None, AUTO_RSYNC_THRESHOLD_BYTES * 4) == "download"


def test_sync_rsync_invokes_resumable_transport(tmp_path):
    root = _model(tmp_path / "source")
    runs = []

    def fake_rsync(argv, on_line):
        runs.append(argv)
        on_line("  1,048,576  50%  100.00MB/s    0:00:01")
        return 0

    manager = ModelSyncManager(
        ssh_trust=lambda target: True,
        rsync_run=fake_rsync,
    )
    events = []
    result = manager.sync(
        "peer.local:8080",
        str(root),
        "rsync",
        ssh_target="user@peer.local",
        destination="/Volumes/models/copy",
        on_progress=events.append,
    )

    argv = runs[0]
    assert "--partial" in argv and "--append-verify" in argv
    assert "BatchMode=yes" in argv[argv.index("-e") + 1]
    assert argv[-2].endswith("/")
    assert argv[-1] == "user@peer.local:/Volumes/models/copy"
    assert result["method"] == "rsync"
    transferring = [e for e in events if e.phase == "transferring"]
    assert transferring, "rsync progress lines must surface as UI events"
    assert transferring[0].bytes_done == 1_048_576
    assert transferring[0].bytes_per_second == pytest.approx(100.0 * 1000**2)
    assert transferring[0].eta_seconds == 1.0
    assert events[-1].phase == "done"
    # Events are also retained for polling UIs.
    assert manager.events[-1].phase == "done"


def test_sync_rsync_requires_trust_and_target(tmp_path):
    root = _model(tmp_path / "source")
    manager = ModelSyncManager(ssh_trust=lambda target: False)

    with pytest.raises(ModelSyncError, match="enrolled SSH trust"):
        manager.sync("peer", str(root), "rsync", ssh_target="user@peer")
    with pytest.raises(ModelSyncError, match="ssh_target"):
        manager.sync("peer", str(root), "rsync", ssh_target=None)


def test_sync_download_fetches_only_shard_files(tmp_path):
    root = _model(tmp_path / "source")
    calls = []

    def fake_download(*, repo_id, allow_patterns, local_dir):
        calls.append((repo_id, tuple(allow_patterns), local_dir))

    manager = ModelSyncManager(hf_download=fake_download)
    destination = tmp_path / "inbound"

    result = manager.sync(
        "peer.local:8080",
        str(root),
        "download",
        destination=destination,
        repo_id="org/model-a",
        start_layer=0,
        end_layer=2,
    )

    repo_id, patterns, local_dir = calls[0]
    assert repo_id == "org/model-a"
    assert Path(local_dir) == destination
    # Layers 0-1 live in model-00000; embeddings and sidecars always travel.
    assert "model-00000.safetensors" in patterns
    assert "model-shared.safetensors" in patterns
    assert "config.json" in patterns
    assert "model-00002.safetensors" not in patterns, "other stages stay on HF"
    assert result["method"] == "download"
    assert result["allow_patterns"] == sorted(patterns)


def test_sync_download_requires_repo_and_destination(tmp_path):
    root = _model(tmp_path / "source")
    manager = ModelSyncManager(hf_download=lambda **kw: None)

    with pytest.raises(ModelSyncError, match="repo ID"):
        manager.sync("peer", str(root), "download", destination=tmp_path / "x")
    with pytest.raises(ModelSyncError, match="destination"):
        manager.sync("peer", str(root), "download", repo_id="org/m")


def test_sync_failure_emits_error_event(tmp_path):
    root = _model(tmp_path / "source")

    def failing_rsync(argv, on_line):
        return 23

    manager = ModelSyncManager(
        ssh_trust=lambda target: True,
        rsync_run=failing_rsync,
    )
    events = []

    with pytest.raises(ModelSyncError, match="status 23"):
        manager.sync(
            "peer", str(root), "rsync",
            ssh_target="user@peer", on_progress=events.append,
        )

    assert events[-1].phase == "error"


def test_sync_auto_selects_rsync_for_large_models(tmp_path):
    root = _model(tmp_path / "source")
    runs = []
    downloads = []
    manager = ModelSyncManager(
        ssh_trust=lambda target: True,
        rsync_run=lambda argv, on_line: runs.append(argv) or 0,
        hf_download=lambda **kw: downloads.append(kw),
    )

    # The fixture is far below 20 GiB, so auto picks download even with trust.
    manager.sync(
        "peer",
        str(root),
        "auto",
        ssh_target="user@peer",
        destination=tmp_path / "d",
        repo_id="org/m",
    )
    assert not runs and len(downloads) == 1

    # Force the decision boundary: a manifest over the threshold picks rsync.
    big = ModelSyncManager(
        ssh_trust=lambda target: True,
        rsync_run=lambda argv, on_line: runs.append(argv) or 0,
    )
    import omlx.cluster.modelsync as ms

    original = ms.AUTO_RSYNC_THRESHOLD_BYTES
    ms.AUTO_RSYNC_THRESHOLD_BYTES = 1
    try:
        big.sync("peer", str(root), "auto", ssh_target="user@peer")
    finally:
        ms.AUTO_RSYNC_THRESHOLD_BYTES = original
    assert len(runs) == 1


# -- rsync helpers ----------------------------------------------------------------


def test_build_rsync_argv_is_resumable_and_noninteractive():
    argv = build_rsync_argv(
        "/models/src", "user@peer.local", "/Volumes/models/dst",
        ssh_identity="~/.ssh/omlx_cluster",
    )

    assert argv[0] == "rsync"
    assert "--partial" in argv
    assert "--append-verify" in argv
    assert "--info=progress2" in argv
    ssh = argv[argv.index("-e") + 1]
    assert "BatchMode=yes" in ssh and "omlx_cluster" in ssh
    assert argv[-2] == "/models/src/"
    assert argv[-1] == "user@peer.local:/Volumes/models/dst"


def test_parse_rsync_progress_lines():
    assert parse_rsync_progress("  1,234,567  42%  123.45MB/s    0:01:23") == (
        1_234_567,
        123.45 * 1000**2,
        83.0,
    )
    assert parse_rsync_progress("100  100%  1.00GB/s    1:02:03")[2] == 3723.0
    assert parse_rsync_progress("sending incremental file list") is None
    assert parse_rsync_progress("") is None


# -- allow-patterns ------------------------------------------------------------


def test_allow_patterns_cover_whole_model_without_layer_range(tmp_path):
    root = _model(tmp_path / "source")
    (root / "modeling_custom.py").write_text("CUSTOM_MODEL = True\n")
    (root / "tokenizer.tiktoken").write_text("token 1\n")
    (root / "chat_examples.jsonl").write_text('{"role":"user"}\n')

    patterns = allow_patterns_for_shard(root)

    assert "model-00000.safetensors" in patterns
    assert "model-00002.safetensors" in patterns
    assert "model-shared.safetensors" in patterns
    assert "config.json" in patterns
    assert "modeling_custom.py" in patterns
    assert "tokenizer.tiktoken" in patterns
    assert "chat_examples.jsonl" in patterns


def test_allow_patterns_reject_inverted_range(tmp_path):
    root = _model(tmp_path / "source")

    with pytest.raises(ValueError, match="0 <= start < end"):
        allow_patterns_for_shard(root, 3, 3)


# -- launch preflight -------------------------------------------------------------


def test_preflight_uses_per_node_model_paths(monkeypatch):
    from omlx.cluster import launch
    from omlx.cluster.launch import _local_runtime_versions, preflight_remote_hosts
    from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION

    versions = _local_runtime_versions()
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(
                versions
                | {
                    "cluster-protocol": CLUSTER_PROTOCOL_VERSION,
                    "python": platform.python_version(),
                    "model-exists": True,
                    "admission-ceiling-bytes": 1024**4,
                }
            ),
            stderr="",
        )

    deployment = _deployment(
        "/nonexistent/shared",
        path_map={"peer": "/Volumes/models/studio-copy"},
    )
    result = preflight_remote_hosts(
        deployment,
        python_executable="/opt/omlx/bin/python",
        runner=runner,
    )

    assert len(result) == 2
    assert len(calls) == 1
    # The peer is probed at its own path, not the coordinator's.
    assert "/Volumes/models/studio-copy" in calls[0][-1]
    assert "/nonexistent/shared" not in calls[0][-1]
