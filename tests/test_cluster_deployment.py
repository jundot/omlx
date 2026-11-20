# SPDX-License-Identifier: Apache-2.0

import base64
import json
import platform
import stat
import subprocess
import zlib
from dataclasses import replace
from pathlib import Path

import pytest

from omlx.cluster import memory_guard
from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    _assignment_from_dict,
    decode_worker_contract,
    decode_worker_path_map,
    validate_model_path_map,
)
from omlx.cluster.performance import NodePerformanceProfile, execution_profile
from omlx.cluster.planner import PipelineAssignment, ShardPlan, synthetic_model_layout
from omlx.cluster.registry import ClusterRegistry

GIB = 1024**3


def _assignments() -> tuple[PipelineAssignment, ...]:
    return (
        PipelineAssignment(
            node_id="large",
            rank=0,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
        ),
        PipelineAssignment(
            node_id="small",
            rank=1,
            start_layer=0,
            end_layer=2,
            layer_weight_bytes=60 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=128 * GIB,
        ),
    )


def _deployment(backend: str = "jaccl") -> ClusterDeployment:
    if backend == "ring":
        hosts = (
            ClusterHost("large", "127.0.0.1", ("192.168.20.1",)),
            ClusterHost("small", "studio.local", ("192.168.20.2",)),
        )
    else:
        hosts = (
            ClusterHost(
                "large",
                "127.0.0.1",
                ("192.168.20.1",),
                (None, "rdma_en5"),
            ),
            ClusterHost(
                "small",
                "studio.local",
                ("192.168.20.2",),
                ("rdma_en5", None),
            ),
        )
    return ClusterDeployment(
        deployment_id="nemotron-ultra",
        model="mlx-community/Nemotron-Ultra-253B-4bit",
        backend=backend,
        hosts=hosts,
        assignments=_assignments(),
        plan_hash="a" * 64,
    )


def test_deployment_round_trip_and_worker_plan_are_json_only():
    deployment = _deployment()

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    plan_hash, assignments = decode_worker_contract(deployment.encode_worker_plan())[:2]

    assert restored == deployment
    assert plan_hash == deployment.plan_hash
    assert assignments == deployment.assignments
    assert deployment.distributed_init_backend == "jaccl"


def test_hostfile_envs_carry_stability_defaults(monkeypatch):
    for name in (
        "MLX_METAL_FAST_SYNCH",
        "MLX_MAX_OPS_PER_BUFFER",
        "MLX_MAX_MB_PER_BUFFER",
        "JACCL_PROGRESS_TIMEOUT_MS",
        "JACCL_TIMEOUT_ACTION",
    ):
        monkeypatch.delenv(name, raising=False)
    deployment = _deployment()

    envs = deployment.hostfile_dict()["envs"]

    # Metal command-buffer caps: an unbounded buffer overruns the GPU
    # driver's ~10 s timeout, which answers with SIGABRT and orphaned wired
    # memory. JACCL knobs pin the wheel's ProgressGuard/teardown-exit posture.
    assert "MLX_METAL_FAST_SYNCH=0" in envs
    assert "MLX_MAX_OPS_PER_BUFFER=16" in envs
    assert "MLX_MAX_MB_PER_BUFFER=512" in envs
    assert "JACCL_PROGRESS_TIMEOUT_MS=30000" in envs
    assert "JACCL_TIMEOUT_ACTION=teardown-exit" in envs
    # Fast fences only for JACCL tensor parallelism.
    for backend, value in (("jaccl", "1"), ("ring", "0")):
        tensor = replace(_deployment(backend), tensor_parallel_size=2)
        assert f"MLX_METAL_FAST_SYNCH={value}" in tensor.hostfile_dict()["envs"]


def test_hostfile_envs_respect_operator_overrides(monkeypatch):
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "1")
    monkeypatch.setenv("MLX_MAX_OPS_PER_BUFFER", "32")
    monkeypatch.setenv("JACCL_PROGRESS_TIMEOUT_MS", "60000")
    deployment = _deployment()

    envs = deployment.hostfile_dict()["envs"]

    assert "MLX_METAL_FAST_SYNCH=1" in envs
    assert "MLX_METAL_FAST_SYNCH=0" not in envs
    assert "MLX_MAX_OPS_PER_BUFFER=32" in envs
    assert "MLX_MAX_OPS_PER_BUFFER=16" not in envs
    assert "JACCL_PROGRESS_TIMEOUT_MS=60000" in envs
    # Untouched knobs keep their tuned defaults.
    assert "MLX_MAX_MB_PER_BUFFER=512" in envs
    assert "JACCL_TIMEOUT_ACTION=teardown-exit" in envs
    # An operator can keep fast fences off for JACCL tensor parallelism too.
    monkeypatch.setenv("MLX_METAL_FAST_SYNCH", "0")
    tensor = replace(deployment, tensor_parallel_size=2)
    assert "MLX_METAL_FAST_SYNCH=0" in tensor.hostfile_dict()["envs"]


def test_deployment_round_trip_preserves_the_selected_context():
    deployment = _deployment()
    deployment = ClusterDeployment(
        deployment_id=deployment.deployment_id,
        model=deployment.model,
        backend=deployment.backend,
        hosts=deployment.hosts,
        assignments=deployment.assignments,
        plan_hash=deployment.plan_hash,
        target_context_tokens=262144,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored.target_context_tokens == 262144
    assert restored.to_dict()["target_context_tokens"] == 262144


def test_deployment_round_trip_preserves_tensor_parallel_size():
    """Tensor parallel size must survive to_dict/from_dict and worker plan encoding."""
    from omlx.cluster.planner import PipelineAssignment

    assignments = (
        PipelineAssignment(
            node_id="large",
            rank=0,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
            tensor_parallel_rank=0,
            tensor_parallel_size=2,
            sharded_weight_bytes=4 * GIB,
        ),
        PipelineAssignment(
            node_id="small",
            rank=1,
            start_layer=2,
            end_layer=6,
            layer_weight_bytes=120 * GIB,
            fixed_weight_bytes=2 * GIB,
            reserve_bytes=8 * GIB,
            capacity_bytes=256 * GIB,
            tensor_parallel_rank=1,
            tensor_parallel_size=2,
            sharded_weight_bytes=4 * GIB,
        ),
    )
    deployment = ClusterDeployment(
        deployment_id="tp-test",
        model="mlx-community/test",
        backend="jaccl",
        hosts=(
            ClusterHost("large", "127.0.0.1", ("192.168.20.1",), (None, "rdma_en5")),
            ClusterHost("small", "studio.local", ("192.168.20.2",), ("rdma_en5", None)),
        ),
        assignments=assignments,
        plan_hash="a" * 64,
        tensor_parallel_size=2,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    assert restored == deployment
    assert restored.tensor_parallel_size == 2

    # Worker plan encoding must also carry tensor_parallel_size
    encoded = deployment.encode_worker_plan()
    import base64
    import json
    import zlib

    compressed = base64.b64decode(encoded, altchars=b"-_")
    raw = zlib.decompress(compressed)
    payload = json.loads(raw)
    assert payload["tensor_parallel_size"] == 2
    assert payload["assignments"][0]["tensor_parallel_rank"] == 0
    assert payload["assignments"][0]["sharded_weight_bytes"] == 4 * GIB


def test_deployment_rejects_non_divisible_tensor_parallel_size():
    """Host count must be divisible by tensor_parallel_size."""
    with pytest.raises(ValueError, match="divisible"):
        ClusterDeployment(
            deployment_id="bad-tp",
            model="model",
            backend="ring",
            hosts=(
                ClusterHost("a", "127.0.0.1", ("10.0.0.1",)),
                ClusterHost("b", "b.local", ("10.0.0.2",)),
                ClusterHost("c", "c.local", ("10.0.0.3",)),
            ),
            assignments=_assignments(),
            plan_hash="c" * 64,
            tensor_parallel_size=2,
        )


def test_deployment_round_trip_preserves_execution_and_performance_profiles():
    original = _deployment()
    profiles = tuple(
        NodePerformanceProfile(
            node_id=host.node_id,
            rank=rank,
            decode_weight_bytes_per_second=100 + rank,
            prefill_weight_bytes_per_second=200 + rank,
            collective_latency_seconds=0.001,
            collective_bandwidth_bytes_per_second=10_000,
            backend=original.backend,
            measured_at="2026-07-26T12:00:00+00:00",
            samples=5,
        )
        for rank, host in enumerate(original.hosts)
    )
    deployment = ClusterDeployment(
        deployment_id=original.deployment_id,
        model=original.model,
        backend=original.backend,
        hosts=original.hosts,
        assignments=original.assignments,
        plan_hash=original.plan_hash,
        execution=execution_profile("throughput"),
        performance_profiles=profiles,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored == deployment
    assert restored.execution.profile == "throughput"
    assert restored.performance_profiles[1].node_id == original.hosts[1].node_id


@pytest.mark.parametrize(
    "target",
    [
        "-oProxyCommand=bad",
        "studio.local;touch /tmp/pwned",
        "studio.local\nbad",
        "",
    ],
)
def test_ssh_target_rejects_option_and_shell_injection(target):
    with pytest.raises(ValueError, match="invalid SSH target"):
        ClusterHost("node", target, ("192.168.1.2",))


def test_jaccl_requires_complete_matrix_with_null_diagonal():
    with pytest.raises(ValueError, match="full RDMA connectivity matrix"):
        ClusterDeployment(
            deployment_id="test",
            model="model",
            backend="jaccl",
            hosts=(
                ClusterHost("large", "127.0.0.1", ("192.168.1.1",)),
                ClusterHost("small", "small.local", ("192.168.1.2",)),
            ),
            assignments=_assignments(),
            plan_hash="b" * 64,
        )


def test_rank_zero_must_be_local_launcher_process():
    deployment = _deployment("ring")
    with pytest.raises(ValueError, match="rank 0"):
        ClusterDeployment(
            deployment_id=deployment.deployment_id,
            model=deployment.model,
            backend=deployment.backend,
            hosts=(
                ClusterHost("large", "large.local", ("192.168.20.1",)),
                deployment.hosts[1],
            ),
            assignments=deployment.assignments,
            plan_hash=deployment.plan_hash,
        )


def test_decode_worker_contract_rejects_trailing_compressed_payload():
    deployment = _deployment()
    encoded = deployment.encode_worker_plan()
    compressed = base64.urlsafe_b64decode(encoded)
    malformed = base64.urlsafe_b64encode(compressed + zlib.compress(b"{}")).decode()

    with pytest.raises(ValueError, match="malformed"):
        decode_worker_contract(malformed)


def test_decode_worker_contract_rejects_unbounded_decompressed_payload():
    raw = json.dumps(
        {
            "schema_version": 1,
            "plan_hash": "a" * 64,
            "assignments": [],
            "padding": "x" * (300 * 1024),
        }
    ).encode()
    encoded = base64.urlsafe_b64encode(zlib.compress(raw)).decode()

    with pytest.raises(ValueError, match="too large"):
        decode_worker_contract(encoded)


# --- What the rank reads back has to be what the planner wrote --------------
#
# ``_assignment_from_dict`` is the only reader of an assignment on the far side
# of both seams that matter: the registry file the admin server reloads, and
# the ``--plan`` argument the rank decodes. A field ``to_dict`` emits and this
# decoder ignores is a value that silently becomes zero on the machine that
# acts on it, with every round-trip test still green — which is exactly what
# happened to the KV cache below.


def _planned_assignment(**overrides) -> PipelineAssignment:
    """An assignment shaped like one the planner really produces."""

    fields = dict(
        node_id="macbook",
        rank=0,
        start_layer=2,
        end_layer=6,
        layer_weight_bytes=40 * GIB,
        fixed_weight_bytes=2 * GIB,
        reserve_bytes=32 * GIB,
        capacity_bytes=107 * GIB,
        role="workstation",
        kv_cache_bytes=20 * GIB,
        kv_bytes_per_token=2_500_000,
        max_context_tokens=13_000,
    )
    fields.update(overrides)
    return PipelineAssignment(**fields)


def test_every_field_the_planner_writes_survives_the_decoder():
    original = _planned_assignment()

    restored = _assignment_from_dict(original.to_dict())

    assert restored == original
    # The number the rank's memory guard is charged, and the engine pool
    # reserves against. It was arriving 20 GiB light because the KV cache was
    # emitted and never read back.
    assert restored.planned_weight_bytes == original.planned_weight_bytes
    assert restored.kv_cache_bytes == 20 * GIB
    assert restored.max_context_tokens == 13_000


def test_the_role_survives_the_worker_plan_and_the_registry_file():
    assignments = (
        _planned_assignment(role="workstation"),
        _planned_assignment(
            node_id="studio",
            rank=1,
            start_layer=0,
            end_layer=2,
            capacity_bytes=256 * GIB,
            reserve_bytes=25 * GIB,
            role="headless",
        ),
    )
    deployment = ClusterDeployment(
        deployment_id="roles",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "studio.local", ("10.0.0.2",)),
        ),
        assignments=assignments,
        plan_hash="d" * 64,
    )

    # The registry writes and reloads this.
    restored = ClusterDeployment.from_dict(json.loads(json.dumps(deployment.to_dict())))
    # The rank decodes this.
    _hash, decoded = decode_worker_contract(deployment.encode_worker_plan())[:2]

    assert [item.role for item in restored.assignments] == [
        "workstation",
        "headless",
    ]
    assert [item.role for item in decoded] == ["workstation", "headless"]


def test_the_memory_tier_survives_the_worker_plan_and_legacy_defaults_safely():
    original = _planned_assignment(memory_guard_tier="safe")
    peer = _planned_assignment(
        node_id="studio",
        rank=1,
        start_layer=0,
        end_layer=2,
        capacity_bytes=256 * GIB,
        reserve_bytes=25 * GIB,
        role="headless",
        memory_guard_tier="aggressive",
    )
    deployment = ClusterDeployment(
        deployment_id="memory-tier",
        model="org/model",
        backend="ring",
        hosts=(
            ClusterHost("macbook", "127.0.0.1", ("10.0.0.1",)),
            ClusterHost("studio", "studio.local", ("10.0.0.2",)),
        ),
        assignments=(original, peer),
        plan_hash="e" * 64,
    )

    restored = ClusterDeployment.from_dict(deployment.to_dict())
    _hash, decoded = decode_worker_contract(deployment.encode_worker_plan())[:2]
    assert [item.memory_guard_tier for item in restored.assignments] == [
        "safe",
        "aggressive",
    ]
    assert [item.memory_guard_tier for item in decoded] == ["safe", "aggressive"]

    legacy = original.to_dict()
    legacy.pop("memory_guard_tier")
    assert _assignment_from_dict(legacy).memory_guard_tier == "balanced"

    legacy["memory_guard_tier"] = "extreme"
    with pytest.raises(ValueError, match="unknown memory guard tier"):
        _assignment_from_dict(legacy)


def test_a_plan_with_no_role_decodes_unchanged():
    payload = _planned_assignment().to_dict()
    payload.pop("role")

    assert _assignment_from_dict(payload).role == ""


def test_a_plan_carrying_an_unknown_role_refuses_to_launch():
    """Fail the launch, not the person at the keyboard.

    A role nobody recognises means the chain that produced it is broken; the
    lenient reading is "headless", which is the fraction that fills the Mac.
    """

    payload = _planned_assignment().to_dict()
    payload["role"] = "workststion"

    with pytest.raises(ValueError, match="unknown node role"):
        _assignment_from_dict(payload)


def test_link_local_zone_ids_are_stripped_from_communication_ips():
    """macOS announces Thunderbolt addresses as fe80::…%en10; mlx's address
    parser cannot read the zone and the rank died at startup on it."""
    host = ClusterHost(
        "node", "peer.local", ("10.0.0.2", "fe80::23:3ee:ec30:9a92%en10")
    )
    assert host.ips == ("10.0.0.2", "fe80::23:3ee:ec30:9a92")


def test_hostfile_advertises_routable_addresses_before_link_local():
    from omlx.cluster.deployment import _hostfile_ips

    host = ClusterHost(
        "node",
        "peer.local",
        ("fe80::1%en4", "10.0.0.2", "fe80::2%en5"),
    )
    assert _hostfile_ips(host) == ["10.0.0.2", "fe80::1", "fe80::2"]


def test_a_link_local_only_host_keeps_its_zone_free_fallback():
    from omlx.cluster.deployment import _hostfile_ips

    host = ClusterHost("node", "peer.local", ("fe80::1%en4",))
    assert _hostfile_ips(host) == ["fe80::1"]


# -- per-node model paths ------------------------------------------------------


def _path_map_deployment(model="/models/shared", path_map=None) -> ClusterDeployment:
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


def test_path_map_resolution_with_fallback():
    deployment = _path_map_deployment(
        path_map={"peer": "/Volumes/models/studio-copy"},
    )

    assert deployment.model_path_for("peer") == "/Volumes/models/studio-copy"
    # Unlisted nodes keep the shared coordinator path.
    assert deployment.model_path_for("local") == "/models/shared"
    # A deployment with no path_map resolves everywhere to the shared path.
    legacy = _path_map_deployment()
    assert legacy.model_path_for("peer") == "/models/shared"


def test_path_map_validation():
    with pytest.raises(ValueError, match="absolute"):
        _path_map_deployment(path_map={"peer": "relative/dir"})
    with pytest.raises(ValueError, match="outside the deployment"):
        _path_map_deployment(path_map={"stranger": "/models/x"})
    with pytest.raises(ValueError, match="absolute"):
        _path_map_deployment(path_map={"peer": "/models/x\nmalformed"})
    assert validate_model_path_map(None) == {}
    assert validate_model_path_map({"a": "/m"}, ("a",)) == {"a": "/m"}


def test_deployment_schema_2_round_trip_preserves_path_map():
    deployment = _path_map_deployment(path_map={"peer": "/Volumes/models/copy"})

    restored = ClusterDeployment.from_dict(deployment.to_dict())

    assert restored == deployment
    assert restored.path_map == {"peer": "/Volumes/models/copy"}
    assert deployment.to_dict()["schema_version"] == 2


def test_schema_1_deployment_decodes_without_path_map():
    payload = _path_map_deployment().to_dict()
    payload["schema_version"] = 1
    del payload["path_map"]

    restored = ClusterDeployment.from_dict(payload)

    assert restored.path_map == {}
    assert restored.model_path_for("peer") == restored.model


def test_path_map_rides_the_worker_contract():
    deployment = _path_map_deployment(path_map={"peer": "/Volumes/models/copy"})

    encoded = deployment.encode_worker_plan()
    plan_hash, assignments, profiles, tp = decode_worker_contract(encoded)

    assert plan_hash == deployment.plan_hash
    assert assignments == deployment.assignments
    assert decode_worker_path_map(encoded) == {"peer": "/Volumes/models/copy"}


def test_schema_1_worker_contract_decodes_to_empty_path_map():
    deployment = _path_map_deployment()
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


def _write_legacy_registry(base: Path, deployment: ClusterDeployment) -> Path:
    entry = deployment.to_dict()
    entry["schema_version"] = 1
    del entry["path_map"]
    path = base / "cluster" / "deployments.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "deployments": [entry]}, indent=2))
    return path


def test_legacy_deployments_json_migrates_on_load(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    deployment = _path_map_deployment(str(model))
    path = _write_legacy_registry(tmp_path, deployment)

    registry = ClusterRegistry(tmp_path)

    loaded = registry.get_for_model(str(model))
    assert loaded is not None
    assert loaded.path_map == {}
    assert loaded.model_path_for("peer") == str(model)
    assert registry.migrated_from == 1
    # The upgrade is persisted: the file now carries schema v2 and an
    # explicit (empty) path_map, at the same private permissions.
    on_disk = json.loads(path.read_text())
    assert on_disk["schema_version"] == 2
    assert on_disk["deployments"][0]["schema_version"] == 2
    assert on_disk["deployments"][0]["path_map"] == {}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # A v2 file loads cleanly on the next start, with no migration flagged.
    assert ClusterRegistry(tmp_path).migrated_from is None


def test_registry_round_trip_preserves_path_map(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    deployment = _path_map_deployment(
        str(model), path_map={"peer": "/Volumes/models/copy"}
    )
    registry = ClusterRegistry(tmp_path)

    registry.upsert(deployment)
    restored = ClusterRegistry(tmp_path).get_for_model(str(model))

    assert restored is not None
    assert restored.path_map == {"peer": "/Volumes/models/copy"}
    assert restored.model_path_for("peer") == "/Volumes/models/copy"


def test_preflight_uses_per_node_model_paths(monkeypatch):
    from omlx.cluster.launch import _local_runtime_versions, preflight_remote_hosts
    from omlx.cluster.models import CLUSTER_PROTOCOL_VERSION

    versions = _local_runtime_versions()
    calls = []
    # The local rank measures live memory, which other test workers consume.
    monkeypatch.setattr(
        memory_guard, "ceiling_breakdown", lambda tier: {"hard_limit": 1024**4}
    )

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

    deployment = _path_map_deployment(
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
