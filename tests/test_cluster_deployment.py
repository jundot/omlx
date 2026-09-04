# SPDX-License-Identifier: Apache-2.0

import base64
import json
import zlib

import pytest

from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    _assignment_from_dict,
    decode_worker_plan,
)
from omlx.cluster.performance import NodePerformanceProfile, execution_profile
from omlx.cluster.planner import PipelineAssignment

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
    plan_hash, assignments = decode_worker_plan(deployment.encode_worker_plan())

    assert restored == deployment
    assert plan_hash == deployment.plan_hash
    assert assignments == deployment.assignments
    assert deployment.hostfile_dict()["envs"] == ["MLX_METAL_FAST_SYNCH=1"]
    assert deployment.distributed_init_backend == "jaccl"


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


def test_decode_worker_plan_rejects_trailing_compressed_payload():
    deployment = _deployment()
    encoded = deployment.encode_worker_plan()
    compressed = base64.urlsafe_b64decode(encoded)
    malformed = base64.urlsafe_b64encode(compressed + zlib.compress(b"{}")).decode()

    with pytest.raises(ValueError, match="malformed"):
        decode_worker_plan(malformed)


def test_decode_worker_plan_rejects_unbounded_decompressed_payload():
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
        decode_worker_plan(encoded)


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
    restored = ClusterDeployment.from_dict(
        json.loads(json.dumps(deployment.to_dict()))
    )
    # The rank decodes this.
    _hash, decoded = decode_worker_plan(deployment.encode_worker_plan())

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
    _hash, decoded = decode_worker_plan(deployment.encode_worker_plan())
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
