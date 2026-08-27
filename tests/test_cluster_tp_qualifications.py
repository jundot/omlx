# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import stat
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from omlx.cluster import routes
from omlx.cluster.deployment import (
    ClusterDeployment,
    ClusterHost,
    decode_worker_contract,
)
from omlx.cluster.planner import ModelLayout, NodeBudget, plan_hybrid
from omlx.cluster.tp_qualifications import (
    TPLayoutQualification,
    TPLayoutQualificationStore,
    TPNodeFingerprint,
    TPQualificationKey,
    TPQualificationProvenance,
    TPRateEvidence,
    configure_tp_layout_qualification_store,
)

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64


def _fingerprint(node_id: str, chip: str, *, kernel: str = H3) -> TPNodeFingerprint:
    return TPNodeFingerprint(
        node_id=node_id,
        chip_name=chip,
        physical_memory_bytes=128 * 1024**3,
        accelerator="metal",
        fabric_identifier=H1,
        omlx_version="0.6.4.dev1",
        mlx_version="0.32.0",
        mlx_lm_version="0.31.3",
        python_version="3.13.13",
        os_name="darwin",
        os_version="25.6.0",
        jaccl_identifier=H2,
        kernel_identifier=kernel,
    )


def _key() -> TPQualificationKey:
    return TPQualificationKey(
        model_identity=H4,
        nodes=(
            _fingerprint("studio", "Apple M3 Ultra"),
            _fingerprint("m5", "Apple M5 Max"),
        ),
        backend="jaccl",
        tensor_parallel_size=2,
        context_bucket=32768,
        execution_profile="balanced",
        microbatch_size=4,
        decode_concurrency=8,
        prompt_concurrency=4,
        prefill_step_size=2048,
        auto_tune=True,
        mtp_enabled=False,
        mtp_depth=None,
    )


def _record(
    *,
    key: TPQualificationKey | None = None,
    promotable: bool = True,
    reason: str = "3:5 beat the matched equal control with exact output",
) -> TPLayoutQualification:
    return TPLayoutQualification(
        key=key or _key(),
        shard_weights=(3, 5),
        equal_control=TPRateEvidence(737.0, 30.5, 3),
        candidate=TPRateEvidence(870.0, 31.1, 3),
        exact=True,
        parity_sha256="a" * 64,
        promotable=promotable,
        reason=reason,
        qualified_at=datetime(2026, 8, 23, tzinfo=UTC).isoformat(),
    )


def _status(node_id: str, chip: str, *, kernel: str = H3) -> dict:
    return {
        "runtime_compatible": True,
        "status": {
            "node": {
                "chip_name": chip,
                "physical_memory_bytes": 128 * 1024**3,
                "accelerator": "metal",
                "fabric_kind": None,
            },
            "runtime": {
                "omlx_version": "0.6.4.dev1",
                "mlx_version": "0.32.0",
                "mlx_lm_version": "0.31.3",
                "python_version": "3.13.13",
                "os_name": "darwin",
                "os_version": "25.6.0",
                "jaccl_identifier": H2,
                "kernel_identifier": kernel,
            },
            "transport": {
                "rdma": {
                    "enabled": True,
                    "control_status": "enabled",
                    "devices": ["rdma_en5"],
                },
                "thunderbolt": {
                    "peer_connected": False,
                    "ports": [],
                },
            },
        },
    }


def test_store_atomic_round_trip_and_mode(tmp_path):
    store = TPLayoutQualificationStore(tmp_path)
    record = _record()
    store.record(record)

    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    payload = json.loads(store.path.read_text())
    assert payload["schema_version"] == 1
    assert len(payload["qualifications"]) == 1

    restored = TPLayoutQualificationStore(tmp_path)
    assert restored.load_error is None
    assert restored.lookup(record.key) == record


@pytest.mark.parametrize(
    "changed",
    [
        lambda key: replace(key, model_identity="b" * 64),
        lambda key: replace(key, nodes=tuple(reversed(key.nodes))),
        lambda key: replace(
            key,
            nodes=(key.nodes[0], replace(key.nodes[1], kernel_identifier="c" * 64)),
        ),
        lambda key: replace(key, context_bucket=262144),
        lambda key: replace(key, microbatch_size=2),
        lambda key: replace(key, mtp_enabled=True, mtp_depth=5),
    ],
)
def test_exact_lookup_rejects_every_key_mismatch(tmp_path, changed):
    store = TPLayoutQualificationStore(tmp_path)
    store.record(_record())
    assert store.lookup(changed(_key())) is None


def test_corrupt_or_unpromotable_store_fails_closed(tmp_path):
    path = tmp_path / "cluster" / "tp-layout-qualifications.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version":1,"qualifications":[{"bad":true}]}')
    corrupt = TPLayoutQualificationStore(tmp_path)
    assert corrupt.load_error
    assert corrupt.lookup(_key()) is None
    with pytest.raises(ValueError, match="refusing to overwrite"):
        corrupt.record(_record())

    path.unlink()
    store = TPLayoutQualificationStore(tmp_path)
    store.record(_record(promotable=False))
    assert store.lookup(_key()) is None


def test_failed_atomic_replace_preserves_old_record(tmp_path, monkeypatch):
    from omlx.cluster import tp_qualifications

    store = TPLayoutQualificationStore(tmp_path)
    first = _record()
    store.record(first)
    before = store.path.read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(tp_qualifications.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.record(_record(reason="new evidence"))
    assert store.path.read_bytes() == before
    assert store.lookup(first.key) == first


def test_route_consumes_only_exact_persistent_match(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    model.joinpath("config.json").write_text('{"model_type":"deepseek_v4"}')
    nodes = [
        routes.ClusterPlanNodeRequest(
            node_id="studio", capacity_bytes=256 * 1024**3
        ),
        routes.ClusterPlanNodeRequest(
            node_id="m5", capacity_bytes=128 * 1024**3
        ),
    ]
    statuses = {
        "studio": _status("studio", "Apple M3 Ultra"),
        "m5": _status("m5", "Apple M5 Max"),
    }
    key = routes._tp_qualification_key(
        model_path=str(model),
        nodes=nodes,
        statuses=statuses,
        backend="jaccl",
        tensor_parallel_size=2,
        target_context_tokens=30_000,
        execution_profile_name="balanced",
        auto_tune=True,
        sampling_rank_only=True,
        mtp_enabled=False,
        mtp_num_draft_tokens=None,
    )
    store = configure_tp_layout_qualification_store(tmp_path)
    store.record(_record(key=key))

    weights, provenance, decision = routes._resolve_tp_layout_qualification(
        model_path=str(model),
        nodes=nodes,
        statuses=statuses,
        backend="jaccl",
        tensor_parallel_size=2,
        target_context_tokens=30_000,
        execution_profile_name="balanced",
        auto_tune=True,
        sampling_rank_only=True,
        mtp_enabled=False,
        mtp_num_draft_tokens=None,
    )
    assert weights == ((3, 5),)
    assert provenance is not None and provenance.source == "persistent"
    assert decision["matched"] is True

    statuses["m5"] = _status("m5", "Apple M5 Max", kernel="d" * 64)
    weights, provenance, decision = routes._resolve_tp_layout_qualification(
        model_path=str(model),
        nodes=nodes,
        statuses=statuses,
        backend="jaccl",
        tensor_parallel_size=2,
        target_context_tokens=30_000,
        execution_profile_name="balanced",
        auto_tune=True,
        sampling_rank_only=True,
        mtp_enabled=False,
        mtp_num_draft_tokens=None,
    )
    assert weights is None and provenance is None
    assert decision["matched"] is False


def test_environment_override_precedes_persistent_record(
    tmp_path, monkeypatch, caplog
):
    model = tmp_path / "model"
    model.mkdir()
    model.joinpath("config.json").write_text('{"model_type":"deepseek_v4"}')
    nodes = [
        routes.ClusterPlanNodeRequest(node_id="studio", capacity_bytes=256 * 1024**3),
        routes.ClusterPlanNodeRequest(node_id="m5", capacity_bytes=128 * 1024**3),
    ]
    statuses = {
        "studio": _status("studio", "Apple M3 Ultra"),
        "m5": _status("m5", "Apple M5 Max"),
    }
    monkeypatch.setenv("OMLX_TP_QUALIFIED_SHARD_WEIGHTS", "5,3")
    monkeypatch.setenv(
        "OMLX_TP_QUALIFIED_MODEL_IDENTITY",
        routes.model_identity_digest(model),
    )
    weights, provenance, decision = routes._resolve_tp_layout_qualification(
        model_path=str(model),
        nodes=nodes,
        statuses=statuses,
        backend="jaccl",
        tensor_parallel_size=2,
        target_context_tokens=30_000,
        execution_profile_name="balanced",
        auto_tune=True,
        sampling_rank_only=True,
        mtp_enabled=False,
        mtp_num_draft_tokens=None,
    )
    assert weights == ((5, 3),)
    assert provenance is not None and provenance.source == "environment_override"
    assert decision["source"] == "environment_override"
    assert "takes precedence" in caplog.text


def test_qualification_provenance_changes_plan_and_approval_signatures():
    model = ModelLayout(
        source="synthetic-qualified",
        fixed_weight_bytes=100,
        layer_weight_bytes=(8_000, 8_000),
        tensor_parallel_heads=8,
        tensor_parallel_divisors=(8,),
        tensor_parallel_shard_units=8,
        supports_tensor_parallel=True,
    )
    nodes = [
        NodeBudget("studio", 1_000_000, rank=0),
        NodeBudget("m5", 1_000_000, rank=1),
    ]
    first = _record()
    second = _record(reason="same vector, independently revised evidence")
    plan_a = plan_hybrid(
        model,
        nodes,
        tensor_parallel_size=2,
        qualified_tensor_shard_weights=((3, 5),),
        tensor_parallel_qualification=TPQualificationProvenance.from_record(first),
    )
    plan_b = plan_hybrid(
        model,
        nodes,
        tensor_parallel_size=2,
        qualified_tensor_shard_weights=((3, 5),),
        tensor_parallel_qualification=TPQualificationProvenance.from_record(second),
    )

    assert [item.tensor_parallel_shard_weight for item in plan_a.assignments] == [3, 5]
    assert plan_a.plan_hash != plan_b.plan_hash
    assert routes._placement_signature(plan_a.to_dict()) != routes._placement_signature(
        plan_b.to_dict()
    )

    deployment = ClusterDeployment(
        deployment_id="qualified-test",
        model="/models/test",
        backend="ring",
        hosts=(
            ClusterHost("studio", "127.0.0.1", ("192.0.2.1",)),
            ClusterHost("m5", "m5.local", ("192.0.2.2",)),
        ),
        assignments=plan_a.assignments,
        plan_hash=plan_a.plan_hash,
        tensor_parallel_size=2,
        tensor_parallel_qualification=plan_a.tensor_parallel_qualification,
    )
    restored = ClusterDeployment.from_dict(deployment.to_dict())
    assert (
        restored.tensor_parallel_qualification
        == plan_a.tensor_parallel_qualification
    )
    plan_hash, assignments, _profiles, tp_size = decode_worker_contract(
        deployment.encode_worker_plan()
    )
    assert plan_hash == plan_a.plan_hash
    assert assignments == plan_a.assignments
    assert tp_size == 2


def _qualification_request() -> dict:
    return {
        "model_path": "/models/ds4",
        "backend": "jaccl",
        "nodes": [
            {"node_id": "studio", "capacity_bytes": 256 * 1024**3},
            {"node_id": "m5", "capacity_bytes": 128 * 1024**3},
        ],
        "hosts": [
            {"node_id": "studio", "ssh": "127.0.0.1", "ips": ["10.0.0.1"]},
            {"node_id": "m5", "ssh": "m5.local", "ips": ["10.0.0.2"]},
        ],
        "tensor_parallel_size": 2,
        "target_context_tokens": 32768,
        "execution_profile": "throughput",
        "auto_tune": True,
        "sampling_rank_only": True,
        "mtp_enabled": False,
        "shard_weights": [3, 5],
        "equal_control": {
            "prefill_tokens_per_second": 737.0,
            "decode_tokens_per_second": 30.5,
            "samples": 3,
            "output_sha256": "a" * 64,
        },
        "candidate": {
            "prefill_tokens_per_second": 870.0,
            "decode_tokens_per_second": 31.1,
            "samples": 3,
            "output_sha256": "a" * 64,
        },
        "reason": "matched physical A/B",
    }


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app)


def test_admin_route_records_live_keyed_tp_qualification(tmp_path, monkeypatch):
    store = configure_tp_layout_qualification_store(tmp_path)
    model = ModelLayout(
        source="synthetic-qualified",
        fixed_weight_bytes=100,
        layer_weight_bytes=(8_000, 8_000),
        tensor_parallel_heads=8,
        tensor_parallel_divisors=(8,),
        tensor_parallel_shard_units=8,
        supports_tensor_parallel=True,
    )
    budgets = [
        NodeBudget("studio", 1_000_000, rank=0),
        NodeBudget("m5", 1_000_000, rank=1),
    ]
    monkeypatch.setattr(routes, "_qualification_statuses", lambda _hosts: {})
    monkeypatch.setattr(routes, "_tp_qualification_key", lambda **_kwargs: _key())
    monkeypatch.setattr(routes, "_model_and_nodes", lambda _request: (model, budgets))

    response = _client().post(
        "/admin/api/cluster/tp-layout-qualifications",
        json=_qualification_request(),
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["shard_weights"] == [3, 5]
    assert store.lookup(_key()) is not None
    listed = _client().get(
        "/admin/api/cluster/tp-layout-qualifications"
    ).json()
    assert len(listed["qualifications"]) == 1


def test_admin_route_records_non_parity_or_weak_candidate_as_rejected(
    tmp_path, monkeypatch
):
    store = configure_tp_layout_qualification_store(tmp_path)
    model = ModelLayout(
        source="synthetic-qualified",
        fixed_weight_bytes=100,
        layer_weight_bytes=(8_000, 8_000),
        tensor_parallel_heads=8,
        tensor_parallel_divisors=(8,),
        tensor_parallel_shard_units=8,
        supports_tensor_parallel=True,
    )
    budgets = [
        NodeBudget("studio", 1_000_000, rank=0),
        NodeBudget("m5", 1_000_000, rank=1),
    ]
    monkeypatch.setattr(routes, "_qualification_statuses", lambda _hosts: {})
    monkeypatch.setattr(routes, "_tp_qualification_key", lambda **_kwargs: _key())
    monkeypatch.setattr(routes, "_model_and_nodes", lambda _request: (model, budgets))
    mismatch = _qualification_request()
    mismatch["candidate"]["output_sha256"] = "b" * 64

    response = _client().post(
        "/admin/api/cluster/tp-layout-qualifications",
        json=mismatch,
    )

    assert response.status_code == 200
    assert response.json()["state"] == "rejected"
    assert response.json()["exact"] is False
    assert store.lookup(_key()) is None
    decision = store.decision(_key())
    assert decision["source"] == "rejected_evidence"
    assert "output hash differs" in decision["reason"]

    weak = _qualification_request()
    weak["candidate"]["prefill_tokens_per_second"] = 740.0
    response = _client().post(
        "/admin/api/cluster/tp-layout-qualifications",
        json=weak,
    )
    assert response.status_code == 200
    assert response.json()["state"] == "rejected"
    assert response.json()["exact"] is True
    assert "promotion policy" in response.json()["reason"]
