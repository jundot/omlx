# SPDX-License-Identifier: Apache-2.0
"""The plan/apply half of Phase A: the policy is visible and it is applied.

The solver tests live in ``test_policy.py``; these drive the real router so the
contract the card reads is the contract the backend produces.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from assistant_harness import (
    APPLY_URL,
    PLAN_URL,
    FakeCoordinator,
    apply_payload,
    assistant_api,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant import provisioning
from omlx.cluster.planner import ModelLayout

GIB = 1024**3


def table_layout(path: str) -> ModelLayout:
    """A checkpoint with an Engram table on every layer, still small enough."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=True,
        tensor_parallel_ssd_layer_bytes=(512 * 1024**2,) * 20,
        table_offload_kind="engram",
        max_context_tokens=32768,
    )


def test_the_plan_reports_the_selected_macs_and_the_resolved_policy(api) -> None:
    client, _, _ = api()
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    assert plan["selected_node_ids"] == plan["rank_order"]
    assert plan["excluded_nodes"] == []
    policy = plan["resource_policy"]
    assert policy["context_tokens"] == plan["context_tokens"]
    assert policy["target_context_tokens"] == plan["context_tokens"]
    assert policy["admission"] == "p50"
    assert policy["table_offload"] == "none"


def test_a_table_offload_plan_is_shown_and_written_on_every_selected_mac(
    api, monkeypatch
) -> None:
    writes: list[tuple[str, str, Mapping[str, Any]]] = []

    def write(target: Any, model_path: str, patch: Mapping[str, Any]) -> None:
        writes.append((target.node_id, model_path, dict(patch)))

    client, _, _ = api(
        coordinator=FakeCoordinator().factory(), settings_writer=write
    )
    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", table_layout)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, table_offload="engram")

    assert plan["resource_policy"]["table_offload"] == "engram"
    assert plan["warnings"]

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "applied"
    assert {node_id for node_id, _, _ in writes} == set(plan["rank_order"])
    assert all(
        patch == {"deepseek_v41_engram_ssd_offload": True}
        for _, _, patch in writes
    )


def test_a_worker_policy_write_happens_after_staging_and_fails_apply(
    api, monkeypatch
) -> None:
    calls: list[str] = []

    def write(target: Any, model_path: str, patch: Mapping[str, Any]) -> None:
        calls.append(target.node_id)
        if target.node_id == "m1":
            raise RuntimeError("the worker has no model settings yet")

    client, _, _ = api(
        coordinator=FakeCoordinator().factory(), settings_writer=write
    )
    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", table_layout)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, table_offload="engram")

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "policy_settings_failed"
    # The coordinator is written before staging; the worker only after it, so
    # a Mac that has not received the copy yet is not asked for model settings.
    assert calls[0] == "m5"
    assert "m1" in calls


def test_a_failed_policy_write_fails_apply_with_one_action(api, monkeypatch) -> None:
    def refusing(target: Any, model_path: str, patch: Mapping[str, Any]) -> None:
        raise RuntimeError("that Mac refused the settings write")

    client, _, _ = api(
        coordinator=FakeCoordinator().factory(), settings_writer=refusing
    )
    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", table_layout)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id, table_offload="engram")

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "failed"
    assert body["error"]["code"] == "policy_settings_failed"
    assert body["error"]["action"] == "diagnostics"


def context_limited_layout(path: str) -> ModelLayout:
    """The KV cache only fits at half the requested context on the small Mac."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=True,
        kv_bytes_per_token_per_layer=1024 * 1024,
        max_context_tokens=65536,
    )


def test_a_reduced_context_survives_plan_apply(tmp_path, monkeypatch) -> None:
    with assistant_api(
        tmp_path, monkeypatch, coordinator=FakeCoordinator().factory()
    ) as (client, _store, _ops):
        monkeypatch.setattr(
            provisioning, "inspect_safetensors_layout", context_limited_layout
        )
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id, context_tokens=32768)

        assert plan["resource_policy"]["context_tokens"] < 32768
        assert plan["context_tokens"] == plan["resource_policy"]["context_tokens"]
        assert "context_reduced" in {
            warning["code"] for warning in plan["warnings"]
        }

        response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
        assert response.status_code == 200, response.text
        body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "applied"
    assert body["context_tokens"] == plan["resource_policy"]["context_tokens"]


def non_pipeline_layout(path: str) -> ModelLayout:
    """An architecture mlx-lm cannot split into pipeline stages."""

    return ModelLayout(
        source=path,
        fixed_weight_bytes=1 * GIB,
        layer_weight_bytes=(2 * GIB,) * 20,
        supports_pipeline=False,
    )


def test_a_non_pipeline_model_is_refused_before_the_coordinator(api, monkeypatch) -> None:
    client, _, _ = api()
    monkeypatch.setattr(
        provisioning, "inspect_safetensors_layout", non_pipeline_layout
    )
    measurement_id = take_measurement(client)

    response = client.post(
        PLAN_URL,
        json={
            "measurement_id": measurement_id,
            "model_path": "/models/qwen",
            "context_tokens": 8192,
            "mtp": False,
        },
    )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "no_feasible_plan"
    assert body["action"] == "choose_pipeline_model"
    assert body["feasible"] is False
    assert body["coordinator_node"] == ""


def test_the_worker_uses_the_interpreter_it_reported(api) -> None:
    # An empty catalogue means the model is not listed yet, so the conditional
    # reload (B2) runs — this test pins the worker interpreter, and the reload
    # is the path that prepares it.
    fake = FakeCoordinator(models=[])
    client, _, _ = api(coordinator=fake.factory())
    nodes = [
        {
            "node_id": "m5",
            "ssh": "172.16.1.11",
            "python_executable": "/opt/m5/python",
        },
        {
            "node_id": "m1",
            "ssh": "172.16.1.14",
            "python_executable": "/opt/m1/python",
        },
    ]
    measurement_id = take_measurement(client, nodes=nodes)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "applied"
    assert fake.reloads >= 1
    hosts = {
        host.node_id: host.python_executable for host in fake.prepared[0].hosts
    }
    # The worker's own interpreter, not the coordinator's `sys.executable`: a
    # worker whose checkout lives elsewhere otherwise fails its SSH preflight.
    assert hosts["m1"] == "/opt/m1/python"
    # Rank 0 is also spawned over SSH; when the measurement reported its
    # interpreter, that exact interpreter is used there too.
    assert hosts["m5"] == "/opt/m5/python"


def test_a_worker_without_a_reported_interpreter_uses_the_published_shim(api) -> None:
    """The slim app's venv has no omlx package; the shim exports PYTHONPATH."""

    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    nodes = [
        {"node_id": "m5", "ssh": "172.16.1.11"},
        {"node_id": "m1", "ssh": "172.16.1.14"},
    ]
    measurement_id = take_measurement(client, nodes=nodes)
    plan = take_plan(client, measurement_id)

    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "applied"
    hosts = {
        host.node_id: host.python_executable for host in fake.prepared[0].hosts
    }
    shim = str(Path(provisioning.DEFAULT_REMOTE_PYTHON).expanduser())
    assert hosts["m5"] == shim
    assert hosts["m1"] == shim


def test_local_worker_is_a_model_source_for_a_remote_coordinator() -> None:
    """The origin's own checkpoint is valid even when it is not rank 0.

    Before this fix, ``model_source`` skipped every ``target.local`` Mac, so a
    plan whose elected coordinator was another Studio could not use the
    origin's complete model copy as the staging source.
    """

    remote = SimpleNamespace(node_id="m12", model_present=False)
    local = SimpleNamespace(node_id="m11", model_present=True)
    record = SimpleNamespace(
        measurement_set=lambda: SimpleNamespace(nodes=(remote, local))
    )
    result = SimpleNamespace(rank_order=("m12", "m11"))
    targets = {
        "m12": SimpleNamespace(node_id="m12", ssh="172.16.1.12", local=False),
        "m11": SimpleNamespace(node_id="m11", ssh="172.16.1.11", local=True),
    }

    assert (
        provisioning.model_source(
            record,
            result,
            targets,
            coordinator=targets["m12"],
            inputs=None,
        )
        == "172.16.1.11"
    )


def test_a_drifted_policy_is_a_plan_mismatch(api, monkeypatch) -> None:
    client, _, _ = api(coordinator=FakeCoordinator().factory())
    monkeypatch.setattr(provisioning, "inspect_safetensors_layout", table_layout)
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)

    # The plan default-offloads the checkpoint's table, so asking for the
    # *resident* table is the drift now.
    response = client.post(
        APPLY_URL,
        json=apply_payload(measurement_id, plan, table_offload="none"),
    )

    assert response.status_code == 409
    assert response.json()["code"] == "plan_mismatch"
    assert response.json()["action"] == "plan"


def test_the_plan_exposes_a_speed_alternative_via_plan_tradeoffs() -> None:
    from types import SimpleNamespace

    from omlx.cluster.assistant.models import _plan_tradeoffs

    def policy(ctx, moe):
        return SimpleNamespace(
            context_tokens=ctx,
            moe_expert_offload=moe,
            moe_resident_fraction=0.25 if moe else 1.0,
            target_context_tokens=1_048_576,
            table_offload="ple",
            table_kind="ple",
            memory_guard_tier=None,
            microbatch_size=1,
            admission="p50",
        )

    quality = SimpleNamespace(
        policy=policy(131072, True), election=SimpleNamespace(selected=None)
    )
    speed = SimpleNamespace(
        policy=policy(65536, False),
        election=SimpleNamespace(
            selected=None,
            plan_signature="a" * 32,
            coordinator_node="a",
            rank_order=("a", "b"),
        ),
    )
    tradeoffs = _plan_tradeoffs(quality, (speed,))
    assert [item.label for item in tradeoffs] == ["speed"]
    assert tradeoffs[0].context_tokens == 65536
    assert tradeoffs[0].resource_policy.moe_expert_offload is False
    assert tradeoffs[0].plan_signature == "a" * 32
    assert tradeoffs[0].against_context_tokens == 131072
    assert tradeoffs[0].against_moe_resident_fraction == 0.25

    # No tradeoff when the main plan already keeps experts resident, and an
    # alternative that still offloads experts is not a speed win.
    resident = SimpleNamespace(
        policy=policy(65536, False), election=SimpleNamespace(selected=None)
    )
    assert _plan_tradeoffs(resident, ()) == []
    assert _plan_tradeoffs(quality, (quality,)) == []
