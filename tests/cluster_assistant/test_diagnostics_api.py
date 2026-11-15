# SPDX-License-Identifier: Apache-2.0
"""The redacted support bundle.

A DAU diagnostics export must be safe to paste into a bug report: no key, no
SSH target, no prompt, no full model path. These tests drive the real route with
the fake cluster and inspect the serialized text, not just individual fields.
"""

from __future__ import annotations

from dataclasses import replace

from assistant_harness import (
    APPLY_URL,
    MODEL_PATH,
    SSH,
    FakeCoordinator,
    apply_payload,
    assistant_api,
    seed_measurement,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant.measurement import MeasurementWarning
from omlx.cluster.assistant.operations import KIND_APPLY

DIAGNOSTICS_URL = "/admin/api/cluster/assistant/diagnostics"


def test_an_empty_bundle_is_app_only_not_an_error(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, _store, _operations):
        response = client.get(DIAGNOSTICS_URL)

    assert response.status_code == 200
    body = response.json()
    assert body["app"]["python"]
    assert body["measurement"] is None
    assert body["operation"] is None
    assert body["topology"] is None
    assert body["events"] == []
    assert body["redactions"]


def test_the_bundle_carries_measurement_plan_and_topology(
    tmp_path, monkeypatch
) -> None:
    with assistant_api(
        tmp_path, monkeypatch, coordinator=FakeCoordinator().factory()
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id)
        applied = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
        operation = wait_for_operation(client, applied.json()["operation_id"])

        response = client.get(
            DIAGNOSTICS_URL,
            params={
                "measurement_id": measurement_id,
                "operation_id": operation["operation_id"],
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["measurement"]["measurement_id"] == measurement_id
    assert body["measurement"]["model"] == "qwen"
    assert {node["node_id"] for node in body["measurement"]["nodes"]} == {
        "m5",
        "m1",
    }
    assert body["operation"]["state"] == "applied"
    assert body["operation"]["rank_order"] == plan["rank_order"]
    assert body["plan"]["plan_signature"] == plan["plan_signature"]
    assert body["topology"]["measured"] is True
    assert body["topology"]["complete"] is True
    assert body["topology"]["links"] == [["m1", "m5"]]
    assert body["health"]["state"] == "applied"

    text = response.text
    assert MODEL_PATH not in text
    assert '"api_key"' not in text
    assert '"password"' not in text
    assert '"authorization"' not in text
    assert f"{SSH['m5']}@" not in text
    assert "prompt" not in text.lower() or "no prompts" in text.lower()


def test_an_unknown_id_stays_an_app_only_bundle(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, _store, _operations):
        response = client.get(
            DIAGNOSTICS_URL,
            params={"measurement_id": "ms-missing", "operation_id": "op-missing"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["measurement"] is None
    assert body["operation"] is None


def test_an_operation_without_its_link_reports_the_missing_pair(
    tmp_path, monkeypatch
) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, operations):
        measurement_id = seed_measurement(
            store, fabric_links=(), fabric_measured=True
        )
        operation = operations.start(
            kind=KIND_APPLY,
            measurement_id=measurement_id,
            plan_signature="a" * 64,
            coordinator_node="m5",
            rank_order=["m5", "m1"],
            context_tokens=8192,
        )

        response = client.get(
            DIAGNOSTICS_URL, params={"operation_id": operation.operation_id}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["topology"]["measured"] is True
    assert body["topology"]["complete"] is False
    assert body["topology"]["missing_links"] == [["m5", "m1"]]


def test_free_text_warnings_are_scrubbed(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = take_measurement(client)
        record = store.get(measurement_id)
        warning = MeasurementWarning(
            code="probe_failed",
            node_id="m1",
            message=(
                f"could not read {MODEL_PATH} on "
                f"andreaskolb@{SSH['m1']} with api_key=sk-secret123"
            ),
        )
        store.finish(
            measurement_id,
            measurement=replace(
                record.measurement_set(), warnings=(warning,)
            ),
            endpoints=dict(record.endpoints),
            fabric_links=record.fabric_links,
            fabric_measured=record.fabric_measured,
        )

        response = client.get(
            DIAGNOSTICS_URL, params={"measurement_id": measurement_id}
        )

    assert response.status_code == 200
    text = response.text
    assert MODEL_PATH not in text
    assert "andreaskolb@" not in text
    assert "sk-secret123" not in text
    assert "[ssh-target]" in text
    assert "[redacted-key]" in text or "api_key=[redacted]" in text


def test_a_measurement_only_bundle_never_mixes_in_another_operation(
    tmp_path, monkeypatch
) -> None:
    """An explicit measurement answers for that measurement only."""

    with assistant_api(
        tmp_path, monkeypatch, coordinator=FakeCoordinator().factory()
    ) as (client, _store, _operations):
        first = take_measurement(client)
        plan = take_plan(client, first)
        applied = client.post(APPLY_URL, json=apply_payload(first, plan))
        wait_for_operation(client, applied.json()["operation_id"])
        second = take_measurement(client)

        response = client.get(
            DIAGNOSTICS_URL, params={"measurement_id": second}
        )

    assert response.status_code == 200
    body = response.json()
    assert body["measurement"]["measurement_id"] == second
    assert body["operation"] is None
    assert body["plan"] is None
