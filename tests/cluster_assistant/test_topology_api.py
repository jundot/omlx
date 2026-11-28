# SPDX-License-Identifier: Apache-2.0
"""The measured line, through the plan/apply API.

A 3-Mac plan is refused with ``topology_incomplete`` when the measured links
cannot form the pipeline order. The 2-Mac plan is offered, but only an explicit
``confirm_two_mac_fallback`` produces (and applies) it; the consent is part of
the signature.
"""

from __future__ import annotations

from assistant_harness import (
    GIB,
    SSH,
    assistant_api,
    plan_payload,
    seed_measurement,
    take_measurement,
)

from omlx.cluster.assistant.measurement import (
    measure_budget_payloads,
    measurement_timestamp,
)
from omlx.cluster.assistant.state import MeasurementEndpoint, ModelIdentity
from omlx.cluster.transport import TransportMatrix

PLAN_URL = "/admin/api/cluster/assistant/plan"
MODEL_PATH = "/models/qwen"
THIRD = "192.0.2.13"


def seed_three_node_measurement(
    store,
    *,
    fabric_links=(),
    fabric_measured=True,
) -> str:
    """m5 + m1 + m2, with the measured links the test names."""

    capacities = {"m5": (107.5, 96.75), "m1": (60.0, 54.0), "m2": (50.0, 40.0)}
    hosts = {"m5": SSH["m5"], "m1": SSH["m1"], "m2": THIRD}
    payloads = [
        {
            "node_id": node_id,
            "capacity_bytes": int(capacities[node_id][0] * GIB),
            "reserve_bytes": int(
                (capacities[node_id][0] - capacities[node_id][1]) * GIB
            ),
            "usable_bytes": int(capacities[node_id][1] * GIB),
            "role": "headless",
            "capacity_source": "admission_ceiling",
            "endpoint_host": hosts[node_id],
            "endpoint_port": 8099,
            "disk_free_bytes": 300 * GIB,
            "model_present": True,
        }
        for node_id in ("m5", "m1", "m2")
    ]
    measurement = measure_budget_payloads(
        payloads, measured_at=measurement_timestamp()
    )
    record = store.start(
        model=ModelIdentity(model_path=MODEL_PATH),
        node_ids=["m5", "m1", "m2"],
    )
    store.finish(
        record.measurement_id,
        measurement=measurement,
        endpoints={
            node_id: MeasurementEndpoint(host=hosts[node_id], port=8099)
            for node_id in hosts
        },
        fabric_links=fabric_links,
        fabric_measured=fabric_measured,
    )
    return record.measurement_id


def test_the_measurement_stores_the_measured_line(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = take_measurement(client)
        record = store.get(measurement_id)

    assert record.fabric_measured is True
    assert record.fabric_links == (("m1", "m5"),)


def test_an_empty_matrix_is_unmeasured_not_a_missing_cable(
    tmp_path, monkeypatch
) -> None:
    def empty(hosts):
        return TransportMatrix(transports=(), backend="ring")

    with assistant_api(
        tmp_path, monkeypatch, transports=empty
    ) as (client, store, _operations):
        measurement_id = take_measurement(client)
        record = store.get(measurement_id)

    assert record.fabric_measured is False
    assert record.fabric_links == ()


def test_a_three_mac_plan_with_no_line_is_refused_with_the_missing_pair(
    tmp_path, monkeypatch
) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = seed_three_node_measurement(
            store, fabric_links=[("m5", "m1")]
        )

        response = client.post(
            PLAN_URL,
            json=plan_payload(
                measurement_id, model_path=MODEL_PATH, node_selection="all"
            ),
        )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "topology_incomplete"
    assert body["missing_links"]
    assert body["fallback"]["feasible"] is True
    assert body["fallback"]["rank_order"] == ["m5", "m1"]
    assert body["fallback"]["topology"]["fallback_used"] is True


def test_only_an_explicit_confirmation_returns_the_two_mac_plan(
    tmp_path, monkeypatch
) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = seed_three_node_measurement(
            store, fabric_links=[("m5", "m1")]
        )
        unconfirmed = client.post(
            PLAN_URL,
            json=plan_payload(
                measurement_id, model_path=MODEL_PATH, node_selection="all"
            ),
        )

        confirmed = client.post(
            PLAN_URL,
            json=plan_payload(
                measurement_id,
                model_path=MODEL_PATH,
                node_selection="all",
                confirm_two_mac_fallback=True,
            ),
        )

    assert unconfirmed.status_code == 409
    assert confirmed.status_code == 200, confirmed.text
    plan = confirmed.json()
    assert plan["rank_order"] == ["m5", "m1"]
    assert plan["topology"]["fallback_used"] is True
    assert any(
        warning["code"] == "two_mac_fallback" for warning in plan["warnings"]
    )


def test_a_two_mac_plan_without_its_link_is_refused(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = seed_measurement(
            store, fabric_links=(), fabric_measured=True
        )

        response = client.post(
            PLAN_URL, json=plan_payload(measurement_id, model_path=MODEL_PATH)
        )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "topology_incomplete"
    assert body["action"] == "adjust_model_or_context"
    assert "fallback" not in body


def test_a_two_mac_plan_with_its_link_is_accepted(tmp_path, monkeypatch) -> None:
    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = seed_measurement(
            store, fabric_links=[("m5", "m1")], fabric_measured=True
        )

        response = client.post(
            PLAN_URL, json=plan_payload(measurement_id, model_path=MODEL_PATH)
        )

    assert response.status_code == 200, response.text
    plan = response.json()
    assert plan["topology"]["measured"] is True
    assert plan["topology"]["complete"] is True
    assert plan["topology"]["missing_links"] == []
