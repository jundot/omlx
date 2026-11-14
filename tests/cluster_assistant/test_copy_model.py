# SPDX-License-Identifier: Apache-2.0
"""Phase C2: source-first model placement.

The copy itself is the existing staging job: apply names the Mac that owns a
complete checkpoint as ``model_source`` and stages from there. No downloader is
involved, and a model that no measured Mac holds asks for a copy instead of
inventing one.
"""

from __future__ import annotations

from assistant_harness import (
    APPLY_URL,
    GIB,
    SSH,
    FakeCoordinator,
    apply_payload,
    assistant_api,
    plan_payload,
    seed_measurement,
    wait_for_operation,
)

PLAN_URL = "/admin/api/cluster/assistant/plan"


def test_a_missing_model_is_staged_from_the_peer_that_has_it(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCoordinator()
    with assistant_api(
        tmp_path, monkeypatch, coordinator=fake.factory()
    ) as (client, store, _operations):
        # m5 is the elected coordinator (m1 cannot hold the rank-0 head), but
        # only m1 holds the checkpoint: the copy must come from m1.
        measurement_id = seed_measurement(store, present=("m1",))
        plan_response = client.post(
            PLAN_URL,
            json=plan_payload(measurement_id, rank0_extra_bytes=20 * GIB),
        )
        assert plan_response.status_code == 200, plan_response.text
        plan = plan_response.json()
        assert plan["coordinator_node"] == "m5"
        assert plan["rank_order"] == ["m5", "m1"]

        response = client.post(
            APPLY_URL,
            json=apply_payload(
                measurement_id, plan, rank0_extra_bytes=20 * GIB
            ),
        )
        body = wait_for_operation(client, response.json()["operation_id"])

    assert body["state"] == "applied"
    # The coordinator was told to copy from m1; the browser never names a path.
    assert fake.staged
    _request, source_host, _prepared = fake.staged[0]
    assert source_host == SSH["m1"]
    assert _request.model_source == SSH["m1"]


def test_a_model_no_measured_mac_holds_asks_for_a_copy(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCoordinator()
    with assistant_api(
        tmp_path, monkeypatch, coordinator=fake.factory()
    ) as (client, store, _operations):
        measurement_id = seed_measurement(store, present=())
        plan = client.post(
            PLAN_URL, json=plan_payload(measurement_id)
        ).json()

        response = client.post(
            APPLY_URL, json=apply_payload(measurement_id, plan)
        )

    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "model_source_missing"
    assert body["action"] == "copy_model"
    # Nothing was staged: the assistant never downloads or invents a source.
    assert fake.staged == []
