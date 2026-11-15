# SPDX-License-Identifier: Apache-2.0
"""The free-space affordance: one model, one measured Mac, one relay.

A copy that cannot fit is refused with ``insufficient_disk`` and the single
action ``free_disk_space`` (the live-run notes). Freeing space is a delete on
the Mac that is short of it, and this route is the only place the assistant
deletes anything:

* it deletes a model that Mac's own measurement listed, never a path the
  caller names,
* it never deletes the model that measurement is about to run,
* it relays to oMLX's per-Mac ``DELETE /admin/api/hf/models/{name}`` over the
  same admin seam apply/start use, so a worker is as reachable as the
  coordinator and the browser never touches SSH.

Every test here is fake-driven: a faked measurement, a faked relay, no Mac.
"""

from __future__ import annotations

import pytest
from assistant_harness import (
    GIB,
    MODEL_PATH,
    SSH,
    STATUS_URL,
    assistant_api,
    seed_measurement,
    take_measurement,
)

from omlx.cluster.assistant import routes as assistant_routes
from omlx.cluster.assistant.start import RemoteAdminError
from omlx.cluster.assistant.state import MeasurementRecord

DELETE_URL = "/admin/api/cluster/assistant/models"


def delete_url(node_id: str, model_id: str, measurement_id: str) -> str:
    return f"{DELETE_URL}/{node_id}/{model_id}?measurement_id={measurement_id}"


class Relay:
    """The per-Mac delete, recorded instead of performed."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self.error = error

    def __call__(
        self, *, port: int, model_name: str, target: str = ""
    ) -> dict[str, object]:
        self.calls.append({"port": port, "model_name": model_name, "target": target})
        if self.error is not None:
            raise self.error
        return {"success": True, "message": f"Model '{model_name}' deleted"}


@pytest.fixture()
def relay(monkeypatch) -> Relay:
    fake = Relay()
    monkeypatch.setattr(assistant_routes, "delete_model_on_mac", fake)
    return fake


def two_models(ssh: str, python_executable: str | None = None):
    """A Mac's catalogue: the measured checkpoint plus a model it no longer needs."""

    return [
        {"model_path": MODEL_PATH, "id": "qwen", "estimated_size": 40 * GIB},
        {"model_path": "/models/old", "id": "old", "estimated_size": 12 * GIB},
    ]


def test_a_measured_model_is_deleted_on_the_mac_that_listed_it(
    tmp_path, monkeypatch, relay
):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == 200, response.text
        assert response.json() == {
            "node_id": "m1",
            "model_id": "old",
            "deleted": True,
        }
        # The relay goes to that Mac, on its measured admin port, and names the
        # folder oMLX's own delete route resolves — never a caller-typed path.
        assert relay.calls == [{"port": 8099, "model_name": "old", "target": SSH["m1"]}]


def test_this_mac_deletes_over_loopback(tmp_path, monkeypatch, relay):
    with assistant_api(
        tmp_path,
        monkeypatch,
        local_addresses=(SSH["m5"],),
        inventory=two_models,
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m5", "old", measurement_id))

        assert response.status_code == 200, response.text
        # No SSH target: the seam calls this Mac's own loopback admin route.
        assert relay.calls == [{"port": 8099, "model_name": "old", "target": ""}]


def test_the_model_the_cluster_is_about_to_run_is_never_deleted(
    tmp_path, monkeypatch, relay
):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m5", "qwen", measurement_id))

        assert response.status_code == 409, response.text
        assert response.json()["code"] == "model_in_use"
        assert response.json()["action"] == "adjust_model_or_context"
        # Refused before any Mac was asked to delete anything.
        assert relay.calls == []


def test_a_model_the_measurement_did_not_list_is_refused(tmp_path, monkeypatch, relay):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "not-there", measurement_id))

        assert response.status_code == 409, response.text
        assert response.json()["code"] == "unknown_model"
        assert response.json()["action"] == "measure"
        assert relay.calls == []


def test_a_mac_outside_the_measurement_is_refused(tmp_path, monkeypatch, relay):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("other-mac", "old", measurement_id))

        assert response.status_code == 409, response.text
        assert response.json()["code"] == "measurement_node_mismatch"
        assert response.json()["action"] == "measure"
        assert relay.calls == []


def test_an_unknown_measurement_is_refused(tmp_path, monkeypatch, relay):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        response = client.delete(delete_url("m1", "old", "ms-nope"))

        assert response.status_code == 404, response.text
        assert response.json()["code"] == "unknown_measurement"
        assert response.json()["action"] == "measure"
        assert relay.calls == []


def test_an_unpaired_mac_asks_to_pair_and_never_uses_this_mac(
    tmp_path, monkeypatch, relay
):
    with assistant_api(
        tmp_path,
        monkeypatch,
        device=lambda node_id: None,
        inventory=two_models,
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == 409, response.text
        assert response.json()["code"] == "node_not_paired"
        assert response.json()["action"] == "pair_mac"
        assert relay.calls == []


@pytest.mark.parametrize(
    ("status", "code", "action", "http_status"),
    [
        (404, "unknown_model", "measure", 409),
        (0, "delete_unreachable", "retry_coordinator", 502),
        (500, "delete_failed", "diagnostics", 502),
    ],
)
def test_the_macs_own_answer_decides_the_one_action(
    tmp_path, monkeypatch, status, code, action, http_status
):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        monkeypatch.setattr(
            assistant_routes,
            "delete_model_on_mac",
            Relay(RemoteAdminError(status, "the delete route answered")),
        )
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == http_status, response.text
        assert response.json()["code"] == code
        assert response.json()["action"] == action


def test_the_status_poll_lists_each_macs_models_with_sizes(tmp_path, monkeypatch):
    """The card may only list a Mac the user picked — so the poll carries it."""

    with assistant_api(tmp_path, monkeypatch, inventory=two_models) as (
        client,
        _store,
        _operations,
    ):
        measurement_id = take_measurement(client)

        status = client.get(STATUS_URL, params={"measurement_id": measurement_id})

        assert status.status_code == 200, status.text
        nodes = {node["node_id"]: node for node in status.json()["nodes"]}
        assert nodes["m1"]["models"] == [
            {"model_id": "old", "model_path": "/models/old", "size_bytes": 12 * GIB},
            {"model_id": "qwen", "model_path": MODEL_PATH, "size_bytes": 40 * GIB},
        ]


def test_a_measurement_without_an_inventory_offers_no_delete(
    tmp_path, monkeypatch, relay
):
    """A record from before this chapter knows no models: refuse, never guess."""

    with assistant_api(tmp_path, monkeypatch) as (client, store, _operations):
        measurement_id = seed_measurement(store)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == 409, response.text
        assert response.json()["code"] == "unknown_model"
        assert response.json()["action"] == "measure"
        assert relay.calls == []


def test_a_record_without_the_inventory_key_still_loads():
    """The added field is optional on disk: an older file keeps loading."""

    record = MeasurementRecord.from_dict(
        {
            "measurement_id": "ms-1",
            "state": "ready",
            "created_at": "2026-11-06T00:00:00+00:00",
            "updated_at": "2026-11-06T00:00:00+00:00",
            "model": {"model_path": MODEL_PATH},
        }
    )

    assert record.inventories == {}


def test_a_malformed_inventory_fails_closed():
    """An entry with no identity is a broken record, not a model to delete."""

    with pytest.raises(ValueError):
        MeasurementRecord.from_dict(
            {
                "measurement_id": "ms-1",
                "state": "ready",
                "created_at": "2026-11-06T00:00:00+00:00",
                "updated_at": "2026-11-06T00:00:00+00:00",
                "model": {"model_path": MODEL_PATH},
                "inventories": {"m1": [{"model_id": "", "model_path": ""}]},
            }
        )


def _old_only_on_m1(ssh: str, python_executable: str | None = None):
    if ssh == SSH["m1"]:
        return [
            {"model_path": MODEL_PATH, "id": "qwen", "estimated_size": 40 * GIB},
            {"model_path": "/models/old", "id": "old", "estimated_size": 12 * GIB},
        ]
    return [{"model_path": MODEL_PATH, "id": "qwen", "estimated_size": 40 * GIB}]


def test_the_last_measured_copy_is_never_deleted(tmp_path, monkeypatch, relay):
    """Deleting the only measured copy would lose the model; refuse.

    "No other Mac listed it" is not proof that no other Mac has it, so the
    safe answer is another model (the free-space panel) rather than data loss.
    """

    with assistant_api(
        tmp_path, monkeypatch, inventory=_old_only_on_m1
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "last_cluster_copy"
        assert body["action"] == "free_disk_space"
        assert relay.calls == []


def test_a_model_with_another_measured_copy_is_still_deletable(
    tmp_path, monkeypatch, relay
):
    with assistant_api(
        tmp_path, monkeypatch, inventory=two_models
    ) as (client, _store, _operations):
        measurement_id = take_measurement(client)

        response = client.delete(delete_url("m1", "old", measurement_id))

        assert response.status_code == 200, response.text
        assert relay.calls == [{"port": 8099, "model_name": "old", "target": SSH["m1"]}]
