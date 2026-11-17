# SPDX-License-Identifier: Apache-2.0
"""Phase B1: the synthetic compute probe after apply, used by the next plan.

The probe is calibration, never a gate. Its profiles are model-independent and
stored on the origin; the next measurement reads them and the election plans
with real compute instead of ``unmeasured_compute``.
"""

from __future__ import annotations

import json
import stat

from assistant_harness import (
    APPLY_URL,
    PERFORMANCE,
    FakeCoordinator,
    apply_payload,
    assistant_api,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant.performance_store import AssistantPerformanceStore


def _probe_reply() -> dict:
    return {"profiles": PERFORMANCE, "backend": "ring", "world_size": 2}


def test_the_store_is_mode_0600_and_has_no_secrets(tmp_path) -> None:
    store = AssistantPerformanceStore(tmp_path / "store")

    store.save(PERFORMANCE, deployment_id="a" * 64, world_size=2, backend="ring")

    mode = stat.S_IMODE(store.path.stat().st_mode)
    assert mode == 0o600
    raw = store.path.read_text()
    assert "node_id" in raw
    for forbidden in ("ssh", "api_key", "model_path", "password", "@"):
        assert forbidden not in raw
    restored = AssistantPerformanceStore(tmp_path / "store").for_nodes(
        ["m5", "m1"]
    )
    assert set(restored) == {"m5", "m1"}


def test_apply_runs_the_probe_and_the_next_measurement_plans_with_it(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCoordinator(probe_reply=_probe_reply())
    with assistant_api(
        tmp_path, monkeypatch, coordinator=fake.factory()
    ) as (client, _store, _ops):
        first = take_measurement(client)
        plan = take_plan(client, first)
        apply = client.post(APPLY_URL, json=apply_payload(first, plan))
        assert apply.status_code == 200, apply.text
        body = wait_for_operation(client, apply.json()["operation_id"])

        assert body["state"] == "applied"
        assert len(fake.probed) == 1
        assert "compute_probe_failed" not in {
            warning["code"] for warning in body["warnings"]
        }

        # The next measurement is model-bound but does not run a fresh probe;
        # it reads the profiles the apply stored.
        second = take_measurement(client)
        plan = take_plan(client, second)

    assert plan["predicted"]["measured"] is True
    assert "unmeasured_compute" not in {
        warning["code"] for warning in plan["warnings"]
    }
    stored = json.loads(
        (tmp_path / "store" / "cluster" / "assistant-performance.json").read_text()
    )
    assert {item["node_id"] for item in stored["profiles"]} == {"m5", "m1"}


def test_a_failed_probe_is_a_warning_and_never_gates_apply(
    tmp_path, monkeypatch
) -> None:
    fake = FakeCoordinator(probe_error=RuntimeError("probe could not launch"))

    with assistant_api(
        tmp_path, monkeypatch, coordinator=fake.factory()
    ) as (client, _store, _ops):
        measurement_id = take_measurement(client)
        plan = take_plan(client, measurement_id)
        apply = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
        body = wait_for_operation(client, apply.json()["operation_id"])

    assert body["state"] == "applied"
    assert "compute_probe_failed" in {
        warning["code"] for warning in body["warnings"]
    }


def test_the_model_bound_measurement_still_wins_over_stored_compute(
    tmp_path, monkeypatch
) -> None:
    """A fresh probe in the same measurement is used before the stored profile."""

    fake = FakeCoordinator(probe_reply=_probe_reply())
    with assistant_api(
        tmp_path, monkeypatch, coordinator=fake.factory()
    ) as (client, _store, _ops):
        first = take_measurement(client)
        plan = take_plan(client, first)
        apply = client.post(APPLY_URL, json=apply_payload(first, plan))
        wait_for_operation(client, apply.json()["operation_id"])

        # The harness's performance probe is wired; asking for it must prefer
        # the fresh profile, while the stored one is still a valid fallback.
        second = take_measurement(client, include_performance_probe=True)
        plan = take_plan(client, second)

    assert plan["predicted"]["measured"] is True


def test_the_store_rejects_a_malformed_file(tmp_path) -> None:
    path = tmp_path / "store" / "cluster" / "assistant-performance.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"schema_version": 99, "profiles": []}')

    store = AssistantPerformanceStore(tmp_path / "store")

    assert store.load_error
    assert store.for_nodes(["m5"]) == {}
