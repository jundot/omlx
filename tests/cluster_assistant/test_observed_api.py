# SPDX-License-Identifier: Apache-2.0
"""``observed``: the one measured rate block, and who may fill it.

The rules these pin are the ones ``the handoff notes`` records:

* the block is nullable and ``None`` for a normal start (the probe is off by
  default), and no rate is stored anywhere else;
* a real completion the coordinator served fills it, through the same
  local/remote seam apply/start use;
* a probe that cannot answer records one warning and the cluster stays
  ``ready`` — a probe never fails or unreadies an operation;
* the plan is not rewritten: ``predicted`` stays unmeasured and the plan's
  ``observed`` stays ``None``.
"""

from __future__ import annotations

import time
from typing import Any

from assistant_harness import (
    APPLY_URL,
    MODEL_PATH,
    FakeCoordinator,
    apply_payload,
    start_payload,
    take_measurement,
    take_plan,
    wait_for_operation,
)

from omlx.cluster.assistant.models import operation_status
from omlx.cluster.assistant.operations import OperationRecord

START_URL = "/admin/api/cluster/assistant/start"
PROBE_WARNING = "performance_probe_failed"


def applied(client, fake: FakeCoordinator) -> dict[str, Any]:
    measurement_id = take_measurement(client)
    plan = take_plan(client, measurement_id)
    response = client.post(APPLY_URL, json=apply_payload(measurement_id, plan))
    assert response.status_code == 200, response.text
    operation = wait_for_operation(client, response.json()["operation_id"])
    assert operation["state"] == "applied", operation
    return plan


def _start(client, plan: dict[str, Any]) -> str:
    response = client.post(START_URL, json=start_payload(plan))
    assert response.status_code == 200, response.text
    return response.json()["operation_id"]


def _wait(client, operation_id: str, predicate, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    body: dict[str, Any] = {}
    while time.monotonic() < deadline:
        response = client.get(
            f"/admin/api/cluster/assistant/operations/{operation_id}"
        )
        assert response.status_code == 200, response.text
        body = response.json()
        if body["state"] == "ready" and predicate(body):
            return body
        if body["state"] == "failed":
            raise AssertionError(body)
        time.sleep(0.01)
    raise AssertionError(f"operation {operation_id} never settled: {body}")


def test_a_normal_start_stores_no_rate_and_runs_no_probe(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory())
    operation_id = _start(client, applied(client, fake))

    operation = _wait(client, operation_id, lambda body: True)

    assert operation["observed"] is None
    assert operation["endpoint"]["ready"] is True
    assert fake.observed_calls == []
    assert PROBE_WARNING not in [item["code"] for item in operation["warnings"]]
    assert "tok_per_s" not in str(operation)


def test_a_real_completion_fills_observed_and_keeps_the_cluster_ready(api) -> None:
    fake = FakeCoordinator()
    client, _, operations = api(
        coordinator=fake.factory(),
        performance_probe=True,
    )
    plan = applied(client, fake)
    operation_id = _start(client, plan)

    operation = _wait(client, operation_id, lambda body: body["observed"] is not None)

    assert operation["state"] == "ready"
    assert operation["endpoint"]["ready"] is True
    assert fake.observed_calls == ["qwen"]
    observed = operation["observed"]
    assert observed["prefill_tok_per_s"] == 320.0
    assert observed["decode_tok_per_s"] == 42.0
    assert observed["ttft_ms"] == 250.0
    assert observed["measured_at"]
    codes = {item["code"] for item in operation["warnings"]}
    assert PROBE_WARNING not in codes
    assert "performance_unmeasured" not in codes
    # The record on disk carries the same block, and the status view matches.
    record = operations.get(operation_id)
    assert record is not None and record.observed is not None
    assert operation_status(record).model_dump()["observed"] == observed


def test_observed_round_trips_and_an_old_record_still_loads(api) -> None:
    fake = FakeCoordinator()
    client, _, operations = api(
        coordinator=fake.factory(),
        performance_probe=True,
    )
    operation_id = _start(client, applied(client, fake))
    _wait(client, operation_id, lambda body: body["observed"] is not None)
    record = operations.get(operation_id)
    assert record is not None

    raw = record.to_dict()
    assert raw["observed"] == record.observed.to_dict()
    assert OperationRecord.from_dict(raw).observed == record.observed
    # Records written before this field existed load with no measurement.
    legacy = {key: value for key, value in raw.items() if key != "observed"}
    assert OperationRecord.from_dict(legacy).observed is None


def test_a_probe_that_cannot_answer_only_adds_a_warning(api) -> None:
    fake = FakeCoordinator(observe_error=RuntimeError("the coordinator did not answer"))
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))

    operation = _wait(
        client,
        operation_id,
        lambda body: PROBE_WARNING in [item["code"] for item in body["warnings"]],
    )

    assert operation["state"] == "ready"
    assert operation["observed"] is None
    assert operation["endpoint"]["ready"] is True
    assert operation["error"] is None
    warning = next(
        item for item in operation["warnings"] if item["code"] == PROBE_WARNING
    )
    assert "did not answer" in warning["message"]


def test_a_reply_without_timing_is_not_a_rate(api) -> None:
    fake = FakeCoordinator(observe_reply={"usage": {}})
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))

    operation = _wait(
        client,
        operation_id,
        lambda body: PROBE_WARNING in [item["code"] for item in body["warnings"]],
    )

    assert operation["state"] == "ready"
    assert operation["observed"] is None


def test_a_reply_with_only_decode_records_the_measured_rate(api) -> None:
    """The distributed non-streaming path reports decode but not TTFT/prefill."""

    fake = FakeCoordinator(
        observe_reply={"usage": {"generation_tokens_per_second": 71.79}}
    )
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))

    operation = _wait(client, operation_id, lambda body: body["observed"] is not None)

    assert operation["observed"]["decode_tok_per_s"] == 71.79
    assert operation["observed"]["ttft_ms"] is None
    assert operation["observed"]["prefill_tok_per_s"] is None
    assert operation["observed"]["measured_at"]
    assert PROBE_WARNING not in [item["code"] for item in operation["warnings"]]


def test_the_plan_is_never_rewritten_by_a_measured_operation(api) -> None:
    fake = FakeCoordinator()
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    plan = applied(client, fake)
    operation_id = _start(client, plan)
    _wait(client, operation_id, lambda body: body["observed"] is not None)

    replanned = take_plan(client, plan["measurement_id"])

    assert replanned["observed"] is None
    # The measured operation never feeds the pure plan.
    assert replanned["predicted"] == plan["predicted"]
    assert "decode_tok_per_s" not in str(replanned["predicted"])


def test_the_probe_asks_for_the_coordinators_serving_model_id(api) -> None:
    fake = FakeCoordinator(models=[{"id": "serving-id", "model_path": MODEL_PATH}])
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))
    _wait(client, operation_id, lambda body: body["observed"] is not None)

    assert fake.observed_calls == ["serving-id"]


def test_streaming_fills_ttft_and_prefill_and_keeps_decode(api) -> None:
    """B2: the final streaming usage chunk supplies TTFT and prefill."""

    fake = FakeCoordinator(
        stream_observe_reply={
            "usage": {
                "time_to_first_token": 0.4,
                "prompt_tokens_per_second": 210.0,
            }
        }
    )
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))

    # Two probes fill the block in order: the non-streaming completion writes
    # decode + its own prefill/TTFT first (320/250 here), then the streaming
    # merge overwrites prefill/TTFT with the final usage chunk's values. A
    # wait on "observed is not None" alone can read the intermediate state
    # under parallel load, so wait for the streaming-derived values instead.
    operation = _wait(
        client,
        operation_id,
        lambda body: (
            body["observed"] is not None
            and body["observed"]["prefill_tok_per_s"] == 210.0
            and body["observed"]["ttft_ms"] == 400.0
        ),
    )

    observed = operation["observed"]
    # The streaming values won; the non-streaming decode survived the merge.
    assert observed["decode_tok_per_s"] == 42.0
    assert observed["prefill_tok_per_s"] == 210.0
    assert observed["ttft_ms"] == 400.0
    assert fake.stream_observed_calls == ["qwen"]
    assert operation["state"] == "ready"
    assert PROBE_WARNING not in [item["code"] for item in operation["warnings"]]


def test_a_streaming_probe_that_fails_keeps_decode_and_adds_one_warning(api) -> None:
    fake = FakeCoordinator(
        stream_observe_error=RuntimeError("the stream closed early")
    )
    client, _, _ = api(coordinator=fake.factory(), performance_probe=True)
    operation_id = _start(client, applied(client, fake))

    operation = _wait(
        client,
        operation_id,
        lambda body: PROBE_WARNING in [item["code"] for item in body["warnings"]],
    )

    assert operation["state"] == "ready"
    assert operation["observed"]["decode_tok_per_s"] == 42.0
    assert fake.stream_observed_calls == ["qwen"]
    warning = next(
        item for item in operation["warnings"] if item["code"] == PROBE_WARNING
    )
    assert "streaming" in warning["message"]
