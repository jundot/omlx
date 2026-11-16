# SPDX-License-Identifier: Apache-2.0
"""The /tune route (fake-driven): fine-tune a ready winner, refuse non-ready."""

from __future__ import annotations

from omlx.cluster.assistant import tuning_wiring
from omlx.cluster.assistant.operations import (
    KIND_START,
    STATE_READY,
    OperationEndpoint,
    OperationProgress,
)
from omlx.cluster.assistant.tuning import FineTuneResult

TUNE_URL = "/admin/api/cluster/assistant/tune"
TUNING_PRESET_URL = "/admin/api/cluster/assistant/tuning-preset"


def _ready_winner(store, *, local=True, rank_order=("a", "b")):
    endpoint = OperationEndpoint(node="a", host="127.0.0.1", port=8099, ready=True)
    op = store.start(
        kind=KIND_START, measurement_id="ms-1", coordinator_node="a",
        coordinator_local=local, rank_order=list(rank_order),
        model_path="/models/dev-1.5b", endpoint=endpoint,
    )
    store.advance(
        op.operation_id, STATE_READY,
        progress=OperationProgress(phase="ready"), endpoint=endpoint,
    )
    return op


def test_tune_runs_fine_tuning_on_a_ready_winner(api, monkeypatch) -> None:
    client, _, store = api()
    op = _ready_winner(store)

    async def fake_fine_tune(complete, *, model_id, **kwargs):
        assert model_id == "dev-1.5b"
        return FineTuneResult(
            chunk=4096, sampler={"sampler": "deterministic"}, mtp=True,
            chunk_rows=[{"chunk": 4096, "tok_per_s": 50.0}],
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake_fine_tune)
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["chunk"] == 4096
    assert body["sampler"]["sampler"] == "deterministic"
    assert body["mtp"] is True
    assert body["model_id"] == "dev-1.5b"


def test_tune_refuses_a_non_ready_operation(api) -> None:
    client, _, store = api()
    op = store.start(kind=KIND_START, measurement_id="ms-1", coordinator_node="a")
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_ready"


def test_tune_refuses_an_unknown_operation(api) -> None:
    client, _, _ = api()
    response = client.post(TUNE_URL, json={"operation_id": "op-nope"})
    assert response.status_code == 409
    assert response.json()["code"] == "operation_not_ready"

def test_tune_skips_mtp_for_a_model_without_an_mtp_head(api, monkeypatch) -> None:
    # The winner's model has no MTP head (the fake path is unreadable), so the
    # route passes mtp_capable=False and the MTP A/B is not applicable.
    client, _, store = api()
    op = _ready_winner(store)
    captured = {}

    async def fake(complete, *, model_id, mtp_capable=True, **kwargs):
        captured["mtp_capable"] = mtp_capable
        return FineTuneResult(
            chunk=2048, sampler={"sampler": "balanced"}, mtp=None,
            mtp_applicable=mtp_capable,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake)
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert captured["mtp_capable"] is False
    assert body["mtp"] is None
    assert body["mtp_applicable"] is False


def test_tune_marks_chunk_not_applicable_via_a_request_field(api, monkeypatch) -> None:
    # The chunk is a model setting, not a chat-request field: the route tells the
    # tuner chunk_applicable=False and reports it, instead of a noise sweep.
    client, _, store = api()
    op = _ready_winner(store)
    captured = {}

    async def fake(complete, *, model_id, mtp_capable=True, chunk_applicable=True, **kw):
        captured["chunk_applicable"] = chunk_applicable
        return FineTuneResult(
            chunk=None, chunk_applicable=chunk_applicable,
            sampler={"sampler": "balanced"}, mtp=None, mtp_applicable=mtp_capable,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake)
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text
    body = response.json()
    assert captured["chunk_applicable"] is False
    assert body["chunk"] is None
    assert body["chunk_applicable"] is False
    assert body["sampler"]["sampler"] == "balanced"


def test_tune_saves_the_preset_per_model_and_hardware(api, monkeypatch) -> None:
    # Tuning is expensive; the winning knobs are saved per (model, node config)
    # so a later Start can reuse them without re-tuning.
    client, _, store = api()
    op = _ready_winner(store)

    async def fake(complete, *, model_id, mtp_capable=True, chunk_applicable=True, **kw):
        return FineTuneResult(
            chunk=None, chunk_applicable=False,
            sampler={"sampler": "balanced"}, mtp=True, mtp_applicable=True,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake)
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    body = response.json()
    assert body["preset_key"]

    from omlx.cluster.assistant import routes as assistant_routes

    preset = assistant_routes._tuning_store().get(
        "/models/dev-1.5b", ",".join(["a", "b"])
    )
    assert preset is not None
    assert preset.sampler["sampler"] == "balanced"
    assert preset.mtp is True


def test_tune_never_erases_an_mtp_depth_a_redeploy_saved(api, monkeypatch) -> None:
    # One combined preset: the sampler sweep must merge into the depth the
    # redeploy sweep measured, not overwrite it with None (PHASE-2B-RUN.md §26).
    client, _, store = api()
    op = _ready_winner(store)

    from omlx.cluster.assistant import routes as assistant_routes
    from omlx.cluster.assistant.tuning_store import TuningPreset

    assistant_routes._tuning_store().save(
        TuningPreset(
            model_signature="/models/dev-1.5b",
            hardware_key="a,b",
            mtp=True,
            mtp_depth=3,
            measured_at="2026-11-14T00:00:00+00:00",
        )
    )

    async def fake(complete, *, model_id, mtp_capable=True, chunk_applicable=True, **kw):
        return FineTuneResult(
            chunk=None, chunk_applicable=False,
            sampler={"sampler": "deterministic"}, mtp=True, mtp_applicable=True,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake)
    response = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert response.status_code == 200, response.text

    preset = assistant_routes._tuning_store().get("/models/dev-1.5b", "a,b")
    assert preset is not None
    assert preset.mtp_depth == 3
    assert preset.sampler["sampler"] == "deterministic"


def test_the_saved_preset_is_readable_for_an_operation(api, monkeypatch) -> None:
    # The card shows what Start will replay: the preset for the operation's
    # model and its SORTED node set (a flipped rank 0 must still find it).
    client, _, store = api()
    op = _ready_winner(store, rank_order=("b", "a"))

    async def fake(complete, *, model_id, mtp_capable=True, chunk_applicable=True, **kw):
        return FineTuneResult(
            chunk=None, chunk_applicable=False,
            sampler={"sampler": "balanced", "temperature": 0.6},
            mtp=True, mtp_applicable=True,
        )

    monkeypatch.setitem(tuning_wiring.__dict__, "fine_tune_constellation", fake)
    tuned = client.post(TUNE_URL, json={"operation_id": op.operation_id})
    assert tuned.status_code == 200, tuned.text

    response = client.get(
        TUNING_PRESET_URL, params={"operation_id": op.operation_id}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model_signature"] == "/models/dev-1.5b"
    assert body["hardware_key"] == "a,b"
    assert body["mtp"] is True
    assert body["mtp_depth"] is None
    assert body["sampler"]["sampler"] == "balanced"
    assert body["sampler"]["temperature"] == 0.6
    assert body["measured_at"]
    assert body["key"] == tuned.json()["preset_key"]


def test_the_preset_read_says_none_when_nothing_was_tuned(api) -> None:
    client, _, store = api()
    op = _ready_winner(store)
    response = client.get(
        TUNING_PRESET_URL, params={"operation_id": op.operation_id}
    )
    assert response.status_code == 200, response.text
    assert response.json() == {"preset": None}


def test_the_preset_read_refuses_an_unknown_operation(api) -> None:
    client, _, _ = api()
    response = client.get(TUNING_PRESET_URL, params={"operation_id": "op-nope"})
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "operation_not_found"
    assert body["action"] == "status"
