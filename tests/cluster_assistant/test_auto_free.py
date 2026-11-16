# SPDX-License-Identifier: Apache-2.0
"""Automatic free-disk on apply: the rule selects, the route deletes and retries."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.responses import JSONResponse

from omlx.cluster.assistant import routes as assistant_routes


def _record():
    return SimpleNamespace(
        model=SimpleNamespace(model_path="/m/Planned"),
        inventories={
            "n1": [
                {"model_id": "Planned", "model_path": "/m/Planned", "size_bytes": 999},
                {"model_id": "Old", "model_path": "/m/Old", "size_bytes": 200},
            ],
            "n2": [
                {"model_id": "Old", "model_path": "/m/Old", "size_bytes": 200},
            ],
        },
    )


def test_auto_free_deletes_the_safe_model_and_reports_freed_bytes(monkeypatch) -> None:
    record = _record()
    deleted_calls: list[tuple] = []

    def fake_delete(*, port, model_name, target=""):
        deleted_calls.append((port, model_name, target))
        return {"deleted": True}

    monkeypatch.setitem(assistant_routes.__dict__, "delete_model_on_mac", fake_delete)
    monkeypatch.setitem(
        assistant_routes.__dict__, "_delete_target", lambda record, node_id: ("", 8099)
    )

    deleted, freed = asyncio.run(
        assistant_routes._auto_free_space(
            record, {"node_id": "n1", "needed_bytes": 500, "free_bytes": 400}
        )
    )

    # The planned model is never deleted; the shared "Old" copy is.
    assert [call[1] for call in deleted_calls] == ["Old"]
    assert deleted[0]["model_id"] == "Old"
    assert freed == {"n1": 200}


def test_auto_free_does_nothing_when_no_safe_candidate_exists(monkeypatch) -> None:
    record = _record()
    monkeypatch.setitem(
        assistant_routes.__dict__,
        "delete_model_on_mac",
        lambda **kwargs: pytest.fail("must not delete the planned/last copy"),
    )
    monkeypatch.setitem(
        assistant_routes.__dict__, "_delete_target", lambda record, node_id: ("", 8099)
    )
    # "Planned" is in use and "Old" is not on this Mac, so nothing may be freed.
    deleted, freed = asyncio.run(
        assistant_routes._auto_free_space(
            record, {"node_id": "n2b", "needed_bytes": 500, "free_bytes": 0}
        )
    )
    assert deleted == []
    assert freed == {}


def test_plan_with_auto_free_retries_with_the_freed_bytes(monkeypatch) -> None:
    record = _record()
    refusal = JSONResponse(
        status_code=409,
        content={
            "code": "insufficient_disk",
            "message": "short",
            "action": "free_disk_space",
            "details": {"node_id": "n1", "needed_bytes": 500, "free_bytes": 400},
        },
    )
    plan_marker = object()
    overrides: list = []

    def fake_plan(record, inputs, *, free_override=None):
        overrides.append(free_override)
        if free_override is None:
            return refusal
        return plan_marker

    async def fake_free(record, details):
        return [{"node_id": "n1", "model_id": "Old"}], {"n1": 200}

    monkeypatch.setitem(assistant_routes.__dict__, "_planned_deployment", fake_plan)
    monkeypatch.setitem(assistant_routes.__dict__, "_auto_free_space", fake_free)

    plan, warnings = asyncio.run(
        assistant_routes._plan_with_auto_free(record, object())
    )

    assert plan is plan_marker
    assert overrides == [None, {"n1": 200}]
    assert [w.code for w in warnings] == [assistant_routes.WARNING_SPACE_FREED]
    assert "Old" in warnings[0].message


def test_plan_with_auto_free_returns_the_refusal_when_nothing_can_be_freed(
    monkeypatch,
) -> None:
    record = _record()
    refusal = JSONResponse(
        status_code=409,
        content={
            "code": "insufficient_disk",
            "message": "short",
            "action": "free_disk_space",
            "details": {"node_id": "n1", "needed_bytes": 500, "free_bytes": 400},
        },
    )

    async def fake_free(record, details):
        return [], {}

    monkeypatch.setitem(
        assistant_routes.__dict__, "_planned_deployment", lambda *a, **k: refusal
    )
    monkeypatch.setitem(assistant_routes.__dict__, "_auto_free_space", fake_free)

    plan, warnings = asyncio.run(
        assistant_routes._plan_with_auto_free(record, object())
    )
    assert plan is refusal
    assert warnings == []
