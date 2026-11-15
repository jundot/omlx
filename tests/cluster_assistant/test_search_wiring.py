# SPDX-License-Identifier: Apache-2.0
"""The evaluate seam adapter (fake-driven): apply -> start -> measure sequencing.

The real apply/start/observe seams are mocked; the adapter's orchestration and
its measured/infeasible/unmeasured branching are tested off-cluster (AGENTS.md).
oMLX's planner split is not exercised here (plan_apply is mocked).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from omlx.cluster.assistant import search_wiring
from omlx.cluster.assistant.election import PlanCandidate
from omlx.cluster.assistant.operations import (
    STATE_APPLIED,
    STATE_READY,
    OperationProgress,
    OperationStore,
)
from omlx.cluster.assistant.start import OperationObserved


def _candidate() -> PlanCandidate:
    return PlanCandidate(
        rank_order=("a", "b"), stages=(), feasible=True, reason="estimate",
        plan_hash="x" * 16, context_tokens=32768,
    )


def _record() -> SimpleNamespace:
    return SimpleNamespace(
        measurement_id="ms-1",
        model=SimpleNamespace(model_path="/models/m", mtp=False),
    )


def _fake_plan() -> SimpleNamespace:
    return SimpleNamespace(
        coordinator=SimpleNamespace(node_id="a", local=True),
        rank_order=("a", "b"),
        endpoint=None,
        request=object(),
    )


def _setup(tmp_path, monkeypatch, *, apply_ok=True, start_ok=True, observed=True):
    store = OperationStore(tmp_path / "store")
    plan = _fake_plan()
    monkeypatch.setattr(search_wiring, "plan_apply", lambda *a, **k: plan)

    def fake_apply(store_, op_id, plan_, ops, **kwargs):
        if apply_ok:
            store_.advance(op_id, STATE_APPLIED, progress=OperationProgress(phase="applied"))

    async def fake_start(store_, op_id, request, ops, **kwargs):
        if start_ok:
            store_.advance(op_id, STATE_READY, progress=OperationProgress(phase="ready"))

    async def fake_observe(store_, op_id, ops, model_path):
        if observed:
            return OperationObserved(
                prefill_tok_per_s=500.0, decode_tok_per_s=50.0, ttft_ms=80.0,
                measured_at="2026-11-13T00:00:00+00:00",
            )
        return None

    monkeypatch.setattr(search_wiring, "run_apply_job", fake_apply)
    monkeypatch.setattr(search_wiring, "run_start_job", fake_start)
    monkeypatch.setattr(search_wiring, "observe_performance", fake_observe)
    return store


def _evaluate(store, **kwargs):
    return asyncio.run(
        search_wiring.evaluate_constellation(
            _candidate(),
            record=_record(),
            store=store,
            seams=None,
            ops_for=lambda plan: SimpleNamespace(),
            **kwargs,
        )
    )


def test_ready_deployment_is_measured(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch)
    result = _evaluate(store)
    assert result.feasible is True
    assert result.decode_tok_per_s == 50.0
    assert result.prefill_tok_per_s == 500.0
    assert result.ttft_ms == 80.0
    assert result.rank_order == ("a", "b")
    assert result.measured is True


def test_apply_failure_is_infeasible(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch, apply_ok=False)
    result = _evaluate(store)
    assert result.feasible is False
    assert result.measured is False


def test_start_failure_is_feasible_but_unmeasured(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch, start_ok=False)
    result = _evaluate(store)
    assert result.feasible is True
    assert result.measured is False
    assert result.decode_tok_per_s is None


def test_ready_without_timing_is_unmeasured(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch, observed=False)
    result = _evaluate(store)
    assert result.feasible is True
    assert result.measured is False


def test_plan_refusal_is_infeasible(tmp_path, monkeypatch) -> None:
    from omlx.cluster.assistant.provisioning import ProvisioningError

    store = OperationStore(tmp_path / "store")

    def refuse(*a, **k):
        raise ProvisioningError("plan_mismatch", "drifted", "plan")

    monkeypatch.setattr(search_wiring, "plan_apply", refuse)
    result = _evaluate(store)
    assert result.feasible is False

def _ops_with_settings(calls: list[str], *, write_ok: bool = True):
    def models():
        calls.append("models")
        return {"models": [{"id": "m", "model_path": "/models/m"}]}

    def write_model_settings(model_id, patch):
        calls.append("write")
        assert model_id == "m"
        assert patch == {"mtp_adaptive_max_depth": 2}
        if not write_ok:
            raise RuntimeError("the coordinator refused the settings write")
        return {"success": True}

    return SimpleNamespace(models=models, write_model_settings=write_model_settings)


def test_a_settings_patch_is_written_before_the_launch(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch)
    calls: list[str] = []

    # Record the start call too, so the ordering is asserted, not assumed.
    original_start = search_wiring.run_start_job

    async def tracking_start(store_, op_id, request, ops, **kwargs):
        calls.append("start")
        await original_start(store_, op_id, request, ops, **kwargs)

    monkeypatch.setattr(search_wiring, "run_start_job", tracking_start)

    result = asyncio.run(
        search_wiring.evaluate_constellation(
            _candidate(),
            record=_record(),
            store=store,
            seams=None,
            ops_for=lambda plan: _ops_with_settings(calls),
            model_settings_patch={"mtp_adaptive_max_depth": 2},
        )
    )

    assert result.feasible is True
    assert result.decode_tok_per_s == 50.0
    assert calls == ["models", "write", "start"]


def test_a_failed_settings_write_drops_the_config(tmp_path, monkeypatch) -> None:
    store = _setup(tmp_path, monkeypatch)
    calls: list[str] = []

    result = asyncio.run(
        search_wiring.evaluate_constellation(
            _candidate(),
            record=_record(),
            store=store,
            seams=None,
            ops_for=lambda plan: _ops_with_settings(calls, write_ok=False),
            model_settings_patch={"mtp_adaptive_max_depth": 2},
        )
    )

    # Measuring the baseline while claiming the tuned config would be a lie.
    assert result.feasible is False
    assert "start" not in calls
