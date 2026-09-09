# SPDX-License-Identifier: Apache-2.0
"""K2 tuner isolation, measured recommendations, and cancellation."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from omlx.admin import ane_tuning
from omlx.model_settings import ModelSettings


@pytest.fixture
def tuner_pool(monkeypatch, tmp_path):
    from omlx.custom_kernels.qwen35_prefill import fast

    base = ModelSettings(qwen35_ane_prefill_enabled=True)
    (tmp_path / "config.json").write_text(json.dumps(dict(num_hidden_layers=3)))
    pool = SimpleNamespace(
        base=base,
        get_entry=lambda _: SimpleNamespace(model_path=tmp_path),
        _settings_manager=SimpleNamespace(get_settings=lambda _: base),
        get_loaded_model_ids=lambda: ["model"],
        _unload_engine=AsyncMock(),
    )
    restored = Mock()
    monkeypatch.setattr(ane_tuning, "_pin_speed_priority", lambda _: "old")
    monkeypatch.setattr(ane_tuning, "_restore_speed_priority", restored)
    monkeypatch.setattr(fast, "qwen35_ane_available", lambda: True)
    monkeypatch.setattr(fast, "_ext", SimpleNamespace(ane_compile_program=object()))
    yield pool
    assert base.qwen35_ane_prefill_enabled
    assert pool._unload_engine.await_count == 2
    restored.assert_called_once_with(pool, "old")


@pytest.mark.asyncio
@pytest.mark.parametrize("peak", [100.5, 101])
async def test_k2_recommendation_keeps_settings_unchanged(
    monkeypatch, tuner_pool, peak
):
    base = tuner_pool.base.to_dict()
    speeds = iter([100, peak, 100])

    async def measure(run, pool, settings, candidate):
        transient = ane_tuning._settings_for_candidate(settings, run.request, candidate)
        assert transient.qwen35_ane_prefill_enabled == candidate.enabled
        return {**ane_tuning._empty_result(candidate), "processing_tps": next(speeds)}

    monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    await ane_tuning.run_tuning(run, tuner_pool)
    assert run.status == "completed" and run.current == run.total
    assert run.recommendation["enabled"] == (peak >= 101)
    assert tuner_pool.base.to_dict() == base


def test_tuner_candidates_follow_dense_and_shared_geometry():
    dense = ane_tuning._k2_candidates(dict(num_hidden_layers=64))
    sparse = ane_tuning._k2_candidates(dict(num_hidden_layers=61, num_experts=192))
    assert len(dense) == 3
    assert len(sparse) == 3
    assert all(row.shared_fraction > 0 for row in sparse[1:])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,cleanup_fails",
    [
        ("budget", False),
        ("success", False),
        ("cancelled", False),
        ("error", False),
        ("success", True),
    ],
)
async def test_k2_cleanup_keeps_run_active(
    monkeypatch, tuner_pool, outcome, cleanup_fails
):
    now = [0]
    monkeypatch.setattr(ane_tuning, "time", SimpleNamespace(monotonic=lambda: now[0]))
    engine = SimpleNamespace(stream_generate=AsyncMock())

    async def load(*args, **kwargs):
        now[0] = 181
        return engine

    tuner_pool.get_engine = AsyncMock(side_effect=load)
    cleaning, finish_cleanup = asyncio.Event(), asyncio.Event()

    async def unload(_):
        if tuner_pool._unload_engine.await_count == 2:
            cleaning.set()
            await finish_cleanup.wait()
            if cleanup_fails:
                raise RuntimeError("cleanup failed")

    tuner_pool._unload_engine.side_effect = unload
    if outcome != "budget":
        error = {
            "cancelled": asyncio.CancelledError(),
            "error": RuntimeError("measurement failed"),
        }.get(outcome)
        measure = AsyncMock(
            side_effect=error,
            return_value={
                **ane_tuning._empty_result(
                    ane_tuning._Candidate("GPU", False, backend="k2")
                ),
                "processing_tps": 100,
            },
        )
        monkeypatch.setattr(ane_tuning, "_measure_candidate", measure)
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(model_id="model", backend="k2")
    )
    monkeypatch.setattr(ane_tuning, "_runs", {run.tuning_id: run})
    run.task = asyncio.create_task(ane_tuning.run_tuning(run, tuner_pool))
    try:
        await asyncio.wait_for(cleaning.wait(), timeout=2)
        assert run.status == "running" and run.phase == "cleaning_up"
        assert ane_tuning.get_active_run() is run
        from omlx.admin.routes import cancel_ane_tuning

        response = await cancel_ane_tuning(run.tuning_id, is_admin=True)
        assert response["status"] == "cleaning_up" and not run.task.cancelling()
    finally:
        finish_cleanup.set()
        await run.task
    expected = "completed" if outcome in ("budget", "success") else outcome
    assert run.status == ("error" if cleanup_fails else expected)
    assert (run.recommendation is not None) == (
        outcome == "success" and not cleanup_fails
    )
    assert ane_tuning.get_active_run() is None
    if cleanup_fails:
        assert "cleanup failed" in run.error_message
    elif outcome == "budget":
        assert "Inconclusive" in run.message and "unchanged" in run.termination_reason
    engine.stream_generate.assert_not_called()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("bits", [None, 4, 8])
@pytest.mark.parametrize("shared", [False, True])
def test_tuning_eligibility_uses_mlp_weight_format(dtype, bits, shared):
    import mlx.core as mx
    import mlx.nn as nn

    projections = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = nn.Linear(64, 64, bias=False)
        linear.set_dtype(getattr(mx, dtype))
        projections[name] = (
            linear.to_quantized(group_size=64, bits=8 if name == "down_proj" else bits)
            if bits
            else linear
        )
    mlp = SimpleNamespace(**projections)
    layer = SimpleNamespace(mlp=SimpleNamespace(shared_experts=mlp) if shared else mlp)
    model = SimpleNamespace(layers=[layer, object()])
    if bits is None:
        with pytest.raises(ValueError, match="eight bits"):
            ane_tuning._validate_k2_tuning_model(model)
    else:
        ane_tuning._validate_k2_tuning_model(model)
        # A checkpoint-level quantization marker must not hide a float down projection.
        mlp.down_proj = nn.Linear(64, 64, bias=False)
        with pytest.raises(ValueError, match="eight bits"):
            ane_tuning._validate_k2_tuning_model(model)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,operations", [(False, 0), (True, 0), (True, 12)])
async def test_long_prefill_measurement_requires_native_execution(
    monkeypatch, enabled, operations
):
    from omlx.custom_kernels.qwen35_prefill import fast

    warmups = []

    async def stream(**options):
        warmups.append(options)
        if False:
            yield

    engine = SimpleNamespace(
        tokenizer=object(),
        stream_generate=stream,
        _model=SimpleNamespace(_omlx_k2_ane_prefill_count=2),
    )
    validate = Mock()
    monkeypatch.setattr(ane_tuning, "_validate_k2_tuning_model", validate)
    monkeypatch.setattr(
        ane_tuning, "_generate_prompt", lambda _, length, profile: [1] * length
    )
    measure = AsyncMock(side_effect=[{"processing_tps": v} for v in (101, 1000, 103)])
    monkeypatch.setattr(ane_tuning, "_run_single_test", measure)
    monkeypatch.setattr(fast, "qwen35_ane_profile_set_enabled", lambda _: True)
    monkeypatch.setattr(fast, "qwen35_ane_profile_reset", lambda: None)
    monkeypatch.setattr(
        fast, "qwen35_ane_profile_snapshot", lambda: {"mlp": {"operations": operations}}
    )
    run = ane_tuning.create_run(
        ane_tuning.ANETuningRequest(
            model_id="model", backend="k2", sequence_length=1024
        )
    )
    candidate = ane_tuning._Candidate("test", enabled, backend="k2")
    call = ane_tuning._measure_candidate(
        run,
        SimpleNamespace(get_engine=AsyncMock(return_value=engine)),
        ModelSettings(),
        candidate,
    )
    if enabled and not operations:
        with pytest.raises(RuntimeError, match="native execution"):
            await call
    else:
        result = await call
        assert result["processing_tps"] == 103
        assert result["samples"] == [101, 1000, 103]
    assert validate.call_count == (0 if enabled else 1)
    assert len(warmups[0]["prompt"]) == 1025 and warmups[0]["skip_cache_store"]
    assert measure.await_count == 3
    assert all(
        c.kwargs["pp_len"] == 4097 and c.kwargs["max_tokens"] == 2
        for c in measure.await_args_list
    )
