# SPDX-License-Identifier: Apache-2.0
"""Fine-tuning sweeps (fake-driven): real-completion orchestration over a seam.

The completion seam is faked; the sweep logic, the lexicographic sampler
objective, the golden-prompt scoring and the hardware-keyed presets are tested
off-cluster (AGENTS.md). Ported methodology from LocalInferenz tuning/profiles.
"""

from __future__ import annotations

import asyncio

from omlx.cluster.assistant.tuning import (
    TuneSample,
    best_chunk,
    best_sampler,
    fine_tune,
    preset_key,
    score_quality,
    sweep_chunks,
    sweep_mtp,
    sweep_samplers,
)


def _run(coro):
    return asyncio.run(coro)


def test_score_quality_counts_expected_hits() -> None:
    assert score_quality("red, green, blue", ("red", "green", "blue")) == 1.0
    assert score_quality("red and blue", ("red", "green", "blue")) == 2 / 3
    assert score_quality("nothing", ("red", "green", "blue")) == 0.0
    assert score_quality("PARIS", ("paris",)) == 1.0


def test_sweep_chunks_keeps_measured_and_skips_failed() -> None:
    async def seam(body):
        chunk = body["prefill_chunk_size"]
        if chunk == 4096:
            return TuneSample(ok=False)
        return TuneSample(ok=True, tok_per_s={2048: 30.0, 8192: 50.0}[chunk])

    rows = _run(sweep_chunks(seam, model_id="m"))
    assert {r["chunk"] for r in rows} == {2048, 8192}
    assert best_chunk(rows) == 8192


def test_best_chunk_empty_is_none() -> None:
    assert best_chunk([]) is None


def test_sweep_samplers_lexicographic_correctness_first() -> None:
    # A fast but wrong sampler must lose to a slower correct one.
    async def seam(body):
        temp = body["temperature"]
        if temp == 0.0:  # deterministic: correct, fewer reasoning tokens
            return TuneSample(
                ok=True, answer="red, green, blue 42 paris",
                completion_tokens=10, reasoning_tokens=0, tok_per_s=40.0,
            )
        # loose: fast but wrong answers, more reasoning bloat
        return TuneSample(
            ok=True, answer="wrong", completion_tokens=50,
            reasoning_tokens=30, tok_per_s=90.0,
        )

    rows = _run(sweep_samplers(seam, model_id="m"))
    winner = best_sampler(rows)
    assert winner["sampler"] == "deterministic"
    assert winner["quality"] == 1.0


def test_best_sampler_tie_breaks_on_reasoning_then_output() -> None:
    rows = [
        {"sampler": "a", "quality": 1.0, "reasoning_tokens": 5,
         "completion_tokens": 10, "tok_per_s": 40.0},
        {"sampler": "b", "quality": 1.0, "reasoning_tokens": 2,
         "completion_tokens": 10, "tok_per_s": 30.0},
    ]
    assert best_sampler(rows)["sampler"] == "b"  # fewer reasoning tokens


def test_sweep_mtp_faster_measured_side_wins() -> None:
    async def seam(body):
        return TuneSample(ok=True, tok_per_s=64.0 if body["mtp"] else 38.0)

    result = _run(sweep_mtp(seam, model_id="m"))
    assert result["mtp"] is True
    assert result["rates"] == {False: 38.0, True: 64.0}


def test_sweep_mtp_unmeasured_side_never_wins() -> None:
    async def seam(body):
        if body["mtp"]:
            return TuneSample(ok=False)
        return TuneSample(ok=True, tok_per_s=38.0)

    result = _run(sweep_mtp(seam, model_id="m"))
    assert result["mtp"] is False  # only off measured


def test_sweep_mtp_neither_measured_is_none() -> None:
    async def seam(body):
        return TuneSample(ok=False)

    assert _run(sweep_mtp(seam, model_id="m"))["mtp"] is None


def test_preset_key_is_hardware_bound() -> None:
    key = preset_key("model-sig", "m5max+m2ultra")
    assert key != preset_key("model-sig", "m1max+m1max")  # other hardware
    assert key != preset_key("other-model", "m5max+m2ultra")
    assert preset_key("model-sig", "hw") == preset_key("model-sig", "hw")


def test_fine_tune_orchestrates_the_sweeps() -> None:
    async def seam(body):
        if "prefill_chunk_size" in body:
            return TuneSample(ok=True, tok_per_s=50.0)
        if "mtp" in body:
            return TuneSample(ok=True, tok_per_s=60.0 if body["mtp"] else 40.0)
        return TuneSample(
            ok=True, answer="red, green, blue 42 paris",
            completion_tokens=10, reasoning_tokens=0, tok_per_s=45.0,
        )

    result = _run(fine_tune(seam, model_id="m"))
    assert result.chunk in (2048, 4096, 8192)
    assert result.sampler is not None
    assert result.mtp is True
    assert result.chunk_rows and result.sampler_rows

def test_sweep_mtp_is_not_applicable_without_an_mtp_head() -> None:
    # A model without an MTP head cannot draft: the A/B is skipped, not measured
    # as two identical no-ops (the request `mtp` field is not a real toggle).
    calls = []

    async def seam(body):
        calls.append(body)
        return TuneSample(ok=True, tok_per_s=50.0)

    result = _run(sweep_mtp(seam, model_id="m", mtp_capable=False))
    assert result["applicable"] is False
    assert result["mtp"] is None
    assert result["rates"] == {}
    assert calls == []  # nothing measured


def test_chunk_sweep_is_not_applicable_via_a_request_field() -> None:
    # The chunk is a per-model setting, not a chat-request field: a per-request
    # sweep measures noise, so fine_tune skips it when chunk_applicable=False.
    calls = []

    async def seam(body):
        calls.append(body)
        if "prefill_chunk_size" in body:
            raise AssertionError("chunk must not be sent as a request field")
        return TuneSample(ok=True, answer="red, green, blue 42 paris",
                          completion_tokens=5, tok_per_s=40.0)

    result = _run(fine_tune(seam, model_id="m", chunk_applicable=False))
    assert result.chunk is None
    assert result.chunk_applicable is False
    assert result.chunk_rows == []
    # The sampler (a real request field) still ran.
    assert result.sampler is not None
