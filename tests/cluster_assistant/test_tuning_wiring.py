# SPDX-License-Identifier: Apache-2.0
"""Fine-tuning wiring (fake-driven): oMLX reply -> TuneSample -> sweeps."""

from __future__ import annotations

import asyncio

from omlx.cluster.assistant.tuning_wiring import (
    completion_seam,
    fine_tune_constellation,
    tune_sample_from_reply,
)


def _reply(rate=50.0, answer="red, green, blue", completion=10, reasoning=0):
    return {
        "usage": {
            "generation_tokens_per_second": rate,
            "completion_tokens": completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
        "choices": [{"message": {"content": answer}}],
    }


def _run(coro):
    return asyncio.run(coro)


def test_tune_sample_reads_usage_and_answer() -> None:
    sample = tune_sample_from_reply(_reply(rate=64.0, answer="Paris", completion=7, reasoning=3))
    assert sample.ok is True
    assert sample.tok_per_s == 64.0
    assert sample.answer == "Paris"
    assert sample.completion_tokens == 7
    assert sample.reasoning_tokens == 3


def test_tune_sample_no_decode_is_not_ok() -> None:
    assert tune_sample_from_reply({"usage": {}}).ok is False
    assert tune_sample_from_reply({"usage": {"generation_tokens_per_second": 0}}).ok is False
    assert tune_sample_from_reply(None).ok is False
    assert tune_sample_from_reply({}).ok is False


def test_completion_seam_sync_and_async() -> None:
    def sync_complete(body):
        return _reply(rate=42.0)

    async def async_complete(body):
        return _reply(rate=43.0)

    assert _run(completion_seam(sync_complete)({})).tok_per_s == 42.0
    assert _run(completion_seam(async_complete)({})).tok_per_s == 43.0


def test_completion_seam_failure_is_skipped() -> None:
    def boom(body):
        raise RuntimeError("coordinator unreachable")

    assert _run(completion_seam(boom)({})).ok is False


def test_fine_tune_constellation_orchestrates() -> None:
    def complete(body):
        if "prefill_chunk_size" in body:
            return _reply(rate=50.0)
        if "mtp" in body:
            return _reply(rate=60.0 if body["mtp"] else 40.0)
        return _reply(rate=45.0, answer="red, green, blue 42 paris")

    result = _run(fine_tune_constellation(complete, model_id="m"))
    assert result.chunk is not None
    assert result.sampler is not None
    assert result.mtp is True