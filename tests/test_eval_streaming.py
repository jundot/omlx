# SPDX-License-Identifier: Apache-2.0
"""Tests for refill-on-completion eval scheduling."""

import asyncio
import time
from types import SimpleNamespace

import pytest

from omlx.eval.base import BaseBenchmark, EvalGenerated, QuestionResult


class _StreamingBenchmark(BaseBenchmark):
    name = "streaming-test"

    async def load_dataset(self, sample_size: int = 0) -> list[dict]:
        return []

    def format_prompt(self, item: dict) -> list[dict[str, str]]:
        return [{"role": "user", "content": str(item["id"])}]

    def extract_answer(self, response: str, item: dict) -> str:
        return response.strip()

    def check_answer(self, predicted: str, item: dict) -> bool:
        return predicted == item.get("answer", "ok")


class _FakeEngine:
    def __init__(
        self, delays=None, responses=None, thinking_response: str | None = None
    ):
        self.delays = delays or {}
        self.responses = responses or {}
        self.thinking_response = thinking_response
        self.started: dict[int, float] = {}
        self.finished: dict[int, float] = {}
        self.calls: list[tuple[int, bool]] = []

    async def chat(self, messages, **kwargs):
        item_id = int(messages[0]["content"])
        ct_kwargs = kwargs.get("chat_template_kwargs") or {}
        enable_thinking = bool(ct_kwargs.get("enable_thinking"))
        self.calls.append((item_id, enable_thinking))
        self.started[item_id] = time.perf_counter()
        await asyncio.sleep(self.delays.get(item_id, 0.0))
        self.finished[item_id] = time.perf_counter()
        if self.thinking_response is not None and not enable_thinking:
            return SimpleNamespace(text=self.thinking_response)
        return SimpleNamespace(text=self.responses.get(item_id, "ok"))


@pytest.mark.asyncio
async def test_refills_generation_slot_before_slow_request_finishes():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(delays={0: 0.05, 1: 0.005, 2: 0.005})

    await bench.run(
        engine,
        [{"id": 0}, {"id": 1}, {"id": 2}],
        batch_size=2,
        enable_thinking=True,
    )

    assert engine.started[2] < engine.finished[0]


@pytest.mark.asyncio
async def test_progress_is_per_question_and_results_keep_input_order():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(delays={0: 0.03, 1: 0.005, 2: 0.01})
    progress = []

    async def on_progress(current, total):
        progress.append((current, total))

    result = await bench.run(
        engine,
        [{"id": 0}, {"id": 1}, {"id": 2}],
        on_progress=on_progress,
        batch_size=2,
        enable_thinking=True,
    )

    assert progress == [(1, 3), (2, 3), (3, 3)]
    assert [qr.question_id for qr in result.question_results] == ["0", "1", "2"]


@pytest.mark.asyncio
async def test_per_question_time_is_not_chunk_average():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(delays={0: 0.03, 1: 0.005})

    result = await bench.run(
        engine,
        [{"id": 0}, {"id": 1}],
        batch_size=2,
        enable_thinking=True,
    )

    times = {qr.question_id: qr.time_seconds for qr in result.question_results}
    assert times["0"] > times["1"]


@pytest.mark.asyncio
async def test_progress_cancellation_propagates():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(delays={0: 0.001, 1: 0.02})

    async def on_progress(current, total):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await bench.run(
            engine,
            [{"id": 0}, {"id": 1}],
            on_progress=on_progress,
            batch_size=2,
            enable_thinking=True,
        )


@pytest.mark.asyncio
async def test_thinking_probe_reruns_probe_and_streams_remaining_with_thinking():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(thinking_response="<think>hidden</think>ok")

    result = await bench.run(
        engine,
        [{"id": 0}, {"id": 1}, {"id": 2}],
        batch_size=2,
        enable_thinking=False,
    )

    assert result.thinking_used is True
    assert engine.calls == [
        (0, False),
        (1, False),
        (0, True),
        (1, True),
        (2, True),
    ]


@pytest.mark.asyncio
async def test_slow_scoring_does_not_block_generation_refill():
    bench = _StreamingBenchmark()
    engine = _FakeEngine(delays={0: 0.001, 1: 0.001, 2: 0.001})
    score_finished: dict[int, float] = {}

    def score_generated(generated: EvalGenerated) -> QuestionResult:
        time.sleep(0.05)
        score_finished[generated.index] = time.perf_counter()
        return QuestionResult(
            question_id=str(generated.index),
            correct=True,
            expected="ok",
            predicted="ok",
            time_seconds=generated.generation_seconds,
        )

    await bench._run_refill_queue(
        engine,
        list(enumerate([{"id": 0}, {"id": 1}, {"id": 2}])),
        batch_size=1,
        sampling_kwargs=None,
        enable_thinking=True,
        score_generated=score_generated,
        on_progress=None,
        total_items=3,
        score_concurrency=1,
    )

    assert engine.started[1] < score_finished[0]
