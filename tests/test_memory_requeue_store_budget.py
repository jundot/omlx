# SPDX-License-Identifier: Apache-2.0
"""The memory-pressure requeue store stops on its time budget or on an abort.

The store runs on the inference thread at the speed of the SSD cache drive.
A stopped store keeps a valid stored prefix; the requeued requests resume from
it (or from nothing) and still stream every token once.
"""

import pytest
from test_scheduler_boundary_completion import (
    _assert_restored_matches_cold,
    _chunked_scheduler_rig,
)

from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def _decoding_pair(scheduler, prompts, max_tokens=12):
    requests = {
        rid: Request(
            request_id=rid,
            prompt=list(tokens),
            sampling_params=SamplingParams(max_tokens=max_tokens, temperature=0),
        )
        for rid, tokens in prompts.items()
    }
    streamed = {rid: [] for rid in requests}
    finals = {}

    def step():
        for output in scheduler.step().outputs:
            streamed[output.request_id].extend(output.new_token_ids)
            if output.finished:
                finals[output.request_id] = output

    for rid in requests:
        scheduler.add_request(requests[rid])
    for _ in range(100):
        step()
        if all(len(r.output_token_ids) >= 5 for r in requests.values()):
            break
    return requests, streamed, finals, step


PROMPTS = {"a": list(range(3, 23)), "b": list(range(4, 24))}  # 5 blocks of 4


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("OMLX_MEMORY_REQUEUE_STORE_BUDGET_S", raising=False)


@pytest.mark.parametrize("kind", ["gdn", "rotating"])
def test_store_stopped_mid_request_keeps_a_valid_prefix(
    mock_tokenizer, tmp_path, kind, monkeypatch
):
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)
    # Stop each request's store after two new blocks, as a spent budget would.
    real_store = BlockAwarePrefixCache.store_cache
    stop_calls: list[int] = []

    def store(self, *args, should_stop=None, **kwargs):
        if should_stop is None:
            return real_store(self, *args, **kwargs)
        calls = {"n": 0}

        def stop_after_two():
            calls["n"] += 1
            return calls["n"] > 2

        result = real_store(self, *args, should_stop=stop_after_two, **kwargs)
        stop_calls.append(calls["n"])
        return result

    monkeypatch.setattr(BlockAwarePrefixCache, "store_cache", store)
    try:
        requests, streamed, finals, step = _decoding_pair(scheduler, PROMPTS)
        sent = {rid: len(r.output_token_ids) for rid, r in requests.items()}
        assert scheduler.request_memory_requeue() == 2
        step()
        assert stop_calls and all(n == 3 for n in stop_calls)
        for rid, request in requests.items():
            assert request.memory_requeues == 1
            assert request.requeued_output_tokens == sent[rid]
            # Two blocks of four tokens stored, the rest is recomputed.
            _assert_restored_matches_cold(
                model, scheduler, prefix, request.prompt_token_ids, 8
            )
        for _ in range(200):
            step()
            if len(finals) == len(requests):
                break
        for rid, request in requests.items():
            assert streamed[rid] == request.output_token_ids
            assert len(streamed[rid]) == 12
            assert finals[rid].completion_tokens == 12
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("kind", ["gdn", "rotating"])
def test_spent_budget_requeues_without_store_and_still_completes(
    mock_tokenizer, tmp_path, kind, monkeypatch, caplog
):
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, kind)
    monkeypatch.setenv("OMLX_MEMORY_REQUEUE_STORE_BUDGET_S", "1e-9")
    try:
        requests, streamed, finals, step = _decoding_pair(scheduler, PROMPTS)
        assert scheduler.request_memory_requeue() == 2
        with caplog.at_level("WARNING", logger="omlx.scheduler"):
            step()
        assert "store budget" in caplog.text
        for request in requests.values():
            assert request.memory_requeues == 1
            table, _remaining = prefix.fetch_cache("probe", request.prompt_token_ids)
            assert table is None or table.num_tokens == 0
            prefix.release_cache("probe")
        for _ in range(200):
            step()
            if len(finals) == len(requests):
                break
        for rid, request in requests.items():
            assert streamed[rid] == request.output_token_ids
            assert len(streamed[rid]) == 12
    finally:
        scheduler.shutdown()


def test_abort_during_the_store_stops_it_between_blocks(
    mock_tokenizer, tmp_path, monkeypatch
):
    """An abort that arrives while a request's KV is being written ends the
    store at the next block (before the fix it only took effect between
    requests, after the whole request was written)."""
    model, scheduler, prefix = _chunked_scheduler_rig(mock_tokenizer, tmp_path, "gdn")
    real_store = BlockAwarePrefixCache.store_cache
    fired = {"done": False}

    def store(self, *args, should_stop=None, **kwargs):
        if should_stop is not None and not fired["done"]:
            fired["done"] = True
            scheduler.abort_request("b")  # arrives mid-store of "a"
        return real_store(self, *args, should_stop=should_stop, **kwargs)

    monkeypatch.setattr(BlockAwarePrefixCache, "store_cache", store)
    try:
        requests, streamed, finals, step = _decoding_pair(scheduler, PROMPTS, 50)
        assert scheduler.request_memory_requeue() == 2
        step()
        assert fired["done"]
        table, _remaining = prefix.fetch_cache("probe", requests["a"].prompt_token_ids)
        assert table is None or table.num_tokens == 0  # "a" stopped at block 1
        prefix.release_cache("probe")
        for _ in range(300):
            step()
            if "a" in finals:
                break
        assert "b" not in scheduler.requests  # the abort went through
        assert finals["a"].completion_tokens == 50
        assert streamed["a"] == requests["a"].output_token_ids
    finally:
        scheduler.shutdown()


def test_budget_env(monkeypatch):
    s = Scheduler.__new__(Scheduler)
    assert Scheduler._memory_requeue_store_budget_s(s) == 30.0
    monkeypatch.setenv("OMLX_MEMORY_REQUEUE_STORE_BUDGET_S", "0")
    assert Scheduler._memory_requeue_store_budget_s(s) == 0.0
    monkeypatch.setenv("OMLX_MEMORY_REQUEUE_STORE_BUDGET_S", "junk")
    assert Scheduler._memory_requeue_store_budget_s(s) == 30.0
