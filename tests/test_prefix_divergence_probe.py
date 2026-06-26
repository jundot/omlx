# SPDX-License-Identifier: Apache-2.0
"""Tests for the DEBUG-only prefix-cache divergence probe (issue #1003)."""

import logging
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omlx.scheduler import Scheduler


class _FakeTokenizer:
    def decode(self, ids):
        return "".join(chr(ord("a") + (i % 26)) for i in ids)


def _make_scheduler(block_size=2048):
    sched = object.__new__(Scheduler)
    sched._cache_probe_seqs = deque(maxlen=4)
    sched.tokenizer = _FakeTokenizer()
    sched.config = MagicMock(spec=[])
    sched.config.paged_cache_block_size = block_size
    return sched


def _request(prompt_ids, cached_tokens=0, request_id="req-new"):
    return SimpleNamespace(
        request_id=request_id,
        prompt_token_ids=prompt_ids,
        cached_tokens=cached_tokens,
    )


class TestCommonPrefixLen:
    def test_identical(self):
        assert Scheduler._common_prefix_len([1, 2, 3], [1, 2, 3]) == 3

    def test_divergent(self):
        assert Scheduler._common_prefix_len([1, 2, 3, 4], [1, 2, 9, 4]) == 2

    def test_length_mismatch(self):
        assert Scheduler._common_prefix_len([1, 2], [1, 2, 3, 4]) == 2

    def test_empty(self):
        assert Scheduler._common_prefix_len([], [1]) == 0


class TestDivergenceProbe:
    def test_logs_divergence_offset_and_context(self, caplog):
        sched = _make_scheduler()
        stored = list(range(100))
        sched._cache_probe_seqs.append(("req-old", stored))
        prompt = list(range(50)) + [999] + list(range(51, 80))

        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt))

        text = caplog.text
        assert "prefix probe vs stored req-old" in text
        assert "common_prefix=50/80" in text
        assert "first divergence at token 50" in text

    def test_full_match_logs_no_divergence_line(self, caplog):
        sched = _make_scheduler()
        stored = list(range(100))
        sched._cache_probe_seqs.append(("req-old", stored))
        # Prompt extends the stored sequence — common prefix covers all
        # reusable tokens, so only the summary line should be emitted.
        prompt = stored + [200, 201]

        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt, cached_tokens=100))

        assert "prefix probe" in caplog.text
        assert "first divergence" not in caplog.text

    def test_picks_best_matching_stored_sequence(self, caplog):
        sched = _make_scheduler()
        sched._cache_probe_seqs.append(("req-a", [9, 9, 9]))
        sched._cache_probe_seqs.append(("req-b", [1, 2, 3, 4, 5]))
        prompt = [1, 2, 3, 7, 7]

        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt))

        assert "vs stored req-b" in caplog.text
        assert "common_prefix=3/5" in caplog.text

    def test_noop_without_stored_sequences(self, caplog):
        sched = _make_scheduler()
        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request([1, 2, 3]))
        assert caplog.text == ""

    def test_noop_with_empty_prompt(self, caplog):
        sched = _make_scheduler()
        sched._cache_probe_seqs.append(("req-old", [1, 2]))
        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request([]))
        assert caplog.text == ""

    def test_decode_failure_is_tolerated(self, caplog):
        sched = _make_scheduler()
        sched.tokenizer = MagicMock()
        sched.tokenizer.decode.side_effect = RuntimeError("boom")
        sched._cache_probe_seqs.append(("req-old", [1, 2, 3, 4]))

        with caplog.at_level(logging.DEBUG, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request([1, 9, 9, 9]))

        assert "first divergence at token 1" in caplog.text
        assert "<decode failed>" in caplog.text
