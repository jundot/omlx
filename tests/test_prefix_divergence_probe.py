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

    def test_large_reprefill_logs_one_info_line(self, caplog):
        """A big re-prefill that shared at least one block with a stored
        sequence gets a single INFO line naming the divergence position
        (#2333/#2349 triage), with no decoded prompt content at INFO."""
        sched = _make_scheduler()
        stored = list(range(10000))
        sched._cache_probe_seqs.append(("req-old", stored))
        # Diverges at 8000: reused 6144 (3 blocks), re-prefills 6000.
        prompt = list(range(8000)) + [999999] * 4143

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt, cached_tokens=6144))

        infos = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(infos) == 1
        msg = infos[0].getMessage()
        assert "re-prefills 5999 of 12143 tokens" in msg
        assert "(reused 6144)" in msg
        assert "req-old" in msg
        assert "shares the first 8000 of 10000 comparable tokens" in msg
        # Prompt content must not leak into INFO (decoded context is DEBUG).
        assert "stored=..." not in msg

    def test_small_reprefill_stays_silent_at_info(self, caplog):
        sched = _make_scheduler()
        stored = list(range(9000))
        sched._cache_probe_seqs.append(("req-old", stored))
        # Only ~1900 tokens re-prefilled: below the INFO threshold.
        prompt = list(range(9000)) + [7] * 100

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt, cached_tokens=7200))

        assert caplog.text == ""

    def test_unrelated_prompt_stays_silent_at_info(self, caplog):
        """Shared prefix below one block means the stored cache was never
        relevant to this prompt: a plain cold prefill logs nothing."""
        sched = _make_scheduler()
        sched._cache_probe_seqs.append(("req-old", list(range(9000))))
        prompt = [500000 + i for i in range(9000)]

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt, cached_tokens=0))

        assert caplog.text == ""

    def test_cache_loss_signature_logged(self, caplog):
        """Stored sequence covers the whole prompt but nothing was served:
        the INFO line must expose the reused=0 vs shared=full mismatch."""
        sched = _make_scheduler()
        stored = list(range(12000))
        sched._cache_probe_seqs.append(("req-old", stored))
        prompt = list(range(12000))

        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            sched._log_prefix_divergence(_request(prompt, cached_tokens=0))

        msg = caplog.text
        assert "re-prefills 12000 of 12000 tokens (reused 0)" in msg
        assert "shares the first 12000 of 12000 comparable tokens" in msg

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


class TestJunctionEnd:
    """A prompt that leaves a recent sequence off the grid plans a shared tail there."""

    @staticmethod
    def _plan(stored, prompt, cached, *, gen_start=0, stateful=True):
        sched = _make_scheduler()
        sched._boundary_snapshot_required = stateful
        if stored is not None:
            sched._cache_probe_seqs.append(("req-old", stored))
        request = _request(prompt, cached_tokens=cached)
        request.generation_prompt_start = gen_start
        sched._log_prefix_divergence(request)
        sched._resolve_junction_end(request)
        return getattr(request, "junction_end", 0)

    def test_plans_the_branch_point_the_lookup_could_not_reuse(self):
        stored = list(range(6000))
        prompt = list(range(5000)) + [-1] * 200
        # 5000 shared tokens, 4096 restorable: 904 tokens the tail saves.
        assert self._plan(stored, prompt, 4096, gen_start=5190) == 5000

    @pytest.mark.parametrize(
        "name,stored,prompt,cached,gen_start,stateful",
        [
            # 392 saved tokens do not pay for a snapshot and its files.
            (
                "small gain",
                list(range(6000)),
                list(range(5000)) + [-1] * 200,
                4608,
                5190,
                True,
            ),
            # On the grid the boundary snapshot already covers the split.
            (
                "on the grid",
                list(range(7000)),
                list(range(6144)) + [-1] * 100,
                4096,
                6230,
                True,
            ),
            # The prompt runs into the generation prompt: a tail replay covers it.
            (
                "at the marker",
                list(range(6000)),
                list(range(5000)) + [-1] * 200,
                4096,
                4990,
                True,
            ),
            # The whole prompt repeats: nothing branches.
            ("exact prefix", list(range(6000)), list(range(5000)), 4096, 0, True),
            # No recent sequence to branch from.
            (
                "no stored sequence",
                None,
                list(range(5000)) + [-1] * 200,
                4096,
                5190,
                True,
            ),
            # Attention-only models resume anywhere a block ends.
            (
                "stateless model",
                list(range(6000)),
                list(range(5000)) + [-1] * 200,
                4096,
                5190,
                False,
            ),
        ],
    )
    def test_plans_nothing(self, name, stored, prompt, cached, gen_start, stateful):
        assert (
            self._plan(stored, prompt, cached, gen_start=gen_start, stateful=stateful)
            == 0
        ), name

    def test_ignores_a_lookup_of_another_request(self):
        sched = _make_scheduler()
        sched._boundary_snapshot_required = True
        sched._cache_probe_seqs.append(("req-old", list(range(6000))))
        sched._log_prefix_divergence(_request(list(range(5000)) + [-1] * 200, 4096))
        other = _request(list(range(5000)) + [-1] * 200, 4096, request_id="req-other")
        other.generation_prompt_start = 5190
        sched._resolve_junction_end(other)
        assert getattr(other, "junction_end", 0) == 0
