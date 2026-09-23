# SPDX-License-Identifier: Apache-2.0
"""Publishing canonical state and adopting it are separate decisions.

The published frontier is the durable canonical prefix the cache can restore;
the adopted frontier is the part of it a foreground request restores. The
default grain of 1 adopts all of it, which must be exactly the behaviour
without this policy; that is pinned first, below.

Recovery publishes every safe block, and it has to: its live state is retired
whenever foreground work arrives, and whatever it processed past the last
publication is lost with it. Adopting that state is a different trade. The
SpecPrefill draft scoring window starts at the prefix a request restores, so
every newly adopted block moves the window and costs a cold draft rescore.

``canonical_state_adoption_grain_blocks`` lets the adopted prefix lag the
durable one by less than a step, so the window moves once per step instead of
once per block. These tests pin what the lag may and may not do: it never
exceeds the durable prefix, never lowers it except when the durable prefix
drops, never touches recovery or publication, applies only when SpecPrefill
would score the request anyway, gives back every reference it drops, and
falls back to the ordinary restore when anything goes wrong.
"""

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

from omlx.cache.paged_cache import PagedCacheManager
from omlx.cache.prefix_cache import BlockAwarePrefixCache
from omlx.canonical_recovery import adopted_frontier, apply_canonical_recovery_settings
from omlx.model_settings import ModelSettings
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig

BLOCK = 4
THRESHOLD = 8


class _Model:
    def __init__(self, num_layers: int = 1):
        self._num_layers = num_layers
        self.layers = [MagicMock() for _ in range(num_layers)]

    @property
    def args(self):
        args = MagicMock()
        args.num_hidden_layers = self._num_layers
        return args


def _cache_data(num_tokens: int):
    keys = mx.arange(num_tokens, dtype=mx.float32).reshape(1, 1, num_tokens, 1)
    values = (keys + 100).astype(mx.float32)
    return [{"state": (keys, values), "cache_type": "KVCache", "class_name": "KVCache"}]


def _make_prefix_cache():
    paged = PagedCacheManager(
        block_size=BLOCK, max_blocks=256, model_name="test-model", initial_blocks=256
    )
    cache = BlockAwarePrefixCache(
        model=_Model(), paged_cache_manager=paged, paged_ssd_cache_manager=None
    )
    # Reconstruction reads tensors back from the SSD tier, which these tests
    # do not configure. What is under test is which blocks the restore keeps,
    # so a reconstruction that returns the table it was given is enough.
    cache.reconstruct_cache = lambda block_table, **_: ["restored"]
    return cache


@pytest.fixture
def prefix_cache():
    return _make_prefix_cache()


def _scheduler(prefix_cache, grain: int = 4, draft: bool = True) -> Scheduler:
    model = MagicMock()
    model.layers = []
    model.minimum_prefill_prefix = None
    model.restore_cache = None
    tokenizer = MagicMock()
    tokenizer.eos_token_id = 2
    sched = Scheduler(
        model=model,
        tokenizer=tokenizer,
        config=SchedulerConfig(
            paged_cache_block_size=BLOCK,
            canonical_state_recovery_enabled=True,
            canonical_state_adoption_grain_blocks=grain,
        ),
    )
    sched.block_aware_cache = prefix_cache
    sched.paged_cache_manager = prefix_cache.paged_cache
    sched._unreconstructible_cache_model = False
    sched._gdn_split_active = MagicMock(return_value=False)
    sched._specprefill_draft_model = MagicMock() if draft else None
    # What SpecPrefill saw as the restored prefix, one entry per request.
    sched.scored_at = []
    sched._try_specprefill_scoring = lambda request: sched.scored_at.append(
        request.cached_tokens
    )
    return sched


def _publish(prefix_cache, tokens):
    """Write a canonical prefix the way the recovery publish does."""
    table = prefix_cache.store_cache(
        "canonical-recovery:s1", tokens, _cache_data(len(tokens))
    )
    assert table is not None and table.block_ids, "the publish itself failed"
    prefix_cache.clear_request_entry("canonical-recovery:s1")
    return table


def _request(prompt_tokens: int, rid: str = "fg", *, specprefill: bool = True, recovery=False):
    request = Request(
        request_id=rid,
        prompt=None,
        prompt_token_ids=list(range(prompt_tokens)),
        sampling_params=SamplingParams(max_tokens=1),
    )
    request._specprefill_enabled = specprefill
    request._specprefill_threshold = THRESHOLD
    request.is_canonical_recovery = recovery
    return request


def _restore(sched, request) -> int:
    sched._prepare_prefix_cache_for_request(request)
    return request.cached_tokens


def _ref_counts(prefix_cache, block_ids):
    return {
        block_id: prefix_cache.paged_cache.allocated_blocks[block_id].ref_count
        for block_id in block_ids
    }


def _snapshot(sched, request, prefix_cache, published_ids):
    table = sched.paged_cache_manager.get_block_table(request.request_id)
    return (
        request.cached_tokens,
        list(request.remaining_tokens or []),
        request.shared_prefix_blocks,
        request.prompt_cache is not None,
        list(table.block_ids) if table is not None else None,
        _ref_counts(prefix_cache, published_ids),
        list(sched.scored_at),
    )


def _without_adoption(sched):
    """The restore as it was before adoption existed: the cap is not called."""
    sched._cap_foreground_adoption = lambda request, block_table: block_table
    return sched


class TestTheDefaultIsTodaysBehaviour:
    """grain=1 must restore exactly what the restore did before the policy."""

    @pytest.mark.parametrize("published_blocks", [0, 1, 3, 4, 5, 8])
    @pytest.mark.parametrize("prompt_blocks", [4, 9, 20])
    @pytest.mark.parametrize("specprefill", [True, False])
    @pytest.mark.parametrize("draft", [True, False])
    @pytest.mark.parametrize("recovery", [True, False])
    def test_restore_is_identical(
        self, published_blocks, prompt_blocks, specprefill, draft, recovery
    ):
        results = []
        for make in (
            lambda c: _scheduler(c, grain=1, draft=draft),
            lambda c: _without_adoption(_scheduler(c, grain=1, draft=draft)),
        ):
            cache = _make_prefix_cache()
            ids = (
                _publish(cache, list(range(published_blocks * BLOCK))).block_ids
                if published_blocks
                else []
            )
            sched = make(cache)
            rid = "canonical-recovery:s1" if recovery else "fg"
            request = _request(
                prompt_blocks * BLOCK + 1, rid, specprefill=specprefill, recovery=recovery
            )
            _restore(sched, request)
            results.append(_snapshot(sched, request, cache, ids))
        assert results[0] == results[1]

    def test_grain_one_touches_nothing(self):
        """Not even a read: the default path returns before looking."""
        sched = _scheduler(_make_prefix_cache(), grain=1)
        sched.paged_cache_manager = MagicMock()
        table = MagicMock()
        assert sched._cap_foreground_adoption(_request(40), table) is table
        assert sched.paged_cache_manager.mock_calls == []
        assert table.mock_calls == []


class TestTheFrontierFunction:
    @pytest.mark.parametrize(
        "durable_blocks,adopted_blocks",
        [(0, 0), (1, 0), (3, 0), (4, 4), (5, 4), (8, 8)],
    )
    def test_four_block_grain(self, durable_blocks, adopted_blocks):
        assert (
            adopted_frontier(
                published_tokens=durable_blocks * 1024, block_size=1024, grain_blocks=4
            )
            == adopted_blocks * 1024
        )

    @pytest.mark.parametrize("durable", [0, 1, 1023, 1024, 5000, 70_000])
    def test_a_grain_of_one_adopts_everything(self, durable):
        assert (
            adopted_frontier(published_tokens=durable, block_size=1024, grain_blocks=1)
            == durable
        )

    @pytest.mark.parametrize("grain", [1, 2, 4, 8])
    def test_never_above_durable_and_never_falls_while_durable_grows(self, grain):
        previous = 0
        for durable in range(0, 40 * 1024, 256):
            adopted = adopted_frontier(
                published_tokens=durable, block_size=1024, grain_blocks=grain
            )
            assert adopted <= durable
            assert adopted >= previous
            previous = adopted

    def test_it_falls_only_with_the_durable_prefix(self):
        """Ground loss is the one allowed drop, and it lands on the new durable
        prefix's own step, not on anything remembered from before."""
        before = adopted_frontier(published_tokens=9 * 1024, block_size=1024, grain_blocks=4)
        after = adopted_frontier(published_tokens=6 * 1024, block_size=1024, grain_blocks=4)
        assert (before, after) == (8 * 1024, 4 * 1024)


class TestTheRestore:
    def test_a_grain_of_one_restores_what_it_did_before(self, prefix_cache):
        _publish(prefix_cache, list(range(5 * BLOCK)))
        assert _restore(_scheduler(prefix_cache, grain=1), _request(10 * BLOCK)) == 5 * BLOCK

    def test_the_restore_lags_to_the_last_whole_step(self, prefix_cache):
        _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        request = _request(10 * BLOCK)
        assert _restore(sched, request) == 4 * BLOCK
        assert request.remaining_tokens == list(range(4 * BLOCK, 10 * BLOCK))
        assert request.shared_prefix_blocks == 4
        assert sched.scored_at == [4 * BLOCK]

    def test_less_than_one_step_restores_nothing_and_holds_nothing(self, prefix_cache):
        table = _publish(prefix_cache, list(range(3 * BLOCK)))
        before = _ref_counts(prefix_cache, table.block_ids)
        sched = _scheduler(prefix_cache, grain=4)
        request = _request(10 * BLOCK)
        assert _restore(sched, request) == 0
        assert request.prompt_cache is None
        assert request.remaining_tokens == request.prompt_token_ids
        assert sched.paged_cache_manager.get_block_table("fg") is None
        assert _ref_counts(prefix_cache, table.block_ids) == before

    def test_dropped_blocks_give_their_references_back(self, prefix_cache):
        table = _publish(prefix_cache, list(range(7 * BLOCK)))
        before = _ref_counts(prefix_cache, table.block_ids)
        sched = _scheduler(prefix_cache, grain=4)
        for i in range(100):
            request = _request(12 * BLOCK, rid=f"fg-{i}")
            assert _restore(sched, request) == 4 * BLOCK
            sched.paged_cache_manager.delete_block_table(request.request_id)
        assert _ref_counts(prefix_cache, table.block_ids) == before


class TestTheLagNeedsAReason:
    """Lag exists only to keep the draft scoring window still. Where no draft
    scoring would happen at the durable prefix, there is nothing to keep still."""

    def test_no_draft_model(self, prefix_cache):
        _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4, draft=False)
        assert _restore(sched, _request(10 * BLOCK)) == 5 * BLOCK

    def test_specprefill_off_for_the_request(self, prefix_cache):
        _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        assert _restore(sched, _request(10 * BLOCK, specprefill=False)) == 5 * BLOCK

    def test_below_threshold_at_the_durable_prefix(self, prefix_cache):
        """Lagging would put this request back above the threshold and buy a
        sparse prefill it did not need. It adopts the durable prefix."""
        _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        prompt = 5 * BLOCK + THRESHOLD  # remaining at D == threshold: dense
        assert _restore(sched, _request(prompt)) == 5 * BLOCK

    def test_the_recovery_jobs_own_restore(self, prefix_cache):
        _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        request = _request(10 * BLOCK, rid="canonical-recovery:s1", recovery=True)
        assert _restore(sched, request) == 5 * BLOCK


class TestNothingIsRemembered:
    def test_a_restarted_scheduler_adopts_the_same_prefix(self, prefix_cache):
        _publish(prefix_cache, list(range(6 * BLOCK)))
        first = _restore(_scheduler(prefix_cache, grain=4), _request(12 * BLOCK, "a"))
        second = _restore(_scheduler(prefix_cache, grain=4), _request(12 * BLOCK, "b"))
        assert first == second == 4 * BLOCK

    def test_a_rewritten_history_carries_nothing_over(self, prefix_cache):
        """Change a token inside block 2: the durable match stops there, and the
        adopted prefix is computed from that alone."""
        _publish(prefix_cache, list(range(8 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        assert _restore(sched, _request(12 * BLOCK, "a")) == 8 * BLOCK
        rewritten = _request(12 * BLOCK, "b")
        rewritten.prompt_token_ids[2 * BLOCK + 1] = -7
        assert _restore(sched, rewritten) == 0


class TestTheDraftWindowMovesOncePerStep:
    @pytest.mark.parametrize("grain,moves", [(1, 12), (4, 3)])
    def test_append_only_session(self, prefix_cache, grain, moves):
        """Recovery publishes one more block before every turn. The draft
        scoring window starts at the restored prefix, so the number of times
        that prefix changes is the number of cold draft rescores."""
        sched = _scheduler(prefix_cache, grain=grain)
        for turn in range(13):
            if turn:
                _publish(prefix_cache, list(range(turn * BLOCK)))
            request = _request(20 * BLOCK + turn, rid=f"t{turn}")
            _restore(sched, request)
            sched.paged_cache_manager.delete_block_table(request.request_id)
        seen = sched.scored_at
        assert sum(a != b for a, b in zip(seen, seen[1:])) == moves
        assert seen[-1] == (12 // grain) * grain * BLOCK


class TestFailClosed:
    def test_an_error_restores_the_full_prefix(self, prefix_cache, caplog):
        table = _publish(prefix_cache, list(range(5 * BLOCK)))
        sched = _scheduler(prefix_cache, grain=4)
        request = _request(10 * BLOCK)
        with patch("omlx.scheduler.adopted_frontier", side_effect=RuntimeError("boom")):
            assert _restore(sched, request) == 5 * BLOCK
        assert "failed closed" in caplog.text
        sched.paged_cache_manager.delete_block_table(request.request_id)
        assert all(
            prefix_cache.paged_cache.allocated_blocks[b].ref_count == 1
            for b in table.block_ids
        )


class TestTheSetting:
    def test_defaults_to_every_block(self):
        assert ModelSettings().canonical_state_adoption_grain_blocks == 1
        assert SchedulerConfig().canonical_state_adoption_grain_blocks == 1

    def test_round_trips(self):
        settings = ModelSettings(canonical_state_adoption_grain_blocks=4)
        restored = ModelSettings.from_dict(settings.to_dict())
        assert restored.canonical_state_adoption_grain_blocks == 4

    def test_rejects_less_than_one(self):
        with pytest.raises(ValueError, match="canonical_state_adoption_grain_blocks"):
            ModelSettings(canonical_state_adoption_grain_blocks=0)

    def test_is_carried_per_model_and_snapshotted(self, prefix_cache):
        config = SchedulerConfig()
        apply_canonical_recovery_settings(
            config, ModelSettings(canonical_state_adoption_grain_blocks=4)
        )
        assert config.canonical_state_adoption_grain_blocks == 4
        sched = _scheduler(prefix_cache, grain=4)
        # Another model's load rewrites the shared config; this engine keeps its own.
        sched.config.canonical_state_adoption_grain_blocks = 1
        assert sched._canonical_adoption_grain_blocks == 4
