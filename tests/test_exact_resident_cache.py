from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import mlx.core as mx
import pytest

from omlx.cache.exact_resident import ExactResidentPrefixCache
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


class _OffsetCache:
    def __init__(self, offset, nbytes=0, rollback_state=None):
        self.offset = offset
        self.nbytes = nbytes
        self.rollback_state = rollback_state
        self.buffer = mx.zeros((max(1, nbytes),), dtype=mx.uint8)


class ArraysCache:
    def __init__(self):
        self.cache = [mx.zeros((1, 2)), mx.ones((1, 2))]
        self.rollback_state = None


class SizedArraysCache:
    def __init__(self, token_count):
        self._inner = ArraysCache()
        self._token_count = token_count
        self.rollback_state = None

    @property
    def cache(self):
        return self._inner.cache


class QSAKVCache:
    def __init__(self, offset=4):
        self.offset = offset
        self.keys = mx.zeros((1, 2, offset, 4))
        self.values = mx.ones((1, 2, offset, 4))
        self._index_keys = mx.zeros((1, offset, 8))
        self._index_position_ids = mx.arange(offset)[None, :]
        self._index_offset = offset
        self.index_keys = self._index_keys
        self.index_position_ids = self._index_position_ids
        self._pooled_index_keys = None
        self._pooled_index_offset = 0
        self._pooled_index_ratio = None
        self._pooled_index_tag = None
        self._omlx_text_position_ids_qualified = False
        self.rollback_state = None

    def _invalidate_pooled_indexer(self):
        self._pooled_index_keys = None
        self._pooled_index_offset = 0
        self._pooled_index_ratio = None
        self._pooled_index_tag = None


def _scheduler(slots=1):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._exact_resident_cache = ExactResidentPrefixCache(slots)
    scheduler._phase_total_ms = defaultdict(float)
    scheduler._phase_count = defaultdict(int)
    scheduler.model = object()
    scheduler._vlm_mtp_drafter = None
    return scheduler


def _request(tokens):
    request = Request("request", list(tokens), SamplingParams())
    request.prompt_token_ids = list(tokens)
    request.num_prompt_tokens = len(tokens)
    return request


def test_exact_resident_transfers_ownership_for_true_suffix():
    tier = ExactResidentPrefixCache(max_entries=1)
    cache = [object()]
    assert tier.put(
        [1, 2, 3],
        cache,
        cache_nbytes=123,
        durable_tokens=2,
    )

    hit = tier.acquire_prefix([1, 2, 3, 4, 5])

    assert hit is not None
    assert hit.cache is cache
    assert hit.cached_tokens == 3
    assert hit.cache_nbytes == 123
    assert hit.durable_tokens == 2
    assert tier.stats()["entries"] == 0


def test_exact_resident_requires_exact_token_identity():
    tier = ExactResidentPrefixCache(max_entries=1)
    assert tier.put([1, 2, 3], [object()])

    assert tier.acquire_prefix([1, 9, 3, 4]) is None
    assert tier.stats()["entries"] == 1


def test_exact_resident_rejects_zero_suffix_without_generic_trim():
    tier = ExactResidentPrefixCache(max_entries=1)
    assert tier.put([1, 2, 3], [object()])

    assert tier.acquire_prefix([1, 2, 3]) is None
    assert tier.stats()["entries"] == 1


def test_exact_resident_is_immediately_claimable_after_response():
    tier = ExactResidentPrefixCache(max_entries=1)
    assert tier.put([1, 2], [object()])

    assert tier.acquire_prefix([1, 2, 3]) is not None


def test_exact_resident_is_bounded_lru():
    tier = ExactResidentPrefixCache(max_entries=2)
    first = [object()]
    second = [object()]
    third = [object()]
    assert tier.put([1], first)
    assert tier.put([2], second)
    assert tier.put([3], third)

    assert tier.stats()["entries"] == 2
    assert tier.stats()["evictions"] == 1
    assert tier.acquire_prefix([1, 4]) is None
    assert tier.acquire_prefix([2, 4]).cache is second


def test_prompt_fallback_never_evicts_longer_terminal_when_budget_is_tight():
    tier = ExactResidentPrefixCache(max_entries=1, max_bytes=1_000)
    terminal = [object()]
    fallback = [object()]
    assert tier.put([1, 2, 3, 4], terminal, cache_nbytes=400)

    assert not tier.put(
        [1, 2],
        fallback,
        cache_nbytes=200,
        protect_longer_prefix=True,
    )
    hit = tier.acquire_prefix([1, 2, 3, 4, 5])
    assert hit is not None
    assert hit.cache is terminal
    assert tier.stats()["protected_rejections"] == 1


def test_prompt_fallback_coexists_with_longer_terminal_under_shared_byte_cap():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    terminal = [object()]
    fallback = [object()]
    assert tier.put([1, 2, 3, 4], terminal, cache_nbytes=400)
    assert tier.put(
        [1, 2],
        fallback,
        cache_nbytes=200,
        protect_longer_prefix=True,
    )

    longest = tier.acquire_prefix([1, 2, 3, 4, 5])
    assert longest is not None
    assert longest.cache is terminal
    divergent = tier.acquire_prefix([1, 2, 9])
    assert divergent is not None
    assert divergent.cache is fallback


def test_prompt_fallback_evicts_unrelated_lru_before_protected_terminal():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    unrelated = [object()]
    terminal = [object()]
    fallback = [object()]
    assert tier.put([9, 9], unrelated, cache_nbytes=100)
    assert tier.put([1, 2, 3, 4], terminal, cache_nbytes=400)

    assert tier.put(
        [1, 2],
        fallback,
        cache_nbytes=200,
        protect_longer_prefix=True,
    )

    assert tier.acquire_prefix([9, 9, 10]) is None
    assert tier.acquire_prefix([1, 2, 3, 4, 5]).cache is terminal
    assert tier.acquire_prefix([1, 2, 8]).cache is fallback


def test_shared_boundary_keeps_newest_terminal_and_evicts_older_longer_branch():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=2_000)
    old_longer = [object()]
    new_shorter = [object()]
    stable = [object()]
    old_tokens = [1, 2, 3, 4, 5, 6]
    new_tokens = [1, 2, 9, 10]
    stable_tokens = [1, 2]

    assert tier.put(old_tokens, old_longer, cache_nbytes=600, durable_tokens=2)
    assert tier.put(new_tokens, new_shorter, cache_nbytes=400, durable_tokens=2)
    assert tier.can_fit_protected_candidate(
        stable_tokens,
        estimated_cache_nbytes=200,
    )
    assert tier.put(
        stable_tokens,
        stable,
        cache_nbytes=200,
        durable_tokens=2,
        protect_longer_prefix=True,
    )

    assert not tier.contains_exact(old_tokens)
    assert tier.contains_exact(new_tokens)
    assert tier.contains_exact(stable_tokens)
    assert tier.stats()["entries"] == 2
    assert tier.stats()["evictions"] == 1

    rewritten = tier.acquire_prefix([1, 2, 42, 43])
    assert rewritten is not None
    assert rewritten.cache is stable
    current_append = tier.acquire_prefix([1, 2, 9, 10, 11])
    assert current_append is not None
    assert current_append.cache is new_shorter


def test_prompt_fallback_preflight_accounts_for_protected_terminal_bytes():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    assert tier.put([1, 2, 3, 4], [object()], cache_nbytes=700)

    assert not tier.can_fit_protected_candidate(
        [1],
        estimated_cache_nbytes=400,
    )
    assert tier.can_fit_protected_candidate(
        [1],
        estimated_cache_nbytes=200,
    )


def test_exact_resident_enforces_byte_budget_and_uint32_tokens():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=100)

    assert not tier.put([1], [object()], cache_nbytes=101)
    assert not tier.put([-1], [object()], cache_nbytes=1)
    assert not tier.put([0x1_0000_0000], [object()], cache_nbytes=1)
    assert not tier.put([1], [object()], cache_nbytes=1, durable_tokens=2)
    assert tier.put([0xFFFFFFFF], [object()], cache_nbytes=100)
    assert tier.stats()["size_bytes"] == 100
    assert tier.stats()["max_bytes"] == 100
    assert tier.stats()["oversize_rejections"] == 1


def test_exact_resident_stats_report_largest_resident_token_count():
    tier = ExactResidentPrefixCache(max_entries=3, max_bytes=1_000)
    assert tier.put([1, 2], [object()], cache_nbytes=10)
    assert tier.put(
        [3, 4, 5, 6],
        [object()],
        cache_nbytes=10,
        durable_tokens=3,
    )

    assert tier.stats()["max_token_count"] == 4
    assert tier.stats()["max_durable_token_count"] == 3
    assert tier.acquire_prefix([3, 4, 5, 6, 7]) is not None
    assert tier.stats()["max_token_count"] == 2
    assert tier.stats()["max_durable_token_count"] == 0
    tier.clear()
    assert tier.stats()["max_token_count"] == 0


def test_disabled_exact_resident_reports_zero_effective_memory_cap():
    tier = ExactResidentPrefixCache(max_entries=0, max_bytes=8 * 1024**3)

    assert tier.stats()["max_entries"] == 0
    assert tier.stats()["max_bytes"] == 0


def test_disabled_scheduler_does_not_report_phantom_fallbacks():
    scheduler = _scheduler(slots=0)

    assert not scheduler._restore_exact_resident_cache(_request([1, 2, 3]))
    stats = scheduler._exact_resident_stats()
    assert stats["misses"] == 0
    assert stats["fallbacks_total"] == 0


def test_exact_resident_cache_has_single_request_ownership():
    tier = ExactResidentPrefixCache(max_entries=1)
    cache = [object()]
    assert tier.put([1, 2], cache)

    with ThreadPoolExecutor(max_workers=2) as pool:
        hits = list(pool.map(tier.acquire_prefix, ([1, 2, 3], [1, 2, 4])))

    assert sum(hit is not None for hit in hits) == 1
    assert next(hit for hit in hits if hit is not None).cache is cache


@pytest.mark.parametrize("batch_size", [1, 2, 4, 6])
def test_exact_resident_b1_b2_b4_b6_has_one_exclusive_claim(batch_size):
    tier = ExactResidentPrefixCache(max_entries=1)
    cache = [object()]
    assert tier.put([1, 2], cache)

    prompts = [[1, 2, 100 + row] for row in range(batch_size)]
    with ThreadPoolExecutor(max_workers=batch_size) as pool:
        hits = list(pool.map(tier.acquire_prefix, prompts))

    assert sum(hit is not None for hit in hits) == 1
    assert tier.stats()["hits"] == 1
    assert tier.stats()["misses"] == batch_size - 1


def test_exact_resident_divergent_entries_claim_independently():
    tier = ExactResidentPrefixCache(max_entries=3, max_bytes=1_000)
    caches = [[object()], [object()], [object()]]
    for base, cache in enumerate(caches, start=1):
        assert tier.put([base, 9], cache, cache_nbytes=100)

    for base, cache in enumerate(caches, start=1):
        hit = tier.acquire_prefix([base, 9, 10])
        assert hit is not None
        assert hit.cache is cache
    assert tier.stats()["entries"] == 0


def test_clear_releases_only_resident_entries():
    tier = ExactResidentPrefixCache(max_entries=2)
    tier.put([1], [object()], cache_nbytes=10)
    tier.put([2], [object()], cache_nbytes=20)

    assert tier.clear() == 2
    assert tier.stats()["entries"] == 0
    assert tier.stats()["size_bytes"] == 0


def test_resize_restores_entry_limit_and_evicts_oldest():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    first = [object()]
    second = [object()]
    assert tier.put([1], first, cache_nbytes=10)
    assert tier.put([2], second, cache_nbytes=20)

    assert tier.resize(1) == 1
    assert tier.stats()["max_entries"] == 1
    assert tier.stats()["entries"] == 1
    assert tier.acquire_prefix([1, 3]) is None
    assert tier.acquire_prefix([2, 3]).cache is second


def test_lifecycle_generation_makes_deferred_publication_atomic():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    generation = tier.generation()

    assert tier.put(
        [1, 2],
        [object()],
        cache_nbytes=10,
        expected_generation=generation,
    )
    tier.clear()
    assert not tier.put(
        [3, 4],
        [object()],
        cache_nbytes=10,
        expected_generation=generation,
    )
    resized_generation = tier.generation()
    tier.resize(1)
    assert tier.generation() == resized_generation + 1


def test_contains_prefix_requires_exact_ledger_and_minimum_boundary():
    tier = ExactResidentPrefixCache(max_entries=2, max_bytes=1_000)
    assert tier.put([1, 2, 3, 4], [object()], cache_nbytes=10)

    assert tier.contains_prefix([1, 2, 3, 4, 9], minimum_tokens=4)
    assert not tier.contains_prefix([1, 2, 3, 4, 9], minimum_tokens=5)
    assert not tier.contains_prefix([1, 2, 8, 4, 9], minimum_tokens=4)


def test_scheduler_stages_and_restores_exact_terminal_cache():
    scheduler = _scheduler()
    completed = _request([1, 2, 3])
    completed._exact_resident_durable_fallback_tokens = 2
    cache = [_OffsetCache(3, nbytes=99)]

    scheduler._stage_exact_resident_cache(completed, cache, [1, 2, 3])
    scheduler._publish_exact_resident_cache(completed)

    next_turn = _request([1, 2, 3, 4, 5])
    assert scheduler._restore_exact_resident_cache(next_turn)
    assert next_turn.prompt_cache is cache
    assert next_turn.cached_tokens == 3
    assert next_turn.remaining_tokens == [4, 5]
    assert next_turn._exact_resident_durable_fallback_tokens == 2


def test_scheduler_validates_every_positioned_and_recurrent_leaf():
    scheduler = _scheduler()
    request = _request([1, 2, 3])
    cache = [_OffsetCache(3, nbytes=9), ArraysCache()]

    scheduler._stage_exact_resident_cache(request, cache, [1, 2, 3])

    assert request._exact_resident_candidate[2] == 25


def test_scheduler_qsa_terminal_validation_covers_all_auxiliary_state():
    scheduler = _scheduler()
    request = _request([1, 2, 3, 4])
    qsa = QSAKVCache(4)

    scheduler._stage_exact_resident_cache(request, [qsa, ArraysCache()], [1, 2, 3, 4])
    assert hasattr(request, "_exact_resident_candidate")

    for mutate in (
        lambda cache: setattr(cache, "keys", mx.zeros((1, 2, 3, 4))),
        lambda cache: setattr(cache, "values", mx.zeros((1, 2, 3, 4))),
        lambda cache: setattr(cache, "_index_offset", 3),
        lambda cache: setattr(cache, "_index_keys", mx.zeros((1, 3, 8))),
        lambda cache: setattr(
            cache, "_index_position_ids", mx.arange(3)[None, :]
        ),
        lambda cache: setattr(cache, "index_keys", mx.zeros((1, 3, 8))),
        lambda cache: setattr(
            cache, "index_position_ids", mx.arange(3)[None, :]
        ),
        lambda cache: setattr(
            cache, "_index_position_ids", mx.zeros((1, 4), dtype=mx.int32)
        ),
        lambda cache: setattr(
            cache, "index_position_ids", mx.zeros((1, 4), dtype=mx.int32)
        ),
    ):
        probe = QSAKVCache(4)
        mutate(probe)
        rejected = _request([1, 2, 3, 4])
        scheduler._stage_exact_resident_cache(
            rejected, [probe, ArraysCache()], [1, 2, 3, 4]
        )
        assert not hasattr(rejected, "_exact_resident_candidate")


def test_scheduler_qsa_accepts_geometric_kv_capacity_at_exact_logical_offset():
    scheduler = _scheduler()
    request = _request([1, 2, 3, 4])
    qsa = QSAKVCache(4)
    qsa.keys = mx.zeros((1, 2, 8, 4))
    qsa.values = mx.ones((1, 2, 8, 4))

    scheduler._stage_exact_resident_cache(
        request,
        [qsa, ArraysCache()],
        [1, 2, 3, 4],
    )

    assert hasattr(request, "_exact_resident_candidate")


def test_scheduler_qsa_pooled_and_text_qualification_must_be_coherent():
    scheduler = _scheduler()
    tokens = [1, 2, 3, 4]

    pooled = QSAKVCache(4)
    pooled._pooled_index_keys = mx.zeros((1, 1, 8))
    pooled._pooled_index_offset = 1
    pooled._pooled_index_ratio = 4
    pooled._pooled_index_tag = object()
    accepted = _request(tokens)
    scheduler._stage_exact_resident_cache(
        accepted, [pooled, ArraysCache()], tokens
    )
    assert hasattr(accepted, "_exact_resident_candidate")
    assert pooled._pooled_index_keys is None
    assert pooled._pooled_index_offset == 0
    assert pooled._pooled_index_ratio is None
    assert pooled._pooled_index_tag is None

    mr = QSAKVCache(4)
    mr._index_position_ids = mx.stack(
        [mx.arange(4), mx.arange(4), mx.arange(4) + 1]
    )[:, None, :]
    mr.index_position_ids = mr._index_position_ids
    rejected_mr = _request(tokens)
    scheduler._stage_exact_resident_cache(
        rejected_mr, [mr, ArraysCache()], tokens
    )
    assert not hasattr(rejected_mr, "_exact_resident_candidate")


def test_scheduler_sized_arrays_proves_timeline_and_counts_inner_arrays():
    scheduler = _scheduler()
    tokens = [1, 2, 3, 4]
    qsa = QSAKVCache(4)
    sized = SizedArraysCache(4)
    accepted = _request(tokens)

    scheduler._stage_exact_resident_cache(accepted, [qsa, sized], tokens)

    candidate = accepted._exact_resident_candidate
    assert candidate[2] == Scheduler._resident_cache_nbytes([qsa, sized])
    # QSA + both arrays owned by SizedArraysCache._inner were counted.
    assert candidate[2] > Scheduler._resident_cache_nbytes([qsa])

    stale = _request(tokens)
    scheduler._stage_exact_resident_cache(
        stale, [QSAKVCache(4), SizedArraysCache(3)], tokens
    )
    assert not hasattr(stale, "_exact_resident_candidate")


def test_scheduler_resident_cache_fails_closed_when_mtp_is_active():
    scheduler = _scheduler()
    scheduler.model = type(
        "MtpModel", (), {"_omlx_mtp_decode_enabled": True}
    )()
    request = _request([1, 2, 3])
    cache = [_OffsetCache(3, nbytes=9)]

    scheduler._stage_exact_resident_cache(request, cache, [1, 2, 3])
    assert not hasattr(request, "_exact_resident_candidate")

    scheduler.model._omlx_mtp_decode_enabled = False
    scheduler._stage_exact_resident_cache(request, cache, [1, 2, 3])
    scheduler._publish_exact_resident_cache(request)
    scheduler.model._omlx_mtp_decode_enabled = True
    assert not scheduler._restore_exact_resident_cache(
        _request([1, 2, 3, 4])
    )
    assert scheduler._exact_resident_cache.stats()["entries"] == 1


def test_scheduler_restores_proven_generic_mtp_terminal_with_mtp_on():
    scheduler = _scheduler()
    scheduler.model = type(
        "MtpModel", (), {"_omlx_mtp_decode_enabled": True}
    )()
    request = _request([1, 2, 3])
    request._mtp_exact_terminal_proved = "mtp-standard-terminal-v1"
    request._exact_resident_durable_fallback_tokens = 2
    cache = [_OffsetCache(3, nbytes=9)]

    scheduler._stage_exact_resident_cache(request, cache, [1, 2, 3])
    assert request._exact_resident_candidate[0] == [1, 2, 3]
    assert scheduler._publish_exact_resident_cache(request)

    next_turn = _request([1, 2, 3, 4])
    assert scheduler._restore_exact_resident_cache(next_turn)
    assert next_turn.cached_tokens == 3
    assert next_turn.remaining_tokens == [4]


def test_scheduler_accepts_target_only_idle_stable_proof_with_mtp_on():
    scheduler = _scheduler()
    scheduler.model = type(
        "MtpModel", (), {"_omlx_mtp_decode_enabled": True}
    )()
    cache = [_OffsetCache(3, nbytes=9)]
    assert scheduler._exact_resident_cache.put(
        [1, 2, 3],
        cache,
        cache_nbytes=9,
        durable_tokens=2,
        terminal_proof="mtp-target-only-stable-v1",
    )

    next_turn = _request([1, 2, 3, 4])
    assert scheduler._restore_exact_resident_cache(next_turn)
    assert next_turn.cached_tokens == 3
    assert next_turn.remaining_tokens == [4]


def test_scheduler_cancelled_lease_is_not_reinserted():
    scheduler = _scheduler()
    completed = _request([1, 2, 3])
    cache = [_OffsetCache(3, nbytes=9)]
    scheduler._stage_exact_resident_cache(completed, cache, [1, 2, 3])
    scheduler._publish_exact_resident_cache(completed)

    claimed = _request([1, 2, 3, 4])
    claimed.request_id = "claimed"
    assert scheduler._restore_exact_resident_cache(claimed)
    assert scheduler._exact_resident_stats()["active_leases"] == 1

    scheduler._cache_freshness_waits = {}
    scheduler._prefix_cache_prepared = set()
    scheduler._throttle_notified_requests = set()
    scheduler._memory_admission_blocked_request_id = None
    scheduler._memory_admission_blocked_since = 0.0
    scheduler._store_cache_admission_blocked_request_id = None
    scheduler._store_cache_admission_blocked_since = 0.0
    scheduler._clear_request_admission_bookkeeping("claimed")

    stats = scheduler._exact_resident_stats()
    assert stats["entries"] == 0
    assert stats["active_leases"] == 0
    assert not scheduler._restore_exact_resident_cache(
        _request([1, 2, 3, 4, 5])
    )


def test_scheduler_rejects_unknown_positionless_leaf():
    scheduler = _scheduler()
    request = _request([1, 2, 3])

    scheduler._stage_exact_resident_cache(
        request,
        [_OffsetCache(3, nbytes=9), object()],
        [1, 2, 3],
    )

    assert not hasattr(request, "_exact_resident_candidate")


def test_scheduler_rejects_skewed_or_rollback_terminal_cache():
    scheduler = _scheduler()
    request = _request([1, 2, 3])

    scheduler._stage_exact_resident_cache(
        request, [_OffsetCache(2)], [1, 2, 3]
    )
    assert not hasattr(request, "_exact_resident_candidate")

    scheduler._stage_exact_resident_cache(
        request,
        [_OffsetCache(3, rollback_state=(object(), object()))],
        [1, 2, 3],
    )
    assert not hasattr(request, "_exact_resident_candidate")


def test_scheduler_resident_cache_is_text_only():
    scheduler = _scheduler()
    request = _request([1, 2, 3])
    request.images = [object()]

    scheduler._stage_exact_resident_cache(
        request, [_OffsetCache(3)], [1, 2, 3]
    )

    assert not hasattr(request, "_exact_resident_candidate")
