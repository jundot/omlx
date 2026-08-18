# SPDX-License-Identifier: Apache-2.0
"""Regression tests for carrying the spill payload across adoption.

The defect this guards against: a request that adopts its prefix prefills only
the remainder, so it captures no block-aligned boundary snapshots of its own.
When the adopted payload was released, neither the spill payload nor the
ordinary store could be built for that request — and the *next* turn
re-prefilled the whole prompt.
"""

from concurrent.futures import Future
from types import SimpleNamespace

from omlx.scheduler import Scheduler

BLOCK = 4


class _RecordingExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, fn, *args):
        self.submissions.append(args)
        future = Future()
        future.set_result(None)
        return future


class _RecordingSnapshotStore:
    def __init__(self):
        self.cleaned = []

    def cleanup_request(self, request_id):
        self.cleaned.append(request_id)


class _Provider(dict):
    """Stands in for _BoundarySnapshotProvider: keyed by token count."""


def _scheduler(own_provider=None, snapshot_need=True):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler._retention_slot = None
    scheduler.config = SimpleNamespace(paged_cache_block_size=BLOCK)
    scheduler.block_aware_cache = object()
    scheduler._store_cache_executor = _RecordingExecutor()
    scheduler._boundary_snapshot_store = _RecordingSnapshotStore()
    scheduler._memory_hard_limit_bytes = 0
    scheduler.memory_monitor = None
    scheduler._split_gdn_snapshot_provider = lambda _rid, _n: own_provider
    scheduler._boundary_override_snapshot_provider = lambda _rid, _tokens: None
    scheduler._detect_boundary_snapshot_need = lambda: snapshot_need
    scheduler._collect_arrays_from_extracted_cache = lambda _cache: []
    scheduler._eval_calls = []
    scheduler._eval_boundary_provider_snapshots = scheduler._eval_calls.append
    return scheduler


def _request(request_id, **extra):
    return SimpleNamespace(
        request_id=request_id,
        _extracted_cache=[{"state": ()}],
        _model_cache_config=None,
        **extra,
    )


def _adopting_request(request_id, prompt, inherited_tokens, provider, owner="r1"):
    request = _request(request_id, prompt_token_ids=list(prompt))
    request._retention_inherited_store = {
        "request_id": owner,
        "snapshot_owner_id": owner,
        "tokens": list(inherited_tokens),
        "cache": [{"class_name": "RotatingKVCache"}],
        "config": None,
        "snapshots": provider,
    }
    return request


def test_adopting_request_reuses_the_inherited_snapshots():
    """Without its own provider, the payload comes from what it adopted."""
    scheduler = _scheduler(own_provider=None)
    provider = _Provider({8: [{"state": ("k", "v")}]})
    tokens = list(range(11))
    request = _adopting_request("r2", tokens, list(range(8)), provider)

    skip_store, payload = scheduler._build_deferred_spill_store(
        "r2", tokens, [{"class_name": "RotatingKVCache"}], request
    )

    assert skip_store is True
    assert payload is not None
    assert payload["snapshots"] is provider
    # Only the blocks the snapshots actually cover.
    assert payload["tokens"] == list(range(8))
    # The hold still belongs to the request that captured them.
    assert payload["snapshot_owner_id"] == "r1"
    assert request._retention_inherited_store is None


def test_without_the_inheritance_the_ordinary_store_still_runs():
    """The pre-34f outcome, kept as the guard: no snapshots anywhere means the
    completion-time store must not be skipped."""
    scheduler = _scheduler(own_provider=None)
    request = _request("r2", prompt_token_ids=list(range(11)))

    skip_store, payload = scheduler._build_deferred_spill_store(
        "r2", list(range(11)), [{"class_name": "RotatingKVCache"}], request
    )

    assert (skip_store, payload) == (False, None)


def test_own_snapshots_win_and_the_inheritance_is_dropped():
    """A request that captured its own blocks must not pin someone else's."""
    own = _Provider({8: [{"state": ("own", "own")}]})
    scheduler = _scheduler(own_provider=own)
    inherited = _Provider({8: [{"state": ("old", "old")}]})
    tokens = list(range(11))
    request = _adopting_request("r2", tokens, list(range(8)), inherited)

    skip_store, payload = scheduler._build_deferred_spill_store(
        "r2", tokens, [{"class_name": "RotatingKVCache"}], request
    )

    assert payload["snapshots"] is own
    assert payload["snapshot_owner_id"] == "r2"
    # Left for _release_inherited_payload to hand back.
    assert request._retention_inherited_store is not None


def test_a_diverged_prefix_walks_down_to_a_boundary_both_agree_on():
    """Partial adoption can land below the inherited payload's own boundary."""
    scheduler = _scheduler(own_provider=None)
    provider = _Provider({4: [{"state": ("k", "v")}], 8: [{"state": ("k", "v")}]})
    inherited_tokens = list(range(8))
    # Diverges at index 5, so only the first block is common.
    tokens = list(range(5)) + [77, 78, 79, 80]
    request = _adopting_request("r2", tokens, inherited_tokens, provider)

    _skip_store, payload = scheduler._build_deferred_spill_store(
        "r2", tokens, [{"class_name": "RotatingKVCache"}], request
    )

    assert payload["tokens"] == list(range(4))


def test_a_fully_diverged_prefix_is_refused():
    scheduler = _scheduler(own_provider=None)
    provider = _Provider({8: [{"state": ("k", "v")}]})
    tokens = [90 + i for i in range(11)]
    request = _adopting_request("r2", tokens, list(range(8)), provider)

    assert scheduler._build_deferred_spill_store(
        "r2", tokens, [{"class_name": "RotatingKVCache"}], request
    ) == (False, None)


def test_a_spilled_inherited_payload_releases_the_capturing_request():
    """The rmtree must target whoever owns the snapshot directory."""
    scheduler = _scheduler(own_provider=None)
    provider = _Provider({8: [{"state": ("k", "v")}]})
    tokens = list(range(11))
    request = _adopting_request("r2", tokens, list(range(8)), provider)
    scheduler._install_retention_slot(
        "r2", request, tokens, [{"class_name": "RotatingKVCache"}], "generic"
    )
    assert scheduler._retention_slot["pending_store"]["snapshot_owner_id"] == "r1"

    # A third conversation takes the slot; the spill must credit r1.
    scheduler._install_retention_slot(
        "r3",
        _request("r3"),
        list(range(40, 51)),
        [{"class_name": "RotatingKVCache"}],
        "generic",
    )
    (submission,) = scheduler._store_cache_executor.submissions
    assert submission[4] is provider
    scheduler._drain_spill_futures()
    assert scheduler._boundary_snapshot_store.cleaned == ["r1"]


def test_the_inheritance_does_not_pin_the_pre_adoption_chain(monkeypatch):
    """Only the snapshots travel: holding the payload's cache would keep the
    pre-adoption buffers resident next to the ones the request grows into."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = _scheduler(own_provider=_Provider({8: [{"state": ("k", "v")}]}))
    old_chain = [{"class_name": "KVCache", "state": ("old_k", "old_v")}]
    scheduler._install_retention_slot(
        "r1", _request("r1"), list(range(10)), old_chain, "generic"
    )
    scheduler._construct_retained_layer = lambda _layer: object()

    request = SimpleNamespace(
        request_id="r2",
        prompt_token_ids=list(range(10)) + [99],
        vlm_extra_keys_for_cache=None,
    )
    assert scheduler._try_adopt_retained_chain(request) is True

    inherited = request._retention_inherited_store
    assert "cache" not in inherited
    assert inherited["snapshots"] is not None
    assert inherited["tokens"] == list(range(8))
