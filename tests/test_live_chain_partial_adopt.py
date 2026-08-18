# SPDX-License-Identifier: Apache-2.0
"""Regression tests for partial adoption of a diverged prefix."""

from types import SimpleNamespace

from omlx.scheduler import Scheduler

BLOCK = 4


class _Provider:
    """Boundary snapshots at every block boundary of the retained chain."""

    def __init__(self, tcs):
        self.tcs = set(tcs)
        self.loaded = []

    def __contains__(self, tc):
        return tc in self.tcs

    def __getitem__(self, tc):
        self.loaded.append(tc)
        # layer 0 is the non-sliceable (GDN) state the snapshot owns; layer 1
        # is the sliceable placeholder the merge refills from the live chain.
        return [
            {"class_name": "ArraysCache", "state": [f"gdn@{tc}"]},
            {"class_name": "KVCache", "state": ()},
        ]


def _scheduler(provider, paged_est=0):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.config = SimpleNamespace(paged_cache_block_size=BLOCK)
    scheduler._boundary_snapshot_store = SimpleNamespace(
        cleanup_request=lambda _rid: None
    )
    scheduler._retention_slot = {
        "tokens": list(range(20)),
        "cache": [
            {"class_name": "ArraysCache", "state": ["gdn@live"]},
            {"class_name": "KVCache", "state": ("k", "v")},
        ],
        "config": None,
        "request_id": "r1",
        # Shaped like a real payload from _build_deferred_spill_store: a stub
        # thinner than the thing it stands for hides breakage in its callers.
        "pending_store": {
            "request_id": "r1",
            "snapshot_owner_id": "r1",
            "tokens": list(range(8)),
            "cache": [{"class_name": "RotatingKVCache"}],
            "config": None,
            "snapshots": provider,
        },
    }
    scheduler._slice_sliceable_layer = lambda _c, k, v, end: (
        f"{k}[:{end}]",
        f"{v}[:{end}]",
    )
    scheduler._construct_retained_layer = lambda layer: layer
    scheduler._collect_arrays_from_extracted_cache = lambda _cache: []
    scheduler._paged_cached_prefix_tokens = lambda _prompt: paged_est
    return scheduler


def _request(prompt):
    return SimpleNamespace(
        request_id="r2", prompt_token_ids=prompt, vlm_extra_keys_for_cache=None
    )


def test_junction_mismatch_adopts_the_block_below_the_divergence(monkeypatch):
    """Diverging near the end must cost one block, not the whole prefill."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    provider = _Provider(range(BLOCK, 21, BLOCK))
    scheduler = _scheduler(provider)
    # shares 0..17, then differs -> largest block boundary <= 18 is 16
    request = _request(list(range(18)) + [900, 901, 902])

    assert scheduler._try_adopt_retained_chain(request) is True
    assert request.cached_tokens == 16
    assert request.remaining_tokens == [16, 17, 900, 901, 902]
    assert provider.loaded == [16]
    # The GDN layer comes from the snapshot, the KV layer from the sliced chain.
    assert request.prompt_cache[0]["state"] == ["gdn@16"]
    assert request.prompt_cache[1]["state"] == ("k[:16]", "v[:16]")
    assert scheduler._retention_slot is None


def test_shorter_prompt_can_still_partially_adopt(monkeypatch):
    """A prompt that does not extend the chain may still share a prefix."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    provider = _Provider(range(BLOCK, 21, BLOCK))
    scheduler = _scheduler(provider)
    request = _request(list(range(10)) + [900])

    assert scheduler._try_adopt_retained_chain(request) is True
    assert request.cached_tokens == 8
    assert request.remaining_tokens == [8, 9, 900]


def test_partial_adoption_leaves_a_token_to_prefill(monkeypatch):
    """A fully cached prompt has nothing left to run a forward pass on."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    provider = _Provider(range(BLOCK, 21, BLOCK))
    scheduler = _scheduler(provider)
    request = _request(list(range(8)))  # common prefix == prompt length

    assert scheduler._try_adopt_retained_chain(request) is True
    assert request.cached_tokens == 4
    assert request.remaining_tokens == [4, 5, 6, 7]


def test_partial_adoption_defers_to_the_paged_path(monkeypatch):
    """Stored blocks already serve this prefix; leave the slot alone."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    provider = _Provider(range(BLOCK, 21, BLOCK))
    scheduler = _scheduler(provider, paged_est=16)
    request = _request(list(range(18)) + [900])

    assert scheduler._try_adopt_retained_chain(request) is False
    assert scheduler._retention_slot is not None


def test_partial_adoption_needs_a_snapshot(monkeypatch):
    """Without per-block snapshots there is no GDN state for the boundary."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = _scheduler(_Provider([]))
    request = _request(list(range(18)) + [900])

    assert scheduler._try_adopt_retained_chain(request) is False
    assert scheduler._retention_slot is not None


def test_no_pending_payload_means_no_partial_adoption(monkeypatch):
    """Configurations that cannot spill also cannot serve a partial prefix."""
    monkeypatch.setenv("OMLX_RETAIN_LIVE_CHAIN", "1")
    scheduler = _scheduler(_Provider(range(BLOCK, 21, BLOCK)))
    scheduler._retention_slot["pending_store"] = None
    request = _request(list(range(18)) + [900])

    assert scheduler._try_adopt_retained_chain(request) is False
