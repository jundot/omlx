# SPDX-License-Identifier: Apache-2.0
"""Reproduction + fix test for #2219.

External VLM MTP routing lives only on the non-chunked prefill exit
(``scheduler.py`` ~8504). Chunked-prefill'd requests complete via
``_insert_prefilled_request`` and go straight to ``BatchGenerator``, so any prompt
long enough to be chunked silently bypasses VLM MTP.

These tests drive ``_insert_prefilled_request`` -- the single choke point both
chunked-completion paths funnel through -- and assert that, when a vlm_mtp drafter
is present and eligible, the request is routed to VLM MTP instead of
BatchGenerator. They FAIL on the unfixed code (reproducing the bug) and pass once
the routing is applied at the top of ``_insert_prefilled_request``.

No model is loaded: the drafter and BatchGenerator are faked, so the routing
decision is tested in isolation.
"""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

import omlx.scheduler as scheduler_mod
from omlx.request import RequestStatus
from omlx.scheduler import Scheduler


def _make_fixture(monkeypatch, drafter_returns_uid):
    calls = {"route": 0, "bg_insert": 0, "events": []}

    def fake_route(request, cache, last_tokens, sampler, sm):
        calls["route"] += 1
        calls["events"].append("route")
        return -7 if drafter_returns_uid else None  # negative uid, or ineligible

    def fake_bg_insert(*args, **kwargs):
        calls["bg_insert"] += 1
        return [101]  # a positive BatchGenerator uid

    # Module-level helpers reached only on the BatchGenerator path.
    monkeypatch.setattr(scheduler_mod, "_register_uid_rows", lambda *a, **k: None)
    monkeypatch.setattr(scheduler_mod, "_batch_generator_all_tokens", lambda r: [])

    sched = SimpleNamespace(
        _vlm_mtp_drafter=object(),  # a drafter is configured
        _route_to_vlm_mtp=fake_route,
        _finalize_chunked_prefill_cache_for_insert=lambda req, cache: None,
        _stream=mx.default_stream(mx.default_device()),
        batch_generator=SimpleNamespace(insert=fake_bg_insert),
        model=SimpleNamespace(),  # no register_rope_delta attr -> skipped
        request_id_to_uid={},
        uid_to_request_id={},
        running={},
        total_prompt_tokens=0,
    )

    request = SimpleNamespace(
        request_id="req-long-text",
        sampling_params=SimpleNamespace(seed=None, max_tokens=1024),
        num_prompt_tokens=32768,  # long enough to have been chunk-prefilled
        rope_deltas=0.0,
        cached_tokens=0,
        batch_uid=None,
        status=None,
        generation_started_at=None,
        last_activity_at=None,
    )
    state = SimpleNamespace(
        cache=[object()],  # non-empty prefilled cache
        last_token=[42],
        sampler=lambda x: x,
        sm=object(),
        per_row_lps=[],
    )
    return sched, request, state, [], calls


def test_chunked_prefilled_request_routes_to_vlm_mtp_when_eligible(monkeypatch):
    """#2219: a chunked-prefill'd request with an eligible vlm_mtp drafter must be
    routed to VLM MTP, not silently dropped into BatchGenerator."""
    sched, request, state, scheduled, calls = _make_fixture(
        monkeypatch, drafter_returns_uid=True
    )

    Scheduler._insert_prefilled_request(sched, request, state, scheduled)

    assert calls["route"] == 1, "vlm_mtp routing was never considered (the #2219 bug)"
    assert (
        calls["bg_insert"] == 0
    ), "request went to BatchGenerator despite eligible vlm_mtp"
    # negative-uid bookkeeping mirrors the non-chunked routing path
    assert request.batch_uid == -7
    assert request.status == RequestStatus.RUNNING
    assert sched.request_id_to_uid["req-long-text"] == -7
    assert sched.uid_to_request_id[-7] == "req-long-text"
    assert sched.running["req-long-text"] is request
    assert request in scheduled
    assert sched.total_prompt_tokens == 32768


def test_seed_is_applied_before_vlm_mtp_sampling(monkeypatch):
    """A successful VLM MTP route must honor the request seed before sampling."""
    sched, request, state, scheduled, calls = _make_fixture(
        monkeypatch, drafter_returns_uid=True
    )
    request.sampling_params.seed = 123

    monkeypatch.setattr(
        scheduler_mod.mx.random,
        "seed",
        lambda seed: calls["events"].append(("seed", seed)),
    )

    Scheduler._insert_prefilled_request(sched, request, state, scheduled)

    assert calls["events"] == [("seed", 123), "route"]


def test_falls_back_to_batch_generator_when_drafter_ineligible(monkeypatch):
    """When _route_to_vlm_mtp declines (e.g. drafter busy under concurrency), the
    request must fall through to BatchGenerator -- not error or double-schedule."""
    sched, request, state, scheduled, calls = _make_fixture(
        monkeypatch, drafter_returns_uid=False
    )

    Scheduler._insert_prefilled_request(sched, request, state, scheduled)

    assert calls["route"] == 1  # routing was considered
    assert calls["bg_insert"] == 1  # but fell back to BatchGenerator
    assert request.batch_uid == 101  # got the BatchGenerator uid
    assert request in scheduled


def test_routed_request_is_scheduled_exactly_once(monkeypatch):
    """A vlm_mtp-routed request must not also hit the BatchGenerator insert."""
    sched, request, state, scheduled, calls = _make_fixture(
        monkeypatch, drafter_returns_uid=True
    )

    Scheduler._insert_prefilled_request(sched, request, state, scheduled)

    assert len(scheduled) == 1
    assert calls["route"] + calls["bg_insert"] == 1  # exactly one path taken
