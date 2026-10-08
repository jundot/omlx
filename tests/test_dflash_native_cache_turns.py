"""Multi-turn insert-growth and pruning behavior of the DFlash native cache.

Each turn runs the full session event flow (prefill boundary captures plus the
end-of-generation publication) over an extending conversation prompt, with the
generated tokens from one turn prepended to the next turn's prompt — the
pattern serving drives every turn. Nothing else in the suite repeats that
cycle, so per-turn accumulation bugs only surface here.
"""

import pytest

from tests.test_dflash_native_cache import install_fixtures, setup  # noqa: F401


def _run_turn(setup, tokens, max_new_tokens=3):
    """Drive one full session (prefill + generation) over ``tokens``.

    Returns (flow, generated_token_ids) with the ids taken from the run's own
    SummaryEvent, so the caller can extend the conversation with what the
    engine actually generated.
    """
    from dflash_mlx.draft_backend import EagerDraftBackend
    from dflash_mlx.engine.events import SummaryEvent
    from dflash_mlx.engine.spec_epoch import (
        SpeculativeSession,
        _RequestState,
        _SessionRequest,
        _YieldPauseTracker,
    )

    cache, draft, backend, provider, context = setup
    flow = cache.for_request(
        model_provider=provider,
        draft_model=draft,
        tokenizer=provider.tokenizer,
        prompt=tokens,
        max_new_tokens=max_new_tokens,
        runtime_context=context,
    )
    from omlx.cache import dflash as bridge

    ops = bridge.NativeCacheTargetOps(provider.target_ops)
    session = SpeculativeSession.open(
        target_model=provider.model,
        draft_model=draft,
        draft_backend=EagerDraftBackend(),
        target_ops=ops,
        supports_prefix_snapshot=True,
        allow_full_context_draft_layers=False,
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
        prefix_snapshot=flow.snapshot,
        quantize_kv_cache=False,
        target_fa_window=0,
        runtime_context=context,
    )
    request = _SessionRequest.from_tokens(
        prompt_tokens=tokens,
        max_new_tokens=max_new_tokens,
        block_tokens=2,
        stop_token_ids=[],
        suppress_token_ids=None,
        prefix_snapshot=flow.snapshot,
        snapshot_service=flow.snapshot_service,
        stable_prefix_len=len(tokens),
        prefix_cache_active=True,
        publish_generation_snapshot=True,
        temperature=0,
        top_p=1,
        top_k=0,
        min_p=0,
    )
    state = _RequestState()
    events = list(session.run_events(request))
    summary = next(e for e in events if isinstance(e, SummaryEvent))
    generated = [int(t) for t in summary.generated_token_ids]
    return flow, events, generated


@pytest.fixture
def four_turn_flow(setup):
    """Run four extending turns; return the last turn's (flow, events)."""
    prompts = [list(range(1, 33))]
    generated: list[int] = []
    flow, events = None, None
    for turn in range(4):
        # The conversation lineage: previous prompt + previously generated
        # tokens + the new user turn.
        prompts.append(prompts[-1] + generated + [90 + turn] * 4)
        flow, events, generated = _run_turn(setup, prompts[-1])
    assert generated, "generation produced no tokens to extend the next turn"
    return flow, events


def test_multi_turn_still_hits_the_whole_prompt(four_turn_flow):
    """Each turn restores the full prompt; generation adds its tokens."""
    from dflash_mlx.engine.events import PrefillCompleteEvent

    flow, events = four_turn_flow
    prefill = next(e for e in events if isinstance(e, PrefillCompleteEvent))
    # The final generation snapshot extends the prompt by the generated
    # tokens, which the next turn's prefill must restore without recompute.
    assert prefill.prefill_tokens_restored == flow.hit_tokens


def test_multi_turn_leaves_no_request_tables(setup):
    """Lookups and stores release their block refs within the turn."""
    prompts = [list(range(1, 33))]
    generated: list[int] = []
    for turn in range(3):
        prompts.append(prompts[-1] + generated + [90 + turn] * 4)
        _run_turn(setup, prompts[-1])
    cache = setup[0]
    paged = cache.prefix.paged_cache
    # Fetch-only callers (for_request's lookups) used to leak one ref-holding
    # block table per lookup, pinning the previous tail against pruning.
    assert len(paged.request_tables) == 0
    assert len(cache.prefix._request_tables) == 0


def test_multi_turn_drop_old_context_files(setup, four_turn_flow):
    """Only the two newest turn tips keep their prefix-context blobs."""
    cache = setup[0]
    stats = cache.ssd.get_stats_dict()
    assert stats["prefix_context_count"] == 2
    # Contexts share the sidecar index with recurrent checkpoints; the metric
    # must count the former without the latter.
    assert 0 < stats["prefix_context_size_bytes"] <= stats["gdn_sidecar_size_bytes"]


def test_multi_turn_tail_blocks_prune_like_general_cache(setup, four_turn_flow):
    """Tail blocks follow the two-turn lineage the general prefix cache keeps."""
    cache = setup[0]
    paged = cache.prefix.paged_cache
    live_tails = [
        h
        for h in cache.prefix._tail_hashes
        if paged.cached_block_hash_to_block.get_block(h) is not None
    ]
    # Steady state: the newest generation tail plus the previous tip kept as
    # the walk-back fallback.
    assert len(live_tails) <= 2


def test_failed_context_save_keeps_previous_context(setup, four_turn_flow, monkeypatch):
    """A failed replacement save must not drop the previous context blob."""
    cache = setup[0]
    ssd = cache.ssd
    before = ssd.get_stats_dict()["prefix_context_count"]
    assert before == 2

    def failing_save(*args, **kwargs):
        return False

    monkeypatch.setattr(ssd, "save_prefix_context", failing_save)
    prompts = [list(range(1, 33))]
    generated: list[int] = []
    for turn in range(4):
        prompts.append(prompts[-1] + generated + [90 + turn] * 4)
        _run_turn(setup, prompts[-1])
    # Nothing new was admitted; the last two real contexts survive.
    assert ssd.get_stats_dict()["prefix_context_count"] == before


def test_context_sizes_reported_for_dashboard(setup, four_turn_flow):
    """The waterfall reports durable drafter-context bytes for diagnosis."""
    fields = setup[0].memory_waterfall_bytes()
    assert fields["prefix_context_files"] == 2
    assert fields["prefix_context_bytes"] > 0
    assert fields["l1_snapshot_draft_context_bytes"] == fields["prefix_context_bytes"]