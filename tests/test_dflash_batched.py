"""Batched DFlash drafter: ring context, batched forward oracle, scheduler hooks."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

from omlx.scheduler import Scheduler
from omlx.speculative import dflash_drafter as dd

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = 6
TARGET_LAYER_IDS = [1, 3, 5]
WINDOW = 12
BLOCK = 4


def _tiny_config(**overrides):
    params = {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "num_target_layers": TARGET_LAYERS,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
        "dflash_config": {
            "block_size": BLOCK,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB - 1,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
    }
    params.update(overrides)
    return DFlash2Config.from_dict(params)


def _tiny_target():
    embed = nn.Embedding(VOCAB, HIDDEN)
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language = SimpleNamespace(
        config={
            "text_config": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": TARGET_LAYERS,
                "vocab_size": VOCAB,
            }
        },
        model=SimpleNamespace(layers=[object()] * TARGET_LAYERS, embed_tokens=embed),
        lm_head=lm_head,
        rollback_speculative_cache=lambda *args, **kwargs: None,
    )
    mx.eval(embed.parameters(), lm_head.parameters())
    return SimpleNamespace(language_model=language)


def _tiny_drafter(seed=0):
    mx.random.seed(seed)
    model = DFlash2DraftModel(_tiny_config())
    # Random weights in float32 so both forwards share tight numerics.
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny")


def _captured(n, seed):
    mx.random.seed(seed)
    return [mx.random.normal((1, n, HIDDEN)) for _ in TARGET_LAYER_IDS]


def _run_cycles(drafter, plan, cycle_offset=0):
    """plan: per cycle, {uid: (segment_len, anchor)}; returns proposals per cycle."""
    outputs = []
    for cycle, rows in enumerate(plan, start=cycle_offset):
        jobs = []
        states = []
        for uid, (n, anchor) in rows.items():
            state = SimpleNamespace(
                uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
            )
            committed = mx.array([anchor], dtype=mx.uint32)
            jobs.append(
                (
                    None,
                    state,
                    _captured(n, seed=1000 * cycle + uid * 7 + n),
                    committed,
                    None,
                )
            )
            states.append((uid, state))
        drafter.draft(jobs)
        outputs.append({uid: state.drafts.tolist() for uid, state in states})
    return outputs


def test_batched_forward_matches_rows_drafted_alone():
    """Padding, masks, vector RoPE offsets and ring writes must not leak across rows."""
    plan = [
        {0: (5, 3), 1: (1, 9)},
        {0: (2, 11), 1: (3, 4), 2: (7, 5)},
        {0: (4, 8), 1: (1, 1), 2: (2, 6)},
        {0: (6, 2), 2: (5, 7)},
        {0: (3, 12), 2: (9, 13)},
    ]
    expected = [{} for _ in plan]
    for uid in (0, 1, 2):
        alone = _tiny_drafter()
        solo_plan = [{uid: rows[uid]} if uid in rows else {} for rows in plan]
        for cycle, rows in enumerate(solo_plan):
            if rows:
                expected[cycle].update(
                    _run_cycles(alone, [rows], cycle_offset=cycle)[0]
                )

    batched = _tiny_drafter()
    got = _run_cycles(batched, plan)
    assert got == expected
    # The ring saw every context token, including the ones that fell out of
    # the window on the long segments.
    assert batched.context_length(0) == sum(rows[0][0] for rows in plan if 0 in rows)
    assert batched.context_length(2) == sum(rows[2][0] for rows in plan if 2 in rows)


def test_predraft_adopt_matches_drafting_committed_rows():
    """A predraft over every verify row equals drafting the committed rows, and
    a discarded one leaves the ring as it was, across ring wrap-around."""

    def cycle(drafter, uid, captured, count, anchor, mode):
        state = SimpleNamespace(
            uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        if mode == "plain":
            committed = mx.array([anchor], dtype=mx.uint32)
            drafter.draft(
                [(None, state, [c[:, :count] for c in captured], committed, None)]
            )
        else:
            assert drafter.predraft(
                None, state, captured, mx.array([count - 1]) + 1, mx.array([anchor])
            )
            if mode == "adopt":
                drafter.adopt_predraft(state, count)
            else:
                drafter.discard_predraft()
                committed = mx.array([anchor], dtype=mx.uint32)
                drafter.draft(
                    [(None, state, [c[:, :count] for c in captured], committed, None)]
                )
        return state.drafts.tolist()

    counts = [2, 4, 1, 4, 3, 2, 4, 1]
    for mode in ("adopt", "discard"):
        plain, other = _tiny_drafter(), _tiny_drafter()
        for d in (plain, other):
            d.seed(0, _captured(WINDOW - 3, seed=77))
            d.draft(
                [
                    (
                        None,
                        SimpleNamespace(
                            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
                        ),
                        [],
                        mx.array([2], dtype=mx.uint32),
                        None,
                    )
                ]
            )
        for step, count in enumerate(counts):
            captured = _captured(BLOCK, seed=500 + step)
            expected = cycle(plain, 0, captured, count, step + 3, "plain")
            assert cycle(other, 0, captured, count, step + 3, mode) == expected
        assert other.context_length(0) == plain.context_length(0)


def test_pending_captures_stay_within_the_window():
    """Decode steps with drafting off keep only the attended rows, at their positions."""
    kept, tail = _tiny_drafter(), _tiny_drafter()
    rows = _captured(30, seed=9)
    for j in range(30):
        kept.observe([0], [layer[:, j : j + 1] for layer in rows])
    assert sum(p.shape[1] for p in kept._rows[0].pending) == kept.ring_slots
    tail._row(0).fed = 30 - tail.ring_slots
    tail.seed(0, [layer[:, 30 - tail.ring_slots :] for layer in rows])
    drafts = []
    for drafter in (kept, tail):
        state = SimpleNamespace(
            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        drafter.draft([(None, state, [], mx.array([5], dtype=mx.uint32), None)])
        drafts.append(state.drafts.tolist())
    assert drafts[0] == drafts[1]
    assert kept.context_length(0) == tail.context_length(0) == 30


def test_release_detaches_rows_and_new_cohort_reuses_ring():
    drafter = _tiny_drafter()
    _run_cycles(drafter, [{0: (3, 1), 1: (2, 2)}])
    assert drafter._cohort is not None and drafter._cohort.uids == (0, 1)
    drafter.release([1])
    assert drafter._cohort is None
    assert drafter._rows[0].keys is not None
    _run_cycles(drafter, [{0: (1, 3), 5: (2, 4)}])
    assert drafter._cohort.uids == (0, 5)
    assert drafter.context_length(0) == 4 and drafter.context_length(5) == 2


def test_prefill_seed_binds_to_uid_and_window_slicing():
    drafter = _tiny_drafter()
    drafter.seed_request("req", _captured(3, seed=1))
    drafter.bind_uid("req", 7)
    assert drafter._request_seeds == {}
    assert len(drafter._rows[7].pending) == 1
    drafter.release_request("req")

    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=drafter))
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    # Chunk [0, 10) ends before the last WINDOW=12 tokens: nothing to capture.
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, scheduler.model, 0, 10, kwargs
        )
        is None
    )
    assert "capture_layer_ids" not in kwargs
    # Chunk [10, 25) overlaps the window starting at 30 - 12 = 18.
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, scheduler.model, 10, 15, kwargs
    )
    assert keep == 8
    assert kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    # A wrapped prefill model (ANE, specprefill) cannot capture.
    assert (
        Scheduler._dflash_prefill_capture(scheduler, request, object(), 10, 15, {})
        is None
    )

    output = SimpleNamespace(hidden_states=_captured(15, seed=2))
    Scheduler._dflash_seed_prefill(scheduler, request, output, keep)
    assert drafter._request_seeds["r"][0].shape == (
        1,
        7,
        HIDDEN * len(TARGET_LAYER_IDS),
    )


def test_sampled_rows_get_sparse_candidate_distributions():
    """Stochastic rows sample from the selector's candidates and expose q."""
    from omlx.utils.sampling import make_sampler

    mx.random.seed(5)
    drafter = _tiny_drafter()
    sampler = make_sampler(temp=1.0)
    rows = []
    for uid, seed in ((0, None), (1, sampler), (2, sampler)):
        rows.append(
            (
                SimpleNamespace(uid=uid),
                drafter._row(uid),
                mx.concatenate(_captured(3, seed=uid + 40), axis=-1),
                mx.array([uid + 1], dtype=mx.int32),
                seed,
            )
        )
    proposals = drafter._draft_batched(rows)
    assert len(proposals) == 3
    greedy_tokens, greedy_q = proposals[0]
    assert greedy_tokens.shape == (1, BLOCK - 1) and greedy_q == []
    for tokens, accept in proposals[1:]:
        assert tokens.shape == (1, BLOCK - 1)
        assert len(accept) == BLOCK - 1
        for position, q in enumerate(accept):
            assert q.shape == (VOCAB,)
            probs = mx.exp(q)
            # Mass lives on at most top_k candidates and includes the draft.
            assert (probs > 0).sum().item() <= drafter.model.candidate_selector.top_k
            assert abs(probs.sum().item() - 1.0) < 1e-3
            assert probs[tokens[0, position]].item() > 0


def test_resolve_block_size_clamps_to_trained_block_and_mtp_limit():
    model = SimpleNamespace(config=SimpleNamespace(block_size=8))
    assert dd.resolve_block_size(model, None) == 8
    assert dd.resolve_block_size(model, 5) == 5
    assert dd.resolve_block_size(model, 16) == 8
    model.config.block_size = 16
    assert dd.resolve_block_size(model, None) == dd.MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    with pytest.raises(ValueError):
        dd.resolve_block_size(model, 1)
