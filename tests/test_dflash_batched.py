"""Batched DFlash drafter: ring context, batched forward oracle, scheduler hooks."""

from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel
from mlx_vlm.speculative.drafters.dspark.config import DSparkConfig
from mlx_vlm.speculative.drafters.dspark.dspark import DSparkDraftModel
from mlx_vlm.speculative.drafters.laguna_dflash import (
    LagunaDFlashDraftModel,
)
from mlx_vlm.speculative.drafters.laguna_dflash import (
    ModelConfig as LagunaDFlashConfig,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    ModelConfig as MuseAssistantConfig,
)
from mlx_vlm.speculative.drafters.muse_glimmer_assistant import (
    MuseGlimmerAssistantDraftModel,
)
from mlx_vlm.speculative.drafters.qwen3_dflash.config import (
    DFlashConfig as V1DFlashConfig,
)
from mlx_vlm.speculative.drafters.qwen3_dflash.dflash import DFlashDraftModel

from omlx.model_settings import ModelSettings
from omlx.request import Request, RequestStatus, SamplingParams
from omlx.scheduler import Scheduler
from omlx.speculative import dflash as dd
from omlx.speculative.dflash import kernels, selection, variants

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


def _tiny_laguna_drafter(seed=0):
    """Fused QKV, per-head output gate, causal block, per-capture norms."""
    config = LagunaDFlashConfig.from_dict(
        {
            "model_type": "laguna",
            "hidden_size": HIDDEN,
            "intermediate_size": 48,
            "num_hidden_layers": len(TARGET_LAYER_IDS),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": VOCAB,
            "draft_vocab_size": VOCAB,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "layer_types": ["sliding_attention"] * len(TARGET_LAYER_IDS),
            "sliding_window": WINDOW,
            "gating": "per-head",
            "eagle_aux_hidden_state_layer_ids": [i + 1 for i in TARGET_LAYER_IDS],
            "dflash_config": {
                "block_size": BLOCK,
                "mask_token_id": VOCAB - 1,
                "target_layer_ids": TARGET_LAYER_IDS,
                "num_target_layers": TARGET_LAYERS,
                "causal": True,
            },
        }
    )
    mx.random.seed(seed)
    model = LagunaDFlashDraftModel(config)
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    target = _tiny_target()
    target.language_model.config = {
        "num_hidden_layers": TARGET_LAYERS,
        "vocab_size": VOCAB,
    }
    model.bind(target)
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny")


def _tiny_muse_drafter(seed=0):
    """Encoder-held context projection, plain q/k/v projections."""
    config = MuseAssistantConfig.from_dict(
        {
            "hidden_size": HIDDEN,
            "intermediate_size": 48,
            "num_hidden_layers": len(TARGET_LAYER_IDS),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 4096,
            "sliding_window": WINDOW,
            "block_size": BLOCK,
            "mask_token_id": VOCAB - 1,
            "target_layer_ids": TARGET_LAYER_IDS,
            "num_target_layers": TARGET_LAYERS,
            "vocab_size": VOCAB,
        }
    )
    mx.random.seed(seed)
    model = MuseGlimmerAssistantDraftModel(config)
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    target = _tiny_target()
    target.language_model.config = SimpleNamespace(
        model_type="muse_glimmer_text",
        hidden_size=HIDDEN,
        num_hidden_layers=TARGET_LAYERS,
        vocab_size=VOCAB,
    )
    model.bind(target)
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny")


def _tiny_dspark_drafter(seed=0):
    """Anchor-slot block layout and Markov-corrected proposals."""
    config = DSparkConfig.from_dict(
        {
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
            # DSpark checkpoints count proposals; the verify block adds the anchor.
            "block_size": BLOCK - 1,
            "dflash_config": {
                "markov_rank": 8,
                "mask_token_id": VOCAB - 1,
                "projector_type": "dspark",
                "target_layer_ids": TARGET_LAYER_IDS,
            },
        }
    )
    mx.random.seed(seed)
    model = DSparkDraftModel(config)
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    return dd.DFlashDrafter(model, block_size=BLOCK, source_path="tiny")


DRAFTER_FAMILIES = pytest.mark.parametrize(
    "make_drafter",
    [_tiny_drafter, _tiny_laguna_drafter, _tiny_muse_drafter, _tiny_dspark_drafter],
    ids=["dflash2", "laguna", "muse", "dspark"],
)


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


@DRAFTER_FAMILIES
@pytest.mark.parametrize("window", [1, WINDOW])
def test_batched_forward_matches_rows_drafted_alone(make_drafter, window):
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
        alone = make_drafter()
        alone._window_size = window
        solo_plan = [{uid: rows[uid]} if uid in rows else {} for rows in plan]
        for cycle, rows in enumerate(solo_plan):
            if rows:
                expected[cycle].update(
                    _run_cycles(alone, [rows], cycle_offset=cycle)[0]
                )

    batched = make_drafter()
    batched._window_size = window
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
    Scheduler._dflash_seed_prefill(scheduler, request, output, keep, 10 + keep)
    assert drafter._request_seeds["r"][0].shape == (
        1,
        7,
        HIDDEN * len(TARGET_LAYER_IDS),
    )
    assert drafter._request_starts["r"] == 18


def _draft_one(drafter, uid, captured, anchor):
    state = SimpleNamespace(uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None)
    drafter.draft([(None, state, captured, mx.array([anchor], dtype=mx.uint32), None)])
    return state.drafts.tolist()


def test_prefix_hit_restores_context_kept_from_finished_request():
    """Cached tokens never reach the drafter; their projected context comes
    from a finished request with the same tokens and drafts like the raw one."""
    raw = _captured(12, seed=3)
    rows = lambda lo, hi: [layer[:, lo:hi] for layer in raw]  # noqa: E731
    tokens = list(range(50, 70))

    drafter = _tiny_drafter()
    drafter.seed_request("a", rows(0, 10), position=0)
    drafter.bind_uid("a", 1)
    _draft_one(drafter, 1, rows(10, 11), 5)
    drafter.release([1])
    drafter.release_request("a", tokens)

    # Other prefixes, or a hit past the kept rows, restore nothing.
    assert drafter.prepare_request("x", [0] * 9, 9) == 0
    assert drafter.prepare_request("y", tokens, 12) == 0

    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=drafter))
    request = SimpleNamespace(prompt_token_ids=tokens[:9] + [1, 2, 3], request_id="b")
    # The first chunk starts at the 9 cached tokens, inside the window.
    Scheduler._dflash_prefill_capture(
        scheduler, request, scheduler.model, 9, 2, {}, first_chunk=True
    )
    restored, start = drafter._request_restored["b"]
    assert (start, restored.shape[1]) == (0, 9)
    drafter.seed_request("b", rows(9, 11), position=9)
    drafter.bind_uid("b", 2)
    assert drafter._rows[2].end == 11
    restored_drafts = _draft_one(drafter, 2, rows(11, 12), 7)

    reference = _tiny_drafter()
    reference.seed(9, rows(0, 11))
    assert restored_drafts == _draft_one(reference, 9, rows(11, 12), 7)


def test_finished_request_keeps_context_at_its_fed_end():
    """Rows queued while the batch decodes without drafting are not part of
    the kept context, so its end is the last fed position."""
    raw = _captured(12, seed=4)
    rows = lambda lo, hi: [layer[:, lo:hi] for layer in raw]  # noqa: E731
    drafter = _tiny_drafter()
    drafter.seed_request("a", rows(0, 6), position=0)
    drafter.bind_uid("a", 1)
    _draft_one(drafter, 1, rows(6, 8), 5)
    for position in range(8, 11):
        drafter.observe((1,), rows(position, position + 1))
    drafter.release_request("a", list(range(50, 70)))
    kept = drafter._kept["a"]
    assert (kept.end, kept.context.shape[1]) == (8, 8)


def test_released_context_is_dropped_once_its_request_is_gone():
    """A row the batch removes after its request finished, or a request that
    finishes without tokens, leaves no released context behind."""
    drafter = _tiny_drafter()
    drafter.seed_request("a", _captured(4, seed=5), position=0)
    drafter.bind_uid("a", 1)
    _draft_one(drafter, 1, _captured(1, seed=6), 5)
    # A scheduler-side stop finishes the request before the batch removal.
    drafter.release_request("a", list(range(50, 70)))
    drafter.release([1])
    assert not drafter._released

    drafter.seed_request("m", _captured(4, seed=7), position=0)
    drafter.bind_uid("m", 2)
    _draft_one(drafter, 2, _captured(1, seed=8), 6)
    drafter.release([2])
    drafter.release_request("m")
    assert not drafter._released


def test_resumed_prefill_keeps_its_first_preparation():
    """A prefill paused for eviction resumes with more cached tokens; the
    seeds of its first attempt already cover them, so nothing is restored
    on top."""
    raw = _captured(12, seed=9)
    rows = lambda lo, hi: [layer[:, lo:hi] for layer in raw]  # noqa: E731
    tokens = list(range(50, 70))
    drafter = _tiny_drafter()
    drafter.seed_request("a", rows(0, 10), position=0)
    drafter.bind_uid("a", 1)
    _draft_one(drafter, 1, rows(10, 11), 5)
    drafter.release([1])
    drafter.release_request("a", tokens)

    assert drafter.prepare_request("b", tokens, 4) == 4
    drafter.seed_request("b", rows(4, 6), position=4)
    assert drafter.prepare_request("b", tokens, 6) == 0
    drafter.bind_uid("b", 2)
    assert drafter._rows[2].end == 6


def test_fully_cached_prompt_binds_at_its_cached_position():
    """With every prompt token but the last cached and no kept context, the
    row still starts at the cached position, so it can be kept later."""
    drafter = _tiny_drafter()
    assert drafter.prepare_request("a", list(range(50, 70)), 19) == 0
    drafter.bind_uid("a", 1)
    assert drafter._rows[1].end == 19
    _draft_one(drafter, 1, _captured(1, seed=10), 5)
    drafter.release_request("a", list(range(50, 71)))
    assert drafter._kept["a"].end == 20


def test_new_batch_generator_resets_drafter_rows():
    """A replaced generator numbers rows from 0 again; rows of the old one
    must not leak into new requests, while kept contexts stay."""
    drafter = _tiny_drafter()
    drafter.seed_request("a", _captured(4, seed=11), position=0)
    drafter.bind_uid("a", 0)
    _draft_one(drafter, 0, _captured(1, seed=12), 5)
    drafter.release([0])
    drafter.release_request("a", list(range(50, 70)))
    drafter.seed_request("b", _captured(3, seed=13), position=0)
    drafter.bind_uid("b", 0)
    drafter.seed_request("c", _captured(2, seed=14), position=0)

    scheduler = SimpleNamespace(
        model=SimpleNamespace(_omlx_drafter=drafter),
        batch_generator=None,
        _create_batch_generator=lambda params: object(),
    )
    Scheduler._ensure_batch_generator(scheduler, SamplingParams())
    assert not drafter._rows and not drafter._request_uids
    assert "a" in drafter._kept and "c" in drafter._request_seeds


@pytest.mark.parametrize("failure", ["corruption", "overflow"])
def test_reprefill_retry_discards_failed_drafter_context(
    failure, mock_model, mock_tokenizer, monkeypatch
):
    drafter = _tiny_drafter()
    tokens = list(range(6))
    drafter.seed_request("finished", _captured(4, seed=90), position=0)
    drafter.bind_uid("finished", 0)
    _draft_one(drafter, 0, _captured(1, seed=91), 7)
    drafter.release([0])
    drafter.release_request("finished", tokens)

    mock_model._omlx_drafter = drafter
    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    monkeypatch.setattr(scheduler, "_create_batch_generator", lambda params: object())
    try:
        request = Request(
            request_id="retry",
            prompt="example",
            prompt_token_ids=tokens,
            num_prompt_tokens=len(tokens),
            sampling_params=SamplingParams(),
            status=RequestStatus.RUNNING,
            cached_tokens=3,
        )
        scheduler.requests[request.request_id] = request
        scheduler.prefilling.append(request)
        assert drafter.prepare_request(request.request_id, tokens, 3) == 3
        drafter.seed_request(request.request_id, _captured(2, seed=92), position=3)

        if failure == "corruption":
            scheduler._recover_from_cache_error()
            scheduler._reschedule_running_requests(is_corruption=True)
        else:
            scheduler._recover_from_generation_overflow_error()
            scheduler._reschedule_generation_overflow_requests()

        assert list(scheduler.waiting) == [request]
        assert request.cached_tokens == 0
        scheduler._ensure_batch_generator(request.sampling_params)
        drafter.seed_request(request.request_id, _captured(5, seed=93), position=0)
        drafter.bind_uid(request.request_id, 0)
        assert drafter._rows[0].end == 5
        assert drafter._rows[0].pending_len == 5

        _draft_one(drafter, 0, _captured(1, seed=94), 7)
        drafter.release([0])
        drafter.release_request(request.request_id, tokens + [7, 8])
        assert drafter.prepare_request("next", tokens, len(tokens)) == len(tokens)
    finally:
        scheduler.shutdown()
        scheduler.deep_reset()


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


@pytest.mark.parametrize(
    "make_drafter, longest",
    [
        (_tiny_drafter, WINDOW - 1),
        (_tiny_laguna_drafter, WINDOW - 1),
        # Muse's reference windows each query; keep the context inside the
        # window of every block query.
        (_tiny_muse_drafter, WINDOW - BLOCK + 1),
        (_tiny_dspark_drafter, WINDOW - 1),
    ],
    ids=["dflash2", "laguna", "muse", "dspark"],
)
def test_short_context_matches_reference_draft_block(make_drafter, longest):
    """Ring slots not yet written must stay out of attention."""
    for length in (1, 3, longest):
        drafter = make_drafter()
        context = mx.concatenate(_captured(length, seed=60 + length), axis=-1)
        anchor = mx.array([7], dtype=mx.int32)
        cache = drafter.model.make_cache()
        for layer_cache in cache:
            layer_cache.offset = 0
        expected = drafter.model.draft_block(
            anchor, context, cache, BLOCK, lambda logits: mx.argmax(logits, axis=-1)
        )
        row = drafter._row(0)
        got = drafter._draft_batched(
            [(SimpleNamespace(uid=0), row, context, anchor, None)]
        )[0][0]
        if isinstance(got, list):
            got = mx.concatenate(got)[None]
        assert got.tolist() == expected.tolist()


def test_mimo_value_scale_and_mask_embedding_match_reference():
    """MiMo scales attention values and embeds masked block positions with a
    trained vector. The draft must equal a reference whose v_proj weights
    carry the scale and whose mask token embeds as that vector."""
    scale = 0.6
    mx.random.seed(11)
    mask = mx.random.normal((HIDDEN,))

    def build():
        config = V1DFlashConfig.from_dict(
            {
                "hidden_size": HIDDEN,
                "intermediate_size": 48,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": VOCAB,
                "max_position_embeddings": 4096,
                "tie_word_embeddings": False,
                "dflash_config": {
                    "mask_token_id": VOCAB - 1,
                    "target_layer_ids": TARGET_LAYER_IDS,
                    "num_target_layers": TARGET_LAYERS,
                    "attention_sink_bias": True,
                },
            }
        )
        mx.random.seed(0)
        model = DFlashDraftModel(config)
        params = {
            key: mx.random.normal(value.shape) * 0.2
            for key, value in nn.utils.tree_flatten(model.parameters())
        }
        model.load_weights(list(params.items()), strict=False)
        return model, _tiny_target()

    model, target = build()
    model.bind(target)
    drafter = dd.DFlashDrafter(
        model,
        block_size=BLOCK,
        source_path="tiny",
        value_scale=scale,
        mask_embedding=mask,
    )

    reference, ref_target = build()
    for layer in reference.layers:
        layer.self_attn.v_proj.weight = layer.self_attn.v_proj.weight * scale
    embed = ref_target.language_model.model.embed_tokens
    embed.weight[VOCAB - 1] = mask
    reference.bind(ref_target)
    mx.eval(reference.parameters(), embed.parameters())

    context = mx.concatenate(_captured(5, seed=90), axis=-1)
    anchor = mx.array([7], dtype=mx.int32)
    cache = reference.make_cache()
    expected = reference.draft_block(
        anchor, context, cache, BLOCK, lambda logits: mx.argmax(logits, axis=-1)
    )
    got = drafter._draft_batched(
        [(SimpleNamespace(uid=0), drafter._row(0), context, anchor, None)]
    )[0][0]
    assert got.tolist() == expected.tolist()


def test_mimo_mask_embedding_loads_for_another_thread(tmp_path):
    """The drafter loads on one executor thread and drafts on the engine's."""
    values = np.arange(HIDDEN, dtype=np.float32)
    path = tmp_path / "mask_embedding.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mask_embedding/byteorder", "little")
        archive.writestr(
            "mask_embedding/data/0",
            (values.view(np.uint32) >> 16).astype(np.uint16).tobytes(),
        )
    with ThreadPoolExecutor(max_workers=1) as executor:
        mask = executor.submit(variants.load_mask_embedding, path, HIDDEN).result()
    assert mask.dtype == mx.bfloat16
    assert (mask.astype(mx.float32) + 1).tolist() == (values + 1).tolist()


def test_prefill_capture_accepts_bound_prefill_of_the_model():
    class Model:
        def _omlx_prefill(self, *args, **kwargs):
            return None

    model = Model()
    model._omlx_drafter = _tiny_drafter()
    scheduler = SimpleNamespace(model=model)
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, model._omlx_prefill, 10, 15, kwargs
    )
    assert keep == 8 and kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    other = Model()
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, other._omlx_prefill, 10, 15, {}
        )
        is None
    )


class _Selector(nn.Module):
    def __init__(self, vocab, rank, hidden):
        super().__init__()
        self.top_k = 16
        self.predecessor_codebook = nn.Embedding(vocab, rank)
        self.successor_codebook = nn.Embedding(vocab, rank)
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)


def test_fused_selector_matches_candidate_sampling():
    """One-launch selector: q equals ``_sample_candidates`` on the same path."""
    from omlx.utils.sampling import make_sampler, top_k_indices

    mx.random.seed(8)
    vocab, rank, hidden, batch, length = 600, 64, 32, 2, BLOCK - 1
    selector = _Selector(vocab, rank, hidden)
    selector.update(
        nn.utils.tree_map(lambda p: (p * 3).astype(mx.bfloat16), selector.parameters())
    )
    states = mx.random.normal((batch, length, hidden)).astype(mx.bfloat16)
    logits = (mx.random.normal((batch, length, vocab)) * 4).astype(mx.bfloat16)
    anchors = mx.array([3, 9], dtype=mx.int32)
    candidates = top_k_indices(logits, 16)
    unary = mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)
    projected = selector.hidden_projection(states).astype(mx.float32)
    for sampler in (make_sampler(temp=1.0, top_p=0.9, top_k=12), None):
        assert selection.fused_select_eligible(selector, states)
        proposals = selection.select_fused(
            selector, states, logits, anchors, [sampler, sampler]
        )
        for row, (tokens, accept) in enumerate(proposals):
            tokens = tokens.reshape(-1).tolist()
            previous = int(anchors[row])
            for position in range(length):
                edges = mx.sum(
                    selector.predecessor_codebook.weight[previous].astype(mx.float32)
                    * projected[row, position]
                    * selector.successor_codebook.weight[
                        candidates[row, position]
                    ].astype(mx.float32),
                    axis=-1,
                )
                scores = (unary[row, position] + edges)[None]
                picked = candidates[row, position].tolist().index(tokens[position])
                if sampler is None:
                    assert picked == int(mx.argmax(scores[0]).item())
                else:
                    _, expected = selection.sample_candidates(scores, sampler)
                    got = accept.logq[position]
                    assert mx.allclose(got, expected[0], atol=1e-4).item()
                    assert got[picked].item() > -float("inf")
                previous = tokens[position]


def test_conv_kernel_matches_grouped_dynamic_convolve():
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
        GroupedDynamicCausalConv,
    )

    mx.random.seed(4)
    conv = GroupedDynamicCausalConv(256, 2, 16)
    conv.base_kernel = (mx.random.normal((2, 2, 256)) * 0.5).astype(mx.bfloat16)
    conv.kernel_projection.weight = (mx.random.normal((64, 256)) * 0.05).astype(
        mx.bfloat16
    )
    x = mx.random.normal((3, BLOCK, 256)).astype(mx.bfloat16)
    expected, dynamic = conv.prepare(x)
    got, packed = kernels.conv_prepare(conv, x)
    assert mx.array_equal(got, expected).item()
    y = expected * 0.5
    assert mx.array_equal(
        kernels.conv_finish(conv, y, packed), conv.finish(y, dynamic)
    ).item()


def test_resolve_block_size_clamps_to_trained_block_and_mtp_limit():
    model = SimpleNamespace(config=SimpleNamespace(block_size=8))
    assert dd.resolve_block_size(model, None) == 8
    assert dd.resolve_block_size(model, 5) == 5
    assert dd.resolve_block_size(model, 16) == 8
    model.config.block_size = 16
    assert dd.resolve_block_size(model, None) == dd.MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    # GLM-5.3's pooling undo log bounds a verify block at 8 rows.
    assert dd.resolve_block_size(model, None, max_rows=8) == 8
    with pytest.raises(ValueError):
        dd.resolve_block_size(model, 1)


def test_verify_mode_selects_adaptive_depth():
    assert dd.resolve_verify_mode(None) is True
    assert dd.resolve_verify_mode("adaptive") is True
    assert dd.resolve_verify_mode("dflash") is False
    assert dd.resolve_verify_mode("off") is False
    with pytest.raises(ValueError):
        dd.resolve_verify_mode("tree")
    # Settings saved before ddtree was removed load as adaptive.
    assert ModelSettings(dflash_verify_mode="ddtree").dflash_verify_mode == "adaptive"


def _tiny_minimax_dspark(seed=0, depth=None, fixed=False):
    from omlx.patches.minimax_m3_dspark import (
        MiniMaxDSparkConfig,
        MiniMaxDSparkDraftModel,
        wrap_minimax_dspark,
    )

    config = MiniMaxDSparkConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rotary_dim=4,
        vocab_size=VOCAB,
        ffn_hidden_size=48,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        swiglu_beta=1.0,
        block_size=BLOCK,
        mask_token_id=VOCAB - 1,
        target_layer_ids=TARGET_LAYER_IDS,
        sliding_window=WINDOW,
        markov_rank=8,
    )
    mx.random.seed(seed)
    model = MiniMaxDSparkDraftModel(config)
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=True)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    return wrap_minimax_dspark(model, "tiny", depth, fixed)


def _reference_minimax_dspark(model, context, anchor):
    """Greedy DSpark block straight from the definition.

    Context positions are 0..n-1 and the block sits at n..n+B-1. Query j
    sees context positions >= n + j - (window - 1) and the whole block; every
    block position proposes, each with the Markov bias of the previous token.
    """
    config = model.config
    n, block = context.shape[1], config.block_size
    slots = config.sliding_window - 1
    ctx = model.hidden_norm(model.fc(context))
    inputs = mx.array([[anchor] + [config.mask_token_id] * (block - 1)])
    h = model._embed_input_tokens(inputs)
    visible = mx.arange(n)[None, :] >= (n + mx.arange(block))[:, None] - slots
    mask = mx.concatenate([visible, mx.ones((block, block), dtype=mx.bool_)], axis=1)
    for layer in model.layers:
        attn = layer.self_attn

        def heads(x, count, length, head_dim=attn.head_dim):
            return x.reshape(1, length, count, head_dim).transpose(0, 2, 1, 3)

        x = layer.input_layernorm(h)
        q = model.rope(
            attn.q_norm(heads(attn.q_proj(x), attn.n_heads, block)), offset=n
        )
        kb = model.rope(
            attn.k_norm(heads(attn.k_proj(x), attn.n_kv_heads, block)), offset=n
        )
        kc = model.rope(
            attn.k_norm(heads(attn.k_proj(ctx), attn.n_kv_heads, n)), offset=0
        )
        vb = heads(attn.v_proj(x), attn.n_kv_heads, block)
        vc = heads(attn.v_proj(ctx), attn.n_kv_heads, n)
        out = mx.fast.scaled_dot_product_attention(
            q,
            mx.concatenate([kc, kb], axis=2),
            mx.concatenate([vc, vb], axis=2),
            scale=attn.scale,
            mask=mask[None, None],
        )
        h = h + attn.o_proj(out.transpose(0, 2, 1, 3).reshape(1, block, -1))
        h = h + layer.mlp(layer.post_attention_layernorm(h))
    logits = model._logits(model.norm(h))[0]
    proposals, previous = [], anchor
    for position in range(block):
        bias = model.markov_head.markov_w2(
            model.markov_head.markov_w1(mx.array([previous]))
        )[0]
        previous = int(mx.argmax(logits[position] + bias).item())
        proposals.append(previous)
    return proposals


def test_minimax_dspark_block_matches_reference_definition():
    drafter = _tiny_minimax_dspark()
    assert drafter.depth == BLOCK
    contexts, anchors = [], [5, 17, 30]
    # 15 then 4 then 6 context tokens: the window (11 slots) drops old ones.
    for cycle, (n, anchor) in enumerate(zip((15, 4, 6), anchors)):
        captured = _captured(n, seed=50 + cycle)
        contexts.append(mx.concatenate(captured, axis=-1))
        state = SimpleNamespace(
            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        drafter.draft(
            [(None, state, captured, mx.array([anchor], dtype=mx.uint32), None)]
        )
        expected = _reference_minimax_dspark(
            drafter.model, mx.concatenate(contexts, axis=1), anchor
        )
        assert state.drafts.tolist() == expected


def test_minimax_dspark_batched_rows_match_rows_drafted_alone():
    plan = [
        {0: (5, 3), 1: (14, 9)},
        {0: (2, 11), 1: (3, 4), 2: (7, 5)},
        {0: (4, 8), 2: (13, 6)},
    ]
    expected = [{} for _ in plan]
    for uid in (0, 1, 2):
        alone = _tiny_minimax_dspark()
        for cycle, rows in enumerate(plan):
            if uid in rows:
                expected[cycle].update(
                    _run_cycles(alone, [{uid: rows[uid]}], cycle_offset=cycle)[0]
                )
    assert _run_cycles(_tiny_minimax_dspark(), plan) == expected


def test_minimax_dspark_sanitize_maps_checkpoint_layout():
    from omlx.patches.minimax_m3_dspark import sanitize_dspark_weights

    prefix = "language_model.model.dspark."
    raw = mx.zeros((4, 64), dtype=mx.uint8)
    weights = {
        f"{prefix}fc.weight": mx.zeros((8, 16)),
        f"{prefix}hidden_norm.weight": mx.zeros((8,)),
        f"{prefix}final_norm.weight": mx.zeros((8,)),
        f"{prefix}layers.0.decoder_layer.self_attn.q_proj.weight": raw,
        f"{prefix}layers.0.decoder_layer.self_attn.q_proj.weight_scale_inv": mx.zeros(
            (4, 2), dtype=mx.uint8
        ),
        f"{prefix}markov_head.markov_w1.weight": mx.zeros((10, 2)),
        f"{prefix}confidence_head.proj.weight": mx.zeros((1, 10)),
        f"{prefix}confidence_head.proj.bias": mx.zeros((1,)),
    }

    sanitized = sanitize_dspark_weights(weights)

    assert set(sanitized) == {
        "fc.weight",
        "hidden_norm.weight",
        "norm.weight",
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.q_proj.scales",
        "markov_head.markov_w1.weight",
    }
    assert sanitized["layers.0.self_attn.q_proj.weight"].dtype == mx.uint32
    assert sanitized["layers.0.self_attn.q_proj.weight"].shape == (4, 16)


def test_dspark_context_only_cycle_keeps_the_next_block():
    first, second = _captured(5, seed=31), _captured(4, seed=32)
    anchors = (mx.array([7], dtype=mx.uint32), mx.array([12], dtype=mx.uint32))

    drafting = _tiny_minimax_dspark()
    every_cycle = SimpleNamespace(uid=0, depth=BLOCK, controller=None)
    drafting.draft([(None, every_cycle, first, anchors[0], None)])
    drafting.draft([(None, every_cycle, second, anchors[1], None)])

    copying = _tiny_minimax_dspark()
    after_copy = SimpleNamespace(uid=0, depth=BLOCK, controller=None)
    copying.extend_context(after_copy, first)
    copying.draft([(None, after_copy, second, anchors[1], None)])

    assert after_copy.drafts.tolist() == every_cycle.drafts.tolist()


def test_minimax_dspark_depth_verifies_a_prefix_of_the_trained_block():
    captured, anchor = _captured(5, seed=33), mx.array([7], dtype=mx.uint32)
    full = _tiny_minimax_dspark()
    capped = _tiny_minimax_dspark(depth=2, fixed=True)
    assert capped.block_size == full.block_size and not capped.adaptive

    states = []
    for drafter in (full, capped):
        state = SimpleNamespace(uid=0, depth=BLOCK, controller=None)
        drafter.draft([(None, state, captured, anchor, None)])
        states.append(state)
    assert states[1].drafts.tolist() == states[0].drafts.tolist()[:2]


def test_dspark_context_only_cycles_queue_at_most_one_ring():
    drafter = _tiny_minimax_dspark()
    state = SimpleNamespace(uid=0, depth=BLOCK, controller=None)
    for cycle in range(6):
        drafter.extend_context(state, _captured(5, seed=40 + cycle))
    row = drafter._rows[0]
    assert row.pending_len == drafter.ring_slots
    assert row.pending_len + row.fed == 30
