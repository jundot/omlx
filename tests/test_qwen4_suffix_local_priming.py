# SPDX-License-Identifier: Apache-2.0
"""Qwen4-only verified-drafter suffix-local prompt-priming contracts."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat
from omlx.patches.mlx_lm_mtp import prompt_priming


def _tiny_config():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    from mlx_vlm.models import qwen4_exp

    text = qwen4_exp.TextConfig(
        model_type="qwen4_exp_text",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=3,
        num_experts=4,
        num_experts_per_tok=2,
        shared_expert_intermediate_size=16,
        moe_intermediate_size=16,
        rms_norm_eps=1e-6,
        vocab_size=64,
        num_key_value_heads=2,
        max_position_embeddings=128,
        hc_count=2,
        hc_lowrank=8,
        head_dim=8,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[1],
        ple_embed_dim=32,
        ple_conv_kernel_size=3,
        ngram_size=3,
        heads_per_ngram=2,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=4,
        split_ngram_parts=4,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=8,
        indexer_compress_ratio=2,
        eos_token_id=1,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 1, 1],
            "rope_theta": 10_000,
            "partial_rotary_factor": 1.0,
        },
    )
    vision = qwen4_exp.VisionConfig(
        model_type="qwen4_exp",
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        out_hidden_size=32,
        num_heads=4,
        patch_size=14,
        in_channels=3,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    return qwen4_exp.ModelConfig(
        text_config=text,
        vision_config=vision,
        model_type="qwen4_exp",
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=58,
        vision_end_token_id=59,
        vocab_size=64,
    )


def _model():
    config = _tiny_config()
    from mlx_vlm.models.qwen4_exp.language import (
        LanguageModel,
        Qwen4ExpMTPModule,
    )

    model = LanguageModel(config.text_config, config)
    model.mtp = Qwen4ExpMTPModule(config.text_config)
    model._enable_mtp_decode_markers()
    mx.eval(model.parameters())
    return model


class _NoSidecarPrefixCache:
    block_size = 4

    def __init__(self):
        self.store_calls = []

    def restore_mtp_prefix_snapshot(self, *args, **kwargs):
        return None

    def store_mtp_prefix_snapshot(self, *args, **kwargs):
        self.store_calls.append((args, kwargs))
        return True


def _arrays(value):
    if isinstance(value, mx.array):
        yield value
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


def _assert_target_cache_equal(left, right):
    assert [type(cache) for cache in left] == [type(cache) for cache in right]
    for left_cache, right_cache in zip(left, right):
        left_arrays = list(_arrays(left_cache.state))
        right_arrays = list(_arrays(right_cache.state))
        assert len(left_arrays) == len(right_arrays)
        mx.eval(*left_arrays, *right_arrays)
        for left_value, right_value in zip(left_arrays, right_arrays):
            assert left_value.shape == right_value.shape
            assert left_value.dtype == right_value.dtype
            assert mx.array_equal(left_value, right_value).item()


def _assert_qsa_ple_cache_parity(left, right):
    """Compare every persisted PLE/QSA array after different forward shapes."""

    assert [type(cache) for cache in left] == [type(cache) for cache in right]
    compared = {"ple": 0, "qsa": 0}
    for left_cache, right_cache in zip(left, right):
        family = "qsa" if type(left_cache).__name__.startswith("QSA") else "ple"
        left_state = left_cache.state
        right_state = right_cache.state
        if family == "ple":
            # Qwen4's linear layer shares one ArraysCache: slots 0/1 are GDN;
            # slots 2/3 are exactly PLE short-conv and n-gram history.
            left_state = left_state[2:]
            right_state = right_state[2:]
        left_arrays = list(_arrays(left_state))
        right_arrays = list(_arrays(right_state))
        assert len(left_arrays) == len(right_arrays)
        mx.eval(*left_arrays, *right_arrays)
        for left_value, right_value in zip(left_arrays, right_arrays):
            compared[family] += 1
            assert left_value.shape == right_value.shape
            assert left_value.dtype == right_value.dtype
            if mx.issubdtype(left_value.dtype, mx.integer):
                assert mx.array_equal(left_value, right_value).item()
            else:
                assert mx.allclose(
                    left_value,
                    right_value,
                    rtol=1e-3,
                    atol=1e-3,
                ).item()
    assert compared["ple"] > 0
    assert compared["qsa"] >= 4


def _restore_target_prefix(model, cache, prefix):
    with prompt_priming.suppress_capture():
        output = model(prefix[None], cache=cache)
        mx.eval(output.logits)


def _greedy(logprobs):
    return mx.argmax(logprobs, axis=-1).astype(mx.uint32)


def _target_continuation(model, history, count):
    """Unprimed target oracle: next token and subsequent greedy tokens."""

    cache = model.make_cache()
    with prompt_priming.suppress_capture():
        output = model(history[None], cache=cache)
        mx.eval(output.logits)
        token = mx.argmax(output.logits[:, -1], axis=-1).astype(mx.uint32)
        next_main = int(token.item())
        tokens = []
        for _ in range(count):
            current = token
            output = model(current[:, None], cache=cache)
            token = mx.argmax(output.logits[:, -1], axis=-1).astype(mx.uint32)
            mx.eval(token)
            tokens.append(int(token.item()))
    # tokens[0] is the prediction after feeding the first next_main token.
    return next_main, tokens


def _suffix_cycle_fixture(model):
    """Build the real target and local Qwen4 head through activation."""

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    prefix = mx.array([2, 3, 4, 5, 6], dtype=mx.int32)
    suffix = mx.array([7, 8, 9], dtype=mx.int32)
    prompt = mx.concatenate([prefix, suffix])
    target_cache = model.make_cache()
    _restore_target_prefix(model, target_cache, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-real-cycle",
        prompt_tokens=prompt.tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    suffix_output = model(suffix[None], cache=target_cache)
    main_tok = mx.argmax(suffix_output.logits[:, -1], axis=-1).astype(mx.uint32)
    activation = model(main_tok[:, None], cache=target_cache, return_hidden=True)
    primed = prompt_priming.take_primed(model, target_cache, main_tok)
    assert isinstance(primed, prompt_priming.SuffixLocalPrimedState)

    state = bg._MtpState(uid=1, chain=True, depth=2, head_clone=False)
    assert bg._adopt_primed_head_state(state, primed, target_cache)
    next_main = mx.argmax(activation.logits[:, -1], axis=-1).astype(mx.uint32)
    state.next_main = next_main
    bg._chain_next_drafts(
        SimpleNamespace(
            model=model,
            logits_processors=[],
            samplers=[None],
            fallback_sampler=_greedy,
        ),
        state,
        activation.hidden_states[0][:, -1:],
        next_main,
        None,
    )
    mx.eval(state.drafts)
    history = mx.concatenate([prompt, main_tok.astype(mx.int32)])
    return target_cache, state, history, int(next_main.item())


@pytest.mark.parametrize("accepted", [0, 1, 2], ids=["reject", "partial", "full"])
def test_real_qwen4_suffix_local_verify_cycle_matches_unprimed_target(
    accepted,
    monkeypatch,
):
    """Real target verify/rollback/bonus stays identical for every accept case."""

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    model = _model()
    target_cache, state, history, next_main = _suffix_cycle_fixture(model)

    # Derive target tokens from an independent, unprimed target replay.
    model._position_ids = None
    model._rope_deltas = None
    oracle_next_main, target_tokens = _target_continuation(model, history, 4)
    assert oracle_next_main == next_main

    drafts = target_tokens[:2]
    if accepted < 2:
        drafts = list(drafts)
        drafts[accepted] = (drafts[accepted] + 1) % model.args.vocab_size
        assert drafts[accepted] != target_tokens[accepted]
    state.drafts = mx.array(drafts, dtype=mx.uint32)
    state.draft_lps = [
        mx.zeros((model.args.vocab_size,), dtype=mx.float32) for _ in drafts
    ]
    state.draft_accept_lps = list(state.draft_lps)

    # Put the correction/bonus exactly on a cache boundary so the real cycle
    # must execute _materialize_mtp_boundary_emit after rollback/full accept.
    model._omlx_mtp_commit_align = 8
    emitted = 8 - (accepted + 1)
    batch = SimpleNamespace(
        model=model,
        prompt_cache=target_cache,
        tokens=[[0] * emitted],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[],
        _token_context=[],
    )

    rollback_calls = []
    original_rollback = model.rollback_speculative_cache

    def tracked_rollback(*args, **kwargs):
        rollback_calls.append((args[2], args[3]))
        return original_rollback(*args, **kwargs)

    model.rollback_speculative_cache = tracked_rollback
    boundary_calls = []
    original_boundary = bg._materialize_mtp_boundary_emit

    def tracked_boundary(gen_batch, mtp_state):
        boundary_calls.append(True)
        return original_boundary(gen_batch, mtp_state)

    monkeypatch.setattr(bg, "_materialize_mtp_boundary_emit", tracked_boundary)
    bg._run_verify_cycle_chain(batch, state)
    mx.eval(*[row[1] for row in state.queue])

    emitted_tokens = [token for token, _lp, _source in state.queue]
    assert emitted_tokens == target_tokens[: accepted + 2]
    assert boundary_calls == [True]
    if accepted < 2:
        assert rollback_calls == [(accepted, 3)]
    else:
        assert rollback_calls == []

    # Verify kept next_main + accepted drafts, then boundary materialization
    # committed the target correction/bonus. The final queued token remains one
    # position ahead, matching the standard decoder pipeline.
    retained_inputs = mx.array(
        [next_main, *target_tokens[: accepted + 1]],
        dtype=mx.int32,
    )
    expected_history = mx.concatenate([history, retained_inputs])
    reference_cache = model.make_cache()
    model._position_ids = None
    model._rope_deltas = None
    with prompt_priming.suppress_capture():
        reference = model(expected_history[None], cache=reference_cache)
        mx.eval(reference.logits)
    _assert_qsa_ple_cache_parity(target_cache, reference_cache)
    assert state.target_expected_offset == len(expected_history)
    assert prompt_priming.target_cache_offset(target_cache) == len(expected_history)
    # Local head history advances by next_main, the verify commit, and boundary
    # materialization; it never jumps to the absolute target offset.
    assert state.hist_offset == 3 + accepted + 3
    assert state.hist_offset < state.target_expected_offset
    assert state.drafts is not None  # boundary path rebuilt the next draft chain


def test_qwen4_suffix_local_priming_preserves_target_output_and_cache():
    """The target is byte-identical; only the verified drafter gains a suffix."""

    model = _model()
    assert (
        model._omlx_mtp_suffix_local_capability
        == "qwen4-verified-text-v1"
    )
    prefix = mx.array([2, 3, 4, 5, 6, 7], dtype=mx.int32)
    suffix = mx.array([8, 9, 10], dtype=mx.int32)
    prompt = mx.concatenate([prefix, suffix])
    control_cache = model.make_cache()
    primed_cache = model.make_cache()
    _restore_target_prefix(model, control_cache, prefix)
    _restore_target_prefix(model, primed_cache, prefix)

    with prompt_priming.suppress_capture():
        control_suffix = model(suffix[None], cache=control_cache)
    prefix_cache = _NoSidecarPrefixCache()
    assert not prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-suffix",
        prompt_tokens=prompt.tolist(),
        cached_tokens=len(prefix),
        prefix_cache=prefix_cache,
    )
    plan = prompt_priming._find_plan(model)
    assert plan is not None and plan.cached_tokens == len(prefix)
    assert prompt_priming.capture_eligible(model, primed_cache)
    primed_suffix = model(suffix[None], cache=primed_cache)
    ctx = prompt_priming._find_ctx(model)
    assert ctx is not None and ctx.suffix_local
    assert ctx.head_hist_offset == len(suffix) - 1
    assert ctx.target_expected_offset == len(prompt)
    assert prompt_priming.mtp_cache_offset(ctx.mtp_cache) == len(suffix) - 1
    assert ctx.snapshot_candidate is None
    assert prefix_cache.store_calls == []

    mx.eval(control_suffix.logits, primed_suffix.logits)
    assert mx.array_equal(control_suffix.logits, primed_suffix.logits).item()
    _assert_target_cache_equal(control_cache, primed_cache)

    main_tok = mx.argmax(primed_suffix.logits[:, -1], axis=-1).astype(mx.int32)
    with prompt_priming.suppress_capture():
        control_activation = model(
            main_tok[:, None], cache=control_cache, return_hidden=True
        )
    primed_activation = model(
        main_tok[:, None], cache=primed_cache, return_hidden=True
    )
    mx.eval(control_activation.logits, primed_activation.logits)
    assert mx.array_equal(
        control_activation.logits, primed_activation.logits
    ).item()
    _assert_target_cache_equal(control_cache, primed_cache)

    primed = prompt_priming.take_primed(model, primed_cache, main_tok)
    assert isinstance(primed, prompt_priming.SuffixLocalPrimedState)
    assert primed.head_hist_offset == len(suffix)
    assert primed.target_expected_offset == len(prompt) + 1
    assert prompt_priming.mtp_cache_offset(primed.mtp_cache) == len(suffix)
    assert prefix_cache.store_calls == []
    # The seam fold touched only the drafter; target output/cache stay exact.
    _assert_target_cache_equal(control_cache, primed_cache)


def test_qwen4_cold_plan_never_captures_or_changes_forward(monkeypatch):
    """Cold Qwen4 stays on the exact priming-OFF model-forward path."""

    model = _model()
    prompt = mx.array([2, 3, 4, 5, 6, 7, 8], dtype=mx.int32)
    control_cache = model.make_cache()
    candidate_cache = model.make_cache()

    with prompt_priming.suppress_capture():
        control = model(prompt[None], cache=control_cache)

    assert not prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-cold-no-capture",
        prompt_tokens=prompt.tolist(),
        cached_tokens=0,
        prefix_cache=_NoSidecarPrefixCache(),
    )
    assert prompt_priming._find_plan(model) is None
    assert prompt_priming._find_ctx(model) is None
    assert not prompt_priming.capture_eligible(model, candidate_cache)

    def unexpected_capture(*_args, **_kwargs):
        raise AssertionError("cold Qwen4 entered prompt capture")

    monkeypatch.setattr(prompt_priming, "maybe_capture", unexpected_capture)
    candidate = model(prompt[None], cache=candidate_cache)

    mx.eval(control.logits, candidate.logits)
    assert candidate.logits.shape == control.logits.shape
    assert mx.array_equal(candidate.logits, control.logits).item()
    assert not getattr(control, "hidden_states", None)
    assert not getattr(candidate, "hidden_states", None)
    _assert_target_cache_equal(control_cache, candidate_cache)
    assert prompt_priming._find_ctx(model) is None


@pytest.mark.parametrize(
    ("extra_keys", "extra_start", "extra_ranges"),
    [
        (("image-hash",), 2, None),
        (None, None, [(2, ("video-hash",))]),
    ],
)
def test_qwen4_suffix_local_multimodal_plan_fails_closed(
    extra_keys,
    extra_start,
    extra_ranges,
):
    model = _model()
    cache = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix = mx.array([6, 7], dtype=mx.int32)
    prompt = mx.concatenate([prefix, suffix])
    _restore_target_prefix(model, cache, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-media",
        prompt_tokens=prompt.tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
        extra_keys=extra_keys,
        extra_key_token_start=extra_start,
        extra_key_ranges=extra_ranges,
    )
    output = model(suffix[None], cache=cache)
    mx.eval(output.logits)
    assert prompt_priming._find_ctx(model) is None


@pytest.mark.parametrize("fault", ["offset", "tokens"])
def test_qwen4_suffix_local_malformed_plan_fails_closed(fault):
    model = _model()
    cache = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix = mx.array([6, 7], dtype=mx.int32)
    _restore_target_prefix(model, cache, prefix)
    planned = mx.concatenate([prefix, suffix]).tolist()
    cached_tokens = len(prefix)
    if fault == "offset":
        cached_tokens += 1
    else:
        planned[-1] = 11
    prompt_priming.prepare_prefix_context(
        model,
        request_id=f"qwen4-malformed-{fault}",
        prompt_tokens=planned,
        cached_tokens=cached_tokens,
        prefix_cache=_NoSidecarPrefixCache(),
    )
    output = model(suffix[None], cache=cache)
    mx.eval(output.logits)
    assert prompt_priming._find_ctx(model) is None


def test_qwen4_suffix_local_activation_rejects_malformed_head_offset():
    model = _model()
    cache = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix = mx.array([6, 7, 8], dtype=mx.int32)
    prompt = mx.concatenate([prefix, suffix])
    _restore_target_prefix(model, cache, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-head-offset",
        prompt_tokens=prompt.tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    output = model(suffix[None], cache=cache)
    ctx = prompt_priming._find_ctx(model)
    assert ctx is not None and ctx.suffix_local
    assert ctx.mtp_cache[0].trim(1) == 1
    main_tok = mx.argmax(output.logits[:, -1], axis=-1).astype(mx.int32)
    model(main_tok[:, None], cache=cache, return_hidden=True)
    assert prompt_priming.take_primed(model, cache, main_tok) is None


def test_qwen4_suffix_local_single_token_suffix_primes_activation_seam():
    """An exact N-1 backbone hit still contributes its one uncached token."""

    model = _model()
    cache = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix = mx.array([6], dtype=mx.int32)
    prompt = mx.concatenate([prefix, suffix])
    _restore_target_prefix(model, cache, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="qwen4-one-token-suffix",
        prompt_tokens=prompt.tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    output = model(suffix[None], cache=cache)
    ctx = prompt_priming._find_ctx(model)
    assert ctx is not None and ctx.suffix_local
    assert ctx.head_hist_offset == 0
    assert prompt_priming.mtp_cache_offset(ctx.mtp_cache) == 0

    main_tok = mx.argmax(output.logits[:, -1], axis=-1).astype(mx.int32)
    model(main_tok[:, None], cache=cache, return_hidden=True)
    primed = prompt_priming.take_primed(model, cache, main_tok)
    assert isinstance(primed, prompt_priming.SuffixLocalPrimedState)
    assert primed.head_hist_offset == 1
    assert primed.target_expected_offset == len(prompt) + 1


class _OffsetCache:
    def __init__(self, offset):
        self.offset = offset

    def trim(self, amount):
        amount = min(self.offset, int(amount))
        self.offset -= amount
        return amount


def test_suffix_local_state_keeps_head_trim_local_and_target_absolute():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    state = bg._MtpState(
        mtp_cache=[_OffsetCache(7)],
        hist_offset=3,
        target_expected_offset=100,
        suffix_local_priming=True,
    )
    bg._trim_committed_mtp_head(state)
    assert state.mtp_cache[0].offset == 3
    assert state.hist_offset == 3
    assert state.target_expected_offset == 100

    target = [_OffsetCache(102)]
    bg._advance_suffix_local_target(state, target, 2)
    assert state.target_expected_offset == 102
    assert state.hist_offset == 3

    bad = SimpleNamespace(offset=104)
    with pytest.raises(bg._MtpStepFallback, match="absolute seam"):
        bg._advance_suffix_local_target(state, [bad], 1)


def test_batch_activation_adopts_only_aligned_suffix_local_state():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    primed = prompt_priming.SuffixLocalPrimedState(
        mtp_cache=[_OffsetCache(3)],
        head_hist_offset=3,
        target_expected_offset=101,
    )
    state = bg._MtpState()
    assert bg._adopt_primed_head_state(state, primed, [_OffsetCache(101)])
    assert state.mtp_cache is primed.mtp_cache
    assert state.hist_offset == 3
    assert state.target_expected_offset == 101
    assert state.suffix_local_priming

    malformed = bg._MtpState()
    assert not bg._adopt_primed_head_state(
        malformed,
        primed,
        [_OffsetCache(100)],
    )
    assert malformed.mtp_cache is None
    assert not malformed.suffix_local_priming


def test_target_cache_offset_requires_every_readable_layer_aligned():
    aligned = [
        _OffsetCache(12),
        SimpleNamespace(caches=[_OffsetCache(12), object()]),
        _OffsetCache(12),
    ]
    assert prompt_priming.target_cache_offset(aligned) == 12
    aligned[2].offset = 11
    assert prompt_priming.target_cache_offset(aligned) is None


def test_batch_activation_rejects_misaligned_multilayer_target():
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    primed = prompt_priming.SuffixLocalPrimedState(
        mtp_cache=[_OffsetCache(3)],
        head_hist_offset=3,
        target_expected_offset=101,
    )
    state = bg._MtpState()
    assert not bg._adopt_primed_head_state(
        state,
        primed,
        [_OffsetCache(101), _OffsetCache(100)],
    )
    assert state.mtp_cache is None


def test_suffix_plan_new_request_replaces_stale_owner():
    model = _model()
    cache = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix_a = mx.array([6, 7], dtype=mx.int32)
    suffix_b = mx.array([8, 9], dtype=mx.int32)
    _restore_target_prefix(model, cache, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="stale-a",
        prompt_tokens=mx.concatenate([prefix, suffix_a]).tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    prompt_priming.prepare_prefix_context(
        model,
        request_id="fresh-b",
        prompt_tokens=mx.concatenate([prefix, suffix_b]).tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    output = model(suffix_b[None], cache=cache)
    mx.eval(output.logits)
    ctx = prompt_priming._find_ctx(model)
    assert ctx is not None
    assert ctx.request_id == "fresh-b"
    assert ctx.suffix_local


def test_suffix_plan_interleaving_cannot_steal_the_new_request_plan():
    model = _model()
    cache_a = model.make_cache()
    cache_b = model.make_cache()
    prefix = mx.array([2, 3, 4, 5], dtype=mx.int32)
    suffix_a = mx.array([6, 7], dtype=mx.int32)
    suffix_b = mx.array([8, 9], dtype=mx.int32)
    _restore_target_prefix(model, cache_a, prefix)
    _restore_target_prefix(model, cache_b, prefix)
    prompt_priming.prepare_prefix_context(
        model,
        request_id="interleave-a",
        prompt_tokens=mx.concatenate([prefix, suffix_a]).tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    # B becomes the scheduler-owned plan before A's stale chunk reaches capture.
    prompt_priming.prepare_prefix_context(
        model,
        request_id="interleave-b",
        prompt_tokens=mx.concatenate([prefix, suffix_b]).tolist(),
        cached_tokens=len(prefix),
        prefix_cache=_NoSidecarPrefixCache(),
    )
    stale = model(suffix_a[None], cache=cache_a)
    mx.eval(stale.logits)
    assert prompt_priming._find_ctx(model) is None
    plan = prompt_priming._find_plan(model)
    assert plan is not None and plan.request_id == "interleave-b"
    # A's mismatched chunk cannot consume or mutate B's immutable plan. B may
    # still start only from its own matching target cache and token suffix.
    current = model(suffix_b[None], cache=cache_b)
    mx.eval(current.logits)
    ctx = prompt_priming._find_ctx(model)
    assert ctx is not None and ctx.request_id == "interleave-b"


def test_suffix_local_b1_to_batch_reconcile_uses_absolute_stream_only(
    monkeypatch,
):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    class _TargetCache:
        def __init__(self):
            self.offset = 0
            self._mtp_undo = None

    def fake_backbone(_model, inputs, cache, n_confirmed=0):
        del n_confirmed
        cache[0].offset = int(inputs.shape[1])
        cache[0]._mtp_undo = object()
        logits = mx.full((1, inputs.shape[1], 64), -10.0)
        logits[:, -1, 5] = 10.0
        return logits, None, None

    monkeypatch.setattr(bg, "_rebuild_singleton_cache", lambda _model: [_TargetCache()])
    monkeypatch.setattr(bg, "_call_backbone", fake_backbone)
    queued_lp = mx.zeros((64,))
    state = bg._MtpState(
        uid=7,
        queue=deque([(42, queued_lp, "draft")]),
        mtp_cache=[_OffsetCache(3)],
        hist_offset=3,
        target_expected_offset=100,
        suffix_local_priming=True,
    )
    batch = SimpleNamespace(
        model=object(),
        uids=[7],
        tokens=[[10, 11, 12, 13]],
        _num_tokens=[4],
        samplers=[None],
        fallback_sampler=_greedy,
        logits_processors=[],
        _next_tokens=mx.array([999], dtype=mx.uint32),
        _next_logprobs=[],
        _token_context=[],
        prompt_cache=[object()],
        _omlx_mtp_state=state,
    )
    assert bg._reconcile_mtp_to_standard(batch, state)
    assert batch._next_tokens.tolist() == [42]
    assert batch.prompt_cache[0].offset == len(batch.tokens[0])
    assert batch.prompt_cache[0]._mtp_undo is None
    # The rebuilt standard target is derived only from the absolute emitted
    # stream; neither local head offset nor stale absolute seam is copied.
    assert not hasattr(batch.prompt_cache[0], "hist_offset")
    assert not hasattr(batch.prompt_cache[0], "target_expected_offset")
