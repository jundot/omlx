# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MTP prompt priming (omlx/patches/mlx_lm_mtp/prompt_priming.py).

Uses a tiny random-weight qwen3_5 TextModel (mlx-lm path) so the capture hook
in the patched ``TextModel.__call__`` and the activation handoff in
``_post_init_mtp`` run for real. The mlx-vlm capture site shares
``maybe_capture`` / ``take_primed``, so the fold math is covered here; its
wiring is exercised by the real-model smoke test.
"""

from types import SimpleNamespace

import pytest

mx = pytest.importorskip("mlx.core")

from omlx.patches.mlx_lm_mtp import prompt_priming


TINY_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "vocab_size": 256,
    "linear_num_value_heads": 2,
    "linear_num_key_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3,
    "full_attention_interval": 2,
    "tie_word_embeddings": True,
    "rms_norm_eps": 1e-5,
    "head_dim": 32,
    "rope_theta": 1000.0,
    "partial_rotary_factor": 0.5,
    "max_position_embeddings": 128,
    "mtp_num_hidden_layers": 1,
}


def _make_tiny_model():
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    args = TextModelArgs.from_dict(TINY_CONFIG)
    model = TextModel(args)
    mx.eval(model.parameters())
    return model


def _make_cache(model):
    from mlx_lm.models.cache import make_prompt_cache

    return make_prompt_cache(model)


def _tokens(n, seed=0):
    mx.random.seed(seed)
    return mx.random.randint(0, TINY_CONFIG["vocab_size"], (n,)).astype(mx.uint32)


def _kv_entries(mtp_cache):
    out = []
    for c in mtp_cache:
        keys, values = c.state
        out.append((keys, values))
    return out


def _reference_head_cache(model, tokens, extra_tok=None):
    """Fold the whole prompt through the head in one shot (oracle)."""
    fresh = _make_cache(model)
    logits, hidden = model(tokens[None, :], cache=fresh, return_hidden=True)
    normed = model.model.norm(hidden)
    ref_cache = model.make_mtp_cache()
    pair_tokens = tokens[1:]
    pair_hidden = normed[:, :-1]
    if extra_tok is not None:
        pair_tokens = mx.concatenate([pair_tokens, extra_tok])
        pair_hidden = normed
    model.mtp(
        pair_hidden,
        pair_tokens[None, :].astype(mx.uint32),
        model.model.embed_tokens,
        ref_cache,
    )
    mx.eval([c.state for c in ref_cache])
    return ref_cache


@pytest.fixture(autouse=True)
def _apply_patch():
    try:
        from omlx.patches.mlx_lm_mtp import qwen35_model, set_mtp_active
    except ImportError:
        pytest.skip("omlx.patches.mlx_lm_mtp not importable")
    if not qwen35_model.apply():
        pytest.skip("qwen35_model patch refused to apply")
    prev = None
    from omlx.patches.mlx_lm_mtp import is_mtp_active

    prev = is_mtp_active()
    set_mtp_active(True)
    yield
    set_mtp_active(prev)


@pytest.fixture()
def model():
    return _make_tiny_model()


def _chunked_prefill(model, cache, tokens, chunks):
    """Drive the patched TextModel.__call__ chunk by chunk (capture rides it)."""
    start = 0
    for size in chunks:
        chunk = tokens[start : start + size]
        model(chunk[None, :], cache=cache)
        start += size
    assert start == tokens.shape[0]


class TestCaptureFold:
    def test_chunked_capture_matches_oneshot_fold(self, model):
        n = 13
        tokens = _tokens(n)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [5, 5, 3])

        folded = prompt_priming.prime_ctx_stats(cache)
        assert folded == n - 1

        ctx = prompt_priming._find_ctx(cache)
        assert ctx is not None and ctx.valid
        assert ctx.mtp_cache[0].offset == n - 1
        mx.eval([c.state for c in ctx.mtp_cache])

        ref_cache = _reference_head_cache(model, tokens)
        for (k, v), (rk, rv) in zip(
            _kv_entries(ctx.mtp_cache), _kv_entries(ref_cache)
        ):
            assert mx.allclose(k, rk, rtol=1e-4, atol=1e-4)
            assert mx.allclose(v, rv, rtol=1e-4, atol=1e-4)

    def test_chunk_size_one_seam_is_dense(self, model):
        """A trailing S==1 forward (the __init__ _step seam) still folds."""
        n = 8
        tokens = _tokens(n, seed=1)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [4, 3, 1])
        assert prompt_priming.prime_ctx_stats(cache) == n - 1

    def test_take_primed_completes_seam(self, model):
        n = 9
        tokens = _tokens(n, seed=2)
        main_tok = _tokens(1, seed=3)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [6, 3])
        # Activation forward runs with return_hidden=True: capture skips it.
        model(main_tok[None, :], cache=cache, return_hidden=True)

        primed = prompt_priming.take_primed(model, cache, main_tok)
        assert primed is not None
        mtp_cache, hist_offset = primed
        assert hist_offset == n
        assert mtp_cache[0].offset == n
        assert prompt_priming._find_ctx(cache) is None

        ref_cache = _reference_head_cache(model, tokens, extra_tok=main_tok)
        mx.eval([c.state for c in mtp_cache])
        for (k, v), (rk, rv) in zip(_kv_entries(mtp_cache), _kv_entries(ref_cache)):
            assert mx.allclose(k, rk, rtol=1e-4, atol=1e-4)
            assert mx.allclose(v, rv, rtol=1e-4, atol=1e-4)


class TestCaptureSkips:
    def test_env_off_disables_capture(self, model, monkeypatch):
        monkeypatch.setenv("OMLX_MTP_PROMPT_PRIMING", "0")
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(6), [6])
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_suppress_capture(self, model):
        cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            _chunked_prefill(model, cache, _tokens(6), [6])
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_single_token_forward_does_not_start_ctx(self, model):
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(1), [1])
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_return_hidden_forward_skipped(self, model):
        cache = _make_cache(model)
        model(_tokens(6)[None, :], cache=cache, return_hidden=True)
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_batch_forward_skipped(self, model):
        cache = _make_cache(model)
        toks = _tokens(12).reshape(2, 6)
        # Batched cache shapes differ; just assert no ctx is created.
        try:
            model(toks, cache=cache)
        except Exception:
            pass
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_offset_rewind_invalidates_and_restarts(self, model):
        tokens = _tokens(12, seed=4)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens[:8], [8])
        assert prompt_priming.prime_ctx_stats(cache) == 7
        # External trim breaks contiguity: the old timeline must not survive.
        for c in cache:
            if hasattr(c, "trim") and type(getattr(c, "offset", None)) is int:
                c.trim(2)
        _chunked_prefill(model, cache, tokens[8:], [4])
        # Restarted mid-prompt: only the new chunk's internal pairs.
        assert prompt_priming.prime_ctx_stats(cache) == 3

    def test_window_cap_disables_long_prompts(self, model, monkeypatch):
        monkeypatch.setenv("OMLX_MTP_PRIME_WINDOW", "4")
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(10, seed=5), [5, 5])
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_take_primed_requires_seam_offset(self, model):
        """No activation forward ran: seam mismatch must discard the ctx."""
        tokens = _tokens(7, seed=6)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [7])
        assert prompt_priming.take_primed(model, cache, _tokens(1, seed=7)) is None
        assert prompt_priming._find_ctx(cache) is None

    def test_ctx_attaches_to_non_kv_entry(self, model):
        """TurboQuant replaces int-offset KVCache entries at end of prefill;
        the ctx must ride an entry conversion never touches (issue found in
        the first real-server smoke: primed=0 with turboquant_kv on)."""
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(6, seed=20), [6])
        target = None
        for c in cache:
            if getattr(c, "_omlx_mtp_prime_ctx", None) is not None:
                target = c
        assert target is not None
        assert type(getattr(target, "offset", None)) is not int

    def test_ctx_survives_kv_entry_replacement(self, model):
        """Simulate the TurboQuant convert: swap every KVCache entry for a
        fresh object carrying the same state, then finish activation."""
        from mlx_lm.models.cache import KVCache

        n = 9
        tokens = _tokens(n, seed=21)
        main_tok = _tokens(1, seed=22)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [6, 3])
        for i, c in enumerate(cache):
            if isinstance(c, KVCache):
                clone = KVCache()
                clone.keys, clone.values, clone.offset = c.keys, c.values, c.offset
                cache[i] = clone
        model(main_tok[None, :], cache=cache, return_hidden=True)
        primed = prompt_priming.take_primed(model, cache, main_tok)
        assert primed is not None
        assert primed[1] == n

    def test_drop_ctx(self, model):
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(6, seed=8), [6])
        assert prompt_priming.prime_ctx_stats(cache) is not None
        prompt_priming.drop_ctx(cache)
        assert prompt_priming.prime_ctx_stats(cache) is None

    def test_deepcopy_clone_is_invalid(self, model):
        import copy

        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(6, seed=9), [6])
        ctx = prompt_priming._find_ctx(cache)
        clone = copy.deepcopy(ctx)
        assert clone.valid is False
        assert clone.folded == 0


class TestActivationHandoff:
    def _gen_batch(self, model, cache, tokens):
        def greedy(lp):
            return mx.argmax(lp, axis=-1).astype(mx.uint32)

        return SimpleNamespace(
            model=model,
            prompt_cache=cache,
            uids=[0],
            samplers=[None],
            fallback_sampler=greedy,
            logits_processors=[],
            tokens=[list(int(t) for t in tokens.tolist())],
            _next_tokens=None,
            _next_logprobs=None,
            _token_context=[None],
        )

    def test_post_init_uses_primed_cache(self, model):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        n = 10
        tokens = _tokens(n, seed=10)
        cache = _make_cache(model)
        # Standard __init__ semantics: prefill everything, then _step on the
        # last token sampled main_tok. Emulate with a chunked prefill over
        # tokens[:-1] plus an S==1 step on tokens[-1].
        _chunked_prefill(model, cache, tokens[:-1], [6, 3])
        logits = model(tokens[-1:][None, :], cache=cache)
        lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
        main_tok = mx.argmax(lp, keepdims=True).astype(mx.uint32)

        gen_batch = self._gen_batch(model, cache, tokens)
        gen_batch._next_tokens = main_tok
        gen_batch._next_logprobs = [lp]

        bg._post_init_mtp(gen_batch)
        state = getattr(gen_batch, "_omlx_mtp_state", None)
        assert state is not None
        # n prompt-pair folds via capture+seam, +1 from _chain_next_drafts.
        assert state.hist_offset == n + 1
        assert state.mtp_cache[0].offset >= n
        assert prompt_priming._find_ctx(cache) is None

    def test_post_init_without_ctx_is_unprimed(self, model):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        n = 10
        tokens = _tokens(n, seed=11)
        cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            _chunked_prefill(model, cache, tokens[:-1], [9])
            logits = model(tokens[-1:][None, :], cache=cache)
        lp = logits[0, -1] - mx.logsumexp(logits[0, -1])
        main_tok = mx.argmax(lp, keepdims=True).astype(mx.uint32)

        gen_batch = self._gen_batch(model, cache, tokens)
        gen_batch._next_tokens = main_tok
        gen_batch._next_logprobs = [lp]

        bg._post_init_mtp(gen_batch)
        state = getattr(gen_batch, "_omlx_mtp_state", None)
        assert state is not None
        assert state.hist_offset == 1
