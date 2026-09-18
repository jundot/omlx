# SPDX-License-Identifier: Apache-2.0
"""Unit tests for MTP prompt priming (omlx/patches/mlx_lm_mtp/prompt_priming.py).

Uses a tiny random-weight qwen3_5 TextModel (mlx-lm path) so the capture hook
in the patched ``TextModel.__call__`` and the activation handoff in
``_post_init_mtp`` run for real. The mlx-vlm capture site shares
``maybe_capture`` / ``take_primed``, so the fold math is covered here; its
wiring is exercised by the real-model smoke test.
"""

import threading
from collections import OrderedDict
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


def _make_batch_cache(model):
    """The cache shape every request takes through ``BatchGenerator``.

    ``PromptProcessingBatch.__init__`` merges the per-request caches
    (mlx-lm ``_merge_caches``), so even a single request runs on ``Batch*``
    entries whose ``offset`` is a 1-element ``mx.array`` rather than an int.
    """
    from mlx_lm.models.cache import ArraysCache, BatchKVCache, KVCache

    batched = []
    for c in _make_cache(model):
        if isinstance(c, KVCache):
            batched.append(BatchKVCache.merge([c]))
        else:
            if isinstance(c, ArraysCache):
                c.left_padding = mx.array([0])
            batched.append(c)
    return batched


def _tokens(n, seed=0):
    mx.random.seed(seed)
    return mx.random.randint(0, TINY_CONFIG["vocab_size"], (n,)).astype(mx.uint32)


def _kv_entries(mtp_cache):
    out = []
    for c in mtp_cache:
        keys, values = c.keys_and_values()
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


class _MemoryMtpPrefixCache:
    """Minimal scheduler/cache contract for prompt-history integration tests."""

    def __init__(self, block_size=8):
        self.block_size = block_size
        self.snapshots = {}

    def _key(self, tokens, boundary):
        return tuple(tokens[:boundary]), int(boundary)

    def store_mtp_prefix_snapshot(self, tokens, boundary, snapshot, **kwargs):
        self.snapshots[self._key(tokens, boundary)] = snapshot
        return True

    def restore_mtp_prefix_snapshot(self, tokens, boundary, **kwargs):
        return self.snapshots.get(self._key(tokens, boundary))


def test_block_prefix_cache_mtp_sidecar_uses_live_chain_hash_and_evicts():
    """The production sidecar is only visible while its backbone tip lives."""
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    class _HashMap:
        def __init__(self):
            self.blocks = {}

        def get_block(self, key):
            return self.blocks.get(key)

    hash_map = _HashMap()
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 4
    cache.paged_cache = SimpleNamespace(
        model_name="tiny-mtp-test",
        cached_block_hash_to_block=hash_map,
    )
    cache._prefix_index = {}
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()

    tokens = list(range(8))
    snapshot = object()
    assert cache.store_mtp_prefix_snapshot(tokens, 8, snapshot)
    tip = cache._mtp_prefix_chain_tip(tokens, 8)
    assert tip is not None
    # Publishing precedes the async backbone store, so the snapshot must not
    # become restorable until the matching ordinary block is live.
    assert cache.restore_mtp_prefix_snapshot(tokens, 8) is None
    hash_map.blocks[tip] = object()
    assert cache.restore_mtp_prefix_snapshot(tokens, 8) is snapshot

    cache._on_block_hash_dropped(tip)
    assert cache.restore_mtp_prefix_snapshot(tokens, 8) is None


def test_block_prefix_cache_mtp_sidecar_lru_four_and_clear_lifecycle():
    """MTP sidecars remain bounded and follow wholesale cache clears."""
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    class _HashMap:
        def __init__(self):
            self.blocks = {}

        def get_block(self, key):
            return self.blocks.get(key)

    hash_map = _HashMap()
    cache = BlockAwarePrefixCache.__new__(BlockAwarePrefixCache)
    cache.block_size = 4
    cache.paged_cache = SimpleNamespace(
        model_name="tiny-mtp-lru-test",
        cached_block_hash_to_block=hash_map,
    )
    cache._prefix_index = {}
    cache._mtp_prefix_snapshots = OrderedDict()
    cache._mtp_prefix_snapshot_lock = threading.RLock()

    entries = []
    for branch in range(5):
        tokens = [branch * 100 + i for i in range(8)]
        snapshot = object()
        assert cache.store_mtp_prefix_snapshot(tokens, 8, snapshot)
        tip = cache._mtp_prefix_chain_tip(tokens, 8)
        assert tip is not None
        hash_map.blocks[tip] = object()
        entries.append((tokens, tip, snapshot))

    assert len(cache._mtp_prefix_snapshots) == 4
    assert cache.restore_mtp_prefix_snapshot(entries[0][0], 8) is None
    assert cache.restore_mtp_prefix_snapshot(entries[-1][0], 8) is entries[-1][2]

    # Individual backbone eviction removes the matching sidecar even if a
    # stale test hash-map entry remains, then the wholesale clear drops all
    # remaining sidecars and the ordinary prefix index together.
    cache._on_block_hash_dropped(entries[-1][1])
    assert cache.restore_mtp_prefix_snapshot(entries[-1][0], 8) is None
    cache._prefix_index[b"ordinary"] = (1, 2, 3)
    cache._on_hash_map_cleared()
    assert not cache._mtp_prefix_snapshots
    assert not cache._prefix_index


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


@pytest.fixture()
def strict_model():
    """Build and run the strict chunk/seam oracle with CPU reductions."""
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield _make_tiny_model()
    finally:
        mx.set_default_device(previous)


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

        folded = prompt_priming.prime_ctx_stats(model)
        assert folded == n - 1

        ctx = prompt_priming._find_ctx(model)
        assert ctx is not None and ctx.valid
        assert ctx.mtp_cache[0].offset == n - 1
        mx.eval([c.state for c in ctx.mtp_cache])

        ref_cache = _reference_head_cache(model, tokens)
        for (k, v), (rk, rv) in zip(
            _kv_entries(ctx.mtp_cache), _kv_entries(ref_cache)
        ):
            assert mx.allclose(k, rk, rtol=1e-4, atol=1e-4)
            assert mx.allclose(v, rv, rtol=1e-4, atol=1e-4)

    def test_warm_prefix_restores_exact_head_history_without_trunk_reforward(
        self, strict_model
    ):
        """A backbone hit at C restores MTP(C-1)+hidden(C-1), then folds
        only the uncached suffix and activation seam.  The resulting head
        cache must equal a one-shot cold oracle exactly within model dtype.
        Repeating scheduler preparation for the same request is idempotent.
        """
        model = strict_model
        tokens = _tokens(13, seed=40)
        main_tok = _tokens(1, seed=41)
        sidecar = _MemoryMtpPrefixCache(block_size=8)

        cold_cache = _make_cache(model)
        assert not prompt_priming.prepare_prefix_context(
            model,
            request_id="cold",
            prompt_tokens=tokens.tolist(),
            cached_tokens=0,
            prefix_cache=sidecar,
        )
        _chunked_prefill(model, cold_cache, tokens, [8, 5])
        cold_ctx = prompt_priming._find_ctx(model)
        assert cold_ctx is not None
        cold_final_pending = cold_ctx.pending_hidden + 0
        mx.eval(cold_final_pending)
        model(main_tok[None, :], cache=cold_cache, return_hidden=True)
        cold_primed = prompt_priming.take_primed(model, cold_cache, main_tok)
        assert cold_primed is not None
        snapshot_key = (tuple(tokens[:8].tolist()), 8)
        assert snapshot_key in sidecar.snapshots
        boundary_pending = sidecar.snapshots[snapshot_key].pending_hidden

        # Build the already-restored backbone cache outside capture.  Start
        # tracing only after sidecar restore: a correct warm path invokes the
        # MTP head for suffix(5)+seam(1), never for the cached trunk(8).
        warm_cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            model(tokens[:8][None, :], cache=warm_cache)
        assert prompt_priming.prepare_prefix_context(
            model,
            request_id="warm",
            prompt_tokens=tokens.tolist(),
            cached_tokens=8,
            prefix_cache=sidecar,
        )
        warm_ctx = prompt_priming._find_ctx(model)
        assert warm_ctx is not None
        assert warm_ctx.folded == 7
        assert warm_ctx.expected_offset == 8
        assert mx.array_equal(warm_ctx.pending_hidden, boundary_pending).item()

        # The scheduler's prepared-set normally prevents this second call;
        # the hook itself also guarantees it cannot reset/double-prime a live
        # request if invoked twice.
        assert prompt_priming.prepare_prefix_context(
            model,
            request_id="warm",
            prompt_tokens=tokens.tolist(),
            cached_tokens=8,
            prefix_cache=sidecar,
        )
        assert prompt_priming._find_ctx(model) is warm_ctx

        mtp_rows = []
        original_mtp_forward = model.mtp_forward

        def traced_mtp_forward(hidden, next_ids, cache, **kwargs):
            mtp_rows.append(int(next_ids.shape[1]))
            return original_mtp_forward(hidden, next_ids, cache, **kwargs)

        model.mtp_forward = traced_mtp_forward
        _chunked_prefill(model, warm_cache, tokens[8:], [5])
        warm_final_ctx = prompt_priming._find_ctx(model)
        assert warm_final_ctx is not None
        assert mx.array_equal(
            warm_final_ctx.pending_hidden, cold_final_pending
        ).item()
        model(main_tok[None, :], cache=warm_cache, return_hidden=True)
        warm_primed = prompt_priming.take_primed(model, warm_cache, main_tok)
        assert warm_primed is not None
        assert warm_primed[1] == len(tokens)
        assert mtp_rows == [5, 1]

        mx.eval(
            [c.state for c in cold_primed[0]],
            [c.state for c in warm_primed[0]],
        )
        for (k, v), (rk, rv) in zip(
            _kv_entries(warm_primed[0]), _kv_entries(cold_primed[0])
        ):
            assert mx.array_equal(k, rk).item()
            assert mx.array_equal(v, rv).item()

    def test_chunk_size_one_seam_is_dense(self, model):
        """A trailing S==1 forward (the __init__ _step seam) still folds."""
        n = 8
        tokens = _tokens(n, seed=1)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [4, 3, 1])
        assert prompt_priming.prime_ctx_stats(model) == n - 1

    def test_take_primed_completes_seam(self, strict_model):
        model = strict_model
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
        assert prompt_priming._find_ctx(model) is None

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
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_suppress_capture(self, model):
        cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            _chunked_prefill(model, cache, _tokens(6), [6])
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_single_token_forward_does_not_start_ctx(self, model):
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(1), [1])
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_return_hidden_forward_skipped(self, model):
        cache = _make_cache(model)
        model(_tokens(6)[None, :], cache=cache, return_hidden=True)
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_batch_forward_skipped(self, model):
        cache = _make_cache(model)
        toks = _tokens(12).reshape(2, 6)
        # Batched cache shapes differ; just assert no ctx is created.
        try:
            model(toks, cache=cache)
        except Exception:
            pass
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_batch_forward_drops_pending_ctx(self, model):
        """A B>1 forward advances the anchor without capture seeing its
        tokens, so a later singleton chunk could read as contiguous across
        it. The pending timeline must not survive one."""
        tokens = _tokens(12, seed=31)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens[:6], [6])
        assert prompt_priming.prime_ctx_stats(model) == 5
        prompt_priming.maybe_capture(
            model,
            mx.zeros((2, 3), dtype=mx.uint32),
            mx.zeros((2, 3, TINY_CONFIG["hidden_size"])),
            cache,
        )
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_offset_rewind_invalidates_and_restarts(self, model):
        tokens = _tokens(12, seed=4)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens[:8], [8])
        assert prompt_priming.prime_ctx_stats(model) == 7
        # External trim breaks contiguity: the old timeline must not survive.
        for c in cache:
            if hasattr(c, "trim") and type(getattr(c, "offset", None)) is int:
                c.trim(2)
        _chunked_prefill(model, cache, tokens[8:], [4])
        # Restarted mid-prompt: only the new chunk's internal pairs.
        assert prompt_priming.prime_ctx_stats(model) == 3

    def test_window_cap_disables_long_prompts(self, model, monkeypatch):
        monkeypatch.setenv("OMLX_MTP_PRIME_WINDOW", "4")
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(10, seed=5), [5, 5])
        assert prompt_priming.prime_ctx_stats(model) is None

    def test_window_caps_folded_span_not_absolute_offset(self, model, monkeypatch):
        """A warm prefix cache leaves only a small remainder to fold; the
        window must cap that folded span (the head-KV it exists to bound),
        not the absolute prompt offset — otherwise every long-context
        warm-cache request runs unprimed even when the remainder is tiny
        (#2909)."""
        monkeypatch.setenv("OMLX_MTP_PRIME_WINDOW", "6")
        tokens = _tokens(12, seed=8)
        cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            _chunked_prefill(model, cache, tokens[:8], [8])
        assert prompt_priming.prime_ctx_stats(model) is None
        # Remainder of 4 tokens at absolute offset 12: over the old
        # absolute-offset guard (12 > 6), within the span guard (4 <= 6).
        _chunked_prefill(model, cache, tokens[8:], [4])
        assert prompt_priming.prime_ctx_stats(model) == 3

    def test_window_overflow_stays_latched_across_small_chunks(
        self, model, monkeypatch
    ):
        """An oversized multi-chunk remainder must not restart priming after
        the first context is dropped."""
        monkeypatch.setenv("OMLX_MTP_PRIME_WINDOW", "4")
        tokens = _tokens(17, seed=9)
        cache = _make_cache(model)
        with prompt_priming.suppress_capture():
            _chunked_prefill(model, cache, tokens[:8], [8])
        _chunked_prefill(model, cache, tokens[8:], [3, 3, 3])
        assert prompt_priming.prime_ctx_stats(model) is None
        ctx = prompt_priming._find_ctx(model)
        assert ctx is not None and ctx.window_exceeded
        assert ctx.expected_offset == 17

    def test_take_primed_requires_seam_offset(self, model):
        """No activation forward ran: seam mismatch must discard the ctx."""
        tokens = _tokens(7, seed=6)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [7])
        assert prompt_priming.take_primed(model, cache, _tokens(1, seed=7)) is None
        assert prompt_priming._find_ctx(model) is None

    def test_ctx_lives_on_host_not_cache(self, model):
        """The slot rides the model instance: cache entries are rebuilt by
        the insert merge (and TurboQuant conversion) on several families, so
        cache-attribute transport silently loses the context (found in the
        first real-server smokes: primed=0 with turboquant_kv / DeepSeek)."""
        cache = _make_cache(model)
        _chunked_prefill(model, cache, _tokens(6, seed=20), [6])
        assert getattr(model, "_omlx_mtp_prime_ctx", None) is not None
        assert all(
            getattr(c, "_omlx_mtp_prime_ctx", None) is None for c in cache
        )

    def test_interleaved_request_restarts_slot(self, model):
        """A second request's prefill on the same model can never continue
        the first request's timeline: its offsets restart at zero, which
        breaks contiguity and restarts the slot."""
        cache_a = _make_cache(model)
        _chunked_prefill(model, cache_a, _tokens(10, seed=23), [10])
        assert prompt_priming.prime_ctx_stats(model) == 9
        cache_b = _make_cache(model)
        _chunked_prefill(model, cache_b, _tokens(6, seed=24), [6])
        assert prompt_priming.prime_ctx_stats(model) == 5
        # Request A activating now must not see B's history.
        model(_tokens(1, seed=25)[None, :], cache=cache_a, return_hidden=True)
        assert prompt_priming.take_primed(model, cache_a, _tokens(1, seed=25)) is None

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
        assert prompt_priming.prime_ctx_stats(model) is not None
        prompt_priming.drop_ctx(model)
        assert prompt_priming.prime_ctx_stats(model) is None


class TestBatchCacheAnchor:
    """Batch caches expose ``offset`` as a 1-element array even at B==1.

    The anchor probe used to require a plain int, so it found no anchor on
    any ``BatchGenerator`` prefill and capture bailed silently — priming
    never activated in the batch engine (#3079).
    """

    def test_anchor_unwraps_size_one_array_offset(self):
        from mlx_lm.models.cache import BatchKVCache

        entry = BatchKVCache([0])
        assert type(entry.offset) is not int
        anchor = prompt_priming._anchor([entry])
        assert anchor is not None
        assert anchor.offset == 0

    def test_anchor_finds_batch_sub_cache_in_container(self):
        """DeepSeek-V4 / GLM-5.2 wrap their layer caches in a CacheList."""
        from mlx_lm.models.cache import BatchKVCache, CacheList

        anchor = prompt_priming._anchor([CacheList(BatchKVCache([0]))])
        assert anchor is not None
        assert anchor.offset == 0

    def test_anchor_skips_multi_row_batch_offset(self):
        """A real B>1 cache has a vector offset: no singleton timeline to
        anchor on, so capture must find nothing rather than guess a row."""
        from mlx_lm.models.cache import BatchKVCache

        assert prompt_priming._anchor([BatchKVCache([0, 0])]) is None

    def test_anchor_view_tracks_the_live_offset(self):
        from mlx_lm.models.cache import BatchKVCache

        entry = BatchKVCache([0])
        anchor = prompt_priming._anchor([entry])
        entry.offset = entry.offset + 7
        assert anchor.offset == 7

    def test_batch_cache_prefill_primes_end_to_end(self, strict_model):
        """Legacy single-head activation over the batch-engine cache shape:
        capture through the seam, matching the one-shot oracle fold."""
        model = strict_model
        n = 9
        tokens = _tokens(n, seed=32)
        main_tok = _tokens(1, seed=33)
        cache = _make_batch_cache(model)
        _chunked_prefill(model, cache, tokens, [6, 3])
        assert prompt_priming.prime_ctx_stats(model) == n - 1

        model(main_tok[None, :], cache=cache, return_hidden=True)
        primed = prompt_priming.take_primed(model, cache, main_tok)
        assert primed is not None
        mtp_cache, hist_offset = primed
        assert hist_offset == n
        assert mtp_cache[0].offset == n
        assert prompt_priming._find_ctx(model) is None

        ref_cache = _reference_head_cache(model, tokens, extra_tok=main_tok)
        mx.eval([c.state for c in mtp_cache])
        for (k, v), (rk, rv) in zip(_kv_entries(mtp_cache), _kv_entries(ref_cache)):
            assert mx.allclose(k, rk, rtol=1e-4, atol=1e-4)
            assert mx.allclose(v, rv, rtol=1e-4, atol=1e-4)


class TestHookFallthrough:
    """``mtp_take_primed`` is registered on the class but answered by only
    some builds: the DeepSeek-V4 patch registers it unconditionally and
    returns None for everything that is not DSpark. Taking that None as the
    final answer made the generic seam unreachable, so priming was
    structurally dead for legacy single-head MTP models (#3079).
    """

    def _prefill_and_activate(self, model, n=9, seed=34):
        tokens = _tokens(n, seed=seed)
        main_tok = _tokens(1, seed=seed + 1)
        cache = _make_cache(model)
        _chunked_prefill(model, cache, tokens, [6, 3])
        assert prompt_priming.prime_ctx_stats(model) == n - 1
        # Activation forward runs with return_hidden=True: capture skips it.
        model(main_tok[None, :], cache=cache, return_hidden=True)
        return cache, main_tok, n

    def _register_hook(self, model, monkeypatch, hook):
        monkeypatch.setattr(
            type(model), "mtp_take_primed", hook, raising=False
        )

    def test_declining_hook_falls_through_to_generic_seam(
        self, model, monkeypatch
    ):
        self._register_hook(model, monkeypatch, lambda self, cache, tok: None)
        cache, main_tok, n = self._prefill_and_activate(model)
        primed = prompt_priming.take_primed(model, cache, main_tok)
        assert primed is not None
        assert primed[1] == n
        assert prompt_priming._find_ctx(model) is None

    def test_owning_hook_result_is_returned(self, model, monkeypatch):
        """A hook that answers owns the whole seam: its result passes
        through and the generic context is left for it to manage."""
        sentinel = (["head-cache"], 123)
        self._register_hook(
            model, monkeypatch, lambda self, cache, tok: sentinel
        )
        cache, main_tok, _ = self._prefill_and_activate(model)
        assert prompt_priming.take_primed(model, cache, main_tok) is sentinel
        assert prompt_priming._find_ctx(model) is not None

    def test_fallthrough_ignores_foreign_ctx(self, model, monkeypatch):
        """Hosts that share the slot (inkling's sliding-window context) pop
        it before declining. If one ever forgets, the generic seam must not
        adopt a context it did not build."""

        class _ForeignCtx:
            pass

        self._register_hook(model, monkeypatch, lambda self, cache, tok: None)
        cache, main_tok, _ = self._prefill_and_activate(model)
        setattr(model, prompt_priming._CTX_ATTR, _ForeignCtx())
        assert prompt_priming.take_primed(model, cache, main_tok) is None


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
        assert prompt_priming._find_ctx(model) is None

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


def test_owned_batch_calibration_history_matches_full_prompt_fold(strict_model):
    """Five ordinary batch steps retain each UID's complete head history."""
    from mlx_lm.generate import _merge_caches

    model = strict_model
    histories = [_tokens(6, seed=81), _tokens(6, seed=82)]
    caches = []
    for uid, tokens in zip([11, 12], histories):
        prompt_priming.prepare_prefix_context(
            model,
            request_id=str(uid),
            prompt_tokens=tokens.tolist(),
            cached_tokens=0,
            prefix_cache=None,
        )
        cache = _make_cache(model)
        model(tokens[None, :], cache=cache)
        prompt_priming.bind_uid(model, str(uid), uid)
        caches.append(cache)
    merged = _merge_caches(caches[::-1])
    with prompt_priming.decode_scope(model, [12, 11]):
        for step in range(5):
            next_tokens = mx.array([[40 + step], [30 + step]])
            mx.eval(model(next_tokens, cache=merged))
            histories[0] = mx.concatenate([histories[0], next_tokens[1]])
            histories[1] = mx.concatenate([histories[1], next_tokens[0]])
    owned = prompt_priming._owned(model)[1]
    for uid, tokens in zip([11, 12], histories):
        ctx = owned.uids[uid][0]
        assert ctx.expected_offset == 11
        assert ctx.folded == 10
        ref = _reference_head_cache(model, tokens)
        for (keys, values), (ref_keys, ref_values) in zip(
            _kv_entries(ctx.mtp_cache), _kv_entries(ref)
        ):
            assert mx.allclose(keys, ref_keys, rtol=1e-4, atol=1e-4)
            assert mx.allclose(values, ref_values, rtol=1e-4, atol=1e-4)


class RecordingHead:
    _omlx_mtp_decode_enabled = True
    _omlx_mtp_chain = True
    mtp = object()

    def mtp_forward(self, hidden, tokens, cache, logits_keep=1):
        cache[0].pairs.extend(zip(hidden[0, :, 0].tolist(), tokens[0].tolist()))


def context(offset):
    ctx = prompt_priming._PrimeCtx(
        mtp_cache=[SimpleNamespace(pairs=[])], folded=offset, expected_offset=offset
    )
    ctx.deferred_pairs = []
    return ctx


@pytest.mark.parametrize("chunks", [(1, 1, 1, 1), (1, 3), (4,)])
@pytest.mark.parametrize("replay_chunk", [1, 2, 512])
def test_deferred_pairs_and_activation_seam(chunks, replay_chunk):
    host, ctx = RecordingHead(), context(10)
    setattr(host, prompt_priming._CTX_ATTR, ctx)
    position = 10
    for count in chunks:
        tokens = mx.arange(position, position + count)[None]
        hidden = (tokens * 10)[..., None]
        position += count
        prompt_priming._capture_single(
            host, tokens, hidden, [SimpleNamespace(offset=position)]
        )
        assert not ctx.mtp_cache[0].pairs
    prompt_priming._flush_deferred_history(host, ctx, chunk_size=replay_chunk)
    assert ctx.mtp_cache[0].pairs == [(100, 11), (110, 12), (120, 13)]
    assert ctx.folded == 13
    result = prompt_priming.take_primed(host, [], mx.array(14), cache_offset=15)
    assert result[1] == 14
    assert result[0][0].pairs == [(100, 11), (110, 12), (120, 13), (130, 14)]
    assert prompt_priming._find_ctx(host) is None


@pytest.mark.parametrize("actual_offset", [9, 10, 12])
def test_discontinuous_capture_rejected_without_mutation(actual_offset):
    host, ctx = RecordingHead(), context(10)
    setattr(host, prompt_priming._CTX_ATTR, ctx)
    with pytest.raises(RuntimeError, match="timeline"):
        prompt_priming._capture_single(
            host,
            mx.array([[10]]),
            mx.array([[[100]]]),
            [SimpleNamespace(offset=actual_offset)],
        )
    assert ctx.expected_offset == 10
    assert ctx.pending_hidden is None
    assert ctx.deferred_pairs == []


def test_uid_reorder_cancel_and_reactivation():
    host = RecordingHead()
    _, owned = prompt_priming._owned(host, create=True)
    first, second = context(10), context(30)
    owned.uids.update({3: (first, None), 7: (second, None)})
    with prompt_priming.decode_scope(host, [7, 3]):
        prompt_priming.maybe_capture(
            host,
            mx.array([[30], [10]]),
            mx.array([[[300]], [[100]]]),
            [SimpleNamespace(offset=mx.array([31, 11]))],
        )
        prompt_priming.maybe_capture(
            host,
            mx.array([[31], [11]]),
            mx.array([[[310]], [[110]]]),
            [SimpleNamespace(offset=mx.array([32, 12]))],
        )
    prompt_priming.release_uids(host, [7])
    assert 7 not in owned.uids
    assert prompt_priming._find_ctx(host) is None
    result = prompt_priming.take_primed(host, [], mx.array(12), uid=3, cache_offset=13)
    assert result[0][0].pairs == [(100, 11), (110, 12)]
    assert owned.uids == {}
    assert second.mtp_cache[0].pairs == []


def test_bad_activation_does_not_run_catchup():
    host, ctx = RecordingHead(), context(10)
    setattr(host, prompt_priming._CTX_ATTR, ctx)
    ctx.deferred_pairs = [(mx.array([[[90]]]), mx.array([[10]]))]
    with pytest.raises(RuntimeError, match="activation seam"):
        prompt_priming.take_primed(host, [], mx.array(11), cache_offset=13)
    assert ctx.mtp_cache[0].pairs == []


def parked_batch():
    host = RecordingHead()
    states = {
        uid: SimpleNamespace(
            chain=True,
            queue=[],
            next_main=mx.array([offset]),
            head_clone=True,
            hist_offset=offset,
            head_history_primed=True,
            mtp_cache=[SimpleNamespace(pairs=[])],
        )
        for uid, offset in [(3, 10), (7, 30)]
    }
    batch = SimpleNamespace(
        model=host,
        uids=[3, 7],
        prompt_cache=[SimpleNamespace(offset=mx.array([10, 30]))],
    )
    return batch, SimpleNamespace(states=states, head=None)


def test_retained_cache_survives_handoff_without_duplicate_pair():
    batch, owner = parked_batch()
    prompt_priming.retain_batch_head_history(batch, owner)
    _, owned = prompt_priming._owned(batch.model)
    for uid in batch.uids:
        assert owned.uids[uid][0].mtp_cache is owner.states[uid].mtp_cache
    with prompt_priming.decode_scope(batch.model, batch.uids):
        prompt_priming.maybe_capture(
            batch.model,
            mx.array([[10], [30]]),
            mx.array([[[100]], [[300]]]),
            [SimpleNamespace(offset=mx.array([11, 31]))],
        )
        prompt_priming.maybe_capture(
            batch.model,
            mx.array([[11], [31]]),
            mx.array([[[110]], [[310]]]),
            [SimpleNamespace(offset=mx.array([12, 32]))],
        )
    for uid, offset in [(3, 10), (7, 30)]:
        result = prompt_priming.take_primed(
            batch.model, [], mx.array(offset + 2), uid=uid, cache_offset=offset + 3
        )
        assert result[0][0].pairs == [
            (offset * 10, offset + 1),
            ((offset + 1) * 10, offset + 2),
        ]
        assert result[1] == offset + 2
    assert owned.uids == {}


@pytest.mark.parametrize("failure", ["queued", "owned"])
def test_retention_preflight_does_not_replace_another_row(failure):
    batch, owner = parked_batch()
    _, owned = prompt_priming._owned(batch.model, create=True)
    existing = context(30)
    if failure == "queued":
        owner.states[7].queue = [object()]
    else:
        owned.uids[7] = (existing, None)
    before = dict(owned.uids)
    with pytest.raises(RuntimeError):
        prompt_priming.retain_batch_head_history(batch, owner)
    assert owned.uids == before
    assert all(not state.mtp_cache[0].pairs for state in owner.states.values())


def test_retention_requires_an_observed_priming_context():
    batch, owner = parked_batch()
    owner.states[7].head_history_primed = False
    prompt_priming.retain_batch_head_history(batch, owner)
    _, owned = prompt_priming._owned(batch.model)
    assert set(owned.uids) == {3}
    assert not owner.states[7].mtp_cache[0].pairs


@pytest.mark.parametrize("frontier", ["queued", "legacy", "missing_main"])
def test_non_drained_handoff_does_not_start_history_retention(frontier, monkeypatch):
    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    batch, owner = parked_batch()
    if frontier == "queued":
        owner.states[7].queue = [object()]
    elif frontier == "legacy":
        owner.states[7].chain = False
    else:
        owner.states[7].next_main = None

    def unexpected(*args):
        raise AssertionError("History retention started outside its valid frontier")

    monkeypatch.setattr(prompt_priming, "retain_batch_head_history", unexpected)
    assert not bg._feed_batch_mains_to_standard(batch, owner)


class HeadHost:
    _omlx_mtp_decode_enabled = True
    _omlx_mtp_chain = True
    _omlx_mtp_multi_request = True
    mtp = object()

    def make_mtp_cache(self):
        return [SimpleNamespace(pairs=[])]

    def mtp_forward(self, hidden, tokens, cache, logits_keep=0):
        cache[0].pairs.extend(
            zip(hidden.reshape(-1).tolist(), tokens.reshape(-1).tolist())
        )


def test_chunked_ragged_prefill_ignores_padding_and_preserves_pending_request():
    host = HeadHost()
    prepare(host, "pending", [30, 31])
    capture(host, [30, 31], 2)
    pending = prompt_priming._find_ctx(host)
    cache = [SimpleNamespace(offset=mx.array([0, 0]))]
    with prompt_priming.prefill_scope(
        host, [11, 12], [[1, 2, 3, 4, 5], [9, 8, 7]], cache
    ):
        for tokens in ([[1, 2], [9, 8]], [[3, 4], [7, 0]], [[5], [0]]):
            inputs = mx.array(tokens)
            cache[0].offset += inputs.shape[1]
            prompt_priming.maybe_capture(host, inputs, inputs[..., None], cache)
    assert prompt_priming._find_ctx(host) is pending
    a = prompt_priming.take_primed(host, [], mx.array(6), uid=11, cache_offset=6)
    b = prompt_priming.take_primed(host, [], mx.array(6), uid=12, cache_offset=4)
    assert a[0][0].pairs == [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    assert b[0][0].pairs == [(9, 8), (8, 7), (7, 6)]
    assert prompt_priming._find_ctx(host) is pending


def test_prefill_scope_extends_owned_scheduler_histories_in_uid_order():
    host = HeadHost()
    for uid, tokens in [(11, [1, 2]), (12, [5, 6, 7])]:
        prepare(host, str(uid), tokens)
        capture(host, tokens, len(tokens))
        prompt_priming.bind_uid(host, str(uid), uid)
    cache = [SimpleNamespace(offset=mx.array([3, 2]))]
    with prompt_priming.prefill_scope(host, [12, 11], [[8, 9], [3]], cache):
        inputs = mx.array([[8, 9], [3, 0]])
        cache[0].offset += 2
        prompt_priming.maybe_capture(host, inputs, inputs[..., None], cache)
    a = prompt_priming.take_primed(host, [], mx.array(4), uid=11, cache_offset=4)
    b = prompt_priming.take_primed(host, [], mx.array(10), uid=12, cache_offset=6)
    assert a[0][0].pairs == [(1, 2), (2, 3), (3, 4)]
    assert b[0][0].pairs == [(5, 6), (6, 7), (7, 8), (8, 9), (9, 10)]


def test_prefill_scope_restores_after_exception_and_skips_empty_batches():
    host = HeadHost()
    with prompt_priming.prefill_scope(host, [], [], []):
        assert prompt_priming._PREFILL_SCOPE.get() is None
    with pytest.raises(ValueError):
        with prompt_priming.prefill_scope(
            host, [11], [[1, 2]], [SimpleNamespace(offset=0)]
        ):
            assert prompt_priming._PREFILL_SCOPE.get() is not None
            raise ValueError("cancelled")
    assert prompt_priming._PREFILL_SCOPE.get() is None
    prompt_priming.release_uids(host, [11])
    assert not prompt_priming._owned(host)[1].uids


def test_single_stream_model_does_not_start_batched_head_prefill():
    host = HeadHost()
    host._omlx_mtp_multi_request = False
    with prompt_priming.prefill_scope(
        host, [11, 12], [[1, 2], [3, 4]], [SimpleNamespace(offset=mx.array([0, 0]))]
    ):
        assert prompt_priming._PREFILL_SCOPE.get() is None
    assert prompt_priming._owned(host)[1] is None
    with prompt_priming.prefill_scope(
        host, [11], [[1, 2]], [SimpleNamespace(offset=0)]
    ):
        assert prompt_priming._PREFILL_SCOPE.get() is not None


def test_generator_remove_and_close_release_prefill_uids_only():
    from collections import deque

    from mlx_lm.generate import BatchGenerator

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    host = HeadHost()
    prepare(host, "waiting", [20, 21])
    capture(host, [20, 21], 2)
    with prompt_priming.prefill_scope(
        host, [11, 12], [[1, 2], [5, 6]], [SimpleNamespace(offset=mx.array([0, 0]))]
    ):
        inputs = mx.array([[1, 2], [5, 6]])
        prompt_priming.maybe_capture(
            host, inputs, inputs[..., None], [SimpleNamespace(offset=mx.array([2, 2]))]
        )

    class Batch:
        def __init__(self, uids):
            self.uids = uids

        def __len__(self):
            return len(self.uids)

        def filter(self, indices):
            self.uids = [self.uids[i] for i in indices]

    generator = BatchGenerator.__new__(BatchGenerator)
    generator.model = host
    generator._old_wired_limit = None
    generator._unprocessed_sequences = deque()
    generator._currently_processing = [object(), object()]
    generator._prompt_batch = Batch([11, 12])
    generator._generation_batch = Batch([])
    state = prompt_priming._owned(host)[1]
    try:
        generator.remove([11])
        assert set(state.uids) == {12}
        generator.close()
        assert not state.uids and set(state.requests) == {"waiting"}
    finally:
        generator.close()


def prepare(host, request_id, tokens):
    prompt_priming.prepare_prefix_context(
        host,
        request_id=request_id,
        prompt_tokens=tokens,
        cached_tokens=0,
        prefix_cache=None,
    )


def capture(host, tokens, offset):
    inputs = mx.array([tokens])
    prompt_priming.maybe_capture(
        host, inputs, inputs[..., None], [SimpleNamespace(offset=offset)]
    )


def test_interleaved_equal_length_requests_keep_distinct_history():
    host = HeadHost()
    prepare(host, "a", [1, 2, 3, 4])
    capture(host, [1, 2], 2)
    prepare(host, "b", [5, 6, 7, 8])
    capture(host, [5, 6], 2)
    prompt_priming.activate_request(host, "a")
    capture(host, [3, 4], 4)
    prompt_priming.bind_uid(host, "a", 11)
    prompt_priming.activate_request(host, "b")
    capture(host, [7, 8], 4)
    prompt_priming.bind_uid(host, "b", 12)
    _, state = prompt_priming._owned(host)
    assert not state.requests
    assert state.uids[11][0] is not state.uids[12][0]
    assert state.uids[11][0].mtp_cache[0].pairs == [(1, 2), (2, 3), (3, 4)]
    assert state.uids[12][0].mtp_cache[0].pairs == [(5, 6), (6, 7), (7, 8)]


def test_decode_reordering_and_unequal_offsets_preserve_pending_request():
    host = HeadHost()
    for uid, tokens in [(11, [1, 2]), (12, [5, 6, 7])]:
        prepare(host, str(uid), tokens)
        capture(host, tokens, len(tokens))
        prompt_priming.bind_uid(host, str(uid), uid)
    prepare(host, "pending", [20, 21])
    capture(host, [20, 21], 2)
    previous = prompt_priming._find_ctx(host)
    with prompt_priming.decode_scope(host, [12, 11]):
        inputs = mx.array([[8], [3]])
        prompt_priming.maybe_capture(
            host, inputs, inputs[..., None], [SimpleNamespace(offset=mx.array([4, 3]))]
        )
    assert prompt_priming._find_ctx(host) is previous
    a = prompt_priming.take_primed(
        host, [SimpleNamespace(offset=4)], mx.array(4), uid=11
    )
    b = prompt_priming.take_primed(
        host, [SimpleNamespace(offset=5)], mx.array(9), uid=12
    )
    assert a[0][0].pairs == [(1, 2), (2, 3), (3, 4)]
    assert b[0][0].pairs == [(5, 6), (6, 7), (7, 8), (8, 9)]
    assert prompt_priming._find_ctx(host) is previous
    assert not prompt_priming._owned(host)[1].uids


def test_identical_prompts_have_separate_contexts_and_cancel_releases_only_owner():
    host = HeadHost()
    for uid in [11, 12]:
        prepare(host, str(uid), [1, 2])
        capture(host, [1, 2], 2)
        prompt_priming.bind_uid(host, str(uid), uid)
    prompt_priming.release_request(host, "11")
    state = prompt_priming._owned(host)[1]
    assert set(state.uids) == {12}
    prompt_priming.release_uids(host, [12])
    assert not state.uids
    prepare(host, "waiting", [1, 2])
    capture(host, [1, 2], 2)
    prompt_priming.clear_owned(host)
    assert not state.requests and not state.uids
    assert prompt_priming._find_ctx(host) is None


def test_missing_uid_does_not_steal_equal_offset_pending_context():
    host = HeadHost()
    prepare(host, "pending", [1, 2])
    capture(host, [1, 2], 2)
    previous = prompt_priming._find_ctx(host)
    assert (
        prompt_priming.take_primed(
            host, [SimpleNamespace(offset=3)], mx.array(3), uid=99
        )
        is None
    )
    assert prompt_priming._find_ctx(host) is previous


def test_offset_rewind_invalidates_head_without_adopting_other_history():
    host = HeadHost()
    prepare(host, "a", [1, 2])
    capture(host, [1, 2], 2)
    prompt_priming.bind_uid(host, "a", 11)
    with prompt_priming.decode_scope(host, [11]):
        capture(host, [3], 2)  # Offset did not advance.
    assert (
        prompt_priming.take_primed(
            host, [SimpleNamespace(offset=3)], mx.array(4), uid=11
        )
        is None
    )


def test_decode_scope_restores_after_exception_and_models_do_not_share_state():
    first, second = HeadHost(), HeadHost()
    prepare(first, "a", [1, 2])
    prepare(second, "a", [5, 6])
    with (
        pytest.raises(ValueError),
        prompt_priming.decode_scope(first, [1]),
    ):  # noqa: SIM117
        with prompt_priming.decode_scope(second, [2]):
            raise ValueError("cancelled")
    assert prompt_priming._DECODE_SCOPE.get() is None
    assert prompt_priming._owned(first)[1] is not prompt_priming._owned(second)[1]


def test_dspark_context_is_owned_by_custom_hook():
    host = HeadHost()
    host._omlx_dspark_decode_enabled = True
    custom = object()
    setattr(host, prompt_priming._CTX_ATTR, custom)
    prepare(host, "a", [1, 2])
    prompt_priming.release_request(host, "a")
    prompt_priming.clear_owned(host)
    assert prompt_priming._find_ctx(host) is custom
    assert prompt_priming._owned(host)[1] is None


@pytest.mark.parametrize("operation", ["reset", "deep_reset", "shutdown"])
def test_scheduler_teardown_releases_owned_histories(
    mock_model, mock_tokenizer, operation
):
    from omlx.scheduler import Scheduler

    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    mock_model._omlx_mtp_decode_enabled = True
    mock_model._omlx_mtp_chain = True
    mock_model.mtp = object()
    prepare(mock_model, "waiting", [1, 2])
    prepare(mock_model, "inserted", [3, 4])
    prompt_priming.bind_uid(mock_model, "inserted", 11)
    state = prompt_priming._owned(mock_model)[1]
    assert state.requests and state.uids
    getattr(scheduler, operation)()
    assert not state.requests and not state.uids


def test_scheduler_abort_releases_only_cancelled_request(mock_model, mock_tokenizer):
    from omlx.request import Request, SamplingParams
    from omlx.scheduler import Scheduler

    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    mock_model._omlx_mtp_decode_enabled = True
    mock_model._omlx_mtp_chain = True
    mock_model.mtp = object()
    scheduler.add_request(
        Request(request_id="waiting", prompt="Hello", sampling_params=SamplingParams())
    )
    prepare(mock_model, "waiting", [1, 2])
    prepare(mock_model, "inserted", [3, 4])
    prompt_priming.bind_uid(mock_model, "inserted", 11)
    state = prompt_priming._owned(mock_model)[1]
    scheduler.abort_request("waiting")
    scheduler._process_pending_aborts()
    assert not state.requests and set(state.uids) == {11}
    prompt_priming.clear_owned(mock_model)


def test_scheduler_completion_releases_unused_priming(mock_model, mock_tokenizer):
    from omlx.scheduler import Scheduler

    scheduler = Scheduler(model=mock_model, tokenizer=mock_tokenizer)
    mock_model._omlx_mtp_decode_enabled = True
    mock_model._omlx_mtp_chain = True
    mock_model.mtp = object()
    prepare(mock_model, "finished", [1, 2])
    prompt_priming.bind_uid(mock_model, "finished", 11)
    state = prompt_priming._owned(mock_model)[1]
    scheduler._cleanup_finished({"finished"})
    assert not state.requests and not state.uids


def test_generation_scope_uses_uids_after_realignment(monkeypatch):
    from contextlib import contextmanager

    from mlx_lm.generate import GenerationBatch

    from omlx.patches.mlx_lm_mtp import batch_generator as bg

    bg.apply()
    batch = SimpleNamespace(model=HeadHost(), uids=[11, 12])
    batch._omlx_realign_rows = lambda: setattr(batch, "uids", [12, 11])

    class ScopeReachedError(Exception):
        pass

    @contextmanager
    def capture_scope(model, uids):
        assert model is batch.model
        assert list(uids) == [12, 11]
        raise ScopeReachedError
        yield  # pragma: no cover

    monkeypatch.setattr(prompt_priming, "decode_scope", capture_scope)
    with pytest.raises(ScopeReachedError):
        GenerationBatch.next(batch)
