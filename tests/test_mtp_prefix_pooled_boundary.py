# SPDX-License-Identifier: Apache-2.0
"""#4471: MTP prefix snapshot for pooled heads through the real priming seam.

The host is glm5_next-shaped and model-free: its head cache is the real
``[KVCache, PoolingCache(4)]`` pair, and ``mtp_forward`` folds one KV row per
pair plus gated window-mean pooled rows (optionally with the DS4 overlap
carry). The gate input differs from the KV input so gate buffers are checked.
Everything else -- ``prepare_prefix_context``, ``maybe_capture``,
``capture_tail_boundary``, ``take_primed`` (seam fold + publication) and the
restore -- is the production code path.
"""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache, PoolingCache
from omlx.patches.mlx_lm_mtp import prompt_priming as pp

R, BLOCK = 4, 8
TOKENS = list(range(100, 140))


def _hidden(toks):
    t = mx.array(toks, dtype=mx.float32)[None, :, None]
    return mx.concatenate([t * 0.5, t * 0.25 + 1], axis=-1)


class PooledHead:
    _omlx_mtp_decode_enabled = True
    _omlx_mtp_chain = True
    mtp = object()

    def __init__(self, overlap=False, pooled=True):
        self.overlap, self.pooled = overlap, pooled

    def make_mtp_cache(self):
        return [KVCache(), PoolingCache(R) if self.pooled else KVCache()]

    def mtp_forward(self, hidden, tokens, cache, logits_keep=0):
        x = hidden + tokens[..., None].astype(hidden.dtype)
        g = 7.0 - 3.0 * x  # distinct gate input
        cache[0].update_and_fetch(x[:, None], x[:, None])
        pool = cache[1]
        if not self.pooled:
            pool.update_and_fetch(x[:, None], x[:, None])
            return None
        pos = pool._pool_len * pool.ratio + pool.remainder
        ready, ready_g, _ = pool.accumulate_windows(x, g, pos)
        if ready.size:
            w = mx.unflatten(ready, 1, (-1, R))
            wg = mx.unflatten(ready_g, 1, (-1, R))
            if self.overlap:
                prev, prev_g = pool.prev_for_prepend()
                if prev is not None:
                    w2 = mx.concatenate([prev, w], axis=1)
                    wg2 = mx.concatenate([prev_g, wg], axis=1)
                else:
                    w2, wg2 = w, wg
                pool.store_prev(w2, wg2, 0)
            pool.update_and_fetch(w.mean(axis=2) + 0.5 * wg.mean(axis=2))
        return None


class Sidecar:
    """In-memory store/restore contract (as in test_mtp_prompt_priming)."""

    block_size = BLOCK

    def __init__(self):
        self.snapshots = {}

    def store_mtp_prefix_snapshot(self, tokens, boundary, snapshot, **kw):
        self.snapshots[(tuple(tokens[:boundary]), boundary)] = snapshot
        return True

    def restore_mtp_prefix_snapshot(self, tokens, boundary, **kw):
        return self.snapshots.get((tuple(tokens[:boundary]), boundary))


def _run(host, sidecar, rid, chunks, *, cached=0, tail=None):
    """Scheduler order: prefill chunks, tail capture, S==1 step, activation."""
    warm = pp.prepare_prefix_context(
        host,
        request_id=rid,
        prompt_tokens=TOKENS,
        cached_tokens=cached,
        prefix_cache=sidecar,
    )
    pos = cached
    for i, n in enumerate(chunks):
        toks = TOKENS[pos : pos + n]
        pos += n
        pp.maybe_capture(
            host, mx.array([toks]), _hidden(toks), [SimpleNamespace(offset=pos)]
        )
        if tail is not None and i == len(chunks) - 2:
            pp.capture_tail_boundary(host, rid, tail)
    primed = pp.take_primed(host, [], mx.array([TOKENS[pos]]), cache_offset=pos + 1)
    return warm, primed


def _oracle(pair_chunks, host=None):
    """Fresh head folded with pairs (hidden[t], token[t+1]) in these chunks."""
    host = host or PooledHead()
    cache, t = host.make_mtp_cache(), 0
    for n in pair_chunks:
        host.mtp_forward(
            _hidden(TOKENS[t : t + n]), mx.array([TOKENS[t + 1 : t + n + 1]]), cache
        )
        t += n
    return cache


def _val(v):
    if isinstance(v, mx.array):
        return v.tolist()
    if isinstance(v, (tuple, list)):
        return [_val(x) for x in v]
    return v


def _state(cache):
    kv, p = cache
    rows = p.pooled.tolist() if p._pool_len else []
    tail = p.buf_kv[:, : p.remainder].tolist() if p.remainder else []
    gate = p.buf_gate[:, : p.remainder].tolist() if p.remainder else []
    return (
        kv.offset,
        kv.keys[..., : kv.offset, :].tolist(),
        p._pool_len,
        rows,
        p.remainder,
        tail,
        gate,
        _val(p.prev_win_kv),
        _val(p.prev_win_gate),
        _val(p._undo),
    )


# Prefill 16 + 4 tokens, tail boundary C=20, S==1 step, then the seam:
# the head ends at 21 tokens (remainder 1), C-1=19 -> a 2-token rewind that
# crosses a completed window and is outside the 1-token seam undo log.
CHUNKS, C = (16, 4, 1), 20


@pytest.mark.parametrize("overlap", [False, True])
def test_tail_snapshot_exact_at_c_minus_1_despite_cross_window_gap(overlap):
    host, sidecar = PooledHead(overlap), Sidecar()
    _, primed = _run(host, sidecar, "cold", CHUNKS, tail=C)
    live = primed[0]
    assert (_state(live)[2], live[1].remainder, primed[1]) == (5, 1, 21)
    snap = sidecar.snapshots.get((tuple(TOKENS[:C]), C))
    assert snap is not None and snap.boundary_tokens == C
    assert _state(snap.mtp_cache) == _state(_oracle([15, 4], PooledHead(overlap)))
    assert snap.pending_hidden.tolist() == _hidden([TOKENS[C - 1]]).tolist()


@pytest.mark.parametrize("overlap", [False, True])
def test_warm_restore_matches_cold_head_and_leaves_snapshot_intact(overlap):
    sidecar = Sidecar()
    _, cold = _run(PooledHead(overlap), sidecar, "cold", CHUNKS, tail=C)
    snap = sidecar.snapshots[(tuple(TOKENS[:C]), C)]
    before = _state(snap.mtp_cache)
    for rid in ("warm-1", "warm-2"):
        warm, primed = _run(PooledHead(overlap), sidecar, rid, (1,), cached=C)
        assert warm and primed[1] == cold[1]
        assert _state(primed[0]) == _state(cold[0])
        assert all(a is not b for a, b in zip(primed[0], snap.mtp_cache))
        assert _state(snap.mtp_cache) == before


def test_capture_does_not_alias_or_change_the_live_head():
    host = PooledHead(overlap=True)
    _, plain = _run(PooledHead(overlap=True), None, "plain", CHUNKS)
    sidecar = Sidecar()
    pp.prepare_prefix_context(
        host,
        request_id="cold",
        prompt_tokens=TOKENS,
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    pos = 0
    for n in (16, 4):
        toks = TOKENS[pos : pos + n]
        pos += n
        pp.maybe_capture(
            host, mx.array([toks]), _hidden(toks), [SimpleNamespace(offset=pos)]
        )
    pp.capture_tail_boundary(host, "cold", C)
    ctx = pp._find_ctx(host)
    head = ctx.snapshot_candidate.mtp_cache
    assert head is not None
    for clone, entry in zip(head, ctx.mtp_cache):
        assert clone is not entry
        for name, value in vars(entry).items():
            if isinstance(value, mx.array):
                assert getattr(clone, name) is not value
    toks = TOKENS[C : C + 1]
    pp.maybe_capture(
        host, mx.array([toks]), _hidden(toks), [SimpleNamespace(offset=C + 1)]
    )
    primed = pp.take_primed(host, [], mx.array([TOKENS[C + 1]]), cache_offset=C + 2)
    assert _state(primed[0]) == _state(plain[0])
    assert _state(head) == _state(_oracle([15, 4], PooledHead(overlap=True)))


def test_kv_only_head_keeps_the_lazy_candidate():
    host, sidecar = PooledHead(pooled=False), Sidecar()
    pp.prepare_prefix_context(
        host,
        request_id="kv",
        prompt_tokens=TOKENS,
        cached_tokens=0,
        prefix_cache=sidecar,
    )
    pp.maybe_capture(
        host,
        mx.array([TOKENS[:16]]),
        _hidden(TOKENS[:16]),
        [SimpleNamespace(offset=16)],
    )
    candidate = pp._find_ctx(host).snapshot_candidate
    assert getattr(candidate, "mtp_cache", None) is None
    pp.take_primed(host, [], mx.array([TOKENS[16]]), cache_offset=17)
    snap = sidecar.snapshots[(tuple(TOKENS[:16]), 16)]
    assert [c.offset for c in snap.mtp_cache] == [15, 15]


def test_unrewindable_in_chunk_boundary_publishes_nothing():
    # One 30-token chunk: block boundary 24 lies 6 tokens behind the head.
    sidecar = Sidecar()
    _, primed = _run(PooledHead(), sidecar, "cold", (30,))
    assert primed is not None and sidecar.snapshots == {}


def test_unsupported_or_mixed_unit_pooled_state_fails_closed():
    batched = [KVCache(), BatchPoolingCache(R, [0])]
    assert pp._cache_at_offset(batched, 0) is None
    head = _oracle([35])  # 8 pooled rows; token target 8 must not match
    assert pp._cache_at_offset(head, 8) is None


def test_older_candidate_is_kept_and_published_when_newest_cannot_rewind():
    # Chunk 1 ends on boundary 16 (head at 15: exact candidate). Chunk 2 ends
    # at 36: boundary 32 is 4 tokens behind the head (remainder 3, no undo log
    # for a 20-token update), so the 16-token candidate must survive.
    sidecar = Sidecar()
    _, primed = _run(PooledHead(overlap=True), sidecar, "cold", (16, 20))
    assert primed is not None
    assert list(sidecar.snapshots) == [(tuple(TOKENS[:16]), 16)]
    snap = sidecar.snapshots[(tuple(TOKENS[:16]), 16)]
    assert _state(snap.mtp_cache) == _state(_oracle([15], PooledHead(overlap=True)))
    assert snap.pending_hidden.tolist() == _hidden([TOKENS[15]]).tolist()


def test_gate_buffers_are_distinct_and_snapshotted_exactly():
    sidecar = Sidecar()
    _, primed = _run(PooledHead(overlap=True), sidecar, "cold", CHUNKS, tail=C)
    head = sidecar.snapshots[(tuple(TOKENS[:C]), C)].mtp_cache
    pool, ref = head[1], _oracle([15, 4], PooledHead(overlap=True))[1]
    assert pool.remainder == 3 and pool.prev_win_gate is not None
    kv_tail, gate_tail = pool.buf_kv[:, :3].tolist(), pool.buf_gate[:, :3].tolist()
    assert gate_tail != kv_tail
    assert gate_tail == ref.buf_gate[:, :3].tolist()
    assert pool.prev_win_gate.tolist() == ref.prev_win_gate.tolist()
    assert pool.prev_win_gate.tolist() != pool.prev_win_kv.tolist()
    assert all(
        getattr(pool, n) is not getattr(primed[0][1], n) for n in ("buf_kv", "buf_gate")
    )
