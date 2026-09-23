# SPDX-License-Identifier: Apache-2.0
"""SpecPrefill on DSA layers: indexer positions must track the attention.

A DSA layer (GLM-5.2, DeepSeek V3.2/V4) rotates twice — once in the attention
and once in the indexer that picks the sparse top-k. SpecPrefill only ever
remapped the attention's RoPE, so the indexer kept scoring at compacted
offsets and selected the wrong tokens; generation after a correct sparse
prefill was then corrupted on exactly those models.

The tests below pin the observable contract: for a multi-token generation, the
positions consumed by *both* RoPE owners must equal the dense positions.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest

import omlx.patches.specprefill as specprefill
from omlx.patches.specprefill import (
    _cache_offset,
    _OffsetAdjustedRoPE,
    _PositionMappedRoPE,
    _rope_owners,
    _slot_cache_offset,
    cleanup_rope,
    sparse_prefill,
)


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _RecordingRoPE:
    """Genuine RoPE stand-in that records the positions it is asked for.

    Only the *decode* path reaches here: during sparse prefill the wrapper
    routes through ``manual_rope`` instead, which ``_capture_prefill`` below
    observes. The distinct ``base`` per owner is what lets that capture tell
    the attention's calls from the indexer's.
    """

    def __init__(self, base: float = 10000.0) -> None:
        self.dims = 4
        self.base = base
        self.scale = 1.0
        self.seen: list[list[int]] = []

    def __call__(self, x: Any, offset: Any = 0) -> Any:
        length = x.shape[2]
        start = int(offset)
        self.seen.append(list(range(start, start + length)))
        return x


_ATTENTION_BASE = 10000.0
_INDEXER_BASE = 500000.0


@contextmanager
def _capture_prefill():
    """Record the positions sparse prefill actually applies, per owner.

    Patches ``manual_rope`` — the real function ``_PositionMappedRoPE`` calls —
    so the assertions observe the production path rather than re-deriving it.
    """
    seen: dict[float, list[int]] = {_ATTENTION_BASE: [], _INDEXER_BASE: []}

    def _fake(x, positions, dims, base=10000.0, scale=1.0):
        seen.setdefault(base, []).extend(int(p) for p in positions.tolist())
        return x

    with patch.object(specprefill, "manual_rope", _fake):
        yield seen


class _KVCache:
    def __init__(self, offset: int = 0) -> None:
        self.offset = offset
        self.state = object()

    def advance(self, n: int) -> None:
        self.offset += n


class _CacheList:
    """Mirrors mlx_lm CacheList: indexable, and deliberately without .offset."""

    def __init__(self, *caches: Any) -> None:
        self.caches = caches

    def __getitem__(self, idx: int) -> Any:
        return self.caches[idx]

    @property
    def state(self) -> Any:
        return [c.state for c in self.caches]


class _Indexer:
    def __init__(self) -> None:
        self.rope = _RecordingRoPE(base=_INDEXER_BASE)


class _DsaAttention:
    def __init__(self, with_indexer: bool = True) -> None:
        self.rope = _RecordingRoPE(base=_ATTENTION_BASE)
        # Shared DSA layers carry indexer=None and reuse the previous full one.
        self.indexer = _Indexer() if with_indexer else None


class _Layer:
    def __init__(self, attn: Any) -> None:
        self.self_attn = attn


class _DsaModel:
    """Single-layer DSA model whose forward advances both cache slots."""

    def __init__(self, with_indexer: bool = True) -> None:
        self.attn = _DsaAttention(with_indexer=with_indexer)
        self.layers = [_Layer(self.attn)]

    def __call__(self, tokens: Any, *, cache: Any) -> Any:
        length = tokens.shape[1]
        entry = cache[0]
        # Both the attention and the indexer rotate, each against its own slot.
        self.attn.rope(mx.zeros((1, 1, length, 4)), offset=entry[0].offset)
        if self.attn.indexer is not None:
            self.attn.indexer.rope(mx.zeros((1, 1, length, 4)), offset=entry[1].offset)
        for c in entry.caches:
            c.advance(length)
        return mx.zeros((1, length, 8))


def _make_cache(system_len: int = 0) -> list[Any]:
    return [_CacheList(_KVCache(system_len), _KVCache(system_len))]


def _decode(model: _DsaModel, cache: list[Any], n_tokens: int) -> None:
    for _ in range(n_tokens):
        model(mx.zeros((1, 1), dtype=mx.int32), cache=cache)


# ---------------------------------------------------------------------------
# Helper-level contracts
# ---------------------------------------------------------------------------


def test_cache_offset_descends_into_cachelist() -> None:
    """CacheList has no .offset — a hasattr check silently reported 0."""
    entry = _CacheList(_KVCache(128), _KVCache(128))
    assert not hasattr(entry, "offset")
    assert _cache_offset(entry) == 128
    assert _cache_offset(entry, default=999) == 128


def test_cache_offset_falls_back_when_unknown() -> None:
    assert _cache_offset(None, default=7) == 7
    assert _cache_offset(object(), default=7) == 7


def test_slot_cache_offset_targets_each_owner() -> None:
    entry = _CacheList(_KVCache(10), _KVCache(20))
    assert _slot_cache_offset(entry, 0) == 10
    assert _slot_cache_offset(entry, 1) == 20


def test_rope_owners_yields_attention_and_indexer() -> None:
    attn = _DsaAttention()
    assert [slot for slot, _ in _rope_owners(attn)] == [0, 1]

    shared = _DsaAttention(with_indexer=False)
    assert [slot for slot, _ in _rope_owners(shared)] == [0]


# ---------------------------------------------------------------------------
# The regression: multi-token generation must match dense
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("system_len", [0, 96])
def test_multi_token_generation_positions_match_dense(system_len: int) -> None:
    """Sparse prefill + decode must consume the same positions as dense.

    Includes a restored system prompt (system_len > 0), which is what the
    CacheList offset descent fixes: without it cache_start collapsed to 0.
    """
    n_conv = 16
    n_generate = 6
    tokens = mx.arange(system_len + n_conv, dtype=mx.int32)
    selected = mx.array(list(range(system_len, system_len + n_conv, 2)))

    # --- dense reference: every position, contiguous ---
    dense = _DsaModel()
    dense_cache = _make_cache(system_len)
    dense(tokens[system_len:][None], cache=dense_cache)
    _decode(dense, dense_cache, n_generate)
    dense_attn = [p for step in dense.attn.rope.seen for p in step]
    dense_index = [p for step in dense.attn.indexer.rope.seen for p in step]

    # --- sparse prefill on the selected subset, then the same decode ---
    sparse = _DsaModel()
    sparse_cache = _make_cache(system_len)
    with _capture_prefill() as prefill_positions:
        sparse_prefill(
            sparse,
            tokens[system_len:],
            selected - system_len,
            sparse_cache,
            position_offset=system_len,
        )
    _decode(sparse, sparse_cache, n_generate)

    kept = selected.tolist()
    total = system_len + n_conv

    # Prefill: the kept tokens keep their ORIGINAL positions, not 0..N-1.
    assert prefill_positions[_ATTENTION_BASE] == kept
    # The indexer must agree with the attention token for token. This is the
    # assertion that fails without the fix: the indexer's rope was never
    # wrapped, so it never went through manual_rope at all.
    assert prefill_positions[_INDEXER_BASE] == kept

    # Decode: every generated token continues from the true prompt length,
    # identically for both owners and identically to dense.
    expected_decode = list(range(total, total + n_generate))
    sparse_attn = [p for step in sparse.attn.rope.seen for p in step]
    sparse_index = [p for step in sparse.attn.indexer.rope.seen for p in step]
    assert sparse_attn == expected_decode
    assert sparse_index == expected_decode
    assert dense_attn[-n_generate:] == expected_decode
    assert dense_index[-n_generate:] == expected_decode


def test_indexer_rope_is_restored_after_generation() -> None:
    """A leaked wrapper would corrupt the next, unrelated request."""
    model = _DsaModel()
    cache = _make_cache()
    tokens = mx.arange(12, dtype=mx.int32)
    genuine_attn = model.attn.rope
    genuine_index = model.attn.indexer.rope

    sparse_prefill(model, tokens, mx.array([0, 2, 4, 6]), cache)

    # Decode wrappers are installed on BOTH owners while generating.
    assert isinstance(model.attn.rope, _OffsetAdjustedRoPE)
    assert isinstance(model.attn.indexer.rope, _OffsetAdjustedRoPE)

    cleanup_rope(model)
    assert model.attn.rope is genuine_attn
    assert model.attn.indexer.rope is genuine_index


def test_shared_dsa_layer_without_indexer_is_untouched() -> None:
    model = _DsaModel(with_indexer=False)
    cache = _make_cache()
    tokens = mx.arange(12, dtype=mx.int32)

    sparse_prefill(model, tokens, mx.array([0, 2, 4, 6]), cache)
    assert isinstance(model.attn.rope, _OffsetAdjustedRoPE)

    cleanup_rope(model)
    assert isinstance(model.attn.rope, _RecordingRoPE)


def test_position_mapped_rope_preserves_dtype() -> None:
    """Guards the fix in 54357e04: a float32 leak becomes an unpromotable
    sdpa mask on GLM DSA, where pe_scores IS the mask."""
    rope = _PositionMappedRoPE(_RecordingRoPE(), mx.arange(8, dtype=mx.int32))
    x = mx.zeros((1, 1, 4, 8), dtype=mx.bfloat16)
    assert rope(x, offset=0).dtype == mx.bfloat16
