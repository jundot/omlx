# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp ``BatchKVCache`` capacity buffer stays bit-exact.

The batch KV bank used to concatenate one 256-column step onto every row each
time decode crossed the physical width, copying the whole bank and stranding
the previous exact-width buffer in MLX's pool every 256 tokens. It now lives
in a capacity buffer on a bounded-ratio ladder and appends write in place.

Only allocation changes. ``_ConcatBatchKV`` below is the pre-change class
(logic verbatim); every public result (fetched K/V, ``state`` including pad columns,
``offset``, ``left_padding``, ``_idx``, masks, extracted rows, masked
attention) must match it bit-for-bit, and columns ``[0, _width)`` of the new
buffer must equal the old physical buffer exactly.
"""

# The reference class keeps the vendored cache's shape names (B, H, L1, ...).
# ruff: noqa: N803, N806

import random

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.mlx_vlm_qwen4_exp_compat import (  # noqa: E402
    apply_mlx_vlm_qwen4_exp_compat_patch,
)

apply_mlx_vlm_qwen4_exp_compat_patch()

from mlx_vlm.models.qwen4_exp.cache import (  # noqa: E402
    BatchKVCache,
    KVCache,
    create_causal_mask,
    dynamic_roll,
)

H, D = 2, 8


class _ConcatBatchKV:
    """``BatchKVCache`` before this change (concatenate growth).

    The logic is verbatim; docstrings, comments and unused helpers are dropped.
    """

    step = 256

    def __init__(self, left_padding):
        self.keys = None
        self.values = None
        self.left_padding = mx.array(left_padding)
        self.offset = mx.array([-pad for pad in left_padding])
        self._idx = 0
        self._right_padding = None

    def update_and_fetch(self, keys, values):
        prev = self._idx
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self._idx += keys.shape[2]
        self.keys[..., prev : self._idx, :] = keys
        self.values[..., prev : self._idx, :] = values
        return self.keys[..., : self._idx, :], self.values[..., : self._idx, :]

    def prepare(self, *, left_padding=None, lengths=None, right_padding=None):
        if left_padding is not None:
            if self.keys is not None:
                raise ValueError(
                    "Left padding can only be added to an empty BatchKVCache"
                )
            left_padding = mx.array(left_padding)
            self.left_padding += left_padding
            self.offset -= left_padding

        if right_padding is not None and max(right_padding) > 0:
            self._right_padding = mx.array(right_padding)

    def finalize(self):
        if self._right_padding is not None:
            padding = self._right_padding
            self.keys = dynamic_roll(self.keys, padding[:, None], axis=2)
            self.values = dynamic_roll(self.values, padding[:, None], axis=2)
            self.offset -= padding
            self.left_padding += padding
            self._right_padding = None

    @property
    def state(self):
        k, v = self.keys, self.values
        if self._idx < k.shape[2]:
            k = k[..., : self._idx, :]
            v = v[..., : self._idx, :]
        return k, v, self.offset, self.left_padding

    @state.setter
    def state(self, v):
        self.keys, self.values, self.offset, self.left_padding = v
        self._idx = self.keys.shape[2]

    def trim(self, n):
        n = min(self._idx, n)
        self._idx -= n
        self.offset -= n
        return n

    def make_mask(self, N: int, return_array: bool = False, **kwargs):
        return create_causal_mask(
            N, offset=self._idx, left_padding=self.left_padding, **kwargs
        )

    def filter(self, batch_indices):
        if self.keys is not None:
            self.keys = self.keys[batch_indices]
            self.values = self.values[batch_indices]
        self.offset = self.offset[batch_indices]
        self.left_padding = self.left_padding[batch_indices]
        if self._right_padding is not None:
            self._right_padding = self._right_padding[batch_indices]

        min_left_pad = self.left_padding.min().item()
        if min_left_pad > 0:
            if self.keys is not None:
                self.keys = self.keys[..., min_left_pad:, :]
                self.values = self.values[..., min_left_pad:, :]
            self._idx -= min_left_pad
            self.left_padding -= min_left_pad

    def extend(self, other):
        if self.keys is None and other.keys is None:
            self.left_padding = mx.concatenate([self.left_padding, other.left_padding])
            self.offset = mx.concatenate([self.offset, other.offset])
            return

        max_idx = max(self._idx, other._idx)
        L1 = L2 = 0
        if self.keys is not None:
            B, H, L1, D = self.keys.shape
            M = self.values.shape[3]
        if other.keys is not None:
            B, H, L2, D = other.keys.shape
            M = other.values.shape[3]
        max_size = max(L1, L2)

        def pad(c):
            k, v = c.keys, c.values
            if k is None:
                Bc = c.offset.shape[0]
                k = mx.array([]).reshape(Bc, H, 0, D)
                v = mx.array([]).reshape(Bc, H, 0, M)
            left = max_idx - c._idx
            right = max_size - k.shape[2] - left
            if right < 0:
                k = k[..., :right, :]
                v = v[..., :right, :]
                right = 0
            if left != 0 or right != 0:
                pad = [(0, 0), (0, 0), (left, right), (0, 0)]
                k = mx.pad(k, pad)
                v = mx.pad(v, pad)
            left_padding = c.left_padding + left
            return k, v, c.offset, left_padding

        self.keys, self.values, self.offset, self.left_padding = map(
            mx.concatenate, zip(*(pad(self), pad(other)))
        )
        self._idx = max_idx

    def extract(self, idx):
        cache = KVCache()
        padding = self.left_padding[idx].item()
        cache.keys = mx.contiguous(self.keys[idx : idx + 1, :, padding : self._idx])
        cache.values = mx.contiguous(self.values[idx : idx + 1, :, padding : self._idx])
        cache.offset = cache.keys.shape[2]
        return cache

    @classmethod
    def merge(cls, caches):
        lengths = [c.size() for c in caches]
        max_length = max(lengths)
        if max_length == 0:
            return cls([0] * len(caches))
        padding = [max_length - length for length in lengths]
        B = len(caches)
        H = max(c.keys.shape[1] for c in caches if c.keys is not None)
        Dk = max(c.keys.shape[3] for c in caches if c.keys is not None)
        Dv = max(c.values.shape[3] for c in caches if c.values is not None)
        dt = next(iter(c.keys.dtype for c in caches if c.keys is not None))
        capacity = max_length + cls.step
        keys = mx.zeros((B, H, capacity, Dk), dtype=dt)
        values = mx.zeros((B, H, capacity, Dv), dtype=dt)
        for i, (p, c) in enumerate(zip(padding, caches)):
            if c.keys is None:
                continue
            keys[i : i + 1, :, p : p + c.offset] = c.keys[..., : c.offset, :]
            values[i : i + 1, :, p : p + c.offset] = c.values[..., : c.offset, :]
        cache = cls(padding)
        cache.keys = keys
        cache.values = values
        cache.offset += max_length
        cache._idx = max_length
        return cache

    def size(self):
        return self._idx


def _bits(a):
    if a.dtype == mx.bfloat16:
        a = a.view(mx.uint16)
    return np.array(a)


def _same(a, b):
    assert (a is None) == (b is None)
    if a is None:
        return
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype
    np.testing.assert_array_equal(_bits(a), _bits(b))


def _rnd(shape, key):
    return mx.random.normal(shape, key=mx.random.key(key)).astype(mx.bfloat16)


def _row(length, key):
    row = KVCache()
    row.update_and_fetch(_rnd((1, H, length, D), key), _rnd((1, H, length, D), key + 1))
    return row


def _logical_width(cache):
    width = getattr(cache, "_width", None)
    if width is not None:
        return width
    return 0 if cache.keys is None else int(cache.keys.shape[2])


def _assert_equivalent(new, ref):
    """Every observable, plus the concatenate-layout prefix, is identical."""
    assert new._idx == ref._idx
    _same(new.offset, ref.offset)
    _same(new.left_padding, ref.left_padding)
    if ref.keys is None:
        assert new.keys is None
        return
    for got, want in zip(new.state, ref.state):
        _same(got, want)
    width = _logical_width(new)
    assert width == ref.keys.shape[2]
    _same(new.keys[..., :width, :], ref.keys)
    _same(new.values[..., :width, :], ref.values)
    # Columns past the concatenate layout are never written.
    if new.keys.shape[2] > width:
        assert not np.any(_bits(new.keys[..., width:, :]))
        assert not np.any(_bits(new.values[..., width:, :]))


def _attend(k, v, mask, key):
    q = _rnd((k.shape[0], H, mask.shape[-2], D), key)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=D**-0.5, mask=mask)


def run_random_sequence(seed, n_ops=40, step=16, new_cls=None):
    """Drive the new class and the verbatim old one through one random run."""
    rng = random.Random(seed)
    new_cls = new_cls or type("_NewKV", (BatchKVCache,), {"step": step})
    ref_cls = type("_RefKV", (_ConcatBatchKV,), {"step": step})
    key = seed * 100_000

    def fresh_rows(n):
        nonlocal key
        rows = []
        for _ in range(n):
            key += 2
            rows.append(_row(rng.randint(1, 5 * step), key))
        return rows

    rows = fresh_rows(rng.randint(1, 3))
    new, ref = new_cls.merge(rows), ref_cls.merge(rows)
    _assert_equivalent(new, ref)
    for _ in range(n_ops):
        op = rng.choices(
            ["append", "trim", "ragged", "filter", "extend", "rebuild", "mask"],
            weights=[8, 3, 4, 1, 1, 1, 1],
        )[0]
        B = int(new.offset.shape[0])
        if op == "append":
            m = rng.choice([1, 2, 3, 4, 5, rng.randint(1, 3 * step)])
            key += 3
            k, v = _rnd((B, H, m, D), key), _rnd((B, H, m, D), key + 1)
            mask = ref.make_mask(m)
            _same(new.make_mask(m), mask)
            got = new.update_and_fetch(k, v)
            want = ref.update_and_fetch(k, v)
            _same(got[0], want[0])
            _same(got[1], want[1])
            # The masked attention the model computes over the fetched bank.
            _same(_attend(*got, mask, key + 2), _attend(*want, mask, key + 2))
        elif op == "trim":
            # Speculative trims only drop draft columns: every row keeps at
            # least one real token past its left padding.
            room = new._idx - int(new.left_padding.max().item()) - 1
            n = rng.randint(0, max(0, min(room, step)))
            assert new.trim(n) == ref.trim(n)
        elif op == "ragged":
            real = [new._idx - p for p in new.left_padding.tolist()]
            right = [rng.randint(0, max(0, min(4, r - 1))) for r in real]
            new.prepare(right_padding=right)
            ref.prepare(right_padding=right)
            new.finalize()
            ref.finalize()
        elif op == "filter" and B > 1:
            keep = sorted(rng.sample(range(B), rng.randint(1, B - 1)))
            new.filter(mx.array(keep))
            ref.filter(mx.array(keep))
        elif op == "extend" and B < 4:
            rows = fresh_rows(rng.randint(1, 2))
            new.extend(new_cls.merge(rows))
            ref.extend(ref_cls.merge(rows))
        elif op == "rebuild":
            # Lightning MTP rebuild: extract every row and merge them again.
            got_rows = [new.extract(i) for i in range(B)]
            want_rows = [ref.extract(i) for i in range(B)]
            for g, w in zip(got_rows, want_rows):
                _same(g.keys, w.keys)
                _same(g.values, w.values)
                assert g.offset == w.offset
            if all(r.offset for r in want_rows):
                new, ref = new_cls.merge(got_rows), ref_cls.merge(want_rows)
        elif op == "mask":
            n = rng.randint(1, 5)
            _same(new.make_mask(n), ref.make_mask(n))
        mx.eval(new.keys, new.values, ref.keys, ref.values)
        _assert_equivalent(new, ref)
    return new, ref


@pytest.mark.parametrize("seed", range(40))
def test_random_operation_sequences_match_concatenate_bit_for_bit(seed):
    run_random_sequence(seed)


def test_default_step_random_sequences_match_concatenate():
    for seed in range(1000, 1004):
        run_random_sequence(seed, n_ops=25, step=256, new_cls=BatchKVCache)


def test_decode_after_merge_writes_in_place():
    length = 16384
    rows = [_row(length, 1), _row(length - 37, 11)]
    batch = BatchKVCache.merge(rows)
    ref = _ConcatBatchKV.merge(rows)
    capacity = batch.keys.shape[2]
    # Bounded reserve: the first step plus [1/32, 1/8) of the length.
    width = length + BatchKVCache.step
    assert width + 128 <= capacity <= width + width // 8
    shapes = set()
    ref_shapes = set()
    for step in range(256):  # 1024 decode tokens
        k, v = _rnd((2, H, 4, D), 100 + step), _rnd((2, H, 4, D), 200 + step)
        got = batch.update_and_fetch(k, v)
        want = ref.update_and_fetch(k, v)
        _same(got[0], want[0])
        shapes.add(batch.keys.shape[2])
        ref_shapes.add(ref.keys.shape[2])
    # The concatenate path regrew (and copied every row) every 256 tokens;
    # the capacity buffer never reallocates inside the reserved span.
    assert len(ref_shapes) == 4
    assert shapes == {capacity}
    _assert_equivalent(batch, ref)


def test_decode_after_extend_writes_in_place():
    length = 16384
    batch = BatchKVCache.merge([_row(length - 900, 1), _row(length - 2000, 11)])
    ref = _ConcatBatchKV.merge([_row(length - 900, 1), _row(length - 2000, 11)])
    for step in range(3):
        k, v = _rnd((2, H, 4, D), 300 + step), _rnd((2, H, 4, D), 400 + step)
        batch.update_and_fetch(k, v)
        ref.update_and_fetch(k, v)
    # A prefilled singleton joins as an exact-width bank (``to_batch``).
    row = _row(length, 21)
    state = (*row.state, mx.array([length]), mx.array([0]))
    donor, ref_donor = BatchKVCache([0]), _ConcatBatchKV([0])
    donor.state = ref_donor.state = state
    batch.extend(donor)
    ref.extend(ref_donor)
    _assert_equivalent(batch, ref)
    # The join lands on the ladder directly, like a merge of the same rows.
    capacity = batch.keys.shape[2]
    width = length + BatchKVCache.step
    assert width <= capacity <= width + width // 8
    shapes = set()
    for step in range(64):
        k, v = _rnd((3, H, 4, D), 500 + step), _rnd((3, H, 4, D), 600 + step)
        _same(batch.update_and_fetch(k, v)[0], ref.update_and_fetch(k, v)[0])
        shapes.add(batch.keys.shape[2])
    assert shapes == {capacity}
    _assert_equivalent(batch, ref)


def test_capacity_growth_is_bounded_ratio():
    batch = BatchKVCache([0, 0])
    capacities = []
    total = 0
    for step in range(3000):
        k = _rnd((2, H, 16, D), step)
        batch.update_and_fetch(k, k)
        total += 16
        cap = int(batch.keys.shape[2])
        if not capacities or capacities[-1] != cap:
            capacities.append(cap)
        # Spare capacity (past the concatenate layout's <= 1 step of slack)
        # stays under max(2 * step, 1/8 of the length).
        step = BatchKVCache.step
        assert cap - batch._idx <= step + max(2 * step, (batch._idx + step) // 8)
    # 48K tokens: the concatenate path grew 188 times (once per 256 tokens).
    assert total // BatchKVCache.step == 187
    assert len(capacities) <= 60
    # Past 32K the grain is 2048 columns: >= 8x fewer reallocations.
    late = [c for c in capacities if c > 32768]
    assert len(late) <= 8


def test_ragged_finalize_rolls_only_the_logical_prefix():
    rows = [_row(300, 1), _row(290, 11)]
    batch = BatchKVCache.merge(rows)
    ref = _ConcatBatchKV.merge(rows)
    k, v = _rnd((2, H, 4, D), 5), _rnd((2, H, 4, D), 6)
    batch.update_and_fetch(k, v)
    ref.update_and_fetch(k, v)
    batch.trim(1)
    ref.trim(1)
    for c in (batch, ref):
        c.prepare(right_padding=[0, 2])
        c.finalize()
    _assert_equivalent(batch, ref)
    # The pad columns rotated in come from the concatenate-layout tail.
    _same(batch.state[0][1:2, :, :2], ref.state[0][1:2, :, :2])


def test_external_assignment_adopts_exact_width():
    batch = BatchKVCache([0])
    keys = _rnd((1, H, 10, D), 1)
    batch.state = (keys, keys, mx.array([10]), mx.array([0]))
    assert batch._width is None
    ref = _ConcatBatchKV([0])
    ref.state = (keys, keys, mx.array([10]), mx.array([0]))
    k = _rnd((1, H, 3, D), 2)
    batch.update_and_fetch(k, k)
    ref.update_and_fetch(k, k)
    _assert_equivalent(batch, ref)
