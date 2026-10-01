# SPDX-License-Identifier: Apache-2.0
"""Batched Qwen4 decode steps and MTP verify windows through gathered QSA.

Batch-one decode and verify attend only the QSA-selected K/V rows. As soon as
two requests shared a batch, every row fell back to a dense SDPA over the full
left-padded width behind a sparse boolean mask, so per-step cost grew with the
sum of all rows' contexts. These tests pin the batched gathered arm to the
dense masked path (same cache state, matching outputs), to the batch-one arm
(a row's output does not depend on its batch mates), and to its fail-closed
eligibility.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from test_qwen4_qsa_batch_row_banks import _config, _ragged_rollback

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat


@pytest.fixture(autouse=True)
def _vendored_qwen4():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()


def _language():
    compat.apply_mlx_vlm_qwen4_exp_compat_patch()
    import mlx_vlm.models.qwen4_exp.language as language

    return language


# budget 8 / ratio 2 -> sparse once a row holds more than 4 complete blocks
# (>= 10 tokens). The 5-token row stays below the crossover.
PREFIXES = (12, 23, 5)


class _Pair:
    """One attention layer and the same batch twice: gathered (fast) and the
    dense masked path (reference)."""

    def __init__(self, monkeypatch, prefixes=PREFIXES, seed=7, **config):
        self.monkeypatch = monkeypatch
        self.language = _language()
        self.config = _config(**config)
        self.attention = self.language.Qwen4ExpAttention(self.config)
        mx.eval(self.attention.parameters())
        mx.random.seed(seed)
        self.inputs = [
            mx.random.normal((1, n, self.config.hidden_size)) for n in prefixes
        ]
        self.fast = self.language.BatchQSAKVCache.merge(self.rows())
        self.reference = self.language.BatchQSAKVCache.merge(self.rows())
        self.calls = []
        original = self.language.Qwen4ExpAttention._gathered_batch

        def counted(attention, *args, **kwargs):
            self.calls.append(args[0].shape[1])
            return original(attention, *args, **kwargs)

        monkeypatch.setattr(self.language.Qwen4ExpAttention, "_gathered_batch", counted)

    def rows(self):
        rows = []
        for prefix in self.inputs:
            row = self.language.QSAKVCache()
            mx.eval(self.attention(prefix, mask="causal", cache=row))
            rows.append(row)
        return rows

    def forward(self, cache, x, gathered, target_verify=False, positions="none"):
        self.monkeypatch.setattr(
            self.language, "_GATHERED_BATCH_DISABLED", not gathered
        )
        batch, length = x.shape[:2]
        mask = self.language._create_qwen3_5_attention_mask(x, cache)
        position_ids = None
        if positions == "mrope":
            position_ids = mx.broadcast_to(
                cache.offset[None, :, None] + mx.arange(length)[None, None],
                (3, batch, length),
            )
        out = self.attention(
            x,
            mask=mask,
            cache=cache,
            position_ids=position_ids,
            target_verify=target_verify,
        )
        mx.eval(out)
        return out

    def step(self, length=1, target_verify=False, positions="none", tol=2e-5):
        batch = self.fast.offset.shape[0]
        x = mx.random.normal((batch, length, self.config.hidden_size))
        before = len(self.calls)
        actual = self.forward(self.fast, x, True, target_verify, positions)
        assert len(self.calls) == before + 1, "the gathered batch arm did not run"
        expected = self.forward(self.reference, x, False, target_verify, positions)
        assert len(self.calls) == before + 1
        assert (
            actual.shape == expected.shape == (batch, length, self.config.hidden_size)
        )
        assert mx.allclose(actual, expected, rtol=tol, atol=tol).item()
        self.assert_same_state()
        return actual

    def both(self, fn):
        fn(self.fast)
        fn(self.reference)
        self.assert_same_state()

    def assert_same_state(self):
        fast, reference = self.fast, self.reference
        assert fast._idx == reference._idx
        assert fast.index_offset == reference.index_offset
        assert fast.offset.tolist() == reference.offset.tolist()
        assert fast.left_padding.tolist() == reference.left_padding.tolist()
        for a, b in zip(fast.state[1:], reference.state[1:]):
            assert mx.array_equal(a, b).item()
        width = fast._idx
        for a, b in ((fast.keys, reference.keys), (fast.values, reference.values)):
            assert mx.array_equal(a[..., :width, :], b[..., :width, :]).item()


@pytest.mark.parametrize("positions", ["none", "mrope"])
def test_decode_verify_and_rollback_match_the_dense_path(monkeypatch, positions):
    pair = _Pair(monkeypatch)
    for _ in range(3):
        pair.step(positions=positions)
    pair.step(length=3, target_verify=True, positions=positions)
    pair.both(_ragged_rollback([3, 1, 2], 3))
    for _ in range(2):
        pair.step(positions=positions)
    pair.step(length=4, target_verify=True, positions=positions)
    pair.both(_ragged_rollback([1, 4, 2], 4))
    pair.both(lambda cache: cache.filter(mx.array([1, 2])))
    for _ in range(2):
        pair.step(positions=positions)
    assert pair.fast._omlx_last_prefill_gathered is True


@pytest.mark.parametrize("length", [1, 3])
def test_each_query_reads_at_most_budget_plus_tail_keys(monkeypatch, length):
    pair = _Pair(monkeypatch)
    widths = []
    original = mx.fast.scaled_dot_product_attention

    def tracked(queries, keys, values, **kwargs):
        widths.append(int(keys.shape[2]))
        return original(queries, keys, values, **kwargs)

    for _ in range(2):
        pair.step()
    batch = len(PREFIXES)
    x = mx.random.normal((batch, length, pair.config.hidden_size))
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", tracked)
    pair.forward(pair.fast, x, True, target_verify=length > 1)
    monkeypatch.setattr(mx.fast, "scaled_dot_product_attention", original)
    # budget 8 + tail 1, never the padded width (> 25 here).
    assert widths and max(widths) <= 8 + 1


def test_a_row_does_not_depend_on_its_batch_mates(monkeypatch):
    pair = _Pair(monkeypatch, seed=5)
    singles = pair.rows()
    x = mx.random.normal((len(PREFIXES), 1, pair.config.hidden_size))
    batched = pair.forward(pair.fast, x, True)
    for i, single in enumerate(singles):
        alone = pair.attention(x[i : i + 1], cache=single)
        assert mx.allclose(batched[i : i + 1], alone, rtol=2e-5, atol=2e-5).item()


def test_served_shape_matches_the_dense_path(monkeypatch):
    """ratio 4 / top-k 512 as served: the native selection and decode kernels."""

    pair = _Pair(monkeypatch, prefixes=(2100, 2600, 1900), seed=5, budget=2048, ratio=4)
    for _ in range(3):
        pair.step(tol=5e-5)
    pair.step(length=4, target_verify=True, tol=5e-5)
    pair.both(_ragged_rollback([2, 4, 1], 4))
    for _ in range(2):
        pair.step(tol=5e-5)


def test_kill_switches_keep_the_dense_path(monkeypatch):
    pair = _Pair(monkeypatch)
    language = pair.language
    x = mx.random.normal((len(PREFIXES), 1, pair.config.hidden_size))
    mask = language._create_qwen3_5_attention_mask(x, pair.fast)
    for switch in ("_GATHERED_BATCH_DISABLED", "_BATCH_ROW_BANKS_ENABLED"):
        monkeypatch.setattr(language, "_GATHERED_BATCH_DISABLED", False)
        monkeypatch.setattr(language, "_BATCH_ROW_BANKS_ENABLED", True)
        monkeypatch.setattr(language, switch, switch == "_GATHERED_BATCH_DISABLED")
        assert (
            pair.attention._gathered_batch_paddings(
                x, mask, pair.fast, None, None, False
            )
            is None
        )


def test_eligibility_fails_closed(monkeypatch):
    pair = _Pair(monkeypatch)
    language, attention, cache = pair.language, pair.attention, pair.fast
    monkeypatch.setattr(language, "_GATHERED_BATCH_DISABLED", False)
    batch = len(PREFIXES)
    decode = mx.zeros((batch, 1, pair.config.hidden_size))
    marker = language._create_qwen3_5_attention_mask(decode, cache)

    def paddings(
        x=decode, mask=marker, c=cache, positions=None, embeddings=None, verify=False
    ):
        return attention._gathered_batch_paddings(
            x, mask, c, positions, embeddings, verify
        )

    assert paddings() == cache.left_padding.tolist()
    # Narrow non-verify windows stay dense, exactly as for a batch-one row.
    narrow = mx.zeros((batch, 3, pair.config.hidden_size))
    assert paddings(x=narrow, mask=None) is None
    window_mask = language._create_qwen3_5_attention_mask(narrow, cache)
    assert paddings(x=narrow, mask=window_mask, verify=True)
    assert paddings(verify=True) is None
    # A caller-supplied mask of another shape (or an additive one) keeps its
    # meaning on the dense path; so do multimodal embeddings.
    assert paddings(x=narrow, mask=window_mask[..., 1:], verify=True) is None
    assert paddings(x=narrow, mask=window_mask.astype(mx.float16), verify=True) is None
    assert paddings(mask=mx.ones((batch, 1, 1, cache._idx + 1), dtype=mx.bool_)) is None
    assert paddings(embeddings=(mx.zeros((1,)), mx.zeros((1,)))) is None
    # Prefill-width windows keep the dense path.
    wide = mx.zeros(
        (batch, language._GATHERED_BATCH_MAX_QUERY + 1, pair.config.hidden_size)
    )
    assert paddings(x=wide, mask=None, verify=True) is None
    # Row-exact verify keeps its own arms.
    monkeypatch.setattr(language, "_row_exact_verify_armed", lambda: True)
    assert paddings(x=narrow, mask=window_mask, verify=True) is None
    monkeypatch.setattr(language, "_row_exact_verify_armed", lambda: False)
    # Singleton caches belong to the batch-one arms.
    assert paddings(x=decode[:1], mask=None, c=language.QSAKVCache()) is None
    # A misaligned indexer bank cannot be sliced per row.
    cache.index_offset = cache._idx - 1
    assert paddings() is None


def test_rows_below_the_crossover_keep_the_dense_path(monkeypatch):
    pair = _Pair(monkeypatch, prefixes=(5, 3))
    monkeypatch.setattr(pair.language, "_GATHERED_BATCH_DISABLED", False)
    x = mx.zeros((2, 1, pair.config.hidden_size))
    mask = pair.language._create_qwen3_5_attention_mask(x, pair.fast)
    assert (
        pair.attention._gathered_batch_paddings(x, mask, pair.fast, None, None, False)
        is None
    )
