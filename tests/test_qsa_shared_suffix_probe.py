# SPDX-License-Identifier: Apache-2.0
"""Serving-family attention and immutable-storage proofs."""

import importlib

import mlx.core as mx
import pytest
from test_qwen4_qsa_decode_gather import _tiny_text_config

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()
language = importlib.import_module("mlx_vlm.models.qwen4_exp.language")
shared = importlib.import_module("mlx_vlm.models.qwen4_exp.shared_suffix")


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("length", [1, 3, 4])
def test_shared_suffix_attention_and_rollback_match_materialized_reference(
    length, verify
):
    mx.random.seed(17)
    attention = language.Qwen4ExpAttention(_tiny_text_config(budget=8, ratio=2))
    source = language.QSAKVCache()
    prefix = mx.random.normal((1, 16, 32))
    mx.eval(attention(prefix, mask="causal", cache=source))
    base = shared.SharedQSAPrefix.freeze(source, attention.indexer)
    row = shared.SharedSuffixQSAKVCache(base)
    reference = language.QSAKVCache()
    reference.state = source.state
    original = [mx.array(x) for x in source.state]
    for step in range(3):
        hidden = mx.random.normal((1, length, 32))
        actual = attention(hidden, cache=row, target_verify=verify)
        expected = attention(
            hidden, mask="causal", cache=reference, target_verify=verify
        )
        mx.eval(actual, expected)
        assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
        assert row.gathered_tokens <= 9
        materialized = row.materialize()
        for got, want in zip(materialized.state, reference.state):
            assert mx.array_equal(got, want).item()
        assert row.offset == reference.offset
        for saved, live in zip(original, source.state):
            assert mx.array_equal(saved, live).item()
        if step == 1:
            row.trim(length)
            reference.trim(length)


def test_rows_keep_one_prefix_and_only_allocate_private_suffixes():
    config = _tiny_text_config(budget=8, ratio=2, max_positions=8192)
    attention = language.Qwen4ExpAttention(config)
    source = language.QSAKVCache()
    source.update_indexer(mx.zeros((1, 4096, 8)), mx.arange(4096)[None])
    source.update_and_fetch(mx.zeros((1, 2, 4096, 8)), mx.zeros((1, 2, 4096, 8)))
    base = shared.SharedQSAPrefix.freeze(source, attention.indexer)
    rows = [shared.SharedSuffixQSAKVCache(base) for _ in range(8)]
    for i, row in enumerate(rows):
        mx.eval(attention(mx.full((1, 1, 32), float(i) / 10), cache=row))
    assert all(row.base is base for row in rows)
    assert all(row.offset == 4097 and row.suffix.offset == 1 for row in rows)
    retained = base.nbytes + sum(row.private_nbytes for row in rows)
    # Compare against the actual upstream batch merge, counting full retained
    # source slabs conservatively. This is tensor capacity, not device peak.
    merged = language.BatchQSAKVCache.merge([source] * 8)
    assert base.nbytes == source.nbytes
    assert retained < merged.nbytes / 2
    rows[0].trim(1)
    assert rows[1].offset == 4097
    assert base.length == 4096


def test_probe_fails_closed_at_unsupported_boundaries():
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = language.QSAKVCache()
    mx.eval(attention(mx.zeros((1, 16, 32)), mask="causal", cache=source))
    row = shared.SharedSuffixQSAKVCache(
        shared.SharedQSAPrefix.freeze(source, attention.indexer)
    )
    with pytest.raises(ValueError, match="immutable"):
        row.trim(1)
    mx.eval(row.state)
    assert len(row.state) == 6  # evaluation state keeps prefix and suffix split
    with pytest.raises(ValueError, match="another"):
        other = language.Qwen4ExpAttention(_tiny_text_config())
        other(mx.zeros((1, 1, 32)), cache=row)
    with pytest.raises(ValueError, match="ordinary causal"):
        attention(mx.zeros((1, 1, 32)), cache=row, position_ids=mx.array([[30]]))


def test_frozen_prefix_survives_source_cache_reuse():
    attention = language.Qwen4ExpAttention(_tiny_text_config())
    source = language.QSAKVCache()
    mx.eval(attention(mx.ones((1, 16, 32)), mask="causal", cache=source))
    base = shared.SharedQSAPrefix.freeze(source, attention.indexer)
    saved = [
        mx.array(x) for x in (base.keys, base.values, base.raw_keys, base.positions)
    ]
    mx.eval(saved)
    source.trim(16)
    mx.eval(attention(mx.zeros((1, 16, 32)), mask="causal", cache=source))
    for original, frozen in zip(
        saved, (base.keys, base.values, base.raw_keys, base.positions)
    ):
        assert mx.array_equal(original, frozen).item()
