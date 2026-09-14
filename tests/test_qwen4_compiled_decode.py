# SPDX-License-Identifier: Apache-2.0
"""Equivalence tests for the mx.compile'd Qwen4-Exp (B, 1) decode step."""

from __future__ import annotations

import mlx.core as mx
import pytest

from omlx.patches import mlx_vlm_qwen4_exp_compat as compat

compat.apply_mlx_vlm_qwen4_exp_compat_patch()

from tests.test_mlx_vlm_qwen4_exp_compat import _tiny_config  # noqa: E402


def _make_lm():
    config = _tiny_config()
    config.text_config.linear_key_head_dim = 32
    config.text_config.linear_value_head_dim = 32
    mx.random.seed(7)
    from mlx_vlm.models import qwen4_exp

    model = qwen4_exp.Model(config)
    model.eval()
    # Mirror production load_weights: fuse the resident PLE table into the
    # packed device-side embedding (tiny shards are plain, so hand-pack).
    from mlx_vlm.models.qwen4_exp.language import fuse_resident_ple_embeddings

    if fuse_resident_ple_embeddings(model, minimum_physical_memory=0) == 0:
        import mlx.nn as nn

        for layer in model.language_model.model.layers:
            ple = getattr(layer, "ple", None)
            emb = getattr(
                getattr(ple, "ple_embedding", None), "ngram_embedding", None
            )
            if type(emb).__name__ == "ShardedEmbedding":
                emb.fused = nn.Embedding(emb.shard_offsets[-1], emb.dims)
    return model.language_model


def _build_caches(lm, batch):
    from mlx_vlm.models import qwen4_exp

    caches = []
    for layer in lm.model.layers:
        if layer.is_linear:
            cache = qwen4_exp.language.ArraysCache(
                size=4 if "ple" in layer else 2
            )
            cache.left_padding = mx.array([0] * batch)
            caches.append(cache)
        else:
            caches.append(qwen4_exp.language.QSAKVCache().to_batch([0] * batch))
    return caches


@pytest.mark.parametrize("batch", [1, 2])
def test_compiled_decode_matches_eager_greedy(batch):
    from mlx_vlm.models.qwen4_exp.compiled_decode import (
        Qwen4ExpCompiledDecodeLane,
    )

    lm = _make_lm()
    length, steps = 24, 16
    mx.random.seed(11)
    tokens = mx.random.uniform(0, 40, (batch, length)).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    eager_caches = _build_caches(lm, batch)
    logits = lm(tokens, cache=eager_caches).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    eager_tokens = []
    for _ in range(steps):
        logits = lm(tok[:, None], cache=eager_caches).logits
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        eager_tokens.append(tok.tolist())

    lm._position_ids = None
    lm._rope_deltas = None
    lane_caches = _build_caches(lm, batch)
    logits = lm(tokens, cache=lane_caches).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

    lane = Qwen4ExpCompiledDecodeLane(
        lm, lane_caches, cap=length + steps + 4
    )
    lane.seed(tok)
    lane_tokens = []
    for _ in range(steps):
        logits = lane.advance(tok)
        mx.eval(logits)
        tok = mx.argmax(logits, axis=-1).astype(mx.uint32)
        mx.eval(tok)
        lane_tokens.append(tok.tolist())

    assert lane_tokens == eager_tokens


def test_snapshot_accepts_left_padded_rows():
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp.compiled_decode import snapshot_from_caches

    lm = _make_lm()
    caches = []
    for layer in lm.model.layers:
        if layer.is_linear:
            cache = qwen4_exp.language.ArraysCache(
                size=4 if "ple" in layer else 2
            )
            cache.left_padding = mx.array([0, 0])
            caches.append(cache)
        else:
            caches.append(
                qwen4_exp.language.QSAKVCache().to_batch([0, 1])
            )
    tokens = mx.random.uniform(0, 40, (2, 12)).astype(mx.uint32)
    lm._position_ids = None
    lm._rope_deltas = None
    lm(tokens, cache=caches).logits
    pos_base, pads, write_col0, leaves = snapshot_from_caches(lm, caches, cap=64)
    assert pads.tolist() == [0, 1]
    assert pos_base.tolist() == [12, 11]


def _greedy(lm, caches, tokens, steps, feed):
    logits = lm(tokens, cache=caches).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    out = []
    for _ in range(steps):
        logits = lm(tok[:, None], cache=caches).logits
        mx.eval(logits)
        tok = feed(logits[:, -1, :])
        out.append(tok.tolist())
    return out


def test_filter_rows_matches_eager_filter():
    from mlx_vlm.models.qwen4_exp.compiled_decode import (
        Qwen4ExpCompiledDecodeLane,
    )

    lm = _make_lm()
    length, pre, post = 20, 6, 8
    mx.random.seed(5)
    tokens = mx.random.uniform(0, 40, (2, length)).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches = _build_caches(lm, 2)
    logits = lm(tokens, cache=caches).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    for _ in range(pre):
        logits = lm(tok[:, None], cache=caches).logits
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    # eager: drop row 0, keep row 1
    for cache in caches:
        cache.filter([1])
    tok = tok[1:2]
    eager = []
    for _ in range(post):
        logits = lm(tok[:, None], cache=caches).logits
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        eager.append(tok.tolist())

    lm._position_ids = None
    lm._rope_deltas = None
    caches2 = _build_caches(lm, 2)
    logits = lm(tokens, cache=caches2).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    lane = Qwen4ExpCompiledDecodeLane(
        lm, caches2, cap=length + pre + post + 8, compiled=False
    )
    lane.seed(tok)
    lane_tokens = []
    for _ in range(pre):
        lg = lane.advance(tok)
        mx.eval(lg)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
    lane.filter_rows([1])
    tok = tok[1:2]
    for _ in range(post):
        lg = lane.advance(tok)
        mx.eval(lg)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)
        lane_tokens.append(tok.tolist())
    assert lane_tokens == eager


def test_extend_rows_equal_width_matches_eager():
    from mlx_vlm.models import qwen4_exp
    from mlx_vlm.models.qwen4_exp.compiled_decode import (
        Qwen4ExpCompiledDecodeLane,
    )

    lm = _make_lm()
    length, pre, post = 20, 4, 8
    mx.random.seed(6)
    tokens = mx.random.uniform(0, 40, (1, length)).astype(mx.uint32)
    # Join at the batch's CURRENT width (length + pre) so the join is
    # unpadded and both paths run the identical uniform algorithm -- the
    # padded join has its own tolerance test below.
    tokens_b = mx.random.uniform(0, 40, (1, length + pre)).astype(mx.uint32)
    joined_first_token = mx.random.uniform(0, 40, (1,)).astype(mx.uint32)

    # eager: row 0 decodes `pre` steps alone, then row B (same width) joins
    lm._position_ids = None
    lm._rope_deltas = None
    caches_a = _build_caches(lm, 1)
    logits = lm(tokens, cache=caches_a).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    for _ in range(pre):
        logits = lm(tok[:, None], cache=caches_a).logits
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches_b = _build_caches(lm, 1)
    lm(tokens_b, cache=caches_b).logits

    for cache_a, cache_b in zip(caches_a, caches_b):
        cache_a.extend(cache_b)
    eager = []
    tok_b = mx.array(joined_first_token)
    for _ in range(post):
        logits = lm(
            mx.concatenate([tok, tok_b])[:, None], cache=caches_a
        ).logits
        mx.eval(logits)
        both = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        tok, tok_b = both[0:1], both[1:2]
        eager.append(both.tolist())

    # lane: same story
    lm._position_ids = None
    lm._rope_deltas = None
    caches_a2 = _build_caches(lm, 1)
    logits = lm(tokens, cache=caches_a2).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    lane = Qwen4ExpCompiledDecodeLane(
        lm, caches_a2, cap=length + pre + post + 8, compiled=False
    )
    lane.seed(tok)
    for _ in range(pre):
        lg = lane.advance(tok)
        mx.eval(lg)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches_b2 = _build_caches(lm, 1)
    lm(tokens_b, cache=caches_b2).logits

    lane.extend_rows([caches_b2])
    tok_b = mx.array(joined_first_token)
    lane_tokens = []
    for _ in range(post):
        lg = lane.advance(mx.concatenate([tok, tok_b]))
        mx.eval(lg)
        both = mx.argmax(lg, axis=-1).astype(mx.uint32)
        tok, tok_b = both[0:1], both[1:2]
        lane_tokens.append(both.tolist())
    assert lane_tokens == eager


def test_extend_rows_padded_join_row0_exact_row1_close():
    """A shorter row joins left-padded: eager serves it through the dense
    per-row ragged kernel while the lane masks a capacity buffer -- equal
    admitted sets, different accumulation order.  The unpadded row must
    stay token-exact; the padded row may drift on near-ties."""
    from mlx_vlm.models.qwen4_exp.compiled_decode import (
        Qwen4ExpCompiledDecodeLane,
    )

    lm = _make_lm()
    length, pre, post = 20, 4, 8
    mx.random.seed(9)
    tokens = mx.random.uniform(0, 40, (1, length)).astype(mx.uint32)
    tokens_b = mx.random.uniform(0, 40, (1, length)).astype(mx.uint32)
    joined_first_token = mx.random.uniform(0, 40, (1,)).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches_a = _build_caches(lm, 1)
    logits = lm(tokens, cache=caches_a).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    for _ in range(pre):
        logits = lm(tok[:, None], cache=caches_a).logits
        mx.eval(logits)
        tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches_b = _build_caches(lm, 1)
    lm(tokens_b, cache=caches_b).logits

    for cache_a, cache_b in zip(caches_a, caches_b):
        cache_a.extend(cache_b)
    eager = []
    tok_b = mx.array(joined_first_token)
    for _ in range(post):
        logits = lm(
            mx.concatenate([tok, tok_b])[:, None], cache=caches_a
        ).logits
        mx.eval(logits)
        both = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
        tok, tok_b = both[0:1], both[1:2]
        eager.append(both.tolist())

    lm._position_ids = None
    lm._rope_deltas = None
    caches_a2 = _build_caches(lm, 1)
    logits = lm(tokens, cache=caches_a2).logits
    mx.eval(logits)
    tok = mx.argmax(logits[:, -1, :], axis=-1).astype(mx.uint32)
    lane = Qwen4ExpCompiledDecodeLane(
        lm, caches_a2, cap=length + pre + post + 8, compiled=False
    )
    lane.seed(tok)
    for _ in range(pre):
        lg = lane.advance(tok)
        mx.eval(lg)
        tok = mx.argmax(lg, axis=-1).astype(mx.uint32)

    lm._position_ids = None
    lm._rope_deltas = None
    caches_b2 = _build_caches(lm, 1)
    lm(tokens_b, cache=caches_b2).logits

    lane.extend_rows([caches_b2])
    assert lane.pads.tolist() == [0, pre]
    tok_b = mx.array(joined_first_token)
    lane_tokens = []
    for _ in range(post):
        lg = lane.advance(mx.concatenate([tok, tok_b]))
        mx.eval(lg)
        both = mx.argmax(lg, axis=-1).astype(mx.uint32)
        tok, tok_b = both[0:1], both[1:2]
        lane_tokens.append(both.tolist())

    # The unpadded original row keeps the exact token stream.
    assert [row[0] for row in lane_tokens] == [row[0] for row in eager]
    # The padded row usually matches too; require at least the first steps
    # before accumulation drift can reach a near-tie.
    same = sum(
        1 for lane_row, eager_row in zip(lane_tokens, eager)
        if lane_row[1] == eager_row[1]
    )
    assert same >= 3, f"padded row matched only {same}/{post} steps"
