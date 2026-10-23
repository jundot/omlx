# SPDX-License-Identifier: Apache-2.0
"""MiniMax M3 sparse attention block selection and attention paths."""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

BLOCK = 4
TOPK = 3


@pytest.fixture(scope="module")
def language():
    from omlx.patches.mlx_vlm_minimax_m3_compat import (
        apply_mlx_vlm_minimax_m3_compat_patch,
    )

    apply_mlx_vlm_minimax_m3_compat_patch()
    import mlx_vlm.models.minimax_m3_vl.language as language

    return language


def _reference_blocks(idx_q, idx_k, qpos, pad):
    """Blocks per (row, index head, query) in each row's own positions."""
    batch, heads, length, _ = idx_q.shape
    result = {}
    for b in range(batch):
        for h in range(heads):
            for t in range(length):
                p = int(qpos[b, t])
                cur = p // BLOCK
                scores = []
                for blk in range(cur + 1):
                    keys = range(blk * BLOCK, min((blk + 1) * BLOCK, p + 1))
                    best = max(
                        float(idx_q[b, h, t] @ idx_k[b, 0, pad[b] + j]) for j in keys
                    )
                    scores.append(1e29 if blk == cur else best)
                order = np.argsort(-np.asarray(scores), kind="stable")
                result[(b, h, t)] = sorted(order[:TOPK].tolist())
    return result


def _reference_attention(q, k, v, blocks, qpos, pad, scale, keep=None, block=BLOCK):
    batch, heads, length, _ = q.shape
    kv_heads = k.shape[1]
    out = np.zeros_like(q)
    for b in range(batch):
        for qh in range(heads):
            h = qh // (heads // kv_heads)
            for t in range(length):
                p = int(qpos[b, t])
                cols = [
                    pad[b] + j
                    for blk in blocks[(b, h, t)]
                    for j in range(blk * block, (blk + 1) * block)
                    if j <= p and (keep is None or keep[b, 0, t, pad[b] + j])
                ]
                logits = scale * (k[b, h, cols] @ q[b, qh, t])
                weights = np.exp(logits - logits.max())
                out[b, qh, t] = (weights / weights.sum()) @ v[b, h, cols]
    return out


def _inputs(batch=2, heads=4, kv_heads=2, length=3, total=40, dim=8, pad=(5, 0)):
    rng = np.random.default_rng(0)
    idx_q = rng.standard_normal((batch, kv_heads, length, dim)).astype(np.float32)
    idx_k = rng.standard_normal((batch, 1, total, dim)).astype(np.float32)
    q = rng.standard_normal((batch, heads, length, dim)).astype(np.float32)
    k = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    v = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    lengths = [total - p for p in pad]
    qpos = np.stack([np.arange(n - length, n) for n in lengths]).astype(np.int32)
    return idx_q, idx_k, q, k, v, qpos, np.asarray(pad, dtype=np.int32)


def test_block_selection_is_per_kv_head_in_row_positions(language):
    idx_q, idx_k, _, _, _, qpos, pad = _inputs()

    block_ids = language._sparse_block_selection(
        mx.array(idx_q),
        mx.array(idx_k),
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        TOPK,
        0,
        1,
    )

    expected = _reference_blocks(idx_q, idx_k, qpos, pad)
    actual = np.sort(np.array(block_ids), axis=-1)
    for (b, h, t), blocks in expected.items():
        assert actual[b, h, t].tolist() == blocks
    # The two index heads choose differently somewhere, so a shared
    # selection across heads would not match the reference.
    assert any(
        expected[(b, 0, t)] != expected[(b, 1, t)] for b in range(2) for t in range(3)
    )


def test_compact_and_masked_paths_match_reference(language):
    idx_q, idx_k, q, k, v, qpos, pad = _inputs()
    scale = 8**-0.5
    block_ids = language._sparse_block_selection(
        mx.array(idx_q),
        mx.array(idx_k),
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        TOPK,
        0,
        1,
    )
    expected = _reference_attention(
        q, k, v, _reference_blocks(idx_q, idx_k, qpos, pad), qpos, pad, scale
    )

    compact = language._compact_sparse_attention(
        mx.array(q),
        mx.array(k),
        mx.array(v),
        block_ids,
        mx.array(qpos),
        mx.array(pad),
        BLOCK,
        scale,
    )
    token_mask = language._sparse_token_mask(
        block_ids, mx.array(qpos), mx.array(pad), k.shape[2], BLOCK
    )
    masked = language._masked_sparse_attention(
        mx.array(q), mx.array(k), mx.array(v), token_mask, scale
    )

    np.testing.assert_allclose(np.array(compact), expected, atol=1e-5)
    np.testing.assert_allclose(np.array(masked), expected, atol=1e-5)


def _tiny_args(sparse=None, **overrides):
    from mlx_vlm.models.minimax_m3_vl.config import TextConfig

    fields = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=1,
        moe_layer_freq=[0],
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_attention_freq": [1],
            "sparse_index_dim": 16,
            "sparse_num_index_heads": 2,
            "sparse_block_size": 8,
            "sparse_topk_blocks": 2,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
            **(sparse or {}),
        },
    )
    return TextConfig(**{**fields, **overrides})


def _tiny_attention(language):
    return language.MiniMaxAttention(_tiny_args(), 0)


def test_long_chunk_drops_positions_that_restate_the_cache_offset(
    language, monkeypatch
):
    seen = []
    can_use = language.MiniMaxAttention._can_use_msa_prefill_attention

    def spy(self, queries, keys, mask, q_positions):
        seen.append(q_positions)
        return can_use(self, queries, keys, mask, q_positions)

    monkeypatch.setattr(
        language.MiniMaxAttention, "_can_use_msa_prefill_attention", spy
    )
    mx.random.seed(5)
    model = language.LanguageModel(
        _tiny_args(vocab_size=97, dense_intermediate_size=32)
    )
    tokens = mx.random.randint(0, 97, (1, 96))
    cache = model.make_cache()
    model(tokens[:, :32], cache=cache)

    model(tokens[:, 32:64], cache=cache, position_ids=mx.arange(32, 64)[None])
    assert seen[-1] is None
    model(tokens[:, 64:], cache=cache, position_ids=mx.arange(70, 102)[None])
    assert seen[-1] is not None


def test_left_padded_batch_decode_matches_single_rows(language):
    mx.random.seed(3)
    attention = _tiny_attention(language)
    lengths = (37, 50)
    prompts = [mx.random.normal((1, n, 64)) for n in lengths]
    steps = [mx.random.normal((1, 1, 64)) for _ in lengths]

    singles, caches = [], []
    for prompt, step, n in zip(prompts, steps, lengths):
        cache = language.MiniMaxM3KVCache()
        attention(prompt, mask="causal", cache=cache)
        singles.append(attention(step, cache=cache, position_ids=mx.array([[n]])))
        cache.trim(1)
        caches.append(cache)

    batch = language.MiniMaxM3BatchKVCache.merge(caches)
    assert batch.left_padding.tolist() == [13, 0]
    batched = attention(
        mx.concatenate(steps, axis=0),
        mask=batch.make_mask(1),
        cache=batch,
        position_ids=mx.array([[lengths[0]], [lengths[1]]]),
    )

    for row, single in enumerate(singles):
        np.testing.assert_allclose(
            np.array(batched[row]), np.array(single[0]), atol=1e-5
        )


@pytest.mark.parametrize("mixed", [False, True])
def test_short_calls_fuse_input_projections_bit_exactly(language, monkeypatch, mixed):
    mx.random.seed(5)
    attention = _tiny_attention(language)

    def spec(path, module):
        if not isinstance(module, nn.Linear):
            return False
        if mixed and path in ("q_proj", "k_proj", "v_proj"):
            return {"group_size": 32, "bits": 5, "mode": "affine"}
        return {"group_size": 32, "bits": 8, "mode": "mxfp8"}

    nn.quantize(attention, class_predicate=spec)
    prompt = mx.random.normal((1, 40, 64))
    steps = [mx.random.normal((1, n, 64)) for n in (1, 4, 8)]

    def run():
        cache = language.MiniMaxM3KVCache()
        outs = [attention(prompt, mask="causal", cache=cache)]
        for step in steps:
            outs.append(attention(step, cache=cache))
            cache.trim(step.shape[1])
        mx.eval(outs)
        return outs

    monkeypatch.setattr(language, "_FUSED_PROJECTION_MAX_ROWS", 0)
    separate = run()
    assert "_omlx_fused_projections" not in attention.__dict__
    monkeypatch.undo()
    fused = run()

    groups = attention.__dict__["_omlx_fused_projections"]
    names = [group.names for group in groups]
    if mixed:
        assert names == [
            ("q_proj", "k_proj", "v_proj"),
            ("index_q_proj", "index_k_proj"),
        ]
    else:
        assert names == [language._INPUT_PROJECTIONS]
    for a, b in zip(separate, fused):
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("mode", ["mxfp8", "affine"])
def test_short_calls_fuse_mlp_gate_and_up_bit_exactly(language, monkeypatch, mode):
    mx.random.seed(11)
    mlp = language.MiniMaxMLP(512, 256)
    quant = {"group_size": 32, "bits": 8, "mode": mode}
    nn.quantize(
        mlp, class_predicate=lambda p, m: quant if isinstance(m, nn.Linear) else False
    )
    xs = [mx.random.normal((1, n, 512)).astype(mx.bfloat16) for n in (1, 4, 8)]

    monkeypatch.setattr(language, "_FUSED_PROJECTION_MAX_ROWS", 0)
    separate = [mlp(x) for x in xs]
    mx.eval(separate)
    monkeypatch.undo()
    fused = [mlp(x) for x in xs]
    mx.eval(fused)

    groups = mlp.__dict__["_omlx_fused_projections"]
    assert [group.names for group in groups] == [("gate_proj", "up_proj")]
    for a, b in zip(separate, fused):
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("rows", [3, 8])
def test_short_row_mxfp8_matmul_matches_one_row_decode(language, rows):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    mx.random.seed(6)
    proj = nn.QuantizedLinear.from_linear(
        nn.Linear(512, 64, bias=False), group_size=32, bits=8, mode="mxfp8"
    )
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([proj(x[:, i : i + 1]) for i in range(rows)], axis=1)
    assert mx.array_equal(qmv_rows.rows_linear(proj, x), expected).item()


def test_row_exact_verify_router_and_mxfp8_match_one_row_decode(language):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(8)
    rows = 2
    gate = nn.Linear(6144, 128, bias=False)
    proj = nn.QuantizedLinear.from_linear(
        nn.Linear(512, 64, bias=False), group_size=32, bits=8, mode="mxfp8"
    )
    x = mx.random.normal((1, rows, 6144)).astype(mx.bfloat16)
    small = x[..., :512]
    expected_gates = mx.concatenate(
        [gate(x[:, i : i + 1].astype(mx.float32)) for i in range(rows)], axis=1
    )
    expected_proj = mx.concatenate(
        [proj(small[:, i : i + 1]) for i in range(rows)], axis=1
    )
    set_verify_qmm_armed(True, row_exact=True)
    try:
        gates = language._router_gates(gate, x)
        projected = qmv_rows.rows_linear(proj, small)
        mx.eval(gates, projected)
    finally:
        set_verify_qmm_armed(False)
    assert mx.array_equal(gates, expected_gates).item()
    assert mx.array_equal(projected, expected_proj).item()


@pytest.mark.parametrize("rows", [2, 8])
def test_row_exact_verify_dense_head_matches_one_row_decode(language, rows):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(10)
    head = nn.Linear(6144, 4096, bias=False)
    head.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, rows, 6144)).astype(mx.bfloat16)
    expected = mx.concatenate([head(x[:, i : i + 1]) for i in range(rows)], axis=1)
    set_verify_qmm_armed(True, row_exact=True)
    try:
        projected = qmv_rows.rows_linear(head, x)
        mx.eval(projected)
    finally:
        set_verify_qmm_armed(False)
    assert mx.array_equal(projected, expected).item()


@pytest.mark.parametrize("prompt", [30, 60])
def test_row_exact_verify_attention_matches_one_row_decode(language, prompt):
    """Six verify rows of 16 query heads per KV head exceed MLX's vector SDPA
    rows. Up to 64 keys the attention is dense; a 60-token prompt puts the
    verify across that threshold. Quantized projections keep the projections
    row-exact, so only the attention can differ."""
    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(1)
    args = _tiny_args(
        sparse={"sparse_block_size": 32, "sparse_index_dim": 64},
        hidden_size=512,
        num_attention_heads=32,
        head_dim=64,
    )
    attention = language.MiniMaxAttention(args, 0)
    attention.set_dtype(mx.bfloat16)
    nn.quantize(attention, group_size=64, bits=8)
    rows = 6
    x = mx.random.normal((1, prompt + rows, 512)).astype(mx.bfloat16)
    cache = language.MiniMaxM3KVCache()
    attention(x[:, :prompt], mask="causal", cache=cache)
    steps = [
        attention(
            x[:, prompt + i : prompt + i + 1],
            cache=cache,
            position_ids=mx.array([[prompt + i]]),
        )
        for i in range(rows)
    ]
    cache.trim(rows)
    set_verify_qmm_armed(True, row_exact=True)
    try:
        verify = attention(
            x[:, prompt:],
            mask="causal",
            cache=cache,
            position_ids=mx.arange(prompt, prompt + rows)[None],
        )
        mx.eval(verify)
    finally:
        set_verify_qmm_armed(False)
    assert mx.array_equal(verify, mx.concatenate(steps, axis=1)).item()


@pytest.mark.parametrize("columns", [96, 65536])
@pytest.mark.parametrize("rows", [5, 7])
def test_row_exact_verify_affine_projection_matches_one_row_decode(
    language, rows, columns
):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(9)
    linear = nn.Linear(512, columns, bias=False)
    linear.set_dtype(mx.bfloat16)
    proj = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=5)
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([proj(x[:, i : i + 1]) for i in range(rows)], axis=1)
    set_verify_qmm_armed(True, row_exact=True)
    try:
        projected = qmv_rows.rows_linear(proj, x)
        mx.eval(projected)
    finally:
        set_verify_qmm_armed(False)
    assert mx.array_equal(projected, expected).item()


@pytest.mark.parametrize("rows", [1, 8])
def test_short_row_nvfp4_experts_match_stock_switch(language, monkeypatch, rows):
    import mlx_vlm.models.minimax_m3_vl.moe_rows as moe_rows

    mx.random.seed(7)
    experts = 16
    switch = language.MiniMaxScaledSwitchGLU(
        512, 512, experts, language.MiniMaxSwiGLUOAI()
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        linear = switch[name]
        linear.weight = mx.random.normal(linear.weight.shape) * 0.05
        switch[name] = linear.to_quantized(group_size=16, bits=4, mode="nvfp4")
    switch.gate_scale = mx.random.uniform(0.5, 30.0, (experts,))
    switch.up_scale = mx.random.uniform(0.5, 30.0, (experts,))
    x = (mx.random.normal((1, rows, 512)) * 2).astype(mx.bfloat16)
    # Every other row repeats the first row's experts, so shared experts are
    # computed once for several rows.
    first = mx.random.permutation(experts)[:4]
    picks = [
        first if r % 2 == 0 else mx.random.permutation(experts)[:4] for r in range(rows)
    ]
    indices = mx.stack(picks)[None].astype(mx.uint32)

    fused = switch(x, indices)
    monkeypatch.setattr(moe_rows, "MAX_ROWS", 0)
    stock = switch(x, indices)
    assert mx.array_equal(fused, stock).item()


@pytest.mark.parametrize("length", [1, 8])
def test_small_l_block_score_kernel_matches_numpy(language, length):
    import mlx_vlm.models.minimax_m3_vl.msa as msa

    batch, heads, dim, total, block = 2, 4, 128, 700, 32
    pad = np.array([37, 0], dtype=np.int32)
    rng = np.random.default_rng(1)
    idx_q = rng.standard_normal((batch, heads, length, dim)).astype(np.float32)
    keys = rng.standard_normal((batch, 1, total + 64, dim)).astype(np.float32)
    qpos = np.stack([np.arange(total - p - length, total - p) for p in pad]).astype(
        np.int32
    )
    q_mx = mx.array(idx_q).astype(mx.bfloat16)
    k_mx = mx.array(keys).astype(mx.bfloat16)[:, :, :total]

    scores = msa.small_l_block_scores(
        q_mx, k_mx, mx.array(qpos), mx.array(pad), block, 0, 1
    )
    assert scores is not None
    scores = -np.array(scores)

    q32 = np.array(q_mx.astype(mx.float32))
    k32 = np.array(k_mx.astype(mx.float32))
    num_blocks = -(-total // block)
    for b in range(batch):
        for h in range(heads):
            for t in range(length):
                p = int(qpos[b, t])
                for blk in range(num_blocks):
                    got = scores[b, h, t, blk]
                    if blk > p // block:
                        assert got == -np.inf
                    elif blk == p // block:
                        assert got == 1e29
                    else:
                        cols = [
                            j + pad[b]
                            for j in range(blk * block, (blk + 1) * block)
                            if j + pad[b] < total
                        ]
                        expected = max(float(q32[b, h, t] @ k32[b, 0, c]) for c in cols)
                        assert abs(got - expected) <= 1e-3 * max(1.0, abs(expected))


def test_small_l_sparse_attention_kernel_matches_reference(language):
    import mlx_vlm.models.minimax_m3_vl.msa as msa

    batch, heads, kv_heads, length, dim = 2, 8, 2, 8, 128
    total, block, topk = 700, 32, 4
    pad = np.array([37, 0], dtype=np.int32)
    rng = np.random.default_rng(2)
    q, k, v = (
        mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
        for shape in (
            (batch, heads, length, dim),
            (batch, kv_heads, total, dim),
            (batch, kv_heads, total, dim),
        )
    )
    keep = rng.random((batch, 1, length, total)) > 0.1
    qpos = np.stack([np.arange(total - p - length, total - p) for p in pad]).astype(
        np.int32
    )
    block_ids = mx.array(
        np.stack(
            [rng.permutation(qpos[b, -1] // block + 1)[:topk] for b in range(batch)]
        )[:, None, None, :]
        .repeat(kv_heads, 1)
        .repeat(length, 2),
        dtype=mx.int32,
    )
    scale = dim**-0.5

    out = msa.small_l_sparse_attention(
        q, k, v, block_ids, mx.array(qpos), mx.array(pad), block, scale, mx.array(keep)
    )
    assert out is not None

    q32, k32, v32 = (np.array(x.astype(mx.float32)) for x in (q, k, v))
    expected = _reference_attention(
        q32, k32, v32, np.array(block_ids), qpos, pad, scale, keep, block
    )
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)), expected, atol=2e-2, rtol=2e-2
    )

    # A verify row computes exactly like a one-token decode step, whatever
    # order the selection returns the blocks in.
    last = msa.small_l_sparse_attention(
        q[:, :, -1:],
        k,
        v,
        block_ids[:, :, -1:, ::-1],
        mx.array(qpos[:, -1:]),
        mx.array(pad),
        block,
        scale,
        mx.array(keep[:, :, -1:]),
    )
    assert mx.array_equal(last, out[:, :, -1:]).item()
