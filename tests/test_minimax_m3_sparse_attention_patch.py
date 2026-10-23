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


@pytest.mark.parametrize("rows", [2, 3, 8])
def test_short_row_mxfp8_matmul_matches_one_row_decode(language, monkeypatch, rows):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    mx.random.seed(6)
    proj = nn.QuantizedLinear.from_linear(
        nn.Linear(512, 64, bias=False), group_size=32, bits=8, mode="mxfp8"
    )
    x = mx.random.normal((1, rows, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([proj(x[:, i : i + 1]) for i in range(rows)], axis=1)
    # MLX's multi-row kernel splits K differently and rounds a few outputs
    # in ten thousand differently, so check the call takes the exact rows.
    exact = qmv_rows.exact_rows_qmv
    calls = []
    monkeypatch.setattr(
        qmv_rows, "exact_rows_qmv", lambda *a, **k: calls.append(1) or exact(*a, **k)
    )
    assert mx.array_equal(qmv_rows.rows_linear(proj, x), expected).item()
    assert calls


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


@pytest.mark.parametrize("armed", [True, False])
@pytest.mark.parametrize("rows", [2, 8])
def test_row_exact_verify_dense_head_matches_one_row_decode(language, rows, armed):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(10)
    head = nn.Linear(6144, 4096, bias=False)
    head.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, rows, 6144)).astype(mx.bfloat16)
    expected = mx.concatenate([head(x[:, i : i + 1]) for i in range(rows)], axis=1)
    # A decode batch (not armed) takes the one-row arithmetic too.
    set_verify_qmm_armed(armed, row_exact=armed)
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


def test_small_l_sparse_attention_kernel_matches_reference(language, monkeypatch):
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

    # Spreading the query heads over more threads changes no bit (here out
    # has one thread per key for all four heads of a KV head).
    for heads_per_thread in (1, 2):
        monkeypatch.setattr(msa, "_SMALL_L_HEADS_PER_THREAD", heads_per_thread)
        spread = msa.small_l_sparse_attention(
            q,
            k,
            v,
            block_ids,
            mx.array(qpos),
            mx.array(pad),
            block,
            scale,
            mx.array(keep),
        )
        assert mx.array_equal(spread, out).item()
    monkeypatch.undo()

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


@pytest.mark.parametrize(
    "shape", [(1, 1, 6144), (1, 7, 6144), (8, 128), (2, 1, 4, 128), (5, 4100)]
)
@pytest.mark.parametrize("out", ["model", "float32"])
def test_decode_rms_norm_matches_mlx(language, shape, out):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(11)
    x = (mx.random.normal(shape) * 3).astype(mx.bfloat16)
    scale = 1.0 + mx.random.normal((shape[-1],)) * 0.3
    dtype = x.dtype if out == "model" else mx.float32
    y = dk.rms_norm(x, scale, 1e-6, dtype)
    assert y is not None
    expected = mx.fast.rms_norm(x, scale, 1e-6).astype(dtype)
    assert mx.array_equal(y, expected).item()


def test_decode_rms_norm_leaves_longer_calls_to_mlx(language):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    x = mx.zeros((9, 128), dtype=mx.bfloat16)
    assert dk.rms_norm(x, mx.ones((128,)), 1e-6, mx.bfloat16) is None


def test_rms_norm_caches_its_fp32_scale(language):
    from mlx.utils import tree_flatten

    norm = language.MiniMaxRMSNorm(64, eps=1e-6)
    norm.weight = (mx.random.normal((64,)) * 0.3).astype(mx.bfloat16)
    x = (mx.random.normal((2, 64)) * 3).astype(mx.bfloat16)
    expected = mx.fast.rms_norm(x, norm.weight.astype(mx.float32) + 1, 1e-6)
    assert mx.array_equal(norm(x), expected.astype(mx.bfloat16)).item()
    assert mx.array_equal(norm(x, dtype=mx.float32), expected).item()
    scale = norm._scale()
    assert norm._scale() is scale
    assert set(dict(tree_flatten(norm.parameters()))) == {"weight"}
    # Loading a new weight array rebuilds the scale.
    norm.weight = mx.zeros((64,), dtype=mx.bfloat16)
    assert mx.array_equal(norm._scale(), mx.ones((64,))).item()


def _head_inputs(batch, length):
    heads = (64, 4, 4, 1)  # M3.1-Flash q, k, index q, index k
    xs = [
        (mx.random.normal((batch, length, h, 128)) * 3).astype(mx.bfloat16)
        for h in heads
    ]
    return xs, [1.0 + mx.random.normal((128,)) * 0.3 for _ in heads]


@pytest.mark.parametrize(
    "batch,length,offset",
    [
        (1, 1, 4097),
        (1, 1, [-3]),
        (2, 1, [37, 50]),
        (1, 4, 300),
        # A left-padded prefill puts its padding at negative positions.
        (2, 3, [-2, 5]),
        (8, 1, [0, 1, 2, 3, 4, 5, 6, 7]),
    ],
)
@pytest.mark.parametrize("dims,rope_scale", [(64, 1.0), (128, 1.0), (64, 0.25)])
def test_decode_head_norm_rope_matches_mlx(
    language, batch, length, offset, dims, rope_scale
):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(13)
    xs, scales = _head_inputs(batch, length)
    if isinstance(offset, list):
        offset = mx.array(offset)
    rotated = dk.head_norm_rope(xs, scales, 1e-6, offset, dims, 1e7, rope_scale)
    assert rotated is not None
    for x, scale, y in zip(xs, scales, rotated):
        normed = mx.fast.rms_norm(x, scale, 1e-6).transpose(0, 2, 1, 3)
        expected = mx.fast.rope(
            normed, dims, traditional=False, base=1e7, scale=rope_scale, offset=offset
        ).astype(x.dtype)
        assert mx.array_equal(y, expected).item()


def test_decode_head_norm_rope_coverage(language):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    def covered(xs, scales, offset, dims):
        return dk.head_norm_rope(xs, scales, 1e-6, offset, dims, 1e7, 1.0) is not None

    xs, scales = _head_inputs(2, 4)
    assert covered(xs, scales, mx.array([3, 0]), 64)
    assert not covered(_head_inputs(3, 3)[0], scales, 0, 64)
    assert not covered(xs, scales, mx.array([1, 2, 3]), 64)
    assert not covered(xs, scales, 0, 48)
    assert not covered(xs, [s.astype(mx.bfloat16) for s in scales], 0, 64)


def test_short_calls_fuse_head_norms_and_rope_bit_exactly(language, monkeypatch):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(9)
    args = _tiny_args(
        head_dim=128, rotary_dim=64, rope_theta=1e7, sparse={"sparse_index_dim": 128}
    )
    attention = language.MiniMaxAttention(args, 0)
    for name in ("q_norm", "k_norm", "index_q_norm", "index_k_norm"):
        attention[name].weight = mx.random.normal((128,)) * 0.3
    attention.set_dtype(mx.bfloat16)
    prompt = mx.random.normal((1, 40, 64)).astype(mx.bfloat16)
    steps = [mx.random.normal((1, n, 64)).astype(mx.bfloat16) for n in (1, 3, 8)]

    def run():
        outs, caches = [], []
        for n in (37, 40):
            cache = language.MiniMaxM3KVCache()
            outs.append(attention(prompt[:, :n], mask="causal", cache=cache))
            for step in steps:
                outs.append(attention(step, cache=cache))
                cache.trim(step.shape[1])
            caches.append(cache)
        batch = language.MiniMaxM3BatchKVCache.merge(caches)
        outs.append(
            attention(
                mx.concatenate([steps[0], steps[0]]),
                mask=batch.make_mask(1),
                cache=batch,
                position_ids=mx.array([[37], [40]]),
            )
        )
        mx.eval(outs)
        return outs

    calls = []
    head_norm_rope = dk.head_norm_rope

    def spy(xs, *args):
        calls.append(xs[0].shape[:2])
        return head_norm_rope(xs, *args)

    monkeypatch.setattr(dk, "head_norm_rope", spy)
    fused = run()
    assert calls == [(1, 1), (1, 3), (1, 8)] * 2 + [(2, 1)]
    monkeypatch.setattr(dk, "head_norm_rope", lambda *args: None)
    separate = run()
    for a, b in zip(separate, fused):
        assert mx.array_equal(a, b).item()


@pytest.mark.parametrize("rows", [1, 3, 8])
def test_decode_router_logits_match_one_row_matmul(language, rows):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(17)
    weight = mx.random.normal((128, 6144)) * 0.02
    x = (mx.random.normal((1, rows, 6144)) * 2).astype(mx.bfloat16)
    logits = dk.moe_router_logits(x, weight)
    assert logits is not None and logits.shape == (1, rows, 128)
    for r in range(rows):
        one = x[:, r : r + 1].astype(mx.float32) @ weight.T
        assert mx.array_equal(logits[:, r : r + 1], one).item()
    # Under 16 inputs per expert MLX's gemv splits K differently.
    assert dk.moe_router_logits(x[..., :1024], weight[:, :1024]) is None


@pytest.mark.parametrize("rows", [1, 2, 8])
@pytest.mark.parametrize("down", [False, True])
def test_decode_router_select_matches_moe_select(language, rows, down):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    rng = np.random.default_rng(rows)
    logits = rng.normal(size=(1, rows, 128)).astype(np.float32) * 3
    bias = rng.normal(size=128).astype(np.float32) * 0.1
    # Five experts tie for the top; argpartition keeps the lower indices.
    tied = [3, 10, 11, 12, 13]
    logits[..., tied] = 8.0
    bias[tied] = 1.0
    logits, bias = mx.array(logits), mx.array(bias)
    down_scale = mx.array(rng.uniform(0.5, 2.0, 128).astype(np.float32))
    routed = dk.moe_router_select(logits, bias, 4, 2.0, down_scale if down else None)
    assert routed is not None
    inds, weights = language._minimax_moe_select(logits, bias, 4, 2.0, "sigmoid")
    if down:
        weights = weights * down_scale[inds]
    assert inds[0, 0].tolist() == [3, 10, 11, 12]
    assert mx.array_equal(routed[0], inds).item()
    assert mx.array_equal(routed[1], weights).item()


def _moe_block(language, experts=64, hidden=1024, inter=64):
    args = _tiny_args(
        hidden_size=hidden,
        intermediate_size=inter,
        shared_intermediate_size=inter,
        num_local_experts=experts,
        pack_shared_expert=False,
        routed_expert_scales=True,
    )
    block = language.MiniMaxSparseMoeBlock(args)
    block.gate.weight = mx.random.normal((experts, hidden)) * 0.05
    block.e_score_correction_bias = mx.random.normal((experts,)) * 0.1
    switch = block.switch_mlp
    switch.gate_scale = mx.random.uniform(0.5, 4.0, (experts,))
    switch.up_scale = mx.random.uniform(0.5, 4.0, (experts,))
    switch.down_scale = mx.random.uniform(0.5, 2.0, (experts,))
    return block


@pytest.mark.parametrize("rows,armed", [(1, False), (3, True)])
def test_short_calls_route_bit_exactly(language, monkeypatch, rows, armed):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    from omlx.patches.qwen35_verify_qmm import set_verify_qmm_armed

    mx.random.seed(23)
    block = _moe_block(language)
    x = mx.random.normal((1, rows, 1024)).astype(mx.bfloat16)

    def run():
        set_verify_qmm_armed(armed, row_exact=armed)
        try:
            y = block(x)
            mx.eval(y)
        finally:
            set_verify_qmm_armed(False)
        return y

    used = []

    def spy(name):
        fn = getattr(dk, name)

        def call(*args):
            used.append(name)
            return fn(*args)

        monkeypatch.setattr(dk, name, call)

    spy("moe_router_logits")
    spy("moe_router_select")
    fused = run()
    assert used == ["moe_router_logits", "moe_router_select"]
    monkeypatch.setattr(dk, "moe_router_select", lambda *args: None)
    assert mx.array_equal(run(), fused).item()


def test_batched_rows_route_on_one_row_logits(language, monkeypatch):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(43)
    block = _moe_block(language)
    x = mx.random.normal((3, 1, 1024)).astype(mx.bfloat16)
    seen = []
    select = dk.moe_router_select

    def spy(logits, *args):
        seen.append(logits)
        return select(logits, *args)

    monkeypatch.setattr(dk, "moe_router_select", spy)
    mx.eval(block(x))
    expected = [block.gate(x[i : i + 1].astype(mx.float32)) for i in range(3)]
    assert mx.array_equal(seen[0], mx.concatenate(expected)).item()


def _quantized_moe_block(language):
    block = _moe_block(language, experts=16, inter=512)
    for module, spec in (
        (block.switch_mlp, {"group_size": 16, "bits": 4, "mode": "nvfp4"}),
        (block.shared_experts, {"group_size": 32, "bits": 8, "mode": "mxfp8"}),
    ):
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = module[name]
            linear.weight = mx.random.normal(linear.weight.shape) * 0.05
            module[name] = linear.to_quantized(**spec)
    return block


def test_one_token_experts_match_the_ops(language, monkeypatch):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(29)
    block = _quantized_moe_block(language)
    # Larger inputs push the SwiGLU-OAI gate and up values past the clip.
    xs = [
        (mx.random.normal((1, 1, 1024)) * scale).astype(mx.bfloat16)
        for scale in (0.5, 2.0, 8.0)
        for _ in range(8)
    ]
    calls = []
    moe_experts = dk.moe_experts

    def spy(*args):
        y = moe_experts(*args)
        calls.append(y is not None)
        return y

    monkeypatch.setattr(dk, "moe_experts", spy)
    fused = [block(x) for x in xs]
    assert calls == [True] * len(xs)
    monkeypatch.setattr(dk, "moe_experts", lambda *args: None)
    separate = [block(x) for x in xs]
    monkeypatch.setattr(dk, "moe_experts", spy)
    # The shared projections are now row views of their fused buffer.
    again = [block(x) for x in xs]
    mx.eval(fused, separate, again)
    for a, b, c in zip(fused, separate, again):
        assert mx.array_equal(a, b).item()
        assert mx.array_equal(c, b).item()


def test_one_token_experts_combine_in_mlx_order(language):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(31)
    block = _quantized_moe_block(language)
    switch, shared = block.switch_mlp, block.shared_experts
    # Expert 1 copies expert 0 and their large opposite weights cancel, so
    # the order of the FP32 sum over the experts shows in the rounded output.
    for name in ("gate_proj", "up_proj", "down_proj"):
        for key in ("weight", "scales"):
            bank = switch[name][key]
            switch[name][key] = mx.concatenate([bank[:1], bank[:1], bank[2:]])
    for key in ("gate_scale", "up_scale"):
        scale = switch[key]
        switch[key] = mx.concatenate([scale[:1], scale[:1], scale[2:]])
    inds = mx.array([[[0, 1, 2, 3]]], dtype=mx.uint32)
    weights = mx.array([[[1000.37, -1000.29, 0.53, 0.0071]]])
    for scale in (0.5, 2.0, 8.0):
        x = (mx.random.normal((1, 1, 1024)) * scale).astype(mx.bfloat16)
        expected = language._combine_routed(switch(x, inds), weights) + shared(x)
        y = dk.moe_experts(x, inds, weights, switch, shared)
        assert y is not None
        assert mx.array_equal(y, expected).item()


@pytest.mark.parametrize("k,n", [(512, 64), (1024, 512), (2048, 8192)])
def test_one_row_mxfp8_kernel_matches_quantized_matmul(language, k, n):
    import mlx_vlm.models.minimax_m3_vl.decode_kernels as dk

    mx.random.seed(59)
    proj = nn.QuantizedLinear.from_linear(
        nn.Linear(k, n, bias=False), group_size=32, bits=8, mode="mxfp8"
    )
    for scale in (0.5, 4.0):
        x = (mx.random.normal((1, 1, k)) * scale).astype(mx.bfloat16)
        y = dk.mxfp8_one_row(x, proj.weight, proj.scales)
        assert y is not None
        assert mx.array_equal(y, proj(x)).item()


def test_decode_batch_affine_head_matches_one_row_decode(language, monkeypatch):
    import mlx_vlm.models.minimax_m3_vl.qmv_rows as qmv_rows

    mx.random.seed(61)
    # As wide as a vocabulary head: a decode batch takes the row-exact tile.
    head = nn.QuantizedLinear.from_linear(
        nn.Linear(512, 65536, bias=False), group_size=64, bits=8
    )
    head.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 3, 512)).astype(mx.bfloat16)
    expected = mx.concatenate([head(x[:, i : i + 1]) for i in range(3)], axis=1)
    tiled = qmv_rows.row_exact_qmv.quantized_linears_tiled
    calls = []
    monkeypatch.setattr(
        qmv_rows.row_exact_qmv,
        "quantized_linears_tiled",
        lambda *a, **k: calls.append(1) or tiled(*a, **k),
    )
    assert mx.array_equal(qmv_rows.rows_linear(head, x), expected).item()
    assert calls


@pytest.mark.parametrize("blocks", [17, 40, 125, 1000, 4096])
def test_block_selection_kernel_matches_argpartition(language, blocks):
    import mlx_vlm.models.minimax_m3_vl.msa as msa

    rng = np.random.default_rng(blocks)
    # Few distinct values, so ties fall across the k-th boundary.
    choices = np.array([-1e30, -1e29, -3.0, -2.0, -1.0, 0.0, np.inf], dtype=np.float32)
    negated = mx.array(rng.choice(choices, size=(2, 4, 3, blocks)))
    expected = mx.argpartition(negated, kth=15, axis=-1)[..., :16].astype(mx.int32)
    selected = msa.small_l_select_blocks(negated, 16)
    assert selected is not None
    assert mx.array_equal(selected, expected).item()


@pytest.mark.parametrize("length", [1, 5])
def test_small_l_sparse_attention_staging_changes_no_bit(language, monkeypatch, length):
    import mlx_vlm.models.minimax_m3_vl.msa as msa

    # M3.1's layout: blocks as long as the head dimension, 16 query heads per
    # KV head.
    batch, heads, kv_heads, dim = 2, 32, 2, 128
    total, block, topk = 1200, 128, 4
    pad = np.array([45, 0], dtype=np.int32)
    rng = np.random.default_rng(3)
    q, k, v = (
        mx.array(rng.standard_normal(shape).astype(np.float32)).astype(mx.bfloat16)
        for shape in (
            (batch, heads, length, dim),
            (batch, kv_heads, total, dim),
            (batch, kv_heads, total, dim),
        )
    )
    keep = mx.array(rng.random((batch, 1, length, total)) > 0.1)
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

    def attend(values, mask, padding):
        return msa.small_l_sparse_attention(
            q, k, values, block_ids, mx.array(qpos), padding, block, dim**-0.5, mask
        )

    # Value rows 132 apart cannot take the 16-byte loads staging uses, so the
    # kernel reads each value from device memory instead.
    v_rows = mx.pad(v, [(0, 0), (0, 0), (0, 0), (0, 4)])[..., :dim]

    kernel = msa._MSA_SMALL_L_PARTIALS
    templates = []
    monkeypatch.setattr(
        msa,
        "_MSA_SMALL_L_PARTIALS",
        lambda **kw: templates.append(dict(kw["template"])) or kernel(**kw),
    )
    outs = [attend(v, keep, mx.array(pad)), attend(v, None, None)]
    assert templates[0]["GT"] == 8 and templates[0]["STAGE_V"]

    # Every grouping, chunk and value layout computes the same bits, including
    # one threadgroup for all 16 heads reading values from device memory.
    for heads_per_group, chunk in ((16, 32), (8, 32), (4, 32), (8, 64)):
        monkeypatch.setattr(msa, "_SMALL_L_HEADS_PER_THREADGROUP", heads_per_group)
        monkeypatch.setattr(msa, "_SMALL_L_VALUE_CHUNK", chunk)
        for values in (v, v_rows):
            assert mx.array_equal(attend(values, keep, mx.array(pad)), outs[0]).item()
            assert mx.array_equal(attend(values, None, None), outs[1]).item()
