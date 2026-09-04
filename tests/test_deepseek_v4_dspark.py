# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4-Flash-0731 embedded DSpark regression tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest


@pytest.fixture(scope="module")
def dsv4():
    from omlx.patches.deepseek_v4 import apply_deepseek_v4_patch
    from omlx.patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch

    apply_deepseek_v4_patch()
    apply_mlx_lm_mtp_patch()
    import mlx_lm.models.deepseek_v4 as module

    return module


def _tiny_config(dsv4):
    return dsv4.ModelArgs.from_dict(
        {
            "model_type": "deepseek_v4",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "moe_intermediate_size": 4,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "num_experts_per_tok": 1,
            "num_hash_layers": 0,
            "q_lora_rank": 4,
            "qk_rope_head_dim": 4,
            "head_dim": 4,
            "o_groups": 2,
            "o_lora_rank": 4,
            "index_n_heads": 2,
            "index_head_dim": 4,
            "index_topk": 2,
            "hc_mult": 4,
            "compress_ratios": [0, 0, 0, 0, 0, 0],
            # This legacy field deliberately coexists with DSpark in 0731.
            "num_nextn_predict_layers": 1,
            "dspark_block_size": 3,
            "dspark_noise_token_id": 31,
            "dspark_target_layer_ids": [0, 1, 2],
            "dspark_markov_rank": 4,
        }
    )


def test_model_args_preserve_dspark_tail_compress_ratios(dsv4):
    args = _tiny_config(dsv4)
    assert args.dspark_target_layer_ids == [0, 1, 2]
    assert len(args.compress_ratios) == 6


def test_ratio128_verify_boundary_matches_m1_pooling(dsv4):
    from omlx.patches.deepseek_v4.cache_extras import BatchPoolingCache

    config = _tiny_config(dsv4)
    compressor = dsv4.Compressor(config, compress_ratio=128, head_dim=4)
    prefix_kv = mx.random.normal((1, 126, 4), dtype=mx.bfloat16)
    prefix_gate = mx.random.normal((1, 126, 4), dtype=mx.bfloat16)
    cache = BatchPoolingCache(128, [0])
    compressor.consume(prefix_kv, prefix_gate, cache, mx.array([0]))
    mx.eval(cache.buf_kv, cache.buf_gate)

    sequential_cache = copy.deepcopy(cache)
    block_cache = copy.deepcopy(cache)
    block_kv = mx.random.normal((1, 3, 4), dtype=mx.bfloat16)
    block_gate = mx.random.normal((1, 3, 4), dtype=mx.bfloat16)
    sequential = [
        compressor.consume(
            block_kv[:, idx : idx + 1],
            block_gate[:, idx : idx + 1],
            sequential_cache,
            mx.array([126 + idx]),
        )
        for idx in range(3)
    ]
    block = dsv4._consume_verify_rows(
        compressor,
        block_kv,
        block_gate,
        block_cache,
        mx.array([126]),
    )
    mx.eval(*sequential, *block)

    assert all(
        mx.array_equal(expected, actual).item()
        for expected, actual in zip(sequential, block)
    )
    assert block_cache.remainder == sequential_cache.remainder
    assert block_cache._pool_lengths == sequential_cache._pool_lengths


def test_dspark_wins_over_legacy_nextn_discriminator(dsv4):
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        model = dsv4.Model(_tiny_config(dsv4))
    finally:
        set_mtp_active(False)

    assert model._omlx_dspark_decode_enabled is True
    assert len(model.mtp) == 3
    assert all(isinstance(stage, dsv4.DSparkBlock) for stage in model.mtp)
    assert not any(isinstance(stage, dsv4.MTPBlock) for stage in model.mtp)


def test_dspark_target_tap_and_parallel_draft_shapes(dsv4):
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        model = dsv4.Model(_tiny_config(dsv4))
    finally:
        set_mtp_active(False)

    target_cache = model.make_cache()
    logits, target_hidden = model(
        mx.array([[1, 2, 3]], dtype=mx.uint32),
        cache=target_cache,
        return_hidden=True,
    )
    draft_cache = model.make_mtp_cache()
    draft_logits, draft_hidden = model.dspark_forward(
        target_hidden[:, -1:],
        mx.array([[4]], dtype=mx.uint32),
        draft_cache,
        draft_length=3,
    )
    mx.eval(logits, target_hidden, draft_logits, draft_hidden)

    assert logits.shape == (1, 3, 32)
    assert target_hidden.shape == (1, 3, 24)
    assert draft_logits.shape == (1, 3, 32)
    assert draft_hidden.shape == (1, 3, 8)
    assert [cache.offset for cache in draft_cache] == [1, 1, 1]


def test_dspark_query_block_matches_requested_depth(dsv4, monkeypatch):
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        model = dsv4.Model(_tiny_config(dsv4))
    finally:
        set_mtp_active(False)

    seen_widths = []
    original_call = dsv4.DSparkBlock.__call__

    def traced_call(self, hidden, *args, **kwargs):
        seen_widths.append(int(hidden.shape[1]))
        return original_call(self, hidden, *args, **kwargs)

    monkeypatch.setattr(dsv4.DSparkBlock, "__call__", traced_call)
    target_cache = model.make_cache()
    _, target_hidden = model(
        mx.array([[1]], dtype=mx.uint32),
        cache=target_cache,
        return_hidden=True,
    )
    draft_logits, _ = model.dspark_forward(
        target_hidden,
        mx.array([[1]], dtype=mx.uint32),
        model.make_mtp_cache(),
        draft_length=2,
    )
    mx.eval(draft_logits)

    assert draft_logits.shape == (1, 2, 32)
    assert seen_widths == [2, 2, 2]


def test_dspark_context_cache_keeps_reference_physical_ring_order(dsv4):
    cache = dsv4.DSparkContextCache(4)

    def append(*positions):
        values = mx.array(positions, dtype=mx.float32).reshape(1, 1, -1, 1)
        cache.append(values)
        mx.eval(cache.keys)

    append(0, 1, 2, 3)
    assert cache.keys.reshape(-1).tolist() == [0, 1, 2, 3]

    append(4)
    assert cache.offset == 5
    assert cache.keys.reshape(-1).tolist() == [4, 1, 2, 3]

    append(5, 6)
    assert cache.offset == 7
    assert cache.keys.reshape(-1).tolist() == [4, 5, 6, 3]

    append(7, 8, 9, 10, 11)
    assert cache.offset == 12
    assert cache.keys.reshape(-1).tolist() == [8, 9, 10, 11]


def test_dspark_sanitize_keeps_direct_stage_layout(dsv4):
    fake = SimpleNamespace(
        args=SimpleNamespace(
            num_hidden_layers=1,
            num_nextn_predict_layers=1,
            dspark_block_size=3,
            dspark_target_layer_ids=[0, 1, 2],
            n_mtp_layers=0,
            n_routed_experts=2,
            o_groups=2,
            o_lora_rank=4,
        ),
        mtp=[object(), object(), object()],
        _omlx_dspark_decode_enabled=True,
        _omlx_mtp_decode_enabled=True,
    )
    weights = {
        "mtp.0.main_proj.weight": mx.zeros((8, 24)),
        "mtp.0.attn.wo_a.weight": mx.zeros((8, 16)),
        "mtp.0.hc_attn_base": mx.zeros((1,)),
        "mtp.2.hc_head_fn": mx.zeros((4, 32)),
        "mtp.2.markov_head.markov_w1.weight": mx.zeros((32, 4)),
        "mtp.2.confidence_head.proj.weight": mx.zeros((1, 12)),
    }
    for expert in range(2):
        for name in ("w1", "w2", "w3"):
            weights[f"mtp.0.ffn.experts.{expert}.{name}.weight"] = mx.zeros((4, 8))

    out = dsv4.Model.sanitize(fake, weights)

    assert "mtp.0.main_proj.weight" in out
    assert out["mtp.0.attn.wo_a.weight"].shape == (2, 4, 16)
    assert "mtp.0.attn_hc.base" in out
    assert "mtp.2.hc_head.fn" in out
    assert "mtp.2.markov_head.markov_w1.weight" in out
    assert "mtp.2.confidence_head.proj.weight" in out
    assert "mtp.0.ffn.switch_mlp.gate_proj.weight" in out
    assert not any(".block." in key for key in out if key.startswith("mtp."))


def test_dspark_generation_batch_samples_markov_chain(dsv4):
    from omlx.patches.mlx_lm_mtp import set_mtp_active, set_mtp_depth
    from omlx.patches.mlx_lm_mtp.batch_generator import (
        _dspark_next_drafts,
        _MtpState,
    )

    set_mtp_active(True)
    set_mtp_depth(3)
    try:
        model = dsv4.Model(_tiny_config(dsv4))
    finally:
        set_mtp_active(False)

    def greedy(logprobs):
        return mx.argmax(logprobs, axis=-1)

    greedy.temp = 0.0
    batch = SimpleNamespace(
        model=model,
        samplers=[None],
        fallback_sampler=greedy,
        logits_processors=[[]],
    )
    state = _MtpState(depth=3, mtp_cache=model.make_mtp_cache())
    _dspark_next_drafts(
        batch,
        state,
        mx.zeros((1, 1, 24)),
        mx.array([4], dtype=mx.uint32),
        None,
    )
    mx.eval(state.drafts)

    assert state.drafts.shape == (3,)
    assert len(state.draft_lps) == 3
    assert len(state.draft_accept_lps) == 3
    assert state.hist_offset == 1


@pytest.mark.parametrize(
    ("mode", "bits", "group_size"),
    [("affine", 8, 64), ("mxfp8", 8, 32)],
)
def test_verify_singleton_batch_qmv_matches_decode_rows(mode, bits, group_size):
    from omlx.patches.deepseek_v4.decode_consistency import (
        _batched_singleton_qmv,
    )

    mx.random.seed(7)
    rows, input_dims, output_dims = 4, 256, 128
    weight = mx.random.normal((output_dims, input_dims), dtype=mx.bfloat16)
    qweight, scales, *biases = mx.quantize(
        weight,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    inputs = mx.random.normal((1, rows, input_dims), dtype=mx.bfloat16)

    def call(value):
        return mx.quantized_matmul(
            value,
            qweight,
            scales,
            biases[0] if biases else None,
            transpose=True,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    expected = mx.concatenate(
        [call(inputs[:, idx : idx + 1]) for idx in range(rows)],
        axis=1,
    )
    actual = _batched_singleton_qmv(inputs, call)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize(
    ("mode", "group_size"),
    [("mxfp8", 32), ("affine", 64)],
)
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
def test_verify_exact_qmv_kernel_matches_decode_rows(mode, group_size, rows):
    from omlx.patches.deepseek_v4.verify_qmv import exact_verify_qmv

    mx.random.seed(31 + rows)
    input_dims, output_dims = 4096, 1024
    linear = nn.Linear(input_dims, output_dims, bias=True)
    linear.weight = mx.random.normal(
        (output_dims, input_dims),
        dtype=mx.bfloat16,
    )
    linear.bias = mx.random.normal((output_dims,), dtype=mx.bfloat16)
    module = nn.QuantizedLinear.from_linear(
        linear,
        group_size=group_size,
        bits=8,
        mode=mode,
    )
    inputs = mx.random.normal((1, rows, input_dims), dtype=mx.bfloat16)

    expected = mx.concatenate(
        [module(inputs[:, idx : idx + 1]) for idx in range(rows)],
        axis=1,
    )
    actual = exact_verify_qmv(module, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("rows", [2, 3, 5, 6])
def test_verify_exact_mxfp8_qmv_pair_matches_decode_rows(rows):
    from omlx.patches.deepseek_v4.verify_qmv import (
        exact_verify_qmv_pair,
        pair_eligible,
    )

    mx.random.seed(233 + rows)
    input_dims, output_dims = 512, 512
    modules = []
    for _ in range(2):
        linear = nn.Linear(input_dims, output_dims, bias=True)
        linear.weight = mx.random.normal(
            (output_dims, input_dims),
            dtype=mx.bfloat16,
        )
        linear.bias = mx.random.normal((output_dims,), dtype=mx.bfloat16)
        modules.append(
            nn.QuantizedLinear.from_linear(
                linear,
                group_size=32,
                bits=8,
                mode="mxfp8",
            )
        )
    inputs = mx.random.normal((1, rows, input_dims), dtype=mx.bfloat16)

    if not pair_eligible(*modules, inputs):
        pytest.skip("native DSpark QMV pair kernel is unavailable")
    expected = [
        mx.concatenate(
            [module(inputs[:, idx : idx + 1]) for idx in range(rows)],
            axis=1,
        )
        for module in modules
    ]
    actual = exact_verify_qmv_pair(*modules, inputs)
    mx.eval(*expected, *actual)

    assert mx.array_equal(actual[0], expected[0]).item()
    assert mx.array_equal(actual[1], expected[1]).item()


def test_verify_qmv_pair_rejects_dense_linears():
    from omlx.patches.deepseek_v4.verify_qmv import pair_eligible

    inputs = mx.zeros((1, 3, 512), dtype=mx.bfloat16)
    assert not pair_eligible(nn.Linear(512, 512), nn.Linear(512, 512), inputs)


def test_verify_batched_gemv_matches_decode_rows():
    from omlx.patches.deepseek_v4.decode_consistency import (
        matmul,
        set_armed,
    )

    mx.random.seed(11)
    inputs = mx.random.normal((1, 4, 257), dtype=mx.float32)
    weight = mx.random.normal((257, 63), dtype=mx.float32)
    expected = mx.concatenate(
        [inputs[:, idx : idx + 1] @ weight for idx in range(4)],
        axis=1,
    )
    set_armed(True)
    try:
        actual = matmul(inputs, weight)
    finally:
        set_armed(False)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
def test_verify_exact_dense_gemv_matches_decode_rows(rows):
    from omlx.patches.deepseek_v4.verify_qmv import exact_verify_gemv

    mx.random.seed(47 + rows)
    input_dims, output_dims = 512, 4096
    module = nn.Linear(input_dims, output_dims, bias=True)
    module.weight = mx.random.normal(
        (output_dims, input_dims),
        dtype=mx.bfloat16,
    )
    module.bias = mx.random.normal((output_dims,), dtype=mx.bfloat16)
    inputs = mx.random.normal((1, rows, input_dims), dtype=mx.bfloat16)

    expected = mx.concatenate(
        [module(inputs[:, idx : idx + 1]) for idx in range(rows)],
        axis=1,
    )
    actual = exact_verify_gemv(module, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("rows", [1, 3, 5])
def test_dspark_head_gemv_matches_promoted_fp32_projection(rows):
    from omlx.patches.deepseek_v4.verify_qmv import dspark_head_gemv

    mx.random.seed(53 + rows)
    input_dims, output_dims = 512, 4096
    module = nn.Linear(input_dims, output_dims, bias=True)
    module.weight = mx.random.normal(
        (output_dims, input_dims),
        dtype=mx.bfloat16,
    )
    module.bias = mx.random.normal((output_dims,), dtype=mx.bfloat16)
    inputs = mx.random.normal((1, rows, input_dims), dtype=mx.bfloat16)

    expected = inputs.astype(mx.float32) @ module.weight.T.astype(
        mx.float32
    ) + module.bias.astype(mx.float32)
    actual = dspark_head_gemv(module, inputs)
    mx.eval(expected, actual)

    assert actual.dtype == mx.float32
    if rows == 1:
        assert mx.array_equal(actual, expected).item()
    else:
        # The custom kernel keeps the M=1 GEMV reduction for every row while
        # MLX selects GEMM for the promoted M>1 reference.
        assert mx.allclose(actual, expected, rtol=0, atol=4e-5).item()


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
def test_verify_exact_multi_qmv_matches_decode_rows(rows):
    from mlx_lm.models.mla import MultiLinear
    from omlx.patches.deepseek_v4.verify_qmv import exact_verify_multi_qmv

    mx.random.seed(59 + rows)
    groups, input_dims, output_dims = 8, 512, 128
    module = MultiLinear(input_dims, output_dims, groups).to_quantized(
        group_size=32,
        bits=8,
        mode="mxfp8",
    )
    inputs = mx.random.normal(
        (groups, rows, input_dims),
        dtype=mx.bfloat16,
    )

    expected = mx.concatenate(
        [module(inputs[:, idx : idx + 1]) for idx in range(rows)],
        axis=1,
    )
    actual = exact_verify_multi_qmv(module, inputs)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("head_dim", [128, 512])
@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6])
def test_dspark_attention_kernel_matches_its_decode_path(head_dim, rows):
    from omlx.patches.deepseek_v4.verify_attention import exact_attention

    mx.random.seed(71 + head_dim + rows)
    heads, key_length = 64, 128
    queries = mx.random.normal(
        (1, heads, rows, head_dim),
        dtype=mx.bfloat16,
    )
    key_rows = [
        mx.random.normal((1, 1, key_length, head_dim), dtype=mx.bfloat16)
        for _ in range(rows)
    ]
    sinks = mx.random.normal((heads,), dtype=mx.bfloat16)

    expected = mx.concatenate(
        [
            exact_attention(
                queries[:, :, idx : idx + 1],
                [key_rows[idx]],
                head_dim**-0.5,
                sinks,
            )
            for idx in range(rows)
        ],
        axis=2,
    )
    actual = exact_attention(queries, key_rows, head_dim**-0.5, sinks)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("rows", [2, 3, 5, 6])
def test_dspark_ring_gemm_matches_materialized_decode_rows(rows):
    from omlx.custom_kernels.glm_moe_dsa import fast
    from omlx.patches.deepseek_v4.verify_attention import rowwise_gemm

    if not fast.has_symbol("dspark_ring_gemm"):
        pytest.skip("native DSpark physical-ring GEMM is unavailable")

    mx.random.seed(83 + rows)
    source = mx.random.normal((128 + rows, 512), dtype=mx.bfloat16)
    index_rows = []
    for row in range(rows):
        snapshot = list(range(128))
        for update in range(row + 1):
            snapshot[(123 + update) % 128] = 128 + update
        index_rows.append(snapshot)
    indices = mx.array(index_rows, dtype=mx.uint32)
    gathered = mx.take(source, indices, axis=0)

    queries = mx.random.normal((rows, 64, 512), dtype=mx.bfloat16)
    expected_scores = rowwise_gemm(queries, gathered, True)
    actual_scores = fast.dspark_ring_gemm(queries, source, indices, True)

    weights = mx.random.normal((rows, 64, 128), dtype=mx.bfloat16)
    expected_values = rowwise_gemm(weights, gathered, False)
    actual_values = fast.dspark_ring_gemm(weights, source, indices, False)
    mx.eval(expected_scores, actual_scores, expected_values, actual_values)

    assert mx.array_equal(actual_scores, expected_scores).item()
    assert mx.array_equal(actual_values, expected_values).item()


@pytest.mark.parametrize("rows", [2, 3, 5])
def test_dspark_ring_sparse_attention_matches_materialized_path(dsv4, rows):
    from omlx.custom_kernels.glm_moe_dsa import fast

    if not fast.has_symbol("dspark_ring_gemm"):
        pytest.skip("native DSpark physical-ring GEMM is unavailable")

    mx.random.seed(109 + rows)
    source = mx.random.normal((128 + rows, 512), dtype=mx.bfloat16)
    index_rows = []
    for row in range(rows):
        snapshot = list(range(128))
        for update in range(row + 1):
            snapshot[(121 + update) % 128] = 128 + update
        index_rows.append(snapshot)
    indices = mx.array(index_rows, dtype=mx.uint32)
    local_kv = mx.take(source, indices, axis=0)[:, None]
    q = mx.random.normal((rows, 64, 1, 512), dtype=mx.bfloat16)
    pooled = mx.random.normal((rows, 640, 512), dtype=mx.bfloat16)
    topk = mx.broadcast_to(
        mx.arange(512, dtype=mx.uint32)[None, None],
        (rows, 1, 512),
    )
    sinks = mx.random.normal((64,), dtype=mx.bfloat16)
    scale = 512**-0.5

    expected = dsv4._sparse_pooled_attention(
        q,
        local_kv,
        pooled,
        topk,
        None,
        None,
        scale,
        sinks,
        decode_consistent=True,
    )
    actual = dsv4._sparse_pooled_ring_attention(
        q,
        source,
        indices,
        pooled,
        topk,
        scale,
        sinks,
    )
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_dspark_multitoken_prefill_uses_vectorized_sparse_attention(dsv4):
    mx.random.seed(2490)
    q = mx.random.normal((1, 2, 3, 4))
    local_kv = mx.random.normal((1, 1, 5, 4))
    pooled = mx.random.normal((1, 6, 4))
    topk = mx.array([[[0, 1], [2, 3], [4, 5]]], dtype=mx.uint32)
    sinks = mx.random.normal((2,))

    expected = dsv4._sparse_pooled_attention(
        q,
        local_kv,
        pooled,
        topk,
        None,
        None,
        4**-0.5,
        sinks,
        decode_consistent=False,
    )
    actual = dsv4._sparse_pooled_attention(
        q,
        local_kv,
        pooled,
        topk,
        None,
        None,
        4**-0.5,
        sinks,
        decode_consistent=True,
    )
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_dspark_attention_keeps_pool_boundary_lengths_separate():
    from omlx.patches.deepseek_v4.verify_attention import exact_attention

    mx.random.seed(907)
    heads, head_dim = 64, 512
    lengths = (159, 160, 160)
    queries = mx.random.normal(
        (1, heads, len(lengths), head_dim),
        dtype=mx.bfloat16,
    )
    key_rows = [
        mx.random.normal((1, 1, length, head_dim), dtype=mx.bfloat16)
        for length in lengths
    ]
    sinks = mx.random.normal((heads,), dtype=mx.bfloat16)
    expected = mx.concatenate(
        [
            exact_attention(
                queries[:, :, idx : idx + 1],
                [key_rows[idx]],
                head_dim**-0.5,
                sinks,
            )
            for idx in range(len(lengths))
        ],
        axis=2,
    )
    actual = exact_attention(queries, key_rows, head_dim**-0.5, sinks)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


def test_dspark_indexer_batches_adjacent_pool_lengths(dsv4):
    mx.random.seed(929)
    lengths = (513, 514, 514)
    indexer = SimpleNamespace(index_topk=512, n_heads=64, scale=128**-0.5)
    pooled_rows = [
        mx.random.normal((1, length, 128), dtype=mx.bfloat16) for length in lengths
    ]
    projected_q = mx.random.normal((1, 64, 3, 128), dtype=mx.bfloat16)
    projected_weights = mx.random.normal((1, 3, 64), dtype=mx.bfloat16)

    expected = [
        dsv4._batch_indexer_rows(
            indexer,
            [pooled_rows[idx]],
            projected_q[:, :, idx : idx + 1],
            projected_weights[:, idx : idx + 1],
        )[0]
        for idx in range(3)
    ]
    actual = dsv4._batch_indexer_rows(
        indexer,
        pooled_rows,
        projected_q,
        projected_weights,
    )
    mx.eval(*expected, *actual)

    assert all(
        mx.array_equal(reference, candidate).item()
        for reference, candidate in zip(expected, actual)
    )


@pytest.mark.parametrize("key_length", [156, 512, 640, 1024])
@pytest.mark.parametrize("rows", [2, 3, 6])
def test_dspark_attention_matches_stock_m1_fallback(dsv4, key_length, rows):
    from omlx.patches.deepseek_v4.verify_attention import exact_attention

    mx.random.seed(811 + key_length + rows)
    heads, head_dim = 64, 512
    queries = mx.random.normal(
        (1, heads, rows, head_dim),
        dtype=mx.bfloat16,
    )
    key_rows = [
        mx.random.normal(
            (1, 1, key_length, head_dim),
            dtype=mx.bfloat16,
        )
        for _ in range(rows)
    ]
    sinks = mx.random.normal((heads,), dtype=mx.bfloat16)
    scale = head_dim**-0.5

    expected = mx.concatenate(
        [
            dsv4.scaled_dot_product_attention(
                queries[:, :, idx : idx + 1],
                key_rows[idx],
                key_rows[idx],
                cache=None,
                scale=scale,
                mask=None,
                sinks=sinks,
            )
            for idx in range(rows)
        ],
        axis=2,
    )
    actual = exact_attention(queries, key_rows, scale, sinks)
    mx.eval(expected, actual)

    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("batch_cache", [False, True])
def test_vectorized_verify_ring_snapshots_match_m1_updates(dsv4, batch_cache):
    from mlx_lm.models.cache import BatchRotatingKVCache

    def make_cache():
        if batch_cache:
            return BatchRotatingKVCache(max_size=4, left_padding=[0])
        return dsv4.RotatingKVCache(max_size=4)

    def clone(source):
        cloned = make_cache()
        for field in (
            "keys",
            "values",
            "offset",
            "_idx",
            "_offset",
            "rotated",
            "left_padding",
        ):
            if not hasattr(source, field):
                continue
            value = getattr(source, field)
            if isinstance(value, mx.array):
                value = value + 0
            setattr(cloned, field, value)
        mx.eval(cloned.keys, cloned.values)
        return cloned

    cache = make_cache()
    empty = mx.zeros((1, 1, 1, 0), dtype=mx.bfloat16)
    for position in range(6):
        key = mx.full((1, 1, 1, 8), position, dtype=mx.bfloat16)
        cache.update_and_fetch(key, empty)
        mx.eval(cache.keys, cache.values)

    expected_cache = clone(cache)
    actual_cache = clone(cache)
    block = mx.stack(
        [mx.full((1, 1, 8), position, dtype=mx.bfloat16) for position in range(6, 9)],
        axis=2,
    )

    expected_rows = []
    for idx in range(block.shape[2]):
        row, _ = expected_cache.update_and_fetch(
            block[..., idx : idx + 1, :],
            empty,
        )
        expected_rows.append(row + 0)
    actual_rows = dsv4._consume_rotating_verify_rows(actual_cache, block)
    mx.eval(*expected_rows, *actual_rows, expected_cache.keys, actual_cache.keys)

    assert len(actual_rows) == len(expected_rows)
    for actual, expected in zip(actual_rows, expected_rows):
        assert mx.array_equal(actual, expected).item()
    assert actual_cache.meta_state == expected_cache.meta_state
    assert mx.array_equal(actual_cache.keys, expected_cache.keys).item()
    assert mx.array_equal(actual_cache.values, expected_cache.values).item()
    if batch_cache:
        assert mx.array_equal(actual_cache.offset, expected_cache.offset).item()
        assert mx.array_equal(
            actual_cache.left_padding,
            expected_cache.left_padding,
        ).item()


@pytest.mark.parametrize("batch_cache", [False, True])
def test_vectorized_verify_ring_rollback_matches_accepted_prefix(dsv4, batch_cache):
    from mlx_lm.models.cache import BatchRotatingKVCache
    from omlx.patches.mlx_lm_mtp.cache_rollback import set_undo_armed

    def make_cache():
        if batch_cache:
            return BatchRotatingKVCache(max_size=4, left_padding=[0])
        return dsv4.RotatingKVCache(max_size=4)

    def clone(source):
        cloned = make_cache()
        for field in (
            "keys",
            "values",
            "offset",
            "_idx",
            "_offset",
            "rotated",
            "left_padding",
        ):
            if not hasattr(source, field):
                continue
            value = getattr(source, field)
            if isinstance(value, mx.array):
                value = value + 0
            setattr(cloned, field, value)
        mx.eval(cloned.keys, cloned.values)
        return cloned

    cache = make_cache()
    empty = mx.zeros((1, 1, 1, 0), dtype=mx.bfloat16)
    for position in range(6):
        key = mx.full((1, 1, 1, 8), position, dtype=mx.bfloat16)
        cache.update_and_fetch(key, empty)
        mx.eval(cache.keys, cache.values)

    expected = clone(cache)
    actual = clone(cache)
    block = mx.stack(
        [mx.full((1, 1, 8), position, dtype=mx.bfloat16) for position in range(6, 9)],
        axis=2,
    )
    expected.update_and_fetch(block[..., :1, :], empty)

    set_undo_armed(True)
    try:
        dsv4._consume_rotating_verify_rows(actual, block)
    finally:
        set_undo_armed(False)
    assert actual.trim(2) == 2
    mx.eval(expected.keys, expected.values, actual.keys, actual.values)

    assert actual.meta_state == expected.meta_state
    assert mx.array_equal(actual.keys, expected.keys).item()
    assert mx.array_equal(actual.values, expected.values).item()
    if batch_cache:
        assert mx.array_equal(actual.offset, expected.offset).item()
        assert mx.array_equal(actual.left_padding, expected.left_padding).item()
