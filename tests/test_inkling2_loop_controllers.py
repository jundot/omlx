# SPDX-License-Identifier: Apache-2.0
"""Fused BF16 loop-controller matmuls for Inkling 2 looped checkpoints."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from omlx.patches.inkling2 import kernels
from omlx.patches.inkling2 import model as inkling2
from omlx.patches.qwen35_verify_qmm import apply_verify_qmm_patch, set_verify_qmm_armed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")


def _bf16_weights(shape, scale=0.05, seed=0):
    mx.random.seed(seed)
    return (mx.random.normal(shape) * scale).astype(mx.bfloat16).astype(mx.float32)


def _tiny_bf16_loop_model():
    text = dict(
        hidden_size=256,
        num_hidden_layers=18,
        vocab_size=320,
        unpadded_vocab_size=300,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=64,
        sliding_window_size=8,
        rel_extent=8,
        d_rel=8,
        n_routed_experts=8,
        num_experts_per_tok=2,
        intermediate_size=64,
        expert_formats=["mxfp8"] * 18,
        loop_config=dict(core_start=6, core_end=12, num_passes=3, gate_rank=64),
    )
    model = inkling2.Model(inkling2.ModelArgs(model_type="inkling2", text_config=text))
    mx.random.seed(0)
    weights = []
    for name, value in tree_flatten(model.parameters()):
        if value.dtype == mx.uint32:
            raw = mx.random.randint(0, 0x70, (*value.shape[:-1], value.shape[-1] * 4))
            weights.append((name, raw.astype(mx.uint8).view(mx.uint32)))
        elif value.dtype == mx.uint8:
            weights.append((name, mx.full(value.shape, 120, dtype=mx.uint8)))
        elif name.endswith(("loop_down", "loop_gain_forget")):
            w = (mx.random.normal(value.shape) * 0.05).astype(mx.bfloat16)
            weights.append((name, w.astype(mx.float32)))
        else:
            dt = mx.bfloat16 if model.cast_predicate(name) else value.dtype
            weights.append((name, (mx.random.normal(value.shape) * 0.05).astype(dt)))
    model.load_weights(weights)
    return model


@pytest.mark.parametrize("hidden,rank", [(2304, 64), (256, 128)])
@pytest.mark.parametrize(
    "m0,n,qx",
    [(0, 3, 3), (0, 3, 1), (3, 1, 1)],
    ids=["full_attn", "sliding_attn", "mlp"],
)
@pytest.mark.parametrize("batch,length", [(1, 1), (1, 8), (2, 4)])
def test_bf16_loop_kernels_match_fp32_reference(hidden, rank, m0, n, qx, batch, length):
    mx.random.seed(hidden + rank + n + qx + batch * 31 + length)
    mixed = mx.random.normal((batch, length, qx, hidden)).astype(mx.float32)
    messages = mx.random.normal((batch, length, 4, hidden)).astype(mx.float32)
    down = _bf16_weights((n, 2, hidden, rank), seed=1)
    gf = _bf16_weights((n, rank, 2 * hidden), seed=2)
    norm_w = mx.ones((qx, hidden), dtype=mx.float32)

    down_t = mx.contiguous(down.astype(mx.bfloat16).transpose(0, 1, 3, 2))
    gf_t = mx.contiguous(
        gf.astype(mx.bfloat16).reshape(n, rank, 2, hidden).transpose(0, 3, 2, 1)
    )
    # low rounds once to BF16 (so a tie can shift by 1 BF16 ULP); gates stay FP32.
    z = kernels.loop_message_inputs(mixed, messages, m0, n)
    low_ref = kernels.loop_message_low(z @ down.reshape(2 * n, hidden, -1))
    gates_ref = low_ref @ gf
    low_got = kernels.loop_message_down_bf16(mixed, messages, down_t, m0)
    gates_got = kernels.loop_message_gates_bf16(low_ref, gf_t)
    assert mx.allclose(low_got, low_ref, rtol=1e-2, atol=1e-2)
    assert mx.abs(gates_got - gates_ref).max().item() < 1e-4

    want = inkling2._mix_messages_normed(
        mixed, messages, m0, n, down, gf, norm_w, 1e-6, mx.bfloat16, want_f32=True
    )
    got = inkling2._mix_messages_normed(
        mixed,
        messages,
        m0,
        n,
        down,
        gf,
        norm_w,
        1e-6,
        mx.bfloat16,
        want_f32=True,
        bf16_weights=(down_t, gf_t),
    )
    for a, b in zip(got, want):
        assert a.shape == b.shape and a.dtype == b.dtype
        assert mx.allclose(
            a.astype(mx.float32), b.astype(mx.float32), rtol=1e-2, atol=1e-2
        )


def test_bf16_loop_kernels_are_row_exact():
    """Each row of a multi-row verify block equals its one-row decode call."""
    hidden, rank, rows = 256, 64, 8
    mx.random.seed(9)
    mixed = mx.random.normal((1, rows, 3, hidden)).astype(mx.float32)
    messages = mx.random.normal((1, rows, 4, hidden)).astype(mx.float32)
    down = _bf16_weights((3, 2, hidden, rank), seed=3)
    gf = _bf16_weights((3, rank, 2 * hidden), seed=4)
    down_t = mx.contiguous(down.astype(mx.bfloat16).transpose(0, 1, 3, 2))
    gf_t = mx.contiguous(
        gf.astype(mx.bfloat16).reshape(3, rank, 2, hidden).transpose(0, 3, 2, 1)
    )
    norm_w = mx.ones((3, hidden), dtype=mx.float32)

    block = inkling2._mix_messages_normed(
        mixed,
        messages,
        0,
        3,
        down,
        gf,
        norm_w,
        1e-6,
        mx.bfloat16,
        bf16_weights=(down_t, gf_t),
    )[0]
    for r in range(rows):
        step = inkling2._mix_messages_normed(
            mixed[:, r : r + 1],
            messages[:, r : r + 1],
            0,
            3,
            down,
            gf,
            norm_w,
            1e-6,
            mx.bfloat16,
            bf16_weights=(down_t, gf_t),
        )[0]
        assert mx.array_equal(block[:, :, r : r + 1], step)


def test_bf16_loop_end_to_end_layer_and_row_exact_verify(monkeypatch):
    """BF16 loop kernels run through Inkling2Layer compiled/exact traces,
    fall back on prefill / oversized batches, and keep verify rows bit-exact."""
    apply_verify_qmm_patch()
    model = _tiny_bf16_loop_model()
    nn.quantize(
        model,
        group_size=64,
        bits=8,
        class_predicate=lambda _, m: isinstance(m, nn.Linear),
    )

    hits = []
    orig_down = kernels.loop_message_down_bf16

    def spy(mixed, messages, down_t, m0):
        hits.append(mixed.shape[0] * mixed.shape[1])
        return orig_down(mixed, messages, down_t, m0)

    monkeypatch.setattr(kernels, "loop_message_down_bf16", spy)

    ids = mx.random.randint(0, 300, (1, 28))
    # Prefill chunk (> 16 rows) must skip the per-row BF16 kernel.
    c0 = model.make_cache()
    mx.eval(model(ids[:, :20], cache=c0))
    assert hits == []

    # Serial decode (L=1) and row-exact verify (L=8) both use BF16 loop kernels
    # and match bit for bit.
    serial = [model(ids[:, t : t + 1], cache=c0)[0, -1] for t in range(20, 28)]
    assert 1 in hits

    c1 = model.make_cache()
    mx.eval(model(ids[:, :20], cache=c1))
    hits.clear()
    set_verify_qmm_armed(True, row_exact=True)
    try:
        block = model(ids[:, 20:28], cache=c1)[0]
    finally:
        set_verify_qmm_armed(False)
    assert hits and all(r == 8 for r in hits)
    for r, expected in enumerate(serial):
        assert mx.array_equal(block[r], expected)

    # Large batch (B=4, L=5 -> rows=20 > 16) falls back to FP32 GEMM.
    hits.clear()
    mx.eval(model(mx.random.randint(0, 300, (4, 5))))
    assert hits == []


def test_bf16_loop_cache_guards_and_cpu_to_gpu_transition():
    model = _tiny_bf16_loop_model()
    layer = model.model.layers[6]
    before = [(k, v.shape) for k, v in tree_flatten(layer.parameters())]

    # Unsupported shapes or CPU device must not poison the cache.
    assert not kernels.loop_bf16_supported(192, 64)
    assert not kernels.loop_bf16_supported(256, 32)
    assert not kernels.loop_bf16_supported(4224, 64)
    prev = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        assert not kernels.loop_bf16_supported(256, 64)
        assert inkling2._bf16_loop_weights(layer) is None
    finally:
        mx.set_default_device(prev)

    # Back on GPU, exact BF16-widened weights build the cache without mutating parameters.
    packed = inkling2._bf16_loop_weights(layer)
    assert packed is not None
    assert [(k, v.shape) for k, v in tree_flatten(layer.parameters())] == before

    # Non-BF16 FP32 weights must not be truncated.
    mx.random.seed(11)
    layer.loop_down = mx.random.normal(layer.loop_down.shape).astype(mx.float32) * 0.05
    layer.loop_gain_forget = (
        mx.random.normal(layer.loop_gain_forget.shape).astype(mx.float32) * 0.05
    )
    assert inkling2._bf16_loop_weights(layer) is None

    # Replacing loop_down with BF16 values refreshes the cache.
    layer.loop_down = _bf16_weights(layer.loop_down.shape, seed=14)
    layer.loop_gain_forget = _bf16_weights(layer.loop_gain_forget.shape, seed=15)
    refreshed = inkling2._bf16_loop_weights(layer)
    assert refreshed is not None and refreshed[0][0] is not packed[0][0]
