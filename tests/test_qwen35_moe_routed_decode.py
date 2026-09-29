# SPDX-License-Identifier: Apache-2.0
"""Bit-exactness and routing tests for the fused one-token routed experts."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

EXPERTS = 32


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched_block(monkeypatch):
    """Apply router + routed patches, restore the class afterwards."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.qwen35_moe_router import apply_qwen35_moe_router_patch

    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    apply_qwen35_moe_router_patch()  # process-wide and idempotent
    assert cls._omlx_router_fused
    call = cls.__call__
    original = getattr(call, "_omlx_routed_decode_original", call)
    monkeypatch.setattr(routed, "_DISABLED", False)
    monkeypatch.setattr(routed, "_PROVEN", False)
    cls.__call__ = original
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield cls
    cls.__call__ = original
    cls._omlx_router_fused = True
    if "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


def _block(
    hidden, inter, top_k=10, bits=4, group_size=64, seed=0, experts=EXPERTS, quantized_shared=True
):
    """A block laid out like Qwen3.8-Flash-Next oQ: quantized routed experts,
    8-bit shared expert (gs128 where the shape allows), 8-bit gs64
    shared-expert gate, bf16 router. ``quantized_shared=False`` keeps the
    shared expert and its gate in bf16."""
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=hidden,
        moe_intermediate_size=inter,
        shared_expert_intermediate_size=inter,
        num_experts=experts,
        num_experts_per_tok=top_k,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    if quantized_shared:
        shared = block.shared_expert
        shared_gs = 128 if hidden % 128 == 0 and inter % 128 == 0 else 64
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(
                shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), shared_gs, 8)
            )
        block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    return block


def _pair(block, x):
    routed._DISABLED = True
    ref = block(x)
    mx.eval(ref)
    routed._DISABLED = False
    out = block(x)
    mx.eval(out)
    return ref, out


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


@pytest.mark.parametrize(
    "hidden,inter,bits,group_size",
    [
        (2560, 640, 5, 64),  # Qwen3.8-Flash-Next oQ5e
        (2560, 640, 4, 64),
        (1024, 320, 5, 32),
        (1024, 384, 6, 128),
        (1024, 320, 8, 64),
    ],
)
def test_fused_decode_is_bit_identical(hidden, inter, bits, group_size, monkeypatch):
    calls = []
    fused = routed.routed_decode
    monkeypatch.setattr(
        routed, "routed_decode", lambda *a: calls.append(1) or fused(*a)
    )
    for seed in range(2):
        block = _block(hidden, inter, bits=bits, group_size=group_size, seed=seed)
        for step in range(6):
            x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
            # The shared expert and its gate run inside the two launches.
            assert routed.routed_decode_plan(block, x).fold
            ref, out = _pair(block, x)
            assert _same_bits(ref, out)
    assert len(calls) == 12
    assert not routed._DISABLED


def test_bf16_shared_expert_stays_composed_and_bit_identical():
    block = _block(2560, 640, bits=5, quantized_shared=False)
    for step in range(4):
        x = (mx.random.normal((1, 1, 2560)) * (0.5 + step)).astype(mx.bfloat16)
        assert not routed.routed_decode_plan(block, x).fold
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert routed._PROVEN and not routed._DISABLED


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_kernels_match_mlx_mat_vecs(bits):
    """BF16 outputs hide one-ulp FP32 differences, so run both launches in
    FP32 against MLX's FP32 mat-vecs: the gate+up rows after SwiGLU of the
    experts and the shared expert plus the gate row, then every down row
    read through the combine (a one-hot score with the gate at sigmoid 0
    reads one expert; zero scores with the gate at sigmoid 1 the shared one)."""
    from mlx_vlm.models.activations import swiglu

    hidden, inter, gs, top_k = 2560, 640, 64, routed.TOP_K
    f32 = mx.float32
    mx.random.seed(40 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    def qmm(x, weights, group_size, b):
        return mx.quantized_matmul(x, *weights, transpose=True, group_size=group_size, bits=b)

    experts_gate_up = quantized((EXPERTS, 2 * inter, hidden), gs, bits)
    experts_down = quantized((EXPERTS, hidden, inter), gs, bits)
    shared_gate, shared_up = quantized((inter, hidden), 128, 8), quantized((inter, hidden), 128, 8)
    shared_down = quantized((hidden, inter), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = routed._Format
    gate_up_kernel = routed._gate_up_kernel(
        fmt(bits, gs, True), fmt(8, 128, True), fmt(8, 64, False)
    )
    down_kernel = routed._down_kernel(fmt(bits, gs, False), fmt(8, 128, False))
    for step in range(3):
        x = mx.random.normal((1, 1, hidden)) * (0.5 + step)
        ids = mx.random.permutation(EXPERTS)[:top_k].astype(mx.uint32)
        routed_gate_up = mx.gather_qmm(
            mx.expand_dims(x, (-2, -3)), *experts_gate_up, rhs_indices=ids.reshape(1, 1, top_k),
            transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        )
        gate, up = mx.split(routed_gate_up, 2, axis=-1)
        ref_h = mx.concatenate([
            swiglu(gate, up).reshape(-1),
            swiglu(qmm(x, shared_gate, 128, 8), qmm(x, shared_up, 128, 8)).reshape(-1),
            qmm(x, gate_row, 64, 8).reshape(-1),
        ])
        h = gate_up_kernel(
            inputs=[x, *experts_gate_up, ids, *shared_gate, *shared_up, *gate_row],
            template=[
                ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
            ],
            grid=(32, 2 * (1 + inter // 4 + top_k * inter // 4), 1),
            threadgroup=(32, 2, 1),
            output_shapes=[ref_h.shape],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h.view(mx.uint32), ref_h.view(mx.uint32)).item()

        ref_down = mx.gather_qmm(
            mx.expand_dims(h[: top_k * inter].reshape(top_k, inter), -2), *experts_down,
            rhs_indices=ids, transpose=True, group_size=gs, bits=bits, sorted_indices=False,
        ).reshape(top_k, hidden)
        ref_shared = qmm(h[top_k * inter : -1], shared_down, 128, 8)
        for j in range(top_k + 1):
            scores = (mx.arange(top_k) == j).astype(f32)
            saturated_gate = mx.array([-1e30 if j < top_k else 1e30], f32)
            row = down_kernel(
                inputs=[
                    mx.concatenate([h[:-1], saturated_gate]), *experts_down, *shared_down, ids,
                    scores,
                ],
                template=[
                    ("T", f32), ("K", inter), ("N", hidden), ("RPS", 4), ("KS", inter),
                    ("NPART", top_k + 1),
                ],
                grid=(32, (top_k + 1) * hidden // 4, 1),
                threadgroup=(32, top_k + 1, 1),
                output_shapes=[(hidden,)],
                output_dtypes=[f32],
            )[0]
            # The +0 folds of the combine turn -0 into +0, so compare values.
            assert mx.array_equal(row, ref_down[j] if j < top_k else ref_shared).item()


def test_experts_past_the_bound_view_are_read_from_the_stacked_weights():
    """The kernels bind a one-expert view and index the rest; a view copied
    out of the stacked buffer would read the wrong bytes for high experts."""
    from omlx.patches.qwen35_moe_router import fused_moe_combine

    block = _block(1024, 320, bits=5, experts=512)
    plan = routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), mx.bfloat16))
    assert all(o.shape[0] == 1 for o in plan.gate_up_operands + plan.down_operands)
    for ids in ([502 + i for i in range(10)], [511, 0, 500, 256, 1, 510, 3, 499, 7, 400]):
        x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
        inds = mx.array(ids, dtype=mx.uint32).reshape(1, 1, 10)
        scores = mx.softmax(mx.random.normal((1, 1, 10)), axis=-1).astype(mx.bfloat16)
        shared = block["shared_expert"](x)
        gate = block["shared_expert_gate"](x)
        ref = fused_moe_combine(block.switch_mlp(x, inds), scores, shared, gate)
        out = routed.routed_decode(plan, x, inds, scores, shared, gate)
        mx.eval(ref, out)
        assert _same_bits(ref, out)


def test_tied_router_logits_route_like_the_served_block():
    """Duplicated router rows give exactly tied probabilities; ties must pick
    the same experts, in the same order, with the same scores as the served
    router launches."""
    from omlx.patches.qwen35_moe_router import router_logits_row, softmax_topk_row

    block = _block(1024, 320, bits=5, experts=512)
    rows = block.gate.weight[:64]
    block.gate.weight = mx.concatenate([rows] * 8)[mx.random.permutation(512)]
    mx.eval(block.gate.weight)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(mx.bfloat16)
        # Both router launches engage: the gemv and the softmax + top-k.
        assert router_logits_row(x, block.gate.weight) is not None
        assert softmax_topk_row(block.gate(x), 10) is not None
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)



@pytest.mark.parametrize(
    "owner,names",
    [
        ("switch_mlp", ("gate_up_proj", "down_proj")),
        ("shared_expert", ("gate_proj", "up_proj", "down_proj")),
    ],
)
def test_plan_follows_replaced_weights(owner, names):
    block = _block(1024, 320, bits=5)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    first = routed.routed_decode_plan(block, x)
    donor = _block(1024, 320, bits=5, seed=7)
    for name in names:
        for key in ("weight", "scales", "biases"):
            block[owner][name][key] = donor[owner][name][key]
    assert routed.routed_decode_plan(block, x) is not first
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 1024, "inter": 512},  # down would take qmv_fast
        {"hidden": 960, "inter": 320},  # gate+up would take qmv
        {"hidden": 1024, "inter": 320, "top_k": 8},
        {"hidden": 1024, "inter": 320, "bits": 3},
    ],
)
def test_ineligible_shapes_keep_the_composed_body(kwargs):
    block = _block(**kwargs)
    hidden = kwargs["hidden"]
    x = mx.random.normal((1, 1, hidden)).astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


def test_prefill_verify_and_float_rows_keep_the_composed_body():
    block = _block(1024, 320)
    assert routed.routed_decode_plan(block, mx.zeros((1, 4, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((2, 1, 1024), dtype=mx.bfloat16)) is None
    assert routed.routed_decode_plan(block, mx.zeros((1, 1, 1024), dtype=mx.float16)) is None
    x = mx.random.normal((1, 5, 1024)).astype(mx.bfloat16)
    ref, out = _pair(block, x)
    assert _same_bits(ref, out)


def test_block_without_gate_up_fusion_is_not_routed():
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    args = SimpleNamespace(
        hidden_size=1024,
        moe_intermediate_size=320,
        shared_expert_intermediate_size=320,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    x = mx.zeros((1, 1, 1024), dtype=mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


def test_kernel_failure_falls_back_once(monkeypatch):
    block = _block(1024, 320)
    x = mx.random.normal((1, 1, 1024)).astype(mx.bfloat16)
    routed._DISABLED = True
    ref = block(x)
    routed._DISABLED = False

    def broken(*args):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(routed, "routed_decode", broken)
    out = block(x)
    mx.eval(ref, out)
    assert _same_bits(ref, out)
    assert routed._DISABLED
    assert routed.routed_decode_plan(block, x) is None


def test_apply_requires_the_fused_router(_patched_block):
    cls = _patched_block
    del cls._omlx_routed_decode
    cls._omlx_router_fused = False
    assert not routed.apply_qwen35_moe_routed_decode_patch()


def test_apply_is_idempotent(_patched_block):
    cls = _patched_block
    call = cls.__call__
    assert routed.apply_qwen35_moe_routed_decode_patch()
    assert cls.__call__ is call
