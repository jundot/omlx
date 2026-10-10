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
    hidden,
    inter,
    top_k=10,
    bits=4,
    group_size=64,
    seed=0,
    experts=EXPERTS,
    quantized_shared=True,
    dtype=mx.bfloat16,
):
    """A block laid out like Qwen3.8-Flash-Next oQ: quantized routed experts,
    8-bit shared expert (gs128 where the shape allows), 8-bit gs64
    shared-expert gate, ``dtype`` router. ``quantized_shared=False`` keeps the
    shared expert and its gate in ``dtype``."""
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
    block.set_dtype(dtype)
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


@pytest.mark.parametrize("experts", [EXPERTS, 512])
def test_bf16_shared_expert_stays_composed_and_bit_identical(experts):
    # 512 experts select inside the gate+up launch, 32 in the routing launch.
    # The 512-expert block takes the smaller shape to fit CI runner memory.
    hidden, inter = (2560, 640) if experts == EXPERTS else (1024, 320)
    block = _block(hidden, inter, bits=5, quantized_shared=False, experts=experts)
    for step in range(4):
        x = (mx.random.normal((1, 1, hidden)) * (0.5 + step)).astype(mx.bfloat16)
        plan = routed.routed_decode_plan(block, x)
        assert not plan.fold
        assert routed._topk_folds(plan, block.gate(x)) == (experts == 512)
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

    hidden, inter, gs, top_k = 2560, 640, 64, 10
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
                ("TOPK", top_k),
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
                    ("T", f32), ("K", inter), ("N", hidden), ("RPS", routed._down_rows(1)),
                    ("KS", inter), ("NPART", top_k + 1), ("TOPK", top_k),
                ],
                grid=(32, (top_k + 1) * hidden // routed._down_rows(1), 1),
                threadgroup=(32, top_k + 1, 1),
                output_shapes=[(hidden,)],
                output_dtypes=[f32],
            )[0]
            # The +0 folds of the combine turn -0 into +0, so compare values.
            assert mx.array_equal(row, ref_down[j] if j < top_k else ref_shared).item()


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_folded_routing_matches_the_routing_launch(bits):
    """The gate+up launch that selects its rows' experts itself, in FP32 (BF16
    outputs hide one-ulp differences): per row, the selection and scores of
    the routing launch, and the gate+up rows of the launch fed with them.
    Router logits repeat, so probabilities tie."""
    from omlx.patches import qwen35_moe_router as router

    hidden, inter, gs, experts, rows, top_k = 1024, 384, 64, 128, 3, 10
    f32 = mx.float32
    mx.random.seed(50 + bits)

    def quantized(shape, group_size, b):
        return mx.quantize(mx.random.normal(shape) * 0.05, group_size, b)

    experts_gate_up = quantized((experts, 2 * inter, hidden), gs, bits)
    shared = quantized((inter, hidden), 128, 8) + quantized((inter, hidden), 128, 8)
    gate_row = quantized((1, hidden), 64, 8)
    fmt = routed._Format
    formats = (fmt(bits, gs, True), fmt(8, 128, True), fmt(8, 64, False))
    x = mx.random.normal((rows, hidden))
    logits = mx.random.normal((rows, experts // 2)) * 2
    logits = mx.concatenate([logits, logits[:, ::-1]], axis=-1)
    template = [
        ("T", f32), ("K", hidden), ("NI", inter), ("RPS", 2), ("NSG", 2), ("NS", inter),
        ("TOPK", top_k),
    ]
    width = top_k * inter + inter + 1
    blocks = 1 + inter // 4 + top_k * inter // 4
    h, inds, scores = routed._gate_up_topk_kernel(*formats)(
        inputs=[x, *experts_gate_up, logits, *shared, *gate_row],
        template=template + [("NE", experts), ("M", rows), ("YW", width)],
        grid=(32, 2 * blocks * rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, width), (rows, top_k), (rows, top_k)],
        output_dtypes=[f32, mx.uint32, f32],
    )
    mx.eval(router.softmax_topk_rows(logits.astype(mx.bfloat16), top_k))
    ref_inds, ref_scores = router._SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", f32), ("NE", experts), ("K", top_k)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, f32],
    )
    assert mx.array_equal(inds, ref_inds).item()
    assert mx.array_equal(scores.view(mx.uint32), ref_scores.view(mx.uint32)).item()
    for r in range(rows):
        ref_h = routed._gate_up_kernel(*formats)(
            inputs=[x[r], *experts_gate_up, ref_inds[r], *shared, *gate_row],
            template=template,
            grid=(32, 2 * blocks, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(width,)],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h[r].view(mx.uint32), ref_h.view(mx.uint32)).item()


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



@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("top_k", [8, 10])
@pytest.mark.parametrize("bits", [4, 5, 6, 8])
def test_fast_down_and_top8_are_bit_identical(dtype, top_k, bits):
    """Qwen3.5/3.6-35B-A3B layout: hidden 2048 and intermediate 512 put the
    down projection on ``qmv_fast`` too."""
    block = _block(1024, 512, bits=bits, top_k=top_k, dtype=dtype)
    for step in range(8):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = routed.routed_decode_plan(block, x)
        assert plan.fold and plan.dtype == dtype and plan.top_k == top_k
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert not routed._DISABLED


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("quantized_shared", [True, False])
def test_top8_routing_folded_into_gate_up_is_bit_identical(dtype, quantized_shared):
    """128 experts select inside the gate+up launch; an unquantized shared
    expert runs composed and enters the combine."""
    block = _block(
        1024, 512, top_k=8, experts=128, quantized_shared=quantized_shared, dtype=dtype
    )
    for step in range(4):
        x = (mx.random.normal((1, 1, 1024)) * (0.5 + step)).astype(dtype)
        plan = routed.routed_decode_plan(block, x)
        assert plan.fold == quantized_shared
        assert routed._topk_folds(plan, block.gate(x))
        ref, out = _pair(block, x)
        assert _same_bits(ref, out)
    assert routed._PROVEN and not routed._DISABLED


def test_mismatched_dtypes_keep_the_composed_body():
    block = _block(1024, 512, top_k=8, dtype=mx.float16)
    x = mx.zeros((1, 1, 1024), mx.float16)
    # A router of another dtype runs as the block's own linear.
    gate = block.gate["weight"]
    block.gate["weight"] = gate.astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x).router_logits is None
    block.gate["weight"] = gate
    assert routed.routed_decode_plan(block, x.astype(mx.bfloat16)) is None
    down = block.switch_mlp.down_proj
    down["scales"] = down["scales"].astype(mx.bfloat16)
    down["biases"] = down["biases"].astype(mx.bfloat16)
    assert routed.routed_decode_plan(block, x) is None


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
        {"hidden": 960, "inter": 320},  # gate+up would take qmv
        {"hidden": 1024, "inter": 320, "top_k": 6},
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
