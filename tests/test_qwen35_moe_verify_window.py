# SPDX-License-Identifier: Apache-2.0
"""Row-exact verify windows through the fused routed-expert kernels.

Every row of a verify window must equal the fused one-token decode of that
row bit for bit (the served serial path), and must equal what the
verifier's composed MoE returned for the window before."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed
from omlx.patches import qwen35_moe_router as router
from omlx.patches import qwen35_verify_qmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# Qwen3.8-Flash-Next oQ5e: 512 experts, top-10, hidden 2560, expert width 640.
HIDDEN, INTER, EXPERTS = 2560, 640, 512
# Quantized blocks keep the expert shape but not the count: 512 experts peak
# past the CI runner's memory. The fused router needs HIDDEN < 16 x experts
# and experts % 128 == 0.
BLOCK_EXPERTS = 256


class _FakeQwen4Model:
    pass


_FakeQwen4Model.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


@pytest.fixture(autouse=True)
def _patched(monkeypatch):
    """Served patch chain: verify linears, fused router, fused routed decode
    (which installs the verify-window entry). Restores the block class."""
    from mlx_vlm.models.qwen3_5_moe import language as vlm_moe

    from omlx.patches.mlx_vlm_mtp import qwen35_verify_linear

    qwen35_verify_qmm.apply_verify_qmm_patch()
    qwen35_verify_linear.apply()
    assert router.apply_qwen35_moe_router_patch()
    cls = vlm_moe.Qwen3_5MoeSparseMoeBlock
    call = cls.__call__
    had_flag = "_omlx_routed_decode" in cls.__dict__
    for name in ("_DISABLED", "_PROVEN", "_WINDOW_DISABLED", "_WINDOW_PROVEN"):
        monkeypatch.setattr(routed, name, False)
    monkeypatch.setattr(routed, "_VERIFY_WINDOW", True)
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    cls.__call__ = call
    if not had_flag and "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


_BLOCKS: dict = {}


def _quantized_experts(experts, out_dims, in_dims, bits, chunk=32):
    """Stacked gs64 expert weights, quantized a few experts at a time so the
    FP32 draws never exceed one chunk."""
    scale = in_dims**-0.5
    parts = []
    for _ in range(0, experts, chunk):
        w = mx.random.uniform(-scale, scale, (chunk, out_dims, in_dims)).astype(mx.bfloat16)
        parts.append(mx.quantize(w, 64, bits))
        mx.eval(parts[-1])
    return tuple(mx.concatenate(p) for p in zip(*parts, strict=True))


def _block(seed=0, bits=5, experts=BLOCK_EXPERTS):
    """A real-shape oQ block: quantized routed experts, 8-bit gs128 shared
    expert, 8-bit gs64 shared-expert gate, bf16 router."""
    key = (seed, bits, experts)
    if key in _BLOCKS:
        return _BLOCKS[key]
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        num_experts=experts,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        layer = getattr(sm, name).to_quantized(64, bits)
        layer.weight, layer.scales, layer.biases = _quantized_experts(
            experts, layer.output_dims, layer.input_dims, bits
        )
        setattr(sm, name, layer)
    shared = block.shared_expert
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), 128, 8))
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    _BLOCKS.clear()  # one real-shape block resident at a time
    _BLOCKS[key] = block
    return block


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def _serial(block, x):
    """The served one-token decode of every row (the fused routed call)."""
    rows = x.reshape(-1, 1, 1, x.shape[-1])
    assert routed.routed_decode_plan(block, rows[0]).fold
    return mx.concatenate([block(row) for row in rows]).reshape(x.shape)


def _verify(block, x, window):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    routed._VERIFY_WINDOW = window
    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=True)
    try:
        out = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
        routed._VERIFY_WINDOW = True
    mx.eval(out)
    return out

def _check(block, x, engaged):
    """Window rows == one-token rows == the composed verifier's rows (the
    latter only up to the shared router cap: past it the composed verifier
    runs stock argpartition routing, not the one-token arithmetic)."""
    count = len(engaged)
    new = _verify(block, x, window=True)
    ref = _serial(block, x)
    old = _verify(block, x, window=False)
    mx.eval(ref, old)
    # Engaged for the window, declined with the kill switch set.
    assert engaged[count:] == [True, False]
    assert _same_bits(new, ref)
    if x.size // HIDDEN <= router._MAX_ROWS:
        assert _same_bits(new, old)


@pytest.fixture
def engaged(monkeypatch):
    calls = []
    window = routed.routed_verify_window

    def spy(block, x):
        y = window(block, x)
        calls.append(y is not None)
        return y

    monkeypatch.setattr(routed, "routed_verify_window", spy)
    return calls


def _inputs(rows, step, batch=1):
    shape = (batch, rows, HIDDEN)
    if step == 0:  # independent rows, mostly distinct experts
        return (mx.random.normal(shape) * 1.5).astype(mx.bfloat16)
    if step == 1:  # consecutive-token-like rows sharing many experts
        base = mx.random.normal((batch, 1, HIDDEN))
        return (0.97 * base + 0.25 * mx.random.normal(shape)).astype(mx.bfloat16)
    # identical rows: every expert serves every row
    row = mx.random.normal((batch, 1, HIDDEN)).astype(mx.bfloat16)
    return mx.broadcast_to(row, shape)


@pytest.fixture(params=[True, False], ids=["topk_fold", "topk_launch"])
def topk_fold(request, monkeypatch):
    """Routing folded into the gate+up launch, or OMLX_QWEN35_MOE_TOPK_FOLD=0
    (the separate routing launch); cached plans are rebuilt either way."""

    def drop_plans():
        for block in _BLOCKS.values():
            block.__dict__.pop("_omlx_routed_decode_plan", None)

    monkeypatch.setattr(routed, "_TOPK_FOLD", request.param)
    drop_plans()
    yield request.param
    drop_plans()


@pytest.mark.parametrize("seed", [0, 1])
def test_window_rows_equal_one_token_decode(seed, topk_fold, engaged):
    block = _block(seed)
    plan = routed.routed_decode_plan(block, mx.zeros((1, 1, HIDDEN), mx.bfloat16))
    assert (plan.topk_kernel is not None) == topk_fold
    for rows in range(1, routed.WINDOW_MAX_ROWS + 1):
        for step in range(3):
            _check(block, _inputs(rows, step), engaged)
    # Batched verify rows (requests x drafts) are rows too.
    _check(block, _inputs(3, 1, batch=2), engaged)


def test_four_bit_window_rows_equal_one_token_decode(engaged):
    block = _block(2, bits=4)
    for rows in (2, 3, 4, 8):
        for step in range(3):
            _check(block, _inputs(rows, step), engaged)


def test_router_ties_and_high_experts(engaged):
    """Exactly tied and one-ulp-apart router probabilities must select the
    same experts in the same order as the one-token router; experts at the
    end of the stacked weights are read past the bound one-expert view."""
    block = _block(3)
    weight = block.gate.weight
    experts = BLOCK_EXPERTS
    # Exact ties: experts / 8 distinct router rows, each repeated 8 times.
    tied = mx.concatenate([weight[: experts // 8]] * 8)[mx.random.permutation(experts)]
    # Near ties: experts / 16 rows repeated 16 times, each copy one weight
    # element about one bf16 ulp away, so logits tie or differ in the last bit.
    column = (mx.arange(experts) * 7) % HIDDEN
    bump = (mx.arange(HIDDEN)[None, :] == column[:, None]).astype(mx.float32) * 2**-13
    near = mx.concatenate([weight[: experts // 16]] * 16) + bump
    # High experts: the last 14 router rows lean on a direction the inputs
    # carry, so every row routes to the last 14 experts.
    direction = mx.random.normal((HIDDEN,))
    last = (mx.arange(experts) >= experts - 14)[:, None]
    high = weight + mx.where(last, 0.02 * direction, 0.0)
    for gate, lean in ((tied, 0.0), (near, 0.0), (high, 1.0)):
        block.gate.weight = gate.astype(weight.dtype)
        mx.eval(block.gate.weight)
        for rows in (2, 3, 4, 7):
            for step in range(3):
                x = (_inputs(rows, step) + lean * direction).astype(mx.bfloat16)
                _check(block, x, engaged)
                if lean:
                    logits = router.router_gemv(block.gate.weight)(x.reshape(rows, HIDDEN))
                    inds, _ = router.softmax_topk_rows(logits, 10)
                    assert mx.min(inds).item() >= experts - 14
    block.gate.weight = weight


def test_kernel_failure_falls_back_to_the_composed_verifier(monkeypatch, engaged):
    block = _block(0)
    x = _inputs(4, 1)
    old = _verify(block, x, window=False)

    def broken(*args):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(routed, "_window_down", broken)
    out = _verify(block, x, window=True)
    assert engaged[-1] is False and routed._WINDOW_DISABLED
    assert _same_bits(out, old)


def _quantized(shape, group_size, bits):
    return mx.quantize(mx.random.normal(shape) * 0.05, group_size, bits)


@pytest.mark.parametrize("bits", [4, 5])
def test_fp32_window_kernels_equal_one_token_kernels(bits):
    """BF16 outputs hide one-ulp FP32 differences: run the window launches
    and the one-token launches in FP32 (the one-token ones equal MLX's FP32
    mat-vecs, see test_qwen35_moe_routed_decode) and compare row by row. The
    rows share experts at different slots and read the last experts."""
    f32, top_k, experts, rows = mx.float32, routed.TOP_K, 64, 4
    mx.random.seed(60 + bits)
    fmt = routed._Format
    routed_gu, shared_gu, gate_fmt = fmt(bits, 64, True), fmt(8, 128, True), fmt(8, 64, False)
    routed_d, shared_d = fmt(bits, 64, False), fmt(8, 128, False)
    experts_gate_up = _quantized((experts, 2 * INTER, HIDDEN), 64, bits)
    experts_down = _quantized((experts, HIDDEN, INTER), 64, bits)
    shared = _quantized((INTER, HIDDEN), 128, 8) + _quantized((INTER, HIDDEN), 128, 8)
    gate_row = _quantized((1, HIDDEN), 64, 8)
    shared_down = _quantized((HIDDEN, INTER), 128, 8)
    first = mx.random.permutation(experts)[:top_k]
    ids = mx.stack(
        [
            first,
            first[::-1],  # same experts, other slots
            mx.concatenate([first[:5], mx.arange(experts - 5, experts)]),
            mx.arange(experts - top_k, experts)[::-1],  # the last experts
        ]
    ).astype(mx.uint32)
    x = mx.random.normal((rows, HIDDEN))
    scores = mx.softmax(mx.random.normal((rows, top_k)), axis=-1)
    gate_up_template = [
        ("T", f32), ("K", HIDDEN), ("NI", INTER), ("RPS", 2), ("NSG", 2), ("NS", INTER),
    ]
    blocks = 1 + INTER // 4 + top_k * INTER // 4
    width = top_k * INTER + INTER + 1
    h = routed._gate_up_window_kernel(routed_gu, shared_gu, gate_fmt)(
        inputs=[x, *experts_gate_up, ids, *shared, *gate_row],
        template=gate_up_template + [("M", rows)],
        grid=(32, 2 * blocks * rows, 1),
        threadgroup=(32, 2, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[f32],
    )[0]
    down_template = [
        ("T", f32), ("K", INTER), ("N", HIDDEN), ("RPS", 4), ("KS", INTER), ("NPART", top_k + 1),
    ]
    y = routed._down_window_kernel(routed_d, shared_d)(
        inputs=[h, *experts_down, *shared_down, ids, scores],
        template=down_template + [("M", rows)],
        grid=(32, (top_k + 1) * HIDDEN // 4 * rows, 1),
        threadgroup=(32, top_k + 1, 1),
        output_shapes=[(rows, HIDDEN)],
        output_dtypes=[f32],
    )[0]
    for r in range(rows):
        h_r = routed._gate_up_kernel(routed_gu, shared_gu, gate_fmt)(
            inputs=[x[r], *experts_gate_up, ids[r], *shared, *gate_row],
            template=gate_up_template,
            grid=(32, 2 * blocks, 1),
            threadgroup=(32, 2, 1),
            output_shapes=[(width,)],
            output_dtypes=[f32],
        )[0]
        y_r = routed._down_kernel(routed_d, shared_d)(
            inputs=[h_r, *experts_down, *shared_down, ids[r], scores[r]],
            template=down_template,
            grid=(32, (top_k + 1) * HIDDEN // 4, 1),
            threadgroup=(32, top_k + 1, 1),
            output_shapes=[(HIDDEN,)],
            output_dtypes=[f32],
        )[0]
        assert mx.array_equal(h[r].view(mx.uint32), h_r.view(mx.uint32)).item()
        assert mx.array_equal(y[r].view(mx.uint32), y_r.view(mx.uint32)).item()


def test_fp32_window_router_equals_one_row_router():
    """The router gemv and softmax + top-k of a window, in FP32, row by row
    against the one-row launches (which reproduce MLX's)."""
    f32, rows = mx.float32, 5
    mx.random.seed(71)
    weight = mx.random.normal((EXPERTS, HIDDEN)) * 0.02
    # A few duplicated router rows give exactly tied probabilities.
    weight = mx.concatenate([weight[:480], weight[:32]])
    x = mx.random.normal((rows, HIDDEN))
    launch = router.router_gemv(weight.astype(mx.bfloat16))
    mx.eval(launch(x.astype(mx.bfloat16)))

    def gemv(x, rows):
        return router._GEMV_KERNEL(
            inputs=[x, weight],
            template=[("T", f32), ("K", HIDDEN), ("N", EXPERTS), ("M", rows), ("NSG", 4)],
            grid=(32, EXPERTS * rows, 1),
            threadgroup=(32, 4, 1),
            output_shapes=[(rows, EXPERTS)],
            output_dtypes=[f32],
        )[0]

    logits = gemv(x, rows)
    probe = (mx.random.normal((rows, EXPERTS)) * 3).astype(mx.bfloat16)
    mx.eval(router.softmax_topk_row(probe[:1], 10), router.softmax_topk_rows(probe, 10))
    inds, scores = router._SOFTMAX_TOPK_ROWS_KERNEL(
        inputs=[logits],
        template=[("T", f32), ("NE", EXPERTS), ("K", 10)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, 10), (rows, 10)],
        output_dtypes=[mx.uint32, f32],
    )
    for r in range(rows):
        logits_r = gemv(x[r], 1)[0]
        inds_r, scores_r = router._SOFTMAX_TOPK_KERNEL(
            inputs=[logits_r],
            template=[("T", f32), ("NE", EXPERTS), ("K", 10)],
            grid=(32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(10,), (10,)],
            output_dtypes=[mx.uint32, f32],
        )
        assert mx.array_equal(logits[r].view(mx.uint32), logits_r.view(mx.uint32)).item()
        assert mx.array_equal(inds[r], inds_r).item()
        assert mx.array_equal(scores[r].view(mx.uint32), scores_r.view(mx.uint32)).item()
