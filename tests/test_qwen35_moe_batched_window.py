# SPDX-License-Identifier: Apache-2.0
"""Batched MoE rows through the fused routed-expert window launches.

Batched decode (B >= 2) and batched verify windows (B >= 2, not row-exact
armed) of at most ``WINDOW_MAX_ROWS`` rows run ``routed_verify_window``:
every row must equal the served one-token decode of that row bit for bit.
The composed multi-row block they replace is not batch-invariant: its
combine sums in another order, so some rows differ from their one-token
decode by a bf16 ulp or two. One-row calls, wider calls and blocks outside
the fused layout keep their paths."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

import omlx.patches.qwen35_moe_routed_decode as routed
from omlx.patches import qwen35_moe_router as router
from omlx.patches import qwen35_verify_qmm

pytestmark = pytest.mark.skipif(not mx.metal.is_available(), reason="requires Metal")

# The Flash-Next expert width (640: the down projection runs MLX's qmv with
# a K tail) on a small hidden size and expert count. The fused layout needs
# hidden % 512 == 0, experts % 128 == 0 and hidden < 16 x experts.
HIDDEN, INTER, EXPERTS = 1024, 640, 128


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
    monkeypatch.setattr(routed, "_BATCHED_WINDOW", True)
    assert routed.apply_qwen35_moe_routed_decode_patch()
    yield
    qwen35_verify_qmm.set_verify_qmm_armed(False)
    cls.__call__ = call
    if not had_flag and "_omlx_routed_decode" in cls.__dict__:
        delattr(cls, "_omlx_routed_decode")


_BLOCKS: dict = {}


def _block(seed=0, bits=5, group_size=64):
    """An oQ-style block: quantized routed experts (fused gate+up), 8-bit
    gs128 shared expert, 8-bit gs64 shared-expert gate, bf16 router."""
    key = (seed, bits, group_size)
    if key in _BLOCKS:
        return _BLOCKS[key]
    from mlx_vlm.models.qwen3_5_moe.language import Qwen3_5MoeSparseMoeBlock

    from omlx.patches.qwen35_moe_gate_up import apply_qwen35_moe_gate_up_fusion

    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=HIDDEN,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        num_experts=EXPERTS,
        num_experts_per_tok=10,
    )
    block = Qwen3_5MoeSparseMoeBlock(args)
    block.set_dtype(mx.bfloat16)
    sm = block.switch_mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(sm, name, getattr(sm, name).to_quantized(group_size, bits))
    shared = block.shared_expert
    for name in ("gate_proj", "up_proj", "down_proj"):
        setattr(shared, name, nn.QuantizedLinear.from_linear(getattr(shared, name), 128, 8))
    block.shared_expert_gate = nn.QuantizedLinear.from_linear(block.shared_expert_gate, 64, 8)
    block.eval()
    model = _FakeQwen4Model()
    model.named_modules = lambda: [("mlp.switch_mlp", sm)]
    assert apply_qwen35_moe_gate_up_fusion(model) == 1
    mx.eval(block.parameters())
    _BLOCKS.clear()  # one block resident at a time
    _BLOCKS[key] = block
    return block


@pytest.fixture
def engaged(monkeypatch):
    calls = []
    window = routed.routed_verify_window

    def spy(block, x):
        y = window(block, x)
        calls.append((tuple(x.shape), y is not None))
        return y

    monkeypatch.setattr(routed, "routed_verify_window", spy)
    return calls


def _same_bits(a, b):
    return a.shape == b.shape and mx.array_equal(a.view(mx.uint16), b.view(mx.uint16)).item()


def _serial(block, x):
    """The served one-token decode of every row (the fused routed call)."""
    rows = x.reshape(-1, 1, 1, x.shape[-1])
    assert routed.routed_decode_plan(block, rows[0]).fold
    out = mx.concatenate([block(row) for row in rows]).reshape(x.shape)
    mx.eval(out)
    return out


def _inputs(batch, length, seed):
    mx.random.seed(seed)
    return (mx.random.normal((batch, length, HIDDEN)) * 1.5).astype(mx.bfloat16)


def _decode(block, x, monkeypatch, batched=True):
    monkeypatch.setattr(routed, "_BATCHED_WINDOW", batched)
    out = block(x)
    mx.eval(out)
    monkeypatch.setattr(routed, "_BATCHED_WINDOW", True)
    return out


def _verify(block, x, row_exact=False):
    from mlx_vlm.models.qwen3_5.speculative_verifier import Qwen3_5BatchInvariantForward

    qwen35_verify_qmm.set_verify_qmm_armed(True, row_exact=row_exact)
    try:
        out = Qwen3_5BatchInvariantForward()._feed_forward(block, x)
    finally:
        qwen35_verify_qmm.set_verify_qmm_armed(False)
    mx.eval(out)
    return out


@pytest.mark.parametrize(
    "bits,group_size", [(4, 64), (5, 64), (6, 64), (8, 64), (5, 32), (6, 128)]
)
def test_batched_decode_rows_equal_one_token_decode(monkeypatch, engaged, bits, group_size):
    block = _block(bits, bits, group_size)
    for batch in (2, 3, 5, 8):
        x = _inputs(batch, 1, batch)
        out = _decode(block, x, monkeypatch)
        assert engaged == [((batch, 1, HIDDEN), True)]
        engaged.clear()
        assert _same_bits(out, _serial(block, x))


def test_batched_verify_rows_equal_one_token_decode(monkeypatch, engaged):
    block = _block(2)
    for batch, length in ((2, 1), (2, 3), (2, 4), (3, 2), (4, 2)):
        x = _inputs(batch, length, 10 * batch + length)
        out = _verify(block, x)
        assert engaged == [((batch, length, HIDDEN), True)]
        engaged.clear()
        assert _same_bits(out, _serial(block, x))
        # The kill switch keeps the verifier's multi-row MoE.
        monkeypatch.setattr(routed, "_BATCHED_WINDOW", False)
        _verify(block, x)
        monkeypatch.setattr(routed, "_BATCHED_WINDOW", True)
        assert engaged == []


def test_composed_batched_rows_are_not_one_token_decode(monkeypatch, engaged):
    """Negative control: the bit-for-bit check above can fail. With the kill
    switch set the composed block serves the batch, and some of its rows
    differ from the same rows' one-token decode; the window rows on the same
    inputs do not."""
    block = _block(5)
    differing = 0
    for seed in range(8):
        x = _inputs(8, 1, 100 + seed)
        ref = _serial(block, x)
        composed = _decode(block, x, monkeypatch, batched=False)
        assert engaged == []
        window = _decode(block, x, monkeypatch)
        assert engaged == [((8, 1, HIDDEN), True)]
        engaged.clear()
        assert _same_bits(window, ref)
        same_row = mx.all(composed.view(mx.uint16) == ref.view(mx.uint16), axis=(1, 2))
        differing += int((~same_row).sum().item())
        # Only the composed combine's summation order separates them.
        assert mx.allclose(composed, ref, rtol=0, atol=1e-2).item()
    assert differing > 0


def test_one_row_wide_and_ineligible_calls_keep_their_paths(monkeypatch, engaged):
    block = _block(1)
    mx.eval(block(_inputs(1, 1, 0)))  # B1 decode: the one-token launches
    mx.eval(block(_inputs(1, 4, 0)))  # B1 prefill chunk: composed
    assert engaged == []
    x = _inputs(3, 3, 0)  # 9 rows: over the window, composed
    out = _decode(block, x, monkeypatch)
    assert engaged == [((3, 3, HIDDEN), False)]
    assert _same_bits(out, _decode(block, x, monkeypatch, batched=False))
    engaged.clear()
    block = _block(1, bits=3)  # outside the fused formats: composed
    x = _inputs(2, 1, 0)
    out = _decode(block, x, monkeypatch)
    assert engaged == [((2, 1, HIDDEN), False)]
    assert _same_bits(out, _decode(block, x, monkeypatch, batched=False))


def test_kernel_failure_keeps_the_composed_block(monkeypatch, engaged):
    block = _block(4)
    x = _inputs(4, 1, 4)
    composed = _decode(block, x, monkeypatch, batched=False)

    def broken(*args):
        raise RuntimeError("no pipeline")

    monkeypatch.setattr(routed, "routed_window", broken)
    out = _decode(block, x, monkeypatch)
    assert engaged == [((4, 1, HIDDEN), False)] and routed._WINDOW_DISABLED
    assert _same_bits(out, composed)


def test_b1_verify_keeps_upstream_routing(engaged):
    block = _block(3)
    x = _inputs(1, 3, 5)
    _verify(block, x)  # unarmed B1 (row-exact verify disabled): composed
    assert engaged == []
    out = _verify(block, x, row_exact=True)  # the row-exact window
    assert engaged == [((1, 3, HIDDEN), True)]
    assert _same_bits(out, _serial(block, x))
