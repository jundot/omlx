# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-MoE blocks: the shared Qwen3.5 blocks with the sparse MoE block.

Attention, GatedDeltaNet, the K/V cache and the rotary rounding come from
``families/_qwen35_layers.py``, so they stay byte-identical to the dense
family this derives from; this module supplies the routed expert block and
the layer that binds it in place of the dense MLP.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ... import oqh
from .. import _qwen35_kernels as kernels
from .._qwen35_layers import (
    Attention,
    GatedDeltaNet,
    KVCache,
    Weights,
    causal_conv,
    rotary_frequencies,
    rotate,
)
from .._qwen35_layers import (
    Layer as _Layer,
)
from .._qwen35_plan import BANK_PROJECTIONS

__all__ = [
    "Attention",
    "GatedDeltaNet",
    "KVCache",
    "Layer",
    "MoE",
    "Weights",
    "causal_conv",
    "rotate",
    "rotary_frequencies",
    "stacked_sum",
]


# --- copied from qwen4_exp so this module carries no cross-family import


def stacked_sum(x: mx.array) -> mx.array:
    """``x.sum(-2)`` over fewer than 32 rows, rounded as on the host.

    MLX's Metal reduce adds the rows eight apart first and then those eight
    partial sums in order, all in the input dtype.
    """
    rows = x.shape[-2]
    if x.dtype == mx.float32 or rows >= 32:
        return x.sum(axis=-2)
    partial = []
    for first in range(min(rows, 8)):
        total = x[..., first, :]
        for row in range(first + 8, rows, 8):
            total = x[..., row, :] + total
        partial.append(total)
    total = partial[0]
    for part in partial[1:]:
        total = part + total
    return total


class MoE:
    """Routed expert MLPs plus a gated shared expert."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.top_k = text["num_experts_per_tok"]
        self.inner = text["moe_intermediate_size"]
        self.normalize = text.get("norm_topk_prob", True)
        self.router = w.linear("gate")
        self.shared_gate = w.linear("shared_expert.gate_proj")
        self.shared_up = w.linear("shared_expert.up_proj")
        self.shared_down = w.linear("shared_expert.down_proj")
        self.shared_scale = w.linear("shared_expert_gate")
        # Every 2-D linear here reads the layer's post-norm rows, so an oQh
        # rotation on one has to be the rotation the others see.
        oqh.check_shared(
            self.router, self.shared_gate, self.shared_up, self.shared_scale
        )
        # The linear that reads the layer's post-norm rows: an oQh one rotates
        # them, and the shared layer asks the block for it.
        self.reader = self.router
        # Rooted at "mlp." by the layer, so the expert banks resolve as
        # mlp.switch_mlp.*, the routers as mlp.gate and the shared expert as
        # mlp.shared_expert.*.
        banks = w.at("switch_mlp")
        # Gate and up share their input rows: one gather computes both.
        self.gate_up = self._bank(banks, BANK_PROJECTIONS[:2])
        self.down = self._bank(banks, BANK_PROJECTIONS[2:])
        self.combine = None
        self.gather = None
        self.swiglu = None
        self.expert_kernel = False

    @staticmethod
    def _bank(w: Weights, names: tuple[str, ...]) -> tuple:
        """(weight, scales, biases, quantization) of expert projections.

        Several projections of the same input are joined along their outputs.
        """
        if any(oqh.is_oqh(w.arrays, w.root + name) for name in names):
            # An oQh bank is a different packed layout; reading it as dense
            # would load the packed codes as weights without saying so.
            raise ValueError("oQh expert banks are not supported by this family")
        if f"{names[0]}.scales" not in w:
            # Dense banks are stored [experts, out, in]; gather_mm wants in first.
            parts = [w[f"{name}.weight"].swapaxes(-1, -2) for name in names]
            return mx.concatenate(parts, axis=-1), None, None, None
        joined = [
            mx.concatenate([w[f"{name}.{part}"] for name in names], axis=1)
            for part in ("weight", "scales", "biases")
        ]
        return *joined, w.quantization(names[0])

    def parameters(self) -> list[mx.array]:
        linears = (
            self.router,
            self.shared_gate,
            self.shared_up,
            self.shared_down,
            self.shared_scale,
        )
        banks = [a for bank in (self.gate_up, self.down) for a in bank[:3]]
        return [a for a in banks if a is not None] + [
            a for layer in linears for a in layer.parameters()
        ]

    def enable_kernels(self) -> None:
        """Pick the CUDA expert kernels this block may use, or none.

        Two independent reasons to use none, and the first is why this family
        never does today:

        * the expert banks are PACKED quantized tuples ``(uint32, scales,
          biases, quant)``. The expert kernels were written and validated
          against qwen4_exp's DENSE banks ``(weight, None, None, None)``; a
          kernel that read packed uint32 as numeric data would still produce a
          speedup and still diverge under a negative control while returning
          garbage on every long chunk. Disabled until a quantized kernel exists.
        * this family's ``kernels`` module is the dense one and carries no
          expert kernel at all, so the probes below would raise AttributeError.

        The reference paths (``rows[order // k]``, ``mx.gather_qmm``,
        ``_swiglu``, ``stacked_sum``) are the correctness baseline, not a
        degraded fallback.
        """
        self._log_dtypes()
        quantized = self.gate_up[3] is not None or self.down[3] is not None
        if quantized:
            kernels.logger.info(
                "MoE expert kernels DISABLED: expert banks are quantized "
                "(no quantized expert kernel exists); using gather_qmm/reference"
            )
            return
        if not hasattr(kernels, "combine_experts"):
            kernels.logger.info(
                "MoE expert kernels DISABLED: kernels module has no expert kernels"
            )
            return
        k, dtype = self.top_k, self.router.weight.dtype
        y = mx.random.normal((64 * k, 1, self.router.weight.shape[1])).astype(dtype)
        positions = mx.argsort(mx.random.uniform(shape=(64 * k,)))
        scores = mx.random.uniform(shape=(64, k)).astype(dtype)
        want = stacked_sum(y[positions].reshape(64, k, -1) * scores[..., None])
        if kernels.close(
            kernels.combine_experts(y, positions, scores), want, "MoE sum"
        ):
            self.combine = kernels.combine_experts
        x = mx.random.normal((64, self.router.weight.shape[1])).astype(dtype)
        src = mx.random.randint(0, 64, (64 * k,)).astype(mx.uint32)
        if x.shape[1] * dtype.size % 16:
            return
        if mx.array_equal(kernels.gather_rows(x, src), x[src]).item():
            self.gather = kernels.gather_rows
        else:
            kernels.logger.warning("Row gather kernel differs; using the reference")
        both = mx.random.normal((64, 1, 2 * self.inner)).astype(dtype)
        if dtype == mx.bfloat16 and self.inner % 8 == 0:
            if mx.array_equal(kernels.swiglu(both), self._swiglu(both)).item():
                self.swiglu = kernels.swiglu
            else:
                kernels.logger.warning("SwiGLU kernel differs; using the reference")
        widths = ((self.gate_up, x.shape[1]), (self.down, self.inner))
        if (
            dtype == mx.bfloat16
            and self.swiglu is not None
            and all(kernels.expert_gemm_fits(bank, width) for bank, width in widths)
        ):
            self.expert_kernel = self._experts_match(dtype)

    def _log_dtypes(self) -> None:
        """Log the router, gate and output dtypes once per layer at load.

        P0-1: the host router is bf16 and ``softmax(precise=True)`` returns the
        input dtype, so top-k is taken over bf16-rounded probabilities. If this
        path produced fp32 it would both change expert choice on near-ties and
        promote the residual stream for every downstream layer and the handoff
        tensor. Measured here rather than assumed.
        """
        w = self.router.weight
        probe = mx.random.normal((1, 8, w.shape[1])).astype(w.dtype)
        logits = self.router(probe)
        gates = mx.softmax(logits, axis=-1, precise=True)
        out = self(probe)
        mx.eval(gates, out)
        kernels.logger.info(
            "MoE dtypes: router_weight=%s logits=%s gates=%s output=%s output_dtype_is_bf16=%s",
            w.dtype,
            logits.dtype,
            gates.dtype,
            out.dtype,
            str(out.dtype) == "bfloat16",
        )

    def _experts_match(self, dtype: mx.Dtype) -> bool:
        """Whether the expert kernel path gives the bits of gather_qmm.

        Enough routes for the sorted gather, and one expert with more rows
        than a launch block holds at once.
        """
        tokens, count = 256, self.gate_up[0].shape[0]
        x = mx.random.normal((1, tokens, self.router.weight.shape[1])).astype(dtype)
        others = mx.random.randint(1, count, (1, tokens, self.top_k - 1))
        chosen = mx.concatenate([mx.zeros((1, tokens, 1), mx.int32), others], -1)
        self.expert_kernel = False
        want = self.experts(x, chosen)[0]
        self.expert_kernel = True
        if mx.array_equal(self.experts(x, chosen)[0], want).item():
            return True
        kernels.logger.warning("Expert kernel differs; using gather_qmm")
        return False

    @staticmethod
    def _gather(x: mx.array, bank: tuple, experts: mx.array) -> mx.array:
        """Rows [n, 1, in] through each row's expert; ``experts`` is sorted."""
        weight, scales, biases, quant = bank
        if quant is None:
            return mx.gather_mm(x, weight, rhs_indices=experts, sorted_indices=True)
        return mx.gather_qmm(
            x,
            weight,
            scales,
            biases,
            rhs_indices=experts,
            transpose=True,
            group_size=quant["group_size"],
            bits=quant["bits"],
            mode=quant.get("mode", "affine"),
            sorted_indices=True,
        )

    @staticmethod
    def _swiglu(both: mx.array) -> mx.array:
        inner = both.shape[-1] // 2
        return nn.silu(both[..., :inner]) * both[..., inner:]

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """(expert ids, weights) [1, rows, top_k] from the router logits."""
        k = self.top_k
        gates = mx.softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, chosen, axis=-1)
        if self.normalize:
            # The host's op, verbatim: mlx_lm Qwen3NextSparseMoeBlock does
            #   scores = scores / scores.sum(axis=-1, keepdims=True)
            # qwen4_exp's family keeps a sequential variant because ITS host adds
            # the k values one at a time; this family deliberately does not, so
            # the divisor is the identical reduction the host runs. With k=8 in
            # bf16 a tree reduce and a sequential chain can differ by an ulp.
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return chosen, scores

    def experts(self, x: mx.array, chosen: mx.array) -> tuple[mx.array, mx.array]:
        """Expert outputs [rows * top_k, 1, hidden] sorted by expert, and the
        row of each (token, choice) in them."""
        assert x.shape[0] == 1, f"the expert path assumes batch one, got {x.shape}"
        k = self.top_k
        flat = chosen.reshape(-1)
        order = mx.argsort(flat)
        ids = flat[order].astype(mx.uint32)
        rows = x.reshape(x.shape[1], -1)
        # gather_qmm takes its sorted expert kernel, whose sums this kernel
        # reproduces, from four routes an expert on.
        if self.expert_kernel and ids.shape[0] >= 4 * self.gate_up[0].shape[0]:
            # Each route reads its token's row; gate, up and SwiGLU in one launch.
            src = (order // k).astype(mx.uint32)
            offsets = kernels.expert_offsets(ids, self.gate_up[0].shape[0])
            hidden = kernels.expert_gemm(rows, self.gate_up, offsets, src, swiglu=True)
            y = kernels.expert_gemm(hidden, self.down, offsets)
            return y[:, None], mx.argsort(order)
        if self.gather is not None:
            routed = self.gather(rows, (order // k).astype(mx.uint32))[:, None]
        else:
            routed = rows[order // k][:, None]
        both = self._gather(routed, self.gate_up, ids)
        hidden = (self.swiglu or self._swiglu)(both)
        return self._gather(hidden, self.down, ids), mx.argsort(order)

    def combined(self, y: mx.array, positions: mx.array, scores: mx.array) -> mx.array:
        """Each token's expert outputs times their weights, summed."""
        if self.combine is not None:
            return self.combine(y, positions, scores[0])[None]
        rows, k = scores.shape[1:]
        return stacked_sum(y[positions].reshape(1, rows, k, -1) * scores[..., None])

    def shared(self, x: mx.array, rot: mx.array | None = None) -> mx.array:
        if self.shared_down.rotation is None:
            shared = self.shared_down(nn.silu(self.shared_gate(x)) * self.shared_up(x))
        else:
            # oQh: the activation is taken in float32 and rotated for down.
            rot = self.shared_gate.rotated(x) if rot is None else rot
            act = oqh.act_rotate(
                self.shared_gate(x, rot),
                self.shared_up(x, rot),
                *self.shared_down.rotation,
            )
            shared = self.shared_down(act, act)
        return mx.sigmoid(self.shared_scale(x, rot)) * shared

    def __call__(self, x: mx.array, rot: mx.array | None = None) -> mx.array:
        chosen, scores = self.route(self.router(x, rot))
        return self.combined(*self.experts(x, chosen), scores) + self.shared(x, rot)


class Layer(_Layer):
    """A qwen3_5_moe layer: the shared blocks with the routed expert block."""

    def __init__(self, w: Weights, text: dict[str, Any], linear: bool):
        super().__init__(w, text, linear, mlp_block=MoE)
