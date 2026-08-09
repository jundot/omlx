
# SPDX-License-Identifier: Apache-2.0
"""Escha-W2 native trellis MoE patch for mlx-lm qwen3_5_moe.

Checkpoints produced with ``tools/convert_escha_mlx.py --expert-format trellis``
ship the routed experts **verbatim** in the eschamoe/EXL3 packed form
(``switch_mlp.<proj>.escha_code/rin/rout``) instead of affine-quantized weights.
This patch swaps mlx-lm's ``SparseMoeBlock`` for a trellis variant that decodes
those codes on the fly with the Metal kernel in ``omlx.custom_kernels.escha``:

    xh   = had128(x * rin)          # MLX ops (Hadamard is symmetric orthonormal)
    ypre = trellis_qgemm(xh, code)  # fused decode+GEMM (Metal)
    y    = had128(ypre) * rout

Quality: the experts keep the original trellis codebook exactly (f16 decode,
bit-identical to the reference runtime), instead of a second affine
quantization.  The router / shared expert / attention / norms are untouched
and load through the stock mlx-lm tree, so this slots into omlx's normal
pre-load patch dispatch.

Activation: ``maybe_apply_pre_load_patches`` calls ``apply_escha_trellis_patch``
when ``config.json`` declares ``quantization_config.quant_method == "eschamoe"``.
The class swap is process-wide but inert for non-escha checkpoints: the
replacement factory delegates to the original ``SparseMoeBlock`` unless the
loaded config flagged escha mode.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.activations import silu

logger = logging.getLogger(__name__)

_PATCHED = False
_ESCHA_MODE = False
_ORIGINAL_SPARSE_MOE_BLOCK = None
_ORIGINAL_MODEL_ARGS_TO_DICT = None

_H128 = None


def _h128() -> mx.array:
    """Normalized Sylvester-Hadamard H(128), f32. Matches the escha reference."""
    global _H128
    if _H128 is None:
        h = mx.array([[1.0]])
        while h.shape[0] < 128:
            h = mx.concatenate(
                [mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0
            )
        _H128 = mx.array(h * (1.0 / (128.0 ** 0.5)))
    return _H128


def _had128(x: mx.array) -> mx.array:
    """Blockwise-128 orthonormal Hadamard along the last axis."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    y = x.reshape(*lead, n // 128, 128)
    out = mx.matmul(y, _h128())
    return out.reshape(*lead, n)


class _TrellisProj(nn.Module):
    """One routed projection: packed EXL3 codes + per-channel scales."""

    def __init__(self, E: int, inp: int, outp: int, K: int):
        super().__init__()
        self.escha_code = mx.zeros((E, inp // 16, outp // 16, 16 * K), mx.int16)
        self.escha_rin = mx.zeros((E, inp), mx.float16)
        self.escha_rout = mx.zeros((E, outp), mx.float16)


class TrellisSwitchMLP(nn.Module):
    """Decode-on-the-fly routed experts (gate_up 2-bit, down 3-bit)."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        E = num_experts
        self.gate_up_proj = _TrellisProj(E, hidden, 2 * intermediate, 2)
        self.down_proj = _TrellisProj(E, intermediate, hidden, 3)

    def _project(self, proj, x, eids, K):
        from omlx.custom_kernels.escha import fast as _fast

        rin = proj.escha_rin[eids].astype(mx.float32)
        rout = proj.escha_rout[eids].astype(mx.float32)
        # The custom kernel must see materialized buffers: pass only evaluated
        # inputs and materialize its output before any dependent op, otherwise
        # lazy graph nodes alias/reuse storage and corrupt the result.
        scaled = x * rin
        mx.eval(scaled)                      # kernel inputs must be materialized
        xh = _had128(scaled)
        mx.eval(xh)
        p = _fast.eschamoe_gather_qgemm(
            mx.asarray(xh, mx.float32), proj.escha_code, eids, K
        )
        return _had128(p) * rout             # output stays lazy (per-layer eval
                                             # in omlx/streaming)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        return self.forward(x, indices)

    def forward(self, x: mx.array, indices: mx.array) -> mx.array:
        """x [..., H], indices [..., topk] -> [..., topk, intermediate]
        (leading dims are flattened for the gather, restored on return)."""
        lead = x.shape[:-1]
        k = indices.shape[-1]
        N = 1
        for d in lead:
            N *= int(d)
        x2 = x.reshape(N, -1)
        inds2 = indices.reshape(N, k)
        xg = mx.repeat(x2, k, axis=0)                      # [M, H]
        flat = inds2.reshape(-1)                           # [M]
        order = mx.argsort(flat, axis=0)
        eids = mx.take(flat, order, axis=0)
        xs = mx.take(xg, order, axis=0)

        gu = self._project(self.gate_up_proj, xs, eids, 2)          # [M, 2I]
        gate, up = gu[..., :512], gu[..., 512:]
        act = silu(gate) * up
        mx.eval(act)                                                   # [M, I]
        out = self._project(self.down_proj, act, eids, 3)             # [M, H]

        inv = mx.argsort(order, axis=0)
        y = mx.take(out, inv, axis=0)
        return y.reshape(*lead, k, -1)


class _EschaSparseMoeBlock(nn.Module):
    """Drop-in for Qwen3NextSparseMoeBlock with a trellis switch_mlp."""

    def __init__(self, args: Any):
        super().__init__()
        dim = args.hidden_size
        intermediate_size = args.moe_intermediate_size
        self.norm_topk_prob = args.norm_topk_prob
        self.num_experts = num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok

        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.switch_mlp = TrellisSwitchMLP(num_experts, dim, intermediate_size)
        from mlx_lm.models.qwen3_next import Qwen3NextMLP

        self.shared_expert = Qwen3NextMLP(dim, intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        shared_y = self.shared_expert(x)
        shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
        return y + shared_y


def set_escha_mode(flag: bool) -> None:
    global _ESCHA_MODE
    _ESCHA_MODE = bool(flag)


def is_escha_mode() -> bool:
    return _ESCHA_MODE


def apply_escha_trellis_patch() -> bool:
    """Install the factory swap. Idempotent; inert for non-escha checkpoints."""
    global _PATCHED, _ORIGINAL_SPARSE_MOE_BLOCK, _ORIGINAL_MODEL_ARGS_TO_DICT
    if _PATCHED:
        return True
    try:
        import mlx_lm.models.qwen3_5 as _q35

        _ORIGINAL_SPARSE_MOE_BLOCK = _q35.SparseMoeBlock

        def _sparse_moe_factory(args):
            if _ESCHA_MODE:
                return _EschaSparseMoeBlock(args)
            return _ORIGINAL_SPARSE_MOE_BLOCK(args)

        _q35.SparseMoeBlock = _sparse_moe_factory

        # Flag the mode from the checkpoint config *before* the tree is built.
        for _mod in ("qwen3_5", "qwen3_5_moe"):
            try:
                mod = __import__(f"mlx_lm.models.{_mod}", fromlist=["ModelArgs"])
                orig = mod.ModelArgs.from_dict.__func__
                _ORIGINAL_MODEL_ARGS_TO_DICT = orig

                @classmethod
                def _wrapped_from_dict(cls, params, _orig=orig):
                    qc = params.get("quantization_config") or {}
                    set_escha_mode(
                        isinstance(qc, dict)
                        and qc.get("quant_method") == "eschamoe"
                    )
                    return _orig(cls, params)

                mod.ModelArgs.from_dict = _wrapped_from_dict
            except Exception:  # pragma: no cover - one arch may be absent
                continue

        _PATCHED = True
        logger.info("escha trellis patch installed (idempotent)")
        return True
    except Exception as exc:  # pragma: no cover
        logger.debug("escha trellis patch unavailable: %s", exc)
        return False
