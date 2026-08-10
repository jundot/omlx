
# SPDX-License-Identifier: Apache-2.0
"""Escha-W2 native trellis MoE patch for mlx-lm qwen3_5_moe.

Checkpoints produced with ``tools/convert_escha_mlx.py --expert-format trellis``
ship the routed experts verbatim in the eschamoe/EXL3 packed form
(``switch_mlp.<proj>.escha_code/rin/rout``). This patch swaps mlx-lm's
``SparseMoeBlock`` for a trellis variant that decodes the codes **on first
use** (per-expert LRU of dense bf16 weights, pure-MLX decode -> lazy-graph
safe) and then runs plain mlx matmuls: the decode platform is affine-like and
the values are the exact escha trellis decode (no second quantization).

The routed-expert dense weight is baked as: W = (H Wtilde H * rin * rout).T,
matching the reference runtime's `xh = had(x*rin); y = had(y_pre)*rout`.

Activation: ``maybe_apply_pre_load_patches`` calls ``apply_escha_trellis_patch``
when config.json declares ``quantization_config.quant_method == "eschamoe"``.
The class swap is inert for non-escha checkpoints (delegates to stock).
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.activations import silu

logger = logging.getLogger(__name__)

_PATCHED = False
_ESCHA_MODE = False
_IN_MTP = False
_ORIGINAL_SPARSE_MOE_BLOCK = None



_H128 = None


def _h128() -> mx.array:
    global _H128
    if _H128 is None:
        h = mx.array([[1.0]])
        while h.shape[0] < 128:
            h = mx.concatenate(
                [mx.concatenate([h, h], 1), mx.concatenate([h, -h], 1)], 0
            )
        _H128 = h * (1.0 / (128.0 ** 0.5))
    return _H128


def _had128(x: mx.array) -> mx.array:
    """Blockwise-128 orthonormal Hadamard along the last axis."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    y = x.reshape(*lead, n // 128, 128)
    y = mx.matmul(y.reshape(-1, n // 128, 128), _h128())
    return mx.moveaxis(y.reshape(*lead, n), -1, 1) if False else y.reshape(*lead, n)


class _DenseExpertCache(nn.Module):
    """One routed projection: packed trellis codes + channel scales, decoded
    per-row by the small-batch Metal kernel (``eschamoe_gather_qmv``)."""

    def __init__(self, E: int, inp: int, outp: int, K: int, cap: int = 16):
        super().__init__()
        self.escha_code = mx.zeros((E, inp // 16, outp // 16, 16 * K), mx.int16)
        self.escha_rin = mx.zeros((E, inp), mx.float16)
        self.escha_rout = mx.zeros((E, outp), mx.float16)
        self._K = K

    def __call__(self, x, eids):
        from omlx.custom_kernels.escha import fast as _fast
        rin = self.escha_rin[eids].astype(mx.float32)
        rout = self.escha_rout[eids].astype(mx.float32)
        xh = _had128(x * rin)
        mx.eval(xh)                          # kernel inputs must be materialized
        p = _fast.eschamoe_gather_qmv(xh, self.escha_code, eids, self._K)
        return _had128(p) * rout


class TrellisSwitchMLP(nn.Module):
    """Routed experts: one fused kernel per layer (gate_up decode -> in-group
    had -> SwiGLU -> down decode -> GEMM). Host keeps ONE input materialization
    per layer (xh1) and one trailing had(* rout2)."""

    def __init__(self, num_experts: int, hidden: int, intermediate: int):
        super().__init__()
        self.gate_up_proj = _DenseExpertCache(num_experts, hidden, 2 * intermediate, K=2)
        self.down_proj = _DenseExpertCache(num_experts, intermediate, hidden, K=3)

    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        from omlx.custom_kernels.escha import fast as _fast
        lead = x.shape[:-1]
        k = indices.shape[-1]
        N = 1
        for d in lead:
            N *= int(d)
        x2 = x.reshape(N, -1)
        inds2 = indices.reshape(N, k)
        xg = mx.repeat(x2, k, axis=0)
        eids = inds2.reshape(-1).astype(mx.uint32)

        rin1 = self.gate_up_proj.escha_rin[eids].astype(mx.float32)
        rout1 = self.gate_up_proj.escha_rout[eids].astype(mx.float32)
        rin2 = self.down_proj.escha_rin[eids].astype(mx.float32)
        rout2 = self.down_proj.escha_rout[eids].astype(mx.float32)

        xh1 = _had128(xg * rin1)
        # No per-layer materialization: the fused kernel has no intra-layer
        # kernel dependency, so all 40 layers can share one lazy graph (one
        # sync per step) without corruption.  (ESCHA_FORCE_EVAL=1 re-enables
        # the conservative per-layer sync for debugging.)
        if __import__("os").environ.get("ESCHA_FORCE_EVAL"):
            mx.eval(xh1)
        ypre = _fast.eschamoe_fused_layer(
            xh1, self.gate_up_proj.escha_code, self.down_proj.escha_code,
            eids, rout1, rin2, rout2,
        )
        y = _had128(ypre) * rout2
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
    global _PATCHED, _ORIGINAL_SPARSE_MOE_BLOCK
    if _PATCHED:
        return True
    try:
        import mlx_lm.models.qwen3_5 as _q35

        _ORIGINAL_SPARSE_MOE_BLOCK = _q35.SparseMoeBlock

        def _sparse_moe_factory(args):
            if _ESCHA_MODE and not _IN_MTP:
                return _EschaSparseMoeBlock(args)
            return _ORIGINAL_SPARSE_MOE_BLOCK(args)

        _q35.SparseMoeBlock = _sparse_moe_factory

        # The MTP head's decoder layer also builds a SparseMoeBlock, but with
        # regular dense affine weights (not trellis codes): exclude it.
        try:
            from ..patches.mlx_lm_mtp import apply_mlx_lm_mtp_patch
            apply_mlx_lm_mtp_patch()      # idempotent; registers MTPDecoderLayer
        except Exception:
            pass
        _mtp_dl = getattr(_q35, "MTPDecoderLayer", None)
        if _mtp_dl is not None and not getattr(_mtp_dl, "_omlx_escha_guarded", False):
            _mtp_dl_orig_init = _mtp_dl.__init__

            def _mtp_guarded_init(self, args, _orig=_mtp_dl_orig_init):
                global _IN_MTP
                _IN_MTP = True
                try:
                    _orig(self, args)
                finally:
                    _IN_MTP = False

            _mtp_dl.__init__ = _mtp_guarded_init
            _mtp_dl._omlx_escha_guarded = True

        for _mod in ("qwen3_5", "qwen3_5_moe"):
            try:
                mod = __import__(f"mlx_lm.models.{_mod}", fromlist=["ModelArgs"])
                orig = mod.ModelArgs.from_dict.__func__

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
