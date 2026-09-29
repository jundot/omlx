import logging
from functools import partial
from typing import Any, Optional

from omlx.utils.layer_pipeline import LayerPipeline
import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..deepseek_v4.hyper_connection import HyperConnection as _HyperConnection
from ..deepseek_v4.hyper_connection import hc_expand as _hc_expand
from ..fast_ops import exact_hc_norm
from ..linear import DECODE_BLOCK_SIZE
from mlx_lm.models.mla import MultiLinear
from omlx.custom_kernels.nax import is_nax_available
from omlx.patches import glm53_kda_prework
from omlx.patches.mlx_vlm_glm5_next_compat import decode_kernels as _decode_kernels
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU, _sort_threshold
from omlx.patches.glm_moe_dsa.sparse_mla_nax import sparse_mla_attention_nax
from omlx.patches.glm_moe_dsa.deepseek_v32 import (
    Model as DSV32Model,
    group_expert_select,
)
from omlx.patches.glm_moe_dsa.sparse_mla import (
    exact_block_token_attention,
    q8_vup_flat,
    sparse_mla_attention,
)
from omlx.patches.qwen35_verify_qmm import _is_armed as _verify_qmm_armed
from omlx.patches.qwen35_verify_qmm import is_row_exact_armed as _row_exact_armed
from omlx.patches.glm_moe_dsa.indexer_nax import (
    indexer_scores_nax,
    max_rows_per_call,
    nax_indexer_available,
)
from .config import ModelConfig, TextConfig
from . import hc_prefill
from .gated_delta import gated_delta_update
from .linear import fused_quantized_matmul, linear_forward


logger = logging.getLogger(__name__)
_NATIVE_INDEXER_WARNED = False


# Causal row blocks of the dense-prefix attention (1 = one call).
_DENSE_ROW_BLOCKS = 8


def _cache_parts(cache):
    """(kv, pool) halves of a sparse-layer cache; either half may be missing."""
    try:
        return cache[0], cache[1]
    except (TypeError, IndexError, KeyError):
        return None, None


# Single-sequence decode (L == 1) and short verify blocks (L <= 8, the
# DECODE_BLOCK_SIZE of the shared HC helpers) run fused kernels that
# reproduce the reference op graph bit for bit; see decode_kernels.py.
# They are validated on M5 (NAX) GPUs and used there.
_DECODE_FUSION = is_nax_available()
_DECODE_BLOCK = 8

# One-token decode forwards start evaluating every this many layers.
# A step is ~800 dependent dispatches whose Python graph build takes ~2.7 ms;
# mlx keeps at most ~10 command buffers in flight and, with its default
# per-buffer size budget (every expert or projection weight input counts in
# full), a step commits ~170 buffers, so the GPU drains the previous step's
# last few buffers long before the next graph is built and encoded (~1.3 ms
# idle per token). Encoding the first layers while the later ones are being
# built keeps it fed. The values computed are unchanged.
_DECODE_EVAL_EVERY = 8


def _decode_hc_pre(connection, norm, x: mx.array):
    """HC collapse plus the branch's input RMSNorm for B == 1, L <= 8.

    Two dispatches (``hc_mix`` and ``exact_hc_norm``) replace the
    cast/RMS/mix-GEMV/sinkhorn-collapse/RMSNorm chain; every row equals the
    reference ``connection(x)`` followed by ``norm`` (the one-token GEMV
    reduction, which the reference also uses per token for L <= 8).
    Returns None when the shape or module state is not covered.
    """
    if (
        not _DECODE_FUSION
        or connection.training
        or connection.hc_mult != 4
        or x.ndim != 4
        or x.shape[0] != 1
        or not 1 <= x.shape[1] <= _DECODE_BLOCK
    ):
        return None
    mixes = _decode_kernels.hc_mix(x, connection.fn, connection.norm_eps)
    if mixes is None:
        return None
    return exact_hc_norm(connection, norm, x, mixes)


def _switch_projections(sw):
    """A SwitchGLU's expert projections in the order its forward passes them
    to _sort_threshold (fused gate_up_proj or separate gate/up, then down)."""
    if "gate_up_proj" in sw:
        return (sw.gate_up_proj, sw.down_proj)
    return (sw.up_proj, sw.gate_proj, sw.down_proj)


def verify_qmm_routed(rows: int) -> bool:
    """Whether a ``QuantizedLinear`` call of ``rows`` rows takes the armed
    MTP verify routes (``qwen35_verify_qmm``) instead of the reference qmm.
    Fused kernels replay the reference qmm, so they decline then."""
    return (
        rows > 1
        and getattr(nn.QuantizedLinear, "_omlx_verify_qmm_patched", False)
        and (_verify_qmm_armed() or _row_exact_armed())
    )


def _multi_linear(x, layers):
    """``[linear_forward(layer, x) for layer in layers]`` for a decode/verify
    block ``x`` [1, L, D]: layers sharing (bits, group size) run as one exact
    ``decode_kernels.multi_qmv`` dispatch, the others as the reference call.
    Returns None when no two layers could share a dispatch."""
    if verify_qmm_routed(x.shape[1]):
        return None
    groups = {}
    for i, layer in enumerate(layers):
        key = (getattr(layer, "bits", None), getattr(layer, "group_size", None))
        groups.setdefault(key, []).append(i)
    if all(len(g) == 1 for g in groups.values()):
        return None
    L, D = x.shape[1], x.shape[2]
    outs = [None] * len(layers)
    for group in groups.values():
        res = None
        if len(group) > 1:
            res = _decode_kernels.multi_qmv(x.reshape(L, D), [layers[i] for i in group])
            if res is not None:
                res = [r.reshape(1, L, -1) for r in res]
        if res is None:
            res = [linear_forward(layers[i], x) for i in group]
        for i, r in zip(group, res):
            outs[i] = r
    return outs


def _decode_hc_expand(x: mx.array, residual: mx.array, post, comb) -> mx.array:
    """``hc_expand`` with the one-token case in a single exact dispatch."""
    if (
        _DECODE_FUSION
        and x.ndim == 3
        and x.shape[:2] == (1, 1)
    ):
        out = _decode_kernels.hc_expand_one(x, residual, post, comb)
        if out is not None:
            return out
    return hc_expand(x, residual, post, comb)


class _HCDeferred:
    """A one-token half-layer output whose HC expand is still pending:
    ``h = hc_expand(y, residual, post, comb)``.

    The next half-layer's ``_decode_hc_pre_deferred`` recomputes h exactly
    while it loads it (and stores it, the next residual) instead of a
    separate expand dispatch; ``materialize`` runs the expand itself. ``mm``
    is hc_expand_one's NAX comb product of ``residual`` ([HC * D] fp32).
    """

    __slots__ = ("y", "residual", "post", "comb", "mm")

    def __init__(self, y, residual, post, comb, mm):
        self.y = y
        self.residual = residual
        self.post = post
        self.comb = comb
        self.mm = mm

    def arrays(self) -> list:
        return [self.y, self.residual, self.post, self.comb, self.mm]

    def materialize(self) -> mx.array:
        return _decode_hc_expand(self.y, self.residual, self.post, self.comb)


def _hc_defer_ok(connection, norm, dtype, width: int) -> bool:
    return (
        _DECODE_FUSION
        and not connection.training
        and _decode_kernels.hc_defer_supported(connection, norm, dtype, width)
    )


def _decode_hc_pre_deferred(connection, norm, x):
    """One-token HC pre of ``x`` ([1, 1, HC, D] or an ``_HCDeferred``).

    Returns ``(branch input, h, post, comb, mm)``: the normalized branch
    input and h (this half-layer's residual) from one ``hc_pre_fused``
    dispatch, and post/comb/mm from ``hc_post_mm``, which only the next
    half-layer reads. The branch input depends on that kernel without reading
    it, so it is scheduled first and runs beside the branch's first kernel.
    The values are those of ``_decode_hc_pre`` and ``_decode_hc_expand``.
    """
    dk = _decode_kernels
    if isinstance(x, _HCDeferred):
        xn, mixes, h = dk.hc_pre_fused(connection, norm, deferred=(x.y, x.mm, x.post))
    else:
        xn, mixes, _ = dk.hc_pre_fused(connection, norm, x=x)
        h = x
    post, comb, mm = dk.hc_post_mm(connection, mixes, h)
    xn = mx.depends(xn, [post])
    return xn, h, post, comb, mm


def _array_slots(tree, slots: list) -> list:
    """Append ``(container, key)`` for every array held in ``tree``: the items
    of a module (a dict) and of its nested dicts and lists."""
    items = tree.items() if isinstance(tree, dict) else enumerate(tree)
    for key, value in items:
        if isinstance(value, mx.array):
            slots.append((tree, key))
        elif isinstance(value, (dict, list)):
            _array_slots(value, slots)
    return slots


def compile_ffn_block(layer, method):
    """``mx.compile(method)`` for a decoder layer's FFN half, with the arrays
    of the modules it reads traced as inputs rather than as constants.

    MLX 0.32.2 leaks every multi-output primitive that is an intermediate of a
    compiled trace: rewiring the trace releases the old outputs by assignment,
    which skips the sibling-cycle break their destructor runs (fixed upstream
    in ml-explore/mlx#4453), so each such group stays allocated together with
    everything it references. The one-token FFN is built from such kernels
    (router logits, the route-selecting gate/up, the HC pre/post), so with the
    weights as trace constants every compiled MoE layer kept its routed and
    shared gate/up weights alive after the model was unloaded: ~2.5 GB per
    layer, 40-80 GB of GLM-5.3's 169 GB. Traced as inputs, the weights are
    shape-only placeholders in the trace and a leaked group holds no weight
    memory. Every call passes the same arrays, so the compiled graph and its
    values are unchanged.
    """
    slots = []
    for module in (layer.ffn_hc, layer.post_attention_layernorm, layer.mlp):
        _array_slots(module, slots)
    arrays = [container[key] for container, key in slots]

    def traced(arrays, *args):
        saved = [container[key] for container, key in slots]
        for (container, key), value in zip(slots, arrays):
            container[key] = value
        try:
            return method(*args)
        finally:
            for (container, key), value in zip(slots, saved):
                container[key] = value

    compiled = mx.compile(traced)
    return lambda *args: compiled(arrays, *args)


def _mla_head_proj(layer, x: mx.array) -> mx.array:
    """``layer(x)`` for the MLA per-head projections (embed_q, unembed_out);
    one token through ``decode_kernels.mla_head_qmv`` (same values)."""
    if _DECODE_FUSION and x.ndim == 4 and x.shape[2] == 1:
        out = _decode_kernels.mla_head_qmv(x, layer)
        if out is not None:
            return out
    return layer(x)


def glm5_next_cast_predicate(key: str) -> bool:
    """Keep numerically sensitive GLM-5.3 parameters in FP32."""
    return not (
        "e_score_correction_bias" in key
        or ".attn_hc." in key
        or ".ffn_hc." in key
        or key.endswith("A_log")
        or key.endswith("dt_bias")
        or key.endswith("mlp.gate.weight")
    )


class HyperConnection(_HyperConnection):
    """mlx-vlm's hyper-connection with fused, batch-invariant prefill kernels.

    Blocks longer than ``DECODE_BLOCK_SIZE`` take ``hc_prefill.hc_pre``;
    decode and short verify blocks keep the canonical path.
    """

    def __call__(self, x: mx.array):
        if x.ndim == 4 and x.shape[1] > DECODE_BLOCK_SIZE:
            fused = hc_prefill.hc_pre(self, x)
            if fused is not None:
                return fused
        return super().__call__(x)


def hc_expand(x, residual, post, comb, **kwargs):
    """``hc_expand`` with a single-pass kernel for prefill-length blocks."""
    if x.ndim == 3 and x.shape[1] > DECODE_BLOCK_SIZE and not kwargs:
        fused = hc_prefill.hc_expand(x, residual, post, comb)
        if fused is not None:
            return fused
    return _hc_expand(x, residual, post, comb, **kwargs)


class Glm5NextRMSNormGated(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(self, hidden_states: mx.array, gate: mx.array) -> mx.array:
        dt = hidden_states.dtype
        x = hidden_states.astype(mx.float32)
        var = (x * x).mean(-1, keepdims=True)
        x = x * mx.rsqrt(var + self.eps)
        x = self.weight.astype(mx.float32) * x
        x = x * mx.sigmoid(gate.astype(mx.float32))
        return x.astype(dt)


class Glm5NextForgetGate(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = mx.zeros(self.qkv_dim)
        self.A_log = mx.zeros(self.num_heads)
        self.safe_gate_lower_bound = config.linear_lower_bound

    def __call__(self, hidden_states: mx.array) -> mx.array:
        B, S, _ = hidden_states.shape
        fg = self.f_b_proj(self.f_a_proj(hidden_states))
        g = (fg.astype(mx.float32) + self.dt_bias.astype(mx.float32)).reshape(
            B, S, self.num_heads, self.head_dim
        )
        decay = mx.exp(self.A_log.astype(mx.float32)).reshape(1, 1, self.num_heads, 1)
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * mx.sigmoid(decay * g)
        g_softplus = mx.where(g > 20.0, g, mx.log(1.0 + mx.exp(g)))
        return -decay * g_softplus


def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + eps)


def recurrent_kimi_delta(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    g: mx.array,
    beta: mx.array,
    state: Optional[mx.array] = None,
):
    # Reference O(S) recurrence for Kimi Delta Attention, kept as the readable
    # spec and the equivalence oracle for tests. The forward path runs this on
    # the shared fused gated_delta kernel (see Glm5NextLinearAttention).
    dt = query.dtype
    query = _l2norm(query.astype(mx.float32))
    key = _l2norm(key.astype(mx.float32))
    value = value.astype(mx.float32)
    g = g.astype(mx.float32)
    beta = beta.astype(mx.float32)
    B, S, H, Dk = key.shape
    Dv = value.shape[-1]
    query = query * (Dk**-0.5)
    if state is None:
        state = mx.zeros((B, H, Dk, Dv), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)
    outs = []
    for i in range(S):
        q_i = query[:, i]
        k_i = key[:, i]
        v_i = value[:, i]
        g_i = mx.exp(g[:, i])[..., None]
        b_i = beta[:, i][..., None]
        state = state * g_i
        kv_mem = (state * k_i[..., None]).sum(axis=-2)
        delta = (v_i - kv_mem) * b_i
        state = state + k_i[..., None] * delta[..., None, :]
        out_i = (state * q_i[..., None]).sum(axis=-2)
        outs.append(out_i)
    out = mx.stack(outs, axis=1).astype(dt)
    return out, state


class KdaStepCapture:
    """A fused KDA verify block, kept so a speculative rollback is exact.

    Holds the block's ``kda_decode_step`` inputs (the fused input projection
    ``proj`` and, when the gate projections ran outside the kernel, their
    rows ``a_pre``/``gate_pre``) and the layer's entry conv/recurrent states.
    Each kernel row depends only on the rows before it (causal short conv,
    per-row l2norm and gate projections, sequential delta rule), so
    ``replay(n)`` over the first ``n`` rows returns bit for bit the states
    the block held after row ``n``.
    """

    __slots__ = ("layer", "proj", "conv_state", "state", "a_pre", "gate_pre")

    def __init__(self, layer, proj, conv_state, state, a_pre=None, gate_pre=None):
        self.layer = layer
        self.proj = proj
        self.conv_state = conv_state
        self.state = state
        self.a_pre = a_pre
        self.gate_pre = gate_pre

    @property
    def width(self) -> int:
        return int(self.proj.shape[1])

    def replay(self, n: int):
        """``(conv_state, recurrent_state)`` after the first ``n`` rows."""
        n = int(n)
        if not 1 <= n <= self.width:
            raise ValueError(f"KDA replay of {n} rows from a {self.width}-row block")

        def rows(a):
            return None if a is None else a[:, :n]

        result = self.layer._kda_kernel_step(
            rows(self.proj), self.conv_state, self.state,
            rows(self.a_pre), rows(self.gate_pre),
        )
        if result is None:
            raise RuntimeError("fused KDA kernel declined a captured verify block")
        return result[1], result[2]


class Glm5NextLinearAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.linear_num_heads
        self.head_dim = config.linear_head_dim
        self.qkv_dim = self.num_heads * self.head_dim
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.q_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.qkv_dim, bias=False)

        self.conv_dim = self.qkv_dim * 3
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=0,
        )

        self.forget_gate = Glm5NextForgetGate(config)
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.o_norm = Glm5NextRMSNormGated(self.head_dim, eps=config.rms_norm_eps)
        self.o_proj = nn.Linear(self.qkv_dim, self.hidden_size, bias=False)
        self.fuse_in = True
        self._fused_ready = False
        # Decode-only grouped input projection when q/k/v/f_a/g_a/b do not
        # share one quantization (None: not built yet, False: not covered).
        self._decode_groups = None

    def _fused_in_proj(self, inputs, split=True):
        # q,k,v,f_a,g_a,b all take `inputs`; fuse into one matmul via a lossless
        # output-axis concat of the (quantized) weights, built once and cached.
        # split=False returns (unsplit output, split points), or None when the
        # projections cannot be fused.
        if not self._fused_ready:
            mods = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.forget_gate.f_a_proj,
                self.g_a_proj,
                self.b_proj,
            ]
            quantized = [hasattr(m, "scales") for m in mods]
            if any(quantized) and not all(quantized):
                if not split:
                    return None
                return tuple(linear_forward(m, inputs) for m in mods)
            if all(quantized):
                specs = {
                    (m.group_size, m.bits, getattr(m, "mode", "affine")) for m in mods
                }
                if len(specs) != 1:
                    if not split:
                        return None
                    return tuple(linear_forward(m, inputs) for m in mods)
            pts, acc = [], 0
            for m in mods[:-1]:
                acc += m.weight.shape[0]
                pts.append(acc)
            self._split_pts = pts
            self._fq = hasattr(mods[0], "scales")
            self._fw = mx.concatenate([m.weight for m in mods], axis=0)
            if self._fq:
                self._fs = mx.concatenate([m.scales for m in mods], axis=0)
                self._fb = mx.concatenate([m.biases for m in mods], axis=0)
                self._gs, self._bits = mods[0].group_size, mods[0].bits
            self._fused_ready = True
        if self._fq:
            out = fused_quantized_matmul(
                inputs,
                self._fw,
                self._fs,
                self._fb,
                bits=self._bits,
                group_size=self._gs,
            )
        else:
            out = inputs @ self._fw.T
        if not split:
            return out, self._split_pts
        return mx.split(out, self._split_pts, axis=-1)

    def _build_decode_groups(self):
        """One fused weight per quantization for projections that mix bit
        widths (e.g. a 5-bit v_proj among 8-bit q/k/gates): row blocks in
        q|k|v|f_a|g_a|b order within each group. False when not covered."""
        mods = [
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.forget_gate.f_a_proj,
            self.g_a_proj,
            self.b_proj,
        ]
        for m in mods:
            if (
                not isinstance(m, nn.QuantizedLinear)
                or getattr(m, "mode", "affine") != "affine"
                or "bias" in m
                or getattr(m, "biases", None) is None
            ):
                return False
        order = {}
        for i, m in enumerate(mods):
            order.setdefault((int(m.bits), int(m.group_size)), []).append(i)
        if len(order) < 2:
            return False
        groups = []
        for (bits, gs), members in order.items():
            parts = [(mods[i].weight, mods[i].scales, mods[i].biases) for i in members]
            if len(parts) == 1:
                w, sc, bi = parts[0]
            else:
                w, sc, bi = (mx.concatenate(list(t), axis=0) for t in zip(*parts))
            groups.append((bits, gs, members, w, sc, bi))
        # Runs of consecutive projections that sit next to each other in one
        # group's output: (group index, first row, last row) in q..b order.
        rows = [m.weight.shape[0] for m in mods]
        where = {}
        for g, (_, _, members, *_rest) in enumerate(groups):
            start = 0
            for i in members:
                where[i] = (g, start, start + rows[i])
                start += rows[i]
        runs = []
        for i in range(len(mods)):
            g, a, b = where[i]
            if runs and runs[-1][0] == g and runs[-1][2] == a:
                runs[-1] = (g, runs[-1][1], b)
            else:
                runs.append((g, a, b))
        pts, acc = [], 0
        for n in rows[:-1]:
            acc += n
            pts.append(acc)
        self._split_pts = pts
        return (groups, runs)

    def _grouped_in_proj(self, inputs):
        """``[linear_forward(m, inputs) for m in q..b]`` concatenated, with one
        matmul per quantization (each row is its own layer's matmul row)."""
        if not self._decode_groups:
            return None
        groups, runs = self._decode_groups
        outs = [
            fused_quantized_matmul(inputs, w, sc, bi, bits=bits, group_size=gs)
            for bits, gs, _, w, sc, bi in groups
        ]
        return mx.concatenate([outs[g][..., a:b] for g, a, b in runs], axis=-1)

    def _decode_step(self, inputs, mask, cache, capture=None):
        """Fused layer body for one sequence and S <= 8 tokens (bit-identical).

        One kernel (``decode_kernels.kda_decode_step``) replaces the conv,
        SiLU, l2norm, gate projections, delta rule and RMSNormGated ops
        between the fused input projection and o_proj. Returns None when the
        layer or inputs are not covered.

        ``capture`` (a list, for speculative verify blocks) receives a
        ``KdaStepCapture`` of the block's kernel inputs and entry states, from
        which ``KdaStepCapture.replay`` rebuilds the caches after any prefix.
        """
        B, S, _ = inputs.shape
        fg = self.forget_gate
        if (
            not _DECODE_FUSION
            or B != 1
            or not 1 <= S <= _DECODE_BLOCK
            or mask is not None
            or cache is None
            or getattr(cache, "lengths", None) is not None
            or not self.fuse_in
            or fg.safe_gate_lower_bound is None
            or self.conv_kernel_size != 4
        ):
            return None
        if not self._fused_ready and self._decode_groups is None:
            self._fused_in_proj(inputs)
            if not self._fused_ready:
                self._decode_groups = self._build_decode_groups()
        if self._fused_ready:
            if not self._fq:
                return None
            proj = fused_quantized_matmul(
                inputs, self._fw, self._fs, self._fb, bits=self._bits, group_size=self._gs
            )
        else:
            proj = self._grouped_in_proj(inputs)
            if proj is None:
                return None
        _, _, v_end, fa_end, ga_end = self._split_pts
        a_pre = gate_pre = None
        f_b, g_b = fg.f_b_proj, self.g_b_proj
        bits = {getattr(m, "bits", None) for m in (f_b, g_b)}
        # The kernel replays the gate projections for 4/8-bit rows (qmv_quad)
        # and, for one token, 5-bit rows (qmv); otherwise they run here.
        in_kernel = {4, 8} | ({5} if S == 1 else set())
        if not (isinstance(f_b, nn.QuantizedLinear) and bits <= in_kernel and len(bits) == 1):
            a_pre = linear_forward(f_b, proj[..., v_end:fa_end])
            gate_pre = linear_forward(g_b, proj[..., fa_end:ga_end])
        conv_in, state_in = cache[0], cache[1]
        result = self._kda_kernel_step(proj, conv_in, state_in, a_pre, gate_pre)
        if result is None:
            return None
        y, conv_state, state = result
        if capture is not None:
            capture.append(KdaStepCapture(self, proj, conv_in, state_in, a_pre, gate_pre))
        cache[0] = conv_state
        cache[1] = state
        cache.advance(S)
        return linear_forward(self.o_proj, y)

    def _kda_kernel_step(self, proj, conv_state, state, a_pre, gate_pre):
        """``decode_kernels.kda_decode_step`` for this layer: (y, conv, state)."""
        fg = self.forget_gate
        _, _, v_end, fa_end, ga_end = self._split_pts
        return _decode_kernels.kda_decode_step(
            proj,
            conv_state,
            self.conv1d.weight,
            fg.A_log,
            fg.dt_bias,
            state,
            self.o_norm.weight,
            heads=self.num_heads,
            head_dim=self.head_dim,
            off_fa=v_end,
            off_ga=fa_end,
            off_b=ga_end,
            q_scale=self.head_dim**-0.5,
            l2_eps=1e-6,
            norm_eps=self.o_norm.eps,
            lower_bound=fg.safe_gate_lower_bound,
            f_b=fg.f_b_proj,
            g_b=self.g_b_proj,
            a_pre=a_pre,
            gate_pre=gate_pre,
        )

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        fused = self._decode_step(inputs, mask, cache)
        if fused is not None:
            return fused
        B, S, _ = inputs.shape
        if glm53_kda_prework.glm53_kda_prefill_eligible(self, inputs, mask, cache):
            return glm53_kda_prework.glm53_kda_prefill(self, inputs, cache)
        has_right_padding = cache is not None and cache.lengths is not None
        if has_right_padding:
            mask = mx.arange(S)[None] < cache.lengths[:, None]
        if self.fuse_in:
            q_o, k_o, v_o, fa_o, ga_o, b_o = self._fused_in_proj(inputs)
            mixed = mx.concatenate([q_o, k_o, v_o], axis=-1)
        else:
            mixed = mx.concatenate(
                [self.q_proj(inputs), self.k_proj(inputs), self.v_proj(inputs)], axis=-1
            )
            fa_o = self.forget_gate.f_a_proj(inputs)
            ga_o = self.g_a_proj(inputs)
            b_o = self.b_proj(inputs)
        if mask is not None and mask.dtype == mx.bool_:
            mixed = mx.where(mask[..., None], mixed, 0)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=inputs.dtype
            )
        conv_input = mx.concatenate([conv_state, mixed], axis=1)
        if cache is not None:
            state_size = self.conv_kernel_size - 1
            if has_right_padding:
                valid_lengths = mx.sum(mask, axis=-1).astype(mx.int32)
                state_indices = valid_lengths[:, None] + mx.arange(state_size)[None]
                state_indices = mx.broadcast_to(
                    state_indices[..., None],
                    (B, state_size, self.conv_dim),
                )
                cache[0] = mx.contiguous(
                    mx.take_along_axis(conv_input, state_indices, axis=1)
                )
            else:
                cache[0] = mx.contiguous(conv_input[:, -state_size:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = mx.split(conv_out, [self.qkv_dim, 2 * self.qkv_dim], axis=-1)
        q = q.reshape(B, S, self.num_heads, self.head_dim)
        k = k.reshape(B, S, self.num_heads, self.head_dim)
        v = v.reshape(B, S, self.num_heads, self.head_dim)

        fg = self.forget_gate
        a = linear_forward(fg.f_b_proj, fa_o).reshape(
            B, S, self.num_heads, self.head_dim
        )
        in_dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * (self.head_dim**-0.5)).astype(in_dtype)
        k = _l2norm(k.astype(mx.float32)).astype(in_dtype)

        state = cache[1] if cache is not None else None
        out, state = gated_delta_update(
            q,
            k,
            v,
            a,
            b_o,
            fg.A_log.reshape(self.num_heads, 1),
            fg.dt_bias.reshape(self.num_heads, self.head_dim),
            state=state,
            mask=mask if mask is not None and mask.dtype == mx.bool_ else None,
            lower_bound=fg.safe_gate_lower_bound,
        )
        if cache is not None:
            cache[1] = state
            cache.advance(S)

        gate = linear_forward(self.g_b_proj, ga_o).reshape(
            B, S, self.num_heads, self.head_dim
        )
        out = self.o_norm(out, gate).reshape(B, S, -1)
        return linear_forward(self.o_proj, out)


class Glm5NextIndexer(nn.Module):
    def __init__(self, args: TextConfig):
        super().__init__()
        self.dim = args.hidden_size
        self.n_heads = args.index_n_heads
        self.head_dim = args.index_head_dim
        self.index_topk = args.index_topk
        self.index_kpool = args.index_kpool
        self.index_kpool_always_select_tail = args.index_kpool_always_select_tail
        self.q_lora_rank = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim**-0.5
        self.weight_scale = self.n_heads**-0.5 * self.softmax_scale
        self.index_kpool_compress_ape = mx.zeros((self.index_kpool, self.head_dim))
        self.index_kpool_compress_gate = mx.zeros((self.head_dim, self.dim))

    def _compress_windows(self, keys, gate_scores):
        B, S, hd = keys.shape
        kp = self.index_kpool
        if S == 0:
            return mx.zeros((B, 0, hd), dtype=keys.dtype)
        usable = (S // kp) * kp
        keys = keys[:, :usable].reshape(B, -1, kp, hd)
        gate_scores = gate_scores[:, :usable].reshape(B, -1, kp, hd)
        logits = gate_scores + self.index_kpool_compress_ape[None, None]
        probs = mx.softmax(logits, axis=2)
        return mx.sum(probs * keys, axis=2)

    @staticmethod
    def _processed(cache):
        processed = getattr(cache, "_processed", None)
        if processed is not None:
            return list(processed)
        return int(cache.size() * cache.ratio + cache.remainder)

    @staticmethod
    def _pool_lengths(cache):
        lengths = getattr(cache, "_pool_lengths", None)
        if lengths is not None:
            return list(lengths)
        return int(cache.size())

    def _native_scores(self, q, pool_keys, weights):
        global _NATIVE_INDEXER_WARNED
        if (
            q.shape[2] != self.n_heads
            or self.n_heads != 32
            or self.head_dim != 128
            or q.dtype not in (mx.float16, mx.bfloat16)
            or pool_keys.dtype != q.dtype
        ):
            return None
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast

            if not fast.has_symbol("dsa_indexer_scores"):
                return None
            qt = q.transpose(0, 2, 1, 3)
            keys = pool_keys[:, None]
            q_pad = (-qt.shape[2]) % 64
            k_pad = (-keys.shape[2]) % 64
            if q_pad:
                qt = mx.pad(qt, [(0, 0), (0, 0), (0, q_pad), (0, 0)])
                weights = mx.pad(weights, [(0, 0), (0, q_pad), (0, 0)])
            if k_pad:
                keys = mx.pad(keys, [(0, 0), (0, 0), (0, k_pad), (0, 0)])
            scores = fast.dsa_indexer_scores(
                qt,
                keys,
                weights,
                causal=False,
            )
            return scores[:, 0, : q.shape[1], : pool_keys.shape[1]]
        except (AttributeError, RuntimeError, TypeError, ValueError):
            if not _NATIVE_INDEXER_WARNED:
                logger.warning(
                    "GLM-5.3 native DSA indexer failed; using the MLX fallback",
                    exc_info=True,
                )
                _NATIVE_INDEXER_WARNED = True
            return None

    @staticmethod
    def _native_topk(scores, topk):
        if topk != 512:
            return None
        try:
            from omlx.custom_kernels.glm_moe_dsa import fast

            if fast.has_symbol("dsa_topk_indices"):
                return fast.dsa_topk_indices(scores[:, None], topk)[:, 0]
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
        return None

    def _fast_short_select(
        self, q, x, pool_keys, before, after, pool_lengths, kv_cache, S,
        select_k, tail_on, output_width, w_raw=None,
    ):
        """Decode/verify (S <= 8, one sequence) selection on fused kernels.

        Same scores, top-k kernel and index expansion as the general path;
        see ``decode_kernels.dsa_decode_scores``.  Returns None to fall back.
        """
        if isinstance(before, list):
            if len(before) != 1 or after[0] - before[0] != S:
                return None
            before, pool_lengths = before[0], pool_lengths[0]
        if not isinstance(before, int) or not isinstance(pool_lengths, int):
            return None
        if select_k * self.index_kpool > self.index_topk:
            return None
        dk = _decode_kernels
        weights = w_raw if w_raw is not None else linear_forward(self.weights_proj, x)
        weights = (weights * self.weight_scale).astype(q.dtype)
        if (
            select_k == 512
            and self.n_heads == 32
            and self.head_dim == 128
            and q.dtype == mx.bfloat16
            and pool_keys.dtype == mx.bfloat16
            and nax_indexer_available()
        ):
            # Same scores as the reference path, which uses the NAX indexer here.
            scores = indexer_scores_nax(
                q[0], pool_keys[0], weights[0], before, pool_lengths, self.index_kpool
            )
            scores = None if scores is None else scores[None]
        else:
            scores = dk.dsa_decode_scores(
                q, pool_keys, weights, before, pool_lengths, self.index_kpool
            )
        if scores is None:
            return None
        # Same output as the native top-k (which covers select_k == 512 only).
        selected = dk.dsa_topk_rows(scores, select_k) if select_k == 512 else None
        if selected is None:
            selected = self._native_topk(scores, select_k)
        if selected is None:
            return None
        left_padding = getattr(kv_cache, "left_padding", None)
        if left_padding is None:
            left_padding = mx.zeros((1,), dtype=mx.int32)
        return dk.dsa_expand_topk(
            selected,
            before,
            pool_lengths,
            left_padding,
            self.index_kpool,
            self.index_kpool - 1 if tail_on else 0,
            output_width,
        )

    def __call__(
        self, x, qr, mask, cache=None, kv_cache=None, score_from=0, projected=None
    ):
        B, S, _ = x.shape
        # ``projected``: (wq_b(qr), wk(x), weights_proj(x)) already computed by
        # the attention's fused decode projections (entries may be None).
        q_raw, k_raw, w_raw = projected if projected is not None else (None, None, None)
        if q_raw is None:
            q_raw = linear_forward(self.wq_b, qr)
        if k_raw is None:
            k_raw = linear_forward(self.wk, x)
        q = q_raw.reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(k_raw).reshape(B, S, self.head_dim)
        gate_scores = x @ self.index_kpool_compress_gate.swapaxes(-1, -2)

        if cache is not None:
            before = self._processed(cache)
            cache_offset = (
                mx.array(before, dtype=mx.int32) if isinstance(before, list) else before
            )
            ready_k, ready_gate, _ = cache.accumulate_windows(
                k, gate_scores, cache_offset
            )
            compressed = self._compress_windows(ready_k, ready_gate)
            pool_keys = cache.update_and_fetch(compressed)
            after = self._processed(cache)
            pool_lengths = self._pool_lengths(cache)
            if isinstance(after, list):
                before_a = mx.array(before, dtype=mx.int32)
                valid_lengths = mx.array(
                    [a - b for a, b in zip(after, before)], dtype=mx.int32
                )
                valid_cur = mx.arange(S)[None] < valid_lengths[:, None]
                total_max = max(after)
            else:
                before_a = mx.full((B,), before, dtype=mx.int32)
                valid_cur = mx.ones((B, S), dtype=mx.bool_)
                total_max = after
        else:
            before = 0
            before_a = mx.zeros((B,), dtype=mx.int32)
            valid_cur = mx.ones((B, S), dtype=mx.bool_)
            usable = (S // self.index_kpool) * self.index_kpool
            pool_keys = self._compress_windows(k[:, :usable], gate_scores[:, :usable])
            pool_lengths = usable // self.index_kpool
            total_max = S

        # The pool has already advanced, even when sparse selection is unnecessary.
        # score_from lets the attention layer score only the rows past the dense
        # prefix while the pool still advances over every row.
        if score_from >= S:
            return None
        if getattr(self, "bypass_short", True) and total_max <= self.index_topk:
            return None

        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)
        if (
            S <= _DECODE_BLOCK
            and score_from == 0
            and B == 1
            and _DECODE_FUSION
            and getattr(self, "fast_decode", True)
        ):
            fast = self._fast_short_select(
                q, x, pool_keys, before, after, pool_lengths, kv_cache, S,
                select_k, tail_on, output_width, w_raw=w_raw,
            )
            if fast is not None:
                return fast
        pool_idx = mx.arange(P)
        pool_end = (pool_idx + 1) * self.index_kpool - 1
        if isinstance(pool_lengths, list):
            pool_lengths_a = mx.array(pool_lengths, dtype=mx.int32)
        else:
            pool_lengths_a = mx.full((B,), pool_lengths, dtype=mx.int32)
        left_padding = getattr(kv_cache, "left_padding", None)
        if left_padding is None:
            left_padding = mx.zeros((B,), dtype=mx.int32)

        # Tensor-unit scores with the causal pool mask folded in, for all
        # scored rows of the chunk at once (single sequence).
        before_s = before[0] if isinstance(before, list) and B == 1 else before
        pool_len_s = (
            pool_lengths[0]
            if isinstance(pool_lengths, list) and B == 1
            else pool_lengths
        )
        nax_scores = (
            B == 1
            and isinstance(before_s, int)
            and isinstance(pool_len_s, int)
            and select_k == 512
            and self.n_heads == 32
            and self.head_dim == 128
            and q.dtype == mx.bfloat16
            and pool_keys.dtype == mx.bfloat16
            and nax_indexer_available()
        )
        tail_rows = S - score_from
        if nax_scores:
            chunk = min(tail_rows, max_rows_per_call(P))
        else:
            chunk = 512 if tail_rows > 512 else tail_rows
        out = []
        for c0 in range(score_from, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            q_chunk = q[:, c0:c1]
            if w_raw is not None and c0 == 0 and c1 == S:
                weights = w_raw
            else:
                weights = linear_forward(self.weights_proj, x[:, c0:c1])
            weights = (weights * self.weight_scale).astype(q_chunk.dtype)
            query_pos = before_a[:, None] + mx.arange(c0, c1)[None]
            index_scores = None
            if nax_scores:
                index_scores = indexer_scores_nax(
                    q_chunk[0],
                    pool_keys[0],
                    weights[0],
                    before_s + c0,
                    pool_len_s,
                    self.index_kpool,
                )
            if index_scores is not None:
                index_scores = index_scores[None]
                valid_candidates = None
            else:
                index_scores = self._native_scores(q_chunk, pool_keys, weights)
                if index_scores is None:
                    head_scores = q_chunk @ pool_keys[:, None].swapaxes(-1, -2)
                    index_scores = mx.sum(
                        weights[..., None]
                        * mx.maximum(head_scores, mx.array(0, head_scores.dtype)),
                        axis=2,
                    )
                valid_candidates = (
                    pool_idx[None, None] < pool_lengths_a[:, None, None]
                ) & (pool_end[None, None] <= query_pos[..., None])
                index_scores = mx.where(valid_candidates, index_scores, -1e30)
            selected = self._native_topk(index_scores, select_k)
            if selected is None:
                selected = mx.argpartition(-index_scores, kth=select_k - 1, axis=-1)[
                    ..., :select_k
                ]
            if valid_candidates is None:
                # Same validity rule as valid_candidates, evaluated only at
                # the selected pools.
                selected_valid = (selected < pool_lengths_a[:, None, None]) & (
                    (selected + 1) * self.index_kpool - 1 <= query_pos[..., None]
                )
            else:
                selected_valid = mx.take_along_axis(
                    valid_candidates, selected, axis=-1
                )
            selected_indices = (
                selected[..., None] * self.index_kpool
                + mx.arange(self.index_kpool)[None, None, None]
                + left_padding[:, None, None, None]
            )
            topk = selected_indices.reshape(B, cs, -1)
            sv = mx.broadcast_to(
                selected_valid[..., None], (B, cs, select_k, self.index_kpool)
            ).reshape(B, cs, -1)
            topk = mx.where(sv, topk, -1)
            if tail_on:
                tail_width = self.index_kpool - 1
                tail_count = (query_pos + 1) % self.index_kpool
                tail_start = query_pos + 1 - tail_count
                tail_offsets = mx.arange(tail_width)
                tail = tail_start[..., None] + tail_offsets
                tail_valid = tail_offsets[None, None] < tail_count[..., None]
                tail = tail + left_padding[:, None, None]
                topk = mx.concatenate([topk, mx.where(tail_valid, tail, -1)], axis=-1)
            if topk.shape[-1] < output_width:
                pad = mx.full(
                    (B, cs, output_width - topk.shape[-1]), -1, dtype=topk.dtype
                )
                topk = mx.concatenate([topk, pad], axis=-1)
            topk = topk[..., :output_width]
            topk = mx.where(valid_cur[:, c0:c1][..., None], topk, -1)
            out.append(topk)
        topk = out[0] if len(out) == 1 else mx.concatenate(out, axis=1)
        return topk[:, None].astype(mx.int32)


class Glm5NextSparseAttention(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.use_nope = config.mla_use_nope or config.qk_rope_head_dim == 0
        # GLM-5-Next is NoPE by design (qk_rope_head_dim=0, mla_use_nope=True); the
        # config carries no rope parameters. Fail loudly rather than run wrong math
        # if a future config ever requests a RoPE MLA.
        if not self.use_nope:
            raise NotImplementedError(
                "glm5_next implements NoPE MLA only; qk_rope_head_dim>0 with "
                "mla_use_nope=False is not supported."
            )
        self.q_head_dim = config.qk_nope_head_dim
        self.scale = self.q_head_dim**-0.5

        self.q_a_proj = nn.Linear(
            self.hidden_size, self.q_lora_rank, bias=config.attention_bias
        )
        self.q_a_layernorm = nn.RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank, self.num_heads * self.q_head_dim, bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            self.hidden_size, self.kv_lora_rank, bias=config.attention_bias
        )
        self.kv_a_layernorm = nn.RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(
            self.qk_nope_head_dim, self.kv_lora_rank, self.num_heads
        )
        self.unembed_out = MultiLinear(
            self.kv_lora_rank, self.v_head_dim, self.num_heads
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.indexer = Glm5NextIndexer(config)

    def _dense_prefix_rows(self, length, mask, cache):
        """(leading dense rows, past length) for the causal-prefix bypass.

        A query at cache position ``pos`` sees ``floor((pos + 1) / kp)``
        complete kpool windows plus the partial-window tail, so whenever that
        candidate count is at most ``index_topk // kp`` the top-k block set
        equals the causal prefix ``[0..pos]`` and the row is plain dense causal
        attention. Returns ``(0, 0)`` (fail closed) whenever eligibility is
        not proven: without the always-select-tail the selection would omit
        the partial window, and batched/padded states keep the original path.
        """
        kp = self.indexer.index_kpool
        if kp < 1 or (kp > 1 and not self.indexer.index_kpool_always_select_tail):
            return 0, 0
        kv, pool = _cache_parts(cache)
        past_len = 0
        if kv is not None:
            # A merged variable-length batch keeps a per-row (array) offset.
            offset = getattr(kv, "offset", None)
            if (
                type(kv).__name__ != "KVCache"
                or getattr(kv, "left_padding", None) is not None
                or not isinstance(offset, int)
                or offset < 0
            ):
                return 0, 0
            past_len = offset
        if pool is not None and isinstance(Glm5NextIndexer._processed(pool), list):
            return 0, 0
        if mask is not None:
            if not isinstance(mask, mx.array) or mask.ndim < 2:
                return 0, 0
            if mask.shape[-1] < past_len + length or mask.shape[-2] != length:
                return 0, 0
        select_max = self.indexer.index_topk // kp
        boundary = (select_max + 1) * kp - 1
        return min(length, max(0, boundary - past_len)), past_len

    def _dense_flat(self, q, kv_latent, mask, rows, past_len):
        """Dense causal attention over the first ``rows`` rows.

        Uses the expanded per-head K/V of the short-context path. The fused
        SDPA kernel has no 512-wide head, so latent-space SDPA would
        materialize the full score matrix.

        With an explicit boolean mask (the engine's prefill; SDPA runs its
        unfused fallback: scores, masked select, softmax, P x V) the rows run
        in causal blocks: block ``[a, b)`` only takes keys ``[0, past_len +
        b)``. Every key past that is masked for all rows of the block, i.e.
        gets probability exactly 0, and the softmax maps each score to the
        same thread whatever the row length, so each block reproduces its
        rows of the one-call result bitwise while skipping ~44% of the
        score / value work (8 blocks).
        """
        kv_rows = kv_latent[:, :, : past_len + rows]
        k = self.embed_q(kv_rows, transpose=False)
        v = self.unembed_out(kv_rows)
        q_rows = q[:, :, :rows]
        if k.dtype != q_rows.dtype:
            k = k.astype(q_rows.dtype)
            v = v.astype(q_rows.dtype)
        n_blocks = _DENSE_ROW_BLOCKS if mask is not None else 1
        n_blocks = max(1, min(n_blocks, rows // 256))
        if n_blocks == 1:
            dense_mask = (
                "causal" if mask is None else mask[..., :rows, : past_len + rows]
            )
            out = mx.fast.scaled_dot_product_attention(
                q_rows, k, v, scale=self.scale, mask=dense_mask
            )
        else:
            bounds = [0]
            bounds += [(rows * i // n_blocks) // 64 * 64 for i in range(1, n_blocks)]
            bounds.append(rows)
            outs = []
            for a, b in zip(bounds[:-1], bounds[1:]):
                end = past_len + b
                outs.append(
                    mx.fast.scaled_dot_product_attention(
                        q_rows[:, :, a:b],
                        k[:, :, :end],
                        v[:, :, :end],
                        scale=self.scale,
                        mask=mask[..., a:b, :end],
                    )
                )
            out = mx.concatenate(outs, axis=2)
        return out.transpose(0, 2, 1, 3).reshape(q.shape[0], rows, -1)

    def _finish(self, flat, out_dense):
        if out_dense is not None:
            flat = mx.concatenate([out_dense, flat], axis=1)
        return linear_forward(self.o_proj, flat)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        length = x.shape[1]
        if length > 8:
            dense_rows, past_len = self._dense_prefix_rows(length, mask, cache)
            if dense_rows:
                return self._forward(x, mask, cache, dense_rows, past_len)
        return self._forward(x, mask, cache)

    def _forward(
        self,
        x: mx.array,
        mask: Optional[mx.array],
        cache: Optional[Any],
        dense_rows: int = 0,
        past_len: int = 0,
    ) -> mx.array:
        B, L, D = x.shape

        projected = self._decode_projections(x, cache)
        if projected is None:
            qr = self.q_a_layernorm(linear_forward(self.q_a_proj, x))
            q = linear_forward(self.q_b_proj, qr)
            compressed_kv = linear_forward(self.kv_a_proj_with_mqa, x)
            indexer_projected = None
        else:
            qr, q, compressed_kv, indexer_projected = projected
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)
        # One token with index selection: embed_q only needs q, so compute it
        # here and order the indexer's query projection after it; it then runs
        # beside the selection chain instead of after it (same values).
        q_latent_early = None
        if (
            L == 1
            and mask is None
            and indexer_projected is not None
            and indexer_projected[0] is not None
        ):
            q_latent_early = _mla_head_proj(self.embed_q, q)
            indexer_projected = (
                mx.depends(indexer_projected[0], [q_latent_early]),
            ) + tuple(indexer_projected[1:])
        kv_latent = self.kv_a_layernorm(compressed_kv)
        kv_latent = mx.expand_dims(kv_latent, axis=1)

        if cache is not None:
            # NoPE attention only needs the latent once. A zero-width value
            # cache avoids storing a duplicate 512-wide tensor per sparse
            # layer while retaining the standard KVCache lifecycle.
            empty_values = mx.zeros((B, 1, L, 0), dtype=kv_latent.dtype)
            kv_latent, _ = cache[0].update_and_fetch(kv_latent, empty_values)
        else:
            cache = [None] * 2

        topk_indices = self.indexer(
            x,
            qr,
            mask,
            cache=cache[1],
            kv_cache=cache[0],
            score_from=dense_rows,
            projected=indexer_projected,
        )
        out_dense = None
        if dense_rows and topk_indices is not None:
            # The leading rows' selection was the causal prefix: dense SDPA over
            # the prefix, with the indexer scoring only the tail rows.
            # When the indexer bypasses selection for the whole forward, keep
            # the pre-existing all-rows path and drop the dense split entirely.
            out_dense = self._dense_flat(q, kv_latent, mask, dense_rows, past_len)
            q = q[:, :, dense_rows:]
            L -= dense_rows
            if mask is not None:
                mask = mask[..., dense_rows:, :]
        attn_mask = mask
        q_latent = q_latent_early
        if topk_indices is not None:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            gathered = None
            if L == 1 and mask is None and _DECODE_FUSION:
                gathered = _decode_kernels.dsa_gather_selected(
                    kv_latent, topk_indices[:, :, 0, :]
                )
            if gathered is not None:
                kv_latent, attn_mask = gathered
            elif L == 1:
                clamped = mx.clip(topk_indices[:, :, 0, :], 0, Kv - 1)
                idx = clamped[..., None]
                kv_latent = mx.take_along_axis(
                    kv_latent,
                    mx.broadcast_to(idx, idx.shape[:-1] + (kv_latent.shape[-1],)),
                    axis=2,
                )
                sel_mask = valid_sel[:, :, 0, :][:, :, None, :]
                if mask is not None and mask.dtype == mx.bool_:
                    # Single-stream decode passes mask=None here; under continuous
                    # batching the batched cache supplies a left-pad mask that can be
                    # 4-D ([B, 1, 1, Kv]) while `clamped` is 3-D. At S=1 the mask is
                    # purely per-key (no causal), so reduce it to [B, Kv] and gather the
                    # selected key positions -- rank-agnostic and batch-safe.
                    mkeys = mask.reshape(B, -1, Kv)[:, 0, :]
                    gathered = mx.take_along_axis(
                        mx.broadcast_to(mkeys[:, None, :], (B, clamped.shape[1], Kv)),
                        clamped,
                        axis=-1,
                    )
                    sel_mask = sel_mask & gathered[:, :, None, :]
                attn_mask = sel_mask
            elif L <= 8:
                return self._finish(
                    self._gathered_attention(q, kv_latent, topk_indices), out_dense
                )
            else:
                q_latent = self.embed_q(q)
                # Native DSA requires FP16/BF16 inputs; quantized projections can yield FP32.
                # Cast at the kernel boundary and preserve the residual stream dtype.
                native_dtype = (
                    mx.float16 if q_latent.dtype == mx.float32 else q_latent.dtype
                )
                q_latent = q_latent.astype(native_dtype)
                kv_latent_native = kv_latent.astype(native_dtype)
                # Tensor-unit kernel (M5): same fp32 math as the native
                # kernel, at any context the indexer runs for.
                output = sparse_mla_attention_nax(
                    q_latent, kv_latent_native, topk_indices, self.scale
                )
                if output is None and Kv >= 4096:
                    q_pe = mx.zeros(q_latent.shape[:-1] + (64,), dtype=native_dtype)
                    k_pe = mx.zeros(kv_latent.shape[:-1] + (64,), dtype=native_dtype)
                    output = sparse_mla_attention(
                        q_latent,
                        q_pe,
                        kv_latent_native,
                        k_pe,
                        topk_indices,
                        self.scale,
                    )
                if output is not None:
                    output_flat = q8_vup_flat(
                        output,
                        self.unembed_out,
                        key_length=Kv,
                    )
                    if output_flat is None:
                        output = self.unembed_out(output)
                        output_flat = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self._finish(output_flat, out_dense)

                k = self.embed_q(kv_latent, transpose=False).astype(native_dtype)
                v = self.unembed_out(kv_latent).astype(native_dtype)
                output = exact_block_token_attention(
                    q.astype(native_dtype),
                    k,
                    v,
                    topk_indices,
                    self.scale,
                )
                if output is not None:
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return self._finish(output, out_dense)

                shape = list(topk_indices.shape)
                shape[-1] = Kv + 1
                safe_idx = mx.where(valid_sel, topk_indices, Kv)
                sparse_mask = mx.zeros(shape, dtype=mx.bool_)
                sparse_mask = mx.put_along_axis(
                    sparse_mask, safe_idx, mx.array(True), axis=-1
                )[..., :Kv]
                if mask is not None and mask.dtype == mx.bool_:
                    sparse_mask = sparse_mask & mask
                attn_mask = sparse_mask

        if (
            cache is not None
            and cache[0] is not None
            and cache[1] is not None
            and cache[1].pooled is not None
        ):
            deps = tuple(v for v in cache[1].state if isinstance(v, mx.array))
            if deps:
                cache[0].keys = mx.depends(cache[0].keys, deps)

        # Short verification blocks use the same latent-space attention as
        # decode. Expanding every cached key and value into all heads makes
        # verification cost grow with the complete context length.
        if L <= 8:
            q = q_latent if q_latent is not None else _mla_head_proj(self.embed_q, q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        if L <= 8:
            output = _mla_head_proj(self.unembed_out, output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self._finish(output, out_dense)

    def _decode_projections(self, x, cache):
        """Decode/verify (B == 1, L <= 8) projections, one dispatch per input
        and quantization.

        q_a, kv_a and the indexer's wk (and weights_proj when the indexer
        will select) read ``x``; q_b (and the indexer's wq_b when selecting)
        read ``qr``. Projections sharing an input and (bits, group size) run
        as one ``decode_kernels.multi_qmv`` dispatch with the per-row
        arithmetic of the separate matmuls; the rest are the reference calls.
        Returns ``(qr, q, compressed_kv, indexer_projected)`` or None.
        """
        B, L, D = x.shape
        if not _DECODE_FUSION or B != 1 or L > _DECODE_BLOCK:
            return None
        if cache is None:
            return None
        idx = self.indexer
        processed = idx._processed(cache[1])
        total = (max(processed) if isinstance(processed, list) else processed) + L
        selecting = not (getattr(idx, "bypass_short", True) and total <= idx.index_topk)
        x_layers = [self.q_a_proj, self.kv_a_proj_with_mqa, idx.wk]
        if selecting:
            x_layers.append(idx.weights_proj)
        outs = _multi_linear(x, x_layers)
        if outs is None:
            return None
        qr = self.q_a_layernorm(outs[0])
        qr_outs = _multi_linear(qr, [self.q_b_proj] + ([idx.wq_b] if selecting else []))
        if qr_outs is None:
            qr_outs = [linear_forward(self.q_b_proj, qr)]
        indexer_projected = (
            qr_outs[1] if selecting and len(qr_outs) > 1 else None,
            outs[2],
            outs[3] if selecting else None,
        )
        return qr, qr_outs[0], outs[1], indexer_projected

    def _gathered_attention(self, q, kv_latent, topk_indices):
        """Latent-space gather for short query blocks; returns pre-o_proj flat."""
        B, H, L, _ = q.shape
        Kv = kv_latent.shape[2]
        dim = kv_latent.shape[-1]
        selected = topk_indices[:, 0]
        topk = selected.shape[-1]
        q_embedded = self.embed_q(q)
        clamped = mx.clip(selected, 0, Kv - 1)
        gathered = mx.take_along_axis(
            mx.broadcast_to(kv_latent[:, 0, None], (B, L, Kv, dim)),
            mx.broadcast_to(clamped[..., None], (B, L, topk, dim)),
            axis=2,
        )
        q_latent = q_embedded.transpose(0, 2, 1, 3).reshape(B * L, H, 1, dim)
        gathered = gathered.reshape(B * L, 1, topk, dim)
        valid = (selected >= 0).reshape(B * L, 1, 1, topk)
        output = scaled_dot_product_attention(
            q_latent,
            gathered,
            gathered,
            cache=None,
            scale=self.scale,
            mask=valid,
        )
        output = output.reshape(B, L, H, dim).transpose(0, 2, 1, 3)
        output = self.unembed_out(output).transpose(0, 2, 1, 3).reshape(B, L, -1)
        return output


@partial(mx.compile, shapeless=True)
def _clamped_swiglu(x_up: mx.array, x_gate: mx.array, limit: float) -> mx.array:
    # One fused elementwise kernel instead of clip/clip/silu/multiply passes
    # over the expert activations; the same ops in the same dtypes, so the
    # result is bit-identical to the eager chain.
    x_gate = mx.clip(x_gate, a_min=None, a_max=limit)
    x_up = mx.clip(x_up, a_min=-limit, a_max=limit)
    return nn.silu(x_gate) * x_up


class Glm5NextClampedSwiGLU(nn.Module):
    def __init__(self, limit: Optional[float]):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        if self.limit is not None:
            return _clamped_swiglu(x_up, x_gate, float(self.limit))
        return nn.silu(x_gate) * x_up


class Glm5NextMLP(nn.Module):
    def __init__(self, config, intermediate_size=None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.limit = config.swiglu_limit
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        if (
            _DECODE_FUSION
            and self.limit is not None
            and x.ndim == 3
            and x.shape[:2] == (1, 1)
        ):
            # One token: gate/up + clamped SwiGLU in one exact dispatch.
            act = _decode_kernels.mlp_gate_up_swiglu(
                x.reshape(1, -1), self.gate_proj, self.up_proj, self.limit
            )
            if act is not None:
                return linear_forward(self.down_proj, act.reshape(1, 1, -1))
        gate = linear_forward(self.gate_proj, x)
        up = linear_forward(self.up_proj, x)
        if self.limit is not None:
            return linear_forward(
                self.down_proj, _clamped_swiglu(up, gate, float(self.limit))
            )
        return linear_forward(self.down_proj, nn.silu(gate) * up)


class Glm5NextMoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.routed_scaling_factor = config.routed_scaling_factor
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))

    def __call__(self, x):
        if (
            _DECODE_FUSION
            and x.ndim == 3
            and x.shape[:2] == (1, 1)
            and self.n_group == 1
        ):
            # One token: the reference logits come from the one-row fp32
            # gemv, which the fused router reproduces (multi-row calls use
            # a different matmul and keep the reference path).
            routed = _decode_kernels.moe_router(
                x.reshape(1, -1),
                self.weight,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
            )
            if routed is not None:
                indices, scores = routed
                return indices.reshape(1, 1, -1), scores.reshape(1, 1, -1)
        if (
            _DECODE_FUSION
            and x.ndim == 3
            and x.shape[0] == 1
            and 2 <= x.shape[1] <= _DECODE_BLOCK
            and self.n_group == 1
        ):
            # Verify block: the reference logits come from MLX's NAX split-K
            # GEMM, which moe_router_rows reproduces op for op.
            routed = _decode_kernels.moe_router_rows(
                x.reshape(x.shape[1], -1),
                self.weight,
                self.e_score_correction_bias,
                self.top_k,
                self.routed_scaling_factor,
                self.norm_topk_prob,
            )
            if routed is not None:
                indices, scores = routed
                return indices.reshape(1, x.shape[1], -1), scores.reshape(1, x.shape[1], -1)
        logits = x.astype(mx.float32) @ self.weight.astype(mx.float32).T
        return group_expert_select(
            logits,
            self.e_score_correction_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
        )


class Glm5NextMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.switch_mlp = SwitchGLU(
            config.hidden_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=Glm5NextClampedSwiGLU(config.swiglu_limit),
        )
        self.gate = Glm5NextMoEGate(config)
        self.shared_experts = None
        if config.n_shared_experts is not None:
            self.shared_experts = Glm5NextMLP(
                config,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
            )

    def __call__(self, x):
        y = self._decode_select(x)
        if y is not None:
            return y
        indices, scores = self.gate(x)
        y = self._decode_experts(x, indices, scores)
        if y is not None:
            return y
        y = self.switch_mlp(x, indices, scores=scores, weighted_sum=True)
        if y.ndim == x.ndim + 1:
            y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y

    def _decode_select(self, x):
        """One token: router logits, then the gate/up kernel replaying the
        router's top-k selection in each routed threadgroup (one dependent
        dispatch less than router select + gate/up), then down/combine.
        Bit-identical to the gate + _decode_experts path; None when not
        covered."""
        dk = _decode_kernels
        gate = self.gate
        sw = self.switch_mlp
        shared = self.shared_experts
        if (
            not _DECODE_FUSION
            or x.ndim != 3
            or x.shape[:2] != (1, 1)
            or gate.n_group != 1
            or shared is None
            or not isinstance(sw, SwitchGLU)
            or gate.top_k >= _sort_threshold(*_switch_projections(sw))
        ):
            return None
        limit = getattr(sw.activation, "limit", None)
        if limit is None or shared.limit != limit:
            return None
        x2 = x.reshape(1, -1)
        logits = dk.moe_router_logits(x2, gate.weight, gate.e_score_correction_bias)
        if logits is None:
            return None
        if "gate_up_proj" in sw:
            routed_gate, routed_up = sw.gate_up_proj, None
        else:
            routed_gate, routed_up = sw.gate_proj, sw.up_proj
        fused = dk.moe_gate_up_swiglu(
            x2, None, limit, routed_gate, routed_up, shared.gate_proj, shared.up_proj,
            select=(*logits, gate.top_k, gate.routed_scaling_factor, gate.norm_topk_prob),
        )
        if fused is None:
            return None
        act, routes, weights = fused
        y = dk.moe_down_combine(act, routes, weights, sw.down_proj, shared.down_proj)
        return None if y is None else y.reshape(x.shape)

    def _decode_experts(self, x, indices, scores):
        """Fused expert path for one sequence whose routes SwitchGLU leaves unsorted.

        Two dispatches replace the gathered gate/up/down products, the
        clamped SwiGLU, the routing-weighted sum and the shared-expert add
        (bit-identical: each row reproduces the one-token qmv arithmetic
        the unsorted gather path uses). For L > 1 the shared expert keeps
        its own multi-row projection and is added in the down kernel.
        Returns None when the shape is not covered.
        """
        sw = self.switch_mlp
        shared = self.shared_experts
        if (
            not _DECODE_FUSION
            or x.ndim != 3
            or x.shape[0] != 1
            or indices.shape[:2] != x.shape[:2]
            or not isinstance(sw, SwitchGLU)
            or indices.size >= _sort_threshold(*_switch_projections(sw))
        ):
            return None
        # Separate gate/up projections, or one fused [gate; up] gate_up_proj
        # (MoE gate/up fusion); the kernels read either layout in place.
        if "gate_up_proj" in sw:
            routed_gate, routed_up = sw.gate_up_proj, None
        else:
            routed_gate, routed_up = sw.gate_proj, sw.up_proj
        limit = getattr(sw.activation, "limit", None)
        if limit is None or (shared is not None and shared.limit != limit):
            return None
        T, D = x.shape[1], x.shape[2]
        x2 = x.reshape(T, D)
        routes = indices.reshape(T, -1).astype(mx.uint32)
        weights = scores.reshape(T, -1)
        dk = _decode_kernels
        if shared is not None and T == 1:
            act = dk.moe_gate_up_swiglu(
                x2, routes, limit, routed_gate, routed_up,
                shared.gate_proj, shared.up_proj,
            )
            if act is None:
                return None
            y = dk.moe_down_combine(act, routes, weights, sw.down_proj, shared.down_proj)
        else:
            fused = None
            if shared is not None:
                # One dispatch also computes the shared expert's gate/up with
                # the multi-row qmv_wide arithmetic its own T > 1 call uses.
                fused = dk.moe_gate_up_swiglu(
                    x2, routes, limit, routed_gate, routed_up,
                    shared.gate_proj, shared.up_proj, shared_wide=True,
                )
            if fused is not None:
                act, shared_act = fused
                # The shared expert's down projection (qmv_wide rows) folds
                # into the combine kernel when covered.
                y = dk.moe_down_combine(
                    act, routes, weights, sw.down_proj, shared.down_proj,
                    shared_act=shared_act,
                )
                if y is not None:
                    return y.reshape(x.shape)
                shared_y = linear_forward(
                    shared.down_proj, shared_act.reshape(1, T, -1)
                ).reshape(T, D)
            else:
                act = dk.moe_gate_up_swiglu(x2, routes, limit, routed_gate, routed_up)
                if act is None:
                    return None
                shared_y = None if shared is None else shared(x).reshape(T, D)
            y = dk.moe_down_combine(act, routes, weights, sw.down_proj, shared_y=shared_y)
        return None if y is None else y.reshape(x.shape)


class Glm5NextDecoderLayer(nn.Module):
    def __init__(self, config: TextConfig, layer_idx: int):
        super().__init__()
        layer_type = config.layer_types[layer_idx]
        self.is_linear = layer_type == "linear_attention"
        if self.is_linear:
            self.self_attn = Glm5NextLinearAttention(config)
        else:
            self.self_attn = Glm5NextSparseAttention(config)

        is_sparse = (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and config.mlp_layer_types[layer_idx] == "sparse"
        )
        self.mlp = Glm5NextMoE(config) if is_sparse else Glm5NextMLP(config)

        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)
        self.compile_ffn = True
        self._ffn_c = None
        self._ffn_dc = None

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        defer: bool = False,
    ) -> mx.array:
        if _DECODE_FUSION:
            # Settle (eagerly, once) how MLX's eager fp32 sigmoid evaluates;
            # the fused router inside the compiled FFN block follows it and
            # cannot probe while being traced.
            _decode_kernels.eager_sigmoid_precise(mx.float32)
        # One-token decode can leave this layer's last HC expand to the next
        # layer's fused HC pre (``defer``: returns an _HCDeferred); an
        # _HCDeferred input is always accepted.
        if defer or isinstance(x, _HCDeferred):
            out = self._decode_deferred(x, mask, cache)
            if out is not None:
                return out if defer else out.materialize()
            if isinstance(x, _HCDeferred):
                x = x.materialize()
        residual = x
        fused = _decode_hc_pre(self.attn_hc, self.input_layernorm, x)
        if fused is None:
            xc, post, comb = self.attn_hc(x)
            xn = self.input_layernorm(xc)
        else:
            xn, post, comb = fused
        r = self.self_attn(xn, mask, cache)
        x = _decode_hc_expand(r, residual, post, comb)
        # Compile the FFN block only for single-stream decode (B=1, S=1) -- the shape it
        # was validated on and where its win lives. Compiling the 288-expert MoE at a
        # batched or prefill shape spikes memory (it can OOM alongside the resident
        # weights), so those shapes take the eager path.
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] == 1:
            if self._ffn_c is None:
                self._ffn_c = compile_ffn_block(self, self._ffn_block)
            return self._ffn_c(x)
        return self._ffn_block(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        # Stateless FFN half (no cache) -> compiles cleanly at a fixed decode shape.
        residual = x
        fused = _decode_hc_pre(self.ffn_hc, self.post_attention_layernorm, x)
        if fused is None:
            xc, post, comb = self.ffn_hc(x)
            xn = self.post_attention_layernorm(xc)
        else:
            xn, post, comb = fused
        m = self.mlp(xn)
        return _decode_hc_expand(m, residual, post, comb)

    def _decode_deferred(self, x, mask, cache) -> Optional[_HCDeferred]:
        """The one-token layer with both HC pres from ``_decode_hc_pre_deferred``
        (each folding in the previous expand); None when not covered."""
        if isinstance(x, _HCDeferred):
            dtype, width = x.y.dtype, x.y.shape[-1]
        elif x.ndim == 4 and x.shape[:2] == (1, 1):
            dtype, width = x.dtype, x.shape[-1]
        else:
            return None
        if not (
            _hc_defer_ok(self.attn_hc, self.input_layernorm, dtype, width)
            and _hc_defer_ok(self.ffn_hc, self.post_attention_layernorm, dtype, width)
        ):
            return None
        xn, h, post, comb, mm = _decode_hc_pre_deferred(
            self.attn_hc, self.input_layernorm, x
        )
        r = self.self_attn(xn, mask, cache)
        if self.compile_ffn:
            if self._ffn_dc is None:
                self._ffn_dc = compile_ffn_block(self, self._ffn_block_deferred)
            return _HCDeferred(*self._ffn_dc(r, h, post, comb, mm))
        return _HCDeferred(*self._ffn_block_deferred(r, h, post, comb, mm))

    def _ffn_block_deferred(self, y, residual, post, comb, mm):
        xn, h, post, comb, mm = _decode_hc_pre_deferred(
            self.ffn_hc,
            self.post_attention_layernorm,
            _HCDeferred(y, residual, post, comb, mm),
        )
        return self.mlp(xn), h, post, comb, mm


class Glm5NextModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.hc_mult = config.hc_mult
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Glm5NextDecoderLayer(config, idx) for idx in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ssm_idx = next((i for i, l in enumerate(self.layers) if l.is_linear), 0)
        self.fa_idx = next((i for i, l in enumerate(self.layers) if not l.is_linear), 0)

    def __call__(
        self,
        inputs: mx.array,
        cache: Optional[Any] = None,
        inputs_embeds: Optional[mx.array] = None,
    ) -> mx.array:
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds

        if cache is None:
            cache = [None] * len(self.layers)

        fa_cache = cache[self.fa_idx]
        fa_mask = create_attention_mask(
            h, fa_cache[0] if fa_cache else None, return_array=True
        )
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx])

        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[2])
        )
        h = mx.contiguous(h)

        # Evaluate layer by layer to bound prefill memory, but pipelined: the
        # GPU runs layer i while the host builds layer i + 1 (at most two
        # layers in flight). Decode and verify blocks only start evaluation
        # early. The MTP replacement loop must use the same policy.
        prefill = h.shape[1] >= 256
        # Each completed layer is waited for and the allocator cache is
        # released (layer-specific buffer sizes would otherwise accumulate).
        # The last layer stays lazy: a prefill chunk only needs its cache update.
        pipeline = (
            LayerPipeline(on_evaluated=mx.clear_cache, lazy_last=True)
            if prefill
            else None
        )
        # One-token decode: start encoding the step every few layers while the
        # rest of the graph is still being built (scheduling only, see
        # _DECODE_EVAL_EVERY).
        eval_every = _DECODE_EVAL_EVERY if h.shape[1] == 1 else 0
        n_layers = len(self.layers)
        # One token: each layer's last HC expand runs inside the next layer's
        # first HC pre (see _decode_hc_pre_deferred); the last one here.
        defer = h.shape[:2] == (1, 1)

        for i, (layer, c) in enumerate(zip(self.layers, cache)):
            mask = ssm_mask if layer.is_linear else fa_mask
            if defer:
                h = layer(h, mask=mask, cache=c, defer=i + 1 < n_layers)
            else:
                h = layer(h, mask=mask, cache=c)
            if pipeline is not None:
                pipeline.push(h)
            elif eval_every and (i + 1) % eval_every == 0 and i + 1 < n_layers:
                mx.async_eval(h.arrays() if isinstance(h, _HCDeferred) else h)
        if pipeline is not None:
            pipeline.drain()

        h = h.mean(axis=2)
        return self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, args: TextConfig, config: ModelConfig = None):
        super().__init__()
        self.args = args
        self.config = args
        self.model_type = args.model_type
        self.model = Glm5NextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: Optional[mx.array] = None,
        inputs_embeds: Optional[mx.array] = None,
        cache: Optional[Any] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> LanguageModelOutput:
        if inputs is None:
            inputs = kwargs.get("input_ids")
        out = self.model(inputs, cache=cache, inputs_embeds=inputs_embeds)
        # Only the last few positions' logits are ever needed for generation; slicing
        # before the (vocab-wide) projection skips it on discarded prefill positions.
        nlk = kwargs.get("num_logits_to_keep", 0)
        if nlk:
            out = out[:, -nlk:, :]
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = linear_forward(self.lm_head, out)
        return LanguageModelOutput(logits=out)

    def sanitize(self, weights):
        weights = {k: v for k, v in weights.items() if "mtp." not in k}
        weights = DSV32Model.sanitize(self, weights)

        remapped = {}
        conv_parts = {}
        fg_parts = (
            "A_log",
            "dt_bias",
            "f_a_proj.weight",
            "f_a_proj.scales",
            "f_a_proj.biases",
            "f_b_proj.weight",
            "f_b_proj.scales",
            "f_b_proj.biases",
        )
        for k, v in weights.items():
            nk = k.replace(".hc_attn_", ".attn_hc.").replace(".hc_ffn_", ".ffn_hc.")

            fused = False
            for part in ("q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"):
                suffix = ".self_attn." + part
                if nk.endswith(suffix):
                    prefix = nk[: -len(part)]
                    conv_parts.setdefault(prefix, {})[part[0]] = v
                    fused = True
                    break
            if fused:
                continue

            for p in fg_parts:
                suffix = ".self_attn." + p
                if nk.endswith(suffix):
                    nk = nk[: -len(p)] + "forget_gate." + p
                    break

            remapped[nk] = v

        for prefix, parts in conv_parts.items():
            if all(c in parts for c in ("q", "k", "v")):
                remapped[prefix + "conv1d.weight"] = mx.concatenate(
                    [parts["q"], parts["k"], parts["v"]], axis=0
                )
            else:
                for c, w in parts.items():
                    remapped[prefix + c + "_conv1d.weight"] = w

        weights = remapped
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.ndim == 3 and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
        for k, v in list(weights.items()):
            keep_fp32 = (
                ".attn_hc." in k
                or ".ffn_hc." in k
                or k.endswith("A_log")
                or k.endswith("dt_bias")
                or k.endswith("mlp.gate.weight")
                or k.endswith("e_score_correction_bias")
            )
            if (
                keep_fp32
                and mx.issubdtype(v.dtype, mx.floating)
                and v.dtype != mx.float32
            ):
                weights[k] = v.astype(mx.float32)
        return weights

    @property
    def layers(self):
        return self.model.layers

    @property
    def cast_predicate(self):
        return glm5_next_cast_predicate

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or "e_score_correction_bias" in path:
                return False
            if ".indexer" in path:
                return {"group_size": 64, "bits": 8}
            return True

        return predicate

    def make_cache(self):
        caches = []
        for layer in self.layers:
            if layer.is_linear:
                caches.append(ArraysCache(size=2))
            else:
                from mlx_lm.models.cache import PoolingCache

                caches.append(
                    CacheList(
                        KVCache(), PoolingCache(layer.self_attn.indexer.index_kpool)
                    )
                )
        return caches
