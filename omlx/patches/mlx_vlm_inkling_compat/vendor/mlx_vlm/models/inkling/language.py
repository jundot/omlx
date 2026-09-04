# Vendored from mlx-vlm PR #1756 (pcuenca/mlx-vlm@73e4cb6, models/inkling),
# with the batching fixes from PR #1763 and decode fast paths from PR #1759.
# oMLX modifications, each marked with an "OMLX:" comment at the site:
# - batched right-padded prefill support: conv_mask wiring from
#   ArraysCache.make_mask, lengths-aware conv state writes, per-sequence
#   key-validity masking and log-tau positions in attention, and per-layer
#   cache.advance. Upstream never passes conv_mask (single-request
#   mlx_vlm.generate has no padding), so batch prefill would pollute the
#   short-conv states and attend to padded keys.
# - explicit error when a config with dense MLP layers lacks
#   dense_intermediate_size (upstream crashes with an opaque TypeError).
import os  # OMLX: sliding-window slice kill switch
from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

# OMLX: compute-only optimization — sliding layers (35 of 42 on Inkling
# Small) never attend past their 512-token window, but upstream runs SDPA
# over the FULL cached sequence and cuts via the additive mask, making
# every sliding layer O(S) per decoded token. Slicing K/V to the window
# before SDPA is numerically equivalent (masked keys carry -1e30 and
# contribute exactly 0 after softmax; distances are shift-invariant).
# Kill switch for A/B measurement: OMLX_INKLING_SLIDING_SLICE=0.
_SLIDING_WINDOW_SLICE = os.environ.get("OMLX_INKLING_SLIDING_SLICE", "1") != "0"

from ..base import LanguageModelOutput, scaled_dot_product_attention
from ..cache import ArraysCache, CacheList, KVCache
from ..mlp import SwiGLUMLP
from ..switch_layers import SwitchGLU, _gather_sort, _scatter_unsort
from .config import TextConfig as ModelConfig


def _clone_cache_tree(value):
    if isinstance(value, mx.array):
        return mx.array(value)
    if isinstance(value, tuple):
        return tuple(_clone_cache_tree(v) for v in value)
    if isinstance(value, list):
        return [_clone_cache_tree(v) for v in value]
    if isinstance(value, dict):
        return {k: _clone_cache_tree(v) for k, v in value.items()}
    return value


_CACHE_DICT_STATE = object()


def _subcaches(cache):
    return getattr(cache, "caches", None) or (cache,)


def _snapshot_cache_state(caches):
    """Copy cache state, including empty KV and ArraysCache metadata."""
    snapshot = []
    for cache in caches:
        if cache is None:
            snapshot.append(None)
            continue
        states = []
        for subcache in _subcaches(cache):
            if (
                isinstance(subcache, ArraysCache)
                or getattr(subcache, "keys", False) is None
            ):
                states.append((_CACHE_DICT_STATE, _clone_cache_tree(vars(subcache))))
            else:
                states.append(_clone_cache_tree(subcache.state))
        snapshot.append(states)
    arrays = [v for _, v in tree_flatten(snapshot) if isinstance(v, mx.array)]
    if arrays:
        mx.eval(arrays)
    return snapshot


def _restore_cache_state(caches, snapshot):
    for cache, states in zip(caches, snapshot):
        if cache is None or states is None:
            continue
        for subcache, state in zip(_subcaches(cache), states):
            if isinstance(state, tuple) and state and state[0] is _CACHE_DICT_STATE:
                subcache.__dict__.clear()
                subcache.__dict__.update(_clone_cache_tree(state[1]))
            else:
                subcache.state = _clone_cache_tree(state)


# OMLX: S / LQ / Q_OFF / B are RUNTIME params (uint32 buffer), not template
# constants. Every template tuple compiles its own Metal pipeline, and these
# values change every decode step (cache growth) and every MTP verify/head
# cycle — templating them forced a pipeline recompile per step on the seven
# global layers and per pass on the MTP head blocks. Only genuinely static
# per-layer constants (dtype, H, D_REL, REL_EXTENT, SLIDING) stay templated.
_MASK_SRC = r"""
    uint j  = thread_position_in_grid.x;   // key   position [0, S)
    uint i  = thread_position_in_grid.y;   // query position [0, LQ)
    uint bh = thread_position_in_grid.z;   // b * H + h
    uint S = params[0], LQ = params[1], B = params[3];
    int q_off = (int)params[2];
    if (i >= LQ || j >= S || bh >= B * H) return;
    uint b = bh / H, h = bh % H;
    int dist = (int(i) + q_off) - int(j);   // backward distance
    T val;
    if (dist < 0) {
        val = (T)(-1e30f);                                   // causal
    } else if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);                                   // sliding-window cap
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        uint rbase = ((b * LQ + i) * H + h) * D_REL;
        uint pcol = (uint)dist;
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + d] * (float)proj[d * REL_EXTENT + pcol];
        val = (T)acc;
    } else {
        val = (T)0;                                          // in-context, outside band
    }
    out[((b * H + h) * LQ + i) * S + j] = val;
"""
_mask_kernel = mx.fast.metal_kernel(
    name="inkling_banded_mask",
    input_names=["rel", "proj", "params"],
    output_names=["out"],
    source=_MASK_SRC,
)


def _rup(a, m):
    return ((a + m - 1) // m) * m


def banded_additive_mask(rel, proj, q_offset, S, sliding, rel_extent):
    """rel: [B, LQ, H, d_rel]; proj: [d_rel, rel_extent] -> additive mask [B, H, LQ, S]."""
    B, LQ, H, d_rel = rel.shape
    dtype = rel.dtype
    if mx.default_device() == mx.gpu:
        params = mx.array([S, LQ, int(q_offset), B], dtype=mx.uint32)
        return _mask_kernel(
            inputs=[rel, proj, params],
            template=[
                ("T", dtype),
                ("H", H),
                ("D_REL", d_rel),
                ("REL_EXTENT", rel_extent),
                ("SLIDING", sliding),
            ],
            grid=(_rup(S, 8), _rup(LQ, 8), B * H),
            threadgroup=(8, 8, 1),
            output_shapes=[(B, H, LQ, S)],
            output_dtypes=[dtype],
        )[0]
    rl = (rel @ proj).transpose(0, 2, 1, 3)
    qp = mx.arange(LQ) + q_offset
    kp = mx.arange(S)
    dist = qp[:, None] - kp[None, :]
    gidx = mx.broadcast_to(mx.clip(dist, 0, rel_extent - 1)[None, None], (B, H, LQ, S))
    pb = mx.take_along_axis(rl, gidx, axis=-1)
    pb = mx.where((dist >= rel_extent)[None, None], mx.array(0.0, dtype), pb)
    neg = dist < 0
    if sliding > 0:
        neg = neg | (dist >= sliding)
    return mx.where(neg[None, None], mx.array(-1e30, dtype), pb).astype(dtype)


def _next_conv_state(state, inputs, mask):
    state_size = state.shape[1]
    if state_size == 0:
        return state
    if mask is None:
        return mx.concatenate([state, inputs], axis=1)[:, -state_size:]

    valid_count = mx.sum(mask, axis=1).astype(mx.int32)
    valid_start = mx.argmax(mask.astype(mx.int32), axis=1)
    logical = mx.arange(state_size)[None, :] + valid_count[:, None] - state_size
    state_indices = state_size + logical
    input_indices = state_size + valid_start[:, None] + mx.maximum(logical, 0)
    indices = mx.where(logical >= 0, input_indices, state_indices)
    combined = mx.concatenate([state, inputs], axis=1)
    indices = mx.broadcast_to(indices[..., None], (*indices.shape, inputs.shape[-1]))
    return mx.take_along_axis(combined, indices, axis=1)


def _cache_padding_mask(cache, length):
    """Combine left- and right-padding metadata for short convolution."""
    if cache is None:
        return None
    positions = mx.arange(length)[None, :]
    mask = None
    left_padding = getattr(cache, "left_padding", None)
    if left_padding is not None:
        mask = positions >= left_padding[:, None]
    lengths = getattr(cache, "lengths", None)
    if lengths is not None:
        length_mask = positions < lengths[:, None]
        mask = length_mask if mask is None else mask & length_mask
    return mask


def _record_conv_stash(cache, conv_idx, state, inputs):
    """Preserve the virtual padded input used by oMLX MTP rewinds."""
    stash = getattr(cache, "_omlx_verify_xp", None)
    if stash is None:
        stash = {}
        cache._omlx_verify_xp = stash
    xp = mx.concatenate([state, inputs], axis=1)
    rows = getattr(cache, "_omlx_stash_rows", 0)
    if rows:
        prev = stash.get(conv_idx)
        roll = xp if prev is None else mx.concatenate([prev, inputs], axis=1)
        if roll.shape[1] > rows:
            roll = roll[:, -rows:, :]
        stash[conv_idx] = roll
    else:
        stash[conv_idx] = xp


_SCONV_SRC = r"""
    uint c = thread_position_in_grid.x;
    uint b = thread_position_in_grid.y;
    if (c >= C || b >= B) return;
    float w0 = (float)w[c * K + 0];
    float w1 = (float)w[c * K + 1];
    float w2 = (float)w[c * K + 2];
    float w3 = (float)w[c * K + 3];
    for (uint i = 0; i < L; ++i) {
        float acc = 0.0f;
        for (uint k = 0; k < K; ++k) {
            int r = (int)(i + k) - (int)(K - 1);
            float v = (r < 0)
                ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
                : (float)x[(b * L + (uint)r) * C + c];
            float wk = (k == 0) ? w0 : (k == 1) ? w1 : (k == 2) ? w2 : w3;
            acc += wk * v;
        }
        float conv_r = (float)((T)acc);
        T inner = (T)(conv_r + (float)x[(b * L + i) * C + c]);
        if (HAS_RES) {
            out[(b * L + i) * C + c] =
                (T)((float)inner + (float)res[(b * L + i) * C + c]);
        } else {
            out[(b * L + i) * C + c] = inner;
        }
    }
    for (uint s = 0; s < K - 1; ++s) {
        int r = (int)(L + s) - (int)(K - 1);
        nstate[(b * (K - 1) + s) * C + c] = (r < 0)
            ? state[(b * (K - 1) + (uint)(r + (int)(K - 1))) * C + c]
            : (float)x[(b * L + (uint)r) * C + c];
    }
"""
_sconv_kernel = mx.fast.metal_kernel(
    name="omlx_inkling_sconv_decode",
    input_names=["x", "state", "w", "res"],
    output_names=["out", "nstate"],
    source=_SCONV_SRC,
)


class InklingShortConvolution(nn.Module):
    """Depthwise causal 1-D conv over the previous ``kernel_size - 1`` states, plus a
    residual add. Kept in fp32 for stability (matches the reference). ``conv_idx`` selects
    this conv's slot in the layer's shared conv cache."""

    def __init__(self, channels: int, kernel_size: int, conv_idx: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_idx = conv_idx
        self.conv = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )

    def __call__(
        self,
        x: mx.array,
        cache=None,
        mask: Optional[mx.array] = None,
        residual: Optional[mx.array] = None,
    ):
        dt = x.dtype
        K = self.kernel_size
        if (
            cache is not None
            and mask is None
            and K == 4
            and x.shape[1] <= 8
            and mx.default_device() == mx.gpu
        ):
            B, L, C = x.shape
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((B, K - 1, C), dtype=mx.float32)
            out, nstate = _sconv_kernel(
                inputs=[
                    x,
                    state,
                    self.conv.weight.reshape(-1),
                    residual if residual is not None else x,
                ],
                template=[
                    ("T", dt),
                    ("B", B),
                    ("L", L),
                    ("C", C),
                    ("K", K),
                    ("HAS_RES", residual is not None),
                ],
                grid=(_rup(C, 32), B, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(B, L, C), (B, K - 1, C)],
                output_dtypes=[dt, mx.float32],
            )
            cache[self.conv_idx] = nstate
            _record_conv_stash(cache, self.conv_idx, state, x.astype(mx.float32))
            return out

        xf = x.astype(mx.float32)
        res = xf
        if mask is not None:
            xf = mx.where(mask[..., None], xf, 0)
        if cache is not None:
            state = cache[self.conv_idx]
            if state is None:
                state = mx.zeros((xf.shape[0], K - 1, xf.shape[-1]), dtype=xf.dtype)
            xp = mx.concatenate([state, xf], axis=1)
            cache[self.conv_idx] = _next_conv_state(state, xf, mask)
            _record_conv_stash(cache, self.conv_idx, state, xf)
        else:
            xp = mx.pad(xf, [(0, 0), (K - 1, 0), (0, 0)])
        out = self.conv(xp.astype(self.conv.weight.dtype)).astype(mx.float32)
        out = (out + res).astype(dt)
        return out if residual is None else residual + out


def qkvr_fusion_enabled(config, layer_idx: int, *, mtp: bool = False) -> bool:
    """Return whether one layer can use the load-time stacked projection."""
    attr = "mtp_qkvr_fused_layers" if mtp else "qkvr_fused_layers"
    policy = getattr(config, attr, None)
    if policy is None or layer_idx >= len(policy):
        return True
    return bool(policy[layer_idx])


class InklingAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        qkvr_fused: Optional[bool] = None,
    ):
        super().__init__()
        self.is_sliding = config.layer_is_sliding(layer_idx)
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.n_heads = (
            config.swa_num_attention_heads
            if self.is_sliding
            else config.num_attention_heads
        )
        self.n_kv = (
            config.swa_num_key_value_heads
            if self.is_sliding
            else config.num_key_value_heads
        )
        self.sliding = config.sliding_window_size if self.is_sliding else 0
        self.rel_extent = (
            config.sliding_window_size if self.is_sliding else config.rel_extent
        )
        self.d_rel = config.d_rel
        self.scale = 1.0 / self.head_dim
        self.log_floor = None if self.is_sliding else config.log_scaling_n_floor
        self.log_alpha = config.log_scaling_alpha

        self.qkvr_fused = (
            qkvr_fusion_enabled(config, layer_idx)
            if qkvr_fused is None
            else bool(qkvr_fused)
        )
        self.qkvr_dims = (
            self.n_heads * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_kv * self.head_dim,
            self.n_heads * self.d_rel,
        )
        if self.qkvr_fused:
            self.qkvr_proj = nn.Linear(
                config.hidden_size, sum(self.qkvr_dims), bias=False
            )
        else:
            dq, dk, dv, dr = self.qkvr_dims
            self.q_proj = nn.Linear(config.hidden_size, dq, bias=False)
            self.k_proj = nn.Linear(config.hidden_size, dk, bias=False)
            self.v_proj = nn.Linear(config.hidden_size, dv, bias=False)
            self.r_proj = nn.Linear(config.hidden_size, dr, bias=False)
        self.o_proj = nn.Linear(
            self.n_heads * self.head_dim, config.hidden_size, bias=False
        )
        self.k_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=0
        )
        self.v_sconv = InklingShortConvolution(
            self.n_kv * self.head_dim, config.sconv_kernel_size, conv_idx=1
        )
        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rel_proj = mx.zeros((self.d_rel, self.rel_extent))

    def __call__(self, x, cache=None, conv_mask=None):
        B, L, _ = x.shape
        kv = cache[0] if cache is not None else None
        conv = cache[1] if cache is not None else None

        dq, dk, dv, _ = self.qkvr_dims
        if self.qkvr_fused:
            qkvr = self.qkvr_proj(x)
            q = qkvr[..., :dq]
            k = qkvr[..., dq : dq + dk]
            v = qkvr[..., dq + dk : dq + dk + dv]
            r = qkvr[..., dq + dk + dv :]
        else:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            r = self.r_proj(x)
        k = self.k_sconv(k, cache=conv, mask=conv_mask)
        v = self.v_sconv(v, cache=conv, mask=conv_mask)
        r = r.reshape(B, L, self.n_heads, self.d_rel)

        q = self.q_norm(q.reshape(B, L, self.n_heads, self.head_dim)).transpose(
            0, 2, 1, 3
        )
        k = self.k_norm(k.reshape(B, L, self.n_kv, self.head_dim)).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_kv, self.head_dim).transpose(0, 2, 1, 3)

        if kv is not None:
            k, v = kv.update_and_fetch(k, v)
        S = k.shape[2]
        # OMLX: buffer-coordinate query offset. Upstream read ``kv.offset``,
        # which is a per-sequence ARRAY on a merged BatchKVCache (batched
        # requests) and cannot feed the scalar Q_OFF kernel template. The
        # banded distances are shift-invariant under left padding, so
        # buffer coordinates (first query at S - L) give identical
        # distances for every layout; for the plain single-request cache
        # S - L equals the pre-update kv.offset upstream used.
        offset = S - L

        # OMLX: sliding layers only attend the last (sliding + L - 1)
        # keys; slice K/V to that window so per-token cost stays O(window)
        # instead of O(S). Distances are shift-invariant under the uniform
        # cut; log-tau never applies here (log_floor is None on sliding
        # layers), and the padding bounds below shift with ``cut``.
        cut = 0
        if _SLIDING_WINDOW_SLICE and self.sliding > 0 and S > self.sliding + L - 1:
            cut = S - (self.sliding + L - 1)
            k = k[:, :, cut:, :]
            v = v[:, :, cut:, :]
            S = k.shape[2]
            offset = S - L

        mask = banded_additive_mask(
            r, self.rel_proj.astype(x.dtype), offset, S, self.sliding, self.rel_extent
        )

        # OMLX: mask padded key positions per sequence. Right-padded
        # prefill chunks carry ``lengths`` on the conv cache (remaining
        # valid tokens; the absolute valid end in buffer coordinates is
        # (S - L) + lengths_b, which also stays correct once lengths goes
        # negative for an already-consumed short sequence). Merged batches
        # carry ``left_padding`` on the KV cache (front holes, valid keys
        # are j >= lp_b). Both can be active for a restored-prefix batch,
        # so the conditions combine.
        lengths = getattr(conv, "lengths", None) if conv is not None else None
        left_padding = getattr(kv, "left_padding", None) if kv is not None else None
        valid_k = None
        if lengths is not None:
            valid_k = mx.arange(S) < (offset + lengths)[:, None]
        if left_padding is not None:
            # OMLX: left_padding is an absolute buffer position; shift it
            # into the sliced coordinate frame.
            lower = mx.arange(S) >= (left_padding - cut)[:, None]
            valid_k = lower if valid_k is None else (valid_k & lower)
        if valid_k is not None:
            mask = mx.where(
                valid_k[:, None, None, :], mask, mx.array(-1e30, mask.dtype)
            )

        if self.log_floor is not None:
            qpos = (mx.arange(L) + offset + 1).astype(mx.float32)
            if left_padding is not None:
                # OMLX: per-sequence true positions under left padding.
                qpos = qpos[None, :] - left_padding[:, None].astype(mx.float32)
                tau_shape = (B, 1, L, 1)
            else:
                tau_shape = (1, 1, L, 1)
            tau = 1.0 + self.log_alpha * mx.log(mx.maximum(qpos / self.log_floor, 1.0))
            tau = tau.reshape(tau_shape).astype(x.dtype)
            q = q * tau
            mask = mx.where(mask > -1e29, mask * tau, mask)

        out = scaled_dot_product_attention(
            q, k, v, cache=None, scale=self.scale, mask=mask
        )
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out)


class InklingDenseMLP(SwiGLUMLP):
    """Dense SwiGLU MLP (shared ``SwiGLUMLP``) with a learned output scale."""

    def __init__(self, config: ModelConfig):
        # OMLX: fail with a clear message instead of an opaque TypeError.
        # Real Inkling checkpoints always set dense_intermediate_size in
        # config.json; guessing a fallback would build wrong shapes.
        if config.dense_intermediate_size is None:
            raise ValueError(
                "inkling config declares dense MLP layers but has no "
                "dense_intermediate_size; the checkpoint config.json is "
                "expected to define it"
            )
        super().__init__(config.hidden_size, config.dense_intermediate_size)
        self.global_scale = mx.ones((1,))

    def __call__(self, x):
        return super().__call__(x) * self.global_scale


class InklingSwitchGLU(SwitchGLU):
    def __init__(self, input_dims, hidden_dims, num_experts, **kwargs):
        super().__init__(input_dims, hidden_dims, num_experts, **kwargs)
        self.gate_scale = mx.ones((num_experts,))  # s13 (fused gate/up scale2)
        self.out_scale = mx.ones((num_experts,))  # s13 * s2
        self._scales_trivial = None

    def _per_expert(self, scale, idx, like):
        s = scale[idx].astype(like.dtype)
        return s.reshape(s.shape + (1,) * (like.ndim - s.ndim))

    def scales_trivial(self):
        if self._scales_trivial is None:
            self._scales_trivial = bool(
                (mx.all(self.gate_scale == 1) & mx.all(self.out_scale == 1)).item()
            )
        return self._scales_trivial

    def __call__(self, x, indices) -> mx.array:
        if self.scales_trivial():
            return super().__call__(x, indices)
        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)
        x_up = self.up_proj(x, idx, sorted_indices=do_sort)
        x_gate = self.gate_proj(x, idx, sorted_indices=do_sort)
        x_gate = x_gate * self._per_expert(self.gate_scale, idx, x_gate)
        x = self.down_proj(self.activation(x_up, x_gate), idx, sorted_indices=do_sort)
        x = x * self._per_expert(self.out_scale, idx, x)
        if do_sort:
            x = _scatter_unsort(x, inv_order, indices.shape)
        return x.squeeze(-2)


_ROUTE_SRC = r"""
    uint lane = thread_position_in_grid.x;
    uint n = thread_position_in_grid.y;
    if (n >= N) return;
    auto lg = logits + (size_t)n * (R + SH);
    float ws = wscale[0];
    constexpr int PER = (R + 31) / 32;
    float sc[PER];
    float tl_[PER];
    bool taken[PER];
    for (int t = 0; t < PER; ++t) {
        uint j = lane + (uint)t * 32u;
        taken[t] = false;
        if (j < R) {
            float l = (float)lg[j];
            tl_[t] = l;
            sc[t] = 1.0f / (1.0f + metal::exp(-l)) + (float)corr[j];
        } else {
            sc[t] = -INFINITY;
        }
    }
    uint bidx[K];
    float btl[K];
    for (uint kk = 0; kk < K; ++kk) {
        float lb = -INFINITY; int lt = -1;
        for (int t = 0; t < PER; ++t)
            if (!taken[t] && sc[t] > lb) { lb = sc[t]; lt = t; }
        float gb = simd_max(lb);
        ushort wl = (ushort)simd_min(lb == gb ? lane : 32u);
        uint wj = simd_shuffle(lt >= 0 ? lane + (uint)lt * 32u : 0u, wl);
        float wtl = simd_shuffle(lt >= 0 ? tl_[lt] : 0.0f, wl);
        if (lane == (uint)wl && lt >= 0 && sc[lt] == gb) taken[lt] = true;
        bidx[kk] = wj; btl[kk] = wtl;
    }
    float lp[K + SH];
    float m = -INFINITY;
    for (uint t = 0; t < K + SH; ++t) {
        float tv = (t < K) ? btl[t] : (float)lg[R + (t - K)];
        float a = -tv;
        float lad = metal::max(a, 0.0f)
                  + metal::log(1.0f + metal::exp(-metal::fabs(a)));
        lp[t] = -lad;
        m = metal::max(m, lp[t]);
    }
    float se = 0.0f;
    for (uint t = 0; t < K + SH; ++t) se += metal::exp(lp[t] - m);
    float lse = m + metal::log(se);
    if (lane < K) {
        idx[(size_t)n * K + lane] = bidx[lane];
        wk[(size_t)n * K + lane] = (T)(metal::exp(lp[lane] - lse) * ws);
    }
    T gv[SH];
    for (uint s = 0; s < SH; ++s) gv[s] = (T)(metal::exp(lp[K + s] - lse) * ws);
    device T* gp = gamma + (size_t)n * SH * I;
    for (uint t = lane; t < SH * I; t += 32u) gp[t] = gv[t / I];
"""
_route_kernel = mx.fast.metal_kernel(
    name="omlx_inkling_moe_route",
    input_names=["logits", "corr", "wscale"],
    output_names=["idx", "wk", "gamma"],
    source=_ROUTE_SRC,
)


_DOWN_COMBINE = True

_DOWN_COMBINE_SRC = r"""
    uint lane = thread_index_in_simdgroup;
    uint sg   = simdgroup_index_in_threadgroup;
    uint row  = threadgroup_position_in_grid.y;
    uint n    = threadgroup_position_in_grid.z;
    threadgroup float partial[8];
    if (sg < K) {
        uint e = idx[(size_t)n * K + sg];
        const device uint* wr = wq + ((size_t)e * OUT + row) * (IN / 8u);
        const device T* sr = sc + ((size_t)e * OUT + row) * GROUPS;
        const device T* br = bi + ((size_t)e * OUT + row) * GROUPS;
        const device T* xr = xin + ((size_t)n * K + sg) * IN;
        float s = (float)sr[lane];
        float b = (float)br[lane];
        float accq = 0.0f, accx = 0.0f;
        uint base = lane * 8u;
        uint xbase = lane * 64u;
        for (uint u = 0; u < 8u; ++u) {
            uint w8 = wr[base + u];
            for (uint t = 0; t < 8u; ++t) {
                float xv = (float)xr[xbase + u * 8u + t];
                accq += (float)((w8 >> (4u * t)) & 0xFu) * xv;
                accx += xv;
            }
        }
        float dot = simd_sum(s * accq + b * accx);
        if (lane == 0) {
            float dv = (float)((T)dot);
            partial[sg] = (float)((T)(dv * (float)wk[(size_t)n * K + sg]));
        }
    } else if (lane == 0) {
        partial[sg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        float tot = 0.0f;
        for (uint t = 0; t < 8u; ++t) tot += partial[t];
        out[(size_t)n * OUT + row] = (T)tot;
    }
"""
_down_combine_kernel = mx.fast.metal_kernel(
    name="omlx_inkling_moe_down_combine",
    input_names=["xin", "wq", "sc", "bi", "idx", "wk"],
    output_names=["out"],
    source=_DOWN_COMBINE_SRC,
)


@partial(mx.compile, shapeless=True)
def _swiglu_scaled(gate, up, scale):
    return nn.silu(gate) * up * scale


class InklingSharedExpertsDense(nn.Module):
    """Concatenate always-on shared experts into three dense projections."""

    def __init__(self, input_dims: int, hidden_dims: int, num_experts: int):
        super().__init__()
        self.gate_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.up_proj = nn.Linear(input_dims, num_experts * hidden_dims, bias=False)
        self.down_proj = nn.Linear(num_experts * hidden_dims, input_dims, bias=False)

    def __call__(self, x, gamma):
        return self.down_proj(_swiglu_scaled(self.gate_proj(x), self.up_proj(x), gamma))


def fuse_qkvr(weights, config=None):
    """Stack compatible q/k/v/r tensors while preserving mixed-quant layers."""
    out = dict(weights)
    prefixes = {
        key[: -len("q_proj.weight")]
        for key in weights
        if key.endswith(".self_attn.q_proj.weight")
    }
    for prefix in prefixes:
        mtp = ".mtp.blocks." in prefix
        marker = ".mtp.blocks." if mtp else ".model.layers."
        try:
            layer_idx = int(prefix.split(marker, 1)[1].split(".", 1)[0])
        except (IndexError, ValueError):
            layer_idx = 0
        if config is not None and not qkvr_fusion_enabled(config, layer_idx, mtp=mtp):
            continue
        for leaf in ("weight", "scales", "biases"):
            parts = [out.pop(f"{prefix}{name}_proj.{leaf}", None) for name in "qkvr"]
            if all(value is not None for value in parts):
                out[f"{prefix}qkvr_proj.{leaf}"] = mx.concatenate(parts, axis=0)
            elif any(value is not None for value in parts):
                raise ValueError(f"partial q/k/v/r {leaf} set under {prefix}")
    return out


def shared_experts_to_dense(weights):
    """Reshape shared SwitchGLU tensors into the dense concatenated layout."""
    out = {}
    for key, value in weights.items():
        if ".shared_experts." in key and getattr(value, "ndim", 0) == 3:
            if ".down_proj." in key:
                value = value.transpose(1, 0, 2).reshape(value.shape[1], -1)
            else:
                value = value.reshape(-1, value.shape[2])
        out[key] = value
    return out


class InklingSparseMoE(nn.Module):
    """Sigmoid-gated fine-grained MoE: top-k routed experts (+ correction-bias selection)
    plus always-on shared experts, weighted by a logsigmoid/logsumexp softmax."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.intermediate_size = config.intermediate_size
        self.gate_weight = mx.zeros((self.n_routed + self.n_shared, config.hidden_size))
        self.e_score_correction_bias = mx.zeros((self.n_routed,))
        self.global_scale = mx.ones((1,))
        self.switch_mlp = InklingSwitchGLU(
            config.hidden_size, config.intermediate_size, self.n_routed
        )
        self.shared_experts = InklingSharedExpertsDense(
            config.hidden_size, config.intermediate_size, self.n_shared
        )
        self._wscale = None

    def _route(self, logits):
        """Select experts and compute routed/shared weights."""
        n_tokens = logits.shape[0]
        if self._wscale is None:
            self._wscale = mx.array(
                [self.route_scale], dtype=mx.float32
            ) * self.global_scale.astype(mx.float32)
        if mx.default_device() == mx.gpu:
            return _route_kernel(
                inputs=[logits, self.e_score_correction_bias, self._wscale],
                template=[
                    ("T", logits.dtype),
                    ("N", n_tokens),
                    ("R", self.n_routed),
                    ("SH", self.n_shared),
                    ("K", self.top_k),
                    ("I", self.intermediate_size),
                ],
                grid=(32, n_tokens, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[
                    (n_tokens, self.top_k),
                    (n_tokens, self.top_k),
                    (n_tokens, self.n_shared * self.intermediate_size),
                ],
                output_dtypes=[mx.uint32, logits.dtype, logits.dtype],
            )

        scores = mx.sigmoid(logits.astype(mx.float32))
        sfc = scores[:, : self.n_routed] + self.e_score_correction_bias
        idx = mx.argpartition(-sfc, self.top_k - 1, axis=-1)[:, : self.top_k]
        routed_logits = logits[:, : self.n_routed]
        shared_logits = logits[:, -self.n_shared :]
        tl = mx.concatenate(
            [mx.take_along_axis(routed_logits, idx, axis=-1), shared_logits], axis=-1
        ).astype(mx.float32)
        lp = -mx.logaddexp(mx.zeros_like(tl), -tl)
        w = (
            mx.exp(lp - mx.logsumexp(lp, axis=-1, keepdims=True))
            * self.route_scale
            * self.global_scale
        )
        topk_w = w[:, : self.top_k].astype(logits.dtype)
        gamma = mx.repeat(
            w[:, -self.n_shared :].astype(logits.dtype),
            self.intermediate_size,
            axis=-1,
        )
        return idx.astype(mx.uint32), topk_w, gamma

    def __call__(self, x):
        B, L, D = x.shape
        xf = x.reshape(-1, D)
        gate_weight = self.gate_weight
        if gate_weight.dtype != x.dtype:
            gate_weight = gate_weight.astype(x.dtype)
        logits = xf @ gate_weight.T
        idx, topk_w, gamma = self._route(logits)

        switch_mlp = self.switch_mlp
        down_proj = switch_mlp.down_proj
        if (
            _DOWN_COMBINE
            and self.top_k <= 8
            and xf.shape[0] <= 8
            and mx.default_device() == mx.gpu
            and getattr(down_proj, "bits", None) == 4
            and getattr(down_proj, "group_size", None) == 64
            and getattr(down_proj, "mode", "affine") == "affine"
            and getattr(down_proj, "biases", None) is not None
            and down_proj.input_dims == 2048
            and down_proj.scales.dtype == x.dtype
            and switch_mlp.scales_trivial()
        ):
            xe = mx.expand_dims(xf, (-2, -3))
            act = switch_mlp.activation(
                switch_mlp.up_proj(xe, idx), switch_mlp.gate_proj(xe, idx)
            )
            yr = _down_combine_kernel(
                inputs=[
                    act,
                    down_proj.weight,
                    down_proj.scales,
                    down_proj.biases,
                    idx,
                    topk_w,
                ],
                template=[
                    ("T", x.dtype),
                    ("OUT", down_proj.output_dims),
                    ("IN", down_proj.input_dims),
                    ("GROUPS", down_proj.input_dims // 64),
                    ("K", self.top_k),
                ],
                grid=(256, down_proj.output_dims, xf.shape[0]),
                threadgroup=(256, 1, 1),
                output_shapes=[(xf.shape[0], down_proj.output_dims)],
                output_dtypes=[x.dtype],
            )[0]
        else:
            yr = (switch_mlp(xf, idx) * topk_w[..., None]).sum(axis=-2)
        ys = self.shared_experts(xf, gamma)
        return (yr + ys).reshape(B, L, D).astype(x.dtype)


class InklingDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        *,
        qkvr_fused: Optional[bool] = None,
    ):
        super().__init__()
        self.self_attn = InklingAttention(config, layer_idx, qkvr_fused=qkvr_fused)
        self.mlp = (
            InklingDenseMLP(config)
            if config.layer_is_dense(layer_idx)
            else InklingSparseMoE(config)
        )
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=2
        )
        self.mlp_sconv = InklingShortConvolution(
            config.hidden_size, config.sconv_kernel_size, conv_idx=3
        )

    def __call__(self, x, cache=None, conv_mask=None):
        conv = cache[1] if cache is not None else None
        if conv_mask is None:
            conv_mask = _cache_padding_mask(conv, x.shape[1])
        r = self.self_attn(self.input_layernorm(x), cache=cache, conv_mask=conv_mask)
        h = self.attn_sconv(r, cache=conv, mask=conv_mask, residual=x)
        r = self.mlp(self.post_attention_layernorm(h))
        out = self.mlp_sconv(r, cache=conv, mask=conv_mask, residual=h)
        # OMLX: advance the shared conv cache once per consumed chunk so
        # lengths/left_padding track the remaining sequence across chunked
        # prefill (qwen3_5 pattern; no-op without padding bookkeeping).
        if conv is not None and hasattr(conv, "advance"):
            conv.advance(x.shape[1])
        return out


class InklingModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if config.use_embed_norm
            else None
        )
        self.layers = [
            InklingDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed(self, input_ids):
        h = self.embed_tokens(input_ids)
        if self.embed_norm is not None:
            h = self.embed_norm(h)
        return h

    def __call__(
        self,
        inputs,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
        skip_final_norm: bool = False,
    ):
        h = input_embeddings if input_embeddings is not None else self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # OMLX: derive the padding mask once per forward from the first
        # layer's conv cache (every layer shares the same bookkeeping and
        # advances its own copy equally). None when no padding is active,
        # which keeps the single-request path byte-identical to upstream.
        conv_mask = None
        first = cache[0] if cache else None
        if first is not None:
            conv0 = first[1]
            conv_mask = _cache_padding_mask(conv0, h.shape[1])
        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c, conv_mask=conv_mask)
        return h if skip_final_norm else self.norm(h)


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = InklingModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _logits_from_norm(self, h):
        h = h / self.config.logits_mup_width_multiplier
        if self.config.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(h)
        else:
            logits = self.lm_head(h)
        uv = self.config.unpadded_vocab_size
        if uv is not None and uv < logits.shape[-1]:
            logits = logits[..., :uv]
        return logits

    def __call__(
        self,
        inputs=None,
        cache=None,
        input_embeddings=None,
        inputs_embeds=None,
        return_hidden: bool = False,
        return_shared_kv: bool = False,
        skip_logits: bool = False,
        **kwargs,
    ):
        if inputs is None:
            inputs = kwargs.get("input_ids")
        if inputs_embeds is None:
            inputs_embeds = input_embeddings
        pre_norm = self.model(inputs, cache, inputs_embeds, skip_final_norm=True)
        logits = (
            None if skip_logits else self._logits_from_norm(self.model.norm(pre_norm))
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=[pre_norm] if return_hidden else None,
            shared_kv_states={} if return_shared_kv else None,
        )

    def speculative_logits_from_hidden(self, hidden: mx.array) -> mx.array:
        return self._logits_from_norm(self.model.norm(hidden))

    def speculative_argmax_from_hidden(self, hidden: mx.array) -> Optional[mx.array]:
        return mx.argmax(self.speculative_logits_from_hidden(hidden), axis=-1)

    def speculative_verify_hidden(self, inputs: mx.array, cache):
        snapshot = _snapshot_cache_state(cache)
        out = self(
            inputs,
            cache=cache,
            return_hidden=True,
            return_shared_kv=True,
            skip_logits=True,
        )
        return out.hidden_states[-1], out.shared_kv_states, (snapshot, inputs)

    def rollback_speculative_cache(
        self, caches, gdn_states, accepted, block_size
    ) -> int:
        if isinstance(accepted, mx.array):
            accepted = int(accepted.max().item()) if accepted.size else 0
        elif not isinstance(accepted, int):
            accepted = max(int(a) for a in accepted)
        snapshot, verify_inputs = gdn_states
        _restore_cache_state(caches, snapshot)
        keep = accepted + 1
        if keep > 0:
            self(verify_inputs[:, :keep], cache=caches, skip_logits=True)
        return accepted

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [CacheList(KVCache(), ArraysCache(4)) for _ in self.model.layers]
