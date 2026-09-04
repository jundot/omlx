import logging
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from ..cache import ArraysCache, CacheList, KVCache
from ..deepseek_v4.hyper_connection import HyperConnection, hc_expand
from mlx_lm.models.mla import MultiLinear
from omlx.patches.deepseek_v4.switch_layers import SwitchGLU
from omlx.patches.glm_moe_dsa.deepseek_v32 import (
    Model as DSV32Model,
    group_expert_select,
)
from omlx.patches.glm_moe_dsa.sparse_mla import (
    exact_block_token_attention,
    q8_vup_flat,
    sparse_mla_attention,
)
from .config import ModelConfig, TextConfig
from .gated_delta import gated_delta_update
from .linear import fused_quantized_matmul, linear_forward

logger = logging.getLogger(__name__)
_NATIVE_INDEXER_WARNED = False


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

    def _fused_in_proj(self, inputs):
        # q,k,v,f_a,g_a,b all take `inputs`; fuse into one matmul via a lossless
        # output-axis concat of the (quantized) weights, built once and cached.
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
                return tuple(linear_forward(m, inputs) for m in mods)
            if all(quantized):
                specs = {
                    (m.group_size, m.bits, getattr(m, "mode", "affine")) for m in mods
                }
                if len(specs) != 1:
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
        return mx.split(out, self._split_pts, axis=-1)

    def __call__(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, S, _ = inputs.shape
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

    def __call__(self, x, qr, mask, cache=None, kv_cache=None):
        B, S, _ = x.shape
        q = linear_forward(self.wq_b, qr).reshape(B, S, self.n_heads, self.head_dim)
        k = self.k_norm(linear_forward(self.wk, x)).reshape(B, S, self.head_dim)
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
            before_a = mx.zeros((B,), dtype=mx.int32)
            valid_cur = mx.ones((B, S), dtype=mx.bool_)
            usable = (S // self.index_kpool) * self.index_kpool
            pool_keys = self._compress_windows(k[:, :usable], gate_scores[:, :usable])
            pool_lengths = usable // self.index_kpool
            total_max = S

        # The pool has already advanced, even when sparse selection is unnecessary.
        if getattr(self, "bypass_short", True) and total_max <= self.index_topk:
            return None

        P = pool_keys.shape[1]
        select_k = min(self.index_topk // self.index_kpool, P)
        pool_idx = mx.arange(P)
        pool_end = (pool_idx + 1) * self.index_kpool - 1
        if isinstance(pool_lengths, list):
            pool_lengths_a = mx.array(pool_lengths, dtype=mx.int32)
        else:
            pool_lengths_a = mx.full((B,), pool_lengths, dtype=mx.int32)
        left_padding = getattr(kv_cache, "left_padding", None)
        if left_padding is None:
            left_padding = mx.zeros((B,), dtype=mx.int32)
        tail_on = self.index_kpool_always_select_tail and self.index_kpool > 1
        output_width = self.index_topk + (self.index_kpool - 1 if tail_on else 0)

        chunk = 512 if S > 512 else S
        out = []
        for c0 in range(0, S, chunk):
            c1 = min(c0 + chunk, S)
            cs = c1 - c0
            q_chunk = q[:, c0:c1]
            weights = linear_forward(self.weights_proj, x[:, c0:c1])
            weights = (weights * self.weight_scale).astype(q_chunk.dtype)
            index_scores = self._native_scores(q_chunk, pool_keys, weights)
            if index_scores is None:
                head_scores = q_chunk @ pool_keys[:, None].swapaxes(-1, -2)
                index_scores = mx.sum(
                    weights[..., None]
                    * mx.maximum(head_scores, mx.array(0, head_scores.dtype)),
                    axis=2,
                )
            query_pos = before_a[:, None] + mx.arange(c0, c1)[None]
            valid_candidates = (
                pool_idx[None, None] < pool_lengths_a[:, None, None]
            ) & (pool_end[None, None] <= query_pos[..., None])
            index_scores = mx.where(valid_candidates, index_scores, -1e30)
            selected = self._native_topk(index_scores, select_k)
            if selected is None:
                selected = mx.argpartition(-index_scores, kth=select_k - 1, axis=-1)[
                    ..., :select_k
                ]
            selected_valid = mx.take_along_axis(valid_candidates, selected, axis=-1)
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

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qr = self.q_a_layernorm(linear_forward(self.q_a_proj, x))
        q = linear_forward(self.q_b_proj, qr)
        q = q.reshape(B, L, self.num_heads, self.q_head_dim).transpose(0, 2, 1, 3)

        compressed_kv = linear_forward(self.kv_a_proj_with_mqa, x)
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
        )
        attn_mask = mask
        if topk_indices is not None:
            Kv = kv_latent.shape[2]
            valid_sel = topk_indices >= 0
            if L == 1:
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
                return self._gathered_attention(q, kv_latent, topk_indices)
            else:
                q_latent = self.embed_q(q)
                q_pe = mx.zeros(q_latent.shape[:-1] + (64,), dtype=q_latent.dtype)
                k_pe = mx.zeros(kv_latent.shape[:-1] + (64,), dtype=kv_latent.dtype)
                output = None
                if Kv >= 4096:
                    output = sparse_mla_attention(
                        q_latent,
                        q_pe,
                        kv_latent,
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
                    return linear_forward(self.o_proj, output_flat)

                k = self.embed_q(kv_latent, transpose=False)
                v = self.unembed_out(kv_latent)
                output = exact_block_token_attention(
                    q,
                    k,
                    v,
                    topk_indices,
                    self.scale,
                )
                if output is not None:
                    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                    return linear_forward(self.o_proj, output)

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

        if L == 1:
            q = self.embed_q(q)
            k = v = kv_latent
        else:
            k = self.embed_q(kv_latent, transpose=False)
            v = self.unembed_out(kv_latent)

        output = scaled_dot_product_attention(
            q, k, v, cache=cache, scale=self.scale, mask=attn_mask
        )
        if L == 1:
            output = self.unembed_out(output)

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return linear_forward(self.o_proj, output)

    def _gathered_attention(self, q, kv_latent, topk_indices):
        B, H, L, _ = q.shape
        Kv = kv_latent.shape[2]
        dim = kv_latent.shape[-1]
        selected = topk_indices[:, 0]
        topk = selected.shape[-1]
        clamped = mx.clip(selected, 0, Kv - 1)
        gathered = mx.take_along_axis(
            mx.broadcast_to(kv_latent[:, 0, None], (B, L, Kv, dim)),
            mx.broadcast_to(clamped[..., None], (B, L, topk, dim)),
            axis=2,
        )
        q_latent = self.embed_q(q).transpose(0, 2, 1, 3).reshape(B * L, H, 1, dim)
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
        return linear_forward(self.o_proj, output)


class Glm5NextClampedSwiGLU(nn.Module):
    def __init__(self, limit: Optional[float]):
        super().__init__()
        self.limit = limit

    def __call__(self, x_up: mx.array, x_gate: mx.array) -> mx.array:
        if self.limit is not None:
            x_gate = mx.clip(x_gate, a_min=None, a_max=self.limit)
            x_up = mx.clip(x_up, a_min=-self.limit, a_max=self.limit)
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
        gate = linear_forward(self.gate_proj, x)
        up = linear_forward(self.up_proj, x)
        if self.limit is not None:
            gate = mx.clip(gate, a_min=None, a_max=self.limit)
            up = mx.clip(up, a_min=-self.limit, a_max=self.limit)
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
        indices, scores = self.gate(x)
        y = self.switch_mlp(x, indices, scores=scores, weighted_sum=True)
        if y.ndim == x.ndim + 1:
            y = (y * scores[..., None]).sum(axis=-2).astype(x.dtype)
        if self.shared_experts is not None:
            y = y + self.shared_experts(x)
        return y


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

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x
        xc, post, comb = self.attn_hc(x)
        r = self.self_attn(self.input_layernorm(xc), mask, cache)
        x = hc_expand(r, residual, post, comb)
        # Compile the FFN block only for single-stream decode (B=1, S=1) -- the shape it
        # was validated on and where its win lives. Compiling the 288-expert MoE at a
        # batched or prefill shape spikes memory (it can OOM alongside the resident
        # weights), so those shapes take the eager path.
        if self.compile_ffn and x.shape[0] == 1 and x.shape[1] == 1:
            if self._ffn_c is None:
                self._ffn_c = mx.compile(self._ffn_block)
            return self._ffn_c(x)
        return self._ffn_block(x)

    def _ffn_block(self, x: mx.array) -> mx.array:
        # Stateless FFN half (no cache) -> compiles cleanly at a fixed decode shape.
        residual = x
        xc, post, comb = self.ffn_hc(x)
        m = self.mlp(self.post_attention_layernorm(xc))
        return hc_expand(m, residual, post, comb)


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

        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, mask=mask, cache=c)

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
        fg_parts = ("A_log", "dt_bias", "f_a_proj.weight", "f_b_proj.weight")
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
