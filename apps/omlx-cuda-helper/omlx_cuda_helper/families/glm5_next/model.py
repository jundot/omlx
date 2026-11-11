# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 prefill forward of one layer, for one sequence.

Every op keeps the dtype MLX type promotion gives it, so a checkpoint with
mixed tensor dtypes rounds where the host does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

from ...tensors import unpack
from . import kernels

_L2_EPS = 1e-6
_K_NORM_EPS = 1e-6
# Rows per call of the reference indexer scores and sparse attention.
_SCORE_ROWS = 512
_ATTEND_ROWS = 64
# Causal row blocks of the dense attention; a block reads only its own keys.
_DENSE_BLOCKS = 8


@dataclass(frozen=True)
class Config:
    hidden: int
    hc_mult: int
    hc_eps: float
    hc_iters: int
    norm_eps: float
    kda_heads: int
    kda_dim: int
    kda_kernel: int
    kda_lower_bound: float | None
    heads: int
    q_dim: int
    v_dim: int
    kv_rank: int
    index_heads: int
    index_dim: int
    index_topk: int
    index_kpool: int
    index_tail: bool
    top_k: int
    norm_topk_prob: bool
    routed_scale: float
    swiglu_limit: float | None

    @classmethod
    def from_dict(cls, text: dict[str, Any]) -> Config:
        linear = text.get("linear_attn_config") or {}
        lower = linear.get("gate_lower_bound", text.get("linear_lower_bound", -5.0))
        if lower is None and linear.get("safe_gate", True):
            lower = -5.0
        rope = text.get("qk_rope_head_dim", 0)
        if rope and not text.get("mla_use_nope", True):
            raise ValueError("GLM-5.3 sparse attention with rotary keys is not built")
        if text.get("n_group", 1) != 1:
            raise ValueError("GLM-5.3 expert groups are not built")
        return cls(
            hidden=text["hidden_size"],
            hc_mult=text.get("hc_mult", 4),
            hc_eps=text.get("hc_eps", 1e-6),
            hc_iters=text.get("hc_sinkhorn_iters", 20),
            norm_eps=text["rms_norm_eps"],
            kda_heads=linear.get("num_heads", text.get("linear_num_heads", 64)),
            kda_dim=linear.get("head_dim", text.get("linear_head_dim", 128)),
            kda_kernel=linear.get(
                "short_conv_kernel_size", text.get("linear_conv_kernel_dim", 4)
            ),
            kda_lower_bound=lower,
            heads=text["num_attention_heads"],
            q_dim=text["qk_nope_head_dim"],
            v_dim=text["v_head_dim"],
            kv_rank=text["kv_lora_rank"],
            index_heads=text["index_n_heads"],
            index_dim=text["index_head_dim"],
            index_topk=text["index_topk"],
            index_kpool=text.get("index_kpool", 4),
            index_tail=text.get("index_kpool_always_select_tail", True),
            top_k=text["num_experts_per_tok"],
            norm_topk_prob=text.get("norm_topk_prob", True),
            routed_scale=text["routed_scaling_factor"],
            swiglu_limit=text.get("swiglu_limit", 10.0),
        )


class Projection:
    """Bias-free linear layers ``x @ W.T`` that read one input.

    ``W`` is [out, in], or [count, out, in] with one weight per leading row of
    ``x`` [..., count, rows, in]. A packed weight is unpacked once, at load.
    """

    def __init__(self, weight, scales=None, biases=None, quant=None):
        self.floats = weight
        if scales is not None:
            self.floats = unpack(weight, scales, biases, quant)

    @classmethod
    def load(cls, weights: dict[str, mx.array], prefixes, quant) -> Projection:
        """The projections ``prefixes``, their outputs side by side."""
        if isinstance(prefixes, str):
            prefixes = [prefixes]
        parts = [
            [weights.get(f"{prefix}.{leaf}") for prefix in prefixes]
            for leaf in ("weight", "scales", "biases")
        ]
        packed = [scales is not None for scales in parts[1]]
        packing = {str(quant(prefix)) for prefix in prefixes} if packed[0] else {""}
        dtypes = {weight.dtype for weight in parts[0]}
        if len(dtypes) != 1 or any(packed) != all(packed) or len(packing) != 1:
            raise ValueError(f"{prefixes} are packed differently")
        joined = [
            None if part[0] is None else mx.concatenate(part, axis=0) for part in parts
        ]
        return cls(*joined, quant(prefixes[0]) if packed[0] else None)

    def parameters(self) -> list[mx.array]:
        return [self.floats]

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.floats.swapaxes(-1, -2)


class HyperConnection:
    """Mix the residual streams into a branch input and plan the write back."""

    def __init__(self, weights: dict[str, mx.array], prefix: str, config: Config):
        fn = weights[f"{prefix}.fn"].astype(mx.float32)
        self.fn_t = mx.contiguous(fn.T)
        self.base = weights[f"{prefix}.base"].astype(mx.float32)
        self.scale = weights[f"{prefix}.scale"].astype(mx.float32)
        self.streams = config.hc_mult
        self.iters = config.hc_iters
        self.eps = config.hc_eps
        self.norm_eps = config.norm_eps

    def parameters(self) -> list[mx.array]:
        return [self.fn_t, self.base, self.scale]

    def mixes(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """(pre, post, comb) of streams ``x`` [B, L, streams, D], in float32."""
        n = self.streams
        z = mx.fast.rms_norm(x.astype(mx.float32).flatten(-2), None, self.norm_eps)
        mixes = z @ self.fn_t
        pre = mx.sigmoid(mixes[..., :n] * self.scale[0] + self.base[:n]) + self.eps
        post = 2 * mx.sigmoid(
            mixes[..., n : 2 * n] * self.scale[1] + self.base[n : 2 * n]
        )
        comb = mixes[..., 2 * n :].reshape(*mixes.shape[:-1], n, n) * self.scale[2]
        comb = comb + self.base[2 * n :].reshape(n, n)
        comb = mx.softmax(comb, axis=-1, precise=True) + self.eps
        comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        for _ in range(max(self.iters - 1, 0)):
            comb = comb / (comb.sum(axis=-1, keepdims=True) + self.eps)
            comb = comb / (comb.sum(axis=-2, keepdims=True) + self.eps)
        return pre, post, comb

    def reference(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        pre, post, comb = self.mixes(x)
        y = x.astype(mx.float32)
        return (pre[..., None] * y).sum(axis=2).astype(x.dtype), post, comb

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        fused = kernels.hc_pre(self, x)
        return self.reference(x) if fused is None else fused


def expand_reference(branch, residual, post, comb) -> mx.array:
    y = post[..., None] * branch[:, :, None, :].astype(mx.float32)
    res = residual.astype(mx.float32)
    for t in range(res.shape[2]):
        y = y + comb[:, :, t, :, None] * res[:, :, t, None, :]
    return y.astype(branch.dtype)


def hc_expand(
    branch: mx.array, residual: mx.array, post: mx.array, comb: mx.array
) -> mx.array:
    """``post * branch + comb^T @ residual`` on the streams."""
    fused = kernels.hc_expand(branch, residual, post, comb)
    return expand_reference(branch, residual, post, comb) if fused is None else fused


class RMSNorm:
    def __init__(self, weights: dict[str, mx.array], prefix: str, eps: float):
        self.weight, self.eps = weights[f"{prefix}.weight"], eps

    def parameters(self) -> list[mx.array]:
        return [self.weight]

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _layer_norm(x: mx.array, weight: mx.array, bias: mx.array, eps: float):
    """LayerNorm whose scale and shift round once, as on Metal; MLX CUDA rounds
    the product and the sum separately."""
    dtype = mx.result_type(x, weight, bias)
    y = x.astype(mx.float32)
    y = y - y.mean(axis=-1, keepdims=True)
    y = y * mx.rsqrt((y * y).mean(axis=-1, keepdims=True) + eps)
    y = y.astype(dtype).astype(mx.float32)
    y = weight.astype(mx.float32) * y + bias.astype(mx.float32)
    return y.astype(dtype)


def _clamped_swiglu(up: mx.array, gate: mx.array, limit: float | None) -> mx.array:
    if limit is not None:
        gate = mx.clip(gate, a_min=None, a_max=limit)
        up = mx.clip(up, a_min=-limit, a_max=limit)
    return gate * mx.sigmoid(gate) * up


class Mlp:
    def __init__(self, weights: dict[str, mx.array], prefix: str, quant, limit):
        names = [f"{prefix}.gate_proj", f"{prefix}.up_proj"]
        self.gate_up = Projection.load(weights, names, quant)
        self.down = Projection.load(weights, f"{prefix}.down_proj", quant)
        self.limit = limit

    def parameters(self) -> list[mx.array]:
        return [*self.gate_up.parameters(), *self.down.parameters()]

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = mx.split(self.gate_up(x), 2, axis=-1)
        return self.down(_clamped_swiglu(up, gate, self.limit))


class ExpertBank:
    """One projection of every routed expert, [experts, out, in]."""

    def __init__(self, weight, scales=None, biases=None, quant=None):
        self.weight, self.scales, self.biases, self.quant = (
            weight,
            scales,
            biases,
            quant,
        )
        if scales is None:
            self.weight = mx.contiguous(weight.swapaxes(-1, -2))

    @classmethod
    def load(cls, weights: dict[str, mx.array], prefix: str, quant) -> ExpertBank:
        packed = (weights.get(f"{prefix}.{leaf}") for leaf in ("scales", "biases"))
        return cls(weights[f"{prefix}.weight"], *packed, quant(prefix))

    def parameters(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, x: mx.array, experts: mx.array) -> mx.array:
        """Rows ``x`` [routes, 1, in] through their expert; sorted by expert."""
        if self.scales is None:
            return mx.gather_mm(
                x, self.weight, rhs_indices=experts, sorted_indices=True
            )
        return mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=experts,
            transpose=True,
            group_size=self.quant["group_size"],
            bits=self.quant["bits"],
            mode=self.quant.get("mode", "affine"),
            sorted_indices=True,
        )


def routed(banks, x, experts, weights, limit) -> mx.array:
    """Each token's experts through the MLX gather, summed in float32."""
    gate, up, down = banks
    batch, length, k = experts.shape
    flat = experts.flatten()
    order = mx.argsort(flat)
    rows = x.reshape(batch * length, 1, -1)[order // k]
    chosen = flat[order]
    y = down(_clamped_swiglu(up(rows, chosen), gate(rows, chosen), limit), chosen)
    y = y[mx.argsort(order)].reshape(batch, length, k, -1)
    return (y * weights[..., None]).sum(axis=-2).astype(x.dtype)


class Moe:
    def __init__(
        self, weights: dict[str, mx.array], prefix: str, quant, config: Config
    ):
        self.router_t = mx.contiguous(
            weights[f"{prefix}.gate.weight"].astype(mx.float32).T
        )
        self.bias = weights[f"{prefix}.gate.e_score_correction_bias"].astype(mx.float32)
        self.gate, self.up, self.down = (
            ExpertBank.load(weights, f"{prefix}.switch_mlp.{name}", quant)
            for name in ("gate_proj", "up_proj", "down_proj")
        )
        self.shared = None
        if f"{prefix}.shared_experts.gate_proj.weight" in weights:
            shared = f"{prefix}.shared_experts"
            self.shared = Mlp(weights, shared, quant, config.swiglu_limit)
        self.top_k = config.top_k
        self.norm = config.norm_topk_prob
        self.scale = config.routed_scale
        self.limit = config.swiglu_limit

    def parameters(self) -> list[mx.array]:
        out = [self.router_t, self.bias]
        for bank in (self.gate, self.up, self.down):
            out += bank.parameters()
        return out + (self.shared.parameters() if self.shared else [])

    def route(self, x: mx.array) -> tuple[mx.array, mx.array]:
        """(experts, weights) [B, L, top_k] of each token."""
        logits = x.astype(mx.float32) @ self.router_t
        scores = mx.sigmoid(logits)
        k = self.top_k
        experts = mx.argpartition(-(scores + self.bias), kth=k - 1, axis=-1)[..., :k]
        weights = mx.take_along_axis(scores, experts, axis=-1)
        if k > 1 and self.norm:
            weights = weights / weights.sum(axis=-1, keepdims=True)
        return experts, weights * self.scale

    def __call__(self, x: mx.array) -> mx.array:
        experts, weights = self.route(x)
        y = kernels.moe_routed(self, x, experts, weights)
        if y is None:
            banks = (self.gate, self.up, self.down)
            y = routed(banks, x, experts, weights, self.limit)
        if self.shared is not None:
            y = y + self.shared(x)
        return y


def _l2norm(x: mx.array) -> mx.array:
    return x * mx.rsqrt((x * x).sum(axis=-1, keepdims=True) + _L2_EPS)


@dataclass
class KdaState:
    conv: mx.array | None = None
    rec: mx.array | None = None


class Kda:
    """Linear attention: short conv, gated delta rule, gated norm."""

    def __init__(
        self, weights: dict[str, mx.array], prefix: str, quant, config: Config
    ):
        self.heads, self.dim = config.kda_heads, config.kda_dim
        self.kernel = config.kda_kernel
        self.lower_bound = config.kda_lower_bound
        gate = f"{prefix}.forget_gate"
        # q, k, v, the two gate bottlenecks and the write strength.
        names = ("q_proj", "k_proj", "v_proj", "forget_gate.f_a_proj", "g_a_proj")
        names = [f"{prefix}.{name}" for name in (*names, "b_proj")]
        self.in_proj = Projection.load(weights, names, quant)
        self.gate_dim = weights[f"{names[3]}.weight"].shape[0]
        self.conv = weights[f"{prefix}.conv1d.weight"]
        self.f_b = Projection.load(weights, f"{gate}.f_b_proj", quant)
        self.g_b = Projection.load(weights, f"{prefix}.g_b_proj", quant)
        self.o_proj = Projection.load(weights, f"{prefix}.o_proj", quant)
        self.a_log = weights[f"{gate}.A_log"].astype(mx.float32)
        self.dt_bias = weights[f"{gate}.dt_bias"].astype(mx.float32)
        self.o_norm = weights[f"{prefix}.o_norm.weight"]
        self.norm_eps = config.norm_eps

    def parameters(self) -> list[mx.array]:
        linears = (self.in_proj, self.f_b, self.g_b, self.o_proj)
        return [
            *[a for linear in linears for a in linear.parameters()],
            self.conv,
            self.a_log,
            self.dt_bias,
            self.o_norm,
        ]

    def new_state(self) -> KdaState:
        return KdaState()

    def gate_decay(self, a: mx.array) -> mx.array:
        """Per-channel forget factor [B, S, heads, dim] in float32."""
        fused = kernels.kda_decay(self, a)
        return self.decay_reference(a) if fused is None else fused

    def decay_reference(self, a: mx.array) -> mx.array:
        rate = mx.exp(self.a_log).reshape(self.heads, 1)
        a = a + self.dt_bias.reshape(self.heads, self.dim)
        if self.lower_bound is not None:
            return mx.exp(self.lower_bound * mx.sigmoid(rate * a))
        return mx.exp(-rate * mx.logaddexp(a, 0))

    def prework(self, proj: mx.array, conv: mx.array):
        """(q, k, v, next conv state): short conv, SiLU, key normalization."""
        batch, length, _ = proj.shape
        heads, dim = self.heads, self.dim
        qkv = 3 * heads * dim
        window = mx.concatenate([conv, proj[..., :qkv]], axis=1)
        mixed = mx.conv1d(window, self.conv, groups=qkv)
        mixed = mixed * mx.sigmoid(mixed)
        q, k, v = (
            part.reshape(batch, length, heads, dim)
            for part in mx.split(mixed, 3, axis=-1)
        )
        dtype = q.dtype
        q = (_l2norm(q.astype(mx.float32)) * dim**-0.5).astype(dtype)
        k = _l2norm(k.astype(mx.float32)).astype(dtype)
        return q, k, v, mx.contiguous(window[:, -(self.kernel - 1) :, :])

    def norm_gate(self, out: mx.array, gate: mx.array) -> mx.array:
        """RMS norm over each head, its gain and the output gate, in float32."""
        y = out.astype(mx.float32)
        y = y * mx.rsqrt((y * y).mean(axis=-1, keepdims=True) + self.norm_eps)
        y = self.o_norm.astype(mx.float32) * y
        return (y * mx.sigmoid(gate.astype(mx.float32))).astype(out.dtype)

    def __call__(self, x: mx.array, state: KdaState) -> mx.array:
        batch, length, _ = x.shape
        heads, dim = self.heads, self.dim
        qkv = 3 * heads * dim
        proj = self.in_proj(x)
        fa, ga, b = mx.split(
            proj[..., qkv:], [self.gate_dim, 2 * self.gate_dim], axis=-1
        )
        a = self.f_b(fa).reshape(batch, length, heads, dim)
        gate = self.g_b(ga).reshape(batch, length, heads, dim)
        conv = state.conv
        if conv is None:
            conv = mx.zeros((batch, self.kernel - 1, qkv), dtype=x.dtype)
        if state.rec is None:
            state.rec = mx.zeros((batch, heads, dim, dim), dtype=mx.float32)
        fused = kernels.kda_prework(self, proj, conv)
        q, k, v, state.conv = self.prework(proj, conv) if fused is None else fused
        out, state.rec = mx.fast.gated_delta_update(
            q, k, v, self.gate_decay(a), mx.sigmoid(b), initial_state=state.rec
        )
        fused = kernels.kda_norm_gate(self, out, gate)
        if fused is None:
            fused = self.norm_gate(out, gate)
        return self.o_proj(fused.reshape(batch, length, -1))

    def state_arrays(self, layer: int, state: KdaState) -> dict[str, mx.array]:
        return {f"{layer}.conv": state.conv, f"{layer}.rec": state.rec}

    def load_state(self, state: KdaState, slots: dict[str, mx.array]) -> None:
        state.conv, state.rec = slots["conv"], slots["rec"]


@dataclass
class SparseState:
    # Latent keys of every position so far, [B, 1, positions, rank].
    kv: mx.array | None = None
    # Pooled index keys, one per complete window, [B, windows, index_dim].
    pool: mx.array | None = None
    # Index keys and pool gate scores of the positions after the last window.
    tail_kv: mx.array | None = None
    tail_gate: mx.array | None = None
    sent: tuple[int, int] = field(default=(0, 0))

    @property
    def positions(self) -> int:
        return 0 if self.kv is None else self.kv.shape[2]

    @property
    def windows(self) -> int:
        return 0 if self.pool is None else self.pool.shape[1]


class SparseAttention:
    """Latent attention over the keys an indexer selects for each query."""

    def __init__(
        self, weights: dict[str, mx.array], prefix: str, quant, config: Config
    ):
        self.config = config

        def load(name: str) -> Projection:
            return Projection.load(weights, f"{prefix}.{name}", quant)

        self.q_a, self.q_b = load("q_a_proj"), load("q_b_proj")
        self.kv_a = load("kv_a_proj_with_mqa")
        self.q_norm = RMSNorm(weights, f"{prefix}.q_a_layernorm", config.norm_eps)
        self.kv_norm = RMSNorm(weights, f"{prefix}.kv_a_layernorm", config.norm_eps)
        # Per-head maps between the latent keys and a head's query and value
        # spaces: embed [heads, rank, q_dim], unembed [heads, v_dim, rank].
        self.embed, self.unembed = load("embed_q"), load("unembed_out")
        self.o_proj = load("o_proj")
        self.wq_b, self.wk = load("indexer.wq_b"), load("indexer.wk")
        self.weights_proj = load("indexer.weights_proj")
        index = f"{prefix}.indexer"
        self.k_norm = (
            weights[f"{index}.k_norm.weight"],
            weights[f"{index}.k_norm.bias"],
        )
        self.ape = weights[f"{index}.index_kpool_compress_ape"]
        self.pool_gate = mx.contiguous(weights[f"{index}.index_kpool_compress_gate"].T)
        self.scale = config.q_dim**-0.5
        self.weight_scale = config.index_heads**-0.5 * config.index_dim**-0.5

    def parameters(self) -> list[mx.array]:
        linears = (self.q_a, self.q_b, self.kv_a, self.embed, self.unembed)
        linears += (self.o_proj, self.wq_b, self.wk, self.weights_proj)
        return [
            *[a for linear in linears for a in linear.parameters()],
            *self.q_norm.parameters(),
            *self.kv_norm.parameters(),
            *self.k_norm,
            self.ape,
            self.pool_gate,
        ]

    def new_state(self) -> SparseState:
        return SparseState()

    def _pool(self, keys: mx.array, gate: mx.array) -> mx.array:
        """Gate-weighted mean of each window of ``index_kpool`` index keys."""
        batch, rows, dim = keys.shape
        kp = self.config.index_kpool
        keys = keys.reshape(batch, rows // kp, kp, dim)
        logits = gate.reshape(batch, rows // kp, kp, dim) + self.ape[None, None]
        return mx.sum(mx.softmax(logits, axis=2) * keys, axis=2)

    def _advance_pool(self, state: SparseState, x: mx.array) -> None:
        keys = _layer_norm(self.wk(x), *self.k_norm, _K_NORM_EPS)
        gate = x @ self.pool_gate
        if state.tail_kv is not None:
            keys = mx.concatenate([state.tail_kv, keys], axis=1)
            gate = mx.concatenate([state.tail_gate, gate], axis=1)
        kp = self.config.index_kpool
        usable = keys.shape[1] // kp * kp
        if usable:
            pooled = self._pool(keys[:, :usable], gate[:, :usable])
            state.pool = (
                pooled
                if state.pool is None
                else mx.concatenate([state.pool, pooled], axis=1)
            )
        if usable < keys.shape[1]:
            state.tail_kv = mx.contiguous(keys[:, usable:])
            state.tail_gate = mx.contiguous(gate[:, usable:])
        else:
            state.tail_kv = state.tail_gate = None

    def _dense_rows(self, length: int, past: int) -> int:
        """Leading rows whose selection is the whole causal prefix: up to
        ``index_topk // kpool`` complete windows the indexer keeps them all."""
        config = self.config
        kp = config.index_kpool
        if past + length <= config.index_topk:
            return length
        if kp > 1 and not config.index_tail:
            return 0
        boundary = (config.index_topk // kp + 1) * kp - 1
        return min(length, max(0, boundary - past))

    def window_scores(self, q, pool, weights, first: int) -> mx.array:
        """Reference scores [rows, windows] of the queries from ``first`` on."""
        position = first + mx.arange(q.shape[0])
        kp = self.config.index_kpool
        return mx.concatenate(
            [
                _index_scores(
                    q[a : a + _SCORE_ROWS],
                    pool,
                    weights[a : a + _SCORE_ROWS],
                    position[a : a + _SCORE_ROWS],
                    kp,
                )
                for a in range(0, q.shape[0], _SCORE_ROWS)
            ],
            axis=0,
        )

    @staticmethod
    def top(scores: mx.array, count: int) -> mx.array:
        """Reference selection of the ``count`` best windows of each row."""
        return mx.concatenate(
            [
                top_windows(scores[a : a + _SCORE_ROWS], count)
                for a in range(0, scores.shape[0], _SCORE_ROWS)
            ],
            axis=0,
        )

    def _select(
        self, x: mx.array, qr: mx.array, state: SparseState, start: int, past: int
    ) -> mx.array:
        """Key positions [rows, width] of chunk rows from ``start``; -1 pads."""
        config = self.config
        kp, heads, dim = config.index_kpool, config.index_heads, config.index_dim
        rows = x.shape[1] - start
        pool = state.pool[0]
        count = min(config.index_topk // kp, pool.shape[0])
        q = self.wq_b(qr[0, start:]).reshape(rows, heads, dim)
        weights = (self.weights_proj(x[0, start:]) * self.weight_scale).astype(q.dtype)
        position = past + start + mx.arange(rows)
        picked = kernels.select_windows(q, pool, weights, past + start, kp, count)
        if picked is None:
            scores = self.window_scores(q, pool, weights, past + start)
            picked = self.top(scores, count)
        valid = (picked + 1) * kp - 1 <= position[:, None]
        tokens = picked[..., None] * kp + mx.arange(kp)
        tokens = mx.where(valid[..., None], tokens, -1).reshape(rows, count * kp)
        if config.index_tail and kp > 1:
            tail_count = (position + 1) % kp
            tail = (position + 1 - tail_count)[:, None] + mx.arange(kp - 1)
            tail = mx.where(mx.arange(kp - 1) < tail_count[:, None], tail, -1)
            tokens = mx.concatenate([tokens, tail], axis=-1)
        return tokens.astype(mx.int32)

    def dense_keys(self, kv: mx.array, dtype) -> tuple[mx.array, mx.array]:
        """Keys and values of every head, expanded from latent keys ``kv``."""
        keys = kv @ self.embed.floats
        values = kv @ self.unembed.floats.swapaxes(-1, -2)
        if keys.dtype != dtype:
            keys, values = keys.astype(dtype), values.astype(dtype)
        return keys, values

    def dense_attend(self, q, keys, values, rows: int, past: int) -> mx.array:
        """Causal attention of the first ``rows`` rows, [B, rows, heads * v].

        Scores, weights and outputs each round to the activation dtype.
        """
        q = q[:, :, :rows] * mx.array(self.scale, dtype=q.dtype)
        blocks = max(1, min(_DENSE_BLOCKS, rows // 256))
        bounds = [rows * i // blocks // 64 * 64 for i in range(1, blocks)]
        outs = []
        for a, b in zip([0, *bounds], [*bounds, rows]):
            end = past + b
            scores = q[:, :, a:b] @ keys[:, :, :end].swapaxes(-1, -2)
            seen = mx.arange(past + a, end)[:, None] >= mx.arange(end)[None]
            scores = mx.where(seen, scores, mx.finfo(scores.dtype).min)
            weights = mx.softmax(scores, axis=-1, precise=True)
            outs.append(weights @ values[:, :, :end])
        out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
        return out.transpose(0, 2, 1, 3).reshape(q.shape[0], rows, -1)

    def _dense(self, q: mx.array, kv: mx.array, rows: int, past: int) -> mx.array:
        keys, values = self.dense_keys(kv[:, :, : past + rows], q.dtype)
        return self.dense_attend(q, keys, values, rows, past)

    def attend_reference(self, q_latent, kv, tokens) -> mx.array:
        """Latent attention of each row over its ``tokens``, by MLX ops."""
        return mx.concatenate(
            [
                _attend_selected(
                    q_latent[0, :, a : a + _ATTEND_ROWS],
                    kv[0, 0],
                    tokens[a : a + _ATTEND_ROWS],
                    self.scale,
                )
                for a in range(0, tokens.shape[0], _ATTEND_ROWS)
            ],
            axis=1,
        )[None]

    def _sparse(self, q: mx.array, kv: mx.array, tokens: mx.array) -> mx.array:
        """Attention of rows over their selected keys, [B, rows, heads * v]."""
        q_latent = self.embed(q)
        latent = kernels.sparse_attention(q_latent, kv, tokens, self.scale)
        if latent is None:
            latent = self.attend_reference(q_latent, kv, tokens)
        out = self.unembed(latent)
        return out.transpose(0, 2, 1, 3).reshape(q.shape[0], tokens.shape[0], -1)

    def __call__(self, x: mx.array, state: SparseState) -> mx.array:
        config = self.config
        batch, length, _ = x.shape
        qr = self.q_norm(self.q_a(x))
        q = self.q_b(qr).reshape(batch, length, config.heads, config.q_dim)
        q = q.transpose(0, 2, 1, 3)
        latent = self.kv_norm(self.kv_a(x))[:, None]
        past = state.positions
        kv = latent if state.kv is None else mx.concatenate([state.kv, latent], axis=2)
        state.kv = kv
        self._advance_pool(state, x)

        dense = self._dense_rows(length, past)
        parts = []
        if dense:
            parts.append(self._dense(q, kv, dense, past))
        if dense < length:
            tokens = self._select(x, qr, state, dense, past)
            parts.append(self._sparse(q[:, :, dense:], kv, tokens))
        flat = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=1)
        return self.o_proj(flat)

    def state_arrays(self, layer: int, state: SparseState) -> dict[str, mx.array]:
        """Latent and pooled rows added since the last call, and the tail."""
        kv_from, pool_from = state.sent
        out = {f"{layer}.kv": state.kv[:, :, kv_from:]}
        if state.pool is not None:
            out[f"{layer}.pool"] = state.pool[:, pool_from:]
        if state.tail_kv is not None:
            out[f"{layer}.tail_kv"] = state.tail_kv
            out[f"{layer}.tail_gate"] = state.tail_gate
        state.sent = (state.positions, state.windows)
        return out

    def load_state(self, state: SparseState, slots: dict[str, mx.array]) -> None:
        state.kv = slots.get("kv")
        pool = slots.get("pool")
        state.pool = pool if pool is not None and pool.shape[1] else None
        state.tail_kv, state.tail_gate = slots.get("tail_kv"), slots.get("tail_gate")
        state.sent = (state.positions, state.windows)


def _index_scores(
    q: mx.array, pool: mx.array, weights: mx.array, position: mx.array, kpool: int
) -> mx.array:
    """Scores [rows, windows] of pooled keys for queries ``q`` [rows, heads, dim].

    The head sum runs in float32 and rounds once to the query dtype. A window
    that is not complete at the query position scores lowest.
    """
    head = q.astype(mx.float32) @ pool.astype(mx.float32).T
    scores = (mx.maximum(head, 0) * weights.astype(mx.float32)[..., None]).sum(axis=1)
    scores = scores.astype(q.dtype)
    end = (mx.arange(pool.shape[0]) + 1) * kpool - 1
    return mx.where(end[None] <= position[:, None], scores, -1e30)


def top_windows(scores: mx.array, count: int) -> mx.array:
    """The ``count`` best windows of each row, ascending.

    Windows tied with the last one kept are taken lowest index first.
    """
    cut = mx.sort(scores, axis=-1)[:, -count][:, None]
    above = scores > cut
    tied = scores == cut
    room = count - above.sum(axis=-1, keepdims=True)
    keep = above | (tied & (mx.cumsum(tied.astype(mx.int32), axis=-1) <= room))
    slot = mx.where(keep, mx.cumsum(keep.astype(mx.int32), axis=-1) - 1, count)
    windows = mx.broadcast_to(mx.arange(scores.shape[-1])[None], scores.shape)
    out = mx.zeros((scores.shape[0], count + 1), dtype=mx.int32)
    return mx.put_along_axis(out, slot, windows.astype(mx.int32), axis=-1)[:, :count]


def _attend_selected(
    q: mx.array, kv: mx.array, tokens: mx.array, scale: float
) -> mx.array:
    """Latent attention in float32, rounded once: [heads, rows, rank]."""
    picked = kv[mx.maximum(tokens, 0)].astype(mx.float32)
    queries = q.astype(mx.float32).transpose(1, 0, 2)
    scores = (queries @ picked.swapaxes(-1, -2)) * scale
    scores = mx.where((tokens >= 0)[:, None, :], scores, -mx.inf)
    out = mx.softmax(scores, axis=-1, precise=True) @ picked
    return out.astype(q.dtype).transpose(1, 0, 2)


class Layer:
    def __init__(self, weights: dict[str, mx.array], quant, config: Config):
        """``quant(module)`` gives the packing of a packed module's weight."""
        self.linear = "self_attn.conv1d.weight" in weights
        attention = Kda if self.linear else SparseAttention
        self.attn = attention(weights, "self_attn", quant, config)
        if "mlp.gate.weight" in weights:
            self.mlp = Moe(weights, "mlp", quant, config)
        else:
            self.mlp = Mlp(weights, "mlp", quant, config.swiglu_limit)
        self.attn_hc = HyperConnection(weights, "attn_hc", config)
        self.ffn_hc = HyperConnection(weights, "ffn_hc", config)
        self.attn_norm = RMSNorm(weights, "input_layernorm", config.norm_eps)
        self.ffn_norm = RMSNorm(weights, "post_attention_layernorm", config.norm_eps)

    def parameters(self) -> list[mx.array]:
        return [
            *self.attn.parameters(),
            *self.mlp.parameters(),
            *self.attn_hc.parameters(),
            *self.ffn_hc.parameters(),
            *self.attn_norm.parameters(),
            *self.ffn_norm.parameters(),
        ]

    def __call__(self, h: mx.array, state: Any) -> mx.array:
        x, post, comb = self.attn_hc(h)
        h = hc_expand(self.attn(self.attn_norm(x), state), h, post, comb)
        x, post, comb = self.ffn_hc(h)
        return hc_expand(self.mlp(self.ffn_norm(x)), h, post, comb)
