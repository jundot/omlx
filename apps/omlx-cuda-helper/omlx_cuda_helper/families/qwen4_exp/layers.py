# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Qwen3.8 Flash-Next blocks, batch one and text only.

Each block follows the oMLX implementation op by op and in the same dtypes,
so 16-bit activations round as they do on the host. The gated delta net and
the MoE block are ports of the mlx-vlm classes that implementation builds on
(MIT, Copyright (c) 2025 Prince Canuma).
"""

from __future__ import annotations

import functools
import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...tensors import from_host, unpack
from ...wire import HostArray
from . import kernels
from .attention import Attention, QSACache
from .ops import (
    Weights,
    depthwise_conv,
    row_sum,
    stacked_sum,
    stream_norm,
    unit_scale,
)


def checked_conv(taps: mx.array, dilation: int, dtype: mx.Dtype):
    """The conv kernel if it equals ``depthwise_conv`` on these taps, else None."""
    x = mx.random.normal((1, 64 + (taps.shape[1] - 1) * dilation, taps.shape[0]))
    x = x.astype(dtype)
    got = kernels.depthwise_conv(x, taps, dilation)
    if mx.array_equal(got, depthwise_conv(x, taps, dilation)).item():
        return kernels.depthwise_conv
    kernels.logger.warning("Depthwise conv kernel differs; using the reference")
    return None


def checked_square_sums(scale: mx.array, eps: float, dtype: mx.Dtype):
    """The stream sum-of-squares kernel if ``stream_norm`` agrees with it."""
    x = mx.random.normal((1, 64, scale.size)).astype(dtype)
    got = stream_norm(x, scale, eps, kernels.stream_square_sums)
    if kernels.close(got, stream_norm(x, scale, eps), "Stream norm"):
        return kernels.stream_square_sums
    return None


@functools.cache
def _stream_mixer(streams: int, hidden: int):
    """Compiled ``(up, normed) -> block input`` for one residual geometry."""

    def mix(up: mx.array, normed: mx.array) -> mx.array:
        # The host kernel takes this sigmoid in float32 and rounds it once.
        gate = mx.sigmoid(up.astype(mx.float32)).astype(normed.dtype)
        # The streams add up one by one in the residual dtype.
        mixed = gate[..., :hidden] * normed[..., :hidden]
        for stream in range(1, streams):
            lo = stream * hidden
            mixed = mixed + gate[..., lo : lo + hidden] * normed[..., lo : lo + hidden]
        return mixed * (1.0 / streams)

    return mx.compile(mix)


class HyperConnection:
    """Reads one block input from the residual streams and gates its write."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.streams, self.hidden = text.get("hc_count", 4), text["hidden_size"]
        self.eps = text["rms_norm_eps"]
        self.scale = unit_scale(w["hc_norm.weight"]).reshape(self.streams, self.hidden)
        self.down = w.linear("input_mix_weight_down")
        self.up = w.linear("input_mix_weight_up")
        self.inject = w.float32("block_inject_weight")
        self.mix = _stream_mixer(self.streams, self.hidden)
        self.square_sums = None

    def parameters(self) -> list[mx.array]:
        linears = (self.down, self.up)
        own = [self.scale, self.inject]
        return own + [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        dtype = self.down.weight.dtype
        self.square_sums = checked_square_sums(self.scale, self.eps, dtype)

    def __call__(self, h: mx.array) -> tuple[mx.array, mx.array]:
        """(block input [1, L, hidden], write gates [1, L, streams])."""
        normed = stream_norm(h, self.scale, self.eps, self.square_sums)
        mix = nn.silu(self.down(normed) / self.streams)
        return self.mix(self.up(mix), normed), self.write_gates(normed)

    def write_gates(self, normed: mx.array) -> mx.array:
        """Per-stream gates [1, L, streams] of the block's residual write.

        They read the inject weights without their 16-bit rounding and take
        their sigmoid in float32, as the host kernel does.
        """
        raw = (normed.astype(mx.float32) @ self.inject.T).astype(normed.dtype)
        raw = (raw / self.streams).astype(mx.float32)
        return 2 * mx.sigmoid(raw).astype(normed.dtype)


def write(h: mx.array, branch: mx.array, inject: mx.array) -> mx.array:
    """Add a block output to every residual stream, scaled by its gate."""
    return h + (branch[..., None, :] * inject[..., None]).reshape(h.shape)


def _norm_gate(activation):
    """Compiled output norm and gate of the recurrence result ``y``."""

    def gate(y: mx.array, inverse: mx.array, weight: mx.array, z: mx.array):
        # The norm rounds before its weight; the gate product is float32.
        y = (y.astype(mx.float32) * inverse).astype(weight.dtype) * weight
        return (y.astype(mx.float32) * activation(z.astype(mx.float32))).astype(
            weight.dtype
        )

    return mx.compile(gate)


_sigmoid_gate, _silu_gate = _norm_gate(mx.sigmoid), _norm_gate(nn.silu)


class GatedDeltaNet:
    """Linear attention: a causal conv and the gated delta recurrence."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.key_heads = text["linear_num_key_heads"]
        self.value_heads = text["linear_num_value_heads"]
        self.key_dim = text["linear_key_head_dim"]
        self.value_dim = text["linear_value_head_dim"]
        self.eps = text["rms_norm_eps"]
        self.sigmoid_gate = (
            text.get("output_gate_type") or text.get("hidden_act") or "sigmoid"
        ) == "sigmoid"
        self.qkv = w.linear("in_proj_qkv")
        self.z = w.linear("in_proj_z")
        self.b = w.linear("in_proj_b")
        self.a = w.linear("in_proj_a")
        self.out = w.linear("out_proj")
        self.taps = w["conv1d.weight"][..., 0].astype(mx.float32)
        self.conv = depthwise_conv
        self.decay = mx.exp(w["A_log"].astype(mx.float32))
        self.dt_bias = w["dt_bias"]
        self.norm_weight = w["norm.weight"]

    def parameters(self) -> list[mx.array]:
        linears = (self.qkv, self.z, self.b, self.a, self.out)
        own = [self.taps, self.decay, self.dt_bias, self.norm_weight]
        return own + [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        self.conv = checked_conv(self.taps, 1, self.qkv.weight.dtype) or self.conv

    def prework(self, qkv: mx.array, state: mx.array | None) -> tuple[mx.array, ...]:
        """(q, k, v, next conv window) from the projected rows [1, rows, C]."""
        rows = qkv.shape[1]
        if state is None:
            window = self.taps.shape[1] - 1
            state = mx.zeros((1, window, qkv.shape[-1]), dtype=qkv.dtype)
        conv_in = mx.concatenate([state, qkv], axis=1)
        conv_out = nn.silu(self.conv(conv_in, self.taps, 1))
        keys = self.key_heads * self.key_dim
        q = conv_out[..., :keys].reshape(1, rows, self.key_heads, self.key_dim)
        k = conv_out[..., keys : 2 * keys].reshape(q.shape)
        v = conv_out[..., 2 * keys :]
        v = v.reshape(1, rows, self.value_heads, self.value_dim)
        # L2 norm with the epsilon after the sum; the query takes 1/sqrt(dim).
        q = q * mx.rsqrt(row_sum(mx.square(q)) + 1e-6)
        k = k * mx.rsqrt(row_sum(mx.square(k)) + 1e-6)
        return q * self.key_dim**-0.5, k, v, conv_in[:, rows:]

    def gates(self, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
        """(decay in float32, update rate) of each step and value head."""
        return mx.exp(-self.decay * nn.softplus(a + self.dt_bias)), mx.sigmoid(b)

    def finish(self, y: mx.array, z: mx.array) -> mx.array:
        """The recurrence output normalized and gated, [1, rows, value width]."""
        squares = mx.square(y.astype(mx.float32))
        inverse = mx.rsqrt(mx.mean(squares, axis=-1, keepdims=True) + self.eps)
        gate = _sigmoid_gate if self.sigmoid_gate else _silu_gate
        gated = gate(y, inverse, self.norm_weight, z.reshape(y.shape))
        return gated.reshape(1, y.shape[1], -1)

    def __call__(self, x: mx.array, cache: list) -> mx.array:
        q, k, v, cache[0] = self.prework(self.qkv(x), cache[0])
        gamma, beta = self.gates(self.a(x), self.b(x))
        y, cache[1] = mx.fast.gated_delta_update(q, k, v, gamma, beta, cache[1])
        return self.out(self.finish(y, self.z(x)))


def ple_rows(inputs: dict[str, HostArray], name: str, rows: int) -> mx.array:
    """The chunk's n-gram embeddings [1, rows, dim] from the rows the host read."""
    ids = inputs.get(name + "ids")
    if ids is None or ids[0].shape[1] != rows:
        raise RuntimeError(f"No PLE rows for this chunk ({name}ids)")
    data = [from_host(inputs[f"{name}{n}"]) for n in range(3) if f"{name}{n}" in inputs]
    scale = from_host(inputs[name + "ws"])
    bits, group_size, cast, fp8 = (int(v) for v in inputs[name + "q"][0])
    values = data[0]
    if fp8:
        values = mx.from_fp8(values, dtype=mx.bfloat16)
    if bits:
        values = unpack(
            values, data[1], data[2], {"group_size": group_size, "bits": bits}
        )
    if cast:
        values = values.astype(scale.dtype)
    return (values * scale).reshape(1, rows, -1)


class PLE:
    """N-gram embedding block; the host owns the table and sends the rows."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.streams, self.hidden = text.get("hc_count", 4), text["hidden_size"]
        self.eps = text["rms_norm_eps"]
        shape = (self.streams, self.hidden)
        self.key = w.linear("key_proj")
        self.value = w.linear("value_proj")
        self.key_scale = unit_scale(w["norm_key.weight"]).reshape(shape)
        self.query_scale = unit_scale(w["norm_query.weight"]).reshape(shape)
        self.conv_scale = unit_scale(w["norm_conv.weight"]).reshape(shape)
        self.taps = w["conv1d.weight"][..., 0].astype(mx.float32)
        self.conv = depthwise_conv
        self.square_sums = None
        ngram = text.get("ngram_size", 3)
        self.dilation, self.context = ngram, ngram - 1
        self.window = (self.taps.shape[1] - 1) * self.dilation
        self.dim = text.get("ple_embed_dim") or self.hidden
        eos = text.get("eos_token_id")
        self.eos = eos[0] if isinstance(eos, list) else eos

    def parameters(self) -> list[mx.array]:
        own = [self.key_scale, self.query_scale, self.conv_scale, self.taps]
        return own + self.key.parameters() + self.value.parameters()

    def enable_kernels(self) -> None:
        dtype = self.key.weight.dtype
        self.conv = checked_conv(self.taps, self.dilation, dtype) or self.conv
        self.square_sums = checked_square_sums(self.key_scale, self.eps, dtype)

    def __call__(
        self, h: mx.array, tokens: mx.array, rows: mx.array | None, cache: list
    ) -> mx.array:
        """``rows`` is None only for a warm-up chunk, which reads a zero table."""
        length = h.shape[1]
        history = cache[3]
        if history is None:
            history = mx.full((1, self.context), self.eos, dtype=mx.int64)
        history = mx.concatenate([history, tokens.astype(mx.int64)[None]], axis=-1)
        cache[3] = history[:, length:]

        if rows is None:
            rows = mx.zeros((1, length, self.dim), dtype=h.dtype)
        embeddings = rows.astype(h.dtype)
        sums = self.square_sums
        keys = stream_norm(self.key(embeddings), self.key_scale, self.eps, sums)
        queries = stream_norm(h, self.query_scale, self.eps, sums)
        gated = self.gated(keys, queries, self.value(embeddings))
        normed = stream_norm(gated, self.conv_scale, self.eps, sums)
        conv, cache[2] = self.convolved(normed, cache[2])
        return gated + nn.silu(conv)

    def gated(self, keys: mx.array, queries: mx.array, value: mx.array) -> mx.array:
        """The value gated per stream by the match of its key and the query."""
        shape = (1, keys.shape[1], self.streams, self.hidden)
        gate = row_sum(keys.reshape(shape) * queries.reshape(shape))
        gate = gate / math.sqrt(self.hidden)
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        return (mx.sigmoid(gate) * value[..., None, :]).reshape(keys.shape)

    def convolved(
        self, normed: mx.array, state: mx.array | None
    ) -> tuple[mx.array, mx.array]:
        """(dilated conv of the rows, next conv window)."""
        if state is None:
            state = mx.zeros((1, self.window, normed.shape[-1]), dtype=normed.dtype)
        conv_in = mx.concatenate([state, normed], axis=1)
        return (
            self.conv(conv_in, self.taps, self.dilation),
            conv_in[:, normed.shape[1] :],
        )


class MoE:
    """Routed expert MLPs plus a gated shared expert."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.top_k = text["num_experts_per_tok"]
        self.normalize = text.get("norm_topk_prob", True)
        self.router = w.linear("gate")
        self.shared_gate = w.linear("shared_expert.gate_proj")
        self.shared_up = w.linear("shared_expert.up_proj")
        self.shared_down = w.linear("shared_expert.down_proj")
        self.shared_scale = w.linear("shared_expert_gate")
        banks = w.at("switch_mlp")
        # Gate and up share their input rows: one gather computes both.
        self.gate_up = self._bank(banks, ("gate_proj", "up_proj"))
        self.down = self._bank(banks, ("down_proj",))
        self.combine = None

    @staticmethod
    def _bank(w: Weights, names: tuple[str, ...]) -> tuple:
        """(weight, scales, biases, quantization) of expert projections.

        Several projections of the same input are joined along their outputs.
        """
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
        """Use the combine kernel if it adds expert rows like ``stacked_sum``."""
        k, dtype = self.top_k, self.router.weight.dtype
        y = mx.random.normal((64 * k, 1, self.router.weight.shape[1])).astype(dtype)
        positions = mx.argsort(mx.random.uniform(shape=(64 * k,)))
        scores = mx.random.uniform(shape=(64, k)).astype(dtype)
        want = stacked_sum(y[positions].reshape(64, k, -1) * scores[..., None])
        if kernels.close(
            kernels.combine_experts(y, positions, scores), want, "MoE sum"
        ):
            self.combine = kernels.combine_experts

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

    def route(self, logits: mx.array) -> tuple[mx.array, mx.array]:
        """(expert ids, weights) [1, rows, top_k] from the router logits."""
        k = self.top_k
        gates = mx.softmax(logits, axis=-1, precise=True)
        chosen = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, chosen, axis=-1)
        if self.normalize:
            # The host adds these few values one by one in their dtype.
            total = scores[..., :1]
            for i in range(1, k):
                total = scores[..., i : i + 1] + total
            scores = scores / total
        return chosen, scores

    def experts(self, x: mx.array, chosen: mx.array) -> tuple[mx.array, mx.array]:
        """Expert outputs [rows * top_k, 1, hidden] sorted by expert, and the
        row of each (token, choice) in them."""
        k = self.top_k
        flat = chosen.reshape(-1)
        order = mx.argsort(flat)
        ids = flat[order].astype(mx.uint32)
        routed = x.reshape(x.shape[1], 1, -1)[order // k]
        both = self._gather(routed, self.gate_up, ids)
        inner = both.shape[-1] // 2
        hidden = nn.silu(both[..., :inner]) * both[..., inner:]
        return self._gather(hidden, self.down, ids), mx.argsort(order)

    def combined(self, y: mx.array, positions: mx.array, scores: mx.array) -> mx.array:
        """Each token's expert outputs times their weights, summed."""
        if self.combine is not None:
            return self.combine(y, positions, scores[0])[None]
        rows, k = scores.shape[1:]
        return stacked_sum(y[positions].reshape(1, rows, k, -1) * scores[..., None])

    def shared(self, x: mx.array) -> mx.array:
        shared = self.shared_down(nn.silu(self.shared_gate(x)) * self.shared_up(x))
        return mx.sigmoid(self.shared_scale(x)) * shared

    def __call__(self, x: mx.array) -> mx.array:
        chosen, scores = self.route(self.router(x))
        return self.combined(*self.experts(x, chosen), scores) + self.shared(x)


class Layer:
    """One decoder layer: optional PLE, attention and MoE on the streams."""

    def __init__(self, w: Weights, text: dict[str, Any], kind: str, ple: bool):
        self.ple = PLE(w.at("ple"), text) if ple else None
        self.attn_hc = HyperConnection(w.at("attn_hyper_connection"), text)
        self.mlp_hc = HyperConnection(w.at("mlp_hyper_connection"), text)
        self.qsa = kind == "qsa"
        if self.qsa:
            self.mixer = Attention(w.at("self_attn"), text)
        else:
            self.mixer = GatedDeltaNet(w.at("linear_attn"), text)
        self.mlp = MoE(w.at("mlp"), text)

    def parameters(self) -> list[mx.array]:
        blocks = [self.attn_hc, self.mlp_hc, self.mixer, self.mlp]
        if self.ple is not None:
            blocks.append(self.ple)
        return [a for block in blocks for a in block.parameters()]

    def enable_kernels(self) -> None:
        """Switch each block to the CUDA kernels that pass their self-check."""
        blocks = [self.attn_hc, self.mlp_hc, self.mixer, self.mlp]
        for block in blocks if self.ple is None else [*blocks, self.ple]:
            block.enable_kernels()

    def new_cache(self) -> Any:
        if self.qsa:
            return QSACache()
        return [None] * (4 if self.ple is not None else 2)

    def __call__(
        self, h: mx.array, tokens: mx.array, rows: mx.array | None, cache: Any
    ) -> mx.array:
        if self.ple is not None:
            h = h + self.ple(h, tokens, rows, cache)
        mixed, inject = self.attn_hc(h)
        h = write(h, self.mixer(mixed, cache), inject)
        mixed, inject = self.mlp_hc(h)
        return write(h, self.mlp(mixed), inject)
