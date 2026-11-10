# SPDX-License-Identifier: Apache-2.0
"""Prefill forward of the Qwen3.5 blocks, batch one and text only.

Each block follows the path the oMLX host takes for a prefill chunk, op by op
and in the same dtypes, so 16-bit activations round as they do there. The
blocks are ports of the mlx-vlm classes (MIT, Copyright (c) 2025 Prince
Canuma); the attention follows the host's fused kernel, which keeps scores,
softmax and value sums in float32.
"""

from __future__ import annotations

import math
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ... import checkpoint
from ...tensors import Linear
from . import kernels

# Bytes of the float32 score sheet one attention step holds.
_SHEET_BYTES = 1 << 30


class Weights:
    """Lazy checkpoint tensors under one name prefix, with their quantization."""

    def __init__(self, arrays: dict[str, mx.array], config: dict[str, Any], root: str):
        self.arrays, self.config, self.root = arrays, config, root

    def at(self, path: str) -> Weights:
        return Weights(self.arrays, self.config, f"{self.root}{path}.")

    def __getitem__(self, name: str) -> mx.array:
        return self.arrays[self.root + name]

    def linear(self, name: str) -> Linear:
        """A dense linear layer in the dtype the checkpoint computes in."""
        path = self.root + name
        return Linear(self.arrays, path, checkpoint.quantization(self.config, path))


def causal_conv(x: mx.array, taps: mx.array) -> mx.array:
    """Depthwise conv of [1, rows + K - 1, C] with float32 ``taps`` [C, K].

    The taps add up in float32 in order and round once, like the host conv.
    """
    rows = x.shape[1] - taps.shape[1] + 1
    acc = x[:, :rows].astype(mx.float32) * taps[:, 0]
    for i in range(1, taps.shape[1]):
        acc = acc + x[:, i : i + rows].astype(mx.float32) * taps[:, i]
    return acc.astype(x.dtype)


class GatedDeltaNet:
    """Linear attention: a causal conv and the gated delta recurrence."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.key_heads = text["linear_num_key_heads"]
        self.value_heads = text["linear_num_value_heads"]
        self.key_dim = text["linear_key_head_dim"]
        self.value_dim = text["linear_value_head_dim"]
        self.eps = text["rms_norm_eps"]
        self.qkv = w.linear("in_proj_qkv")
        self.z = w.linear("in_proj_z")
        self.b = w.linear("in_proj_b")
        self.a = w.linear("in_proj_a")
        self.out = w.linear("out_proj")
        self.taps = w["conv1d.weight"][..., 0].astype(mx.float32)
        self.decay = mx.exp(w["A_log"].astype(mx.float32))
        self.dt_bias = w["dt_bias"]
        self.norm_weight = w["norm.weight"]
        self.conv = causal_conv

    def enable_kernels(self) -> None:
        """Use the conv kernel if it equals ``causal_conv`` on these taps."""
        rows = 64 + self.taps.shape[1] - 1
        x = mx.random.normal((1, rows, self.taps.shape[0]))
        x = x.astype(self.qkv.weight.dtype)
        got = kernels.causal_conv(x, self.taps)
        if mx.array_equal(got, causal_conv(x, self.taps)).item():
            self.conv = kernels.causal_conv
        else:
            kernels.logger.warning("Causal conv kernel differs; using the reference")

    def parameters(self) -> list[mx.array]:
        linears = (self.qkv, self.z, self.b, self.a, self.out)
        own = [self.taps, self.decay, self.dt_bias, self.norm_weight]
        return own + [a for layer in linears for a in layer.parameters()]

    def prework(self, qkv: mx.array, window: mx.array | None) -> tuple[mx.array, ...]:
        """(q, k, v, next conv window) from the projected rows [1, rows, C]."""
        rows = qkv.shape[1]
        if window is None:
            width = self.taps.shape[1] - 1
            window = mx.zeros((1, width, qkv.shape[-1]), dtype=qkv.dtype)
        conv_in = mx.concatenate([window, qkv], axis=1)
        conv_out = nn.silu(self.conv(conv_in, self.taps))
        keys = self.key_heads * self.key_dim
        q = conv_out[..., :keys].reshape(1, rows, self.key_heads, self.key_dim)
        k = conv_out[..., keys : 2 * keys].reshape(q.shape)
        v = conv_out[..., 2 * keys :]
        v = v.reshape(1, rows, self.value_heads, self.value_dim)
        # L2 norm as an RMS norm; the query also takes the 1/sqrt(dim) of
        # the readout. Both scales multiply the rounded norm.
        scale = self.key_dim**-0.5
        eps = 1e-6 * scale**2
        q = (scale**2) * mx.fast.rms_norm(q, None, eps)
        k = scale * mx.fast.rms_norm(k, None, eps)
        return q, k, v, conv_in[:, rows:]

    def gates(self, a: mx.array, b: mx.array) -> tuple[mx.array, mx.array]:
        """(decay in float32, update rate) of each step and value head."""
        return mx.exp(-self.decay * nn.softplus(a + self.dt_bias)), mx.sigmoid(b)

    def finish(self, y: mx.array, z: mx.array) -> mx.array:
        """The recurrence output normalized and gated, [1, rows, value width]."""
        y = mx.fast.rms_norm(y, self.norm_weight, self.eps)
        gate = nn.silu(z.reshape(y.shape).astype(mx.float32))
        gated = (gate * y.astype(mx.float32)).astype(y.dtype)
        return gated.reshape(1, y.shape[1], -1)

    def __call__(self, x: mx.array, cache: list) -> mx.array:
        q, k, v, cache[0] = self.prework(self.qkv(x), cache[0])
        gamma, beta = self.gates(self.a(x), self.b(x))
        y, cache[1] = mx.fast.gated_delta_update(q, k, v, gamma, beta, cache[1])
        return self.out(self.finish(y, self.z(x)))


class KVCache:
    """Rotated keys and values of one attention layer for one prompt."""

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        # (keys, values) of the last chunk.
        self.appended: tuple[mx.array, ...] = ()

    def append(self, keys: mx.array, values: mx.array) -> None:
        """Add rows [1, kv heads, n, dim]."""
        self.appended = (keys, values)
        if self.keys is not None:
            keys = mx.concatenate([self.keys, keys], axis=2)
            values = mx.concatenate([self.values, values], axis=2)
        self.keys, self.values = keys, values
        self.offset = keys.shape[2]

    def arrays(self) -> list[mx.array]:
        return [a for a in (self.keys, self.values) if a is not None]


def rotary_frequencies(text: dict[str, Any]) -> mx.array:
    """Inverse frequencies of the rotary dims as this backend rounds them.

    Only a warm-up chunk uses them: a prompt's chunks carry the frequencies
    of the host, whose last bits differ.
    """
    rope = text.get("rope_parameters") or {}
    heads = text["num_attention_heads"]
    head_dim = text.get("head_dim") or text["hidden_size"] // heads
    rotary = int(head_dim * rope.get("partial_rotary_factor", 0.25))
    exponent = mx.arange(0, rotary, 2).astype(mx.float32) / rotary
    return 1.0 / (rope.get("rope_theta", 10_000_000) ** exponent)


def rotate(x: mx.array, positions: mx.array, frequencies: mx.array) -> mx.array:
    """Rotate the leading dims of [1, heads, n, dim] by text positions [n].

    Angles, sines and the rotation are float32 and the result rounds once.
    """
    half = frequencies.shape[0]
    angle = positions.astype(mx.float32)[:, None] * frequencies
    cos, sin = mx.cos(angle), mx.sin(angle)
    a = x[..., :half].astype(mx.float32)
    b = x[..., half : 2 * half].astype(mx.float32)
    parts = [(a * cos - b * sin).astype(x.dtype), (b * cos + a * sin).astype(x.dtype)]
    if x.shape[-1] > 2 * half:
        parts.append(x[..., 2 * half :])
    return mx.concatenate(parts, axis=-1)


class Attention:
    """Gated causal attention over every cached token."""

    def __init__(self, w: Weights, text: dict[str, Any]):
        self.heads = text["num_attention_heads"]
        self.kv_heads = text["num_key_value_heads"]
        self.head_dim = text.get("head_dim") or text["hidden_size"] // self.heads
        self.eps = text["rms_norm_eps"]
        self.q_proj = w.linear("q_proj")
        self.k_proj = w.linear("k_proj")
        self.v_proj = w.linear("v_proj")
        self.o_proj = w.linear("o_proj")
        self.q_norm = w["q_norm.weight"]
        self.k_norm = w["k_norm.weight"]
        self.causal_weights = None

    def parameters(self) -> list[mx.array]:
        linears = (self.q_proj, self.k_proj, self.v_proj, self.o_proj)
        own = [self.q_norm, self.k_norm]
        return own + [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        """Use the causal softmax kernel if ``attend`` gives the same result."""
        dtype = self.q_proj.weight.dtype
        rows, total = 40, 104
        keys = mx.random.normal((1, self.kv_heads, total, self.head_dim)).astype(dtype)
        values = mx.random.normal(keys.shape).astype(dtype)
        queries = mx.random.normal((1, self.heads, rows, self.head_dim)).astype(dtype)
        # The reference runs on plain ops, also when a kernel is in use.
        self.causal_weights = None
        want = self.attend(queries, keys, values, step=16)
        self.causal_weights = kernels.causal_weights
        if not kernels.close(
            self.attend(queries, keys, values, step=16), want, "Causal softmax"
        ):
            self.causal_weights = None

    def attend(
        self, queries: mx.array, keys: mx.array, values: mx.array, step: int = 0
    ) -> mx.array:
        """Causal attention of the last rows of the cache over all of it.

        ``queries`` [1, heads, rows, dim] are the newest ``rows`` tokens of
        ``keys`` / ``values`` [1, kv heads, total, dim]. Returns
        [1, rows, heads, dim]. Scores, softmax and value sums are float32 and
        the output rounds once. ``step`` query rows share a score sheet; 0
        sizes it to ``_SHEET_BYTES``.
        """
        rows, total = queries.shape[2], keys.shape[2]
        past = total - rows
        kv, dim = self.kv_heads, self.head_dim
        groups = self.heads // kv
        # Exact for a head width that is a power of four; else it moves the
        # float32 scores by less than their own rounding.
        scale = 1.0 / math.sqrt(dim)
        grouped = queries.reshape(1, kv, groups, rows, dim)
        # Transposed once: every step multiplies by the same keys.
        keys = mx.contiguous(keys.astype(mx.float32).swapaxes(-1, -2))
        values = values.astype(mx.float32)
        step = step or max(1, _SHEET_BYTES // (4 * self.heads * total))
        outputs = []
        for start in range(0, rows, step):
            stop = min(start + step, rows)
            n = stop - start
            q = grouped[..., start:stop, :].astype(mx.float32) * scale
            scores = q.reshape(1, kv, groups * n, dim) @ keys
            if self.causal_weights is not None:
                weights, norm = self.causal_weights(
                    scores.reshape(-1, total), past + start + 1, n
                )
                weights = weights.reshape(scores.shape)
                out = (weights @ values) / norm.reshape(1, kv, groups * n, 1)
            else:
                visible = past + mx.arange(start, stop)[:, None] >= mx.arange(total)
                scores = scores.reshape(1, kv, groups, n, total)
                weights = mx.softmax(mx.where(visible, scores, -mx.inf), axis=-1)
                out = weights.reshape(1, kv, groups * n, total) @ values
            outputs.append(out.astype(queries.dtype).reshape(1, kv, groups, n, dim))
            # One step at a time: each holds a float32 sheet of its scores.
            mx.eval(outputs[-1])
        out = outputs[0] if len(outputs) == 1 else mx.concatenate(outputs, axis=-2)
        return out.reshape(1, self.heads, rows, dim).transpose(0, 2, 1, 3)

    def __call__(self, x: mx.array, cache: KVCache, frequencies: mx.array) -> mx.array:
        rows = x.shape[1]
        projected = self.q_proj(x).reshape(1, rows, self.heads, 2 * self.head_dim)
        queries, gate = projected[..., : self.head_dim], projected[..., self.head_dim :]
        keys = self.k_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        values = self.v_proj(x).reshape(1, rows, self.kv_heads, self.head_dim)
        queries = mx.fast.rms_norm(queries, self.q_norm, self.eps).transpose(0, 2, 1, 3)
        keys = mx.fast.rms_norm(keys, self.k_norm, self.eps).transpose(0, 2, 1, 3)
        positions = mx.arange(cache.offset, cache.offset + rows, dtype=mx.int32)
        queries = rotate(queries, positions, frequencies)
        cache.append(rotate(keys, positions, frequencies), values.transpose(0, 2, 1, 3))
        output = self.attend(queries, cache.keys, cache.values)
        gate = mx.sigmoid(gate.reshape(1, rows, -1))
        return self.o_proj(output.reshape(1, rows, -1) * gate)


class Layer:
    """One decoder layer: linear or full attention, then the MLP."""

    def __init__(self, w: Weights, text: dict[str, Any], linear: bool):
        self.linear = linear
        self.eps = text["rms_norm_eps"]
        self.input_norm = w["input_layernorm.weight"]
        self.post_norm = w["post_attention_layernorm.weight"]
        if linear:
            self.mixer = GatedDeltaNet(w.at("linear_attn"), text)
        else:
            self.mixer = Attention(w.at("self_attn"), text)
        self.gate = w.linear("mlp.gate_proj")
        self.up = w.linear("mlp.up_proj")
        self.down = w.linear("mlp.down_proj")

    def parameters(self) -> list[mx.array]:
        linears = (self.gate, self.up, self.down)
        own = [self.input_norm, self.post_norm, *self.mixer.parameters()]
        return own + [a for layer in linears for a in layer.parameters()]

    def enable_kernels(self) -> None:
        """Switch the mixer to the CUDA kernels that pass their self-check."""
        self.mixer.enable_kernels()

    def new_cache(self) -> Any:
        return [None, None] if self.linear else KVCache()

    def mlp(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))

    def __call__(
        self, x: mx.array, cache: Any, frequencies: mx.array | None
    ) -> mx.array:
        normed = mx.fast.rms_norm(x, self.input_norm, self.eps)
        if self.linear:
            h = x + self.mixer(normed, cache)
        else:
            h = x + self.mixer(normed, cache, frequencies)
        return h + self.mlp(mx.fast.rms_norm(h, self.post_norm, self.eps))
