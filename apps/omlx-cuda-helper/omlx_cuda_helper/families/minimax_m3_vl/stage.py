# SPDX-License-Identifier: Apache-2.0
"""Layers [first, last) of MiniMax M3 / M3.1 on one GPU, prefill only.

A fresh port of the oMLX MiniMax M3 prefill forward (model code derived from
mlx-vlm, MIT, Copyright (c) 2025 Prince Canuma).
"""

from __future__ import annotations

import functools
import logging
import math
import time
from typing import Any

import mlx.core as mx
import numpy as np

from ... import checkpoint
from ...tensors import Linear, from_host, load, to_host
from ...wire import HostArray
from . import ROOT, Config, kernels, read_config, sparse

logger = logging.getLogger("omlx_cuda_helper.minimax_m3_vl")

# mlx-lm writes this divisor into its NVFP4 per-expert global scales.
_NVFP4_GLOBAL_SCALE = 6.0 * 448.0
# Largest relative error of a kernel path against its portable twin; a wrong
# kernel is off by orders of magnitude more.
_KERNEL_TOLERANCE = 1e-2
Arrays = dict[str, HostArray]


def _dequantize(weight, scales, biases, quant: dict) -> mx.array:
    """Quantized values as Metal unpacks them: an affine value is formed in
    float32 and rounded once (CUDA rounds twice); FP modes are exact in bf16."""
    sizes = {"group_size": quant["group_size"], "bits": quant["bits"]}
    mode = quant.get("mode", "affine")
    if mode != "affine":
        return mx.dequantize(weight, scales, mode=mode, dtype=mx.bfloat16, **sizes)
    exact = (scales.astype(mx.float32), biases.astype(mx.float32))
    return mx.dequantize(weight, *exact, mode=mode, **sizes).astype(scales.dtype)


def _norm_scale(weights: dict[str, mx.array], name: str, gemma: bool) -> mx.array:
    # Float32, so adding 1 to a bfloat16 weight does not round every channel.
    scale = weights[name].astype(mx.float32)
    return scale + 1 if gemma else scale


def _rms_norm(x: mx.array, scale: mx.array, eps: float, dtype: mx.Dtype) -> mx.array:
    """RMS norm with a float32 scale, computed in float32 and rounded once."""
    return mx.fast.rms_norm(x, scale, eps).astype(dtype)


# exp2(-(i / pairs) * log2(base)) as Metal returns it for M3.1 and M3: some are
# one float32 above the correctly rounded value, which later angles amplify.
_METAL_FREQUENCIES = {
    (64, 10000000.0): (
        0x3F800000, 0x3F1AB32B, 0x3EBAF81B, 0x3E61F836, 0x3E088D78, 0x3DA50958,
        0x3D47763F, 0x3CF11177, 0x3C91AD3A, 0x3C301054, 0x3BD4CA17, 0x3B809680,
        0x3B1B690C, 0x3ABBD3F1, 0x3A6301E3, 0x3A092E07, 0x39A5CB62, 0x394860C0,
        0x38F22CE7, 0x3892587F, 0x3830DF56, 0x37D5C443, 0x37812DB0, 0x371C1FCD,
        0x36BCB0C0, 0x36640CCB, 0x3609CF51, 0x35A68E4A, 0x35494C59, 0x34F349A5,
        0x34930497, 0x3431AF44,
    ),
    (64, 5000000.0): (
        0x3F800000, 0x3F1E165D, 0x3EC33F3B, 0x3E71242A, 0x3E14E963, 0x3DB7EA1C,
        0x3D63251B, 0x3D0C44BF, 0x3CAD3D5F, 0x3C55F607, 0x3C04208A, 0x3BA32F41,
        0x3B498AD3, 0x3AF8EA9A, 0x3A99B687, 0x3A3DD82F, 0x39EA7801, 0x3990CA8A,
        0x3932D353, 0x38DCDC15, 0x38886324, 0x38287232, 0x37D00A69, 0x37807898,
        0x371EAB4B, 0x36C3F72E, 0x36720760, 0x361575A9, 0x35B8975F, 0x3563FB1F,
        0x350CC8EC, 0x34ADE091,
    ),
}  # fmt: skip


@functools.lru_cache(maxsize=4)
def _rope_tables(cfg: Config, start: int, length: int) -> tuple[mx.array, mx.array]:
    """cos and sin [L, pairs] float32 of the angles Metal's RoPE forms for
    positions ``start`` on: position times frequency, rounded to float32."""
    pairs = cfg.rope_dims // 2
    bits = _METAL_FREQUENCIES.get((cfg.rope_dims, cfg.rope_base))
    if bits is None:
        exponent = -np.arange(pairs, dtype=np.float32) / np.float32(pairs)
        exponent = exponent * np.float32(math.log2(cfg.rope_base))
        frequencies = np.exp2(exponent.astype(np.float64)).astype(np.float32)
    else:
        frequencies = np.array(bits, dtype=np.uint32).view(np.float32)
    positions = np.arange(start, start + length, dtype=np.float32)
    positions = np.float32(cfg.rope_scale) * positions
    angles = (positions[:, None] * frequencies).astype(np.float64)
    return (
        mx.array(np.cos(angles).astype(np.float32)),
        mx.array(np.sin(angles).astype(np.float32)),
    )


def _rope(x: mx.array, cfg: Config, start: int, dtype: mx.Dtype, kernel: bool):
    """Rotate the first ``rope_dims`` features of float32 ``x`` [1, L, heads,
    dim] and return [1, heads, L, dim] in ``dtype``."""
    if kernel:
        cos, sin = _rope_tables(cfg, start, x.shape[1])
        return kernels.rope(x[0], cos, sin, dtype)
    x = mx.fast.rope(
        x.transpose(0, 2, 1, 3),
        cfg.rope_dims,
        traditional=False,
        base=cfg.rope_base,
        scale=cfg.rope_scale,
        offset=start,
    )
    return x.astype(dtype)


def _swiglu(up, gate, up_scale, gate_scale, alpha: float, limit: float, beta: float):
    """The clipped SwiGLU in float32, rounded once to the input dtype."""
    glu = mx.minimum(gate.astype(mx.float32) * gate_scale, limit)
    linear = mx.clip(up.astype(mx.float32) * up_scale, -limit, limit)
    return (glu * mx.sigmoid(alpha * glu) * (linear + beta)).astype(up.dtype)


class _Dense:
    """Bias-free linears of one input as one dense weight, unpacked at load to
    the values Metal's quantized matmul multiplies by."""

    def __init__(self, weights: dict[str, mx.array], prefixes: list[str], config):
        parts = []
        for prefix in prefixes:
            weight = weights[f"{prefix}.weight"]
            scales = weights.get(f"{prefix}.scales")
            if scales is not None:
                quant = checkpoint.quantization(config, prefix)
                biases = weights.get(f"{prefix}.biases")
                weight = _dequantize(weight, scales, biases, quant)
            parts.append(weight)
        self.weight = parts[0] if len(parts) == 1 else mx.concatenate(parts)
        ends = np.cumsum([part.shape[0] for part in parts])
        self.splits = [int(end) for end in ends[:-1]]

    def __call__(self, x: mx.array):
        out = x @ self.weight.T
        return mx.split(out, self.splits, axis=-1) if self.splits else out


class _Embedding:
    def __init__(self, weights: dict[str, mx.array], prefix: str, config: dict):
        self.weight = weights[f"{prefix}.weight"]
        self.scales = weights.get(f"{prefix}.scales")
        self.biases = weights.get(f"{prefix}.biases")
        self.quant = checkpoint.quantization(config, prefix)

    def parameters(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, ids: mx.array) -> mx.array:
        if self.scales is None:
            return self.weight[ids]
        biases = None if self.biases is None else self.biases[ids]
        return _dequantize(self.weight[ids], self.scales[ids], biases, self.quant)


class _MLP:
    def __init__(self, weights, prefix: str, config: dict, cfg: Config):
        names = [f"{prefix}.gate_proj", f"{prefix}.up_proj"]
        self.gate_up = _Dense(weights, names, config)
        self.down = _Dense(weights, [f"{prefix}.down_proj"], config)
        self.act = (cfg.alpha, cfg.limit, cfg.beta)

    def linears(self) -> list[_Dense]:
        return [self.gate_up, self.down]

    def __call__(self, x: mx.array) -> mx.array:
        gate, up = self.gate_up(x)
        return self.down(_swiglu(up, gate, 1.0, 1.0, *self.act))


class _Bank:
    """One projection of every routed expert, packed as in the checkpoint."""

    def __init__(self, weights, prefix: str, config: dict):
        self.weight = weights[f"{prefix}.weight"]
        self.scales = weights.get(f"{prefix}.scales")
        self.biases = weights.get(f"{prefix}.biases")
        self.quant = checkpoint.quantization(config, prefix)

    def parameters(self) -> list[mx.array]:
        return [a for a in (self.weight, self.scales, self.biases) if a is not None]

    def __call__(self, x: mx.array, experts: mx.array) -> mx.array:
        """``x`` [rows, 1, in] through the expert of each row; rows are sorted
        by expert."""
        if self.scales is None:
            return mx.gather_mm(
                x,
                self.weight.swapaxes(-1, -2),
                rhs_indices=experts,
                sorted_indices=True,
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


class _MoE:
    """Routed experts plus the shared expert, a packed bank's last row or an MLP.

    Gate and up expert scales apply in the activation and down's in the routing
    weight, so the banks stay as stored.
    """

    def __init__(self, weights, prefix: str, config: dict, cfg: Config):
        self.cfg = cfg
        switch = f"{prefix}.switch_mlp"
        router = f"{prefix}.gate"
        if f"{router}.scales" in weights:
            quant = checkpoint.quantization(config, router)
            self.router = Linear(weights, router, quant, dense=False)
        else:
            self.router = Linear(weights, router, None, dtype=mx.float32)
        bias = weights.get(f"{prefix}.e_score_correction_bias")
        self.bias = None if bias is None else bias.astype(mx.float32)
        self.fused = f"{switch}.gate_up_proj.weight" in weights
        if self.fused:
            self.gate_up = _Bank(weights, f"{switch}.gate_up_proj", config)
        else:
            self.gate = _Bank(weights, f"{switch}.gate_proj", config)
            self.up = _Bank(weights, f"{switch}.up_proj", config)
        self.down = _Bank(weights, f"{switch}.down_proj", config)
        self.experts = self.router.weight.shape[0]
        # A packed bank holds the shared expert after the routed ones.
        self.shared_row = self.down.weight.shape[0] > self.experts
        self.shared = None
        if f"{prefix}.shared_experts.gate_proj.weight" in weights:
            self.shared = _MLP(weights, f"{prefix}.shared_experts", config, cfg)
        self.scales = {
            name: self._expert_scale(weights, switch, name)
            for name in ("gate", "up", "down")
        }
        # The router weight as [K, experts], the layout the router kernel reads.
        self.columns = None
        if self.router.quant is None:
            self.columns = mx.contiguous(self.router.weight.T)
        # Set by the stage after its kernel checks.
        self.kernel = self.router_kernel = False
        self.act = mx.array([cfg.alpha, cfg.limit, cfg.beta], dtype=mx.float32)

    @staticmethod
    def _expert_scale(weights, switch: str, name: str) -> mx.array | None:
        scale = weights.get(f"{switch}.{name}_scale")
        if scale is None:
            scale = weights.get(f"{switch}.{name}_proj.global_scale")
            if scale is None:
                return None
            scale = scale.astype(mx.float32) / _NVFP4_GLOBAL_SCALE
        return scale.astype(mx.float32)

    def linears(self) -> list[_Dense]:
        return [] if self.shared is None else self.shared.linears()

    def parameters(self) -> list[mx.array]:
        banks = [self.gate_up] if self.fused else [self.gate, self.up]
        extra = [a for a in (self.bias, *self.scales.values()) if a is not None]
        return [
            p for m in (self.router, self.down, *banks) for p in m.parameters()
        ] + extra

    def route_gates(self, gates: mx.array):
        """Experts [rows, k] and their float32 weights, down scales folded in."""
        # A close port of oMLX's MiniMax expert selection (mlx-vlm lineage, MIT).
        cfg = self.cfg
        if cfg.scoring == "sigmoid":
            scores = mx.sigmoid(gates)
        else:
            scores = mx.softmax(gates, axis=-1, precise=True)
        biased = scores if self.bias is None else scores + self.bias
        picked = mx.argpartition(-biased, kth=cfg.top_k - 1, axis=-1)[:, : cfg.top_k]
        weights = mx.take_along_axis(scores, picked, axis=-1)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-20)
        weights = weights * cfg.routed_scale
        if self.scales["down"] is not None:
            weights = weights * self.scales["down"][picked]
        return picked, weights

    def gates(self, rows: mx.array) -> mx.array:
        """Float32 router logits of each token row."""
        if self.router_kernel:
            return kernels.router(rows, self.columns)
        return self.router(rows.astype(mx.float32)).astype(mx.float32)

    def _route(self, rows: mx.array) -> tuple[mx.array, mx.array]:
        """Experts and weights of each token row, with a packed shared expert."""
        picked, weights = self.route_gates(self.gates(rows))
        if self.shared_row:
            count = rows.shape[0]
            shared = mx.full((count, 1), self.experts, dtype=picked.dtype)
            picked = mx.concatenate([picked, shared], axis=-1)
            weights = mx.concatenate([weights, mx.ones((count, 1))], axis=-1)
        return picked, weights

    def kernel_applies(self, dtype: mx.Dtype) -> bool:
        """Whether the expert kernels cover this block: separate FP banks."""
        plain = not self.fused and not self.shared_row and not self.cfg.pack_shared
        return plain and kernels.covers_experts(self.gate, self.up, self.down, dtype)

    def _experts(self, rows, source, experts, inverse, weights) -> mx.array:
        """The weighted sum of each token's routed experts, in float32.

        ``source`` is the token row and ``experts`` the expert of each routed
        row, sorted by expert; ``inverse`` maps the token order back to it.
        """
        cfg = self.cfg
        routed = rows[source][:, None, :]
        if self.fused:
            gate, up = mx.split(self.gate_up(routed, experts), 2, axis=-1)
        else:
            up, gate = self.up(routed, experts), self.gate(routed, experts)
        up_scale, gate_scale = 1.0, 1.0
        if self.scales["up"] is not None:
            up_scale = self.scales["up"][experts][:, None, None]
        if self.scales["gate"] is not None:
            gate_scale = self.scales["gate"][experts][:, None, None]
        act = _swiglu(up, gate, up_scale, gate_scale, cfg.alpha, cfg.limit, cfg.beta)
        y = self.down(act, experts)[:, 0]
        y = y[inverse].reshape(*weights.shape, -1)
        # Routing weights stay float32 and the expert outputs sum in float32.
        return (y.astype(mx.float32) * weights[..., None]).sum(axis=1)

    def _kernel_experts(self, rows, source, experts, inverse, weights) -> mx.array:
        """``_experts`` on the expert kernels, rounded to the row dtype."""
        ones = mx.ones((self.experts,), dtype=mx.float32)
        act = kernels.gate_up(
            rows,
            source,
            experts,
            self.gate,
            self.up,
            ones if self.scales["gate"] is None else self.scales["gate"],
            ones if self.scales["up"] is None else self.scales["up"],
            self.act,
        )
        down = kernels.gather(act, self.down, experts)
        return kernels.weighted_sum(down, inverse, weights)

    def __call__(self, x: mx.array) -> mx.array:
        rows = x.reshape(-1, x.shape[-1])
        picked, weights = self._route(rows)
        each = picked.shape[-1]
        flat = picked.reshape(-1)
        order = mx.argsort(flat)
        routed = (rows, order // each, flat[order], mx.argsort(order), weights)
        packed = self.shared is not None and self.cfg.pack_shared
        if self.kernel:
            out = self._kernel_experts(*routed)
        else:
            out = self._experts(*routed)
            if packed:
                # The host packs this layout into the bank, where the shared
                # expert joins the float32 sum with weight one.
                out = out + self.shared(rows).astype(mx.float32)
            out = out.astype(x.dtype)
        if self.shared is not None and not packed:
            out = out + self.shared(rows)
        return out.reshape(x.shape)


class _Cache:
    """K/V rows of one layer, and its index key rows on a sparse-index layer."""

    def __init__(self):
        self.keys = self.values = self.index = None
        self.offset = 0

    def append(self, keys: mx.array, values: mx.array, index: mx.array | None):
        if self.keys is None:
            self.keys, self.values, self.index = keys, values, index
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
            if index is not None:
                self.index = mx.concatenate([self.index, index], axis=2)
        self.offset = self.keys.shape[2]


class _Attention:
    def __init__(self, weights, prefix: str, config: dict, cfg: Config, index: bool):
        self.cfg, self.index = cfg, index
        names = ["q_proj", "k_proj", "v_proj"]
        norms = ["q_norm", "k_norm"] if cfg.qk_norm else []
        if index:
            if cfg.index_heads not in (1, cfg.kv_heads):
                raise ValueError("sparse attention needs an index head per KV head")
            names += ["index_q_proj", "index_k_proj"]
            norms += ["index_q_norm", "index_k_norm"]
        self.inputs = _Dense(weights, [f"{prefix}.{name}" for name in names], config)
        self.o = _Dense(weights, [f"{prefix}.o_proj"], config)
        self.norms = {
            name: _norm_scale(weights, f"{prefix}.{name}.weight", cfg.gemma_norm)
            for name in norms
        }
        # Set by the stage after its kernel checks.
        self.kernel = False

    def linears(self) -> list[_Dense]:
        return [self.inputs, self.o]

    def _heads(self, y, heads: int, norm: str, start: int, dtype):
        """Split into heads, normalize and rotate in float32, round once."""
        y = y.reshape(1, y.shape[1], heads, -1)
        if norm in self.norms:
            y = _rms_norm(y, self.norms[norm], self.cfg.eps, mx.float32)
        return _rope(y.astype(mx.float32), self.cfg, start, dtype, self.kernel)

    def __call__(self, x: mx.array, cache: _Cache):
        """Attention output and the cache rows (K, V, index K) of the chunk."""
        cfg = self.cfg
        length, start, dtype = x.shape[1], cache.offset, x.dtype
        queries, keys, values, *index = self.inputs(x)
        queries = self._heads(queries, cfg.heads, "q_norm", start, dtype)
        keys = self._heads(keys, cfg.kv_heads, "k_norm", start, dtype)
        values = values.reshape(1, length, cfg.kv_heads, -1).transpose(0, 2, 1, 3)
        index_queries = index_keys = None
        if self.index:
            index_queries = self._heads(
                index[0], cfg.index_heads, "index_q_norm", start, dtype
            )
            index_keys = self._heads(index[1], 1, "index_k_norm", start, dtype)
        cache.append(keys, values, index_keys)
        scale = cfg.head_dim**-0.5
        # Up to the kept blocks the selection keeps every key.
        if self.index and cache.offset > cfg.block * cfg.top_blocks:
            out = sparse.attention(
                queries,
                cache.keys,
                cache.values,
                index_queries,
                cache.index,
                start,
                scale,
                cfg,
                self.kernel,
            )
        elif self.kernel:
            out = kernels.attend_all(queries, cache.keys, cache.values, start, scale)
        else:
            out = mx.fast.scaled_dot_product_attention(
                queries,
                cache.keys,
                cache.values,
                scale=scale,
                mask="causal" if length > 1 else None,
            )
        out = self.o(out.transpose(0, 2, 1, 3).reshape(1, length, -1))
        return out, (keys, values, index_keys)


class _Layer:
    def __init__(self, weights, index: int, config: dict, cfg: Config):
        prefix = f"{ROOT}layers.{index}"
        self.index, self.eps = index, cfg.eps
        self.attention = _Attention(
            weights, f"{prefix}.self_attn", config, cfg, cfg.sparse[index]
        )
        gemma = cfg.gemma_norm
        self.dtype = weights[f"{prefix}.input_layernorm.weight"].dtype
        self.norm = _norm_scale(weights, f"{prefix}.input_layernorm.weight", gemma)
        self.post_norm = _norm_scale(
            weights, f"{prefix}.post_attention_layernorm.weight", gemma
        )
        self.moe = cfg.moe[index]
        if self.moe:
            self.mlp = _MoE(weights, f"{prefix}.block_sparse_moe", config, cfg)
        else:
            self.mlp = _MLP(weights, f"{prefix}.mlp", config, cfg)

    def linears(self) -> list[_Dense]:
        return self.attention.linears() + self.mlp.linears()

    def parameters(self) -> list[mx.array]:
        own = [self.norm, self.post_norm, *self.attention.norms.values()]
        own += [linear.weight for linear in self.linears()]
        return own + (self.mlp.parameters() if self.moe else [])

    def __call__(self, x: mx.array, cache: _Cache):
        dtype = x.dtype
        out, rows = self.attention(_rms_norm(x, self.norm, self.eps, dtype), cache)
        h = x + out
        h = h + self.mlp(_rms_norm(h, self.post_norm, self.eps, dtype))
        return h, rows


class MiniMaxStage:
    def __init__(self, model_dir: str, first: int, last: int):
        self.first, self.last = first, last
        self.device = mx.device_info().get("device_name", "GPU")
        cfg = read_config(model_dir)
        config = checkpoint.load_config(model_dir)
        names = checkpoint.tensors(model_dir)
        if f"{ROOT}layers.{first}.input_layernorm.weight" not in names:
            raise ValueError(f"{model_dir} is not an MLX-layout MiniMax M3 checkpoint")
        self.embedding = None
        if first == 0:
            prefix = f"{ROOT}embed_tokens"
            wanted = [n for n in names if n.startswith(prefix + ".")]
            self.embedding = _Embedding(load(model_dir, wanted), prefix, config)
            mx.eval(self.embedding.parameters())
        self.vocab = names[f"{ROOT}embed_tokens.weight"].shape[0] if first == 0 else 0
        self.layers = []
        for index in range(first, last):
            prefix = f"{ROOT}layers.{index}."
            wanted = [n for n in names if n.startswith(prefix)]
            layer = _Layer(load(model_dir, wanted), index, config, cfg)
            mx.eval(layer.parameters())
            # The packed arrays of the unpacked linears are no longer needed.
            mx.clear_cache()
            self.layers.append(layer)
        self.dtype = self.layers[0].dtype
        self.hidden = self.layers[0].norm.shape[0]
        if kernels.available():
            for layer in self.layers:
                attention = layer.attention
                attention.kernel = kernels.covers_attention(
                    cfg, self.dtype, attention.index
                )
                if layer.moe:
                    mlp = layer.mlp
                    mlp.kernel = mlp.kernel_applies(self.dtype)
                    # The router kernel takes a float32 weight and bfloat16 rows.
                    mlp.router_kernel = (
                        mlp.columns is not None and self.dtype == mx.bfloat16
                    )
            self._verify_kernels(cfg)

    def _verify_kernels(self, cfg: Config) -> None:
        """Run each kernel path and its portable twin on one of this stage's
        layers; a kernel that disagrees is switched off for the whole stage."""
        # Past the dense limit and ending inside a block.
        rows = cfg.block * (cfg.top_blocks + 2) + 37
        key = mx.random.key(0)
        h = mx.random.normal((1, rows, self.hidden), key=key).astype(self.dtype)
        attentions = [layer.attention for layer in self.layers]
        experts = [layer.mlp for layer in self.layers if layer.moe]
        checks = (
            ("attention", "kernel", attentions, lambda m: m(h, _Cache())[0]),
            ("routed experts", "kernel", experts, lambda m: m(h)),
            ("router", "router_kernel", experts, lambda m: m.gates(h[0])),
        )
        for name, switch, modules, run in checks:
            module = next((m for m in modules if getattr(m, switch)), None)
            if module is None:
                continue
            try:
                got = run(module).astype(mx.float32)
                setattr(module, switch, False)
                want = run(module).astype(mx.float32)
                error = (mx.linalg.norm(got - want) / mx.linalg.norm(want)).item()
            except RuntimeError as failure:
                # A kernel that does not compile on this GPU raises here.
                logger.warning("the %s kernel failed: %s", name, failure)
                error = float("inf")
            setattr(module, switch, True)
            if not error <= _KERNEL_TOLERANCE:
                logger.warning(
                    "the %s kernel differs from the portable path (relative "
                    "error %.3g); using the portable path",
                    name,
                    error,
                )
                for other in modules:
                    setattr(other, switch, False)
        mx.clear_cache()

    def warm(self, tokens: int) -> float:
        seconds = 0.0
        # The first pass compiles the kernels; the second one is timed.
        for _ in range(2):
            boundary = None
            if self.first:
                h = mx.random.normal((1, tokens, self.hidden)).astype(self.dtype)
                boundary = {"h": to_host(h)}
            ids = np.arange(tokens, dtype=np.int32) % max(1, min(1000, self.vocab - 3))
            began = time.perf_counter()
            self.run(
                self.new_job(self.last - self.first),
                start=0,
                tokens=ids + 3,
                boundary=boundary,
                inputs={},
                following=0,
                capture=None,
            )
            seconds = time.perf_counter() - began
        mx.clear_cache()
        return seconds

    def memory_bytes(self) -> int:
        return mx.get_active_memory()

    def new_job(self, layers: int) -> list[_Cache]:
        return [_Cache() for _ in range(layers)]

    def load_prefix(self, job: list[_Cache], state: Arrays) -> None:
        for layer, cache in zip(self.layers, job):
            keys = state.get(f"{layer.index}.k")
            if keys is None or not keys[0].shape[2]:
                continue
            index = state.get(f"{layer.index}.ik")
            cache.append(
                from_host(keys),
                from_host(state[f"{layer.index}.v"]),
                None if index is None else from_host(index),
            )
        mx.eval([a for c in job for a in (c.keys, c.values, c.index) if a is not None])

    def run(
        self,
        job: list[_Cache],
        *,
        start: int,
        tokens: np.ndarray,
        boundary: Arrays | None,
        inputs: Arrays,
        following: int,
        capture: dict[str, Any] | None,
    ) -> tuple[Arrays, Arrays]:
        if boundary is None:
            h = self.embedding(mx.array(tokens))[None]
        else:
            h = from_host(boundary["h"])
        length = h.shape[1]
        out: Arrays = {}
        state: Arrays = {}
        for layer, cache in zip(self.layers, job):
            h, (keys, values, index) = layer(h, cache)
            held = (cache.keys, cache.values, cache.index)
            mx.eval(h, [a for a in held if a is not None])
            i = layer.index
            state[f"{i}.k"], state[f"{i}.v"] = to_host(keys), to_host(values)
            if index is not None:
                state[f"{i}.ik"] = to_host(index)
            if capture and i in capture["layers"] and start + length > capture["from"]:
                out[f"cap.{i}"] = to_host(h[:, max(0, capture["from"] - start) :])
        out["h"] = to_host(h)
        return out, state


def load_stage(model_dir: str, first: int, last: int) -> MiniMaxStage:
    return MiniMaxStage(model_dir, first, last)
