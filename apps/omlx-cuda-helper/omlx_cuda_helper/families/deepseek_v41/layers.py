# SPDX-License-Identifier: Apache-2.0
"""One DeepSeek V4.1 layer for prefill: packed sparse attention and MoE.

Ported from oMLX (omlx/patches/deepseek_v41/language.py). A layer's cache is
a list of the seven slots the host keeps: offset, window KV, compressed KV,
index keys, partial compressor KV, partial compressor gates, n-gram history.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from ...tensors import unpack
from . import kernels, ops, quant
from .config import Config, expert_bits

WINDOW, COMPRESSED, INDEX_KEYS, PARTIAL_KV, PARTIAL_GATE = 1, 2, 3, 4, 5
# Relative L2 error the expert tile kernels may have against gather_qmm: their
# float32 sums differ in order, which moves a few 16-bit and FP8 values.
_TILE_ERROR = 3e-3

logger = logging.getLogger(__name__)


class Projection:
    """A bias-free linear layer, dense or unpacked from a quantized weight.

    A quantized weight is unpacked once at load into the activation type (fp32
    with ``wide_input``), rounded once from fp32 as the host's kernel sees it.
    The product then runs in the types of its operands, as on the host.
    """

    def __init__(
        self,
        weights: dict[str, mx.array],
        name: str,
        config: Config,
        wide_input: bool = False,
    ):
        weight = weights[f"{name}.weight"]
        spec = config.quantized.get(name)
        self.quantize_input = False
        if spec is not None:
            self.quantize_input = bool(spec.get("quantize_input", True))
            weight = unpack(
                weight,
                weights[f"{name}.scales"],
                weights.get(f"{name}.biases"),
                {"group_size": 32, **spec},
                mx.float32 if wide_input else None,
            )
        elif wide_input:
            weight = weight.astype(mx.float32)
        self.weight_t = mx.contiguous(weight.T)

    def __call__(self, x: mx.array, quantized: mx.array | None = None) -> mx.array:
        """``quantized`` is the FP8 round trip of x when the caller has it."""
        if self.quantize_input:
            x = _quantize(x) if quantized is None else quantized
        return ops.matmul(x, self.weight_t)


def _arrays(value):
    """Every array a layer object holds."""
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, (Projection, ExpertBank, Block)):
        yield from _arrays(list(vars(value).values()))
    elif isinstance(value, dict):
        yield from _arrays(list(value.values()))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _arrays(item)


# Experts repacked at once when a bank is widened at load.
_REPACK_EXPERTS = 4
# Banks from this many weights on are checked at load: an expert kernel that
# offsets experts in 32 bits reads the last ones from the wrong place.
_WIDE_BANK = 1 << 32


def _widen(weight: mx.array, bits: int, wide: int) -> mx.array:
    """Packed affine codes of ``bits`` bits as ``wide``-bit codes."""
    width = weight.shape[-1] * 32 // bits
    per_word = 32 // wide
    bit = mx.arange(32, dtype=mx.uint32)
    out = []
    for begin in range(0, weight.shape[0], _REPACK_EXPERTS):
        part = weight[begin : begin + _REPACK_EXPERTS]
        # The codes are one little-endian bit stream per row.
        ones = ((part[..., None] >> bit) & 1).reshape(*part.shape[:-1], width, bits)
        codes = mx.sum(ones << bit[:bits], axis=-1)
        codes = codes.reshape(*part.shape[:-1], width // per_word, per_word)
        packed = mx.sum(codes << bit[:per_word] * wide, axis=-1).astype(mx.uint32)
        mx.eval(packed)
        out.append(packed)
    return mx.concatenate(out, 0)


class ExpertBank:
    """One projection of every routed expert: dense [E, out, in] or packed.

    A packed bank multiplies in the activation type with the expert kernel of MLX.
    """

    def __init__(self, weights: dict[str, mx.array], name: str, config: Config):
        self.weight = weights[f"{name}.weight"]
        self.spec = config.quantized.get(name)
        self.quantize_input = False
        if self.spec is None:
            return
        self.quantize_input = bool(self.spec.get("quantize_input", True))
        self.scales = weights[f"{name}.scales"]
        self.biases = weights.get(f"{name}.biases")
        self.bits = expert_bits(self.spec)
        if self.bits != self.spec["bits"]:
            self.weight = _widen(self.weight, self.spec["bits"], self.bits)
        experts, rows = self.weight.shape[:2]
        if experts * rows * self.weight.shape[-1] * 32 // self.bits >= _WIDE_BANK:
            self._check_last_expert(name)

    def _check_last_expert(self, name: str) -> None:
        """Raise unless the kernel reads the last expert from its own weights."""
        last = self.weight.shape[0] - 1
        quant = {"group_size": 32, **self.spec, "bits": self.bits}
        biases = None if self.biases is None else self.biases[last]
        dense = unpack(self.weight[last], self.scales[last], biases, quant)
        x = mx.random.normal((8, 1, dense.shape[-1]), key=mx.random.key(0))
        x = x.astype(dense.dtype)
        got = self(x, mx.full((8,), last, mx.uint32))
        want = x.astype(mx.float32) @ dense.astype(mx.float32).T
        differ = mx.mean(got != want.astype(got.dtype)).item()
        if differ > 0.05:
            raise RuntimeError(
                f"{name}: the expert kernel of this MLX build gives wrong "
                f"values for expert {last} ({100 * differ:.0f}% of them); it "
                "needs 64-bit expert offsets"
            )

    def __call__(self, x: mx.array, experts: mx.array) -> mx.array:
        """x [R, 1, in] through the expert of each row; ``experts`` ascends."""
        if self.spec is None:
            return x @ self.weight[experts].swapaxes(-1, -2)
        out = mx.gather_qmm(
            x,
            self.weight,
            self.scales,
            self.biases,
            rhs_indices=experts,
            transpose=True,
            group_size=self.spec.get("group_size", 32),
            bits=self.bits,
            mode=self.spec["mode"],
            sorted_indices=True,
        )
        return out.astype(x.dtype)


def _quantize(x: mx.array, bits: int = 8) -> mx.array:
    """quant.quantize, on the CUDA kernel when it runs for this dtype."""
    if kernels.use_pointwise(x.dtype):
        return kernels.transform(x, bits)
    return quant.quantize(x, bits)


def _rope(x, positions, config, compressed, inverse=False, bits=0):
    """RoPE of x [1, L, ..., D] at ``positions``, then quantize(bits) if set."""
    tables = ops.rope_table(positions, config, compressed)
    if kernels.use_pointwise(x.dtype):
        return kernels.transform(x, bits, tables, inverse)
    out = ops.rope_apply(x, *tables, inverse)
    return quant.quantize(out, bits) if bits else out


def _swiglu(gate, up, weights, limit, quantize):
    """SwiGLU of an expert in the activation dtype, then FP8 when the down
    projection takes quantized input. ``weights`` holds one value per row."""
    if kernels.use_pointwise(gate.dtype):
        return kernels.swiglu(gate, up, weights, limit, quantize)
    if weights is not None:
        weights = weights[:, None, None]
    y = ops.swiglu(gate, up, weights, gate.dtype, limit)
    return quant.quantize(y) if quantize else y


def _attend(q, window, pooled, wi, ci, sink, scale):
    """ops.sparse_attention, on the CUDA kernels when they run for this dtype."""
    if kernels.attention_tc_fits(q) and kernels.use_attention_tc(
        q.dtype, *q.shape[-2:]
    ):
        out = kernels.sparse_attention_tc(q, window, pooled, wi, ci, sink, scale)
        # Evaluated here, the layer's peak memory stays about 1.3 GiB lower at
        # 8192 rows than when the rest of the layer runs with it.
        mx.eval(out)
        return out
    if kernels.use_attention(q.dtype):
        return kernels.sparse_attention(q, window, pooled, wi, ci, sink, scale)
    return ops.sparse_attention(q, window, pooled, wi, ci, sink, scale)


def _hc_mixes(x, fn_t, scale, base, config):
    if x.dtype == mx.bfloat16 and kernels.use_hc_mixes(config, fn_t):
        return kernels.hc_mixes(x, fn_t, scale, base, config)
    return ops.hc_mixes(x, fn_t, scale, base, config)


def _hc_pre_norm(x, pre, weight, eps):
    if kernels.use_hc_pre_norm(x.dtype, x.shape[-1]):
        return kernels.hc_pre_norm(x, pre, weight, eps)
    return ops.hc_pre_norm(x, pre, weight, eps)


def _hc_post(x, residual, post, comb):
    if kernels.use_pointwise(x.dtype):
        return kernels.hc_post(x, residual, post, comb)
    return ops.hc_post(x, residual, post, comb)


class Block:
    def __init__(self, weights: dict[str, mx.array], layer: int, config: Config):
        c = self.config = config
        self.layer = layer
        base = f"language_model.layers.{layer}."
        self.ratio = c.compress_ratios[layer]
        self.kv_source = layer in c.kv_source_layers
        self.index_source = bool(self.ratio) and layer in c.index_source_layers
        if self.index_source and 0 <= c.candidate_source_layer <= layer:
            raise ValueError(
                f"Layer {layer} uses candidate blocks, which this helper does not run"
            )

        def tensor(name):
            return weights[base + name]

        def projection(name, wide_input=False):
            return Projection(weights, base + name, c, wide_input)

        self.attn_norm, self.ffn_norm = tensor("attn_norm.weight"), tensor(
            "ffn_norm.weight"
        )
        self.hc = {
            kind: (
                mx.contiguous(tensor(f"hc_{kind}_fn").astype(mx.float32).T),
                tensor(f"hc_{kind}_scale"),
                tensor(f"hc_{kind}_base"),
            )
            for kind in ("attn", "ffn")
        }
        self.sink = tensor("attn.attn_sink").astype(mx.float32)
        self.wq_a, self.wq_b = projection("attn.wq_a"), projection("attn.wq_b")
        self.wkv, self.wo_b = projection("attn.wkv"), projection("attn.wo_b")
        self.q_norm, self.kv_norm = tensor("attn.q_norm.weight"), tensor(
            "attn.kv_norm.weight"
        )
        wo_a = tensor("attn.wo_a.weight").reshape(c.o_groups, c.o_lora_rank, -1)
        self.wo_a_t = mx.contiguous(wo_a.swapaxes(1, 2))
        if self.ratio and self.kv_source:
            # A compressor of a ratio above 1 pools fp32 projections.
            wide = self.ratio > 1
            self.c_wkv = projection("attn.compressor.wkv", wide)
            self.c_norm = tensor("attn.compressor.norm.weight")
            if wide:
                self.c_wgate = projection("attn.compressor.wgate", wide)
        if self.index_source:
            self.i_wq_b = projection("attn.indexer.wq_b")
            self.i_weights = projection("attn.indexer.weights_proj")
            if self.kv_source:
                self.i_wk = projection("attn.indexer.wk")
                self.i_k_norm = tensor("attn.indexer.k_norm.weight")
        self.gate_t = mx.contiguous(tensor("ffn.gate.weight").astype(mx.float32).T)
        self.gate_bias = tensor("ffn.gate.bias")
        self.experts = [
            ExpertBank(weights, f"{base}ffn.experts.{n}", c) for n in ("w1", "w3", "w2")
        ]
        # Set by check_kernels once the expert tile kernels reproduce gather_qmm;
        # "mx" for the block-scaled tiles.
        self.tiles: bool | str = False
        self.shared = [
            projection(f"ffn.shared_experts.{n}") for n in ("w1", "w3", "w2")
        ]
        self.engram = None
        if layer in c.engram_layer_ids:
            self.engram = (
                projection("engram.wkv"),
                tensor("engram.q_weight").astype(mx.float32)
                * tensor("engram.k_weight").astype(mx.float32),
            )

    def parameters(self) -> list[mx.array]:
        return list(_arrays(self))

    def add_engram(self, h, rows, inverse):
        """Gate the Engram value of each position into every stream.

        ``rows`` are the table rows of the unique id tuples, ``inverse`` maps
        positions [1, L] to them (None when rows are already per position).
        """
        c = self.config
        wkv, weight = self.engram
        kv = wkv(rows)
        if inverse is not None:
            kv = kv[inverse]
        width = c.dim * c.hc_mult
        key = kv[..., :width].astype(mx.float32).reshape(h.shape)
        value = kv[..., width:].astype(mx.float32)
        x = h.astype(mx.float32)
        inv = mx.rsqrt(mx.mean(x * x, -1) + c.norm_eps) * mx.rsqrt(
            mx.mean(key * key, -1) + c.norm_eps
        )
        dot = mx.sum(x * weight * key, -1) * inv * c.dim**-0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        # A zero dot product takes the positive branch of the reference.
        gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)
        return (x + gate[..., None] * value[..., None, :]).astype(h.dtype)

    def _compress(self, x, cache, start):
        """New compressed rows of this chunk before RoPE, or None."""
        c, r = self.config, self.ratio
        if r == 1:
            return ops.norm(self.c_wkv(x), self.c_norm, c.norm_eps)
        f = x.astype(mx.float32)
        kv, gate = self.c_wkv(f), self.c_wgate(f)
        rem = start % r
        if rem:
            kv = mx.concatenate([cache[PARTIAL_KV][:, :rem], kv], 1)
            gate = mx.concatenate([cache[PARTIAL_GATE][:, :rem], gate], 1)
        cutoff = kv.shape[1] // r * r
        cache[PARTIAL_KV], cache[PARTIAL_GATE] = kv[:, cutoff:], gate[:, cutoff:]
        if not cutoff:
            return None
        kv = kv[:, :cutoff].reshape(1, -1, r, kv.shape[-1])
        gate = gate[:, :cutoff].reshape(1, -1, r, gate.shape[-1])
        pooled = mx.sum(kv * mx.softmax(gate, axis=2), axis=2)
        return ops.norm(pooled.astype(x.dtype), self.c_norm, c.norm_eps)

    def _index(self, x, qr, latent, cache, shared, start):
        """Top-k compressed rows of each query: [1, L, K] int32."""
        c, ratio, length = self.config, self.ratio, x.shape[1]
        if self.kv_source:
            keys = cache[INDEX_KEYS]
            if keys is None:
                keys = quant.pack(mx.zeros((1, 0, c.index_head_dim)), bits=4)
            keys = keys[:, : start // ratio]
            if latent is not None:
                rows = range(start // ratio * ratio, (start + length) // ratio * ratio)
                new = ops.norm(self.i_wk(latent), self.i_k_norm, c.norm_eps)
                new = quant.pack(_rope(new, rows[::ratio], c, True), bits=4)
                keys = mx.concatenate([keys, new], 1)
            shared["index_k"] = cache[INDEX_KEYS] = keys
        q = self.i_wq_b(qr).reshape(1, length, c.index_n_heads, c.index_head_dim)
        q = _rope(q, range(start, start + length), c, True, bits=4)
        scale = mx.array(c.index_head_dim**-0.5 * c.index_n_heads**-0.5, mx.float32)
        # The reference rounds the head weights to the activation dtype.
        weights = (self.i_weights(x) * scale).astype(x.dtype).astype(mx.float32)
        keys = quant.unpack(shared["index_k"][0], x.dtype, bits=4)
        if kernels.use_index(x.dtype):
            tensor_cores = kernels.index_tc_fits(q) and kernels.use_index_tc(
                *q.shape[-2:]
            )
            return kernels.index_topk(
                q, keys, weights, start, ratio, c.index_topk, tensor_cores
            )
        return ops.index_topk(q, keys, weights, start, ratio, c.index_topk)

    def _attention(self, x, cache, shared, start):
        c, ratio, length = self.config, self.ratio, x.shape[1]
        positions = range(start, start + length)
        quantizes = self.wq_a.quantize_input or self.wkv.quantize_input
        xq = _quantize(x) if quantizes else None
        qr = ops.norm(self.wq_a(x, xq), self.q_norm, c.norm_eps)
        q = self.wq_b(qr).reshape(1, length, c.n_heads, c.head_dim)
        q = _rope(q, positions, c, bool(ratio))
        kv = ops.norm(self.wkv(x, xq), self.kv_norm, c.norm_eps)
        new = quant.pack(_rope(kv, positions, c, bool(ratio)))
        old = cache[WINDOW]
        old_len = min(start, c.window_size, 0 if old is None else old.shape[1])
        window = mx.concatenate([old[:, :old_len], new], 1) if old_len else new
        cache[WINDOW] = window[:, -c.window_size :]
        wi = ops.window_rows(start, length, c.window_size, old_len)
        ci = pooled = None
        if ratio:
            latent = None
            if self.kv_source:
                latent = self._compress(x, cache, start)
                if cache[COMPRESSED] is None:
                    cache[COMPRESSED] = quant.pack(
                        mx.zeros((1, 0, c.head_dim)), 4, 16, True
                    )
                shared["kv"] = cache[COMPRESSED][:, : start // ratio]
            if self.index_source:
                shared["idx"] = self._index(x, qr, latent, cache, shared, start)
            if latent is not None:
                rows = range(start // ratio * ratio, (start + length) // ratio * ratio)
                new = quant.pack(_rope(latent, rows[::ratio], c, True), 4, 16, True)
                shared["kv"] = cache[COMPRESSED] = mx.concatenate(
                    [shared["kv"], new], 1
                )
            ci = shared["idx"][0]
            pooled = quant.unpack(shared["kv"][0], x.dtype, 4, 16, True)
        out = _attend(
            q,
            quant.unpack(window[0], x.dtype),
            pooled,
            wi,
            ci,
            self.sink,
            c.head_dim**-0.5,
        )
        # The reference quantizes the head outputs before the grouped wo_a.
        out = _rope(out, positions, c, bool(ratio), inverse=True, bits=8)
        return self.wo_b(self.group_out(out))

    def group_out(self, out):
        """Head outputs [1, L, H, D] through the grouped low-rank wo_a."""
        length, groups = out.shape[1], self.config.o_groups
        grouped = out.reshape(length, groups, -1).swapaxes(0, 1)
        low = ops.matmul(grouped, self.wo_a_t)
        return low.swapaxes(0, 1).reshape(1, length, -1)

    def route(self, flat):
        """(experts [L, k] by descending score, route weights [L, k])."""
        c = self.config
        raw = (flat.astype(mx.float32) @ self.gate_t) / c.gate_temp
        if c.score_func == "softmax":
            scores = mx.softmax(raw, -1)
        elif c.score_func == "sigmoid":
            scores = mx.sigmoid(raw)
        else:
            scores = mx.sqrt(mx.logaddexp(raw, 0))
        k = c.n_activated_experts
        chosen = mx.argsort(-(scores + self.gate_bias), axis=-1)[:, :k]
        weights = mx.take_along_axis(scores, chosen, -1)
        if c.norm_topk_prob and k > 1:
            weights = weights / (mx.sum(weights, -1, keepdims=True) + 1e-20)
        return chosen, weights * c.route_scale

    def _routed(self, flat, xq, chosen, weights, tiles):
        """(each route's expert output [R, D] sorted by expert, inverse order).

        ``xq`` is the FP8 round trip of ``flat``; ``tiles`` runs the expert
        tile kernels instead of gather_qmm, "mx" the block-scaled ones.
        """
        c, k = self.config, chosen.shape[-1]
        w1, w3, w2 = self.experts
        # Routes sorted by expert let the expert kernels batch each expert.
        experts = chosen.reshape(-1).astype(mx.uint32)
        order = mx.argsort(experts)
        experts, token = experts[order], order // k
        route_weights = weights.reshape(-1)[order]
        if tiles == "mx":
            codes, scales = kernels.expert_gate_up_mx(
                *kernels.fp8_codes(flat),
                token.astype(mx.uint32),
                experts,
                w1,
                w3,
                route_weights,
                c.swiglu_limit,
            )
            routed = kernels.expert_down_mx(codes, scales, experts, w2)
        elif tiles:
            act = kernels.expert_gate_up(
                xq if w1.quantize_input else flat,
                token.astype(mx.uint32),
                experts,
                w1,
                w3,
                route_weights,
                c.swiglu_limit,
                w2.quantize_input,
            )
            routed = kernels.expert_down(act, experts, w2)
        else:
            rows = {q: (xq if q else flat)[token][:, None] for q in (False, True)}
            gate = w1(rows[w1.quantize_input], experts)
            up = w3(rows[w3.quantize_input], experts)
            y = _swiglu(gate, up, route_weights, c.swiglu_limit, w2.quantize_input)
            routed = w2(y, experts)
        return routed, mx.argsort(order)

    def check_kernels(self) -> None:
        """Turn the expert tile kernels on when they reproduce gather_qmm on a
        sample of routes through this layer's banks."""
        w1, w3, _ = self.experts
        if not (
            kernels.available()
            and kernels.experts_fit(self.experts)
            and w1.quantize_input == w3.quantize_input
        ):
            return
        c = self.config
        keys = mx.random.split(mx.random.key(5), 3)
        flat = mx.random.normal((96, c.dim), key=keys[0]).astype(self.attn_norm.dtype)
        shape = (96, c.n_activated_experts)
        chosen = mx.random.randint(0, c.n_routed_experts, shape, key=keys[1])
        weights = mx.random.uniform(0.1, 1.0, shape, key=keys[2])
        xq = _quantize(flat)
        want = self._routed(flat, xq, chosen, weights, False)[0]
        want = want.reshape(chosen.size, -1)
        try:
            got = self._routed(flat, xq, chosen, weights, True)[0]
            error = kernels.relative_error(got, want)
        except RuntimeError as failure:
            logger.warning("DeepSeek V4.1 expert tile kernels are off: %r", failure)
            return
        self.tiles = error <= _TILE_ERROR
        if not self.tiles:
            logger.warning(
                "DeepSeek V4.1 expert tile kernels are off: layer %d differs from "
                "gather_qmm by %.2e",
                self.layer,
                error,
            )
            return
        # The block-scaled mma takes the activations' FP8 round trip as codes.
        if not (
            kernels.mx_experts_fit(self.experts)
            and all(w.quantize_input for w in self.experts)
        ):
            return
        try:
            got = self._routed(flat, xq, chosen, weights, "mx")[0]
            error = kernels.relative_error(got, want)
        except RuntimeError as failure:
            logger.warning("DeepSeek V4.1 MX expert tiles are off: %r", failure)
            return
        if error <= _TILE_ERROR:
            self.tiles = "mx"
        else:
            logger.warning(
                "DeepSeek V4.1 MX expert tiles are off: layer %d differs from "
                "gather_qmm by %.2e",
                self.layer,
                error,
            )

    def _moe(self, x):
        c, dtype = self.config, x.dtype
        flat = x.reshape(-1, x.shape[-1])
        chosen, weights = self.route(flat)
        w1, w3, w2 = self.experts
        s1, s3, s2 = self.shared
        quantizes = any(p.quantize_input for p in (w1, w3, s1, s3))
        # Quantization is row-local, so one round trip serves every route.
        xq = _quantize(flat) if quantizes else flat
        routed, inverse = self._routed(flat, xq, chosen, weights, self.tiles)
        y = _swiglu(s1(flat, xq), s3(flat, xq), None, c.swiglu_limit, s2.quantize_input)
        # y is already quantized when the down projection asks for it.
        shared = s2(y, y)
        if kernels.use_combine(dtype):
            out = kernels.combine(routed, inverse, shared)
        else:
            out = ops.combine(routed, inverse, shared)
        return out.reshape(x.shape)

    def __call__(self, h, pre, cache, shared, start):
        """Run the layer on streams h [1, L, n, D]; returns (h, next pre)."""
        c = self.config
        ap, ao, ac = _hc_mixes(h, *self.hc["attn"], c)
        x = _hc_pre_norm(h, pre, self.attn_norm, c.norm_eps)
        h = _hc_post(self._attention(x, cache, shared, start), h, ao, ac)
        fp, fo, fc = _hc_mixes(h, *self.hc["ffn"], c)
        x = _hc_pre_norm(h, ap, self.ffn_norm, c.norm_eps)
        return _hc_post(self._moe(x), h, fo, fc), fp
