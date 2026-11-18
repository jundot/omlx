"""The oQh pass of an oQ build.

The main oQ pass writes every tensor except the oQh targets. This pass walks
the original model layer by layer like the oQe refinement, collects each
target's input Gram, quantizes it to oQh and writes new shards. The other
affine tensors the walk sees get the usual oQe code refinement.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from .. import oq
from . import format as fmt
from . import quantize

logger = logging.getLogger(__name__)


def _float_weight(module) -> mx.array:
    """The module's weight as floats, dequantizing a quantized calibration source."""
    w = module.weight
    if mx.issubdtype(w.dtype, mx.floating):
        return w
    return mx.dequantize(
        w,
        module.scales,
        getattr(module, "biases", None),
        group_size=module.group_size,
        bits=module.bits,
        mode=getattr(module, "mode", "affine"),
    )


class OqhWriter:
    """Writes oQh targets; the other refinable tensors go to the oQe refine writer."""

    def __init__(
        self,
        refine,
        target_bits: dict,
        dtype,
        shard_index: int,
        scheme,
        group_size: int | None = None,
    ):
        self.refine = refine
        self.scheme = scheme
        self.output = refine.output
        self.weight_map = refine.weight_map
        self.target_bits = dict(target_bits)
        self.dtype = dtype
        self.shard_index = shard_index
        # None: one scale per row; else group scales and biases of this size.
        self.group_size = group_size
        self.entries: dict[str, dict] = {}
        self._out: dict[str, mx.array] = {}
        self._signs: dict[str, mx.array] = {}

    def _signs_for(self, name: str, in_dim: int):
        key, block = self.scheme.rotation(name, in_dim)
        signs = self._signs.get(key)
        if signs is None:
            signs = mx.array(fmt.make_signs(key, in_dim)).astype(mx.float32)
            self._signs[key] = signs
        return signs, block

    def _quantize(self, weight, gram, bits, signs, block):
        """({suffix: tensor}, before, after) for one weight or expert range."""
        if self.group_size is None:
            packed, rscale, before, after = quantize.quantize(
                weight, gram, bits, signs, block
            )
            return {"weight": packed, "rscale": rscale}, before, after
        packed, scales, biases, before, after = quantize.quantize_affine(
            weight, gram, bits, signs, block, self.group_size
        )
        parts = {"weight": packed, "oqh_scales": scales, "oqh_biases": biases}
        return parts, before, after

    def _emit(self, name, parts, signs, block, bits) -> None:
        for suffix, value in parts.items():
            if suffix != "weight":
                value = value.astype(self.dtype)
            self._out[f"{name}.{suffix}"] = value
        self._out[f"{name}.oqh_signs"] = signs.reshape(-1, block).astype(self.dtype)
        self.entries[name] = fmt.config_entry(bits, block, self.group_size)

    def _quantize_dense(self, name, weight, gram, stats) -> None:
        bits = self.target_bits[name]
        signs, block = self._signs_for(name, int(weight.shape[-1]))
        parts, before, after = self._quantize(weight, gram, bits, signs, block)
        self._emit(name, parts, signs, block, bits)
        stats["oqh_tensors"] = stats.get("oqh_tensors", 0) + 1
        stats["oqh_before"] = stats.get("oqh_before", 0.0) + before
        stats["oqh_after"] = stats.get("oqh_after", 0.0) + after

    def _quantize_switch(self, collector, owner, name, weight, stats) -> None:
        bits = self.target_bits[name]
        n_exp, in_dim = int(weight.shape[0]), int(weight.shape[-1])
        signs, block = self._signs_for(name, in_dim)
        ranges: list[dict] = []
        before = after = 0.0
        out_dim = int(weight.shape[-2])
        for first, stop in collector.expert_ranges(owner, n_exp, in_dim, out_dim):
            gram = oq._shrunk_gram(collector.switch_gram(owner, first, stop))
            parts, b0, b1 = self._quantize(weight[first:stop], gram, bits, signs, block)
            ranges.append(parts)
            before += b0
            after += b1
            del gram
            mx.clear_cache()
        parts = {k: mx.concatenate([r[k] for r in ranges], 0) for k in ranges[0]}
        self._emit(name, parts, signs, block, bits)
        stats["oqh_tensors"] = stats.get("oqh_tensors", 0) + 1
        stats["oqh_before"] = stats.get("oqh_before", 0.0) + before
        stats["oqh_after"] = stats.get("oqh_after", 0.0) + after

    def process(self, collector, modules: dict, stats: dict) -> None:
        mine = {n: m for n, m in modules.items() if n in self.target_bits}
        rest = {n: m for n, m in modules.items() if n not in self.target_bits}
        owners: dict[str, list] = {}
        for name, module in mine.items():
            owners.setdefault(collector.aliases.get(name, name), []).append(
                (name, module)
            )
        for owner, items in owners.items():
            if owner in collector.grams:
                gram = oq._shrunk_gram(collector.grams[owner])
                for name, module in items:
                    self._quantize_dense(name, _float_weight(module), gram, stats)
                del gram
            elif owner in collector.switch_rows:
                for name, module in items:
                    self._quantize_switch(
                        collector, owner, name, _float_weight(module), stats
                    )
                collector.switch_rows.pop(owner)
            mx.clear_cache()
        oq._refine_captured(collector, rest, self, stats)

    def spec(self, base: str):
        return self.refine.spec(base)

    def read(self, base: str):
        return self.refine.read(base)

    def write(self, base: str, packed) -> None:
        self.refine.write(base, packed)

    def layer_done(self, layer_idx: int) -> None:
        self.refine.layer_done(layer_idx)
        if sum(v.nbytes for v in self._out.values()) >= oq._MAX_SHARD_BYTES:
            self._flush_oqh()

    def _flush_oqh(self) -> None:
        if not self._out:
            return
        self.shard_index += 1
        shard = f"model-{self.shard_index:05d}-of-PLACEHOLDER.safetensors"
        mx.save_safetensors(
            str(self.output / shard), self._out, metadata={"format": "mlx"}
        )
        for key in self._out:
            self.weight_map[key] = shard
        self._out = {}
        mx.clear_cache()

    def finish(self) -> None:
        self.refine.finish()
        self._flush_oqh()
        missing = sorted(set(self.target_bits) - set(self.entries))
        if missing:
            raise RuntimeError(
                f"oQh: {len(missing)} tensors got no calibration input, "
                f"first: {missing[:3]}"
            )
