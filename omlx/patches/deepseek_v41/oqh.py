# SPDX-License-Identifier: MIT
"""oQh routed experts of a DeepSeek V4.1 export.

The main export leaves out the experts the oQh plan covers. This pass loads the
source model, runs forwards over the calibration text a few layers at a time
(each stops at the next block) while the oQe Gram collector records each routed
expert's inputs (w3 shares w1's), quantizes those experts against their Grams
and adds them to the export, one module per writer call so each stays in one
shard.
"""

import contextlib
import logging

import mlx.core as mx

from ... import oq
from ...oqh import format as fmt
from ...oqh import quantize, targets
from .loading import load
from .quantization import QuantizedProjection

logger = logging.getLogger(__name__)

_EXPERTS = ("w1", "w3", "w2")


class _GroupDoneError(Exception):
    """Raised by the first block after the layers a pass collects."""


def _raise_stop(self, *args, **kwargs):
    raise _GroupDoneError


def _stop_before(block) -> type:
    """Make block raise _GroupDoneError when called; returns its class to restore."""
    base = type(block)
    block.__class__ = type(f"Stop{base.__name__}", (base,), {"__call__": _raise_stop})
    return base


def plan_widths(inventory, allocation, layer_scores) -> dict[str, int]:
    """{module: bits} of the routed experts within their affine plan's bytes."""
    shapes, planned = {}, {}
    for name, spec in allocation.items():
        shape = tuple(inventory[name]["shape"])
        if spec.get("mode", "affine") != "affine":
            continue
        if not targets.DEEPSEEK_V41.is_target(name, shape):
            continue
        shapes[name] = shape
        planned[name] = spec
    return targets.reinvest(shapes, planned, layer_scores)


def planned_bytes(inventory, allocation, names) -> int:
    """Bytes the affine plan gives these modules."""
    return sum(
        targets._affine_bytes(tuple(inventory[n]["shape"]), allocation[n])
        for n in names
    )


def _float_experts(module, first: int, stop: int) -> mx.array:
    if isinstance(module, QuantizedProjection):
        return mx.dequantize(
            module.weight[first:stop],
            module.scales[first:stop],
            module.get("biases")[first:stop] if "biases" in module else None,
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        ).astype(mx.float32)
    return module.weight[first:stop].astype(mx.float32)


def _quantize_module(collector, owner, name, module, bits):
    n_exp = int(module.weight.shape[0])
    in_dim = oq.OQImatrixCollector._module_in_dim(module)
    key, block = targets.DEEPSEEK_V41.rotation(name, in_dim)
    signs = mx.array(fmt.make_signs(key, in_dim)).astype(mx.float32)
    packed, rscale = [], []
    before = after = 0.0
    for first, stop in collector.expert_ranges(owner, n_exp, in_dim):
        gram = oq._shrunk_gram(collector.switch_gram(owner, first, stop))
        weight = _float_experts(module, first, stop)
        p, r, b0, b1 = quantize.quantize(weight, gram, bits, signs, block)
        packed.append(p)
        rscale.append(r)
        before, after = before + b0, after + b1
        del gram, weight
        mx.clear_cache()
    values = {
        f"{name}.weight": mx.concatenate(packed),
        f"{name}.rscale": mx.concatenate(rscale),
        f"{name}.oqh_signs": signs.reshape(-1, block).astype(mx.bfloat16),
    }
    mx.eval(values)
    return values, fmt.config_entry(bits, block), before, after


def write_experts(
    source,
    writer,
    widths: dict[str, int],
    *,
    num_samples: int,
    seq_length: int,
    layers_per_pass: int = 2,
    progress=None,
) -> dict:
    """Quantize the planned experts from the source model and add them to writer.

    Returns {"entries": {module: config entry}, "bytes": written bytes,
    "hessian_before": ..., "hessian_after": ...}.
    """
    by_layer: dict[int, dict[str, int]] = {}
    for name, bits in widths.items():
        layer = int(name.split("layers.")[1].split(".")[0])
        by_layer.setdefault(layer, {})[name] = bits
    layers = sorted(by_layer)
    model, processor = load(source, engram_ssd_offload=True)
    entries, written = {}, 0
    before = after = 0.0
    try:
        tokens = oq._load_calibration_data(
            processor.tokenizer,
            oq._OQE_REFINE_CALIB_DATASET,
            num_samples,
            seq_length,
        )
        if tokens is None or len(tokens) == 0:
            raise ValueError("V4.1 oQh requires nonempty calibration data")
        core = model.language_model
        for start in range(0, len(layers), layers_per_pass):
            group = layers[start : start + layers_per_pass]
            collectors = {}
            # The collector evaluates what it records, so the forward can stop
            # at the block after the group.
            nxt = group[-1] + 1
            stop = core.layers[nxt] if nxt < len(core.layers) else None
            restore = None if stop is None else _stop_before(stop)
            try:
                for i in group:
                    collector = oq._OQGramCollector()
                    prefix = f"language_model.layers.{i}.ffn.experts."
                    collector.install(core.layers[i].ffn.experts, name_prefix=prefix)
                    collectors[i] = collector
                for row in tokens:
                    for collector in collectors.values():
                        collector.begin_forward()
                    cache = core.make_cache()
                    with contextlib.suppress(_GroupDoneError):
                        mx.eval(core(row[None], cache=cache)[:, -1])
                    del cache
            finally:
                if restore is not None:
                    stop.__class__ = restore
                for i, collector in collectors.items():
                    collector.restore(core.layers[i].ffn.experts)
            for i in group:
                collector = collectors[i]
                experts = core.layers[i].ffn.experts
                for part in _EXPERTS:
                    name = f"language_model.layers.{i}.ffn.experts.{part}"
                    if name not in by_layer[i]:
                        continue
                    owner = collector.aliases.get(name, name)
                    if owner not in collector.switch_rows:
                        raise RuntimeError(f"V4.1 oQh: no calibration input for {name}")
                    values, entry, b0, b1 = _quantize_module(
                        collector,
                        owner,
                        name,
                        getattr(experts, part),
                        by_layer[i][name],
                    )
                    writer.add(values)
                    written += sum(v.nbytes for v in values.values())
                    entries[name] = entry
                    before, after = before + b0, after + b1
                    del values
                collector.switch_rows.clear()
                mx.clear_cache()
            if progress is not None:
                progress(min(start + layers_per_pass, len(layers)), len(layers))
            logger.info(
                "V4.1 oQh: layers %s done (%d/%d)",
                group,
                min(start + layers_per_pass, len(layers)),
                len(layers),
            )
    finally:
        model.close()
        del model, processor
        mx.synchronize()
        mx.clear_cache()
    return {
        "entries": entries,
        "bytes": written,
        "hessian_before": before,
        "hessian_after": after,
    }
