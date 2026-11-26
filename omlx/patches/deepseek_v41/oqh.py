# SPDX-License-Identifier: MIT
"""oQh routed experts of a DeepSeek V4.1 export.

The main export leaves out the experts the oQh plan covers. This pass loads the
source model and runs the calibration text through it a few layers at a time
(iter_expert_grams); the pass quantizes those experts against their Grams and
adds them to the export, one module per writer call so each stays in one shard.
"""

import logging

import mlx.core as mx

from ... import oq
from ...oqh import format as fmt
from ...oqh import quantize, targets
from .loading import load
from .quantization import QuantizedProjection

logger = logging.getLogger(__name__)

_EXPERTS = ("w1", "w3", "w2")


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


def float_experts(module, first: int, stop: int) -> mx.array:
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
    out_dim = int(module.weight.shape[-2])
    for first, stop in collector.expert_ranges(owner, n_exp, in_dim, out_dim):
        gram = oq._shrunk_gram(collector.switch_gram(owner, first, stop))
        weight = float_experts(module, first, stop)
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


def expert_layer(name: str) -> int:
    return int(name.split("layers.")[1].split(".")[0])


def expert_owner(collector, name: str) -> str:
    """The collector entry holding name's inputs (w3 shares w1's)."""
    owner = collector.aliases.get(name, name)
    if owner not in collector.switch_rows:
        raise RuntimeError(f"V4.1: no calibration input for {name}")
    return owner


def iter_expert_grams(
    source,
    layers,
    *,
    num_samples: int,
    seq_length: int,
    layers_per_pass: int = 2,
):
    """Yield (layer, routed experts, Gram collector) for each of layers, in order.

    Loads the source model and runs the oQe refinement calibration text through
    it a few layers at a time, as pipeline stages: each row's stream (h, pre and
    the shared cross-layer state) carries over to the next group, so no layer
    runs twice. A layer's collector holds its routed experts' inputs until the
    caller resumes the generator.
    """
    layers = sorted(layers)
    model, processor = load(source, engram_ssd_offload=True)
    try:
        tokens = oq._load_calibration_data(
            processor.tokenizer,
            oq._OQE_REFINE_CALIB_DATASET,
            num_samples,
            seq_length,
        )
        if tokens is None or len(tokens) == 0:
            raise ValueError("V4.1 expert calibration requires nonempty text")
        core = model.language_model
        stages = (core.start_idx, core.end_idx)
        # Each row's stream after the layers run so far, the next stage's input.
        streams = [None] * len(tokens)
        stage_start = 0
        for start in range(0, len(layers), layers_per_pass):
            group = layers[start : start + layers_per_pass]
            collectors = {}
            core.start_idx, core.end_idx = stage_start, group[-1] + 1
            try:
                for i in group:
                    collector = oq._OQGramCollector()
                    prefix = f"language_model.layers.{i}.ffn.experts."
                    collector.install(core.layers[i].ffn.experts, name_prefix=prefix)
                    collectors[i] = collector
                for k, row in enumerate(tokens):
                    for collector in collectors.values():
                        collector.begin_forward()
                    capture = {}
                    kwargs = {"skip_logits": True, "_stage_capture": capture}
                    if streams[k] is not None:
                        kwargs["_stage_input"] = streams[k]
                    mx.eval(core(row[None], cache=core.make_cache(), **kwargs))
                    shared = capture["shared"].values()
                    mx.eval(
                        capture["h"],
                        capture["pre"],
                        [v for v in shared if isinstance(v, mx.array)],
                    )
                    streams[k] = capture
            finally:
                core.start_idx, core.end_idx = stages
                for i, collector in collectors.items():
                    collector.restore(core.layers[i].ffn.experts)
            stage_start = group[-1] + 1
            for i in group:
                yield i, core.layers[i].ffn.experts, collectors[i]
                collectors[i].switch_rows.clear()
                mx.clear_cache()
    finally:
        model.close()
        del model, processor
        mx.synchronize()
        mx.clear_cache()


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
        by_layer.setdefault(expert_layer(name), {})[name] = bits
    entries, written = {}, 0
    before = after = 0.0
    grams = iter_expert_grams(
        source,
        by_layer,
        num_samples=num_samples,
        seq_length=seq_length,
        layers_per_pass=layers_per_pass,
    )
    for done, (i, experts, collector) in enumerate(grams, start=1):
        for part in _EXPERTS:
            name = f"language_model.layers.{i}.ffn.experts.{part}"
            if name not in by_layer[i]:
                continue
            values, entry, b0, b1 = _quantize_module(
                collector,
                expert_owner(collector, name),
                name,
                getattr(experts, part),
                by_layer[i][name],
            )
            writer.add(values)
            written += sum(v.nbytes for v in values.values())
            entries[name] = entry
            before, after = before + b0, after + b1
            del values
        if progress is not None:
            progress(done, len(by_layer))
        logger.info("V4.1 oQh: layer %d done (%d/%d)", i, done, len(by_layer))
    return {
        "entries": entries,
        "bytes": written,
        "hessian_before": before,
        "hessian_after": after,
    }
