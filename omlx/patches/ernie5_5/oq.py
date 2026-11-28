# SPDX-License-Identifier: Apache-2.0
"""oQ export of the ERNIE 5.5 FP4 and FP8 releases.

Only the routed experts are requantized. Every other tensor keeps its release
format (mxfp8 or floating point), and the DSpark draft keeps its source experts.
Calibration runs the model's own cached forward one decoder layer at a time:
each layer gets a fresh request cache per sample, as a single-shot prefill does.
"""

import re
from pathlib import Path

import mlx.core as mx
from transformers import PreTrainedTokenizerFast

from ... import oq
from .config import ModelConfig
from .language import LanguageModel
from .loading import load

_EXPERT_KEY = re.compile(
    r"(model\.(?:layers|mtp)\.\d+\.block_sparse_moe\.experts)\.(\d+)\.(w[123])\.weight"
)
_AFFINE_EXPERT = re.compile(
    r"(language_model\.model\.layers\.\d+\.ffn\.switch_mlp\.\w+_proj)\.biases"
)
_DRAFT_PREFIX = "model.mtp."
_GROUP_PROJ = ".linear_o_group_proj"
_PREFIX = "language_model."
_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_PERTURB_GROUP_SIZE = 64


def sanitizer(config):
    """oQ's replayable sanitize: stacked experts under the runtime module paths."""
    language = LanguageModel(ModelConfig.from_dict(config))

    def sanitize(weights):
        groups = {}
        for key in list(weights):
            match = _EXPERT_KEY.fullmatch(key)
            if match:
                groups.setdefault((match[1], match[3]), []).append((int(match[2]), key))
        for (stem, projection), members in groups.items():
            weights[f"{stem}.{projection}.weight"] = mx.stack(
                [weights.pop(key) for _, key in sorted(members)]
            )
        draft = {
            k + ".weight" if k.endswith(_GROUP_PROJ) else k: weights.pop(k)
            for k in list(weights)
            if k.startswith(_DRAFT_PREFIX)
        }
        out = language.sanitize(weights)
        out.update(draft)
        return {_PREFIX + key: value for key, value in out.items()}

    sanitize._omlx_cast_predicate = lambda name: False
    return sanitize


def check_level(config, oq_level):
    quantization = config.get("quantization_config") or {}
    if oq_level > 4 and quantization.get("expert_dtype") == "mxfp4":
        raise ValueError(
            f"oQ{oq_level:g} needs more than the 4-bit experts of the ERNIE 5.5 "
            "FP4 release. oQ4 keeps those experts as they are; quantize higher "
            "levels from the FP8 release."
        )


def _prefix(index):
    return f"{_PREFIX}model.layers.{index}.ffn.switch_mlp."


def _experts(layer):
    return getattr(layer.ffn, "switch_mlp", None)


def _embedded(language, tokens):
    width = language.config.hc_mult
    hidden = []
    for row in tokens:
        h = language.model.embed_tokens(row[None])
        hidden.append(
            mx.broadcast_to(h[:, :, None], (*h.shape[:2], width, h.shape[-1]))
        )
    return hidden


def _forward(language, index, hidden, before_row=None):
    layer = language.layers[index]
    outputs = []
    for h in hidden:
        if before_row is not None:
            before_row()
        out = layer(h, language.make_cache()[index])
        mx.eval(out)
        outputs.append(out)
    return outputs


def _calibration(model_path, dataset, num_samples, seq_length):
    model, _ = load(model_path, processor=False)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(model_path) / "tokenizer.json")
    )
    tokens = oq._load_calibration_data(tokenizer, dataset, num_samples, seq_length)
    if tokens is None or len(tokens) == 0:
        raise ValueError("ERNIE 5.5 calibration requires nonempty calibration data")
    return model, tokens


def collect_imatrix(model, tokens, progress=None):
    """Routed expert input statistics from the cached forward."""
    language = model.language_model
    collector = oq.OQImatrixCollector()
    hidden = _embedded(language, tokens)
    for index, layer in enumerate(language.layers):
        experts = _experts(layer)
        if experts is not None:
            collector.install(experts, name_prefix=_prefix(index))
        hidden = _forward(language, index, hidden)
        if experts is not None:
            collector.restore(experts)
        if progress is not None:
            progress(index + 1, len(language.layers))
    return collector.entries


def collect_checkpoint_imatrix(
    model_path,
    *,
    calib_dataset,
    num_samples,
    seq_length,
    progress_callback,
    progress_start,
    progress_end,
):
    """oQe imatrix of a release checkpoint, as oQ's collector returns it."""
    model, tokens = _calibration(model_path, calib_dataset, num_samples, seq_length)

    def progress(done, total):
        oq._emit_progress(
            progress_callback,
            "imatrix",
            progress_start + (progress_end - progress_start) * done / total,
            f"ERNIE 5.5 calibration layer {done}/{total}",
        )

    entries = collect_imatrix(model, tokens, progress)
    return entries, {
        "forward": "layerwise_cached_forward",
        "processed_samples": len(tokens),
        "coverage": oq._imatrix_expert_coverage_stats(entries),
        "requires_expert_counts": True,
        "uncalibrated_policy": "preserve_source_precision",
    }


def _requantize(experts, bits):
    for name in _PROJECTIONS:
        module = getattr(experts, name)
        weight = mx.dequantize(
            module.weight,
            module.scales,
            module.get("biases"),
            group_size=module.group_size,
            bits=module.bits,
            mode=module.mode,
        )
        module.weight, module.scales, module.biases = mx.quantize(
            weight, group_size=_PERTURB_GROUP_SIZE, bits=bits, mode="affine"
        )
        module.bits, module.mode = bits, "affine"
        module.group_size = _PERTURB_GROUP_SIZE
        mx.eval(module.weight, module.scales, module.biases)


def measure_sensitivity(
    model_path,
    *,
    bits,
    calib_dataset,
    num_samples,
    seq_length,
):
    """Relative layer output error with the routed experts at ``bits``."""
    model, tokens = _calibration(model_path, calib_dataset, num_samples, seq_length)
    language = model.language_model
    hidden = _embedded(language, tokens)
    scores = {}
    for index, layer in enumerate(language.layers):
        outputs = _forward(language, index, hidden)
        experts = _experts(layer)
        if experts is not None:
            _requantize(experts, bits)
            perturbed = _forward(language, index, hidden)
            error = power = 0.0
            for reference, actual in zip(outputs, perturbed):
                reference = reference.astype(mx.float32)
                error += mx.sum(mx.square(reference - actual)).item()
                power += mx.sum(mx.square(reference)).item()
            scores[index] = error / max(power, 1e-10)
        hidden = outputs
    return scores


def _expert_grams(source, layers):
    model, tokens = _calibration(
        source,
        oq._OQE_REFINE_CALIB_DATASET,
        oq._OQE_REFINE_SAMPLES,
        oq._OQE_REFINE_SEQ_LENGTH,
    )
    language = model.language_model
    hidden = _embedded(language, tokens)
    for index, layer in enumerate(language.layers):
        if index not in layers:
            hidden = _forward(language, index, hidden)
            continue
        experts = layer.ffn.switch_mlp
        collector = oq._OQGramCollector()
        collector.install(experts, name_prefix=_prefix(index))
        hidden = _forward(language, index, hidden, collector.begin_forward)
        collector.restore(experts)
        modules = {
            _prefix(index) + name: getattr(experts, name) for name in _PROJECTIONS
        }
        yield index, modules, collector
        collector.switch_rows.clear()
        mx.clear_cache()


def write_oqh(source, writer, stats, progress):
    """oQh pass over the routed experts, against source Grams."""
    layers = {oq._extract_layer_index(name) for name in writer.target_bits}
    grams = _expert_grams(source, layers)
    for done, (index, modules, collector) in enumerate(grams, start=1):
        writer.process(collector, modules, stats)
        writer.layer_done(index)
        progress(done, len(layers))


def refine_experts(source, output, weight_map, quantization, *, progress):
    """oQe code refinement of the affine routed experts, against source Grams."""
    names = {match[1] for key in weight_map if (match := _AFFINE_EXPERT.fullmatch(key))}
    if not names:
        return {"status": "skipped", "reason": "no affine routed experts"}
    layers = {oq._extract_layer_index(name) for name in names}
    return oq._refine_expert_grams(
        output,
        weight_map,
        quantization,
        names,
        _expert_grams(source, layers),
        progress,
    )
