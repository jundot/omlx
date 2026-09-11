# SPDX-License-Identifier: Apache-2.0
"""Budget-checked oQ3/oQ4 export of the mixed-precision V4.1 release.

Engram follows Qwen4-Exp's uniform affine group-32 embedding policy and is
excluded from the remaining weights' bit budget. If all other source weights
fit their independent budget at original precision, that allocation dominates
every lossy alternative: neither sensitivity ranking nor an imatrix can improve
it. oQ3 measures layer sensitivity and projection input statistics, then uses
the official mixed-bit allocator and weighted affine arithmetic for oQ3e.
"""

import json
import logging
import math
import shutil
import struct
from contextlib import closing
from pathlib import Path

import mlx.core as mx
import numpy as np

from .convert import iter_source_weights, source_engram_tables, strip_draft_config
from .sharding import ShardWriter
from .storage import TensorFile, decode_array

logger = logging.getLogger(__name__)


def requantize_projection(
    values, name, source_spec, *, bits, importance=None, progress=None
):
    """Convert one projection with bounded floating workspace and oQe arithmetic."""
    from ...oq import _quantize_chunked

    if bits not in (2, 3, 4, 6, 8):
        raise ValueError("V4.1 affine projections require 2, 3, 4, 6 or 8 bits")
    weight = values[name + ".weight"]
    if weight.ndim not in (2, 3):
        raise ValueError("Expected a dense or stacked expert projection")
    width = weight.shape[-1]
    if source_spec is not None:
        width = width * 32 // source_spec["bits"]
    if width % 64:
        raise ValueError("V4.1 affine projection width must divide into 64 channels")
    # Slice before dequantization. In particular, do not expand a complete
    # stacked expert tensor to BF16 while its packed source is still live.
    rows_per_chunk = max(1, (128 * 1024**2) // (2 * width))
    experts = weight.shape[0] if weight.ndim == 3 else 1
    packed_experts, scale_experts, bias_experts = [], [], []
    for expert in range(experts):
        current = weight[expert] if weight.ndim == 3 else weight
        packed_rows, scale_rows, bias_rows = [], [], []
        imp = importance
        if importance is not None and importance.ndim == 2:
            imp = importance[expert]
        for start in range(0, current.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, current.shape[0])
            chunk = current[start:stop]
            if source_spec is not None:

                def part(suffix, expert=expert, start=start, stop=stop):
                    value = values.get(name + suffix)
                    if value is None:
                        return None
                    value = value[expert] if weight.ndim == 3 else value
                    return value[start:stop]

                chunk = mx.dequantize(
                    chunk,
                    part(".scales"),
                    part(".biases"),
                    group_size=source_spec.get("group_size", 32),
                    bits=source_spec["bits"],
                    mode=source_spec["mode"],
                )
            q, s, b = _quantize_chunked(
                chunk.astype(mx.bfloat16), 64, bits, "affine", importance=imp
            )
            mx.eval(q, s, b)
            packed_rows.append(q)
            scale_rows.append(s)
            bias_rows.append(b)
            if progress is not None:
                progress(expert * current.shape[0] + stop, experts * current.shape[0])
        packed_experts.append(mx.concatenate(packed_rows))
        scale_experts.append(mx.concatenate(scale_rows))
        bias_experts.append(mx.concatenate(bias_rows))
    join = mx.stack if weight.ndim == 3 else lambda arrays: arrays[0]
    result = {
        name + ".weight": join(packed_experts),
        name + ".scales": join(scale_experts),
        name + ".biases": join(bias_experts),
    }
    mx.eval(result)
    spec = {
        "bits": bits,
        "group_size": 64,
        "mode": "affine",
        "quantize_input": (
            source_spec.get("quantize_input", True) if source_spec else False
        ),
    }
    return result, spec


def source_budget(
    source, config, mapping, *, preserve_mtp, text_only=False, engram_bits=4
):
    """Count actual exported bytes, including scales, norms and draft heads."""
    headers = {}
    for filename in dict.fromkeys(mapping.values()):
        with (source / filename).open("rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            headers[filename] = json.loads(f.read(size))
    params = output_bytes = draft_tensors = 0
    engram_params = engram_bytes = 0
    for name, filename in mapping.items():
        if name.startswith("mtp."):
            if not preserve_mtp:
                continue
            draft_tensors += 1
        if text_only and name.startswith(("vision.", "aligner.", "image_")):
            continue
        if name.endswith(".scale"):
            continue
        info = headers[filename][name]
        shape, dtype = info["shape"], info["dtype"]
        count = math.prod(shape) * (2 if dtype in ("I8", "U8") else 1)
        if ".engram.embed." in name:
            if len(shape) != 2 or shape[-1] % 32:
                raise ValueError("Engram rows must divide into 32-channel groups")
            engram_params += count
            engram_bytes += count * engram_bits // 8 + count // 32 * 4
            continue
        params += count
        if dtype.startswith("F8_E4M3"):
            output_bytes += (
                count * 2
                if name.endswith("wo_a.weight")
                or name in ("embed.weight", "head.weight")
                else count + count // 32
            )
        elif dtype in ("I8", "U8"):
            output_bytes += count // 2 + count // 32
        else:
            output_bytes += info["data_offsets"][1] - info["data_offsets"][0]
    if preserve_mtp and draft_tensors == 0:
        raise ValueError("Requested MTP preservation but source has no draft weights")
    return {
        "remaining_weights": {
            "logical_parameters": params,
            "tensor_bytes": output_bytes,
            "effective_bpw": output_bytes * 8 / max(params, 1),
        },
        "engram": {
            "logical_parameters": engram_params,
            "tensor_bytes": engram_bytes,
            "effective_bpw": engram_bytes * 8 / max(engram_params, 1),
            "bits": engram_bits,
            "group_size": 32,
            "mode": "affine",
        },
        "tensor_bytes": output_bytes + engram_bytes,
        "preserved_source_draft_tensors": draft_tensors,
    }


def quantize_engram(
    source,
    destination,
    table,
    *,
    rows_per_chunk=65536,
    progress=None,
    module_name=None,
    bits=4,
):
    """Write three safetensors arrays in bounded row chunks, never a full table."""
    if bits not in (2, 3, 4, 6, 8):
        raise ValueError("Unsupported Engram affine bit width")
    if rows_per_chunk <= 0:
        raise ValueError("Engram chunk size must be positive")
    reader = TensorFile(source / table["weight_file"])
    try:
        rows, width = reader.header[table["weight_key"]]["shape"]
    finally:
        reader.close()
    if width % 32:
        raise ValueError("Engram width must divide by 32")
    shapes = {
        "weight": (rows, width * bits // 32),
        "scales": (rows, width // 32),
        "biases": (rows, width // 32),
    }
    header, offset = {}, 0
    for key, shape in shapes.items():
        length = math.prod(shape) * (4 if key == "weight" else 2)
        header[key] = {
            "dtype": "U32" if key == "weight" else "BF16",
            "shape": list(shape),
            "data_offsets": [offset, offset + length],
        }
        offset += length
    names = {key: f"{module_name}.{key}" if module_name else key for key in shapes}
    encoded = json.dumps(
        {names[key]: value for key, value in header.items()}, separators=(",", ":")
    ).encode()
    encoded += b" " * (-len(encoded) % 8)
    data_start = 8 + len(encoded)
    partial = destination.with_suffix(destination.suffix + ".partial")
    with partial.open("xb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        out.truncate(data_start + offset)
        for start in range(0, rows, rows_per_chunk):
            selected = np.arange(
                start, min(rows, start + rows_per_chunk), dtype=np.int64
            )
            reader = TensorFile(source / table["weight_file"])
            scales = None
            try:
                values = decode_array(*reader.read(table["weight_key"], selected))
                if table.get("scale_key"):
                    scales = (
                        reader
                        if not table.get("scale_file")
                        or table["scale_file"] == table["weight_file"]
                        else TensorFile(source / table["scale_file"])
                    )
                    factor = decode_array(*scales.read(table["scale_key"], selected))
                    values = (
                        values.reshape(len(selected), -1, 32) * factor[..., None]
                    ).reshape(len(selected), width)
                quantized = mx.quantize(
                    values.astype(mx.bfloat16), group_size=32, bits=bits, mode="affine"
                )
                mx.eval(quantized)
            finally:
                reader.close()
                if scales is not None and scales is not reader:
                    scales.close()
            for key, value in zip(shapes, quantized):
                # NumPy has no native BF16; retain its raw 16-bit representation.
                raw = np.asarray(value if key == "weight" else value.view(mx.uint16))
                row_bytes = raw.shape[-1] * raw.dtype.itemsize
                out.seek(
                    data_start + header[key]["data_offsets"][0] + start * row_bytes
                )
                out.write(raw.tobytes())
            if progress is not None:
                progress(min(rows, start + rows_per_chunk), rows)
            del values, quantized
    partial.rename(destination)
    return {
        "weight_file": str(destination.name),
        "weight_key": names["weight"],
        "scale_file": str(destination.name),
        "scale_key": names["scales"],
        "bias_key": names["biases"],
        "bits": bits,
        "group_size": 32,
        "mode": "affine",
    }


def quantize(
    source,
    destination,
    *,
    oq_level=4,
    enhanced=False,
    preserve_mtp=False,
    target_bpw=None,
    hard_cap_bpw=None,
    text_only=False,
    dtype="bfloat16",
    progress_callback=None,
    group_size=64,
    sensitivity_model_path="",
    sensitivity_map_override=None,
    imatrix_cache_path="",
    imatrix_reuse_cache=True,
    imatrix_strict=False,
    imatrix_num_samples=128,
    imatrix_seq_length=512,
):
    from ...oq import (
        _OQ_BPW_TARGETS,
        _emit_progress,
        _get_predicate_bits,
        _lookup_imatrix_importance,
    )

    if oq_level not in (3, 4) or dtype != "bfloat16" or group_size != 64:
        raise ValueError("V4.1 supports oQ3/oQ4 BF16 export with group size 64")
    if oq_level == 4 and (sensitivity_model_path or sensitivity_map_override):
        raise ValueError(
            "V4.1 original-precision oQ4 does not use sensitivity overrides"
        )
    if text_only:
        raise ValueError(
            "V4.1 oQ export currently preserves the vision encoder and routing biases"
        )
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41" or "omlx_deepseek_v41" in config:
        raise ValueError("V4.1 oQ export currently requires the original checkpoint")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    budget = source_budget(
        source,
        config,
        mapping,
        preserve_mtp=preserve_mtp,
        text_only=text_only,
        engram_bits=oq_level,
    )
    default_target, default_cap = _OQ_BPW_TARGETS[oq_level]
    target = default_target if target_bpw is None else target_bpw
    cap = default_cap if hard_cap_bpw is None else hard_cap_bpw
    if not 0 < target <= cap or not math.isfinite(cap):
        raise ValueError("Expected finite 0 < target_bpw <= hard_cap_bpw")
    if oq_level == 4 and budget["remaining_weights"]["effective_bpw"] > target:
        raise ValueError(
            "Original-precision allocation exceeds the requested oQ4 budget; "
            "V4.1 calibrated affine allocation is currently supported for oQ3"
        )
    allocation, imatrix, preparation = {}, None, {}
    if oq_level == 3:
        allocation, imatrix, preparation = prepare_affine_conversion(
            source,
            destination,
            config,
            mapping,
            budget,
            target=target,
            cap=cap,
            preserve_mtp=preserve_mtp,
            sensitivity_model_path=sensitivity_model_path,
            sensitivity_map_override=sensitivity_map_override,
            imatrix_cache_path=imatrix_cache_path,
            imatrix_reuse_cache=imatrix_reuse_cache,
            imatrix_strict=imatrix_strict,
            imatrix_num_samples=imatrix_num_samples,
            imatrix_seq_length=imatrix_seq_length,
            progress_callback=progress_callback,
        )
        budget["remaining_weights"].update(
            {key: preparation[key] for key in ("tensor_bytes", "effective_bpw")}
        )
        budget["tensor_bytes"] = (
            budget["remaining_weights"]["tensor_bytes"]
            + budget["engram"]["tensor_bytes"]
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination.parent).free < budget["tensor_bytes"] + 1024**3:
        raise OSError("Insufficient space for V4.1 export plus 1 GiB headroom")
    destination.mkdir()
    report = {
        **budget,
        "oq_level": oq_level,
        "enhanced_requested": enhanced,
        "target_bpw": target,
        "hard_cap_bpw": cap,
        "allocation": "original_precision_fits_independent_remaining_budget",
        "budget_scope": "remaining_weights_excluding_engram",
        "calibration": "not_needed_no_linear_requantization",
        "imatrix_applied_modules": [],
        "storage_layout": "unified_safetensors_index",
    }
    if allocation:
        report.update(
            allocation="measured_layer_sensitivity_affine", calibration=preparation
        )
    (destination / "conversion.inprogress.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    logger.info(
        "V4.1 planned non-Engram weights: %.4f bpw; total %.2f GiB",
        budget["remaining_weights"]["effective_bpw"],
        budget["tensor_bytes"] / 1024**3,
    )
    writer = ShardWriter(destination)
    tables = source_engram_tables(mapping)
    exported_tables = {}
    engram_start, engram_span = (20, 20) if oq_level == 3 else (0, 30)
    projection_start = engram_start + engram_span
    for i, (name, table) in enumerate(tables.items()):
        if _get_predicate_bits(name + ".weight", config, oq_level, 64) != (
            oq_level,
            32,
            "affine",
        ):
            raise ValueError("V4.1 Engram policy differs from Qwen4-Exp")
        filename = f".model-engram-{i:02d}.safetensors"
        last = [-1]

        def progress(done, total, i=i, last=last):
            percent = int(100 * done / total)
            if percent != last[0]:
                last[0] = percent
                logger.info("Engram %d/%d: %d%%", i + 1, len(tables), percent)
                _emit_progress(
                    progress_callback,
                    "quantizing",
                    engram_start
                    + engram_span * (i + done / total) / max(len(tables), 1),
                    f"Engram {i + 1}/{len(tables)}: {percent}%",
                )

        spec = quantize_engram(
            source,
            destination / filename,
            table,
            progress=progress,
            module_name=name,
            bits=oq_level,
        )
        writer.add_file(
            filename, [spec[key] for key in ("weight_key", "scale_key", "bias_key")]
        )
        exported_tables[name] = spec
    quantized = {}
    written_bytes = 0
    with closing(
        iter_source_weights(source, config, mapping, preserve_mtp=preserve_mtp)
    ) as batches:
        for i, (values, specs) in enumerate(batches):
            if text_only:
                values = {
                    k: v
                    for k, v in values.items()
                    if not k.startswith(("vision.", "aligner.", "image_"))
                }
                specs = {k: v for k, v in specs.items() if k + ".weight" in values}
            if not values:
                continue
            for key in list(values):
                if not key.endswith(".weight"):
                    continue
                name = key.removesuffix(".weight")
                if name not in allocation:
                    continue
                importance = None
                if enhanced:
                    shape = tuple(values[key].shape)
                    if name in specs:
                        shape = (*shape[:-1], shape[-1] * 32 // specs[name]["bits"])
                    importance = _lookup_imatrix_importance(
                        imatrix,
                        key,
                        shape,
                        config=config,
                        strict=True,
                        report=None,
                    )
                converted, spec = requantize_projection(
                    values,
                    name,
                    specs.get(name),
                    bits=allocation[name]["bits"],
                    importance=importance,
                    progress=lambda done, total, name=name, written=written_bytes: _emit_progress(
                        progress_callback,
                        "quantizing",
                        projection_start
                        + (99 - projection_start)
                        * written
                        / max(budget["remaining_weights"]["tensor_bytes"], 1),
                        f"Quantizing {name}: {done}/{total} rows",
                    ),
                )
                values.update(converted)
                specs[name] = spec
                if enhanced:
                    report["imatrix_applied_modules"].append(name)
            writer.add(values)
            written_bytes += sum(value.nbytes for value in values.values())
            quantized.update(specs)
            if i % 25 == 0:
                logger.info(
                    "Exported %d projection groups (including DSpark=%s)",
                    i + 1,
                    preserve_mtp,
                )
                _emit_progress(
                    progress_callback,
                    "quantizing",
                    projection_start
                    + (99 - projection_start)
                    * written_bytes
                    / max(budget["remaining_weights"]["tensor_bytes"], 1),
                    f"Exported {i + 1} projection groups",
                )
    output_map = writer.finish()
    report["shard_count"] = len(set(output_map.values()))
    for spec in exported_tables.values():
        spec["weight_file"] = output_map[spec["weight_key"]]
        spec["scale_file"] = output_map[spec["scale_key"]]
    if written_bytes != budget["remaining_weights"]["tensor_bytes"]:
        raise ValueError("Exported non-Engram bytes differ from the independent budget")
    config.pop("quantization_config", None)
    if not preserve_mtp:
        strip_draft_config(config)
    if text_only:
        config.pop("vision_config", None)
    config["omlx_deepseek_v41"] = {
        "version": 1,
        "quantized_modules": quantized,
        "engram_tables": exported_tables,
        "engram_in_index": True,
        "preserve_mtp": preserve_mtp,
        "excluded_draft_tensors": (
            0 if preserve_mtp else sum(k.startswith("mtp.") for k in mapping)
        ),
        "quantization_report": report,
    }
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "LICENSE",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": output_map}, indent=2) + "\n"
    )
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "conversion.inprogress.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    (destination / "conversion.inprogress.json").rename(
        destination / "quantization_report.json"
    )
    _emit_progress(progress_callback, "complete", 100.0, "V4.1 export complete")


def prepare_affine_conversion(
    source,
    destination,
    config,
    mapping,
    budget,
    *,
    target,
    cap,
    preserve_mtp,
    sensitivity_model_path,
    sensitivity_map_override,
    imatrix_cache_path,
    imatrix_reuse_cache,
    imatrix_strict,
    imatrix_num_samples,
    imatrix_seq_length,
    progress_callback,
):
    from ...oq import (
        _OQE_CALIB_DATASET,
        _emit_progress,
        _load_or_collect_imatrix,
        _normalize_sensitivity_map_override,
        _source_imatrix_signature,
    )
    from .loading import load
    from .oq_inventory import build_affine_plan, projection_inventory
    from .sensitivity import measure_sensitivity

    if imatrix_num_samples < 1 or imatrix_seq_length < 1:
        raise ValueError(
            "V4.1 calibration sample count and sequence length must be positive"
        )
    calibration_source = (
        Path(sensitivity_model_path) if sensitivity_model_path else source
    )
    signed_config = dict(config)
    if calibration_source != source:
        proxy_config = json.loads((calibration_source / "config.json").read_text())
        if proxy_config.get("model_type") != "deepseek_v41" or (
            proxy_config.get("text_config") != config.get("text_config")
        ):
            raise ValueError(
                "V4.1 calibration proxy must have matching text architecture"
            )
        signed_config["_oq_calibration_proxy"] = _source_imatrix_signature(
            calibration_source,
            proxy_config,
            num_samples=imatrix_num_samples,
            seq_length=imatrix_seq_length,
            calib_dataset=_OQE_CALIB_DATASET,
        )
    if not imatrix_cache_path:
        imatrix_cache_path = str(
            destination.parent
            / ".oqe_imatrix"
            / f"{source.name}-oQe-s{imatrix_num_samples}-l{imatrix_seq_length}.npz"
        )
    imatrix = _load_or_collect_imatrix(
        str(source),
        signed_config,
        cache_path=imatrix_cache_path,
        reuse_cache=imatrix_reuse_cache,
        num_samples=imatrix_num_samples,
        seq_length=imatrix_seq_length,
        strict=imatrix_strict,
        trust_remote_code=False,
        progress_callback=progress_callback,
        progress_start=1,
        progress_end=10,
        load_path_factory=lambda: str(calibration_source),
    )
    coverage = imatrix.metadata.get("expert_coverage", {})
    sensitivity_config = {
        **config,
        "num_hidden_layers": config.get("text_config", config).get("num_hidden_layers")
        or config.get("text_config", config).get("n_layers", 0),
    }
    sensitivity = _normalize_sensitivity_map_override(
        sensitivity_config, sensitivity_map_override
    )
    if sensitivity is None:
        _emit_progress(
            progress_callback, "sensitivity", 10, "Measuring V4.1 layer distortion"
        )
        model, processor = load(calibration_source, engram_ssd_offload=True)
        try:

            def progress(phase, done, total):
                start, span = (10, 3) if phase == "capture" else (13, 7)
                _emit_progress(
                    progress_callback,
                    "sensitivity",
                    start + span * done / total,
                    f"V4.1 {phase} {done}/{total}",
                )

            sensitivity = measure_sensitivity(
                model, processor.tokenizer, bits=3, progress=progress
            )
        finally:
            model.close()
            del model, processor
            mx.synchronize()
            mx.clear_cache()
    inventory = projection_inventory(source, config, mapping, preserve_mtp=preserve_mtp)
    allocation, report = build_affine_plan(
        inventory,
        budget,
        config,
        imatrix,
        sensitivity,
        target=target,
        cap=cap,
    )
    report.update(
        imatrix_cache_path=imatrix.path,
        imatrix_cache_reused=imatrix.reused,
        expert_coverage=coverage,
        sensitivity=sensitivity,
        calibration_model=str(calibration_source),
        uncalibrated_policy="preserve_source_precision",
    )
    return allocation, imatrix, report
