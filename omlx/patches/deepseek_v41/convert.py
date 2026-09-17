# SPDX-License-Identifier: MIT
"""Stream the official checkpoint into MLX MXFP shards plus mmap Engram tables.

Usage: python -m omlx.patches.deepseek_v41.convert --hf-path SRC --mlx-path DST
The source stays immutable. Engram files are hardlinked when possible, copied
otherwise. Use --preserve-mtp to retain all embedded DSpark stages.
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np

from .sharding import ShardWriter
from .storage import TensorFile, decode_array


def repack_weight(raw, dtype, scale, scale_dtype, force_dense=False):
    """Losslessly repack E4M3/E2M1 bytes and E8M0 scales for MLX QMM."""
    if dtype.startswith("F8_E4M3") or dtype in ("I8", "U8"):
        if scale is None or not scale_dtype.startswith("F8_E8M0"):
            raise ValueError("Quantized weights require published E8M0 scales")
        if np.any(scale == 255):
            raise ValueError("NaN E8M0 scale")
        bits = 8 if dtype.startswith("F8_E4M3") else 4
        if raw.ndim != 2 or raw.shape[-1] % 4:
            raise ValueError("Packed matrix must be rank two with 4-byte-aligned rows")
        scales = scale
        if bits == 8:
            scales = np.repeat(scale, 32, axis=0)[: raw.shape[0]]
        width = raw.shape[-1] * (8 // bits)
        if scales.shape != (raw.shape[0], width // 32):
            raise ValueError("Unexpected block scale shape")
        weight = mx.array(raw.view(np.uint8).copy().view("<u4"))
        values = {"weight": weight, "scales": mx.array(scales)}
        spec = {"bits": bits, "mode": f"mxfp{bits}"}
        if force_dense:
            value = mx.dequantize(weight, values["scales"], group_size=32, **spec)
            return {"weight": value.astype(mx.bfloat16)}, None
        return values, spec
    if scale is not None:
        raise ValueError("Unexpected scale attached to a floating weight")
    return {"weight": decode_array(raw, dtype)}, None


def mapped(key):
    if key.startswith(("vision.", "aligner.", "image_")):
        return key.replace(".mlp.", ".ffn.") if key.startswith("vision.") else key
    return "language_model." + key


def source_engram_tables(mapping):
    tables = {}
    for key, filename in mapping.items():
        if ".engram.embed." not in key or not key.endswith(".weight"):
            continue
        prefix = key.rsplit(".", 1)[0]
        scale_key = prefix + ".scale"
        tables[mapped(prefix)] = {
            "weight_key": key,
            "weight_file": filename,
            "scale_key": scale_key if scale_key in mapping else None,
            "scale_file": mapping.get(scale_key),
        }
    return tables


def strip_draft_config(config):
    """An export without draft weights must not advertise DSpark stages."""
    for values in (config, config.get("text_config", {})):
        for key in ("n_mtp_layers", "num_nextn_predict_layers", "dspark_block_size"):
            if key in values:
                values[key] = 0
        if "dspark_target_layer_ids" in values:
            values["dspark_target_layer_ids"] = []


def iter_source_weights(source, config, mapping, *, preserve_mtp=False):
    """Repack one projection at a time for either direct loading or export."""
    source = Path(source)
    readers, consumed = {}, set()

    def read(key):
        filename = mapping[key]
        if filename not in readers:
            readers[filename] = TensorFile(source / filename)
        return readers[filename].read(key)

    def matrix(key, force_dense=False):
        raw, dtype = read(key)
        scale_key = key.removesuffix(".weight") + ".scale"
        scale, sd = read(scale_key) if scale_key in mapping else (None, None)
        consumed.add(key)
        if scale_key in mapping:
            consumed.add(scale_key)
        return repack_weight(raw, dtype, scale, sd, force_dense)

    def release_readers():
        # Every tensor read owns its bytes. Drop source mappings before yielding
        # so their resident pages do not accumulate beside the loaded model.
        for reader in readers.values():
            reader.close()
        readers.clear()

    for table in source_engram_tables(mapping).values():
        consumed.add(table["weight_key"])
        if table["scale_key"]:
            consumed.add(table["scale_key"])
    try:
        for key in mapping:
            if key in consumed or (key.startswith("mtp.") and not preserve_mtp):
                continue
            match = re.match(
                r"((?:layers|mtp)\.\d+\.ffn\.experts)\.(\d+)\.(w[123])\.weight$", key
            )
            if match:
                base, _, projection = match.groups()
                text_config = config.get("text_config", config)
                count = text_config["n_routed_experts"]
                if base.startswith("mtp."):
                    count = text_config.get("dspark_n_routed_experts") or count
                values, spec = [], None
                for expert in range(count):
                    parts, current = matrix(f"{base}.{expert}.{projection}.weight")
                    if expert and current != spec:
                        raise ValueError("Mixed expert formats within one projection")
                    spec = current
                    values.append(parts)
                prefix = mapped(f"{base}.{projection}")
                tensors = {
                    prefix + "." + name: mx.stack([v[name] for v in values])
                    for name in values[0]
                }
                del values
                release_readers()
                yield tensors, {prefix: spec} if spec else {}
                del tensors
            elif key.endswith(".weight"):
                force_dense = key.endswith(
                    (
                        "wo_a.weight",
                        ".markov_head.embed.weight",
                        ".markov_head.head.weight",
                    )
                ) or key in (
                    "head.weight",
                    "embed.weight",
                )
                values, spec = matrix(key, force_dense)
                prefix = mapped(key.removesuffix(".weight"))
                release_readers()
                yield (
                    {prefix + "." + name: value for name, value in values.items()},
                    {prefix: spec} if spec else {},
                )
                del values
            elif key.endswith(".scale"):
                continue
            else:
                raw, dtype = read(key)
                value = decode_array(raw, dtype)
                release_readers()
                yield {mapped(key): value}, {}
                del value
                consumed.add(key)
                del raw
        leftover = (
            set(mapping)
            - consumed
            - {k for k in mapping if k.startswith("mtp.") and not preserve_mtp}
        )
        if leftover:
            raise ValueError(f"Unconverted target tensors: {sorted(leftover)[:10]}")
    finally:
        release_readers()


def convert(source, destination, *, preserve_mtp=False):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Conversion output already exists: {destination}")
    config = json.loads((source / "config.json").read_text())
    if config.get("model_type") != "deepseek_v41":
        raise ValueError("Expected the official deepseek_v41 checkpoint")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    destination.mkdir(parents=True)
    (destination / "engram").mkdir()
    tables = source_engram_tables(mapping)
    for table in tables.values():
        for label in ("weight_file", "scale_file"):
            filename = table.get(label)
            if filename is None:
                continue
            target = destination / "engram" / filename
            if not target.exists():
                try:
                    os.link(source / filename, target)
                except OSError:
                    shutil.copyfile(source / filename, target)
            table[label] = str(target.relative_to(destination))
    writer, quantized = ShardWriter(destination), {}
    for values, specs in iter_source_weights(
        source, config, mapping, preserve_mtp=preserve_mtp
    ):
        writer.add(values)
        quantized.update(specs)
    output_map = writer.finish()
    config.pop("quantization_config", None)
    if not preserve_mtp:
        strip_draft_config(config)
    config["omlx_deepseek_v41"] = {
        "version": 1,
        "quantized_modules": quantized,
        "engram_tables": tables,
        "preserve_mtp": preserve_mtp,
        "excluded_draft_tensors": (
            0 if preserve_mtp else sum(k.startswith("mtp.") for k in mapping)
        ),
    }
    for name in ("tokenizer.json", "tokenizer_config.json", "LICENSE"):
        if (source / name).is_file():
            shutil.copyfile(source / name, destination / name)
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (destination / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": output_map}, indent=2) + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    parser.add_argument("--preserve-mtp", action="store_true")
    args = parser.parse_args()
    convert(args.hf_path, args.mlx_path, preserve_mtp=args.preserve_mtp)
