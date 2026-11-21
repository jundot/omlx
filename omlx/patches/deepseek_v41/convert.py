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
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlx.core as mx
import numpy as np

from .sharding import ShardWriter
from .storage import TensorFile, decode_array

_FLOAT_DTYPES = frozenset({"BF16", "F16", "F32"})


def source_quantization_spec(config, path):
    """Resolve affine defaults and per-module overrides; False means dense.

    Callers validate tensor shapes against the logical module dimensions.
    """
    for section in (config.get("quantization"), config.get("quantization_config")):
        if not isinstance(section, dict):
            continue
        spec = {
            key: value
            for key, value in section.items()
            if not isinstance(value, (dict, bool))
        }
        entry = None
        for candidate in (path, "language_model." + path, "model." + path):
            if candidate in section:
                entry = section[candidate]
                break
        if entry is False:
            return None
        if isinstance(entry, dict):
            spec.update(entry)
        bits, mode = spec.get("bits"), spec.get("mode", "affine")
        group_size = spec.get("group_size")
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise ValueError(f"Source quantization declares no bit width: {path}")
        if mode != "affine":
            raise ValueError(f"Unsupported source quantization mode: {path}")
        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError(f"Affine source declares no group size: {path}")
        # mlx_lm affine conversions use float activations, without FP8 rounding.
        return {
            "bits": bits,
            "group_size": group_size,
            "mode": mode,
            "quantize_input": False,
        }
    return None


def _repack_affine(raw, scale, scale_dtype, bias, bias_dtype, spec, force_dense):
    """Repack an mlx_lm affine projection: packed U32 plus float metadata."""
    if scale is None or bias is None:
        raise ValueError("Affine weights require both scales and biases")
    if scale_dtype not in _FLOAT_DTYPES or bias_dtype != scale_dtype:
        raise ValueError("Affine scales and biases must share a float dtype")
    bits, group_size = spec["bits"], spec["group_size"]
    if raw.ndim != 2 or raw.dtype != np.dtype("<u4"):
        raise ValueError("Packed affine matrix must be rank two with U32 elements")
    width = raw.shape[-1] * 32 // bits
    if width % group_size:
        raise ValueError("Packed row width is not a multiple of the group size")
    expected = (raw.shape[0], width // group_size)
    if scale.shape != expected or bias.shape != expected:
        raise ValueError("Unexpected affine metadata shape")
    weight = mx.array(raw.view(np.uint8).copy().view("<u4"))
    if force_dense:
        value = mx.dequantize(
            weight,
            decode_array(scale, scale_dtype),
            decode_array(bias, bias_dtype),
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        return {"weight": value.astype(mx.bfloat16)}, None
    return (
        {
            "weight": weight,
            "scales": decode_array(scale, scale_dtype),
            "biases": decode_array(bias, bias_dtype),
        },
        spec,
    )


def repack_weight(
    raw,
    dtype,
    scale,
    scale_dtype,
    force_dense=False,
    *,
    bias=None,
    bias_dtype=None,
    spec=None,
):
    """Losslessly repack E4M3/E2M1 bytes, E8M0 scales, or affine metadata."""
    if dtype == "U32":
        if spec is None:
            raise ValueError("Packed weight without a declared quantization spec")
        return _repack_affine(
            raw, scale, scale_dtype, bias, bias_dtype, spec, force_dense
        )
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


def source_engram_tables(mapping, config):
    tables = {}
    for key, filename in mapping.items():
        if ".engram.embed." not in key or not key.endswith(".weight"):
            continue
        prefix = key.rsplit(".", 1)[0]
        scale_key = prefix + ".scale"
        if scale_key not in mapping:
            scale_key = prefix + ".scales"
        bias_key = prefix + ".biases" if prefix + ".biases" in mapping else None
        spec = source_quantization_spec(config, prefix) if bias_key else None
        if bias_key:
            if spec is None:
                raise ValueError(
                    f"Packed Engram table without a declared quantization: {prefix}"
                )
            if mapping[bias_key] != mapping[scale_key]:
                raise ValueError(f"Engram metadata must share one shard: {prefix}")
        tables[mapped(prefix)] = {
            "weight_key": key,
            "weight_file": filename,
            "scale_key": scale_key if scale_key in mapping else None,
            "scale_file": mapping.get(scale_key),
            "bias_key": bias_key,
            "bits": spec["bits"] if spec else None,
            "group_size": spec["group_size"] if spec else None,
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


# A source checkpoint stores one small tensor pair per expert, so the expert
# walk is hundreds of synchronous preads per projection and dominates load
# time. Four workers match the SSD concurrency peak measured on M3 Ultra; the
# window bounds how many raw expert tensors may sit in memory at once.
LOAD_IO_WORKERS = 4
LOAD_READ_AHEAD = 16
_PROJECTION_FIELDS = ("weight", "scales", "biases", "scale")


def projection_keys(mapping, key):
    """Sibling tensor keys that make up one projection, weight first."""
    prefix = key.removesuffix(".weight")
    return [
        prefix + "." + field
        for field in _PROJECTION_FIELDS
        if prefix + "." + field in mapping
    ]


class ProjectionReadAhead:
    """Fetch a projection's expert tensors ahead on worker threads.

    Workers keep private TensorFile handles because TensorFile.read holds a
    per-instance lock, and every handle is closed on exit -- before the caller
    yields -- so the release-mappings-before-yield invariant still holds.
    Workers=1 reads synchronously, which is the unmodified behaviour.
    """

    def __init__(self, source, mapping, keys, lookaside, workers, depth):
        self._source = source
        self._mapping = mapping
        self._keys = list(keys)
        self._lookaside = lookaside
        self._depth = max(1, depth)
        self._workers = max(1, workers)
        self._pool = None
        self._pending = deque()
        self._next = 0
        self._handles = []
        self._handles_lock = threading.Lock()
        self._local = threading.local()

    def _read(self, key):
        handles = getattr(self._local, "handles", None)
        if handles is None:
            handles = self._local.handles = {}
        filename = self._mapping[key]
        if filename not in handles:
            handle = TensorFile(self._source / filename)
            with self._handles_lock:
                self._handles.append(handle)
            handles[filename] = handle
        return handles[filename].read(key)

    def _fetch(self, key):
        return {
            sibling: self._read(sibling)
            for sibling in projection_keys(self._mapping, key)
        }

    def __enter__(self):
        if self._workers > 1 and len(self._keys) > 1:
            self._pool = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="v41-load-io"
            )
            self._fill()
        return self

    def _fill(self):
        while self._next < len(self._keys) and len(self._pending) < self._depth:
            self._pending.append(self._pool.submit(self._fetch, self._keys[self._next]))
            self._next += 1

    def wait(self):
        """Block until the next expert is fetched, then publish it.

        The lookaside is keyed, so a caller that reads out of step with the
        window falls back to a synchronous read instead of a wrong tensor.
        """
        if self._pool is None:
            return
        self._lookaside.update(self._pending.popleft().result())
        self._fill()

    def close(self):
        try:
            if self._pool is not None:
                while self._pending:
                    self._pending.popleft().cancel()
                self._pool.shutdown(wait=True)
        finally:
            self._pool = None
            for handle in self._handles:
                handle.close()
            self._handles.clear()

    def __exit__(self, *_exc):
        self.close()
        return False


def iter_source_weights(source, config, mapping, *, preserve_mtp=False):
    """Repack one projection at a time for either direct loading or export."""
    source = Path(source)
    # Sorted mlx_lm indexes put biases before weights.
    # Claim metadata first so only its weight emits it.
    readers, consumed = (
        {},
        {key for key in mapping if key.endswith((".scales", ".biases", ".scale"))},
    )
    # Window of expert tensors already read by ProjectionReadAhead.
    prefetched = {}

    def read(key):
        if key in prefetched:
            return prefetched.pop(key)
        filename = mapping[key]
        if filename not in readers:
            readers[filename] = TensorFile(source / filename)
        return readers[filename].read(key)

    def matrix(key, force_dense=False):
        # Affine sources use scales/biases; official MXFP sources use scale.
        prefix = key.removesuffix(".weight")
        fields = {}
        for field in ("weight", "scales", "biases", "scale"):
            sibling = prefix + "." + field
            if sibling in mapping:
                fields[field] = read(sibling)
                consumed.add(sibling)
        consumed.add(key)
        raw, dtype = fields["weight"]
        spec = source_quantization_spec(config, prefix) if dtype == "U32" else None
        scale, scale_dtype = fields.get("scales") or fields.get("scale") or (None, None)
        bias, bias_dtype = fields.get("biases") or (None, None)
        # QuantizedProjection cannot retain a linear bias, so keep that module dense.
        return repack_weight(
            raw,
            dtype,
            scale,
            scale_dtype,
            force_dense or prefix + ".bias" in mapping,
            bias=bias,
            bias_dtype=bias_dtype,
            spec=spec,
        )

    def release_readers():
        # Every tensor read owns its bytes. Drop source mappings before yielding
        # so their resident pages do not accumulate beside the loaded model.
        for reader in readers.values():
            reader.close()
        readers.clear()

    for table in source_engram_tables(mapping, config).values():
        consumed.add(table["weight_key"])
        if table["scale_key"]:
            consumed.add(table["scale_key"])
        if table["bias_key"]:
            consumed.add(table["bias_key"])
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
                keys = [
                    f"{base}.{expert}.{projection}.weight" for expert in range(count)
                ]
                values, spec = [], None
                with ProjectionReadAhead(
                    source, mapping, keys, prefetched, LOAD_IO_WORKERS, LOAD_READ_AHEAD
                ) as ahead:
                    for expert, key in enumerate(keys):
                        ahead.wait()
                        parts, current = matrix(key)
                        if expert and current != spec:
                            raise ValueError(
                                "Mixed expert formats within one projection"
                            )
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
    tables = source_engram_tables(mapping, config)
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
