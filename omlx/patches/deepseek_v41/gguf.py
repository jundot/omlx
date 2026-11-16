# SPDX-License-Identifier: MIT
"""Native packed GGUF backbone with original safetensors Engram/vision/DSpark.

Create a small discoverable MLX folder (no weight copies):
python -m omlx.patches.deepseek_v41.gguf --gguf-path GGUF/UD-IQ2_XXS \
    --hf-path ORIGINAL --mlx-path DEST
The GGUF download may finish later; all backbone parts are checked at load.
"""

import argparse
import fcntl
import json
import math
import shutil
from pathlib import Path

from omlx.gguf import SPLIT_RE, GgufError, _Reader

from .config import ModelConfig

# GGML type: (logical block width, stored bytes, upstream decoder).
FORMATS = {
    8: (32, 34, "q8_0"),
    10: (256, 84, "q2_K"),
    13: (256, 176, "q5_K"),
    14: (256, 210, "q6_K"),
    16: (256, 66, "iq2_xxs"),
    17: (256, 74, "iq2_xs"),
    18: (256, 98, "iq3_xxs"),
    19: (256, 50, "iq1_s"),
    22: (256, 82, "iq2_s"),
}


def row_bytes(width, qtype):
    block, size, _ = FORMATS[qtype]
    if width <= 0 or width % block:
        raise ValueError(f"GGUF row width {width} is not divisible by {block}")
    return width // block * size


def dense_weight(name):
    return name in (
        "language_model.head.weight",
        "language_model.embed.weight",
    ) or name.endswith(".wo_a.weight")


def dual_weight(name):
    """Projections the fused decode kernels read packed and prefill reads dense."""
    return name.endswith(
        (
            ".attn.wq_a.weight",
            ".attn.wkv.weight",
            ".attn.wq_b.weight",
            ".attn.wo_b.weight",
            ".ffn.shared_experts.w1.weight",
            ".ffn.shared_experts.w3.weight",
            ".ffn.shared_experts.w2.weight",
        )
    )


_ENGRAM = "per_layer_token_embd.weight"
_ROOT = {
    "output.weight": "head.weight",
    "output_norm.weight": "norm.weight",
    "token_embd.weight": "embed.weight",
}
_LAYER = {
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "engram_kv.weight": "engram.wkv.weight",
    "engram_gate.weight": "engram.q_weight",
    "attn_norm.weight": "attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    **{
        f"hc_{role}_{field}.weight": f"hc_{role}_{field}"
        for role in ("attn", "ffn")
        for field in ("base", "fn", "scale")
    },
    **{
        f"ffn_{role}_{kind}.weight": f"ffn.{target}.w{n}.weight"
        for role, n in (("gate", 1), ("down", 2), ("up", 3))
        for kind, target in (("exps", "experts"), ("shexp", "shared_experts"))
    },
}


def target_name(name):
    if name in _ROOT:
        return "language_model." + _ROOT[name]
    parts = name.split(".", 2)
    if (
        len(parts) != 3
        or parts[0] != "blk"
        or not parts[1].isdigit()
        or parts[2] not in _LAYER
    ):
        raise GgufError(f"Unsupported DeepSeek V4.1 GGUF tensor: {name}")
    return f"language_model.layers.{parts[1]}." + _LAYER[parts[2]]


def read_header(path):
    """Read bounded headers only, and validate every retained tensor's byte range."""
    path = Path(path)
    with path.open("rb") as file:
        r = _Reader(file)
        if r.take(4) != b"GGUF" or r.scalar("<I") not in (2, 3):
            raise GgufError(f"Expected little-endian GGUF v2/v3: {path}")
        count, nmeta = r.length(1 << 20), r.length(1 << 20)
        meta = {}
        for _ in range(nmeta):
            key = r.string()
            if key in meta:
                raise GgufError(f"Duplicate GGUF metadata: {key}")
            meta[key] = r.value(r.scalar("<I"), keep=False)
        tensors = []
        for _ in range(count):
            name = r.string()
            ndim = r.scalar("<I")
            if not 1 <= ndim <= 4:
                raise GgufError(f"Invalid GGUF rank: {name}")
            shape = tuple(reversed([r.length(1 << 40) for _ in range(ndim)]))
            qtype, offset = r.scalar("<I"), r.scalar("<Q")
            tensors.append(
                dict(name=name, shape=shape, qtype=qtype, offset=offset, file=path)
            )
        alignment = meta.get("general.alignment", 32)
        if (
            not isinstance(alignment, int)
            or alignment <= 0
            or alignment > 4096
            or alignment & (alignment - 1)
        ):
            raise GgufError("Invalid GGUF alignment")
        start = (file.tell() + alignment - 1) // alignment * alignment
    end, names = start, set()
    for t in sorted(tensors, key=lambda t: t["offset"]):
        if t["name"] in names or not all(t["shape"]):
            raise GgufError(f"Invalid or duplicate GGUF tensor: {t['name']}")
        names.add(t["name"])
        t["offset"] += start
        # The replacement Engram table is never read, even if its shard exists.
        if t["name"] == _ENGRAM:
            continue
        qtype, shape = t["qtype"], t["shape"]
        if qtype in (0, 1, 30):
            size = math.prod(shape) * (4 if qtype == 0 else 2)
        elif qtype in FORMATS:
            size = math.prod(shape[:-1]) * row_bytes(shape[-1], qtype)
        else:
            raise GgufError(f"Unsupported GGML quantization type {qtype}: {t['name']}")
        if (
            t["offset"] < end
            or (t["offset"] - start) % alignment
            or t["offset"] + size > path.stat().st_size
        ):
            raise GgufError(
                f"Truncated or overlapping GGUF tensor: {t['name']} in {path.name}"
            )
        t["size"] = size
        end = t["offset"] + size
    return meta, tensors


def backbone_tensors(folder, config):
    """Only the final, table-only shard may be absent; backbone gaps stay errors."""
    files = sorted(Path(folder).glob("*.gguf"))
    if not files:
        raise FileNotFoundError(f"No GGUF parts downloaded in {folder}")
    headers = [read_header(file) for file in files]
    first = headers[0][0]
    count, total = first.get("split.count"), first.get("split.tensors.count")
    if not isinstance(count, int) or count < 2 or not isinstance(total, int):
        raise GgufError("Expected a split DeepSeek V4.1 GGUF checkpoint")
    parts, tensors = set(), []
    for file, (meta, items) in zip(files, headers):
        part = meta.get("split.no")
        match = SPLIT_RE.match(file.name)
        if (
            meta.get("general.architecture", "deepseek41") != "deepseek41"
            or meta.get("split.count") != count
            or meta.get("split.tensors.count") != total
            or not isinstance(part, int)
            or not 0 <= part < count
            or part in parts
            or not match
            or int(match["index"]) != part + 1
            or int(match["count"]) != count
            or match["prefix"] != SPLIT_RE.match(files[0].name)["prefix"]
        ):
            raise GgufError(f"Inconsistent GGUF split: {file.name}")
        parts.add(part)
        if part == count - 1:
            if [t["name"] for t in items] != [_ENGRAM]:
                raise GgufError(
                    "Final GGUF shard is not an Engram-only replacement shard"
                )
        else:
            if any(t["name"] == _ENGRAM for t in items):
                raise GgufError("Engram table must occupy its own final GGUF shard")
            tensors.extend(items)
    missing = set(range(count - 1)) - parts
    if missing:
        raise GgufError(
            f"Missing required GGUF backbone parts (one-based): {sorted(i + 1 for i in missing)}"
        )
    if first.get("general.architecture") != "deepseek41":
        raise GgufError("First GGUF part must declare the deepseek41 architecture")
    if len(tensors) + 1 != total or len({t["name"] for t in tensors}) != len(tensors):
        raise GgufError(
            "GGUF backbone tensor count mismatch; only one Engram tensor may be omitted"
        )
    for key, value in (
        ("block_count", config.n_layers),
        ("embedding_length", config.dim),
        ("expert_count", config.n_routed_experts),
        ("attention.head_count", config.n_heads),
        ("expert_feed_forward_length", config.moe_inter_dim),
    ):
        if first.get("deepseek41." + key) != value:
            raise GgufError(f"Original checkpoint disagrees with GGUF metadata: {key}")
    return tensors


def supplementary(mapping):
    """The main GGUF omits the vision tower and DSpark; retain their source precision."""
    return {
        k: v
        for k, v in mapping.items()
        if k.startswith(("vision.", "aligner.", "image_", "mtp."))
    }


def iter_weights(folder, source, raw, mapping, expected_shapes, preserve_mtp):
    import mlx.core as mx
    import numpy as np

    from .convert import iter_source_weights
    from .gguf_quant import dequantize
    from .storage import decode_array

    tensors = backbone_tensors(folder, ModelConfig.from_dict(raw))
    targets = set()
    for t in tensors:
        name = target_name(t["name"])
        if (
            name not in expected_shapes
            or t["shape"] != expected_shapes[name]
            or name in targets
        ):
            raise GgufError(
                f"Unexpected GGUF target or shape: {t['name']} -> {name} {t['shape']}"
            )
        targets.add(name)
        if t["name"].endswith(".engram_gate.weight"):
            targets.add(name.replace(".q_weight", ".k_weight"))
    extras = supplementary(mapping)
    extras = {
        k: v for k, v in extras.items() if preserve_mtp or not k.startswith("mtp.")
    }
    # Validate coverage before allocating hundreds of GiB; expert source keys
    # belong only to the supplementary MTP path and are consolidated there.
    required = {
        k
        for k in expected_shapes
        if not k.startswith(("vision.", "aligner.", "image_", "language_model.mtp."))
    }
    if targets != required:
        raise GgufError(
            f"Incomplete GGUF backbone; missing={sorted(required-targets)[:8]}, extra={sorted(targets-required)[:8]}"
        )
    for filename in set(extras.values()):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    for t in tensors:
        name, shape, qtype = target_name(t["name"]), t["shape"], t["qtype"]
        with t["file"].open("rb") as reader:
            if hasattr(fcntl, "F_NOCACHE"):
                fcntl.fcntl(reader.fileno(), fcntl.F_NOCACHE, 1)
            reader.seek(t["offset"])
            value = np.fromfile(reader, dtype=np.uint8, count=t["size"])
        if value.size != t["size"]:
            raise GgufError(f"GGUF tensor changed while loading: {t['name']}")
        spec = {}
        if qtype in (0, 1, 30):
            dtype, tag = {0: ("<f4", "F32"), 1: ("<f2", "F16"), 30: ("<u2", "BF16")}[
                qtype
            ]
            value = decode_array(value.view(dtype).reshape(shape), tag)
        else:
            value = mx.array(value.reshape(*shape[:-1], row_bytes(shape[-1], qtype)))
            if dense_weight(name):
                value = dequantize(value, qtype, shape)
            else:
                spec = {
                    name.removesuffix(".weight"): dict(
                        qtype=qtype,
                        input_dims=shape[-1],
                        dense=dual_weight(name),
                    )
                }
        values = {name: value}
        if t["name"].endswith(".engram_gate.weight"):
            values[name.replace(".q_weight", ".k_weight")] = mx.ones(shape, mx.float32)
        yield values, spec
        del value, values
    yield from iter_source_weights(source, raw, extras, preserve_mtp=preserve_mtp)


def prepare(folder, source, destination):
    """Write references, not copies; preparation works before GGUF files arrive."""
    from .convert import source_engram_tables

    folder, source, destination = map(
        lambda p: Path(p).expanduser().resolve(), (folder, source, destination)
    )
    raw = json.loads((source / "config.json").read_text())
    if (
        raw.get("model_type") != "deepseek_v41"
        or raw.get("omlx_deepseek_v41") is not None
    ):
        raise ValueError("Engram source must be the original DeepSeek V4.1 checkpoint")
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    tables = source_engram_tables(mapping, raw)
    config = ModelConfig.from_dict(raw)
    if set(tables) != {
        f"language_model.layers.{i}.engram.embed" for i in config.engram_layer_ids
    }:
        raise ValueError("Original checkpoint is missing Engram tables")
    for table in tables.values():
        for field in ("weight_file", "scale_file"):
            if table.get(field):
                file = source / table[field]
                if not file.is_file():
                    raise FileNotFoundError(file)
                table[field] = str(file)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)
    raw["omlx_deepseek_v41"] = dict(
        version=1, engram_tables=tables, gguf=dict(path=str(folder), source=str(source))
    )
    destination.mkdir(parents=True, exist_ok=False)
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "LICENSE",
    ):
        if (source / filename).is_file():
            shutil.copyfile(source / filename, destination / filename)
    (destination / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf-path", required=True)
    parser.add_argument("--hf-path", required=True)
    parser.add_argument("--mlx-path", required=True)
    args = parser.parse_args()
    print(prepare(args.gguf_path, args.hf_path, args.mlx_path))
