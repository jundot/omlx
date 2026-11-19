# SPDX-License-Identifier: Apache-2.0
"""Convert the pinned public Tencent Hy4-flash BF16 instruct snapshot, offline.

    omlx hy4-flash convert --source SNAPSHOT --dry-run
    omlx hy4-flash convert --source SNAPSHOT --output NEW_DIR [--chunk-mib 8] [--bits 8]
                           [--trunk-bits 8]

Dry-run reads only config, index and headers. With --headers DIR it accepts
standalone <shard>.safetensors.header files (8-byte length + JSON, no payload).
No dry-run authenticates weight payloads. Build reads every original shard,
including the omitted vision/drafter and the four unchanged external tables.
It authenticates the ordered inventory of complete official LFS hashes before
publishing. It never opens sources for write or downloads model code/weights.

By default the sole lossy step is uncalibrated, group-affine RTN in the
signed oQh basis (see quant_io.quantize_rows), 8-bit; --bits 4|5|6 writes
smaller banks with the same rotation and groups. Neither activation data,
Hessians nor GPTQ are used. Only the two released 3-D routed bank names in
each sparse block qualify. The text trunk is byte-exact unless --trunk-bits 8
is given: then the dense attention/MLP projections and the LM head
(quant_io.TRUNK_LINEAR) are also stored as plain MLX group-64 affine RTN, the
same as nn.quantize. Vision and DFly are omitted, not silently downgraded.
The four dedicated BF16 Engram files stay in the source or an independently
verified store: the artifact contains portable locators, not copies or an
implicit memory placement policy.

One source and at most one destination shard open at a time, one row-aligned
chunk on MLX; metadata scales with key count. A chunk bound is not a process
RSS limit. MLX's free allocation cache is disabled for the packing loop. Core
shards are read back before publishing the entire new directory atomically.
The manifest is a local integrity record, not a signed distribution attestation.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from ...oqh import format as fmt
from .quant_io import (
    BITS,
    BLOCK,
    GROUP,
    MAX_HEADER,
    SUPPORTED_BITS,
    SUPPORTED_TRUNK_BITS,
    TRUNK_GROUP,
    TRUNK_LINEAR,
    Spec,
    View,
    Writer,
    bank_specs,
    chunks,
    read_header,
    require,
    trunk_specs,
    unique,
    write_bank,
    write_trunk,
)

REPO = "tencent/Hy4-flash"
REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
TRANSFORMERS_COMMIT = "c2ec850510a37978ea83df226589623d5f2be776"
SCHEMA = "hy4-flash-oqh/v1"
INDEX = "model.safetensors.index.json"
TABLE_END = ".engram_layer.multi_head_embedding.embedding.weight"
# Git blob IDs from /api/models/{REPO}/revision/{REVISION}?blobs=true.
ASSET_BLOBS = {
    "config.json": "53b66f45537b2220dc882deed899b26a1b5356ab",
    INDEX: "20a1caaebc69cf0805157c2af1de6084bf9dae49",
    "LICENSE": "2be6f88597077cc42c495586eaf4fb6c6587e33c",
    "tokenizer.json": "ff6aaf29542a31a718ffc2986100c3ac195f223d",
    "tokenizer_config.json": "59ac77396a0aa48b3e4192911f76a0160561cb52",
    "chat_template.jinja": "8bb8ebcdf8b687c0edd5c5a746f38ab4a98efc9b",
    "generation_config.json": "aa39360634ecaa9bc165a18279aa3200f6e145f9",
}
# SHA256 of canonical JSON (see canonical_sha256), reproducible from the
# revision's range headers and Hub LFS metadata, respectively:
# {tensor: [shard_basename, absolute_byte_offset, dtype, shape]},
# {shard_basename: {"size": whole_file_bytes, "sha256": official_lfs_oid}}.
LAYOUT_SHA256 = "73967e9218c11d0579a22843cd1a26e9596a6808fb3fb086f9164efaada5daae"
LFS_INVENTORY_SHA256 = (
    "86ec55f73cdd605d61bcbded1a81236d20068fb65a4f96304bb20165508cc521"
)
ALGORITHM = {
    "method": "signed-wht-affine-rtn",
    "calibrated": False,
    "bits": BITS,
    "block": BLOCK,
    "group_size": GROUP,
    "sign_key": "full source bank path",
    "transform_dtype": "F32",
    "affine_input_dtype": "F32",
    "packed_dtype": "U32",
    "stored_scale_bias_dtype": "BF16",
}


def canonical_sha256(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def asset_bytes(src, name):
    with (src / name).open("rb") as f:
        st = os.fstat(f.fileno())
        require(
            stat.S_ISREG(st.st_mode) and st.st_size <= MAX_HEADER,
            f"invalid asset size/type: {name}",
        )
        data = f.read(st.st_size + 1)
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    require(blob == ASSET_BLOBS[name], f"source Git blob mismatch: {name}")
    return data


def policy(text):
    layers = text["mlp_layer_types"]
    require(
        len(layers) == text["num_hidden_layers"] and set(layers) <= {"dense", "sparse"},
        "invalid MLP topology",
    )
    e, h, m = (
        text["n_routed_experts"],
        text["routed_expert_hidden_size"],
        text["moe_intermediate_size"],
    )
    banks = {}
    for i, kind in enumerate(layers):
        if kind == "sparse":
            prefix = f"model.layers.{i}.mlp.experts."
            banks[prefix + "gate_up_proj"] = Spec("BF16", (e, 2 * m, h))
            banks[prefix + "down_proj"] = Spec("BF16", (e, h, m))
    tables = {}
    heads, dim = text["engram_n_head_per_ngram"], text["engram_n_embed_per_ngram"]
    for number in text["engram_layer_ids"]:  # one-based in config; zero-based on wire
        vocab = text["engram_layer_head_vocab_sizes"][str(number)]
        require(
            0 < number <= len(layers)
            and heads > 0
            and dim % heads == 0
            and all(len(v) == heads for v in vocab),
            "invalid Engram geometry",
        )
        tables[f"model.layers.{number - 1}{TABLE_END}"] = (
            number - 1,
            Spec("BF16", (sum(map(sum, vocab)), dim // heads)),
        )
    return banks, tables


def trunk_record(bits, modules):
    return {
        "method": "mlx-affine-rtn",
        "calibrated": False,
        "bits": bits,
        "group_size": TRUNK_GROUP,
        "mode": "affine",
        "stored_scale_bias_dtype": "BF16",
        "modules": sorted(modules),
    }


def disposition(key, spec, banks, tables, bits=BITS, trunk_bits=None):
    if key in banks:
        require(spec == banks[key], f"wrong routed bank dtype/shape: {key}")
        return "routed", bank_specs(key, spec, bits)
    if key in tables:
        require(spec == tables[key][1], f"wrong external table dtype/shape: {key}")
        return "external", {}
    if key.startswith("vit."):
        return "omitted_vision", {}
    if key.startswith("model.dfly_layers."):
        return "omitted_drafter", {}
    require(
        ".mlp.experts." not in key and not key.endswith(TABLE_END),
        f"unexpected bank/table: {key}",
    )
    require(
        key.startswith(("model.layers.", "model.hc_head."))
        or key in {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"},
        f"unexpected source namespace: {key}",
    )
    if trunk_bits and TRUNK_LINEAR.fullmatch(key):
        return "trunk", trunk_specs(key, spec, trunk_bits)
    return "native", {key: spec}


def stamp(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


@dataclass
class Plan:
    text: dict
    banks: dict[str, Spec]
    tables: dict[str, tuple[int, Spec]]
    inputs: dict[str, dict[str, View]]
    stamps: dict
    outputs: dict[str, dict[str, Spec]]  # keyed by originating source shard
    table_records: dict
    counts: dict
    bits: int = BITS
    trunk_bits: int | None = None
    trunk: dict[str, Spec] | None = None

    def trunk_record(self):
        modules = [k.removesuffix(".weight") for k in self.trunk or {}]
        return trunk_record(self.trunk_bits, modules) if self.trunk_bits else None

    def quantization(self):
        q = dict(ALGORITHM, bits=self.bits)
        if self.trunk_bits:
            q["trunk"] = self.trunk_record()
        return q

    def audit(self):
        return {
            "schema": SCHEMA,
            "text_only": True,
            "source": {"repo": REPO, "revision": REVISION},
            "source_layout_sha256": LAYOUT_SHA256,
            "payloads_verified": False,
            "counts": self.counts,
            "quantization": self.quantization(),
            "routed_source_shapes": {k: list(s.shape) for k, s in self.banks.items()},
            "external_tables": self.table_records,
        }


def inspect_source(src, headers=None, bits=BITS, trunk_bits=None):
    """Header-only plan. The CLI has no option to substitute source pins."""
    require(bits in SUPPORTED_BITS, f"unsupported routed bank bits: {bits}")
    require(
        trunk_bits is None or trunk_bits in SUPPORTED_TRUNK_BITS,
        f"unsupported trunk bits: {trunk_bits}",
    )
    cfg = json.loads(asset_bytes(src, "config.json"), object_pairs_hook=unique)
    index = json.loads(asset_bytes(src, INDEX), object_pairs_hook=unique)
    text = cfg["text_config"]
    require(
        cfg["model_type"] == text["model_type"] == "hy_v4_flash"
        and cfg["dtype"] == "bfloat16"
        and not any(
            k in obj
            for k in ("quantization", "quantization_config", fmt.CONFIG_KEY)
            for obj in (cfg, text)
        ),
        "expected original BF16 Flash config",
    )
    banks, tables = policy(text)
    mapping = index["weight_map"]
    require(
        (banks.keys() | tables.keys()) <= mapping.keys(),
        "missing expected routed bank/table",
    )
    groups = defaultdict(set)
    for key, name in mapping.items():
        require(
            re.fullmatch(r"model-[0-9]{5}-of-[0-9]{5}[.]safetensors", name),
            f"invalid source shard name: {name}",
        )
        groups[name].add(key)
    root, suffix = (src, "") if headers is None else (headers, ".header")
    actual = {p.name for p in root.glob("*.safetensors" + suffix)}
    require(
        actual == {n + suffix for n in groups},
        "missing or unexpected source shards/headers",
    )
    inputs, stamps, outputs, records, basis = {}, {}, {}, {}, {}
    out_keys, counts, sizes, trunk = set(), Counter(), Counter(), {}
    for name, keys in sorted(groups.items()):
        with (root / (name + suffix)).open("rb", buffering=0) as f:
            stamps[name] = stamp(os.fstat(f.fileno()))
            views = read_header(f, detached=headers is not None)
        require(views.keys() == keys, f"source index/header mismatch: {name}")
        inputs[name] = views
        shard_out = {}
        for k, v in views.items():
            s = v.spec
            basis[k] = [name, v.start, s.dtype, s.shape]
            action, out = disposition(k, s, banks, tables, bits, trunk_bits)
            if action == "trunk":
                trunk[k] = s
            require(not out_keys.intersection(out), f"duplicate output key for {k}")
            out_keys.update(out)
            counts[action] += 1
            sizes[action] += s.nbytes
            if action == "external":
                require(
                    len(views) == 1, f"Engram table is not in a dedicated shard: {k}"
                )
                block = tables[k][0]
                records[str(block)] = {
                    "block": block,
                    "source_key": k,
                    "source_shard": name,
                    "source_offset": v.start,
                    "dtype": s.dtype,
                    "shape": list(s.shape),
                    "row_stride": 2 * s.shape[-1],
                    "byte_length": s.nbytes,
                    "file_size": v.start + s.nbytes,
                }
            shard_out.update(out)
        if shard_out:
            outputs[name] = shard_out
    if trunk_bits:
        require(trunk, "no trunk linears matched")
        sizes["native"] += sizes.pop("trunk")
    total = sum(sizes.values())
    require(
        total == index["metadata"]["total_size"], "source index total_size mismatch"
    )
    require(
        canonical_sha256(basis) == LAYOUT_SHA256,
        "source tensor layout differs from pinned revision",
    )
    counts = {
        "source_shards": len(inputs),
        "source_tensors": len(mapping),
        "source_payload_bytes": total,
        "routed_banks": counts["routed"],
        "routed_source_bytes": sizes["routed"],
        "native_tensors": counts["native"] + counts["trunk"],
        "native_bytes": sizes["native"],
        "external_tables": counts["external"],
        "external_bytes": sizes["external"],
        "omitted_vision": counts["omitted_vision"],
        "omitted_drafter": counts["omitted_drafter"],
        "omitted_bytes": sizes["omitted_vision"] + sizes["omitted_drafter"],
        "core_shards": len(outputs),
        "core_tensors": len(out_keys),
        "core_payload_bytes": sum(
            s.nbytes for part in outputs.values() for s in part.values()
        ),
    }
    if trunk_bits:
        counts["trunk_linears"] = len(trunk)
        counts["trunk_source_bytes"] = sum(s.nbytes for s in trunk.values())
    return Plan(
        text,
        banks,
        tables,
        inputs,
        stamps,
        outputs,
        records,
        counts,
        bits,
        trunk_bits,
        trunk or None,
    )


def hashed(data_chunks, whole):
    for data in data_chunks:
        whole.update(data)
        yield data


def convert_shard(path, dest, plan, budget):
    name = path.name
    table_hashes = {}
    whole = hashlib.sha256()
    with (
        path.open("rb", buffering=0) as f,
        dest.open("x+b", buffering=0) if dest else nullcontext() as out,
    ):
        require(
            stamp(os.fstat(f.fileno())) == plan.stamps[name],
            f"source changed after audit: {name}",
        )
        views = plan.inputs[name]
        # Views are in payload order. Include the *actual* header in the LFS hash.
        for data in chunks(f, next(iter(views.values())).start, budget):
            whole.update(data)
        writer = Writer(out, plan.outputs[name]) if out else None
        for key, v in views.items():
            require(f.tell() == v.start, f"source position mismatch: {key}")
            limit = budget
            trunk = plan.trunk is not None and key in plan.trunk
            if key in plan.banks or trunk:
                stride = 2 * v.spec.shape[-1]
                limit = budget // stride * stride
            stream = hashed(chunks(f, v.spec.nbytes, limit), whole)
            if key in plan.banks:
                write_bank(key, v.spec, stream, writer, plan.bits)
            elif trunk and writer is not None:
                write_trunk(key, v.spec, stream, writer, plan.trunk_bits)
            elif key in plan.tables:
                digest = hashlib.sha256()
                for data in stream:
                    digest.update(data)
                table_hashes[str(plan.tables[key][0])] = digest.hexdigest()
            elif writer is not None and key in writer.views:
                for data in stream:
                    writer.append(
                        key, data
                    )  # no floating-point conversion, including F32/I32
            else:
                for _ in stream:  # omitted vision/drafter bytes are still authenticated
                    pass
        st = os.fstat(f.fileno())
        require(
            f.tell() == st.st_size and stamp(st) == plan.stamps[name],
            f"source changed during read: {name}",
        )
        source = {"size": st.st_size, "sha256": whole.hexdigest()}
        core = writer.finish(budget) if writer is not None else None
    return source, core, table_hashes


def json_bytes(obj):
    return (json.dumps(obj, sort_keys=True, indent=2) + "\n").encode()


def write_asset(path, data):
    with path.open("xb") as f:
        require(f.write(data) == len(data), f"short write: {path.name}")
        f.flush()
        os.fsync(f.fileno())
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def publish(stage, output):
    """Atomic directory rename that cannot clobber even a racing empty dir."""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        status = libc.renamex_np(
            os.fsencode(stage), os.fsencode(output), 4
        )  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        status = libc.renameat2(
            -100, os.fsencode(stage), -100, os.fsencode(output), 1
        )  # RENAME_NOREPLACE
    else:
        raise OSError("atomic no-replace directory rename not available")
    if status:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), str(output))


def build(src, output, *, budget=8 << 20, progress=None, bits=BITS, trunk_bits=None):
    src = src.resolve(strict=True)
    require(not os.path.lexists(output), "output already exists; no overwrite/resume")
    output = output.resolve()
    require(
        output.parent.is_dir() and not output.is_relative_to(src),
        "expected a new output outside source under an existing parent",
    )
    require(type(budget) is int and 0 < budget <= 64 << 20, "invalid input chunk size")
    plan = inspect_source(src, bits=bits, trunk_bits=trunk_bits)
    require(
        all(
            2 * s.shape[-1] <= budget
            for s in [*plan.banks.values(), *(plan.trunk or {}).values()]
        ),
        "chunk smaller than a BF16 row",
    )
    assets = {
        k: asset_bytes(src, k) for k in ASSET_BLOBS if k not in {"config.json", INDEX}
    }
    need = (
        plan.counts["core_payload_bytes"] + sum(map(len, assets.values())) + (16 << 20)
    )
    require(
        shutil.disk_usage(output.parent).free >= need,
        f"insufficient free space; need at least {need} bytes",
    )
    import mlx.core as mx

    config = dict(plan.text, dtype="bfloat16")
    config[fmt.CONFIG_KEY] = {
        "version": fmt.FORMAT_VERSION,
        "tensors": {k: fmt.config_entry(plan.bits, BLOCK, GROUP) for k in plan.banks},
    }
    if plan.trunk_bits:
        # Kept out of the shared oQh block: these are stock MLX QuantizedLinear tensors.
        config["trunk_quantization"] = plan.trunk_record()
    manifest = {
        "schema": SCHEMA,
        "text_only": True,
        "counts": plan.counts,
        "source": {
            "repo": REPO,
            "revision": REVISION,
            "asset_git_blobs": ASSET_BLOBS,
            "layout_sha256": LAYOUT_SHA256,
            "lfs_inventory_sha256": LFS_INVENTORY_SHA256,
            "shards": {},
        },
        "producer": {
            "mlx_version": version("mlx"),
            "transformers_reference_commit": TRANSFORMERS_COMMIT,
            "input_chunk_bytes": budget,
        },
        "quantization": plan.quantization(),
        "external_tables": plan.table_records,
        "core_shards": {},
        "assets": {},
        "inference_qualified": False,
    }
    # Never let a partial directory masquerade as a finished checkpoint. On a
    # hard crash only the private temp sibling may remain (no automatic resume).
    with tempfile.TemporaryDirectory(prefix=".hy4-convert-", dir=output.parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir(mode=0o700)
        file_map = {
            name: f"model-{i:05}-of-{len(plan.outputs):05}.safetensors"
            for i, name in enumerate(plan.outputs, 1)
        }
        weight_map = {}
        old_cache = mx.set_cache_limit(0)
        try:
            for name in plan.inputs:
                if progress:
                    progress(name)
                out_name = file_map.get(name)
                record, core, table_hashes = convert_shard(
                    src / name, stage / out_name if out_name else None, plan, budget
                )
                manifest["source"]["shards"][name] = record
                if out_name:
                    manifest["core_shards"][out_name] = core
                    weight_map.update(dict.fromkeys(plan.outputs[name], out_name))
                for block, digest in table_hashes.items():
                    manifest["external_tables"][block].update(
                        file_sha256=record["sha256"], payload_sha256=digest
                    )
        finally:
            mx.set_cache_limit(old_cache)
        require(
            canonical_sha256(manifest["source"]["shards"]) == LFS_INVENTORY_SHA256,
            "source LFS SHA-256 inventory differs from pinned revision",
        )
        for name in plan.inputs:
            require(
                stamp((src / name).stat()) == plan.stamps[name],
                f"source changed before publish: {name}",
            )
        assets["config.json"] = json_bytes(config)
        assets[INDEX] = json_bytes(
            {
                "metadata": {"total_size": plan.counts["core_payload_bytes"]},
                "weight_map": weight_map,
            }
        )
        for name, data in assets.items():
            manifest["assets"][name] = write_asset(stage / name, data)
        write_asset(stage / "manifest.json", json_bytes(manifest))
        publish(stage, output)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Stream the pinned BF16 Hy4 Flash snapshot to text-only oQh RTN."
    )
    p.add_argument(
        "--source", required=True, type=Path, help="local original BF16 snapshot"
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="audit all headers and predict output; read no payloads, write nothing",
    )
    mode.add_argument(
        "--output", type=Path, help="build into a NEW directory outside the snapshot"
    )
    p.add_argument(
        "--headers",
        type=Path,
        help="dry-run only: directory of detached .safetensors.header files",
    )
    p.add_argument(
        "--chunk-mib",
        type=int,
        default=8,
        choices=range(1, 65),
        metavar="1..64",
        help="source chunk budget, default 8 MiB; not a total memory limit",
    )
    p.add_argument(
        "--bits",
        type=int,
        default=BITS,
        choices=SUPPORTED_BITS,
        help="routed expert bank bits, default 8; 4 fits 128 GB Macs",
    )
    p.add_argument(
        "--trunk-bits",
        type=int,
        choices=SUPPORTED_TRUNK_BITS,
        help="also store the dense attention/MLP projections and LM head as MLX "
        "group-64 affine RTN; default keeps the trunk byte-exact BF16",
    )
    args = p.parse_args(argv)
    if args.headers is not None and not args.dry_run:
        p.error("--headers is only valid with --dry-run")
    try:
        if args.dry_run:
            result = inspect_source(
                args.source, args.headers, args.bits, args.trunk_bits
            ).audit()
        else:
            m = build(
                args.source,
                args.output,
                budget=args.chunk_mib << 20,
                bits=args.bits,
                trunk_bits=args.trunk_bits,
                progress=lambda n: print(n, file=sys.stderr, flush=True),
            )
            result = {
                "schema": SCHEMA,
                "counts": m["counts"],
                "source_revision": REVISION,
                "manifest": str(args.output / "manifest.json"),
            }
    except (OSError, ValueError, KeyError) as e:
        print(f"hy4-flash convert: {e}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
