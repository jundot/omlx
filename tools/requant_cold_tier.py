"""Requantize a streaming model's expert banks into a cold precision tier (Fase I5).

Reads a checkpoint's `switch_mlp` stacked banks (oQ4e etc.), dequantizes and
requantizes them at a lower bit width (same group size / mode), and writes a
parallel shard set under `<model>/expert_cold/` with the SAME shard filenames
and key names. At runtime, `expert_streaming_cold_tier` makes the backing
store route expert reads to those files — cutting 25% (3-bit) or half
(2-bit) of the bytes per token that pin decode to the NVMe's I/O floor.

Layout: the runtime (`cold_tier_status` + the cold readers) resolves keys by
scanning `expert_cold/` headers, so the tier does NOT need to mirror the
source shard layout tensor-for-tensor. This tool groups banks by their
WEIGHT's source shard (from model.safetensors.index.json), loads each
component (weight/scales/biases) from wherever the index says it lives, and
writes the requantized triple together into the output shard named after the
weight's source shard. This handles checkpoints whose packing splits a
bank's weight and scales/biases across different shards (the Qwen3.8 JANG
quants) — the previous same-shard requirement silently skipped them.

The requantized tensors keep the source group size; bits and group size are
recorded in each shard's `__metadata__` (`omlx_cold_bits` /
`omlx_cold_group_size`) so the runtime can build the gather_qmm without
guessing. Source bits/group size/mode come from config.json's `quantization`
block (per-tensor overrides honored, with or without the `.weight` suffix
on the override key — the JANG configs key them by module path). Only
affine-mode banks with a `.biases` key are converted — the affine bias term
must ride along or the runtime's dequantize would reconstruct shifted values.

Quality is NOT decided here — bench/ppl_expert_streaming.py (I4) is the
gate. 3-bit is the conservative default; 2-bit is the max-bytes option.

Usage:
    .venv/bin/python tools/requant_cold_tier.py \
        --model "/Volumes/SSD 4TB/AI Models/GLM-5.3-Flash-oQ4e" --bits 3
    # report what would be written (no writes)
    .venv/bin/python tools/requant_cold_tier.py --model ... --bits 3 --check
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import mlx.core as mx

EXPERT_BANK_MARKERS = (
    ".switch_mlp.gate_proj.",
    ".switch_mlp.up_proj.",
    ".switch_mlp.down_proj.",
    ".switch_mlp.gate_up_proj.",
)

META_BITS = "omlx_cold_bits"
META_GS = "omlx_cold_group_size"


def bank_prefixes_from_index(weight_map: dict[str, str]) -> set[str]:
    """Stacked-bank key prefixes (…switch_mlp.<proj>) that carry .weight keys."""
    prefixes: set[str] = set()
    for key in weight_map:
        if not key.endswith(".weight"):
            continue
        for marker in EXPERT_BANK_MARKERS:
            idx = key.find(marker)
            if idx > 0:
                prefixes.add(key[: idx + len(marker) - 1])
                break
    return prefixes


def component_map_from_index(
    prefixes: set[str], weight_map: dict[str, str]
) -> dict[str, dict[str, str]]:
    """prefix -> {weight|scales|biases: shard filename} from the global index.

    The index maps EVERY tensor (weights, scales, biases) to its shard, so a
    bank whose weight and scales live in different shards resolves fine —
    the runtime only needs the keys to exist in some expert_cold/ header.
    """
    comps: dict[str, dict[str, str]] = {}
    for p in prefixes:
        entry = {}
        for part in ("weight", "scales", "biases"):
            shard = weight_map.get(f"{p}.{part}")
            if shard:
                entry[part] = shard
        comps[p] = entry
    return comps


def _read_header(path: Path) -> dict:
    with path.open("rb") as f:
        hsize = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(hsize))


def quant_cfg_for(key: str, quant_cfg: dict) -> tuple[int, int, str]:
    """(group_size, bits, mode) for *key* — per-tensor override or defaults.

    Some checkpoints (the Qwen3.8 JANG quants) key their per-tensor
    quantization config by the tensor's *module path* (no `.weight`
    suffix) while the safetensor keys carry it — try both spellings so the
    real per-tensor bits are found instead of falling back to the config
    default (which would crash dequantize with a shape mismatch).
    """
    override = quant_cfg.get(key)
    if not isinstance(override, dict) and key.endswith(".weight"):
        override = quant_cfg.get(key[: -len(".weight")])
    if isinstance(override, dict):
        return (
            int(override.get("group_size", quant_cfg["group_size"])),
            int(override.get("bits", quant_cfg["bits"])),
            str(override.get("mode", quant_cfg.get("mode", "affine"))),
        )
    return (
        int(quant_cfg["group_size"]),
        int(quant_cfg["bits"]),
        str(quant_cfg.get("mode", "affine")),
    )


def requant_group(
    name: str,
    prefixes: list[str],
    comps: dict[str, dict[str, str]],
    model: Path,
    dst_dir: Path,
    quant_cfg: dict,
    bits: int,
    check_only: bool = False,
) -> dict:
    """Requantize the banks of one output shard (grouped by weight's shard)."""
    out_path = dst_dir / name
    # Only affine banks with all three components indexed are convertible.
    convertible = [
        p for p in prefixes
        if all(part in comps[p] for part in ("weight", "scales", "biases"))
        and quant_cfg_for(f"{p}.weight", quant_cfg)[2] == "affine"
    ]
    skipped = sorted(set(prefixes) - set(convertible))
    if skipped:
        return {
            "shard": name,
            "status": "skipped (banks without .bias or non-affine)",
            "src_mib": 0.0,
            "dst_mib": 0.0,
        }
    if not convertible:
        return {"shard": name, "status": "no expert banks", "src_mib": 0.0, "dst_mib": 0.0}

    gs0 = quant_cfg_for(f"{convertible[0]}.weight", quant_cfg)[0]
    if out_path.exists():
        try:
            old = _read_header(out_path)
            meta = old.get("__metadata__") or {}
            if (
                int(meta.get(META_BITS, -1)) == bits
                and int(meta.get(META_GS, -1)) == gs0
                and all(f"{p}.weight" in old for p in convertible)
            ):
                return {"shard": name, "status": "already matches", "src_mib": 0.0, "dst_mib": 0.0}
        except Exception:
            pass
    if check_only:
        return {"shard": name, "status": "missing (would write)", "src_mib": 0.0, "dst_mib": 0.0}

    # Load each DISTINCT source shard once (a group may pull components from
    # several files; a source shard may serve several groups — reload cost is
    # acceptable because groups are processed sequentially).
    src_shards = sorted({comps[p][part] for p in convertible for part in ("weight", "scales", "biases")})
    loaded: dict[str, dict] = {}
    for fname in src_shards:
        loaded[fname] = mx.load(str(model / fname))

    arrays: dict[str, mx.array] = {}
    src_bytes = dst_bytes = 0
    max_err = 0.0
    for prefix in convertible:
        w_key = f"{prefix}.weight"
        gs, src_bits, mode = quant_cfg_for(w_key, quant_cfg)
        w = loaded[comps[prefix]["weight"]][w_key]
        scales = loaded[comps[prefix]["scales"]][f"{prefix}.scales"]
        biases = loaded[comps[prefix]["biases"]][f"{prefix}.biases"]
        dense = mx.dequantize(w, scales, biases, group_size=gs, bits=src_bits)
        w2, s2, b2 = mx.quantize(dense, group_size=gs, bits=bits, mode=mode)
        err = mx.abs(mx.dequantize(w2, s2, b2, group_size=gs, bits=bits) - dense).max().item()
        max_err = max(max_err, err)
        arrays[w_key] = w2
        arrays[f"{prefix}.scales"] = s2.astype(mx.bfloat16)
        arrays[f"{prefix}.biases"] = b2.astype(mx.bfloat16)
        src_bytes += w.nbytes
        dst_bytes += w2.nbytes
        # Drop the dense/requant transients from the Metal buffer pool —
        # banks are multi-GiB and the pool would otherwise grow to OOM.
        mx.clear_cache()

    dst_dir.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(
        str(out_path),
        arrays,
        metadata={META_BITS: str(bits), META_GS: str(gs0)},
    )
    return {
        "shard": name,
        "status": "written",
        "banks": len(convertible),
        "max_requant_err": max_err,
        "src_mib": src_bytes / 2**20,
        "dst_mib": dst_bytes / 2**20,
    }


def requant_shard(
    src: Path,
    dst_dir: Path,
    quant_cfg: dict,
    bits: int,
    check_only: bool = False,
) -> dict:
    """Backcompat per-shard entry point (tests + older scripts).

    Converts every convertible bank whose WEIGHT lives in *src*, loading
    the scales/biases components from wherever the checkpoint index says
    they live (same relaxed layout rule as the global path).
    """
    model = src.parent
    index_path = model / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    else:
        # no index: single-shard checkpoint — banks live in src itself
        header = _read_header(src)
        weight_map = {k: src.name for k in header if k != "__metadata__"}
    prefixes = bank_prefixes_from_index(weight_map)
    comps = component_map_from_index(prefixes, weight_map)
    mine = [p for p in sorted(prefixes) if comps[p].get("weight") == src.name]
    return requant_group(
        src.name, mine, comps, model, dst_dir, quant_cfg, bits, check_only
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="checkpoint directory")
    ap.add_argument(
        "--bits", type=int, default=3, choices=(2, 3), help="cold tier bit width"
    )
    ap.add_argument("--check", action="store_true", help="report only; skip writes")
    ap.add_argument(
        "--out-dir",
        help="write the tier here instead of <model>/expert_cold (deploy the "
        "directory via OMLX_EXPERT_STREAMING_COLD_ROOT at runtime)",
    )
    args = ap.parse_args()

    model = Path(args.model).expanduser().resolve()
    cfg_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    if not index_path.is_file() or not cfg_path.is_file():
        print(
            "checkpoint needs config.json + model.safetensors.index.json",
            file=sys.stderr,
        )
        sys.exit(2)
    quant_cfg = json.loads(cfg_path.read_text()).get("quantization") or {}
    if not quant_cfg:
        print("config.json has no quantization block", file=sys.stderr)
        sys.exit(2)
    weight_map = json.loads(index_path.read_text()).get("weight_map") or {}
    prefixes = bank_prefixes_from_index(weight_map)
    if not prefixes:
        print("no switch_mlp expert banks found in the index", file=sys.stderr)
        sys.exit(2)
    comps = component_map_from_index(prefixes, weight_map)
    # Group by the WEIGHT's source shard: the output shard keeps that name.
    groups: dict[str, list[str]] = {}
    for p in sorted(prefixes):
        wshard = comps[p].get("weight")
        if wshard:
            groups.setdefault(wshard, []).append(p)
    print(
        f"{len(prefixes)} expert bank(s) across {len(groups)} shard(s); "
        f"cold bits = {args.bits}, check={'yes' if args.check else 'no'}"
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else model / "expert_cold"
    totals = {"src_mib": 0.0, "dst_mib": 0.0}
    for name in sorted(groups):
        res = requant_group(name, groups[name], comps, model, out_dir, quant_cfg, args.bits, args.check)
        totals["src_mib"] += res["src_mib"]
        totals["dst_mib"] += res["dst_mib"]
        extra = (
            f"  requant_err<={res['max_requant_err']:.4f}"
            if "max_requant_err" in res
            else ""
        )
        print(
            f"  {res['shard']}: {res['status']}"
            + (f" ({res['banks']} banks)" if "banks" in res else "")
            + extra
        )
    if totals["src_mib"]:
        ratio = totals["dst_mib"] / totals["src_mib"]
        print(
            f"expert banks: {totals['src_mib'] / 1024:.1f} GiB -> "
            f"{totals['dst_mib'] / 1024:.1f} GiB ({ratio:.2f}x)"
        )


if __name__ == "__main__":
    main()
