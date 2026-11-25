#!/usr/bin/env python3
"""Repack a converted Inkling 2 checkpoint for SSD expert offload.

Each routed expert's six tensors (``w13`` and ``w2`` weight, scales and
biases) are written as separate per-expert tensors, back to back, so expert
offload reads one miss with a single ``pread`` instead of six. The expert
bytes are copied unchanged. The file stays standard safetensors; the loader
stacks the per-expert tensors when offload is off.

Optionally, the token embedding is requantized (default 8 -> 4 bits), the
vocabulary head is requantized (off by default), and the BF16 vision tower
linears are quantized (default 8 bits, group size 32: within about half a
point of BF16's own feature error against FP32).

Usage
-----
    python scripts/repack_inkling2_experts.py \\
        --model ~/.omlx/models/mlx-community/Inkling-2-Mini-oQ2.5e-mtp \\
        --output ~/.omlx/models/local/Inkling-2-Mini-oQ2.5e-mtp-packed \\
        [--embed-bits 4] [--lm-head-bits 0] [--vision-bits 8] \\
        [--vision-group-size 32] [--layers-per-shard 8]

Pass 0 for any of the bit widths to keep those tensors as they are.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

_EXPERT_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(w13|w2)\.(weight|scales|biases)$"
)
# Vision linears that see their weight only through ``__call__``. The patch
# embedding casts its input to ``weight.dtype``, so it stays BF16.
_VISION_LINEAR_RE = re.compile(
    r"^vision_tower\..*(q_proj|k_proj|v_proj|out_proj|c_fc|c_proj|proj_in|proj_out)"
    r"\.weight$"
)
_EXPERT_FIELDS = [(p, f) for p in ("w13", "w2") for f in ("weight", "scales", "biases")]
_MX_TO_ST = {
    mx.uint32: "U32",
    mx.bfloat16: "BF16",
    mx.float16: "F16",
    mx.float32: "F32",
}
_ST_ITEMSIZE = {"U32": 4, "BF16": 2, "F16": 2, "F32": 4, "I32": 4, "U8": 1}
_GROUP = 64
_ROWS = 16384


class Source:
    """Raw byte access to a checkpoint's safetensors shards."""

    def __init__(self, model_dir: Path):
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        self.specs: dict[str, tuple[str, list[int], np.memmap, int, int]] = {}
        maps: dict[str, np.memmap] = {}
        for shard in sorted(set(index["weight_map"].values())):
            path = model_dir / shard
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                header = json.loads(f.read(n))
            maps[shard] = np.memmap(path, dtype=np.uint8, mode="r")
            for name, spec in header.items():
                if name == "__metadata__":
                    continue
                start, end = spec["data_offsets"]
                self.specs[name] = (
                    spec["dtype"],
                    spec["shape"],
                    maps[shard],
                    8 + n + start,
                    end - start,
                )

    def raw(self, name: str, start: int = 0, nbytes: int | None = None) -> memoryview:
        _, _, buf, base, size = self.specs[name]
        nbytes = size - start if nbytes is None else nbytes
        return memoryview(buf[base + start : base + start + nbytes])

    def mx(self, name: str) -> mx.array:
        dtype, shape, *_ = self.specs[name]
        np_dtype = {
            "U32": np.uint32,
            "BF16": np.uint16,
            "F16": np.float16,
            "F32": np.float32,
        }[dtype]
        a = mx.array(np.frombuffer(self.raw(name), dtype=np_dtype).reshape(shape))
        return a.view(mx.bfloat16) if dtype == "BF16" else a


def _requantize(src: Source, prefix: str, spec: dict, bits: int) -> dict:
    """Dequantize an affine tensor and quantize it again at ``bits``, by rows."""
    w, s, b = (src.mx(f"{prefix}.{f}") for f in ("weight", "scales", "biases"))
    parts: dict[str, list] = {"weight": [], "scales": [], "biases": []}
    for i in range(0, w.shape[0], _ROWS):
        rows = mx.dequantize(
            w[i : i + _ROWS],
            s[i : i + _ROWS],
            b[i : i + _ROWS],
            group_size=spec["group_size"],
            bits=spec["bits"],
            mode="affine",
        )
        q = mx.quantize(rows, group_size=_GROUP, bits=bits, mode="affine")
        mx.eval(q)
        for k, v in zip(("weight", "scales", "biases"), q):
            parts[k].append(
                np.array(v.view(mx.uint16) if v.dtype == mx.bfloat16 else v)
            )
    return {
        f"{prefix}.{k}": (_MX_TO_ST[t], np.concatenate(parts[k]))
        for k, t in (("weight", mx.uint32), ("scales", s.dtype), ("biases", b.dtype))
    }


def _quantize_dense(src: Source, name: str, bits: int, group: int) -> dict:
    w = src.mx(name)
    q = mx.quantize(w, group_size=group, bits=bits, mode="affine")
    mx.eval(q)
    prefix = name[: -len(".weight")]
    out = {}
    for k, v in zip(("weight", "scales", "biases"), q):
        st = _MX_TO_ST[v.dtype]
        out[f"{prefix}.{k}"] = (st, np.array(v.view(mx.uint16) if st == "BF16" else v))
    return out


def _write_shard(path: Path, tensors: list[tuple[str, str, list[int], object]]) -> int:
    """Write ``(name, dtype, shape, bytes-like)`` in the given order."""
    header: dict = {"__metadata__": {"format": "mlx"}}
    offset = 0
    for name, dtype, shape, data in tensors:
        n = memoryview(data).nbytes
        expect = int(np.prod(shape)) * _ST_ITEMSIZE[dtype]
        if n != expect:
            raise ValueError(f"{name}: {n} bytes for shape {shape} {dtype}")
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + n],
        }
        offset += n
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _, _, _, data in tensors:
            f.write(memoryview(data).cast("B"))
    os.replace(tmp, path)
    return offset


def repack(
    model_dir: Path,
    out_dir: Path,
    embed_bits: int,
    vision_bits: int,
    layers_per_shard: int,
    lm_head_bits: int = 0,
    vision_group: int = 32,
) -> None:
    config = json.loads((model_dir / "config.json").read_text())
    quant = config["quantization"]
    n_layers = config["text_config"]["num_hidden_layers"]
    n_experts = config["text_config"]["n_routed_experts"]
    src = Source(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    other: list = []
    overrides: dict[str, dict] = {}
    replaced: set[str] = set()
    for prefix, bits in (("model.embed", embed_bits), ("lm_head", lm_head_bits)):
        spec = quant.get(prefix) or quant
        if bits and bits != spec["bits"]:
            for name, (dtype, arr) in _requantize(src, prefix, spec, bits).items():
                other.append((name, dtype, list(arr.shape), arr))
                replaced.add(name)
            overrides[prefix] = {"group_size": _GROUP, "bits": bits, "mode": "affine"}
    if vision_bits:
        for name, (dtype, shape, *_) in src.specs.items():
            if dtype == "BF16" and len(shape) == 2 and _VISION_LINEAR_RE.match(name):
                for qname, (qdtype, arr) in _quantize_dense(
                    src, name, vision_bits, vision_group
                ).items():
                    other.append((qname, qdtype, list(arr.shape), arr))
                replaced.add(name)
                overrides[name[: -len(".weight")]] = {
                    "group_size": vision_group,
                    "bits": vision_bits,
                    "mode": "affine",
                }

    experts: dict[int, dict] = {}
    for name, (dtype, shape, *_) in src.specs.items():
        m = _EXPERT_RE.match(name)
        if m:
            if shape[0] != n_experts:
                raise ValueError(f"{name}: expected {n_experts} stacked experts")
            experts.setdefault(int(m[1]), {})[(m[2], m[3])] = name
        elif name not in replaced:
            other.append((name, dtype, shape, src.raw(name)))
    if sorted(experts) != list(range(n_layers)) or any(
        len(d) != len(_EXPERT_FIELDS) for d in experts.values()
    ):
        raise ValueError("Checkpoint is missing stacked Inkling 2 expert tensors")

    groups = [
        list(range(i, min(n_layers, i + layers_per_shard)))
        for i in range(0, n_layers, layers_per_shard)
    ]
    total = len(groups) + 1
    names = [f"model-{i:05d}-of-{total:05d}.safetensors" for i in range(1, total + 1)]
    weight_map: dict[str, str] = {}
    size = _write_shard(out_dir / names[0], other)
    weight_map.update({t[0]: names[0] for t in other})
    print(f"{names[0]}: {len(other)} tensors, {size / 2**30:.2f} GiB")

    for shard, layers in zip(names[1:], groups):
        tensors = []
        for layer in layers:
            d = experts[layer]
            for e in range(n_experts):
                for proj, field in _EXPERT_FIELDS:
                    name = d[(proj, field)]
                    dtype, shape, _, _, nbytes = src.specs[name]
                    slab = nbytes // n_experts
                    tensors.append(
                        (
                            f"model.layers.{layer}.mlp.experts.{e}.{proj}.{field}",
                            dtype,
                            shape[1:],
                            src.raw(name, e * slab, slab),
                        )
                    )
        size += (n := _write_shard(out_dir / shard, tensors))
        weight_map.update({t[0]: shard for t in tensors})
        print(f"{shard}: layers {layers[0]}-{layers[-1]}, {n / 2**30:.2f} GiB")

    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": size},
                "weight_map": dict(sorted(weight_map.items())),
            },
            indent=2,
        )
    )
    for key in ("quantization", "quantization_config"):
        if isinstance(config.get(key), dict):
            config[key].update(overrides)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    skip = {"config.json", "model.safetensors.index.json"}
    for f in model_dir.iterdir():
        if f.is_file() and f.name not in skip and not f.name.startswith("model-"):
            dst = out_dir / f.name
            if dst.exists():
                dst.unlink()
            # Hard-link only the large unchanged weights; small files (README,
            # tokenizer) are copied so editing them never touches the source.
            try:
                if f.suffix != ".safetensors":
                    raise OSError
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
    print(f"wrote {out_dir} ({size / 2**30:.2f} GiB of model weights)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--embed-bits", type=int, default=4)
    ap.add_argument("--lm-head-bits", type=int, default=0)
    ap.add_argument("--vision-bits", type=int, default=8)
    ap.add_argument("--vision-group-size", type=int, default=32)
    ap.add_argument("--layers-per-shard", type=int, default=8)
    args = ap.parse_args()
    model, out = args.model.expanduser().resolve(), args.output.expanduser().resolve()
    if model == out:
        raise SystemExit("--output must differ from --model")
    repack(
        model,
        out,
        args.embed_bits,
        args.vision_bits,
        args.layers_per_shard,
        args.lm_head_bits,
        args.vision_group_size,
    )


if __name__ == "__main__":
    main()
