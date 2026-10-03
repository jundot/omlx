"""Repack qwen4_exp MoE expert (switch_mlp) tensors into a page-aligned,
mmap-streaming artifact (PLE-artifact-style, separate file, non-destructive).

Every tensor's data starts at a 16 KB page boundary so the streaming loader can
create a page-aligned bytesNoCopy MTLBuffer and address each tensor via a
page-aligned `set_buffer` offset (Metal requires >=4-byte offset alignment; ~80%
of this checkpoint's shards are non-4-aligned in absolute file offset).

Artifact layout:
  [8-byte little-endian manifest_len][manifest JSON, then zero-pad to PAGE]
  [tensor 0 data, zero-pad to PAGE][tensor 1 data, zero-pad to PAGE]...
Manifest: {"page_size":N,
           "tensors":{key:{offset,length,dtype,shape,bits,group_size,mode}}}

Quantization params (bits/group_size/mode) are carried explicitly per tensor,
read from config.json["quantization"] (global default + per-module overrides) --
NEVER inferred from tensor shape. This checkpoint is NOT uniform: the 48 layer
blocks are oQ2 2-bit gs32 but the MTP block's switch_mlp is 4-bit gs32 (packed
dim 320 vs 160). Shape-inference of bits/gs is provably ambiguous, so the loader
must consume these manifest fields directly.

Usage: repack.py <out_path> [--model-dir DIR] [--subset N] [--verify]

The source checkpoint directory is resolved (highest precedence first) from
``--model-dir DIR``, then the ``OMLX_QWEN4_MODEL_DIR`` env var, then the example
default below. It is NOT hardcoded -- point it at wherever the oQ2-MTP checkpoint
lives on your machine.
"""
import json
import os
import struct
import sys

PAGE = 16384
# Example checkpoint this artifact format was authored against. Override via
# --model-dir or OMLX_QWEN4_MODEL_DIR; do not rely on this literal path.
DEFAULT_MODEL_DIR = (
    "/Users/alytaphoenix/.omlx/models/Vontra/Qwen3.8-Flash-Next-MLX-oQ2-MTP"
)
D = os.environ.get("OMLX_QWEN4_MODEL_DIR", DEFAULT_MODEL_DIR)


def _align(n, a=PAGE):
    return (n + a - 1) // a * a


def _load_quant_config():
    """config.json quantization block: global bits/group_size/mode + per-module
    overrides (keyed by module path, e.g. '...switch_mlp.gate_proj')."""
    q = json.load(open(os.path.join(D, "config.json")))["quantization"]
    return q, q["bits"], q["group_size"], q["mode"]


_QCFG = None


def _quant_params(tensor_key):
    """(bits, group_size, mode) for a tensor, from config -- never shape-inferred.
    Override keys are MODULE paths, so strip the trailing .weight/.scales/.biases."""
    global _QCFG
    if _QCFG is None:
        _QCFG = _load_quant_config()
    q, gb, gg, gm = _QCFG
    o = q.get(tensor_key.rsplit(".", 1)[0])
    if isinstance(o, dict):
        return int(o["bits"]), int(o["group_size"]), o.get("mode", gm)
    return gb, gg, gm


def _shard_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def enumerate_experts():
    """Return [(key, shard_path, abs_offset, length, dtype, shape)] for every
    switch_mlp tensor across all 49 MoE blocks (48 layers + MTP), grouped so a
    subset takes whole blocks."""
    idx = json.load(open(os.path.join(D, "model.safetensors.index.json")))["weight_map"]
    headers = {}
    out = []
    for key, fn in idx.items():
        if ".switch_mlp." not in key:
            continue
        path = os.path.join(D, fn)
        if fn not in headers:
            headers[fn] = _shard_header(path)
        hdr, data_start = headers[fn]
        e = hdr[key]
        s, en = e["data_offsets"]
        out.append((key, path, data_start + s, en - s, e["dtype"], e["shape"]))
    # sort by (layer index parsed from key, key) for deterministic block grouping
    import re

    def layer_of(k):
        m = re.search(r"layers\.(\d+)\.", k)
        return int(m.group(1)) if m else 9999  # MTP block (no layers.N) sorts last

    out.sort(key=lambda t: (layer_of(t[0]), t[0]))
    return out


def repack(out_path, subset=None, verify=False):
    tensors = enumerate_experts()
    if subset is not None:
        # keep whole blocks: 9 tensors/block, so subset*9 tensors
        tensors = tensors[: subset * 9]
    print(f"repacking {len(tensors)} tensors "
          f"({len(tensors)//9} blocks) -> {out_path}")

    # Build manifest with page-aligned offsets (after the manifest block).
    manifest = {"page_size": PAGE, "tensors": {}}
    # placeholder to size the manifest; offsets depend on manifest length, so
    # do a two-pass: assume a generous manifest size, then place data.
    # Simpler: reserve one page for length header + manifest up to (blocks*~1KB).
    def _entry(cur, key, length, dtype, shape):
        bits, gs, mode = _quant_params(key)
        return {"offset": cur, "length": length, "dtype": dtype, "shape": shape,
                "bits": bits, "group_size": gs, "mode": mode}

    est_manifest = 8 + len(json.dumps({
        "page_size": PAGE,
        "tensors": {k: _entry(0, k, ln, dt, sh)
                    for (k, _, _, ln, dt, sh) in tensors},
    }).encode()) + 4096
    data_start = _align(est_manifest)

    cur = data_start
    for (key, _path, _off, length, dtype, shape) in tensors:
        manifest["tensors"][key] = _entry(cur, key, length, dtype, shape)
        cur += _align(length)
    total = cur
    mb = json.dumps(manifest).encode()
    assert 8 + len(mb) <= data_start, "manifest bigger than reserved region"

    # Write.
    written = 0
    with open(out_path, "wb") as w:
        w.write(struct.pack("<Q", len(mb)))
        w.write(mb)
        w.write(b"\x00" * (data_start - 8 - len(mb)))  # pad to data_start
        for (key, path, off, length, dtype, shape) in tensors:
            tgt = manifest["tensors"][key]["offset"]
            assert w.tell() == tgt, (w.tell(), tgt, key)
            with open(path, "rb") as src:
                src.seek(off)
                remaining = length
                while remaining:
                    chunk = src.read(min(1 << 22, remaining))
                    w.write(chunk)
                    remaining -= len(chunk)
            pad = _align(length) - length
            if pad:
                w.write(b"\x00" * pad)
            written += 1
            if written % 45 == 0:
                print(f"  {written}/{len(tensors)} ({cur/1024**3:.1f}GB target)")
    print(f"wrote {total/1024**3:.2f} GB")

    if verify:
        import mmap
        import numpy as np
        NPD = {"U32": np.uint32, "BF16": np.uint16}
        with open(out_path, "rb") as af:
            mm = mmap.mmap(af.fileno(), 0, access=mmap.ACCESS_READ)
        mlen = struct.unpack("<Q", mm[:8])[0]
        man = json.loads(mm[8:8 + mlen])
        bad = 0
        for (key, path, off, length, dtype, shape) in tensors:
            t = man["tensors"][key]
            assert t["offset"] % PAGE == 0, f"{key} not page-aligned"
            a = np.frombuffer(mm[t["offset"]: t["offset"] + length], dtype=NPD[dtype])
            with open(path, "rb") as src:
                src.seek(off)
                b = np.frombuffer(src.read(length), dtype=NPD[dtype])
            if not np.array_equal(a, b):
                bad += 1
                print(f"  MISMATCH {key}")
        print(f"verify: {len(tensors)-bad}/{len(tensors)} tensors byte-identical, "
              f"all page-aligned")
    return manifest


if __name__ == "__main__":
    out = sys.argv[1]
    subset = None
    if "--subset" in sys.argv:
        subset = int(sys.argv[sys.argv.index("--subset") + 1])
    if "--model-dir" in sys.argv:
        # Reassign the module global read by _load_quant_config/enumerate_experts.
        D = sys.argv[sys.argv.index("--model-dir") + 1]
    if not os.path.isdir(D):
        raise SystemExit(
            f"model dir not found: {D!r}\n"
            "Pass --model-dir DIR or set OMLX_QWEN4_MODEL_DIR to the oQ2-MTP "
            "checkpoint directory."
        )
    repack(out, subset=subset, verify="--verify" in sys.argv)
