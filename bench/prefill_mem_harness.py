"""Fase J synthetic prefill-memory harness.

Why this exists
---------------
Etapas C, D, E and A1 all change *how much Metal memory a prefill holds*,
but validating them on the real target (qwen4_exp, ~30 GiB) needs a machine
with ~30 GiB free — which is exactly the resource the bug consumes. This
harness reproduces the mechanism at reduced scale so the direction and
magnitude of each change can be measured on any Apple Silicon box.

What it actually measures
-------------------------
The F1 failure mode is purely structural: with no per-layer eval boundary,
every MoE layer's assembled mini-bank stays referenced by the lazy graph
until the chunk-end ``mx.eval``, so the peak holds ~one bank per
layer simultaneously. That property depends on layer count and bank size,
not on model identity — so a synthetic stack of ``StreamingSwitchGLU``
layers over shard-backed banks reproduces it faithfully, scaled down.

It exercises the real code path: a genuine ``ExpertBackingStore`` over
safetensors shards (so ``read_expert_into`` / ``_load_expert_bank_np`` /
promotion all run for real), real ``StreamingQuantizedSwitchLinear`` and
``StreamingSwitchGLU``, and the real per-layer load context.

Controls (A/B)
--------------
--mode accumulate   one eval at the end  (pre-Etapa-C behaviour)
--mode boundary     eval after each layer (post-Etapa-C behaviour)

OMLX_EXPERT_STREAMING_BANK_PROMOTE=1|0   Etapa A1 on/off
OMLX_EXPERT_STREAMING_LAYER_BARRIER=1|0  Etapa B ctx on/off

Weights are zero-filled and the shard is written sparse, so disk cost is
near zero. Numerics are irrelevant here — only allocation is measured.

Usage:
    python bench/prefill_mem_harness.py --layers 48 --experts 32 \
        --hidden 1024 --moe-hidden 896 --tokens 2048 --mode accumulate
    python bench/prefill_mem_harness.py --layers 48 --experts 32 --mode boundary
"""

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mlx.core as mx  # noqa: E402

_DTY_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1}
_PROJS = (
    ("gate_proj", "moe_hidden", "hidden"),
    ("up_proj", "moe_hidden", "hidden"),
    ("down_proj", "hidden", "moe_hidden"),
)


def write_sparse_shard(path: Path, tensors: dict) -> int:
    """Write a zero-filled safetensors file, leaving the data region sparse.

    Calls ``truncate`` instead of writing the payload so APFS backs it with
    no allocated blocks — a multi-GB synthetic bank costs ~0 real disk.
    """
    header = {}
    offset = 0
    for key, (shape, dtype) in tensors.items():
        nbytes = int(np.prod(shape)) * _DTY_BYTES[dtype]
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    blob = json.dumps(header).encode()
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.truncate(len(blob) + 8 + offset)
    return offset


def build_model_dir(
    root: Path,
    layers: int,
    experts: int,
    hidden: int,
    moe_hidden: int,
    group_size: int,
) -> tuple[dict, int]:
    """Create a synthetic quantized MoE model dir. Returns (weight_map, bytes)."""
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict = {}
    for li in range(layers):
        for proj, out_name, in_name in _PROJS:
            o = moe_hidden if out_name == "moe_hidden" else hidden
            i = hidden if in_name == "hidden" else moe_hidden
            base = f"model.layers.{li}.mlp.switch_mlp.{proj}"
            # 4-bit affine: 8 packed values per uint32 word.
            tensors[f"{base}.weight"] = ((experts, o, i // 8), "U32")
            tensors[f"{base}.scales"] = ((experts, o, i // group_size), "BF16")
            tensors[f"{base}.biases"] = ((experts, o, i // group_size), "BF16")
    total = write_sparse_shard(root / "model.safetensors", tensors)
    weight_map = {k: "model.safetensors" for k in tensors}
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return weight_map, total


def build_glus(model_dir: Path, layers, experts, hidden, moe_hidden, group_size, bits):
    """Assemble a stack of real StreamingSwitchGLU layers over the shards."""
    from omlx.patches.expert_streaming.shard_bank import ExpertBackingStore
    from omlx.patches.expert_streaming.streaming_switch import (
        ExpertLRUCache,
        StreamingQuantizedSwitchLinear,
        StreamingSwitchGLU,
    )

    backing = ExpertBackingStore(model_dir)
    estimate = experts * (moe_hidden * hidden // 2 + 2 * (moe_hidden * hidden // group_size))
    cache = ExpertLRUCache(
        budget_bytes=0,  # prefill_bypass below means nothing is retained anyway
        per_expert_bytes=max(1, estimate // 3),
        num_layers=layers,
    )
    # Mirrors the real prefill path: with the hotness seeder active the demand
    # set is not retained, so every layer is an all-miss and Etapa A1 engages.
    cache.prefill_bypass = True

    glus = []
    for li in range(layers):
        glu = StreamingSwitchGLU(
            input_dims=hidden,
            hidden_dims=moe_hidden,
            num_experts=experts,
            layer_idx=li,
            backing=backing,
            cache=cache,
            fused_gate_up=False,
            inverse_scatter=False,
            quantized=True,
            group_size=group_size,
            bits=bits,
            mode="affine",
        )
        for proj, out_name, in_name in _PROJS:
            o = moe_hidden if out_name == "moe_hidden" else hidden
            i = hidden if in_name == "hidden" else moe_hidden
            base = f"model.layers.{li}.mlp.switch_mlp.{proj}"
            setattr(
                glu,
                proj,
                StreamingQuantizedSwitchLinear(
                    layer_idx=li,
                    proj_name=proj,
                    stacked_weight_key=f"{base}.weight",
                    stacked_scales_key=f"{base}.scales",
                    stacked_biases_key=f"{base}.biases",
                    num_experts=experts,
                    input_dims=i,
                    output_dims=o,
                    backing=backing,
                    cache=cache,
                    group_size=group_size,
                    bits=bits,
                    mode="affine",
                    has_bias=False,
                ),
            )
        glus.append(glu)
    return glus, backing


def run(glus, tokens, hidden, experts, mode, seed=0):
    """Run one prefill-shaped forward. Returns (peak_bytes, wall_s)."""
    rng = np.random.default_rng(seed)
    # Covers every expert, so each layer's demand set is the full bank
    # (worst case, and the case where A1's single-promotion path engages).
    idx_np = rng.integers(0, experts, size=(tokens,), dtype=np.int32)
    idx_np[:experts] = np.arange(experts, dtype=np.int32)

    x = mx.zeros((1, tokens, hidden), dtype=mx.bfloat16)
    indices = mx.array(idx_np)
    mx.eval(x, indices)
    mx.reset_peak_memory()

    outs = []
    t0 = time.perf_counter()
    for glu in glus:
        # Keep every layer's output referenced: this is what pins the
        # assembled mini-banks until the chunk-end eval (the F1 failure mode).
        outs.append(glu(x, indices))
        if mode == "boundary":
            mx.eval(outs[-1])
    mx.eval(*outs)
    wall = time.perf_counter() - t0
    return mx.get_peak_memory(), wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--moe-hidden", type=int, default=896)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--mode", choices=("accumulate", "boundary"), default="accumulate")
    ap.add_argument("--root", default=None, help="reuse/keep the synthetic model dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tmp = Path(args.root) if args.root else Path("/tmp/omlx_synth_moe")
    own = args.root is None
    if own and tmp.exists():
        shutil.rmtree(tmp)

    _, shard_bytes = build_model_dir(
        tmp, args.layers, args.experts, args.hidden, args.moe_hidden, args.group_size
    )
    glus, _ = build_glus(
        tmp, args.layers, args.experts, args.hidden, args.moe_hidden, args.group_size, args.bits
    )

    # Warm-up: first touch pulls the sparse file into the page cache and
    # materialises any lazy MLX state, so the measured run is not the cold one.
    run(glus, min(args.tokens, 64), args.hidden, args.experts, mode="boundary", seed=1)
    peak, wall = run(glus, args.tokens, args.hidden, args.experts, mode=args.mode, seed=0)

    gib = peak / 1024**3
    result = {
        "mode": args.mode,
        "layers": args.layers,
        "experts": args.experts,
        "hidden": args.hidden,
        "moe_hidden": args.moe_hidden,
        "tokens": args.tokens,
        "shard_bytes": shard_bytes,
        "peak_bytes": peak,
        "peak_gib": round(gib, 3),
        "wall_s": round(wall, 3),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if own:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
