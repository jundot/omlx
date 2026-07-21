#!/usr/bin/env python3
"""Repack MLX 2-bit ternary checkpoint weights to t5 (base-3) format.

Identity I-D: ternary entropy is log2(3) ≈ 1.585 bpw; 3^5 = 243 ≤ 2^8
gives an exact 5-trits-per-byte encoding at 1.585 bpw vs the current
2-bit slots at 2.0 bpw → ~20% fewer weight bytes.

Format
------
Each group of group_size consecutive quantized values q ∈ {0,1,2} is
packed into ceil(group_size/5) uint8 bytes using base-3:

    byte_b = t_{5b} + t_{5b+1}*3 + t_{5b+2}*9 + t_{5b+3}*27 + t_{5b+4}*81

where t_k = q_k ∈ {0,1,2}.  The last byte of each group has only
(group_size % 5) active trits; the remaining positions are padded with
q=1 (trit=0, contributing nothing to the dot product).

  group_size=128 → 26 bytes/group (130 trits encoded; 2 × q=1 padding)
  group_size=64  → 13 bytes/group (65 trits encoded; 1 × q=1 padding)

The repack is lossless: dequantized values are bit-identical.
Bias tensors are dropped (t5 is always symmetric: dq = scale*(q-1)).

Usage
-----
    # Recommended: name the output to make the format explicit
    python tools/repack_ternary_t5.py \\
        --model /path/to/Bonsai-27B-mlx-2bit \\
        --output /path/to/Bonsai-27B-mlx-t5 \\
        [--group-size 128]  # default: auto-detect from config.json

    # The tool refuses to overwrite the source model directory.
    # Always specify a different --output path (e.g. append "-t5" to the name).

The output directory will contain:
  - All original non-weight files (config.json, tokenizer.*, etc.)
  - Repacked weight shards as safetensors with t5 weights in uint8

Validation
----------
After repacking, run:
    python tools/repack_ternary_t5.py --verify \\
        --model /path/to/2bit-mlx-model \\
        --t5-model /path/to/t5-model \\
        --atol 1e-4

This checks that dequantized values are bit-identical (up to fp order).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

# Actual bits-per-weight for base-3 ternary packing
_T5_BPW = math.log2(3)  # ≈ 1.585


def _suggest_output_name(src: Path) -> Path:
    """Derive output path by replacing the bit-count in the source name.

    e.g. 'Ternary-Bonsai-27B-mlx-2bit' → 'Ternary-Bonsai-27B-mlx-1.585bit'
    Falls back to appending '-1.585bit' if no bit descriptor is found.
    """
    bpw_str = f"{_T5_BPW:.3f}bit"
    new_name = re.sub(r"\d+(?:\.\d+)?-?bit", bpw_str, src.name, count=1, flags=re.IGNORECASE)
    if new_name == src.name:
        new_name = src.name + f"-{bpw_str}"
    return src.parent / new_name


# ---------------------------------------------------------------------------
# Core packing / unpacking
# ---------------------------------------------------------------------------

def pack_t5(quants: np.ndarray, group_size: int) -> np.ndarray:
    """Pack uint8 quants (values 0,1,2) into t5 bytes.

    Parameters
    ----------
    quants : (N, K) uint8 array of quantized values in {0,1,2}
    group_size : int — values per group (64 or 128)

    Returns
    -------
    (N, n_groups * bytes_per_group) uint8 t5 weight tensor
    """
    N, K = quants.shape
    assert K % group_size == 0, f"K={K} not divisible by group_size={group_size}"
    n_groups = K // group_size
    bytes_per_group = math.ceil(group_size / 5)  # 26 for gs=128, 13 for gs=64

    # Reshape to (N, n_groups, group_size)
    q = quants.reshape(N, n_groups, group_size)

    # Pad each group to bytes_per_group*5 trits with q=1 (trit=0, zero contribution)
    pad_len = bytes_per_group * 5 - group_size
    if pad_len > 0:
        q = np.concatenate([q, np.ones((N, n_groups, pad_len), dtype=np.uint8)], axis=2)
    # q: (N, n_groups, bytes_per_group*5)

    # Expose groups of 5 trits, then base-3 encode — fully vectorized, no Python loops
    q = q.reshape(N, n_groups, bytes_per_group, 5)
    v = (q[:, :, :, 0].astype(np.uint32)
         + q[:, :, :, 1] * 3
         + q[:, :, :, 2] * 9
         + q[:, :, :, 3] * 27
         + q[:, :, :, 4] * 81).astype(np.uint8)
    return v.reshape(N, n_groups * bytes_per_group)


def unpack_t5(t5w: np.ndarray, group_size: int, K: int) -> np.ndarray:
    """Unpack t5 bytes back to (N, K) uint8 quants in {0,1,2}.

    Parameters
    ----------
    t5w       : (N, n_groups * bytes_per_group) uint8
    group_size : int
    K          : int — original number of columns

    Returns
    -------
    (N, K) uint8 quants
    """
    N = t5w.shape[0]
    n_groups = K // group_size
    bytes_per_group = math.ceil(group_size / 5)

    # Decode 5 trits from every byte simultaneously — 5-iteration loop, fully vectorized
    v = t5w.reshape(N, n_groups, bytes_per_group).astype(np.uint32)
    trits = np.empty((N, n_groups, bytes_per_group, 5), dtype=np.uint8)
    for j in range(5):
        trits[:, :, :, j] = (v % 3).astype(np.uint8)
        v //= 3

    # Flatten bytes×trits axis, drop padding, reshape to (N, K)
    return trits.reshape(N, n_groups, bytes_per_group * 5)[:, :, :group_size].reshape(N, K)


# ---------------------------------------------------------------------------
# MLX 2-bit unpack helpers
# ---------------------------------------------------------------------------

def unpack_mlx_2bit(w_uint32: np.ndarray, K: int) -> np.ndarray:
    """Unpack MLX standard 2-bit weights (16 values per uint32) to uint8 quants.

    Parameters
    ----------
    w_uint32 : (N, K//16) uint32
    K        : int — number of columns

    Returns
    -------
    (N, K) uint8 quants in {0,1,2,3}  (ternary uses only {0,1,2})
    """
    N = w_uint32.shape[0]
    shifts = np.arange(16, dtype=np.uint32) * 2  # (16,)
    # (N, K//16, 16) → reshape to (N, K): slot-major ordering matches slot::16 stride
    return ((w_uint32[:, :, None] >> shifts) & 0x3).astype(np.uint8).reshape(N, K)


def dequantize_group(quants: np.ndarray, scale: float, bias: float) -> np.ndarray:
    """Dequantize a group: dq = scale * q + bias."""
    return scale * quants.astype(np.float32) + bias


# ---------------------------------------------------------------------------
# Checkpoint repack
# ---------------------------------------------------------------------------

def _load_safetensors_numpy(path: Path) -> dict[str, np.ndarray]:
    """Load a safetensors file as a dict of numpy arrays."""
    try:
        import safetensors.numpy as st_np
        return dict(st_np.load_file(str(path)))
    except ImportError:
        pass
    # Fallback: use mlx
    try:
        import mlx.core as mx
        data = mx.load(str(path))
        return {k: np.array(v) for k, v in data.items()}
    except Exception as e:
        raise RuntimeError(f"Cannot load {path}: {e}. Install safetensors or mlx.") from e


def _save_safetensors_numpy(data: dict[str, np.ndarray], path: Path) -> None:
    try:
        import safetensors.numpy as st_np
        st_np.save_file(data, str(path))
        return
    except ImportError:
        pass
    try:
        import mlx.core as mx
        mx_data = {k: mx.array(v) for k, v in data.items()}
        mx.save_safetensors(str(path), mx_data)
    except Exception as e:
        raise RuntimeError(f"Cannot save {path}: {e}. Install safetensors or mlx.") from e


def repack_shard(
    tensors: dict[str, np.ndarray],
    group_size: int,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Repack all 2-bit weight tensors in a shard to t5 format.

    Rules:
    - Keys ending in '.weight' with dtype uint32 and ndim==2 are weight tensors.
    - Their corresponding '.scales' and '.biases' must exist.
    - After repack: weight dtype becomes uint8 with t5 encoding; '.biases' key is kept.
    - '.scales' is unchanged (same values, same dtype).
    """
    out: dict[str, np.ndarray] = {}

    for key, arr in tensors.items():
        if not key.endswith(".weight"):
            out[key] = arr
            continue

        prefix = key[:-len(".weight")]
        scales_key = prefix + ".scales"
        biases_key = prefix + ".biases"

        # Only repack if 2-bit uint32 weight with matching scales/biases
        if (arr.dtype != np.uint32 or arr.ndim != 2 or
                scales_key not in tensors or biases_key not in tensors):
            out[key] = arr
            continue

        scales = tensors[scales_key]
        biases = tensors[biases_key]

        # Verify symmetry: bias should ≈ -scale (ternary 2-bit Bonsai)
        ratio = biases / (scales + 1e-9)
        if not np.allclose(ratio, -1.0, atol=1e-2):
            if verbose:
                print(f"  skip {key}: not symmetric (bias/scale ratio not ≈ -1)")
            out[key] = arr
            continue

        # Unpack 2-bit → (N, K) quants
        N, K_packed = arr.shape
        K = K_packed * 16  # 16 values per uint32

        # The requested group_size must match the checkpoint's real grouping,
        # otherwise the t5 bytes get laid out on wrong boundaries and the
        # model loads cleanly but decodes shifted trits (silent corruption).
        real_gs = K // scales.shape[-1]
        if real_gs != group_size:
            print(
                f"Error: {key} is quantized at group_size={real_gs} but the "
                f"repack was requested at group_size={group_size}.  Re-run "
                f"with --group-size {real_gs} (or omit it to auto-detect).",
                file=sys.stderr,
            )
            sys.exit(1)

        quants = unpack_mlx_2bit(arr, K)

        # Verify quants are in {0,1,2} (ternary)
        if quants.max() > 2:
            if verbose:
                print(f"  skip {key}: quants > 2 (not ternary)")
            out[key] = arr
            continue

        # Pack to t5
        t5w = pack_t5(quants, group_size)

        if verbose:
            old_bytes = arr.nbytes
            new_bytes = t5w.nbytes
            print(f"  {key}: ({N}, {K_packed}) uint32 → ({t5w.shape[0]}, {t5w.shape[1]}) uint8  "
                  f"({old_bytes/1e6:.1f} MB → {new_bytes/1e6:.1f} MB, "
                  f"{100*(1-new_bytes/old_bytes):.1f}% saved)")

        out[key] = t5w
        out[scales_key] = scales   # keep scales unchanged
        # biases are kept: mlx-lm strict load requires them; t5 decode path ignores them

    return out


def _config_group_size(model_dir: Path) -> int | None:
    """Read quantization.group_size from the model's config.json."""
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text())
    except (OSError, ValueError):
        return None
    quant = config.get("quantization")
    if isinstance(quant, dict) and isinstance(quant.get("group_size"), int):
        return quant["group_size"]
    return None


def repack_model(src: Path, dst: Path, group_size: int, verbose: bool = True) -> None:
    """Repack all weight shards in src model directory to dst.

    The source and destination must be different directories; the tool
    never overwrites the original checkpoint.
    """
    src = src.resolve()
    dst = dst.resolve()
    if src == dst:
        print(
            f"Error: --output must differ from --model.\n"
            f"  Suggested name: {_suggest_output_name(src)}",
            file=sys.stderr,
        )
        sys.exit(1)
    if dst.exists() and any(dst.iterdir()):
        print(
            f"Warning: output directory {dst} already exists and is non-empty.\n"
            f"Files will be overwritten.",
            file=sys.stderr,
        )
    dst.mkdir(parents=True, exist_ok=True)

    weight_files = sorted(src.glob("*.safetensors"))
    if not weight_files:
        print(f"No .safetensors files found in {src}", file=sys.stderr)
        sys.exit(1)

    # Copy non-weight files
    for f in src.iterdir():
        if f.suffix not in (".safetensors",) and f.name != "model.safetensors.index.json":
            dst_f = dst / f.name
            if f.is_file():
                shutil.copy2(f, dst_f)
                if verbose:
                    print(f"  copy {f.name}")

    # Repack weight shards
    for shard in weight_files:
        if verbose:
            print(f"\nRepacking {shard.name}...")
        tensors = _load_safetensors_numpy(shard)
        repacked = repack_shard(tensors, group_size, verbose=verbose)
        out_path = dst / shard.name
        _save_safetensors_numpy(repacked, out_path)
        if verbose:
            print(f"  saved → {out_path}")

    # Also copy / patch the index file if present
    index_src = src / "model.safetensors.index.json"
    if index_src.exists():
        shutil.copy2(index_src, dst / index_src.name)

    if verbose:
        print(f"\nDone. t5 model saved to {dst}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_models(
    src: Path,
    t5: Path,
    group_size: int,
    atol: float = 1e-4,
    verbose: bool = True,
) -> bool:
    """Verify dequantized weights are identical between 2-bit and t5 models."""
    ok = True
    for shard in sorted(src.glob("*.safetensors")):
        src_tensors = _load_safetensors_numpy(shard)
        t5_tensors  = _load_safetensors_numpy(t5 / shard.name)

        for key, arr in src_tensors.items():
            if not key.endswith(".weight"):
                continue
            prefix = key[:-len(".weight")]
            scales_key = prefix + ".scales"
            biases_key = prefix + ".biases"

            if (arr.dtype != np.uint32 or
                    scales_key not in src_tensors or
                    biases_key not in src_tensors):
                continue

            if key not in t5_tensors:
                print(f"MISSING {key} in t5 model")
                ok = False
                continue

            # Dequantize both
            scales = src_tensors[scales_key].astype(np.float32)
            biases = src_tensors[biases_key].astype(np.float32)

            N, K_packed = arr.shape
            K = K_packed * 16
            n_groups = K // group_size
            q_src = unpack_mlx_2bit(arr, K)

            q_t5w = t5_tensors[key]
            q_t5  = unpack_t5(q_t5w, group_size, K)

            # Vectorized dequant: broadcast scales/biases over group_size axis
            s = scales.reshape(N, n_groups, 1)
            b = biases.reshape(N, n_groups, 1)
            dq_src = (s * q_src.reshape(N, n_groups, group_size) + b).reshape(N, K)
            dq_t5  = (s * q_t5.reshape(N,  n_groups, group_size) + b).reshape(N, K)

            if not np.allclose(dq_src, dq_t5, atol=atol):
                max_diff = np.abs(dq_src - dq_t5).max()
                print(f"FAIL {key}: max_diff={max_diff:.6f} > atol={atol}")
                ok = False
            elif verbose:
                print(f"  OK {key}")

    return ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",      required=True, type=Path,
                   help="Source 2-bit MLX model directory")
    p.add_argument("--output",     type=Path, default=None,
                   help="Output t5 model directory (required unless --verify)")
    p.add_argument("--group-size", type=int, default=None,
                   help="Group size (default: auto-detect from config.json)")
    p.add_argument("--verbose",    action="store_true", default=True,
                   help="Print per-tensor progress (default: on)")
    p.add_argument("--quiet",      action="store_true",
                   help="Suppress per-tensor output")
    p.add_argument("--verify",     action="store_true",
                   help="Verify dequantized weights match (requires --t5-model)")
    p.add_argument("--t5-model",   type=Path, default=None,
                   help="t5 model path to verify against (used with --verify)")
    p.add_argument("--atol",       type=float, default=1e-4,
                   help="Absolute tolerance for verification (default: 1e-4)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    verbose = args.verbose and not args.quiet

    if args.group_size is None:
        args.group_size = _config_group_size(args.model)
        if args.group_size is None:
            print(
                "Could not read quantization.group_size from config.json; "
                "pass --group-size explicitly.",
                file=sys.stderr,
            )
            sys.exit(1)
        if verbose:
            print(f"group_size={args.group_size} (from config.json)")

    if args.verify:
        t5_path = args.t5_model or args.output
        if t5_path is None:
            print("--verify requires --t5-model or --output", file=sys.stderr)
            sys.exit(1)
        ok = verify_models(args.model, t5_path, args.group_size, args.atol, verbose)
        sys.exit(0 if ok else 1)

    if args.output is None:
        print("--output is required for repacking", file=sys.stderr)
        sys.exit(1)

    repack_model(args.model, args.output, args.group_size, verbose)


if __name__ == "__main__":
    main()
