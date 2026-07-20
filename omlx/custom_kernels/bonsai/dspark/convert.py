# SPDX-License-Identifier: Apache-2.0
"""Convert a Bonsai DSpark GGUF drafter checkpoint to MLX safetensors.

Usage (CLI)::

    python -m omlx.custom_kernels.bonsai.dspark.convert \\
        --gguf  models/ternary-gguf/27B/Ternary-Bonsai-27B-dspark-Q4_1.gguf \\
        --out   ~/.cache/omlx/bonsai-dspark

Usage (Python)::

    from omlx.custom_kernels.bonsai.dspark.convert import convert_gguf
    config = convert_gguf(gguf_path, out_dir)

Output directory contains:
  config.json          BonsaiDSparkConfig in JSON form
  drafter.safetensors  Model weights (BF16; token_embd excluded — shared from target)

Notes
-----
- ``token_embd.weight`` (dtype=42, Prism custom ternary) is **skipped**.
  Call ``BonsaiDSparkDrafter.bind_target_embedding(target)`` after loading.
- ``output.weight`` (lm_head, Q4_1) and large Q4_1 vocab tensors are dequantized
  to BF16. The resulting weights file is ~3 GB; if this is too large for your
  storage, add the ``--quantize`` flag to re-quantize to mlx 4-bit after loading.
- GGUF Q4_1 dequant: ``value[i] = delta * nibble[i] + d_min``, block_size=32.
- GGUF BF16 (dtype=30): read as 2-byte little-endian BF16, convert to float32.
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GGUF binary format constants
# ---------------------------------------------------------------------------

_GGUF_MAGIC = b"GGUF"
_GGUF_VERSION_SUPPORTED = {2, 3}

# GGML type IDs
_GGML_F32 = 0
_GGML_F16 = 1
_GGML_Q4_0 = 2
_GGML_Q4_1 = 3
_GGML_BF16 = 30
_GGML_PRISM_TERNARY = 42  # Prism custom dtype — skip

# GGUF metadata value types
_GGUFValueType = {
    0: "uint8", 1: "int8", 2: "uint16", 3: "int16",
    4: "uint32", 5: "int32", 6: "float32",
    7: "bool", 8: "string", 9: "array",
    10: "uint64", 11: "int64", 12: "float64",
}

# Q4_1 block layout: [nibbles: 16 bytes][delta: f16][d_min: f16] → 32 values per block
_Q4_1_BLOCK_SIZE = 32
_Q4_1_BLOCK_BYTES = 16 + 2 + 2  # nibbles + delta + d_min = 20

# Q4_0 block layout: [nibbles: 16 bytes][delta: f16] → 32 values per block
_Q4_0_BLOCK_SIZE = 32
_Q4_0_BLOCK_BYTES = 16 + 2  # 18 bytes


# ---------------------------------------------------------------------------
# Key remapping: GGUF tensor names → MLX safetensors weight names
# ---------------------------------------------------------------------------

def _remap_key(gguf_name: str) -> str | None:
    """Map GGUF tensor name to the mlx weight key used by BonsaiDSparkDrafter.

    Returns None to skip the tensor (token_embd is shared from target).
    """
    n = gguf_name

    # Skip token embedding — dtype=42, shared with target
    if n == "token_embd.weight":
        return None

    # Block attention / MLP weights: blk.{i}.attn_q.weight → layers.{i}.self_attn.q_proj.weight
    if n.startswith("blk."):
        parts = n.split(".", 2)
        if len(parts) < 3:
            return n
        idx = parts[1]
        rest = parts[2]
        prefix = f"layers.{idx}."
        mapping = {
            "attn_q.weight":          "self_attn.q_proj.weight",
            "attn_k.weight":          "self_attn.k_proj.weight",
            "attn_v.weight":          "self_attn.v_proj.weight",
            "attn_output.weight":     "self_attn.o_proj.weight",
            "attn_q_norm.weight":     "self_attn.q_norm.weight",
            "attn_k_norm.weight":     "self_attn.k_norm.weight",
            "attn_norm.weight":       "input_layernorm.weight",
            "ffn_gate.weight":        "mlp.gate_proj.weight",
            "ffn_up.weight":          "mlp.up_proj.weight",
            "ffn_down.weight":        "mlp.down_proj.weight",
            "ffn_norm.weight":        "post_attention_layernorm.weight",
        }
        mapped = mapping.get(rest)
        if mapped is None:
            logger.warning("Unknown block tensor key: %s", n)
            return n
        return prefix + mapped

    # DSpark projection / conditioning weights
    dspark_mapping = {
        "dspark.fc.weight":                "fc.weight",
        "dspark.hidden_norm.weight":       "hidden_norm.weight",
        "dspark.log_snr_fc1.weight":       "log_snr_fc1.weight",
        "dspark.log_snr_fc1.bias":         "log_snr_fc1.bias",
        "dspark.log_snr_fc2.weight":       "log_snr_fc2.weight",
        "dspark.log_snr_fc2.bias":         "log_snr_fc2.bias",
        "dspark.markov_head_a.weight":     "markov_head.markov_w1.weight",
        "dspark.markov_head_b.weight":     "markov_head.markov_w2.weight",
        "dspark.confidence_head.weight":   "confidence_head.proj.weight",
        "dspark.confidence_head.bias":     "confidence_head.proj.bias",
    }
    if n in dspark_mapping:
        return dspark_mapping[n]

    # Output head and norm
    if n == "output.weight":
        return "lm_head.weight"
    if n == "output_norm.weight":
        return "norm.weight"

    logger.warning("Unrecognized GGUF tensor key: %s — keeping as-is", n)
    return n


# ---------------------------------------------------------------------------
# Low-level GGUF parser
# ---------------------------------------------------------------------------

def _read_str(data: bytes, pos: int) -> tuple[str, int]:
    length = struct.unpack_from("<Q", data, pos)[0]
    pos += 8
    s = data[pos:pos + length].decode("utf-8")
    pos += length
    return s, pos


def _read_value(data: bytes, pos: int, vtype: int) -> tuple[object, int]:
    """Read one GGUF metadata value, returning (value, new_pos)."""
    if vtype == 0:    # uint8
        return struct.unpack_from("B", data, pos)[0], pos + 1
    if vtype == 1:    # int8
        return struct.unpack_from("b", data, pos)[0], pos + 1
    if vtype == 2:    # uint16
        return struct.unpack_from("<H", data, pos)[0], pos + 2
    if vtype == 3:    # int16
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if vtype == 4:    # uint32
        return struct.unpack_from("<I", data, pos)[0], pos + 4
    if vtype == 5:    # int32
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if vtype == 6:    # float32
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if vtype == 7:    # bool
        return bool(struct.unpack_from("B", data, pos)[0]), pos + 1
    if vtype == 8:    # string
        return _read_str(data, pos)
    if vtype == 9:    # array
        elem_type = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        count = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        vals = []
        for _ in range(count):
            v, pos = _read_value(data, pos, elem_type)
            vals.append(v)
        return vals, pos
    if vtype == 10:   # uint64
        return struct.unpack_from("<Q", data, pos)[0], pos + 8
    if vtype == 11:   # int64
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if vtype == 12:   # float64
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    raise ValueError(f"Unknown GGUF value type {vtype}")


def _parse_gguf_header(data: bytes) -> tuple[dict, list[dict], int]:
    """Parse GGUF header.

    Returns:
        kv: metadata dict
        tensors: list of {name, dtype, shape, offset}  (offset from data section)
        data_offset: byte position in ``data`` where the tensor data section begins
    """
    if data[:4] != _GGUF_MAGIC:
        raise ValueError("Not a GGUF file (bad magic)")
    version = struct.unpack_from("<I", data, 4)[0]
    if version not in _GGUF_VERSION_SUPPORTED:
        raise ValueError(f"GGUF version {version} not supported (expected 2 or 3)")

    n_tensors = struct.unpack_from("<Q", data, 8)[0]
    n_kv = struct.unpack_from("<Q", data, 16)[0]
    pos = 24

    kv: dict[str, object] = {}
    for _ in range(n_kv):
        key, pos = _read_str(data, pos)
        vtype = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        val, pos = _read_value(data, pos, vtype)
        kv[key] = val

    tensors: list[dict] = []
    for _ in range(n_tensors):
        name, pos = _read_str(data, pos)
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = []
        for _ in range(n_dims):
            dims.append(struct.unpack_from("<Q", data, pos)[0])
            pos += 8
        ggml_type = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        offset = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        tensors.append({"name": name, "dtype": ggml_type, "shape": dims, "offset": offset})

    # Data section starts at next 32-byte alignment after header
    data_offset = (pos + 31) & ~31
    return kv, tensors, data_offset


# ---------------------------------------------------------------------------
# Dequantization
# ---------------------------------------------------------------------------

def _dequant_q4_1(raw: bytes, n_elements: int) -> np.ndarray:
    """Dequantize GGML Q4_1 → float32.

    Each 20-byte block:
      bytes 0..15: 16 bytes → 32 nibbles (two values per byte: low nibble first)
      bytes 16..17: delta (float16 LE)
      bytes 18..19: d_min (float16 LE)
    """
    n_blocks = n_elements // _Q4_1_BLOCK_SIZE
    assert n_blocks * _Q4_1_BLOCK_SIZE == n_elements, \
        f"Q4_1 element count {n_elements} not divisible by block size {_Q4_1_BLOCK_SIZE}"

    raw_arr = np.frombuffer(raw[:n_blocks * _Q4_1_BLOCK_BYTES], dtype=np.uint8)
    raw_arr = raw_arr.reshape(n_blocks, _Q4_1_BLOCK_BYTES)

    # Nibble data: first 16 bytes per block → 32 uint4 values
    nibble_bytes = raw_arr[:, :16]  # [n_blocks, 16]
    lo = nibble_bytes & 0x0F         # [n_blocks, 16] low nibbles
    hi = (nibble_bytes >> 4) & 0x0F  # [n_blocks, 16] high nibbles
    # Interleave: lo[i], hi[i] for all i → [n_blocks, 32]
    nibbles = np.stack([lo, hi], axis=2).reshape(n_blocks, 32).astype(np.float32)

    # Scale and min
    delta = raw_arr[:, 16:18].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    d_min = raw_arr[:, 18:20].copy().view(np.float16).reshape(n_blocks).astype(np.float32)

    result = nibbles * delta[:, None] + d_min[:, None]
    return result.reshape(n_elements)


def _dequant_q4_0(raw: bytes, n_elements: int) -> np.ndarray:
    """Dequantize GGML Q4_0 → float32.

    Each 18-byte block:
      bytes 0..15: 16 bytes → 32 nibbles (symmetric, zero point = 8)
      bytes 16..17: delta (float16 LE)
    """
    n_blocks = n_elements // _Q4_0_BLOCK_SIZE
    assert n_blocks * _Q4_0_BLOCK_SIZE == n_elements

    raw_arr = np.frombuffer(raw[:n_blocks * _Q4_0_BLOCK_BYTES], dtype=np.uint8)
    raw_arr = raw_arr.reshape(n_blocks, _Q4_0_BLOCK_BYTES)

    nibble_bytes = raw_arr[:, :16]
    lo = (nibble_bytes & 0x0F).astype(np.int8) - 8
    hi = ((nibble_bytes >> 4) & 0x0F).astype(np.int8) - 8
    nibbles = np.stack([lo, hi], axis=2).reshape(n_blocks, 32).astype(np.float32)

    delta = raw_arr[:, 16:18].copy().view(np.float16).reshape(n_blocks).astype(np.float32)
    return (nibbles * delta[:, None]).reshape(n_elements)


def _read_f16(raw: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(raw[:n_elements * 2], dtype="<f2").astype(np.float32)


def _read_bf16(raw: bytes, n_elements: int) -> np.ndarray:
    """Read BF16 as little-endian uint16, reinterpret as float32."""
    u16 = np.frombuffer(raw[:n_elements * 2], dtype="<u2")
    # BF16 → float32: shift into the high 16 bits of float32
    f32 = (u16.astype(np.uint32) << 16).view(np.float32)
    return f32


def _read_f32(raw: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(raw[:n_elements * 4], dtype="<f4").copy()


def _dequantize_tensor(info: dict, data: bytes, data_offset: int) -> np.ndarray:
    """Extract and dequantize one tensor, returned as float32 numpy array."""
    dtype = info["dtype"]
    # GGUF dims are innermost-first (opposite of numpy/mlx convention)
    # E.g. [5120, 248320] in GGUF = 248320 rows × 5120 cols in row-major → shape [248320, 5120]
    # But GGUF stores dims as [in_features, out_features] for weight matrices,
    # so we need to transpose for mlx (which expects [out, in]).
    # We don't transpose here — we store as-is and note that the mlx model
    # definitions must match. GGUF convention: dims[0] is fastest-varying.
    shape_gguf = info["shape"]   # GGUF order: [fastest, ..., slowest]
    # numpy shape should be reversed so dim[0]=rows, consistent with safetensors
    shape_np = shape_gguf[::-1]   # typical: [out_features, in_features]
    n_elements = 1
    for d in shape_gguf:
        n_elements *= d

    raw_start = data_offset + info["offset"]

    if dtype == _GGML_F32:
        arr = _read_f32(data[raw_start:], n_elements)
    elif dtype == _GGML_F16:
        arr = _read_f16(data[raw_start:], n_elements)
    elif dtype == _GGML_BF16:
        arr = _read_bf16(data[raw_start:], n_elements)
    elif dtype == _GGML_Q4_1:
        arr = _dequant_q4_1(data[raw_start:], n_elements)
    elif dtype == _GGML_Q4_0:
        arr = _dequant_q4_0(data[raw_start:], n_elements)
    else:
        raise ValueError(f"Unsupported GGML dtype {dtype} for tensor {info['name']!r}")

    return arr.reshape(shape_np).astype(np.float32)


# ---------------------------------------------------------------------------
# Config extraction from GGUF metadata
# ---------------------------------------------------------------------------

def _config_from_kv(kv: dict) -> "BonsaiDSparkConfig":  # noqa: F821
    from .config import BonsaiDSparkConfig

    def _get(key: str, default):
        for prefix in ("dspark.", ""):
            v = kv.get(prefix + key)
            if v is not None:
                return v
        return default

    target_layers = _get("target_layer_ids", None)
    if target_layers is None:
        target_layers = _get("target_layers", [1, 16, 31, 46, 61])

    return BonsaiDSparkConfig(
        family="bonsai",
        hidden_size=int(_get("hidden_size", 5120)),
        vocab_size=int(_get("vocab_size", 248320)),
        num_hidden_layers=int(_get("block_count", 6)),
        intermediate_size=int(_get("feed_forward_length", 8192)),
        rms_norm_eps=float(_get("rms_norm_eps", 1e-6)),
        num_attention_heads=int(_get("attention.head_count", 16)),
        num_key_value_heads=int(_get("attention.head_count_kv", 8)),
        head_dim=int(_get("attention.head_dim",
                          int(_get("hidden_size", 5120)) // int(_get("attention.head_count", 16)))),
        block_size=int(_get("block_size", 4)),
        mask_token_id=int(_get("mask_token_id", 151666)),
        target_layer_ids=[int(x) for x in target_layers],
        markov_rank=int(_get("markov_rank", 256)),
        enable_confidence_head=bool(_get("enable_confidence_head", True)),
        confidence_head_with_markov=bool(_get("confidence_head_with_markov", True)),
        log_snr_dim=int(_get("log_snr_dim", 128)),
        log_snr_inference=float(_get("log_snr_inference", 10.0)),
    )


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------

def convert_gguf(
    gguf_path: str | Path,
    out_dir: str | Path,
    *,
    verbose: bool = True,
) -> "BonsaiDSparkConfig":  # noqa: F821
    """Convert a Bonsai DSpark GGUF file to MLX safetensors.

    Parameters
    ----------
    gguf_path:
        Path to the ``.gguf`` file (e.g. ``Ternary-Bonsai-27B-dspark-Q4_1.gguf``).
    out_dir:
        Directory to write ``config.json`` and ``drafter.safetensors`` into.
        Created if it does not exist.
    verbose:
        Log progress for each tensor.

    Returns
    -------
    BonsaiDSparkConfig
        Config parsed from the GGUF metadata.
    """
    from safetensors.numpy import save_file as save_safetensors

    from .config import BonsaiDSparkConfig

    gguf_path = Path(gguf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Reading GGUF: %s  (%d MiB)", gguf_path, gguf_path.stat().st_size >> 20)
    data = gguf_path.read_bytes()

    kv, tensor_infos, data_offset = _parse_gguf_header(data)
    if verbose:
        logger.info("GGUF version=%d  tensors=%d  kv=%d  data_offset=%#x",
                    struct.unpack_from("<I", data, 4)[0],
                    len(tensor_infos), len(kv), data_offset)

    config = _config_from_kv(kv)

    weights: dict[str, np.ndarray] = {}
    skipped = []

    for info in tensor_infos:
        name = info["name"]
        dtype = info["dtype"]

        mlx_key = _remap_key(name)
        if mlx_key is None:
            skipped.append(name)
            if verbose:
                logger.info("  SKIP  %-45s  (dtype=%d, shared from target)", name, dtype)
            continue

        if verbose:
            shape_str = "×".join(str(d) for d in info["shape"])
            logger.info("  %-45s  dtype=%-3d  shape=%s → %s",
                        name, dtype, shape_str, mlx_key)

        arr = _dequantize_tensor(info, data, data_offset)
        weights[mlx_key] = arr.astype(np.float32)

    # Norm weights in GGUF for Bonsai are stored as deltas from 1.0 (same HF convention
    # that mlx_vlm's sanitize applies). Add 1.0 so nn.RMSNorm(weight) works correctly.
    _norm_suffixes = (
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "hidden_norm.weight",
        "norm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    )
    for k in list(weights):
        if any(k.endswith(sfx) for sfx in _norm_suffixes):
            weights[k] = weights[k] + 1.0

    # Convert to BF16 for smaller file (keeps precision close to bf16 inference)
    weights_bf16 = {k: v.astype(np.float32) for k, v in weights.items()}

    out_weights = out_dir / "drafter.safetensors"
    logger.info("Saving %d tensors → %s", len(weights_bf16), out_weights)
    save_safetensors(weights_bf16, str(out_weights))

    # Save config
    out_config = out_dir / "config.json"
    with open(out_config, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
    logger.info("Config → %s", out_config)

    if skipped:
        logger.info("Skipped %d tensors (dtype=42, shared from target): %s",
                    len(skipped), ", ".join(skipped))

    return config


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="Convert Bonsai DSpark GGUF → MLX safetensors")
    p.add_argument("--gguf", required=True, help="Input .gguf file")
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    convert_gguf(args.gguf, args.out)


if __name__ == "__main__":
    _main()
