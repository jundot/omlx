# SPDX-License-Identifier: Apache-2.0
"""Streaming dequant of MiMo V2.5's pre-sharded fused QKV.

MiMo ships attention as a single fused ``qkv_proj`` that is already sharded
for tensor parallelism, alongside a block-128 ``weight_scale_inv``. Each shard
is padded to a block boundary individually, so the padding is *interleaved*
through the tensor rather than trailing it. oQ hides the fused tensor behind
virtual ``q_proj``/``k_proj``/``v_proj`` keys and applies the scale per shard
when the tensor is materialized.

The oracle below deliberately does not reuse ``split_fused_qkv``: it maps every
element to its scale block with explicit integer arithmetic. A helper factored
out of the model and then compared against that same model would agree with
itself even if both were wrong.

Weights are exact e4m3 values and every scale is a power of two, so the two
paths must agree *bit for bit*. A misplaced block shows up as a factor-of-two
error rather than a rounding difference.

Only the full-attention layers are padded; sliding-window layers happen to be
block-aligned and dequantize correctly even under the naive whole-tensor
formula. Tests that exercise only an SWA layer therefore prove nothing, which
``test_trailing_pad_dequant_*`` pins down explicitly.
"""

import json
import struct

import mlx.core as mx
import numpy as np
import pytest

from omlx.oq import (
    _block_dequant_fp8,
    _build_model_sanitizer,
    _discover_sanitize_plan,
    _DiscoveredPlan,
    _LazyTensorIndex,
)

BS = 128
TP = 4
N_COLS = 256

# Geometry chosen so the full-attention shard needs padding (704 -> 768 rows)
# while the sliding-window shard is already block-aligned (1024 rows), which
# is the asymmetry the real checkpoint has.
FULL = {"n_h": 8, "n_kv": 4, "hd": 192, "vhd": 128}
SWA = {"n_h": 8, "n_kv": 8, "hd": 192, "vhd": 128}

# 0 = full attention, 1 = sliding window.
HYBRID_PATTERN = [0, 1, 1, 0]

PARTS = ("q_proj", "k_proj", "v_proj")


def _write_safetensors(path, tensors):
    """Minimal safetensors writer for dtypes numpy cannot represent.

    tensors: {name: (dtype_str, shape, raw_bytes)}
    """
    header = {}
    offset = 0
    for name, (dtype_str, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype_str,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    header_json = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_json)))
        f.write(header_json)
        for _, (_, _, data) in tensors.items():
            f.write(data)


def _shard_rows(geom):
    q_pr = (geom["n_h"] // TP) * geom["hd"]
    k_pr = (geom["n_kv"] // TP) * geom["hd"]
    v_pr = (geom["n_kv"] // TP) * geom["vhd"]
    actual_pr = q_pr + k_pr + v_pr
    padded_pr = -(-actual_pr // BS) * BS
    return q_pr, k_pr, v_pr, actual_pr, padded_pr


def _fused_tensors(seed, geom):
    """Build one layer's fused qkv codes and its block scale.

    Weight values are multiples of 0.5 (exact in e4m3) and scales are powers
    of two, so every product is exact in bfloat16.
    """
    _, _, _, actual_pr, padded_pr = _shard_rows(geom)
    rng = np.random.default_rng(seed)
    raw = rng.integers(-6, 7, size=(TP * actual_pr, N_COLS)).astype(np.float32) * 0.5
    codes = mx.to_fp8(mx.array(raw))
    exps = rng.integers(-2, 3, size=(TP * padded_pr // BS, N_COLS // BS))
    scale = np.exp2(exps).astype(np.float32)
    return codes, scale


def _oracle_scaled(codes, scale, geom):
    """Correctly scaled fused tensor, still in on-disk row order.

    Kept separate from the split so the naive-formula comparisons below line
    up row for row; comparing against the split output would conflate a
    wrong scale with a mere reordering.
    """
    _, _, _, actual_pr, padded_pr = _shard_rows(geom)
    decoded = np.array(mx.from_fp8(codes, dtype=mx.float32))
    # Shard t's row i sits at padded offset t*padded_pr + i, which is what
    # makes the padding interleaved rather than trailing.
    scale_row = np.array(
        [(t * padded_pr + i) // BS for t in range(TP) for i in range(actual_pr)]
    )
    scale_col = np.arange(N_COLS) // BS
    return decoded * np.asarray(scale)[scale_row][:, scale_col]


def _oracle(codes, scale, geom):
    """Reference dequant+split via explicit per-element scale-block lookup."""
    q_pr, k_pr, _, actual_pr, _ = _shard_rows(geom)
    scaled = _oracle_scaled(codes, scale, geom)

    def gather(lo, hi):
        return np.concatenate(
            [scaled[t * actual_pr + lo : t * actual_pr + hi] for t in range(TP)],
            axis=0,
        )

    return (
        gather(0, q_pr),
        gather(q_pr, q_pr + k_pr),
        gather(q_pr + k_pr, actual_pr),
    )


def _trailing_pad_dequant(codes, scale, geom):
    """The wrong formula: pad the fused tensor once, at the bottom.

    This is what treating the fused tensor as a single block grid amounts to.
    It is correct only when no shard needed padding.
    """
    _, _, _, actual_pr, padded_pr = _shard_rows(geom)
    decoded = np.array(mx.from_fp8(codes, dtype=mx.float32))
    rows = TP * actual_pr
    scale_np = np.asarray(scale)
    padded = np.zeros((scale_np.shape[0] * BS, N_COLS), dtype=np.float32)
    padded[:rows] = decoded
    blocked = padded.reshape(scale_np.shape[0], BS, N_COLS // BS, BS)
    out = (blocked * scale_np[:, None, :, None]).reshape(-1, N_COLS)[:rows]
    return out


def _geometry_for(layer_idx):
    return SWA if HYBRID_PATTERN[layer_idx] else FULL


def _config(**overrides):
    cfg = {
        "model_type": "mimo_v2",
        "attention_projection_layout": "fused_qkv",
        "vocab_size": 128,
        "hidden_size": N_COLS,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_hidden_layers": len(HYBRID_PATTERN),
        "num_attention_heads": FULL["n_h"],
        "num_key_value_heads": FULL["n_kv"],
        "head_dim": FULL["hd"],
        "v_head_dim": FULL["vhd"],
        "rope_theta": 10000.0,
        "swa_num_attention_heads": SWA["n_h"],
        "swa_num_key_value_heads": SWA["n_kv"],
        "swa_head_dim": SWA["hd"],
        "swa_v_head_dim": SWA["vhd"],
        "swa_rope_theta": 10000.0,
        "sliding_window_size": 128,
        "add_full_attention_sink_bias": False,
        "add_swa_attention_sink_bias": False,
        "hybrid_layer_pattern": list(HYBRID_PATTERN),
        "moe_layer_freq": [0] * len(HYBRID_PATTERN),
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "topk_method": "greedy",
        "partial_rotary_factor": 1.0,
        "attention_bias": False,
        "layernorm_epsilon": 1e-5,
        "max_position_embeddings": 4096,
    }
    cfg.update(overrides)
    return cfg


def _build_checkpoint(tmp_path, *, include_mtp=True):
    """Write a fused-QKV checkpoint; returns (path, {layer: (codes, scale)})."""
    tensors = {}
    sources = {}
    for layer_idx in range(len(HYBRID_PATTERN)):
        geom = _geometry_for(layer_idx)
        codes, scale = _fused_tensors(layer_idx, geom)
        key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        tensors[key] = ("F8_E4M3", codes.shape, np.array(codes).tobytes())
        tensors[f"{key}_scale_inv"] = ("F32", scale.shape, scale.tobytes())
        sources[layer_idx] = (codes, scale)

    if include_mtp:
        # The MTP head sits at layer index 0 but carries sliding-window
        # geometry. Anything that infers geometry by regexing a layer index
        # out of the key will read it as full attention and misfire.
        codes, scale = _fused_tensors(99, SWA)
        key = "model.mtp.layers.0.self_attn.qkv_proj.weight"
        tensors[key] = ("F8_E4M3", codes.shape, np.array(codes).tobytes())
        tensors[f"{key}_scale_inv"] = ("F32", scale.shape, scale.tobytes())

    shard = str(tmp_path / "model.safetensors")
    _write_safetensors(shard, tensors)
    return shard, sources


@pytest.mark.parametrize("layer_idx", range(len(HYBRID_PATTERN)))
@pytest.mark.parametrize("part_idx", range(3))
def test_virtual_qkv_matches_independent_oracle(tmp_path, layer_idx, part_idx):
    shard, sources = _build_checkpoint(tmp_path)
    idx = _LazyTensorIndex([shard], config=_config())

    key = f"model.layers.{layer_idx}.self_attn.{PARTS[part_idx]}.weight"
    got = np.array(idx[key].astype(mx.float32))

    codes, scale = sources[layer_idx]
    expected = _oracle(codes, scale, _geometry_for(layer_idx))[part_idx]

    assert got.shape == expected.shape
    assert np.array_equal(got, expected), (
        f"{key}: {int((got != expected).sum())} of {got.size} elements differ"
    )


def test_virtual_keys_replace_fused_in_logical_view(tmp_path):
    shard, _ = _build_checkpoint(tmp_path)
    idx = _LazyTensorIndex([shard], config=_config())
    keys = set(idx.keys())

    for layer_idx in range(len(HYBRID_PATTERN)):
        prefix = f"model.layers.{layer_idx}.self_attn"
        assert f"{prefix}.qkv_proj.weight" not in keys
        assert f"{prefix}.qkv_proj.weight_scale_inv" not in keys
        for part in PARTS:
            assert f"{prefix}.{part}.weight" in keys

    # The logical view is what plan discovery consumes.
    logical = idx.logical_metadata()
    geom = FULL
    q_pr, k_pr, v_pr, _, _ = _shard_rows(geom)
    assert logical["model.layers.0.self_attn.q_proj.weight"] == (
        (TP * q_pr, N_COLS),
        "BF16",
    )
    assert logical["model.layers.0.self_attn.k_proj.weight"] == (
        (TP * k_pr, N_COLS),
        "BF16",
    )
    assert logical["model.layers.0.self_attn.v_proj.weight"] == (
        (TP * v_pr, N_COLS),
        "BF16",
    )


def test_mtp_head_does_not_abort_registration(tmp_path):
    """The MTP head must neither misfire geometry checks nor be split."""
    shard, _ = _build_checkpoint(tmp_path, include_mtp=True)
    idx = _LazyTensorIndex([shard], config=_config())

    mtp_key = "model.mtp.layers.0.self_attn.qkv_proj.weight"
    assert mtp_key not in idx._virtual
    assert "model.mtp.layers.0.self_attn.q_proj.weight" not in idx
    # Left alone for the model's own sanitize to drop.
    assert idx.source_shape(mtp_key) is not None


@pytest.mark.parametrize("layer_idx", [0, 3])
def test_trailing_pad_dequant_is_wrong_on_padded_layers(tmp_path, layer_idx):
    """The naive single-grid formula corrupts the layers that need padding."""
    _, sources = _build_checkpoint(tmp_path)
    codes, scale = sources[layer_idx]
    geom = _geometry_for(layer_idx)

    naive = _trailing_pad_dequant(codes, scale, geom)
    correct = _oracle_scaled(codes, scale, geom)
    # Same shape, quietly different values — this is the silent-corruption mode.
    assert naive.shape == correct.shape
    wrong_rows = int((naive != correct).any(axis=1).sum())
    assert wrong_rows > 0.5 * naive.shape[0], (
        f"expected widespread corruption, got {wrong_rows}/{naive.shape[0]} rows"
    )


@pytest.mark.parametrize("layer_idx", [1, 2])
def test_trailing_pad_dequant_looks_correct_on_aligned_layers(tmp_path, layer_idx):
    """Why an SWA-only test proves nothing: the wrong formula passes there."""
    _, sources = _build_checkpoint(tmp_path)
    codes, scale = sources[layer_idx]
    geom = _geometry_for(layer_idx)

    naive = _trailing_pad_dequant(codes, scale, geom)
    correct = _oracle_scaled(codes, scale, geom)
    assert np.array_equal(naive, correct)


def test_block_dequant_refuses_padded_fused_qkv(tmp_path):
    """The generic block dequant cannot express the interleaved layout."""
    _, sources = _build_checkpoint(tmp_path)
    codes, scale = sources[0]
    with pytest.raises(ValueError, match="not divisible by scale shape"):
        _block_dequant_fp8(codes, mx.array(scale), "F8_E4M3", "F32")


def test_streaming_plan_matches_eager_sanitize(tmp_path):
    """Discovery must yield exactly the keys eager sanitize would produce."""
    shard, sources = _build_checkpoint(tmp_path)
    config = _config()
    sanitize_fn = _build_model_sanitizer(config)
    assert sanitize_fn is not None

    eager_inputs = {}
    for layer_idx, (codes, scale) in sources.items():
        key = f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"
        eager_inputs[key] = codes
        eager_inputs[f"{key}_scale_inv"] = mx.array(scale)
    eager = sanitize_fn(dict(eager_inputs))

    idx = _LazyTensorIndex([shard], config=config)
    plan = _discover_sanitize_plan(sanitize_fn, idx)
    planned = _DiscoveredPlan(plan, idx)

    assert set(planned.keys()) == set(eager.keys())
    assert not any("qkv_proj" in k for k in planned)

    # And the replayed values must still match the independent oracle.
    for layer_idx in range(len(HYBRID_PATTERN)):
        geom = _geometry_for(layer_idx)
        expected = _oracle(*sources[layer_idx], geom)
        for part_idx, part in enumerate(PARTS):
            key = f"model.layers.{layer_idx}.self_attn.{part}.weight"
            got = np.array(planned.pop(key).astype(mx.float32))
            assert np.array_equal(got, expected[part_idx]), key


def test_registration_is_noop_without_fused_tensors(tmp_path):
    """oQ outputs and the calibration proxy inherit the config but ship split
    tensors; re-quantizing them must not trip the fused path."""
    geom = FULL
    q_pr, k_pr, v_pr, _, _ = _shard_rows(geom)
    tensors = {}
    for part, rows in zip(PARTS, (TP * q_pr, TP * k_pr, TP * v_pr)):
        data = np.zeros((rows, N_COLS), dtype=np.float32)
        tensors[f"model.layers.0.self_attn.{part}.weight"] = (
            "F32",
            data.shape,
            data.tobytes(),
        )
    shard = str(tmp_path / "model.safetensors")
    _write_safetensors(shard, tensors)

    idx = _LazyTensorIndex([shard], config=_config())
    assert idx._virtual == {}
    assert idx._hidden == set()


def test_geometry_mismatch_refuses(tmp_path):
    """An unrecognised layout must abort rather than dequantize a guess."""
    shard, _ = _build_checkpoint(tmp_path, include_mtp=False)
    bad = _config(head_dim=FULL["hd"] + 64)
    with pytest.raises(ValueError):
        _LazyTensorIndex([shard], config=bad)
