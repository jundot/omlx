"""JANGQ quantization-contract reconstruction (no model download).

Covers the pure packing inference, the per-projection streaming packing
helper, and checkpoint detection on synthetic safetensors directories.
Full synthesis against a live model is covered by the JANG_4S load smoke.
"""

import json
import struct
from types import SimpleNamespace

import pytest

from omlx.patches.expert_streaming import _source_packing
from omlx.patches.mlx_vlm_qwen4_exp_compat.jangq import (
    detect_jangq,
    infer_packing,
    load_cached_policy,
)


def test_infer_packing_routed_gate_3bit():
    assert infer_packing((512, 640, 240), (512, 640, 40), (512, 640, 2560)) == (3, 64)


def test_infer_packing_shared_expert_8bit():
    assert infer_packing((640, 640), (640, 40), (640, 2560)) == (8, 64)


def test_infer_packing_2bit_group32():
    assert infer_packing((8, 2), (8, 1), (8, 32)) == (2, 32)


def test_infer_packing_rejects_mismatched_leading_dims():
    assert infer_packing((512, 640, 240), (512, 640, 40), (256, 640, 2560)) is None


def test_infer_packing_rejects_unsupported_bits():
    # 16-bit implied packing is a dense tensor, not a packed triple.
    assert infer_packing((4, 16), (4, 1), (4, 32)) is None


def test_infer_packing_rejects_rank_mismatch():
    assert infer_packing((640,), (640, 40), (640, 2560)) is None


def test_source_packing_prefers_module_values():
    src = SimpleNamespace(group_size=64, bits=2, mode="affine")
    assert _source_packing(src, 64, 8, "affine") == (64, 2, "affine")


def test_source_packing_missing_attrs_fail_loudly():
    """A projection without packing attrs must raise, never inherit silently."""
    import pytest

    with pytest.raises(ValueError, match="lacks 'group_size'"):
        _source_packing(SimpleNamespace(), 64, 4, "affine")


def _write_fake_model(tmp_path, *, with_quantization=False):
    base = "language_model.layers.0.mlp.switch_mlp.gate_proj"
    tensors = {
        f"{base}.weight": ("U32", (2, 4, 6)),
        f"{base}.scales": ("F16", (2, 4, 1)),
        f"{base}.biases": ("F16", (2, 4, 1)),
    }
    header = {}
    offset = 0
    blob = bytearray()
    for key, (dtype, shape) in tensors.items():
        nbytes = 2 * 4 * 6 * (4 if dtype == "U32" else 2)
        header[key] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        blob += b"\x00" * nbytes
        offset += nbytes
    raw = json.dumps(header).encode()
    (tmp_path / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(raw)) + raw + bytes(blob)
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {"weight_map": {k: "model-00001-of-00001.safetensors" for k in tensors}}
        ),
        encoding="utf-8",
    )
    config = {"model_type": "qwen4_exp", "architectures": ["Qwen4Exp"]}
    if with_quantization:
        config["quantization"] = {"group_size": 64, "bits": 8, "mode": "affine"}
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")


def test_detect_jangq_finds_packed_triple(tmp_path):
    _write_fake_model(tmp_path)
    found = detect_jangq(tmp_path)
    assert found is not None
    assert found["triples"] == 1
    assert found["fingerprint"]


def test_detect_jangq_skips_quantized_checkpoints(tmp_path):
    _write_fake_model(tmp_path, with_quantization=True)
    assert detect_jangq(tmp_path) is None


def test_detect_jangq_rejects_bare_directory(tmp_path):
    assert detect_jangq(tmp_path) is None


def test_load_cached_policy_miss_returns_none():
    assert load_cached_policy("nonexistent-fingerprint") is None
