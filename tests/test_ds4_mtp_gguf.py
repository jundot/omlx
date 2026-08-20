# SPDX-License-Identifier: Apache-2.0
"""DS4 MTP GGUF compatibility validation tests."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from omlx.ds4_gguf import (
    DS4_DSPARK_REQUIRED_TENSORS,
    DS4_MTP_REQUIRED_TENSORS,
    DS4MTPCompatibilityError,
    detect_ds4_mtp_sidecar_kind,
    is_ds4_mtp_gguf_sidecar,
    validate_ds4_mtp_compatibility,
)

_GGUF_TYPE_UINT32 = 4
_GGUF_TYPE_STRING = 8
_GGML_TYPE_F32 = 0


def _write_string(f, value: str) -> None:
    data = value.encode("utf-8")
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def _write_scalar_kv(f, key: str, value) -> None:
    _write_string(f, key)
    if isinstance(value, str):
        f.write(struct.pack("<I", _GGUF_TYPE_STRING))
        _write_string(f, value)
    else:
        f.write(struct.pack("<I", _GGUF_TYPE_UINT32))
        f.write(struct.pack("<I", int(value)))


def _write_tiny_gguf(
    path: Path,
    *,
    metadata: dict[str, object],
    tensors: dict[str, tuple[int, ...]] | None = None,
) -> None:
    tensors = tensors or {}
    with path.open("wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3))
        f.write(struct.pack("<Q", len(tensors)))
        f.write(struct.pack("<Q", len(metadata)))
        for key, value in metadata.items():
            _write_scalar_kv(f, key, value)
        for name, dims in tensors.items():
            _write_string(f, name)
            f.write(struct.pack("<I", len(dims)))
            for dim in dims:
                f.write(struct.pack("<Q", dim))
            f.write(struct.pack("<I", _GGML_TYPE_F32))
            f.write(struct.pack("<Q", 0))


def _main_metadata(name: str = "DeepSeek V4 Flash") -> dict[str, object]:
    return {
        "general.architecture": "deepseek4",
        "general.name": name,
        "deepseek4.embedding_length": 4096,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_feed_forward_length": 2048,
        "deepseek4.nextn_predict_layers": 1,
    }


def _mtp_metadata() -> dict[str, object]:
    return {
        "general.architecture": "deepseek4_mtp_support",
        "general.name": "DeepSeek V4 Flash MTP support",
        "deepseek4.expert_count": 256,
        "deepseek4.mtp_layer_count": 1,
        "deepseek4.nextn_predict_layers": 1,
    }


def _mtp_tensors(*, routed_shape=(4096, 2048, 256)) -> dict[str, tuple[int, ...]]:
    tensors = {name: (1,) for name in DS4_MTP_REQUIRED_TENSORS}
    for name in (
        "mtp.0.enorm.weight",
        "mtp.0.hnorm.weight",
        "mtp.0.norm.weight",
        "mtp.0.attn_norm.weight",
        "mtp.0.ffn_norm.weight",
    ):
        tensors[name] = (4096,)
    tensors["mtp.0.e_proj.weight"] = (4096, 4096)
    tensors["mtp.0.h_proj.weight"] = (4096, 4096)
    tensors["mtp.0.ffn_gate_exps.weight"] = routed_shape
    return tensors


def _dspark_metadata() -> dict[str, object]:
    return {
        "general.architecture": "deepseek4-dspark",
        "general.name": "DeepSeek V4 Flash DSpark support",
    }


def _dspark_tensors() -> dict[str, tuple[int, ...]]:
    return {name: (1,) for name in DS4_DSPARK_REQUIRED_TENSORS}


def test_validate_ds4_mtp_accepts_matching_flash_sidecar(tmp_path):
    main = tmp_path / "DeepSeek-V4-Flash.gguf"
    mtp = tmp_path / "DeepSeek-V4-Flash-MTP.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata())
    _write_tiny_gguf(mtp, metadata=_mtp_metadata(), tensors=_mtp_tensors())

    validate_ds4_mtp_compatibility(main, mtp)


def test_validate_ds4_mtp_accepts_dspark_support(tmp_path):
    main = tmp_path / "DeepSeek-V4-Flash.gguf"
    dspark = tmp_path / "DeepSeek-V4-Flash-DSpark-support.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata())
    _write_tiny_gguf(dspark, metadata=_dspark_metadata(), tensors=_dspark_tensors())

    assert detect_ds4_mtp_sidecar_kind(dspark) == "dspark"
    assert is_ds4_mtp_gguf_sidecar(dspark) is True
    validate_ds4_mtp_compatibility(main, dspark)


def test_validate_ds4_mtp_rejects_incomplete_dspark_support(tmp_path):
    main = tmp_path / "DeepSeek-V4-Flash.gguf"
    dspark = tmp_path / "DeepSeek-V4-Flash-DSpark-support.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata())
    _write_tiny_gguf(
        dspark, metadata=_dspark_metadata(), tensors={"mtp.0.main_proj.weight": (1,)}
    )

    with pytest.raises(
        DS4MTPCompatibilityError, match="DSpark support GGUF is missing"
    ):
        validate_ds4_mtp_compatibility(main, dspark)


def test_validate_ds4_mtp_rejects_pro_main_model(tmp_path):
    main = tmp_path / "DeepSeek-V4-Pro.gguf"
    mtp = tmp_path / "DeepSeek-V4-Flash-MTP.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata("DeepSeek V4 Pro"))
    _write_tiny_gguf(mtp, metadata=_mtp_metadata(), tensors=_mtp_tensors())

    with pytest.raises(DS4MTPCompatibilityError, match="DeepSeek V4 Flash"):
        validate_ds4_mtp_compatibility(main, mtp)


def test_validate_ds4_mtp_rejects_main_model_as_sidecar(tmp_path):
    main = tmp_path / "DeepSeek-V4-Flash.gguf"
    bad_mtp = tmp_path / "DeepSeek-V4-Flash-copy.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata())
    _write_tiny_gguf(bad_mtp, metadata=_main_metadata())

    with pytest.raises(DS4MTPCompatibilityError, match="architecture"):
        validate_ds4_mtp_compatibility(main, bad_mtp)


def test_validate_ds4_mtp_rejects_shape_mismatch(tmp_path):
    main = tmp_path / "DeepSeek-V4-Flash.gguf"
    mtp = tmp_path / "DeepSeek-V4-Flash-MTP.gguf"
    _write_tiny_gguf(main, metadata=_main_metadata())
    _write_tiny_gguf(
        mtp,
        metadata=_mtp_metadata(),
        tensors=_mtp_tensors(routed_shape=(7168, 3072, 384)),
    )

    with pytest.raises(DS4MTPCompatibilityError, match="routed expert shape"):
        validate_ds4_mtp_compatibility(main, mtp)
