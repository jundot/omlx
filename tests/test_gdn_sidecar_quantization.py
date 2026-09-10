# SPDX-License-Identifier: Apache-2.0
"""Storage-only precision tests for split-GDN recurrent sidecars."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    mx = None

pytestmark = pytest.mark.skipif(not HAS_MLX, reason="MLX not available")

from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore
from omlx.cache.paged_ssd_cache import PagedSSDCacheManager


def _extracted() -> list[dict]:
    conv = mx.arange(24, dtype=mx.float32).reshape(1, 3, 8).astype(mx.bfloat16)
    # Different row ranges make a wrong reduction axis measurably worse.
    base = mx.arange(1, 97, dtype=mx.float32).reshape(1, 2, 4, 12)
    recurrent = mx.concatenate((base * 0.01, base * 17.0), axis=1)
    return [
        {
            "state": (conv, recurrent),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        },
        {
            "state": (mx.ones((1, 2), dtype=mx.float32),),
            "meta_state": (),
            "class_name": "KVCache",
            "cache_type": "KVCache",
        },
    ]


def _rht_extracted() -> list[dict]:
    """Use a power-of-two recurrent width for the RHT codec tests."""
    base = mx.arange(1, 1 + 1 * 2 * 4 * 16, dtype=mx.float32).reshape(
        1, 2, 4, 16
    )
    recurrent = (base - 64.5) * 0.125
    return [
        {
            "state": (
                mx.arange(8, dtype=mx.float32).reshape(1, 1, 8).astype(mx.bfloat16),
                recurrent,
            ),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        }
    ]


@pytest.mark.parametrize(
    ("mode", "stored_dtype", "max_rel_error"),
    [("fp32", "F32", 0.0), ("bf16", "BF16", 0.01), ("int8", "I8", 0.02)],
)
def test_storage_precision_roundtrip_and_scope(
    tmp_path, mode: str, stored_dtype: str, max_rel_error: float
):
    store = BoundarySnapshotSSDStore(
        tmp_path,
        gdn_sidecar_state_dtype=mode,
    )
    extracted = _extracted()
    source = np.array(extracted[0]["state"][1], copy=True)
    try:
        tensors, metadata = store._serialize_extracted(extracted, "req", 2048)
        layer_info = json.loads(metadata["layer_info"])

        # Only ArraysCache state_1 changes precision. Conv state and unrelated
        # cache tensors retain their original storage dtype.
        assert tensors["layer_0_state_0"][1] == "BF16"
        assert tensors["layer_0_state_1"][1] == stored_dtype
        assert tensors["layer_1_state_0"][1] == "F32"
        assert np.array_equal(np.asarray(extracted[0]["state"][1]), source)

        if mode == "int8":
            assert tensors["layer_0_state_1__scale"][1] == "F32"
            assert tensors["layer_0_state_1__scale"][2] == [1, 4, 4, 1]
            assert (
                layer_info[0]["state_1_storage_codec"]
                == "int8_rowwise_last_axis_v1"
            )
        elif mode == "bf16":
            assert layer_info[0]["state_1_storage_codec"] == "bf16_v1"
        else:
            assert "state_1_storage_codec" not in layer_info[0]

        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        got = restored[0]["state"][1]
        assert got.dtype == mx.float32
        numerator = mx.sqrt(mx.sum((got - extracted[0]["state"][1]) ** 2))
        denominator = mx.sqrt(mx.sum(extracted[0]["state"][1] ** 2))
        assert float(numerator / denominator) <= max_rel_error
        assert store.gdn_state_dequantizations == (0 if mode == "fp32" else 1)
    finally:
        store.shutdown()


def test_zero_rows_have_finite_positive_scale(tmp_path):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        extracted = _extracted()
        extracted[0]["state"] = (
            extracted[0]["state"][0],
            mx.zeros((1, 2, 4, 12), dtype=mx.float32),
        )
        tensors, metadata = store._serialize_extracted(extracted, "zero", 2048)
        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        assert bool(mx.all(mx.isfinite(restored[0]["state"][1])))
        assert float(mx.max(mx.abs(restored[0]["state"][1]))) == 0.0
    finally:
        store.shutdown()


def test_int8_async_staging_file_roundtrip(tmp_path):
    store = BoundarySnapshotSSDStore(
        tmp_path,
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="int8",
    )
    extracted = _extracted()

    def extract(_snapshot):
        return extracted, None

    try:
        assert store.save("durable", 2048, [object()], extract)
        staged = store.take_staged_file("durable", 2048, timeout_s=5.0)
        assert staged is not None
        arrays, metadata = mx.load(str(staged), return_metadata=True)
        mx.eval(*arrays.values())
        assert arrays["layer_0_state_1"].dtype == mx.int8
        assert arrays["layer_0_state_1__scale"].dtype == mx.float32
        assert metadata["gdn_sidecar_format_version"] == "2"

        restored = store.load_file(staged)
        assert restored is not None
        assert restored[0]["state"][1].dtype == mx.float32
        assert store.gdn_state_dequantizations == 1
    finally:
        store.shutdown()


def test_rht_int8_roundtrip_rotates_last_axis_and_restores_fp32(tmp_path):
    """RHT-int8 stores row scales and reconstructs the original axis shape."""
    store = BoundarySnapshotSSDStore(
        tmp_path,
        gdn_sidecar_state_dtype="rht_int8",
    )
    extracted = _rht_extracted()
    source = extracted[0]["state"][1]
    try:
        tensors, metadata = store._serialize_extracted(extracted, "rht", 2048)
        info = json.loads(metadata["layer_info"])

        quantized = tensors["layer_0_state_1"]
        scale = tensors["layer_0_state_1__scale"]
        assert quantized[1] == "I8"
        assert quantized[2] == [1, 2, 4, 16]
        assert scale[1] == "F32"
        assert scale[2] == [1, 2, 4, 1]
        assert info[0]["state_1_storage_codec"] == (
            "rht_int8_rowwise_last_axis_v1"
        )
        assert info[0]["state_1_original_dtype"] == "float32"
        assert info[0]["state_1_rht_seed"] == "0"
        assert info[0]["state_1_rht_dim"] == "16"
        assert metadata["gdn_sidecar_format_version"] == "2"

        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        got = restored[0]["state"][1]
        assert got.dtype == mx.float32
        assert got.shape == source.shape
        numerator = mx.sqrt(mx.sum((got - source) ** 2))
        denominator = mx.sqrt(mx.sum(source**2))
        assert float(numerator / denominator) <= 0.02
        assert store.gdn_state_dequantizations == 1
    finally:
        store.shutdown()


def test_rht_int16_roundtrip_rotates_last_axis_and_restores_fp32(tmp_path):
    """RHT-int16 keeps the storage-only inverse path in fp32."""
    store = BoundarySnapshotSSDStore(
        tmp_path,
        gdn_sidecar_state_dtype="rht_int16",
    )
    extracted = _rht_extracted()
    source = extracted[0]["state"][1]
    try:
        tensors, metadata = store._serialize_extracted(extracted, "rht16", 2048)
        info = json.loads(metadata["layer_info"])

        quantized = tensors["layer_0_state_1"]
        scale = tensors["layer_0_state_1__scale"]
        assert quantized[1] == "I16"
        assert quantized[2] == [1, 2, 4, 16]
        assert scale[1] == "F32"
        assert scale[2] == [1, 2, 4, 1]
        assert info[0]["state_1_storage_codec"] == (
            "rht_int16_rowwise_last_axis_v1"
        )
        assert info[0]["state_1_original_dtype"] == "float32"
        assert info[0]["state_1_rht_seed"] == "0"
        assert info[0]["state_1_rht_dim"] == "16"
        assert metadata["gdn_sidecar_format_version"] == "2"

        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        got = restored[0]["state"][1]
        assert got.dtype == mx.float32
        assert got.shape == source.shape
        numerator = mx.sqrt(mx.sum((got - source) ** 2))
        denominator = mx.sqrt(mx.sum(source**2))
        assert float(numerator / denominator) <= 0.001
        assert store.gdn_state_dequantizations == 1
    finally:
        store.shutdown()


def test_rht_int8_encoding_is_deterministic_and_codec_is_distinct(tmp_path):
    """Fixed RHT signs make repeated snapshots byte-identical."""
    extracted = _rht_extracted()
    first = BoundarySnapshotSSDStore(tmp_path / "first", gdn_sidecar_state_dtype="rht_int8")
    second = BoundarySnapshotSSDStore(
        tmp_path / "second", gdn_sidecar_state_dtype="rht_int8"
    )
    try:
        tensors_1, metadata_1 = first._serialize_extracted(extracted, "rht", 2048)
        tensors_2, metadata_2 = second._serialize_extracted(extracted, "rht", 2048)
        assert tensors_1 == tensors_2
        assert metadata_1 == metadata_2

        plain = BoundarySnapshotSSDStore(tmp_path / "plain", gdn_sidecar_state_dtype="int8")
        try:
            plain_tensors, plain_metadata = plain._serialize_extracted(
                extracted, "rht", 2048
            )
            plain_info = json.loads(plain_metadata["layer_info"])
            rht_info = json.loads(metadata_1["layer_info"])
            assert rht_info[0]["state_1_storage_codec"] != plain_info[0][
                "state_1_storage_codec"
            ]
            assert tensors_1["layer_0_state_1"][1] == "I8"
            assert plain_tensors["layer_0_state_1"][1] == "I8"
        finally:
            plain.shutdown()
    finally:
        first.shutdown()
        second.shutdown()


def test_rht_int8_async_staging_file_roundtrip(tmp_path):
    store = BoundarySnapshotSSDStore(
        tmp_path,
        pending_max_bytes=1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    extracted = _rht_extracted()

    def extract(_snapshot):
        return extracted, None

    try:
        assert store.save("rht-durable", 2048, [object()], extract)
        staged = store.take_staged_file("rht-durable", 2048, timeout_s=5.0)
        assert staged is not None
        arrays, metadata = mx.load(str(staged), return_metadata=True)
        mx.eval(*arrays.values())
        assert arrays["layer_0_state_1"].dtype == mx.int8
        assert arrays["layer_0_state_1__scale"].dtype == mx.float32
        info = json.loads(metadata["layer_info"])
        assert info[0]["state_1_storage_codec"] == (
            "rht_int8_rowwise_last_axis_v1"
        )

        restored = store.load_file(staged)
        assert restored is not None
        assert restored[0]["state"][1].dtype == mx.float32
        assert restored[0]["state"][1].shape == extracted[0]["state"][1].shape
        assert store.gdn_state_dequantizations == 1
    finally:
        store.shutdown()


def test_rht_int8_corrupt_metadata_fails_closed(tmp_path):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        tensors, metadata = store._serialize_extracted(_rht_extracted(), "bad-rht", 2048)
        info = json.loads(metadata["layer_info"])
        info[0]["state_1_storage_codec"] = "rht_int8_unknown_v1"
        unknown = dict(metadata)
        unknown["layer_info"] = json.dumps(info)
        with pytest.raises(ValueError, match="unsupported GDN storage codec"):
            store._deserialize(tensors, unknown)

        info[0]["state_1_storage_codec"] = "rht_int8_rowwise_last_axis_v1"
        bad_shape = dict(tensors)
        raw, dtype, _shape = bad_shape["layer_0_state_1__scale"]
        # Keep the raw byte count valid so validation reaches the scale-shape
        # guard rather than failing during tensor reshape.
        bad_shape["layer_0_state_1__scale"] = (raw, dtype, [1, 2, 4, 1, 1])
        malformed = dict(metadata)
        malformed["layer_info"] = json.dumps(info)
        with pytest.raises(ValueError, match="invalid GDN .* scale"):
            store._deserialize(bad_shape, malformed)

        missing_rht_metadata = dict(metadata)
        missing_info = json.loads(metadata["layer_info"])
        missing_info[0].pop("state_1_rht_seed")
        missing_rht_metadata["layer_info"] = json.dumps(missing_info)
        with pytest.raises(ValueError, match="missing GDN RHT metadata"):
            store._deserialize(tensors, missing_rht_metadata)

        wrong_seed = dict(metadata)
        wrong_seed_info = json.loads(metadata["layer_info"])
        wrong_seed_info[0]["state_1_rht_seed"] = "1"
        wrong_seed["layer_info"] = json.dumps(wrong_seed_info)
        with pytest.raises(ValueError, match="unsupported GDN RHT seed"):
            store._deserialize(tensors, wrong_seed)
    finally:
        store.shutdown()


def test_missing_scale_and_future_format_fail_closed(tmp_path):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        tensors, metadata = store._serialize_extracted(_extracted(), "bad", 2048)
        missing_scale = dict(tensors)
        missing_scale.pop("layer_0_state_1__scale")
        with pytest.raises(ValueError, match="missing GDN int8 scale"):
            store._deserialize(missing_scale, metadata)

        future = dict(metadata)
        future["gdn_sidecar_format_version"] = "999"
        assert store._deserialize(tensors, future) is None
    finally:
        store.shutdown()


def test_legacy_fp32_metadata_remains_readable(tmp_path):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        tensors, metadata = store._serialize_extracted(_extracted(), "old", 2048)
        info = json.loads(metadata["layer_info"])
        # Rebuild the historical fp32 representation with no format/codec keys.
        recurrent = _extracted()[0]["state"][1]
        from omlx.cache.paged_ssd_cache import _extract_tensor_bytes

        tensors["layer_0_state_1"] = _extract_tensor_bytes(recurrent)
        tensors.pop("layer_0_state_1__scale")
        info[0].pop("state_1_storage_codec")
        info[0].pop("state_1_original_dtype")
        metadata.pop("gdn_sidecar_format_version")
        metadata["layer_info"] = json.dumps(info)

        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        assert restored[0]["state"][1].dtype == mx.float32
        assert np.array_equal(
            np.asarray(restored[0]["state"][1]),
            np.asarray(recurrent),
        )
    finally:
        store.shutdown()


def test_reduced_precision_uses_distinct_sidecar_signature(tmp_path):
    common = dict(
        cache_dir=tmp_path,
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=2,
        expected_block_size=2048,
        expected_layer_cache_types=["KVCache", "ArraysCache"],
        gdn_ssd_split_enabled=True,
    )
    fp32 = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="fp32")
    int8 = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="int8")
    rht_int8 = PagedSSDCacheManager(
        **common, gdn_sidecar_state_dtype="rht_int8"
    )
    try:
        kwargs = dict(
            model_name="model",
            num_layers=2,
            block_size=2048,
            layer_cache_types=["KVCache", "ArraysCache"],
        )
        # Main KV compatibility is precision-independent; only the durable
        # recurrent sidecar namespace changes.
        assert fp32.cache_signature_for(**kwargs) == int8.cache_signature_for(**kwargs)
        fp32_signature = fp32.gdn_cache_signature_for(**kwargs)
        int8_signature = int8.gdn_cache_signature_for(**kwargs)
        rht_signature = rht_int8.gdn_cache_signature_for(**kwargs)
        assert fp32_signature != int8_signature
        assert int8_signature != rht_signature
        assert "gdn_sidecar_state_dtype" not in fp32_signature
        assert json.loads(int8_signature)["gdn_sidecar_state_dtype"] == "int8"
        assert json.loads(rht_signature)["gdn_sidecar_state_dtype"] == "rht_int8"

        # An int8-configured manager can still consume a pre-feature fp32
        # sidecar when no int8 checkpoint has been written for that block.
        staged = tmp_path / "legacy-staged.safetensors"
        staged.write_bytes(b"legacy-fp32-sidecar")
        block_hash = b"legacy-block-hash"
        assert (
            int8.commit_gdn_checkpoint_file(
                block_hash,
                staged,
                token_count=2048,
                model_name="model",
                cache_signature=fp32_signature,
                block_size=2048,
            )
            is not None
        )
        assert int8.has_gdn_checkpoint(block_hash, int8_signature)
        legacy_lookup = int8.get_gdn_checkpoint_file_with_diagnostic(
            block_hash, int8_signature
        )
        assert legacy_lookup is not None
        assert legacy_lookup.requested_state_dtype == "int8"
        assert legacy_lookup.effective_state_codec == "fp32"
        assert legacy_lookup.used_legacy_fp32_fallback is True
        assert int8.gdn_legacy_fp32_fallbacks == 1

        # A current reduced namespace must win over the compatibility fallback.
        current_staged = tmp_path / "current-int8-staged.safetensors"
        current_staged.write_bytes(b"current-int8-sidecar")
        assert (
            int8.commit_gdn_checkpoint_file(
                block_hash,
                current_staged,
                token_count=2048,
                model_name="model",
                cache_signature=int8_signature,
                block_size=2048,
            )
            is not None
        )
        current_lookup = int8.get_gdn_checkpoint_file_with_diagnostic(
            block_hash, int8_signature
        )
        assert current_lookup is not None
        assert current_lookup.requested_state_dtype == "int8"
        assert (
            current_lookup.effective_state_codec
            == "int8_rowwise_last_axis_v1"
        )
        assert current_lookup.used_legacy_fp32_fallback is False
        assert int8.gdn_legacy_fp32_fallbacks == 1

        rht_block_hash = b"rht-only-block-hash"
        rht_staged = tmp_path / "current-rht-staged.safetensors"
        rht_staged.write_bytes(b"current-rht-sidecar")
        assert (
            int8.commit_gdn_checkpoint_file(
                rht_block_hash,
                rht_staged,
                token_count=2048,
                model_name="model",
                cache_signature=rht_signature,
                block_size=2048,
            )
            is not None
        )
        # The plain INT8 lookup candidates are INT8 then FP32, never RHT.
        assert (
            int8.get_gdn_checkpoint_file_with_diagnostic(
                rht_block_hash, int8_signature
            )
            is None
        )
        rht_lookup = int8.get_gdn_checkpoint_file_with_diagnostic(
            rht_block_hash, rht_signature
        )
        assert rht_lookup is not None
        assert rht_lookup.requested_state_dtype == "rht_int8"
        assert (
            rht_lookup.effective_state_codec
            == "rht_int8_rowwise_last_axis_v1"
        )
        assert rht_lookup.used_legacy_fp32_fallback is False
    finally:
        fp32.close()
        int8.close()
        rht_int8.close()


def test_invalid_precision_rejected(tmp_path):
    with pytest.raises(ValueError, match="gdn_sidecar_state_dtype"):
        BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="fp8")


def test_rht_int16_uses_distinct_sidecar_signature_namespace(tmp_path):
    common = dict(
        cache_dir=tmp_path,
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=1,
        expected_block_size=2048,
        expected_layer_cache_types=["ArraysCache"],
        gdn_ssd_split_enabled=True,
    )
    rht8 = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int8")
    rht16 = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int16")
    try:
        kwargs = dict(
            model_name="model",
            num_layers=1,
            block_size=2048,
            layer_cache_types=["ArraysCache"],
        )
        sig8 = rht8.gdn_cache_signature_for(**kwargs)
        sig16 = rht16.gdn_cache_signature_for(**kwargs)
        assert sig8 != sig16
        assert json.loads(sig8)["gdn_sidecar_state_dtype"] == "rht_int8"
        assert json.loads(sig16)["gdn_sidecar_state_dtype"] == "rht_int16"
        assert rht8.cache_signature_for(**kwargs) == rht16.cache_signature_for(
            **kwargs
        )
    finally:
        rht8.close()
        rht16.close()


# --- P0-2: fail-closed on non-finite input and mixed-writer metadata ---


def _recurrent_extracted(recurrent) -> list[dict]:
    """Wrap a recurrent tensor in the minimal Arrays-family layer shape."""
    return [
        {
            "state": (
                mx.zeros((1, 1, 8), dtype=mx.bfloat16),
                recurrent,
            ),
            "meta_state": (),
            "class_name": "ArraysCache",
            "cache_type": "ArraysCache",
        }
    ]


@pytest.mark.parametrize("mode", ["int8", "rht_int8", "rht_int16", "bf16"])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_source_is_refused_before_encoding(tmp_path, mode, bad):
    """A corrupt recurrent state must not become a well-formed sidecar.

    NaN survives round/clip into an undefined int8 value while its row scale
    can still look finite, so restore-side validation alone would not catch it.
    """
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype=mode)
    try:
        values = mx.arange(1, 1 + 2 * 8, dtype=mx.float32).reshape(1, 1, 2, 8)
        poisoned = mx.concatenate(
            [
                mx.full((1, 1, 1, 8), bad, dtype=mx.float32),
                values[:, :, 1:, :],
            ],
            axis=2,
        )
        with pytest.raises(ValueError, match="non-finite GDN recurrent state"):
            store._serialize_extracted(
                _recurrent_extracted(poisoned), "poison", 2048
            )
        assert store.gdn_encode_failures == 1

        # save() degrades the same failure to a safe miss, leaving no file.
        assert (
            store.save("poison", 2048, [object()], lambda _s: (
                _recurrent_extracted(poisoned),
                None,
            ))
            is False
        )
        assert store.gdn_encode_failures == 2
        assert store.has("poison", 2048) is False
    finally:
        store.shutdown()


@pytest.mark.parametrize("mode", ["int8", "rht_int8", "rht_int16"])
def test_non_finite_check_does_not_reject_extreme_finite_states(tmp_path, mode):
    """All-zero, subnormal and wide-dynamic-range rows stay encodable."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype=mode)
    try:
        rows = mx.array(
            [
                [0.0] * 8,
                [5e-324, 1e-38, -1e-38, 0.0, 1e-30, -1e-30, 2e-38, -5e-324],
                [1e30, -1e30, 1.0, -1.0, 0.5, -0.5, 1e20, -1e20],
                [1.0] * 8,
            ],
            dtype=mx.float32,
        ).reshape(1, 1, 4, 8)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), "extremes", 2048
        )
        assert store.gdn_encode_failures == 0
        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        state = restored[0]["state"][1]
        assert state.dtype == mx.float32
        assert bool(mx.all(mx.isfinite(state)))
        # The all-zero row must come back as exact zeros, not scale noise.
        assert bool(mx.all(state[0, 0, 0] == 0.0))
    finally:
        store.shutdown()


def test_rht_overflow_on_near_max_fp32_row_is_refused_at_encode(tmp_path):
    """The RHT is not per-element magnitude preserving.

    A row near the fp32 maximum sums to inf under the Hadamard butterfly even
    though every source element is finite, so the encoder validates the row
    scale as well as the source.
    """
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        rows = mx.full((1, 1, 2, 8), 3.4e38, dtype=mx.float32)
        assert bool(mx.all(mx.isfinite(rows)))
        with pytest.raises(ValueError, match="non-finite GDN row scale"):
            store._serialize_extracted(
                _recurrent_extracted(rows), "overflow", 2048
            )
        assert store.gdn_encode_failures == 1
    finally:
        store.shutdown()


def test_plain_int8_tolerates_near_max_fp32_row(tmp_path):
    """Without the rotation the same row quantizes without overflow."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        rows = mx.full((1, 1, 2, 8), 3.4e38, dtype=mx.float32)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), "nearmax", 2048
        )
        assert store.gdn_encode_failures == 0
        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        assert bool(mx.all(mx.isfinite(restored[0]["state"][1])))
    finally:
        store.shutdown()


@pytest.mark.parametrize(
    ("np_dtype", "dtype_str"),
    # F64 is absent from the safetensors dtype table entirely, so it fails
    # closed one layer earlier; only representable dtypes reach this check.
    [(np.float16, "F16"), (np.uint16, "BF16"), (np.int32, "I32")],
)
def test_scale_dtype_must_be_exactly_fp32(tmp_path, np_dtype, dtype_str):
    """The codec contract stores fp32 scales; a narrower dtype changes rows."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        tensors, metadata = store._serialize_extracted(_extracted(), "scale", 2048)
        key = "layer_0_state_1__scale"
        _raw, _dtype, shape = tensors[key]
        count = int(np.prod(shape))
        recast = dict(tensors)
        recast[key] = (
            np.ones(count, dtype=np_dtype).tobytes(),
            dtype_str,
            shape,
        )
        with pytest.raises(ValueError, match="invalid GDN .* scale dtype"):
            store._deserialize(recast, metadata)
        assert store.gdn_decode_failures == 1
    finally:
        store.shutdown()


@pytest.mark.parametrize("mode", ["int8", "rht_int8", "rht_int16", "bf16"])
@pytest.mark.parametrize("original", [None, "bfloat16", "float16", ""])
def test_reduced_codec_requires_float32_source_dtype(tmp_path, mode, original):
    extracted = (
        _rht_extracted()
        if mode in {"rht_int8", "rht_int16"}
        else _extracted()
    )
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype=mode)
    try:
        tensors, metadata = store._serialize_extracted(extracted, "dtype", 2048)
        info = json.loads(metadata["layer_info"])
        if original is None:
            info[0].pop("state_1_original_dtype")
        else:
            info[0]["state_1_original_dtype"] = original
        tampered = dict(metadata)
        tampered["layer_info"] = json.dumps(info)
        with pytest.raises(ValueError, match="invalid GDN source dtype"):
            store._deserialize(tensors, tampered)
    finally:
        store.shutdown()


def test_rht_metadata_rejected_under_plain_int8_codec(tmp_path):
    """Plain int8 carrying RHT keys would restore in the wrong basis."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        tensors, metadata = store._serialize_extracted(_rht_extracted(), "mix", 2048)
        info = json.loads(metadata["layer_info"])
        assert info[0]["state_1_storage_codec"] == "int8_rowwise_last_axis_v1"
        info[0]["state_1_rht_seed"] = "0"
        info[0]["state_1_rht_dim"] = "16"
        mixed = dict(metadata)
        mixed["layer_info"] = json.dumps(info)
        with pytest.raises(
            ValueError, match="RHT metadata present under non-RHT codec"
        ):
            store._deserialize(tensors, mixed)
    finally:
        store.shutdown()


@pytest.mark.parametrize(
    ("seed", "dim", "match"),
    [
        ("0", "12", "not a power of two"),
        ("0", "-16", "invalid GDN RHT metadata literals"),
        ("0", "16.0", "invalid GDN RHT metadata literals"),
        ("0", " 16", "invalid GDN RHT metadata literals"),
        ("0", "1_6", "invalid GDN RHT metadata literals"),
        ("0", "32", "invalid GDN RHT dimension"),
        ("-1", "16", "invalid GDN RHT metadata literals"),
        ("2", "16", "unsupported GDN RHT seed"),
    ],
)
def test_rht_metadata_literals_and_dimension_fail_closed(
    tmp_path, seed, dim, match
):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        tensors, metadata = store._serialize_extracted(_rht_extracted(), "meta", 2048)
        info = json.loads(metadata["layer_info"])
        info[0]["state_1_rht_seed"] = seed
        info[0]["state_1_rht_dim"] = dim
        tampered = dict(metadata)
        tampered["layer_info"] = json.dumps(info)
        with pytest.raises(ValueError, match=match):
            store._deserialize(tensors, tampered)
    finally:
        store.shutdown()


def test_non_int8_payload_under_int8_codec_fails_closed(tmp_path):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        tensors, metadata = store._serialize_extracted(_rht_extracted(), "payload", 2048)
        key = "layer_0_state_1"
        _raw, _dtype, shape = tensors[key]
        count = int(np.prod(shape))
        widened = dict(tensors)
        widened[key] = (np.ones(count, dtype=np.float32).tobytes(), "F32", shape)
        with pytest.raises(ValueError, match="invalid GDN int8 payload dtype"):
            store._deserialize(widened, metadata)
    finally:
        store.shutdown()


def test_disk_restore_path_applies_the_same_validation(tmp_path):
    """_reconstruct_from_safetensors must fail closed like the pending path."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        tensors, metadata = store._serialize_extracted(_rht_extracted(), "disk", 2048)
        arrays = {
            name: mx.array(
                np.frombuffer(
                    raw,
                    dtype={"I8": np.int8, "F32": np.float32, "BF16": np.uint16}[dtype],
                ).reshape(shape)
            )
            for name, (raw, dtype, shape) in tensors.items()
            if dtype in {"I8", "F32"}
        }
        info = json.loads(metadata["layer_info"])
        info[0]["state_1_original_dtype"] = "bfloat16"
        tampered = dict(metadata)
        tampered["layer_info"] = json.dumps(info)
        with pytest.raises(ValueError, match="invalid GDN source dtype"):
            store._reconstruct_from_safetensors(arrays, tampered)
        assert store.gdn_decode_failures == 1
    finally:
        store.shutdown()


def test_unrepresentable_scale_dtype_is_still_counted_as_a_decode_failure(tmp_path):
    """F64 has no safetensors mapping, so it fails before the dtype check."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="int8")
    try:
        tensors, metadata = store._serialize_extracted(_extracted(), "f64", 2048)
        key = "layer_0_state_1__scale"
        _raw, _dtype, shape = tensors[key]
        recast = dict(tensors)
        recast[key] = (
            np.ones(int(np.prod(shape)), dtype=np.float64).tobytes(),
            "F64",
            shape,
        )
        with pytest.raises(KeyError):
            store._deserialize(recast, metadata)
        assert store.gdn_decode_failures == 1
    finally:
        store.shutdown()


def test_rejected_sidecar_degrades_to_a_cache_miss(tmp_path):
    """A fail-closed decode must surface as None, not as an exception.

    Uses the committed file rather than the pending buffer: the background
    writer can drain a pending entry at any point, after which load() would
    read the intact file on disk and the corruption would not be exercised.
    """
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        def extract(_snapshot):
            return _rht_extracted(), None

        assert store.save("miss", 2048, [object()], extract) is True
        staged = store.take_staged_file("miss", 2048, timeout_s=5.0)
        assert staged is not None
        assert store.load_file(staged) is not None
        assert store.gdn_decode_failures == 0

        arrays, metadata = mx.load(str(staged), return_metadata=True)
        # Materialize before overwriting the same backing file. MLX 0.32.2
        # otherwise retains read primitives that fail after the truncate.
        mx.eval(*arrays.values())
        info = json.loads(metadata["layer_info"])
        info[0]["state_1_original_dtype"] = "bfloat16"
        metadata["layer_info"] = json.dumps(info)
        mx.save_safetensors(str(staged), arrays, metadata=metadata)

        assert store.load_file(staged) is None
        assert store.gdn_decode_failures == 1
    finally:
        store.shutdown()


# --- P0-3: RHT capability gating on unsupported widths ---


@pytest.mark.parametrize("dim", [1, 2, 16, 128, 1024])
def test_rht_encodes_supported_power_of_two_widths(tmp_path, dim):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        rows = mx.arange(1, 1 + 2 * dim, dtype=mx.float32).reshape(1, 1, 2, dim)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), f"dim{dim}", 2048
        )
        info = json.loads(metadata["layer_info"])
        assert tensors["layer_0_state_1"][1] == "I8"
        assert info[0]["state_1_storage_codec"] == "rht_int8_rowwise_last_axis_v1"
        assert info[0]["state_1_rht_dim"] == str(dim)
        assert store.gdn_capability_fallbacks == 0
        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        assert restored[0]["state"][1].dtype == mx.float32
    finally:
        store.shutdown()


@pytest.mark.parametrize("dim", [3, 12, 20, 96])
def test_rht_unsupported_width_stores_plain_fp32(tmp_path, dim):
    """An unsupported width must degrade visibly, not disable the sidecar."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        rows = mx.arange(1, 1 + 2 * dim, dtype=mx.float32).reshape(1, 1, 2, dim)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), f"dim{dim}", 2048
        )
        info = json.loads(metadata["layer_info"])
        # Stored verbatim: no codec key, so the restore path passes it through.
        assert tensors["layer_0_state_1"][1] == "F32"
        assert "state_1_storage_codec" not in info[0]
        assert "layer_0_state_1__scale" not in tensors
        assert store.gdn_capability_fallbacks == 1

        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        got = restored[0]["state"][1]
        assert got.dtype == mx.float32
        assert bool(mx.all(got == rows))
        # No dequantization happened, which is exactly the signal a quality run
        # treats as INVALID for a codec arm.
        assert store.gdn_state_dequantizations == 0
    finally:
        store.shutdown()


def test_rht_capability_error_is_logged_once_per_store(tmp_path, caplog):
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        rows = mx.ones((1, 1, 2, 12), dtype=mx.float32)
        layers = _recurrent_extracted(rows) * 4
        with caplog.at_level("ERROR"):
            for i in range(3):
                store._serialize_extracted(layers, f"rep{i}", 2048)
        errors = [
            r for r in caplog.records if "GDN RHT sidecars unsupported" in r.message
        ]
        assert len(errors) == 1, "a 48-layer model must not log this per layer"
        # Every skipped tensor is still counted: 4 layers x 3 checkpoints.
        assert store.gdn_capability_fallbacks == 12
    finally:
        store.shutdown()


@pytest.mark.parametrize("mode", ["int8", "bf16"])
@pytest.mark.parametrize("dim", [3, 12])
def test_width_constraint_applies_only_to_rht(tmp_path, mode, dim):
    """Only the rotation needs a power-of-two width; the others do not."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype=mode)
    try:
        rows = mx.arange(1, 1 + 2 * dim, dtype=mx.float32).reshape(1, 1, 2, dim)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), f"{mode}{dim}", 2048
        )
        info = json.loads(metadata["layer_info"])
        assert tensors["layer_0_state_1"][1] == ("I8" if mode == "int8" else "BF16")
        assert "state_1_storage_codec" in info[0]
        assert store.gdn_capability_fallbacks == 0
    finally:
        store.shutdown()


def test_zero_dim_recurrent_state_never_reaches_the_codec(tmp_path):
    """Zero-sized states are handled by the shape marker path, not the codec."""
    store = BoundarySnapshotSSDStore(tmp_path, gdn_sidecar_state_dtype="rht_int8")
    try:
        rows = mx.zeros((1, 1, 2, 0), dtype=mx.float32)
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(rows), "zerodim", 2048
        )
        info = json.loads(metadata["layer_info"])
        assert "zero_dim_1" in info[0]
        assert "state_1_storage_codec" not in info[0]
        assert store.gdn_capability_fallbacks == 0
        restored = store._deserialize(tensors, metadata)
        assert restored is not None
        assert tuple(restored[0]["state"][1].shape) == (1, 1, 2, 0)
    finally:
        store.shutdown()


def test_rht_sign_cache_is_bounded(tmp_path):
    from omlx.cache.boundary_snapshot_store import _gdn_rht_sign_values

    assert _gdn_rht_sign_values.cache_info().maxsize is not None
    # A power-of-two width yields a deterministic +/-1 diagonal of that length.
    signs = _gdn_rht_sign_values(128, 0)
    assert len(signs) == 128
    assert set(signs) == {1.0, -1.0}
    assert _gdn_rht_sign_values(128, 0) is signs


# --- P0-4: pending / disk / restart parity ---


def _codec_extracted(mode: str) -> list[dict]:
    return (
        _rht_extracted()
        if mode in {"rht_int8", "rht_int16", "int8", "bf16"}
        else _extracted()
    )


@pytest.mark.parametrize(
    "mode", ["fp32", "bf16", "int8", "rht_int8", "rht_int16"]
)
def test_pending_raw_and_durable_file_restore_identically(tmp_path, mode):
    """The in-memory buffer and the committed file must agree bit for bit.

    They run different decoders (``_deserialize`` over raw bytes vs
    ``_reconstruct_from_safetensors`` over mx.load arrays), so a codec change
    can easily land in one and not the other.
    """
    store = BoundarySnapshotSSDStore(
        tmp_path,
        pending_max_bytes=64 * 1024**2,
        gdn_sidecar_state_dtype=mode,
    )
    extracted = _codec_extracted(mode)
    try:
        # Holding the writer lock keeps the item in _pending_writes, which is
        # otherwise drained by the background thread at an arbitrary moment.
        with store._writer_busy:
            assert store.save(
                "parity", 2048, [object()],
                lambda _s: (extracted, None),
            )
            pending = store.load("parity", 2048)
            assert pending is not None, "expected the pending buffer to serve this"

        staged = store.take_staged_file("parity", 2048, timeout_s=5.0)
        assert staged is not None
        durable = store.load_file(staged)
        assert durable is not None
        # load_file must detach mx.load's lazy arrays from the ephemeral
        # promotion file before ownership returns to the caller.
        staged.unlink()

        pending_state = pending[0]["state"][1]
        durable_state = durable[0]["state"][1]
        assert pending_state.dtype == durable_state.dtype == mx.float32
        assert pending_state.shape == durable_state.shape
        assert bool(mx.all(pending_state == durable_state)), (
            "pending and durable restores diverged"
        )
    finally:
        store.shutdown()


def test_pending_decode_counts_dequantizations_once_per_state(tmp_path):
    store = BoundarySnapshotSSDStore(
        tmp_path,
        pending_max_bytes=64 * 1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    try:
        with store._writer_busy:
            assert store.save(
                "count", 2048, [object()],
                lambda _s: (_rht_extracted(), None),
            )
            assert store.gdn_state_dequantizations == 0
            assert store.load("count", 2048) is not None
            assert store.gdn_state_dequantizations == 1
            assert store.load("count", 2048) is not None
            assert store.gdn_state_dequantizations == 2
    finally:
        store.shutdown()


def test_cancelled_pending_write_leaves_no_sidecar_or_dequantization(tmp_path):
    store = BoundarySnapshotSSDStore(
        tmp_path,
        pending_max_bytes=64 * 1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    try:
        assert store.save(
            "cancel", 2048, [object()],
            lambda _s: (_rht_extracted(), None),
        )
        store.cleanup_request("cancel")
        assert store.has("cancel", 2048) is False
        assert store.load("cancel", 2048) is None
        assert store.gdn_state_dequantizations == 0
        assert store.gdn_decode_failures == 0
    finally:
        store.shutdown()


def _commit_sidecar(manager, block_hash, signature, payload, tmp_path, name):
    staged = tmp_path / f"{name}.safetensors"
    staged.write_bytes(payload)
    committed = manager.commit_gdn_checkpoint_file(
        block_hash,
        staged,
        token_count=2048,
        model_name="model",
        cache_signature=signature,
        block_size=2048,
    )
    assert committed is not None
    return committed


def test_restart_index_scan_selects_the_requested_codec_namespace(tmp_path):
    """All three namespaces coexist for one block and survive a restart.

    The startup scan rebuilds the index from files on disk, so an exact
    namespace match has to be re-derivable without the writing process.
    """
    cache_dir = tmp_path / "cache"
    common = dict(
        cache_dir=cache_dir,
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=2,
        expected_block_size=2048,
        expected_layer_cache_types=["KVCache", "ArraysCache"],
        gdn_ssd_split_enabled=True,
    )
    sig_kwargs = dict(
        model_name="model",
        num_layers=2,
        block_size=2048,
        layer_cache_types=["KVCache", "ArraysCache"],
    )
    block_hash = b"shared-block-hash"
    payloads = {
        "fp32": b"legacy-fp32-payload",
        "int8": b"plain-int8-payload",
        "rht_int8": b"rht-int8-payload",
    }

    writer = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int8")
    signatures = {}
    try:
        for mode in ("fp32", "int8", "rht_int8"):
            manager = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype=mode)
            try:
                signatures[mode] = manager.gdn_cache_signature_for(**sig_kwargs)
            finally:
                manager.close()
        for mode, payload in payloads.items():
            _commit_sidecar(
                writer, block_hash, signatures[mode], payload, tmp_path, mode
            )
    finally:
        writer.close()

    # Fresh managers: the index comes from the startup scan, not from memory.
    for mode in ("fp32", "int8", "rht_int8"):
        restarted = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype=mode)
        try:
            lookup = restarted.get_gdn_checkpoint_file_with_diagnostic(
                block_hash, signatures[mode]
            )
            assert lookup is not None, mode
            assert lookup.file_path.read_bytes() == payloads[mode], mode
            assert lookup.requested_state_dtype == mode
            assert lookup.used_legacy_fp32_fallback is False, mode
            assert restarted.gdn_legacy_fp32_fallbacks == 0
        finally:
            restarted.close()


def test_restart_falls_back_to_legacy_only_when_current_namespace_is_absent(
    tmp_path,
):
    cache_dir = tmp_path / "cache"
    common = dict(
        cache_dir=cache_dir,
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=2,
        expected_block_size=2048,
        expected_layer_cache_types=["KVCache", "ArraysCache"],
        gdn_ssd_split_enabled=True,
    )
    sig_kwargs = dict(
        model_name="model",
        num_layers=2,
        block_size=2048,
        layer_cache_types=["KVCache", "ArraysCache"],
    )
    block_hash = b"legacy-only-block"

    writer = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="fp32")
    try:
        fp32_signature = writer.gdn_cache_signature_for(**sig_kwargs)
        _commit_sidecar(
            writer, block_hash, fp32_signature, b"legacy-only", tmp_path, "legacy"
        )
    finally:
        writer.close()

    restarted = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int8")
    try:
        rht_signature = restarted.gdn_cache_signature_for(**sig_kwargs)
        lookup = restarted.get_gdn_checkpoint_file_with_diagnostic(
            block_hash, rht_signature
        )
        assert lookup is not None
        assert lookup.file_path.read_bytes() == b"legacy-only"
        assert lookup.requested_state_dtype == "rht_int8"
        assert lookup.effective_state_codec == "fp32"
        assert lookup.used_legacy_fp32_fallback is True
        assert restarted.gdn_legacy_fp32_fallbacks == 1
    finally:
        restarted.close()

    # A plain int8 request must never be served the RHT namespace, restart or
    # not: the payload would be restored without the inverse rotation.
    writer = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int8")
    try:
        rht_signature = writer.gdn_cache_signature_for(**sig_kwargs)
        _commit_sidecar(
            writer, b"rht-only-block", rht_signature, b"rht-only", tmp_path, "rht"
        )
    finally:
        writer.close()

    restarted = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="int8")
    try:
        int8_signature = restarted.gdn_cache_signature_for(**sig_kwargs)
        assert (
            restarted.get_gdn_checkpoint_file_with_diagnostic(
                b"rht-only-block", int8_signature
            )
            is None
        )
    finally:
        restarted.close()


def test_restart_restores_real_rht_payload_end_to_end(tmp_path):
    """A committed RHT sidecar decodes correctly in a fresh manager+store."""
    cache_dir = tmp_path / "cache"
    common = dict(
        cache_dir=cache_dir,
        max_size_bytes=1024**3,
        expected_model_name="model",
        expected_num_layers=1,
        expected_block_size=2048,
        expected_layer_cache_types=["ArraysCache"],
        gdn_ssd_split_enabled=True,
    )
    sig_kwargs = dict(
        model_name="model",
        num_layers=1,
        block_size=2048,
        layer_cache_types=["ArraysCache"],
    )
    extracted = _rht_extracted()
    source = extracted[0]["state"][1]
    block_hash = b"real-rht-block"

    store = BoundarySnapshotSSDStore(
        cache_dir, pending_max_bytes=64 * 1024**2,
        gdn_sidecar_state_dtype="rht_int8",
    )
    manager = PagedSSDCacheManager(**common, gdn_sidecar_state_dtype="rht_int8")
    try:
        assert store.save(
            "real", 2048, [object()], lambda _s: (extracted, None)
        )
        staged = store.take_staged_file("real", 2048, timeout_s=5.0)
        assert staged is not None
        signature = manager.gdn_cache_signature_for(**sig_kwargs)
        assert (
            manager.commit_gdn_checkpoint_file(
                block_hash,
                staged,
                token_count=2048,
                model_name="model",
                cache_signature=signature,
                block_size=2048,
            )
            is not None
        )
    finally:
        manager.close()
        store.shutdown()

    restarted_manager = PagedSSDCacheManager(
        **common, gdn_sidecar_state_dtype="rht_int8"
    )
    restarted_store = BoundarySnapshotSSDStore(
        cache_dir, gdn_sidecar_state_dtype="rht_int8"
    )
    try:
        signature = restarted_manager.gdn_cache_signature_for(**sig_kwargs)
        path = restarted_manager.get_gdn_checkpoint_file(block_hash, signature)
        assert path is not None
        restored = restarted_store.load_file(path)
        assert restored is not None
        got = restored[0]["state"][1]
        assert got.dtype == mx.float32
        assert got.shape == source.shape
        rel = float(
            mx.sqrt(mx.sum((got - source) ** 2)) / mx.sqrt(mx.sum(source**2))
        )
        assert rel <= 0.02
        assert restarted_store.gdn_state_dequantizations == 1
        assert restarted_store.gdn_decode_failures == 0
    finally:
        restarted_store.shutdown()
        restarted_manager.close()


# --- P0-5: golden transform values and cross-process determinism ---

# Pinning these makes any change to the sign derivation or the transform a
# visible test failure rather than a silent re-encoding of every sidecar.
_GOLDEN_SIGN_DIGEST_DIM128_SEED0 = (
    "2fcb7f9dc27a9a294821b0afce21462ea63cca8acbd2bf65aa3c9757fd397e6c"
)


def _sign_digest(dim: int, seed: int) -> str:
    from omlx.cache.boundary_snapshot_store import _gdn_rht_sign_values

    signs = _gdn_rht_sign_values(dim, seed)
    return hashlib.sha256(struct.pack(f"<{len(signs)}f", *signs)).hexdigest()


def test_golden_sign_diagonal_for_the_production_width():
    from omlx.cache.boundary_snapshot_store import _gdn_rht_sign_values

    signs = _gdn_rht_sign_values(128, 0)
    assert len(signs) == 128
    assert set(signs) == {1.0, -1.0}
    assert [int(s) for s in signs[:16]] == [
        -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, 1, -1
    ]
    assert _sign_digest(128, 0) == _GOLDEN_SIGN_DIGEST_DIM128_SEED0
    # A different width must not reuse the same diagonal prefix.
    assert _sign_digest(64, 0) != _GOLDEN_SIGN_DIGEST_DIM128_SEED0


def test_golden_forward_and_inverse_on_a_known_vector():
    """dim=4 lands on exact binary values, so no tolerance is needed.

    signs = [-1, 1, 1, -1]; H4 @ [-1, 2, 3, -4] / sqrt(4) = [0, 2, 1, -5].
    """
    from omlx.cache.boundary_snapshot_store import _gdn_rht_sign_values

    assert [int(s) for s in _gdn_rht_sign_values(4, 0)] == [-1, 1, 1, -1]
    source = mx.array([[1.0, 2.0, 3.0, 4.0]], dtype=mx.float32)
    forward = BoundarySnapshotSSDStore._gdn_rht_forward(source, 0)
    assert [float(x) for x in forward[0]] == [0.0, 2.0, 1.0, -5.0]
    inverse = BoundarySnapshotSSDStore._gdn_rht_inverse(forward, 0)
    assert [float(x) for x in inverse[0]] == [1.0, 2.0, 3.0, 4.0]


def test_rht_does_not_advance_the_generation_rng():
    """Sampling must be unaffected by whether a checkpoint was encoded."""
    mx.random.seed(42)
    baseline = mx.random.uniform(shape=(4,))
    mx.eval(baseline)

    mx.random.seed(42)
    BoundarySnapshotSSDStore._gdn_rht_forward(
        mx.ones((1, 128), dtype=mx.float32), 0
    )
    BoundarySnapshotSSDStore._gdn_rht_inverse(
        mx.ones((1, 128), dtype=mx.float32), 0
    )
    after = mx.random.uniform(shape=(4,))
    mx.eval(after)
    assert bool(mx.all(baseline == after))


def test_encoded_sidecar_bytes_are_identical_across_processes(tmp_path):
    """A separate interpreter must produce byte-identical payload+metadata."""
    script = tmp_path / "encode_once.py"
    script.write_text(
        "import hashlib, json, sys, tempfile\n"
        "from pathlib import Path\n"
        "import mlx.core as mx\n"
        "from omlx.cache.boundary_snapshot_store import BoundarySnapshotSSDStore\n"
        "base = mx.arange(1, 1 + 1 * 2 * 4 * 16, dtype=mx.float32).reshape(1, 2, 4, 16)\n"
        "recurrent = (base - 64.5) * 0.125\n"
        "extracted = [{\n"
        "    'state': (\n"
        "        mx.arange(8, dtype=mx.float32).reshape(1, 1, 8).astype(mx.bfloat16),\n"
        "        recurrent,\n"
        "    ),\n"
        "    'meta_state': (),\n"
        "    'class_name': 'ArraysCache',\n"
        "    'cache_type': 'ArraysCache',\n"
        "}]\n"
        "with tempfile.TemporaryDirectory() as d:\n"
        "    store = BoundarySnapshotSSDStore(Path(d), gdn_sidecar_state_dtype='rht_int8')\n"
        "    try:\n"
        "        tensors, metadata = store._serialize_extracted(extracted, 'x', 2048)\n"
        "    finally:\n"
        "        store.shutdown()\n"
        "h = hashlib.sha256()\n"
        "for name in sorted(tensors):\n"
        "    raw, dtype, shape = tensors[name]\n"
        "    h.update(name.encode()); h.update(dtype.encode())\n"
        "    h.update(repr(shape).encode()); h.update(raw)\n"
        "for key in sorted(metadata):\n"
        "    h.update(key.encode()); h.update(str(metadata[key]).encode())\n"
        "sys.stdout.write(h.hexdigest())\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    digests = set()
    for _ in range(2):
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            env=env,
            timeout=180,
        )
        assert result.returncode == 0, result.stderr
        digests.add(result.stdout.strip())
    assert len(digests) == 1, f"cross-process encoding diverged: {digests}"
    assert len(next(iter(digests))) == 64


def test_production_shape_error_and_payload_layout(tmp_path):
    """Characterize the real 27B recurrent shape end to end."""
    shape = (1, 48, 128, 128)
    mx.random.seed(0)
    recurrent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(recurrent)

    # Transform-only roundtrip: fp32 rounding, no quantization involved.
    forward = BoundarySnapshotSSDStore._gdn_rht_forward(recurrent, 0)
    inverse = BoundarySnapshotSSDStore._gdn_rht_inverse(forward, 0)
    transform_only_max_abs = float(mx.max(mx.abs(inverse - recurrent)))
    assert transform_only_max_abs < 1e-3

    errors = {}
    for mode in ("int8", "rht_int8"):
        store = BoundarySnapshotSSDStore(
            tmp_path / mode, gdn_sidecar_state_dtype=mode
        )
        try:
            tensors, metadata = store._serialize_extracted(
                _recurrent_extracted(recurrent), "prod", 2048
            )
            assert tensors["layer_0_state_1"][1] == "I8"
            assert tensors["layer_0_state_1"][2] == list(shape)
            assert tensors["layer_0_state_1__scale"][1] == "F32"
            assert tensors["layer_0_state_1__scale"][2] == [1, 48, 128, 1]
            # int8 payload + fp32 row scales against a 4-byte fp32 source.
            payload = len(tensors["layer_0_state_1"][0])
            scale_bytes = len(tensors["layer_0_state_1__scale"][0])
            source_bytes = 4 * 48 * 128 * 128
            assert payload == source_bytes // 4
            assert (source_bytes / (payload + scale_bytes)) > 3.5

            restored = store._deserialize(tensors, metadata)[0]["state"][1]
            errors[mode] = float(
                mx.mean(mx.abs(restored - recurrent))
                / mx.mean(mx.abs(recurrent))
            )
        finally:
            store.shutdown()

    # The rotation is what buys the accuracy; keep that ordering pinned.
    assert errors["rht_int8"] < errors["int8"]
    assert errors["rht_int8"] < 0.01


def test_rht_int16_production_shape_payload_layout(tmp_path):
    """The production 27B state shape uses two bytes plus FP32 row scales."""
    shape = (1, 48, 128, 128)
    mx.random.seed(1)
    recurrent = mx.random.normal(shape, dtype=mx.float32)
    mx.eval(recurrent)
    store = BoundarySnapshotSSDStore(
        tmp_path / "rht_int16", gdn_sidecar_state_dtype="rht_int16"
    )
    try:
        tensors, metadata = store._serialize_extracted(
            _recurrent_extracted(recurrent), "prod16", 2048
        )
        payload = tensors["layer_0_state_1"]
        scale = tensors["layer_0_state_1__scale"]
        assert payload[1] == "I16"
        assert payload[2] == list(shape)
        assert scale[1] == "F32"
        assert scale[2] == [1, 48, 128, 1]
        source_bytes = 4 * 48 * 128 * 128
        assert len(payload[0]) == source_bytes // 2
        # FP32 -> int16 payload plus one FP32 scale per row; the small scale
        # tensor makes the full recurrent payload ratio just under 2x.
        assert source_bytes / (len(payload[0]) + len(scale[0])) > 1.96
        restored = store._deserialize(tensors, metadata)[0]["state"][1]
        relative = float(
            mx.sqrt(mx.sum((restored - recurrent) ** 2))
            / mx.sqrt(mx.sum(recurrent**2))
        )
        assert relative < 0.0001
    finally:
        store.shutdown()


# --- P0-6: which layer owns the split-disabled + reduced-dtype invariant ---


@pytest.mark.parametrize("mode", ["bf16", "int8", "rht_int8", "rht_int16"])
def test_manager_constructor_does_not_enforce_the_split_invariant(tmp_path, mode):
    """Documented layering, pinned so it cannot drift silently.

    The invariant lives where an operator states intent: ``Settings.validate``
    and the admin settings route both reject a reduced dtype without
    ``gdn_ssd_split_enabled``. The scheduler additionally coerces reduced ->
    fp32 when split is off. These constructors are internal and stay
    permissive; with split disabled the recurrent state is stored embedded in
    the block payload and the codec simply never runs.
    """
    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / mode,
        max_size_bytes=1024**2,
        gdn_ssd_split_enabled=False,
        gdn_sidecar_state_dtype=mode,
    )
    try:
        assert manager._payload_layout == "embedded"
    finally:
        manager.close()


@pytest.mark.parametrize(
    "mode", ["fp32", "bf16", "int8", "rht_int8", "rht_int16"]
)
def test_constructors_normalize_dtype_case(tmp_path, mode):
    store = BoundarySnapshotSSDStore(
        tmp_path / f"store-{mode}", gdn_sidecar_state_dtype=mode.upper()
    )
    try:
        assert store.gdn_sidecar_state_dtype == mode
    finally:
        store.shutdown()

    manager = PagedSSDCacheManager(
        cache_dir=tmp_path / f"mgr-{mode}",
        max_size_bytes=1024**2,
        gdn_ssd_split_enabled=True,
        gdn_sidecar_state_dtype=mode.upper(),
    )
    try:
        assert manager._gdn_sidecar_state_dtype == mode
    finally:
        manager.close()


@pytest.mark.parametrize("bad", ["fp8", "int4", "", "rht", None, 8])
def test_constructors_reject_unknown_dtype(tmp_path, bad):
    with pytest.raises(ValueError, match="gdn_sidecar_state_dtype"):
        BoundarySnapshotSSDStore(tmp_path / "s", gdn_sidecar_state_dtype=bad)
    with pytest.raises(ValueError, match="gdn_sidecar_state_dtype"):
        PagedSSDCacheManager(
            cache_dir=tmp_path / "m",
            max_size_bytes=1024**2,
            gdn_ssd_split_enabled=True,
            gdn_sidecar_state_dtype=bad,
        )
