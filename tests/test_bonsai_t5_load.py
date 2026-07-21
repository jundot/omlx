# SPDX-License-Identifier: Apache-2.0
"""Tests for the Bonsai t5 load / quantized_matmul patch.

Covers:
  - _is_t5_weight_replacement shape and dtype gating
  - _patched_load_weights strict behaviour with t5 uint8 replacements
  - _t5_quantized_matmul routing: t5 uint8 fallback, bits=1 fallback,
    uint32 passthrough, native kernel dispatch (with fakes)
  - apply/remove lifecycle (idempotency, restore of originals)
  - free_t5_biases placeholder swap
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import omlx.patches.bonsai_t5_load as bonsai_t5_load
from omlx.custom_kernels.bonsai.fast import _dequant_1bit
from omlx.patches import bonsai_qmv
from omlx.patches.bonsai_t5_load import (
    _is_t5_weight_replacement,
    _patched_load_weights,
    _t5_quantized_matmul,
    apply_bonsai_t5_load_patch,
    free_t5_biases,
    remove_bonsai_t5_load_patch,
)
from tools.repack_ternary_t5 import pack_t5


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _t5_patch_guard():
    """Never leak the global patch into other tests, even when a test fails."""
    remove_bonsai_t5_load_patch()
    yield
    remove_bonsai_t5_load_patch()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _TinyModel(nn.Module):
    """One 2-bit QuantizedLinear: weight (4, 4) uint32, scales/biases (4, 1)."""

    def __init__(self):
        super().__init__()
        self.proj = nn.QuantizedLinear(64, 4, bias=False, group_size=64, bits=2)


class _TwoLayerModel(nn.Module):
    """One t5-convertible 2-bit layer and one 4-bit layer."""

    def __init__(self):
        super().__init__()
        self.t5 = nn.QuantizedLinear(64, 4, bias=False, group_size=64, bits=2)
        self.q4 = nn.QuantizedLinear(64, 4, bias=False, group_size=64, bits=4)


def _t5_weights_for(model: _TinyModel, seed: int = 0) -> list[tuple[str, mx.array]]:
    """Full strict weight list with the uint32 weight replaced by t5 uint8."""
    curr = dict(tree_flatten(model.parameters()))
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 3, size=(4, 64), dtype=np.uint8)
    return [
        ("proj.weight", mx.array(pack_t5(q, 64))),
        ("proj.scales", curr["proj.scales"]),
        ("proj.biases", curr["proj.biases"]),
    ]


# ---------------------------------------------------------------------------
# _is_t5_weight_replacement
# ---------------------------------------------------------------------------


class TestIsT5WeightReplacement:
    def test_accepts_gs64_single_group(self):
        # K=64: uint32 placeholder (4, 4), t5 uint8 (4, 13).
        curr = mx.zeros((4, 4), dtype=mx.uint32)
        new = mx.zeros((4, 13), dtype=mx.uint8)
        assert _is_t5_weight_replacement("proj.weight", curr, new) is True

    def test_accepts_gs64_multi_group(self):
        # K=192, 3 groups: uint32 (4, 12), t5 uint8 (4, 39).
        curr = mx.zeros((4, 12), dtype=mx.uint32)
        new = mx.zeros((4, 39), dtype=mx.uint8)
        assert _is_t5_weight_replacement("proj.weight", curr, new) is True

    def test_accepts_gs128_layout(self):
        # K=128: uint32 placeholder (4, 8), t5 uint8 (4, 26).
        curr = mx.zeros((4, 8), dtype=mx.uint32)
        new = mx.zeros((4, 26), dtype=mx.uint8)
        assert _is_t5_weight_replacement("proj.weight", curr, new) is True

    @pytest.mark.parametrize(
        ("key", "curr", "new"),
        [
            # Non-weight key with otherwise valid shapes.
            pytest.param(
                "proj.scales",
                mx.zeros((4, 4), dtype=mx.uint32),
                mx.zeros((4, 13), dtype=mx.uint8),
                id="wrong-key",
            ),
            # Current parameter is not the uint32 placeholder.
            pytest.param(
                "proj.weight",
                mx.zeros((4, 4), dtype=mx.float16),
                mx.zeros((4, 13), dtype=mx.uint8),
                id="curr-not-uint32",
            ),
            # Incoming tensor is not uint8.
            pytest.param(
                "proj.weight",
                mx.zeros((4, 4), dtype=mx.uint32),
                mx.zeros((4, 13), dtype=mx.uint32),
                id="new-not-uint8",
            ),
            # Row counts differ.
            pytest.param(
                "proj.weight",
                mx.zeros((8, 4), dtype=mx.uint32),
                mx.zeros((4, 13), dtype=mx.uint8),
                id="row-mismatch",
            ),
            # 13 columns imply K=64 so the placeholder must have 4 columns.
            pytest.param(
                "proj.weight",
                mx.zeros((4, 5), dtype=mx.uint32),
                mx.zeros((4, 13), dtype=mx.uint8),
                id="k-mismatch",
            ),
            # 14 columns divide by neither 13 nor 26.
            pytest.param(
                "proj.weight",
                mx.zeros((4, 4), dtype=mx.uint32),
                mx.zeros((4, 14), dtype=mx.uint8),
                id="non-divisible-cols",
            ),
            # Current parameter is not 2-D.
            pytest.param(
                "proj.weight",
                mx.zeros((4,), dtype=mx.uint32),
                mx.zeros((4, 13), dtype=mx.uint8),
                id="curr-1d",
            ),
            # Incoming tensor is not 2-D.
            pytest.param(
                "proj.weight",
                mx.zeros((4, 4), dtype=mx.uint32),
                mx.zeros((4, 13, 1), dtype=mx.uint8),
                id="new-3d",
            ),
        ],
    )
    def test_rejects(self, key, curr, new):
        assert _is_t5_weight_replacement(key, curr, new) is False


# ---------------------------------------------------------------------------
# _patched_load_weights
# ---------------------------------------------------------------------------


class TestPatchedLoadWeights:
    def test_strict_accepts_t5_replacement(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)
        _patched_load_weights(model, weights, strict=True)
        loaded = dict(tree_flatten(model.parameters()))["proj.weight"]
        assert loaded.dtype == mx.uint8
        assert loaded.shape == (4, 13)

    def test_strict_rejects_wrong_shape_uint32(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)
        weights[0] = ("proj.weight", mx.zeros((4, 5), dtype=mx.uint32))
        with pytest.raises(ValueError, match="Expected shape"):
            _patched_load_weights(model, weights, strict=True)

    def test_strict_rejects_wrong_shape_uint8(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)
        weights[0] = ("proj.weight", mx.zeros((4, 14), dtype=mx.uint8))
        with pytest.raises(ValueError, match="Expected shape"):
            _patched_load_weights(model, weights, strict=True)

    def test_strict_rejects_extra_key(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)
        weights.append(("proj.ghost", mx.zeros((1,), dtype=mx.float16)))
        with pytest.raises(ValueError, match="not in model"):
            _patched_load_weights(model, weights, strict=True)

    def test_strict_rejects_missing_key(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)[:2]
        with pytest.raises(ValueError, match="Missing"):
            _patched_load_weights(model, weights, strict=True)

    def test_strict_rejects_non_array(self):
        model = _TinyModel()
        weights = _t5_weights_for(model)
        weights[0] = ("proj.weight", [[1, 2, 3]])
        with pytest.raises(ValueError, match="Expected mx.array"):
            _patched_load_weights(model, weights, strict=True)

    def test_loads_from_safetensors_path(self, tmp_path):
        model = _TinyModel()
        weights = dict(_t5_weights_for(model))
        path = str(tmp_path / "model.safetensors")
        mx.save_safetensors(path, weights)
        _patched_load_weights(model, path, strict=True)
        loaded = dict(tree_flatten(model.parameters()))["proj.weight"]
        assert loaded.dtype == mx.uint8
        assert loaded.shape == (4, 13)


# ---------------------------------------------------------------------------
# _t5_quantized_matmul: fallback paths (no native extension)
# ---------------------------------------------------------------------------


class TestT5QuantizedMatmulFallback:
    def test_t5_uint8_dequant_fallback_gs64_exact(self, monkeypatch):
        """Identity rows pick out dequantized columns: scale * (q - 1)."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        rng = np.random.default_rng(7)
        N, K = 4, 64
        q = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
        w = mx.array(pack_t5(q, 64))
        scales = mx.ones((N, 1), dtype=mx.float16)
        biases = mx.zeros((N, 1), dtype=mx.float16)
        x = mx.array(np.eye(K, dtype=np.float16)[:8])
        out = _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=2, group_size=64
        )
        expected = (q.astype(np.float32) - 1.0).T[:8]
        np.testing.assert_array_equal(np.array(out.astype(mx.float32)), expected)

    def test_t5_uint8_dequant_fallback_gs64_random(self, monkeypatch):
        """Random x and scales match a numpy reference dequant matmul."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        rng = np.random.default_rng(11)
        N, K, gs = 8, 128, 64
        n_groups = K // gs
        q = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
        scales_np = rng.uniform(0.5, 2.0, size=(N, n_groups)).astype(np.float16)
        x_np = rng.standard_normal((3, K)).astype(np.float16)
        out = _t5_quantized_matmul(
            mx.array(x_np),
            mx.array(pack_t5(q, gs)),
            mx.array(scales_np),
            mx.zeros((N, n_groups), dtype=mx.float16),
            transpose=True,
            bits=2,
            group_size=gs,
        )
        s_exp = np.repeat(scales_np.astype(np.float32), gs, axis=1)
        w_fp = (q.astype(np.float32) - 1.0) * s_exp
        expected = x_np.astype(np.float32) @ w_fp.T
        np.testing.assert_allclose(
            np.array(out.astype(mx.float32)), expected, rtol=1e-2, atol=1e-2
        )

    def test_t5_uint8_dequant_fallback_gs128_exact(self, monkeypatch):
        """bpg=26 layout: group size is inferred as 128 from the byte count."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        rng = np.random.default_rng(13)
        N, K = 4, 128
        q = rng.integers(0, 3, size=(N, K), dtype=np.uint8)
        w = mx.array(pack_t5(q, 128))
        assert w.shape == (N, 26)
        scales = mx.ones((N, 1), dtype=mx.float16)
        biases = mx.zeros((N, 1), dtype=mx.float16)
        x = mx.array(np.eye(K, dtype=np.float16)[:8])
        out = _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=2, group_size=128
        )
        expected = (q.astype(np.float32) - 1.0).T[:8]
        np.testing.assert_array_equal(np.array(out.astype(mx.float32)), expected)

    def test_bits1_fallback_hand_computed(self, monkeypatch):
        """N=2, K=64, gs=32 with explicit bit words and hand-computed output."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        w = mx.array(
            np.array(
                [[0xFFFFFFFF, 0x00000000], [0x00000001, 0x80000000]],
                dtype=np.uint32,
            )
        )
        scales = mx.array([[2.0, 3.0], [1.0, 0.5]], dtype=mx.float16)
        biases = mx.array([[-1.0, 0.5], [0.0, -0.25]], dtype=mx.float16)
        x = mx.ones((1, 64), dtype=mx.float16)
        out = _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=1, group_size=32
        )
        assert out.shape == (1, 2)
        # Row 0: 32 cols at 2*1-1=1.0 plus 32 cols at 3*0+0.5=0.5 -> 48.0.
        assert float(out[0, 0]) == 48.0
        # Row 1: 1.0 (col 0) + 31*0.0 + 31*(-0.25) + 0.25 (col 63) -> -6.5.
        assert float(out[0, 1]) == -6.5

    def test_bits1_fallback_matches_dequant_reference(self, monkeypatch):
        """Random bits: output is exactly x @ _dequant_1bit(w).T."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        rng = np.random.default_rng(17)
        N, K, gs = 8, 128, 64
        w_np = rng.integers(0, 2**32, size=(N, K // 32), dtype=np.uint64)
        w = mx.array(w_np.astype(np.uint32))
        scales = mx.array(rng.uniform(0.5, 1.5, (N, K // gs)).astype(np.float16))
        biases = mx.array(rng.uniform(-0.5, 0.5, (N, K // gs)).astype(np.float16))
        x = mx.array(rng.standard_normal((2, K)).astype(np.float16))
        out = _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=1, group_size=gs
        )
        expected = x @ _dequant_1bit(w, scales, biases, mx.float16, gs).T
        assert mx.array_equal(out, expected).item()

    def test_uint32_bits4_passthrough_matches_stock(self, monkeypatch):
        """4-bit uint32 weights go straight to the original C function."""
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: False)
        wf = mx.random.normal((8, 64)).astype(mx.float16)
        w, scales, biases = mx.quantize(wf, group_size=64, bits=4)
        x = mx.random.normal((2, 64)).astype(mx.float16)
        expected = mx.quantized_matmul(
            x, w, scales=scales, biases=biases, transpose=True, group_size=64, bits=4
        )
        monkeypatch.setattr(
            bonsai_t5_load, "_original_quantized_matmul", mx.quantized_matmul
        )
        out = _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=4, group_size=64
        )
        assert mx.array_equal(out, expected).item()

    def test_uint32_passthrough_forwards_kwargs(self, monkeypatch):
        called = {}

        def fake_qmm(x, w, scales, biases, *, transpose, bits, group_size, **kw):
            called["args"] = (bits, group_size, transpose, kw.get("mode"))
            return mx.zeros((1, 8), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "_original_quantized_matmul", fake_qmm)
        x = mx.zeros((1, 64), dtype=mx.float16)
        w = mx.zeros((8, 8), dtype=mx.uint32)
        scales = mx.ones((8, 1), dtype=mx.float16)
        biases = mx.zeros((8, 1), dtype=mx.float16)
        _t5_quantized_matmul(
            x, w, scales, biases, transpose=True, bits=4, group_size=64, mode="affine"
        )
        assert called["args"] == (4, 64, True, "affine")


# ---------------------------------------------------------------------------
# _t5_quantized_matmul: native kernel dispatch (fakes)
# ---------------------------------------------------------------------------


class TestT5QuantizedMatmulNativeRouting:
    def _t5_inputs(self, M: int):
        w = mx.zeros((4, 13), dtype=mx.uint8)
        scales = mx.ones((4, 1), dtype=mx.float16)
        biases = mx.zeros((4, 1), dtype=mx.float16)
        x = mx.zeros((M, 64), dtype=mx.float16)
        return x, w, scales, biases

    def test_t5_m1_routes_to_qmv(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_qmv(x, w, scales):
            called["fired"] = True
            return mx.zeros((1, 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "bonsai_t5_qmv", fake_qmv)
        x, w, scales, biases = self._t5_inputs(M=1)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called.get("fired") is True

    def test_t5_m3_routes_to_qmv_wide(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_wide(x, w, scales):
            called["fired"] = True
            return mx.zeros((3, 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "bonsai_t5_qmv_wide", fake_wide)
        x, w, scales, biases = self._t5_inputs(M=3)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called.get("fired") is True

    def test_t5_above_threshold_routes_to_qmm(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_qmm(x_flat, w, scales):
            called["M"] = x_flat.shape[0]
            return mx.zeros((x_flat.shape[0], 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "bonsai_t5_qmm", fake_qmm)
        x, w, scales, biases = self._t5_inputs(M=32)
        out = _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called["M"] == 32
        assert out.shape == (32, 4)

    def test_bits1_m1_routes_to_q1_qmv(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_q1(x, w, scales, biases):
            called["fired"] = True
            return mx.zeros((1, 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "bonsai_q1_affine_qmv", fake_q1)
        x = mx.zeros((1, 64), dtype=mx.float16)
        w = mx.zeros((4, 2), dtype=mx.uint32)
        scales = mx.ones((4, 2), dtype=mx.float16)
        biases = mx.zeros((4, 2), dtype=mx.float16)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=1)
        assert called.get("fired") is True

    def test_bits1_m3_routes_to_qmv_wide(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_wide(x, w, scales, biases, bits):
            called["bits"] = bits
            return mx.zeros((3, 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "bonsai_qmv_wide", fake_wide)
        x = mx.zeros((3, 64), dtype=mx.float16)
        w = mx.zeros((4, 2), dtype=mx.uint32)
        scales = mx.ones((4, 2), dtype=mx.float16)
        biases = mx.zeros((4, 2), dtype=mx.float16)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=1)
        assert called.get("bits") == 1


# ---------------------------------------------------------------------------
# apply / remove lifecycle
# ---------------------------------------------------------------------------


class TestPatchLifecycle:
    def test_apply_installs_and_is_idempotent(self):
        orig_lw = nn.Module.load_weights
        orig_qmm = mx.quantized_matmul
        try:
            assert apply_bonsai_t5_load_patch() is True
            assert nn.Module.load_weights is _patched_load_weights
            assert mx.quantized_matmul is _t5_quantized_matmul
            # Second apply is a no-op and reports it.
            assert apply_bonsai_t5_load_patch() is False
            assert nn.Module.load_weights is _patched_load_weights
        finally:
            remove_bonsai_t5_load_patch()
        assert nn.Module.load_weights is orig_lw
        assert mx.quantized_matmul is orig_qmm

    def test_remove_without_apply_is_noop(self):
        orig_lw = nn.Module.load_weights
        orig_qmm = mx.quantized_matmul
        remove_bonsai_t5_load_patch()
        assert nn.Module.load_weights is orig_lw
        assert mx.quantized_matmul is orig_qmm

    def test_installed_patch_serves_bound_load_weights(self):
        """model.load_weights goes through the patch after apply."""
        try:
            apply_bonsai_t5_load_patch()
            model = _TinyModel()
            model.load_weights(_t5_weights_for(model), strict=True)
            loaded = dict(tree_flatten(model.parameters()))["proj.weight"]
            assert loaded.dtype == mx.uint8
        finally:
            remove_bonsai_t5_load_patch()

    def test_prefill_threshold_matches_bonsai_qmv(self):
        # The two module constants are documented as must-match.
        assert (
            bonsai_t5_load._T5_PREFILL_THRESHOLD == bonsai_qmv._T5_PREFILL_THRESHOLD
        )


# ---------------------------------------------------------------------------
# free_t5_biases
# ---------------------------------------------------------------------------


class TestFreeT5Biases:
    def test_frees_only_t5_layer_biases(self):
        model = _TwoLayerModel()
        rng = np.random.default_rng(19)
        q = rng.integers(0, 3, size=(4, 64), dtype=np.uint8)
        model.t5.weight = mx.array(pack_t5(q, 64))
        t5_biases = model.t5.biases
        q4_biases_before = np.array(model.q4.biases)
        expected_freed = int(t5_biases.size) * t5_biases.itemsize

        freed = free_t5_biases(model)

        assert freed == expected_freed
        assert freed > 0
        # t5 layer biases replaced with the tiny placeholder.
        assert model.t5.biases.shape == (1,)
        assert float(model.t5.biases[0]) == 0.0
        # 4-bit layer untouched.
        assert model.q4.biases.shape == (4, 1)
        np.testing.assert_array_equal(np.array(model.q4.biases), q4_biases_before)

    def test_no_t5_layers_frees_nothing(self):
        model = _TwoLayerModel()  # Both weights still uint32.
        biases_before = np.array(model.t5.biases)
        freed = free_t5_biases(model)
        assert freed == 0
        assert model.t5.biases.shape == (4, 1)
        np.testing.assert_array_equal(np.array(model.t5.biases), biases_before)
