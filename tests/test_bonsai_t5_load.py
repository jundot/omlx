# SPDX-License-Identifier: Apache-2.0
"""Tests for Bonsai checkpoint support.

Covers:
  - t5 repack script round trips
  - _is_t5_weight_replacement shape and dtype gating
  - _patched_load_weights strict behaviour with t5 uint8 replacements
  - _t5_quantized_matmul routing: t5 uint8 fallback,
    uint32 passthrough, native kernel dispatch (with fakes)
  - apply/remove lifecycle (idempotency, restore of originals)
  - free_t5_biases placeholder swap
  - pre-load wiring by weight format
  - mlx-vlm 1-bit module rebinding to the stock MLX modules
"""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import mlx_lm.utils as mlx_lm_utils
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_vlm.quantization.one_bit import OneBitEmbedding, OneBitLinear

import omlx.patches.bonsai_t5_load as bonsai_t5_load
from omlx.models.vlm import restore_bonsai_quantized_modules
from omlx.patches.bonsai_t5_load import (
    _is_t5_weight_replacement,
    _patched_load_weights,
    _t5_quantized_matmul,
    apply_bonsai_t5_load_patch,
    free_t5_biases,
    remove_bonsai_t5_load_patch,
)
from omlx.utils.model_loading import maybe_apply_pre_load_patches
from scripts.repack_ternary_t5 import pack_t5, unpack_mlx_2bit, unpack_t5


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
# t5 repack script
# ---------------------------------------------------------------------------


class TestT5Repack:
    """Tests for scripts/repack_ternary_t5.py."""

    def test_pack_unpack_roundtrip_gs128(self):
        rng = np.random.default_rng(0)
        q = rng.integers(0, 3, size=(8, 128), dtype=np.uint8)
        t5w = pack_t5(q, group_size=128)
        assert t5w.shape == (8, 26), f"expected (8,26) got {t5w.shape}"
        q_rt = unpack_t5(t5w, group_size=128, K=128)
        np.testing.assert_array_equal(q, q_rt)

    def test_pack_unpack_roundtrip_gs64(self):
        rng = np.random.default_rng(1)
        q = rng.integers(0, 3, size=(8, 64), dtype=np.uint8)
        t5w = pack_t5(q, group_size=64)
        assert t5w.shape == (8, 13), f"expected (8,13) got {t5w.shape}"
        q_rt = unpack_t5(t5w, group_size=64, K=64)
        np.testing.assert_array_equal(q, q_rt)

    def test_pack_unpack_larger_K(self):
        rng = np.random.default_rng(2)
        K, gs = 7168, 128
        q = rng.integers(0, 3, size=(4, K), dtype=np.uint8)
        t5w = pack_t5(q, group_size=gs)
        n_groups = K // gs
        assert t5w.shape == (4, n_groups * 26)
        q_rt = unpack_t5(t5w, group_size=gs, K=K)
        np.testing.assert_array_equal(q, q_rt)

    def test_padding_trit_is_neutral(self):
        """Padding trits (q=1) must contribute zero to the dot product."""
        # Single group of 128, last 2 positions zero-padded with q=1
        q = np.ones((1, 128), dtype=np.uint8)  # all t=0 (q=1 → dq=0 for scale*(q-1))
        t5w = pack_t5(q, group_size=128)
        # Decode last byte and check it encodes 3 active trits + 2 padding (all q=1)
        # byte v = 1 + 1*3 + 1*9 + 1*27 + 1*81 = 121
        assert t5w[0, 25] == 121  # 1+3+9+27+81

    def test_dequant_matches_2bit_reference(self):
        """t5 and 2-bit dequantize to the same float values."""
        rng = np.random.default_rng(3)
        N, K, gs = 16, 256, 128
        # Generate ternary quants ∈ {0,1,2}
        q = rng.integers(0, 3, size=(N, K), dtype=np.uint8)

        # 2-bit MLX pack: 16 values per uint32
        w2bit = np.zeros((N, K // 16), dtype=np.uint32)
        for i in range(16):
            w2bit |= (q[:, i::16].astype(np.uint32) << (i * 2))

        # Build matching scales and biases
        n_groups = K // gs
        scales = rng.uniform(0.5, 1.5, size=(N, n_groups)).astype(np.float32)
        biases = -scales  # ternary symmetric

        # Dequantize from 2-bit
        q2 = unpack_mlx_2bit(w2bit, K)
        dq2 = sum(
            (scales[:, g:g+1] * q2[:, g*gs:(g+1)*gs] + biases[:, g:g+1])
            for g in range(n_groups)
        )

        # Dequantize from t5
        t5w = pack_t5(q, group_size=gs)
        qt5 = unpack_t5(t5w, group_size=gs, K=K)
        dqt5 = sum(
            (scales[:, g:g+1] * qt5[:, g*gs:(g+1)*gs] + biases[:, g:g+1])
            for g in range(n_groups)
        )

        np.testing.assert_allclose(dq2, dqt5, atol=1e-6)


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

        monkeypatch.setattr(bonsai_t5_load, "t5_qmv", fake_qmv)
        x, w, scales, biases = self._t5_inputs(M=1)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called.get("fired") is True

    def test_t5_m3_routes_to_qmv_wide(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_wide(x, w, scales):
            called["fired"] = True
            return mx.zeros((3, 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "t5_qmv_wide", fake_wide)
        x, w, scales, biases = self._t5_inputs(M=3)
        _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called.get("fired") is True

    def test_t5_above_threshold_routes_to_qmm(self, monkeypatch):
        monkeypatch.setattr(bonsai_t5_load, "has_native", lambda: True)
        called = {}

        def fake_qmm(x_flat, w, scales):
            called["M"] = x_flat.shape[0]
            return mx.zeros((x_flat.shape[0], 4), dtype=mx.float16)

        monkeypatch.setattr(bonsai_t5_load, "t5_qmm", fake_qmm)
        x, w, scales, biases = self._t5_inputs(M=32)
        out = _t5_quantized_matmul(x, w, scales, biases, transpose=True, bits=2)
        assert called["M"] == 32
        assert out.shape == (32, 4)

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

    @pytest.mark.skipif(
        not hasattr(mlx_lm_utils, "infer_quant_config"),
        reason="mlx-lm pin predates infer_quant_config",
    )
    @pytest.mark.parametrize("group_size", [64, 128])
    def test_installed_patch_reports_t5_layers_as_2bit(self, group_size):
        """mlx-lm's width-based inference would read t5 rows as 6 bits."""
        rng = np.random.default_rng(0)
        q = rng.integers(0, 3, size=(4, 2 * group_size), dtype=np.uint8)
        t5 = nn.QuantizedLinear(2 * group_size, 4, group_size=group_size, bits=2)
        q4 = nn.QuantizedLinear(2 * group_size, 4, group_size=group_size, bits=4)
        weights = {
            "t5.weight": mx.array(pack_t5(q, group_size)),
            "t5.scales": t5.scales,
            "q4.weight": q4.weight,
            "q4.scales": q4.scales,
        }
        linear = nn.Linear(2 * group_size, 4, bias=False)
        try:
            apply_bonsai_t5_load_patch()
            infer = mlx_lm_utils.infer_quant_config
            assert infer("t5", linear, weights) == {
                "group_size": group_size,
                "bits": 2,
                "mode": "affine",
            }
            assert infer("q4", linear, weights)["bits"] == 4
        finally:
            remove_bonsai_t5_load_patch()
        assert (
            mlx_lm_utils.infer_quant_config is not bonsai_t5_load._t5_infer_quant_config
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


@pytest.mark.parametrize("mode", ["affine", "mxfp4"])
@pytest.mark.parametrize("positional", [False, True])
def test_installed_matmul_preserves_mlx_call_contract(mode, positional):
    x = mx.ones((2, 64), dtype=mx.float16)
    packed = mx.quantize(
        mx.ones((8, 64), dtype=mx.float16),
        bits=4,
        group_size=32 if mode == "mxfp4" else 64,
        mode=mode,
    )

    def run():
        if positional:
            w, scales, *biases = packed
            return mx.quantized_matmul(
                x,
                w,
                scales,
                biases[0] if biases else None,
                True,
                None,
                None,
                mode,
            )
        return mx.quantized_matmul(x, *packed, mode=mode)

    expected = run()
    mx.eval(expected)
    apply_bonsai_t5_load_patch()
    actual = run()
    assert mx.array_equal(actual, expected).item()


@pytest.mark.parametrize("packing", ["affine", "t5", "invalid_u8"])
@pytest.mark.parametrize("group_size", [64, 128])
def test_preload_dispatch_uses_weight_format(tmp_path, packing, group_size):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "quantization": {"bits": 2, "group_size": group_size},
            }
        )
    )
    model = _TinyModel()
    model.proj = nn.QuantizedLinear(128, 4, bias=False, group_size=group_size, bits=2)
    weights = dict(tree_flatten(model.parameters()))
    x = mx.ones((1, 128), dtype=mx.float16)
    expected = model.proj(x)
    mx.eval(expected)
    if packing == "t5":
        quants = np.ones((4, 128), dtype=np.uint8)
        weights["proj.weight"] = mx.array(pack_t5(quants, group_size))
    elif packing == "invalid_u8":
        weights["proj.weight"] = mx.zeros((4, 14), dtype=mx.uint8)
    # Weight and scales can live in different shards.
    mx.save_safetensors(
        str(tmp_path / "model-00001.safetensors"),
        {"proj.weight": weights.pop("proj.weight")},
    )
    mx.save_safetensors(str(tmp_path / "model-00002.safetensors"), weights)
    original = mx.quantized_matmul
    maybe_apply_pre_load_patches(str(tmp_path))
    assert (mx.quantized_matmul is not original) == (packing == "t5")
    if packing == "invalid_u8":
        return
    loaded = {}
    for shard in sorted(tmp_path.glob("*.safetensors")):
        loaded.update(mx.load(str(shard)))
    model.load_weights(list(loaded.items()))
    if packing == "affine":
        assert mx.array_equal(model.proj(x), expected).item()
    else:
        assert model.proj.weight.dtype == mx.uint8
        mx.eval(model.proj(x))


def test_one_bit_preload_keeps_stock_quantized_matmul(tmp_path):
    (tmp_path / "config.json").write_text(
        '{"model_type": "qwen3", "quantization": {"bits": 1, "group_size": 128}}'
    )
    original = mx.quantized_matmul
    maybe_apply_pre_load_patches(str(tmp_path))
    assert mx.quantized_matmul is original


def test_vlm_one_bit_modules_match_stock_mlx():
    rng = np.random.default_rng(7)
    model = nn.Module()
    model.linear = OneBitLinear(128, 32, bias=False)
    model.embedding = OneBitEmbedding(32, 128)
    for module in (model.linear, model.embedding):
        bits = rng.integers(0, 2**32, size=(32, 4), dtype=np.uint64)
        module.weight = mx.array(bits.astype(np.uint32))
        module.scales = mx.array(rng.uniform(0.5, 1.5, (32, 2)).astype(np.float16))
        module.biases = -0.5 * module.scales
    xs = [mx.array(rng.standard_normal((m, 128)).astype(np.float16)) for m in (1, 2, 8)]
    ids = mx.array([0, 7])
    expected = [model.linear(x) for x in xs]
    expected_rows = model.embedding(ids)
    weights = model.linear.weight, model.embedding.weight
    linear_call = nn.QuantizedLinear.__call__

    assert restore_bonsai_quantized_modules(model) == 2
    assert type(model.linear) is nn.QuantizedLinear
    assert type(model.embedding) is nn.QuantizedEmbedding
    assert model.linear.weight is weights[0]
    assert model.embedding.weight is weights[1]
    assert not model.linear.trainable_parameters()
    assert not model.embedding.trainable_parameters()
    assert restore_bonsai_quantized_modules(model) == 0
    assert nn.QuantizedLinear.__call__ is linear_call
    for x, ref in zip(xs, expected, strict=True):
        assert mx.allclose(model.linear(x), ref, atol=1e-2, rtol=1e-2).item()
    assert mx.array_equal(model.embedding(ids), expected_rows).item()
