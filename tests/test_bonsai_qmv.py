# SPDX-License-Identifier: Apache-2.0
"""Tests for the Bonsai 1-bit / 2-bit qmv decode kernels and patch.

Covers:
  - _arch_gen() parsing
  - _use_qmv_wide() routing table
  - is_nax_available() fallback + env override
  - _verify_abi() with mock extensions
  - bonsai_q1_affine_qmv / bonsai_qmv_wide fallback (no native ext)
  - spec_decode_verify pure-mlx fallback correctness
  - apply/remove bonsai_qmv_patch lifecycle
  - model_loading wiring: patch fires on bits=1/2, skipped on bits=4
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import omlx.custom_kernels.bonsai.fast as bonsai_fast
from omlx.patches.bonsai_qmv import (
    apply_bonsai_qmv_patch,
    is_patch_active,
    remove_bonsai_qmv_patch,
)
from omlx.utils import model_loading
from omlx.utils.model_loading import maybe_apply_pre_load_patches


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_bonsai_caches(monkeypatch):
    """Clear all module-level caches before each test."""
    monkeypatch.setattr(bonsai_fast, "_nax_available_cache", None)
    monkeypatch.setattr(bonsai_fast, "_arch_gen_cache", None)
    yield


@pytest.fixture(autouse=True)
def _remove_patch_after(monkeypatch):
    """Ensure the QuantizedLinear patch is removed after every test."""
    yield
    remove_bonsai_qmv_patch()


# ---------------------------------------------------------------------------
# _arch_gen parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "expected_gen"),
    [
        ("applegpu_g15d", 15),
        ("applegpu_g17s", 17),
        ("applegpu_g18p", 18),
        ("applegpu_G15D", 15),          # case-insensitive
        ("APPLEGPU_G18P", 18),
        ("", 0),
        ("unknown_gpu", 0),
        ("applegpu_gXYs", 0),           # non-numeric gen
    ],
)
def test_arch_gen_parsing(monkeypatch, arch, expected_gen):
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    gen = bonsai_fast._arch_gen()
    assert gen == expected_gen


def test_arch_gen_device_info_exception(monkeypatch):
    monkeypatch.setattr(mx, "device_info", lambda: (_ for _ in ()).throw(RuntimeError("no GPU")))
    assert bonsai_fast._arch_gen() == 0


# ---------------------------------------------------------------------------
# _use_qmv_wide routing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bits", "M", "gen", "expected"),
    [
        # M < 3: never use wide
        (1, 1, 15, False),
        (1, 2, 18, False),
        (2, 1, 15, False),
        (2, 2, 18, False),
        # M >= 3 on gen >= 15 → qmv_wide for both 1-bit and 2-bit
        (1, 3, 15, True),
        (1, 5, 18, True),
        (2, 3, 15, True),
        (2, 5, 17, True),
        # M >= 3 on old hardware (gen < 15) → fall back
        (1, 3, 14, False),
        (2, 3, 14, False),
        (2, 5, 0, False),
    ],
)
def test_use_qmv_wide_routing(monkeypatch, bits, M, gen, expected):
    monkeypatch.setattr(bonsai_fast, "_arch_gen_cache", gen)
    assert bonsai_fast._use_qmv_wide(bits, M) is expected


# ---------------------------------------------------------------------------
# is_nax_available — fallback path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arch", "expected"),
    [
        ("applegpu_g18p", True),
        ("applegpu_g17s", False),   # gen-17 excluded even though M5-class
        ("applegpu_g15d", False),
        ("", False),
    ],
)
def test_is_nax_available_fallback(monkeypatch, arch, expected):
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": arch})
    assert bonsai_fast.is_nax_available() is expected


def test_is_nax_available_prefers_ext(monkeypatch):
    fake_ext = SimpleNamespace(is_nax_available=lambda: True)
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)
    monkeypatch.setattr(mx, "device_info", lambda: {"architecture": "applegpu_g15d"})
    assert bonsai_fast.is_nax_available() is True


# ---------------------------------------------------------------------------
# _verify_abi
# ---------------------------------------------------------------------------


class _MismatchedExt:
    def abi_probe(self, a):
        raise TypeError("incompatible function arguments")


class _HealthyExt:
    def abi_probe(self, a):
        return 1


class _LegacyExt:
    """Pre-probe build — assumed compatible."""


def test_verify_abi_mismatched_disables_ext():
    ext, err = bonsai_fast._verify_abi(_MismatchedExt(), None)
    assert ext is None
    assert isinstance(err, TypeError)


def test_verify_abi_healthy_passes_through():
    ext = _HealthyExt()
    out, err = bonsai_fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


def test_verify_abi_legacy_build_passes_through():
    ext = _LegacyExt()
    out, err = bonsai_fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


def test_verify_abi_none_ext_passes_through():
    sentinel = ImportError("no native build")
    out, err = bonsai_fast._verify_abi(None, sentinel)
    assert out is None
    assert err is sentinel


# ---------------------------------------------------------------------------
# bonsai_q1_affine_qmv — fallback to mx.quantized_matmul
# ---------------------------------------------------------------------------


def _make_q1_tensors(N=64, K=256, group_size=128):
    """Return (x, w, scales, biases) for a 1-bit affine layer.

    mlx packs (32 // bits) values per uint32, so 1-bit → K//32 words.
    """
    x = mx.zeros((1, K), dtype=mx.float16)
    w = mx.zeros((N, K // 32), dtype=mx.uint32)      # 1-bit: 32 values per uint32
    n_groups = K // group_size
    scales = mx.ones((N, n_groups), dtype=mx.float16)
    biases = mx.zeros((N, n_groups), dtype=mx.float16)
    return x, w, scales, biases


def test_q1_qmv_fallback_calls_quantized_matmul(monkeypatch):
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    called = {}

    def fake_qmm(x, w, *, scales, biases, transpose, group_size, bits, stream=None):
        called["args"] = (bits, group_size, transpose)
        return mx.zeros((1, 64), dtype=mx.float16)

    monkeypatch.setattr(mx, "quantized_matmul", fake_qmm)
    x, w, scales, biases = _make_q1_tensors()
    bonsai_fast.bonsai_q1_affine_qmv(x, w, scales, biases)
    assert called["args"] == (1, 128, True)


def test_q1_qmv_routes_to_ext_when_available(monkeypatch):
    called = {}

    def fake_q1(x, w, scales, biases, stream=None):
        called["fired"] = True
        return mx.zeros((1, 64), dtype=mx.float16)

    fake_ext = SimpleNamespace(
        bonsai_q1_affine_qmv=fake_q1,
        abi_probe=lambda a: 1,
    )
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)
    x, w, scales, biases = _make_q1_tensors()
    bonsai_fast.bonsai_q1_affine_qmv(x, w, scales, biases)
    assert called.get("fired") is True


# ---------------------------------------------------------------------------
# bonsai_qmv_wide dispatch
# ---------------------------------------------------------------------------


def _make_q2_tensors(M=3, N=64, K=256, group_size=128):
    x = mx.zeros((M, K), dtype=mx.bfloat16)
    w = mx.zeros((N, K // 16), dtype=mx.uint32)      # packed 2-bit
    n_groups = K // group_size
    scales = mx.ones((N, n_groups), dtype=mx.bfloat16)
    biases = mx.zeros((N, n_groups), dtype=mx.bfloat16)
    return x, w, scales, biases


def test_qmv_wide_2bit_m3_gen15_routes_to_ext(monkeypatch):
    """M=3, bits=2, gen-15 → should call bonsai_q2_affine_qmv_wide."""
    monkeypatch.setattr(bonsai_fast, "_arch_gen_cache", 15)
    called = {}

    def fake_wide(x, w, scales, biases, stream=None):
        called["fired"] = True
        return mx.zeros((3, 64), dtype=mx.bfloat16)

    fake_ext = SimpleNamespace(
        bonsai_q2_affine_qmv_wide=fake_wide,
        abi_probe=lambda a: 1,
    )
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)
    x, w, scales, biases = _make_q2_tensors(M=3)
    bonsai_fast.bonsai_qmv_wide(x, w, scales, biases, bits=2)
    assert called.get("fired") is True


def test_qmv_wide_2bit_m2_falls_back_to_stock(monkeypatch):
    """M=2, bits=2 → _use_qmv_wide returns False → stock mlx."""
    monkeypatch.setattr(bonsai_fast, "_arch_gen_cache", 17)
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    called = {}

    def fake_qmm(x, w, *, scales, biases, transpose, group_size, bits, stream=None):
        called["bits"] = bits
        return mx.zeros((2, 64), dtype=mx.bfloat16)

    monkeypatch.setattr(mx, "quantized_matmul", fake_qmm)
    x, w, scales, biases = _make_q2_tensors(M=2)
    bonsai_fast.bonsai_qmv_wide(x, w, scales, biases, bits=2)
    assert called.get("bits") == 2


def test_qmv_wide_1bit_always_uses_qmv_fast(monkeypatch):
    """bits=1 always routes through qmv_fast (wide shows no benefit on M4 Max)."""
    monkeypatch.setattr(bonsai_fast, "_arch_gen_cache", 18)
    called = {}

    def fake_q1(x, w, scales, biases, stream=None):
        called["fired"] = True
        return mx.zeros((1, 64), dtype=mx.float16)

    fake_ext = SimpleNamespace(
        bonsai_q1_affine_qmv=fake_q1,
        abi_probe=lambda a: 1,
    )
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)
    x = mx.zeros((1, 256), dtype=mx.float16)
    w = mx.zeros((64, 256 // 32), dtype=mx.uint32)   # 1-bit: 32 values per uint32
    scales = mx.ones((64, 2), dtype=mx.float16)
    biases = mx.zeros((64, 2), dtype=mx.float16)
    bonsai_fast.bonsai_qmv_wide(x, w, scales, biases, bits=1)
    assert called.get("fired") is True


# ---------------------------------------------------------------------------
# spec_decode_verify — pure-mlx fallback correctness
# ---------------------------------------------------------------------------


def _logits_from_greedy(token_ids: list[int], V: int) -> mx.array:
    """Make [1, len, V] logits where argmax = token_ids."""
    T = len(token_ids)
    lgt = mx.zeros((1, T, V), dtype=mx.float32)
    # Use numpy-style trick via list-of-lists
    rows = []
    for tok in token_ids:
        row = [0.0] * V
        row[tok] = 10.0
        rows.append(row)
    return mx.array([[rows]], dtype=mx.float32).reshape(1, T, V)


def test_spec_decode_verify_all_accepted(monkeypatch):
    """Draft tokens perfectly match target greedy: all K accepted."""
    monkeypatch.setattr(bonsai_fast, "_ext", None)

    V = 8
    draft = mx.array([[1, 2, 3]], dtype=mx.int32)          # [1, 3]
    # target greedy: positions 0..3 → tokens [1, 2, 3, 5]
    target_logits = _logits_from_greedy([1, 2, 3, 5], V)  # [1, 4, V]

    n_acc, committed = bonsai_fast.spec_decode_verify(draft, target_logits)
    mx.eval(n_acc, committed)

    assert int(n_acc[0]) == 3                               # all 3 accepted
    assert int(committed[0, 0]) == 1
    assert int(committed[0, 1]) == 2
    assert int(committed[0, 2]) == 3
    assert int(committed[0, 3]) == 5                        # corrected token


def test_spec_decode_verify_first_mismatch(monkeypatch):
    """Target disagrees with first draft token: n_accepted=0."""
    monkeypatch.setattr(bonsai_fast, "_ext", None)

    V = 8
    draft = mx.array([[1, 2]], dtype=mx.int32)
    # target greedy at pos 0 = 7 (≠ draft[0]=1) → mismatch immediately
    target_logits = _logits_from_greedy([7, 2, 4], V)      # [1, 3, V]

    n_acc, committed = bonsai_fast.spec_decode_verify(draft, target_logits)
    mx.eval(n_acc, committed)

    assert int(n_acc[0]) == 0
    assert int(committed[0, 0]) == 7                        # corrected at pos 0
    assert int(committed[0, 1]) == 0                        # zeroed out
    assert int(committed[0, 2]) == 0


def test_spec_decode_verify_mid_mismatch(monkeypatch):
    """Mismatch at second token: n_accepted=1."""
    monkeypatch.setattr(bonsai_fast, "_ext", None)

    V = 8
    draft = mx.array([[3, 5]], dtype=mx.int32)
    # target greedy: [3, 6, 2] → match at 0, mismatch at 1
    target_logits = _logits_from_greedy([3, 6, 2], V)      # [1, 3, V]

    n_acc, committed = bonsai_fast.spec_decode_verify(draft, target_logits)
    mx.eval(n_acc, committed)

    assert int(n_acc[0]) == 1
    assert int(committed[0, 0]) == 3                        # accepted draft
    assert int(committed[0, 1]) == 6                        # corrected token
    assert int(committed[0, 2]) == 0


def test_spec_decode_verify_routes_to_ext_when_available(monkeypatch):
    called = {}

    def fake_verify(draft_tokens, target, stream=None):
        called["fired"] = True
        called["target"] = target
        B = draft_tokens.shape[0]
        K = draft_tokens.shape[1]
        return mx.zeros((B,), mx.int32), mx.zeros((B, K + 1), mx.int32)

    fake_ext = SimpleNamespace(bonsai_spec_decode_verify=fake_verify)
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)

    draft = mx.array([[1, 2]], dtype=mx.int32)
    target_logits = mx.zeros((1, 3, 8), dtype=mx.float32)
    bonsai_fast.spec_decode_verify(draft, target_logits)
    assert called.get("fired") is True
    # The native op takes argmaxed int32 token ids, not raw logits.
    assert called["target"].dtype == mx.int32
    assert called["target"].shape == (1, 3)


@pytest.mark.skipif(
    not bonsai_fast.has_symbol("bonsai_spec_decode_verify"),
    reason="requires compiled bonsai extension",
)
def test_spec_decode_verify_native_matches_fallback():
    """Native kernel and pure-mlx fallback agree on n_accepted and the
    committed prefix (positions past n_accepted are unspecified padding)."""
    rng = np.random.default_rng(11)
    for _trial in range(10):
        B = int(rng.integers(1, 5))
        K = int(rng.integers(1, 8))
        V = 32
        draft = mx.array(rng.integers(0, V, (B, K)), dtype=mx.int32)
        logits = mx.array(rng.standard_normal((B, K + 1, V)).astype(np.float32))

        n_nat, c_nat = bonsai_fast.spec_decode_verify(draft, logits)
        mx.eval(n_nat, c_nat)

        orig_ext = bonsai_fast._ext
        try:
            bonsai_fast._ext = None
            n_fb, c_fb = bonsai_fast.spec_decode_verify(draft, logits)
            mx.eval(n_fb, c_fb)
        finally:
            bonsai_fast._ext = orig_ext

        assert mx.array_equal(n_nat, n_fb).item()
        for b in range(B):
            n = int(n_nat[b].item())
            assert c_nat[b, : n + 1].tolist() == c_fb[b, : n + 1].tolist()


# ---------------------------------------------------------------------------
# Symmetric detection and routing (identity I-B)
# ---------------------------------------------------------------------------


def _make_sym_layer(bits: int, N: int = 64, K: int = 256, group_size: int = 128):
    """QuantizedLinear with biases = -scales * ratio (symmetric Bonsai layout)."""
    import mlx.nn as nn_inner
    ratio = 0.5 if bits == 1 else 1.0
    pack = 32 // bits
    layer = nn_inner.QuantizedLinear.__new__(nn_inner.QuantizedLinear)
    scales = mx.ones((N, K // group_size), dtype=mx.float16)
    biases = mx.full((N, K // group_size), -ratio, dtype=mx.float16)
    weight = mx.zeros((N, K // pack), dtype=mx.uint32)
    object.__setattr__(layer, "weight", weight)
    object.__setattr__(layer, "scales", scales)
    object.__setattr__(layer, "biases", biases)
    object.__setattr__(layer, "bits", bits)
    object.__setattr__(layer, "group_size", group_size)
    object.__setattr__(layer, "mode", "affine")
    return layer


def test_is_symmetric_detects_bonsai_1bit():
    from omlx.patches.bonsai_qmv import _is_symmetric
    layer = _make_sym_layer(bits=1)
    assert _is_symmetric(layer, bits=1) is True


def test_is_symmetric_detects_bonsai_2bit():
    from omlx.patches.bonsai_qmv import _is_symmetric
    layer = _make_sym_layer(bits=2)
    assert _is_symmetric(layer, bits=2) is True


def test_is_symmetric_rejects_non_symmetric():
    from omlx.patches.bonsai_qmv import _is_symmetric
    layer = _make_sym_layer(bits=1)
    # Corrupt one bias entry
    bad_biases = mx.full((64, 2), -0.3, dtype=mx.float16)
    object.__setattr__(layer, "biases", bad_biases)
    assert _is_symmetric(layer, bits=1) is False


def test_is_symmetric_cached():
    from omlx.patches.bonsai_qmv import _is_symmetric
    layer = _make_sym_layer(bits=1)
    first = _is_symmetric(layer, bits=1)
    # Alter biases — cached value should still be returned
    object.__setattr__(layer, "biases", mx.zeros((64, 2), dtype=mx.float16))
    second = _is_symmetric(layer, bits=1)
    assert first == second


def test_sym_q1_fast_py_fallback_routes_to_same_mlx(monkeypatch):
    """Symmetric q1 fast fallback calls mx.quantized_matmul with same args as affine."""
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    calls = []

    def recording_qmm(x, w, *, scales, biases, transpose, group_size, bits, stream=None):
        calls.append({"bits": bits, "group_size": group_size})
        return mx.zeros((1, 64), dtype=mx.float16)

    monkeypatch.setattr(mx, "quantized_matmul", recording_qmm)
    x, w, scales, biases = _make_q1_tensors()
    biases_sym = -scales * 0.5
    bonsai_fast.bonsai_q1_affine_qmv_sym(x, w, scales, biases_sym)
    # Fallback when ext is None: sym delegates to affine which calls quantized_matmul
    assert calls, "quantized_matmul should have been called"
    assert calls[0]["bits"] == 1


def test_sym_q2_fast_py_fallback_routes_to_same_mlx(monkeypatch):
    """Symmetric q2 fast fallback calls mx.quantized_matmul."""
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    calls = []

    def recording_qmm(x, w, *, scales, biases, transpose, group_size, bits, stream=None):
        calls.append({"bits": bits})
        return mx.zeros((1, 64), dtype=mx.bfloat16)

    monkeypatch.setattr(mx, "quantized_matmul", recording_qmm)
    x, w, scales, biases = _make_q2_tensors(M=1)
    biases_sym = -scales
    bonsai_fast.bonsai_q2_affine_qmv_sym(x, w, scales, biases_sym)
    assert calls, "quantized_matmul should have been called"
    assert calls[0]["bits"] == 2


def test_sym_routes_to_ext_when_available(monkeypatch):
    called = {}

    def fake_sym(x, w, scales, biases, stream=None):
        called["fired"] = True
        return mx.zeros((1, 64), dtype=mx.float16)

    fake_ext = SimpleNamespace(
        bonsai_q1_affine_qmv_sym=fake_sym,
        abi_probe=lambda a: 1,
    )
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)
    x, w, scales, biases = _make_q1_tensors()
    bonsai_fast.bonsai_q1_affine_qmv_sym(x, w, scales, biases)
    assert called.get("fired") is True


# ---------------------------------------------------------------------------
# apply_bonsai_qmv_patch lifecycle
# ---------------------------------------------------------------------------


def test_patch_applies_when_native_available(monkeypatch):
    monkeypatch.setattr(bonsai_fast, "_ext", SimpleNamespace(abi_probe=lambda a: 1))
    remove_bonsai_qmv_patch()
    result = apply_bonsai_qmv_patch()
    assert result is True
    assert is_patch_active() is True


def test_patch_skipped_when_no_native(monkeypatch):
    monkeypatch.setattr(bonsai_fast, "_ext", None)
    remove_bonsai_qmv_patch()

    from omlx.patches import bonsai_qmv as bonsai_qmv_mod
    monkeypatch.setattr(bonsai_qmv_mod, "has_native", lambda: False)

    result = apply_bonsai_qmv_patch()
    assert result is False
    assert is_patch_active() is False


def test_patch_idempotent(monkeypatch):
    monkeypatch.setattr(bonsai_fast, "_ext", SimpleNamespace(abi_probe=lambda a: 1))
    remove_bonsai_qmv_patch()

    from omlx.patches import bonsai_qmv as bonsai_qmv_mod
    monkeypatch.setattr(bonsai_qmv_mod, "has_native", lambda: True)

    apply_bonsai_qmv_patch()
    original_call = nn.QuantizedLinear.__call__
    apply_bonsai_qmv_patch()  # second call should not re-wrap
    assert nn.QuantizedLinear.__call__ is original_call


def test_remove_restores_original():
    from omlx.patches import bonsai_qmv as bonsai_qmv_mod

    original = nn.QuantizedLinear.__call__
    bonsai_qmv_mod._original_quantized_linear_call = original
    bonsai_qmv_mod._patch_active = True
    nn.QuantizedLinear.__call__ = lambda self, x: x  # type: ignore[method-assign]

    remove_bonsai_qmv_patch()

    assert nn.QuantizedLinear.__call__ is original
    assert is_patch_active() is False


# ---------------------------------------------------------------------------
# model_loading wiring
# ---------------------------------------------------------------------------


def _write_config(tmp_path, body: str) -> str:
    (tmp_path / "config.json").write_text(body)
    return str(tmp_path)


class TestModelLoadingBonsaiWiring:
    def test_bits2_triggers_patch(self, tmp_path, monkeypatch):
        model_dir = _write_config(
            tmp_path,
            '{"model_type": "qwen3_5", "quantization": {"group_size": 128, "bits": 2}}',
        )
        applied = []
        monkeypatch.setattr(
            model_loading,
            "_patch_mlx_lm_load_config",
            lambda: None,
        )
        # Stub out apply_bonsai_qmv_patch inside model_loading
        from omlx.patches import bonsai_qmv as bonsai_qmv_mod
        monkeypatch.setattr(bonsai_qmv_mod, "has_native", lambda: True)
        monkeypatch.setattr(
            bonsai_qmv_mod,
            "apply_bonsai_qmv_patch",
            lambda: applied.append(True) or True,
        )
        maybe_apply_pre_load_patches(model_dir, "test-model", for_vlm=False)
        assert applied, "apply_bonsai_qmv_patch should have been called for bits=2"

    def test_bits1_triggers_patch(self, tmp_path, monkeypatch):
        model_dir = _write_config(
            tmp_path,
            '{"model_type": "bonsai", "quantization": {"group_size": 128, "bits": 1}}',
        )
        applied = []
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        from omlx.patches import bonsai_qmv as bonsai_qmv_mod
        monkeypatch.setattr(bonsai_qmv_mod, "has_native", lambda: True)
        monkeypatch.setattr(
            bonsai_qmv_mod,
            "apply_bonsai_qmv_patch",
            lambda: applied.append(True) or True,
        )
        maybe_apply_pre_load_patches(model_dir, "test-model", for_vlm=False)
        assert applied

    def test_bits4_skips_patch(self, tmp_path, monkeypatch):
        model_dir = _write_config(
            tmp_path,
            '{"model_type": "llama", "quantization": {"group_size": 64, "bits": 4}}',
        )
        applied = []
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        from omlx.patches import bonsai_qmv as bonsai_qmv_mod
        monkeypatch.setattr(
            bonsai_qmv_mod,
            "apply_bonsai_qmv_patch",
            lambda: applied.append(True) or True,
        )
        maybe_apply_pre_load_patches(model_dir, "test-model", for_vlm=False)
        assert not applied, "bits=4 should NOT trigger the bonsai patch"

    def test_no_quantization_field_skips_patch(self, tmp_path, monkeypatch):
        model_dir = _write_config(tmp_path, '{"model_type": "llama"}')
        applied = []
        monkeypatch.setattr(model_loading, "_patch_mlx_lm_load_config", lambda: None)
        from omlx.patches import bonsai_qmv as bonsai_qmv_mod
        monkeypatch.setattr(
            bonsai_qmv_mod,
            "apply_bonsai_qmv_patch",
            lambda: applied.append(True) or True,
        )
        maybe_apply_pre_load_patches(model_dir, "test-model", for_vlm=False)
        assert not applied


# ---------------------------------------------------------------------------
# ABI probe in the bonsai package is included in the shared parametrize suite
# ---------------------------------------------------------------------------


def test_bonsai_local_build_probe_is_healthy():
    """If the local build is available its abi_probe must accept mlx arrays."""
    if not bonsai_fast.is_native_available():
        pytest.skip("bonsai native build unavailable")
    assert bonsai_fast._ext.abi_probe(mx.zeros((3,))) == 3


# ---------------------------------------------------------------------------
# t5 (base-3 ternary packing, Identity I-D) tests
# ---------------------------------------------------------------------------


def _make_t5_layer(N: int = 64, K: int = 256, group_size: int = 128,
                   quants: np.ndarray | None = None) -> nn.QuantizedLinear:
    """QuantizedLinear with t5-format weights (uint8, base-3)."""
    from tools.repack_ternary_t5 import pack_t5

    if quants is None:
        rng = np.random.default_rng(42)
        quants = rng.integers(0, 3, size=(N, K), dtype=np.uint8)

    t5w = pack_t5(quants, group_size)
    n_groups = K // group_size
    scales = mx.ones((N, n_groups), dtype=mx.float16)

    layer = nn.QuantizedLinear.__new__(nn.QuantizedLinear)
    object.__setattr__(layer, "weight", mx.array(t5w))
    object.__setattr__(layer, "scales", scales)
    object.__setattr__(layer, "biases", -scales)  # symmetric: bias = -scale
    object.__setattr__(layer, "bits", 2)
    object.__setattr__(layer, "group_size", group_size)
    object.__setattr__(layer, "mode", "affine")
    return layer


class TestT5Repack:
    """Tests for tools/repack_ternary_t5.py."""

    def test_pack_unpack_roundtrip_gs128(self):
        from tools.repack_ternary_t5 import pack_t5, unpack_t5
        rng = np.random.default_rng(0)
        q = rng.integers(0, 3, size=(8, 128), dtype=np.uint8)
        t5w = pack_t5(q, group_size=128)
        assert t5w.shape == (8, 26), f"expected (8,26) got {t5w.shape}"
        q_rt = unpack_t5(t5w, group_size=128, K=128)
        np.testing.assert_array_equal(q, q_rt)

    def test_pack_unpack_roundtrip_gs64(self):
        from tools.repack_ternary_t5 import pack_t5, unpack_t5
        rng = np.random.default_rng(1)
        q = rng.integers(0, 3, size=(8, 64), dtype=np.uint8)
        t5w = pack_t5(q, group_size=64)
        assert t5w.shape == (8, 13), f"expected (8,13) got {t5w.shape}"
        q_rt = unpack_t5(t5w, group_size=64, K=64)
        np.testing.assert_array_equal(q, q_rt)

    def test_pack_unpack_larger_K(self):
        from tools.repack_ternary_t5 import pack_t5, unpack_t5
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
        from tools.repack_ternary_t5 import pack_t5
        # Single group of 128, last 2 positions zero-padded with q=1
        q = np.ones((1, 128), dtype=np.uint8)  # all t=0 (q=1 → dq=0 for scale*(q-1))
        t5w = pack_t5(q, group_size=128)
        # Decode last byte and check it encodes 3 active trits + 2 padding (all q=1)
        # byte v = 1 + 1*3 + 1*9 + 1*27 + 1*81 = 121
        assert t5w[0, 25] == 121  # 1+3+9+27+81

    def test_dequant_matches_2bit_reference(self):
        """t5 and 2-bit dequantize to the same float values."""
        from tools.repack_ternary_t5 import pack_t5, unpack_t5, unpack_mlx_2bit
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


class TestT5FormatDetection:
    """Tests for _is_t5_format detection in bonsai_qmv patch."""

    def test_detects_t5_gs128(self):
        from omlx.patches.bonsai_qmv import _is_t5_format
        layer = _make_t5_layer(N=64, K=256, group_size=128)
        assert _is_t5_format(layer) is True

    def test_detects_t5_gs64(self):
        from omlx.patches.bonsai_qmv import _is_t5_format
        layer = _make_t5_layer(N=64, K=256, group_size=64)
        assert _is_t5_format(layer) is True

    def test_rejects_uint32_weight(self):
        from omlx.patches.bonsai_qmv import _is_t5_format
        layer = _make_sym_layer(bits=2, N=64, K=256, group_size=128)
        # weight is uint32 (2-bit MLX format), not t5
        assert _is_t5_format(layer) is False

    def test_rejects_wrong_bytes_per_group(self):
        from omlx.patches.bonsai_qmv import _is_t5_format
        import mlx.nn as nn_inner
        # uint8 weight but bytes_per_group=32 (not 13 or 26)
        layer = nn_inner.QuantizedLinear.__new__(nn_inner.QuantizedLinear)
        object.__setattr__(layer, "weight", mx.zeros((64, 64), dtype=mx.uint8))
        object.__setattr__(layer, "scales", mx.ones((64, 2), dtype=mx.float16))
        object.__setattr__(layer, "bits", 2)
        object.__setattr__(layer, "mode", "affine")
        assert _is_t5_format(layer) is False

    def test_detection_cached(self):
        from omlx.patches.bonsai_qmv import _is_t5_format
        layer = _make_t5_layer(N=16, K=128, group_size=128)
        first = _is_t5_format(layer)
        # Change weight — cache should still return first result
        object.__setattr__(layer, "weight", mx.zeros((16, 3), dtype=mx.uint8))
        second = _is_t5_format(layer)
        assert first == second
