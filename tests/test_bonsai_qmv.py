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

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
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
        # M < 3: per-row qmv_fast for both 1-bit and 2-bit (overhead > gain)
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
    """bits=1 always routes through bonsai_q1_affine_qmv (per-row fast path)."""
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

    def fake_verify(draft_tokens, target_logits, stream=None):
        called["fired"] = True
        B = draft_tokens.shape[0]
        K = draft_tokens.shape[1]
        return mx.zeros((B,), mx.int32), mx.zeros((B, K + 1), mx.int32)

    fake_ext = SimpleNamespace(bonsai_spec_decode_verify=fake_verify)
    monkeypatch.setattr(bonsai_fast, "_ext", fake_ext)

    draft = mx.array([[1, 2]], dtype=mx.int32)
    target_logits = mx.zeros((1, 3, 8), dtype=mx.float32)
    bonsai_fast.spec_decode_verify(draft, target_logits)
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
