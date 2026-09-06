# SPDX-License-Identifier: Apache-2.0
"""Tests for NAX (M5 tensor unit) detection and qmm dispatch gating."""

from __future__ import annotations

import types

import pytest

import omlx.custom_kernels.qwen35_prefill.fast as fast
from omlx.custom_kernels.nax import is_nax_available


@pytest.fixture(autouse=True)
def _fresh_nax_state(monkeypatch):
    monkeypatch.setattr(fast, "_nax_available_cache", None)
    monkeypatch.setattr(fast, "_stock_nax_cache", None)
    monkeypatch.setattr(fast, "_qmm_nax_cache", None)
    monkeypatch.delenv("OMLX_NAX", raising=False)
    monkeypatch.delenv("OMLX_QWEN35_QMM_NAX", raising=False)
    yield


@pytest.mark.parametrize(
    ("version", "arch", "expected"),
    [
        ("26.2", "applegpu_g17s", True),
        ("26.2.1", "applegpu_g17d", True),
        ("26.2", "applegpu_g18p", True),
        ("26.2", "applegpu_g17p", False),
        ("26.2", "applegpu_g15d", False),
        ("26.1", "applegpu_g17s", False),
        ("15.5", "applegpu_g17s", False),
        ("26.2", "applegpu_gXYs", False),
        ("26.2", "", False),
        ("garbage", "applegpu_g17s", False),
    ],
)
def test_nax_fallback_mirrors_mlx_gate(version, arch, expected):
    assert fast._nax_available_fallback(version, arch) is expected


def test_is_nax_available_env_override(monkeypatch):
    monkeypatch.setenv("OMLX_NAX", "1")
    assert fast.is_nax_available() is True
    monkeypatch.setenv("OMLX_NAX", "0")
    assert fast.is_nax_available() is False


def test_is_nax_available_uses_fallback_without_ext(monkeypatch):
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", False)
    monkeypatch.setattr(fast, "_nax_available_fallback", lambda: True)
    monkeypatch.setattr(fast, "_stock_mlx_has_nax", lambda: True)
    assert fast.is_nax_available() is True


def test_is_nax_available_requires_stock_nax_kernels(monkeypatch):
    # NAX hardware with a no-NAX mlx wheel (e.g. the macosx_15 sequoia
    # bundle): stock stays classic, so route-to-stock must not engage.
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", False)
    monkeypatch.setattr(fast, "_nax_available_fallback", lambda: True)
    monkeypatch.setattr(fast, "_stock_mlx_has_nax", lambda: False)
    assert fast.is_nax_available() is False


def test_stock_mlx_probe_scans_metallib(tmp_path):
    with_nax = tmp_path / "with_nax.metallib"
    with_nax.write_bytes(b"\x00" * 100 + b"affine_qmm_t_nax_bfloat16_t" + b"\x00" * 100)
    assert fast._stock_mlx_has_nax(with_nax) is True

    without_nax = tmp_path / "without_nax.metallib"
    without_nax.write_bytes(b"\x00" * 100 + b"affine_qmm_t_classic" + b"\x00" * 100)
    assert fast._stock_mlx_has_nax(without_nax) is False

    # Absent metallib (JIT build) falls back to the hardware-only gate.
    assert fast._stock_mlx_has_nax(tmp_path / "missing.metallib") is True


def test_stock_mlx_probe_finds_needle_across_chunks(tmp_path, monkeypatch):
    lib = tmp_path / "boundary.metallib"
    chunk = 1 << 23
    needle = b"affine_qmm_t_nax"
    # Place the needle straddling the first chunk boundary.
    lib.write_bytes(b"\x00" * (chunk - 8) + needle + b"\x00" * 64)
    assert fast._stock_mlx_has_nax(lib) is True


def test_nax_shim_reexports_fast_impl():
    assert is_nax_available is fast.is_nax_available


def test_qmm_nax_kwargs_empty_for_pre_nax_ext(monkeypatch):
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", False)
    assert fast._qmm_nax_kwargs() == {}


def test_qmm_nax_kwargs_on_nax_machine(monkeypatch):
    fake_ext = types.SimpleNamespace(
        is_nax_available=lambda: True,
        nax_qmm_kernels_built=lambda: True,
    )
    monkeypatch.setattr(fast, "_ext", fake_ext)
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", True)
    kwargs = fast._qmm_nax_kwargs()
    assert kwargs["use_nax"] is True
    assert kwargs["nax_variant"] == fast.QMM_NAX_VARIANT


def test_qmm_nax_env_kill_switch(monkeypatch):
    fake_ext = types.SimpleNamespace(
        is_nax_available=lambda: True,
        nax_qmm_kernels_built=lambda: True,
    )
    monkeypatch.setattr(fast, "_ext", fake_ext)
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", True)
    monkeypatch.setenv("OMLX_QWEN35_QMM_NAX", "0")
    assert fast._qmm_nax_kwargs()["use_nax"] is False


def test_qmm_nax_disabled_without_kernels(monkeypatch):
    fake_ext = types.SimpleNamespace(
        is_nax_available=lambda: True,
        nax_qmm_kernels_built=lambda: False,
    )
    monkeypatch.setattr(fast, "_ext", fake_ext)
    monkeypatch.setattr(fast, "_EXT_HAS_NAX", True)
    assert fast._qmm_nax_kwargs()["use_nax"] is False


def test_ane_hybrid_nax_capability_reports_native_state(monkeypatch):
    fake_ext = types.SimpleNamespace(qwen35_ane_hybrid_nax_enabled=lambda: True)
    monkeypatch.setattr(fast, "_ext", fake_ext)
    assert fast.qwen35_ane_hybrid_nax_enabled() is True


def test_ane_hybrid_nax_capability_is_false_for_older_extension(monkeypatch):
    monkeypatch.setattr(fast, "_ext", types.SimpleNamespace())
    assert fast.qwen35_ane_hybrid_nax_enabled() is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("5", 5), ("6", 0), ("-1", 0), ("junk", 0), (" 2 ", 2)],
)
def test_qmm_nax_variant_env_is_validated(monkeypatch, raw, expected):
    monkeypatch.setenv("OMLX_QWEN35_QMM_NAX_VARIANT", raw)
    monkeypatch.setattr(fast, "_qmm_nax_variant_warned", False)
    assert fast._resolve_qmm_nax_variant() == expected


def _nax_qmm_ready() -> bool:
    return (
        fast.is_native_available()
        and fast.is_nax_available()
        and fast.nax_qmm_kernels_built()
        and fast._qmm_use_nax()
    )


@pytest.mark.skipif(not _nax_qmm_ready(), reason="bundled NAX qmm kernels unavailable")
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("variant", list(fast.NAX_QMM_VARIANTS))
def test_every_bundled_nax_qmm_tile_matches_stock(
    monkeypatch, bits, group_size, variant
):
    """Each opt-in tile must reproduce stock MLX (the dropped wn=4 tile did not)."""
    import mlx.core as mx

    n, k, t = 640, 512, 320
    keys = mx.random.split(mx.random.key(bits * 100 + variant), 2)
    w = (mx.random.normal((n, k), key=keys[0]) * 0.05).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    x = mx.random.normal((1, t, k), key=keys[1]).astype(mx.bfloat16)
    ref = mx.quantized_matmul(
        x, wq, scales, biases, transpose=True, group_size=group_size, bits=bits
    )
    monkeypatch.setattr(fast, "QMM_NAX_VARIANT", variant)
    native = getattr(fast, f"qwen35_q{bits}_affine_qmm_t")
    out = native(x, wq, scales, biases, 8, group_size)
    ref32 = ref.astype(mx.float32)
    err = mx.abs(out.astype(mx.float32) - ref32).max().item()
    scale = mx.abs(ref32).max().item()
    # Reduction order costs at most a bf16 ulp or two; the dropped wn=4 tile
    # was off by whole units (err ~ scale).
    assert err <= 0.02 * scale, (
        f"variant {variant} q{bits}/gs{group_size}: max err {err} (scale {scale})"
    )
