# SPDX-License-Identifier: Apache-2.0
"""Tests for the custom-kernel nanobind ABI probe (issue #2139).

An extension built with a nanobind whose ABI tag differs from the mlx
wheel's imports cleanly and lists every symbol, but rejects every mlx
array at call time. ``_verify_abi`` must catch that once at import and
disable the native symbols instead of letting each routed call raise.
"""

import pytest

from omlx.custom_kernels.glm_moe_dsa import fast as glm_fast
from omlx.custom_kernels.minimax_m3 import fast as minimax_fast
from omlx.custom_kernels.qwen35_prefill import fast as qwen35_fast

ALL_FAST = (qwen35_fast, glm_fast, minimax_fast)


class _MismatchedExt:
    """Mimics a wrong-nanobind build: symbols exist, every call raises."""

    def abi_probe(self, a):
        raise TypeError(
            "abi_probe(): incompatible function arguments. The following "
            "argument types are supported: ..."
        )


class _HealthyExt:
    def abi_probe(self, a):
        return 1


class _LegacyExt:
    """A build predating the probe symbol: assumed compatible."""


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_mismatched_build_is_disabled_with_import_error(fast):
    ext, err = fast._verify_abi(_MismatchedExt(), None)
    assert ext is None
    assert isinstance(err, TypeError)


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_healthy_build_passes_through(fast):
    ext = _HealthyExt()
    out, err = fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_legacy_build_without_probe_passes_through(fast):
    ext = _LegacyExt()
    out, err = fast._verify_abi(ext, None)
    assert out is ext
    assert err is None


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_missing_extension_passes_through(fast):
    sentinel = ImportError("no native build")
    out, err = fast._verify_abi(None, sentinel)
    assert out is None
    assert err is sentinel


@pytest.mark.parametrize("fast", ALL_FAST, ids=lambda m: m.__name__)
def test_local_build_probe_is_healthy(fast):
    """The in-tree builds must expose abi_probe and accept mlx arrays."""
    if not fast.is_native_available():
        pytest.skip(f"{fast.__name__} native build unavailable")
    import mlx.core as mx

    assert fast._ext.abi_probe(mx.zeros((3,))) == 3