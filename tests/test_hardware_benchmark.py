# SPDX-License-Identifier: Apache-2.0
"""Tests for hardware detection functions used in omlx.ai benchmark integration."""

import pytest

from omlx.utils.hardware import (
    _OWNER_HASH_ALPHABET,
    compute_owner_hash,
    get_os_version,
    parse_chip_info,
)


class TestParseChipInfo:
    def test_m4_pro(self):
        assert parse_chip_info("Apple M4 Pro") == ("M4", "Pro")

    def test_m3_max(self):
        assert parse_chip_info("Apple M3 Max") == ("M3", "Max")

    def test_m2_ultra(self):
        assert parse_chip_info("Apple M2 Ultra") == ("M2", "Ultra")

    def test_m1_base(self):
        assert parse_chip_info("Apple M1") == ("M1", "")

    def test_m4_base(self):
        assert parse_chip_info("Apple M4") == ("M4", "")

    def test_m5_pro(self):
        assert parse_chip_info("Apple M5 Pro") == ("M5", "Pro")

    def test_fallback(self):
        assert parse_chip_info("Apple Silicon") == ("M1", "")

    def test_empty_string(self):
        assert parse_chip_info("") == ("M1", "")


class TestComputeOwnerHash:
    def test_deterministic(self):
        h1 = compute_owner_hash("UUID-123", "M4", 12, 64)
        h2 = compute_owner_hash("UUID-123", "M4", 12, 64)
        assert h1 == h2

    def test_different_inputs_differ(self):
        h1 = compute_owner_hash("UUID-123", "M4", 12, 64)
        h2 = compute_owner_hash("UUID-456", "M4", 12, 64)
        assert h1 != h2

    def test_format(self):
        h = compute_owner_hash("test-uuid", "M4", 16, 128)
        # SHA-256 hex = 64 chars + 1 verify char = 65
        assert len(h) == 65
        # Last char should be in alphabet
        assert h[-1] in _OWNER_HASH_ALPHABET
        # Hash body should be hex
        assert all(c in "0123456789abcdef" for c in h[:-1])

    def test_verify_char_correct(self):
        h = compute_owner_hash("test-uuid", "M3", 10, 32)
        body = h[:-1]
        verify = h[-1]
        expected_sum = sum(ord(c) for c in body)
        expected_char = _OWNER_HASH_ALPHABET[expected_sum % 36]
        assert verify == expected_char

    def test_none_gpu_cores(self):
        # Should not crash with None gpu_cores
        h = compute_owner_hash("test-uuid", "M1", None, 8)
        assert len(h) == 65


class TestGetOsVersion:
    def test_returns_string(self):
        result = get_os_version()
        assert isinstance(result, str)
        assert result.startswith("macOS")
