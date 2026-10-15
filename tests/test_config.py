# SPDX-License-Identifier: Apache-2.0
"""Tests for omlx.config module."""

import pytest

from omlx.config import parse_size


class TestParseSize:
    """Test cases for parse_size function."""

    def test_parse_bytes(self):
        """Test parsing byte values."""
        assert parse_size("100B") == 100
        assert parse_size("0B") == 0
        assert parse_size("1024B") == 1024

    def test_parse_kilobytes(self):
        """Test parsing KB values."""
        assert parse_size("1KB") == 1024
        assert parse_size("100KB") == 100 * 1024
        assert parse_size("1.5KB") == int(1.5 * 1024)

    def test_parse_megabytes(self):
        """Test parsing MB values."""
        assert parse_size("1MB") == 1024**2
        assert parse_size("512MB") == 512 * 1024**2
        assert parse_size("2.5MB") == int(2.5 * 1024**2)

    def test_parse_gigabytes(self):
        """Test parsing GB values."""
        assert parse_size("1GB") == 1024**3
        assert parse_size("16GB") == 16 * 1024**3
        assert parse_size("32.5GB") == int(32.5 * 1024**3)

    def test_parse_terabytes(self):
        """Test parsing TB values."""
        assert parse_size("1TB") == 1024**4
        assert parse_size("2TB") == 2 * 1024**4

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        assert parse_size("1gb") == 1024**3
        assert parse_size("1Gb") == 1024**3
        assert parse_size("1gB") == 1024**3
        assert parse_size("1GB") == 1024**3

    def test_parse_with_whitespace(self):
        """Test parsing with leading/trailing whitespace."""
        assert parse_size("  1GB  ") == 1024**3
        assert parse_size("\t16GB\n") == 16 * 1024**3

    def test_parse_plain_number(self):
        """Test parsing plain number as bytes."""
        assert parse_size("1024") == 1024
        assert parse_size("0") == 0

    def test_parse_invalid_raises_error(self):
        """Test that invalid input raises ValueError."""
        with pytest.raises(ValueError):
            parse_size("invalid")
        with pytest.raises(ValueError):
            parse_size("abc123")
        with pytest.raises(ValueError):
            parse_size("1XB")  # Invalid unit

    @pytest.mark.parametrize(
        "value", ["infMB", "-infMB", "nanMB", "1e999MB", "1e308TB"]
    )
    def test_parse_non_finite_raises_value_error(self, value):
        """Non-finite unit values are rejected as invalid size strings."""
        with pytest.raises(ValueError, match="Invalid size string"):
            parse_size(value)

