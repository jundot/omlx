# SPDX-License-Identifier: Apache-2.0
"""Gate collision tests for DeepSeek-V4 vs V4.1 predicates."""
from __future__ import annotations

from omlx.patches.deepseek_v41.predicates import (
    is_deepseek_v4,
    is_deepseek_v41,
    is_deepseek_v4_family,
)


def test_v41_not_classified_as_v4():
    assert is_deepseek_v41("deepseek_v41")
    assert is_deepseek_v41("deepseek_v41_text")
    assert not is_deepseek_v4("deepseek_v41")
    assert not is_deepseek_v4("deepseek_v41_text")


def test_v4_still_matches():
    assert is_deepseek_v4("deepseek_v4")
    assert is_deepseek_v4("deepseek_v4_mtp")
    assert not is_deepseek_v41("deepseek_v4")


def test_family():
    assert is_deepseek_v4_family("deepseek_v4")
    assert is_deepseek_v4_family("deepseek_v41")
    assert not is_deepseek_v4_family("llama")


def test_naive_startswith_is_the_bug_we_fixed():
    # Document why predicates exist.
    assert "deepseek_v41".startswith("deepseek_v4")
    assert not is_deepseek_v4("deepseek_v41")
