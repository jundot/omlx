# SPDX-License-Identifier: Apache-2.0
"""Model-type predicates that avoid the deepseek_v4 / deepseek_v41 collision.

``str.startswith("deepseek_v4")`` matches both ``deepseek_v4`` and
``deepseek_v41``. Use these helpers everywhere we dispatch patches.
"""
from __future__ import annotations


def is_deepseek_v41(model_type: str | None) -> bool:
    if not isinstance(model_type, str):
        return False
    return model_type == "deepseek_v41" or model_type.startswith("deepseek_v41")


def is_deepseek_v4(model_type: str | None) -> bool:
    """True for DeepSeek-V4 / V4-Flash, NOT V4.1."""
    if not isinstance(model_type, str):
        return False
    if is_deepseek_v41(model_type):
        return False
    return model_type == "deepseek_v4" or model_type.startswith("deepseek_v4_")


def is_deepseek_v4_family(model_type: str | None) -> bool:
    """True for either V4 or V4.1 (shared tooling that is version-agnostic)."""
    return is_deepseek_v4(model_type) or is_deepseek_v41(model_type)
