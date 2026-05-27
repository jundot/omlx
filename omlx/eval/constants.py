# SPDX-License-Identifier: Apache-2.0
"""Constants and enums for the evaluation framework.

This module provides canonical benchmark keys and OQ level values
to eliminate "magic strings" across the codebase.
"""

from __future__ import annotations

from enum import Enum, auto


class BenchmarkKey(str, Enum):
    """Canonical benchmark identifiers used throughout the evaluation suite.

    These keys are used consistently across the eval framework,
    API endpoints, and frontend/backend DTOs.
    """

    ARC_CHALLENGE = "arc_challenge"
    HUMAN_EVAL = "humaneval"
    LIVECODEBENCH = "livecodebench"
    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    GSM8K = "gsm8k"
    MBPP = "mbpp"
    MATHQA = "mathqa"
    CMMLU = "cmmlu"
    JMMLU = "jmmlu"
    KMMLU = "kmmlu"
    HELLASWAG = "hellaswag"
    BBQ = "bbq"
    TRUTHFULQA = "truthfulqa"
    SAFETY_BENCH = "safetybench"
    WINOGRANDE = "winogrande"


class OQLevel:
    """Quantization level identifiers.

    Used to validate and normalize OQ level inputs across the API,
    admin panel, and quantization engine.
    """

    VALID = {2, 3, 3.5, 4, 5, 6, 8}

    @classmethod
    def is_valid(cls, level: float) -> bool:
        return level in cls.VALID

    @classmethod
    def from_string(cls, s: str) -> float | None:
        """Parse a string into an OQ level, or None if invalid."""
        try:
            return float(s)
        except (ValueError, TypeError):
            return None


# Pre-built alias for backward compatibility
OQ_LEVELS = OQLevel.VALID
