# SPDX-License-Identifier: Apache-2.0
"""Decode fast-path kernels used by oMLX runtime patches."""

from .fast import NATIVE_AVAILABLE, rms_norm_residual

__all__ = ["NATIVE_AVAILABLE", "rms_norm_residual"]
