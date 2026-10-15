# SPDX-License-Identifier: Apache-2.0
"""Optional exact decode SDPA kernel used by Qwen4 QSA."""

from .fast import NATIVE_AVAILABLE, sdpa_decode

__all__ = ["NATIVE_AVAILABLE", "sdpa_decode"]
