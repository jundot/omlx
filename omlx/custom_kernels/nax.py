# SPDX-License-Identifier: Apache-2.0
"""Shared NAX (M5 tensor unit) detection for custom-kernel dispatch.

The implementation lives in quant_matmul.fast, which prefers the native
extension's mirror of mlx metal::is_nax_available() and falls back to parsing
mx.device_info() when the extension is missing.
"""

from omlx.custom_kernels.quant_matmul.fast import is_nax_available

__all__ = ["is_nax_available"]
