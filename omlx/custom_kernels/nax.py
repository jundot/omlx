# SPDX-License-Identifier: Apache-2.0
"""Shared NAX (M5 tensor unit) detection for custom-kernel dispatch.

The implementation lives in qwen35_prefill.fast, which prefers the native
extension's mirror of mlx metal::is_nax_available() and falls back to parsing
mx.device_info() when the extension predates the NAX split. Import from here
in patches that are not Qwen-specific.
"""

from omlx.custom_kernels.qwen35_prefill.fast import is_nax_available

__all__ = ["is_nax_available"]
