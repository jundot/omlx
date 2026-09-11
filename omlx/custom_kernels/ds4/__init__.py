# SPDX-License-Identifier: Apache-2.0
"""Optional DeepSeek V4 kernel-provider contracts and provenance."""

from .provider import (
    DS4_FLASH_FINGERPRINT,
    DS4CheckpointLayout,
    DS4DispatchResult,
    DS4ExecutionMode,
    DS4KernelProvider,
    DS4KernelRequest,
    DS4ProviderCapability,
    DS4ProviderDecisionCode,
    DS4ProviderSupport,
    dispatch_ds4_kernel,
    ds4_flash_fingerprint_mismatches,
    is_exact_ds4_flash_config,
    resolve_ds4_provider,
)

__all__ = [
    "DS4CheckpointLayout",
    "DS4DispatchResult",
    "DS4ExecutionMode",
    "DS4_FLASH_FINGERPRINT",
    "DS4KernelProvider",
    "DS4KernelRequest",
    "DS4ProviderCapability",
    "DS4ProviderDecisionCode",
    "DS4ProviderSupport",
    "dispatch_ds4_kernel",
    "ds4_flash_fingerprint_mismatches",
    "is_exact_ds4_flash_config",
    "resolve_ds4_provider",
]
