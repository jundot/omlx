# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator: a helper Mac runs the first layers of each prompt chunk.

This Mac keeps the whole model and does all decoding. For each prefill chunk
the helper runs layers ``[0, k)`` and sends back the tensors that cross the cut
plus the chunk's cache state for those layers; this Mac runs ``[k, n)`` and then
applies that state, so every cache ends the chunk at the same offset.
"""

from .adapter import StageAdapter
from .adapters import adapter_for

__all__ = ["StageAdapter", "adapter_for"]
