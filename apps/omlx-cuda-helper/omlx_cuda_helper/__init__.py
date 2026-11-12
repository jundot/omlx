# SPDX-License-Identifier: Apache-2.0
"""Prefill Accelerator helper for oMLX on Linux with NVIDIA GPUs.

It shares no code with oMLX but speaks its cluster discovery, pairing and
prefill-helper wire protocols.
"""

__version__ = "0.1.2"

# Framing and ops of the prefill data channel (oMLX ``wire.PROTOCOL``).
DATA_PROTOCOL = 1
# Ops of the control channel between a paired host and this helper.
CONTROL_VERSION = 1
# A stage worker prints this before the JSON that says it is ready.
READY_PREFIX = "OMLX_CUDA_HELPER_READY "
