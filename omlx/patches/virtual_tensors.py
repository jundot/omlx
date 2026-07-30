# SPDX-License-Identifier: Apache-2.0
"""Virtual source-tensor registration for the oQ streaming quantizer.

Some checkpoints store a tensor in a layout the model's ``sanitize`` has to
restructure before use — for example a fused, tensor-parallel-sharded
``qkv_proj`` that must be dequantized per shard and split into q/k/v.

oQ's streaming path discovers sanitize's key mapping by running it over
recorder proxies, which can only replay single-source unary ops. Restructuring
that needs a second live tensor (a block scale) cannot be recorded, so it has
to happen where the source tensors are actually read instead.

This module is the seam. Model-specific registrars declare *virtual* keys on
the lazy tensor index — a logical name, its shape and dtype, a callable that
produces it, and the on-disk keys it consumes (which become hidden). The
quantizer itself stays generic: it knows only that an index may carry virtual
tensors, never which models need them.

Registrars must be conservative:

- gate on ``config`` (model type / declared layout), and
- detect their candidates from the tensors actually present, never assuming
  they exist — the same config is reused for oQ outputs and for the
  calibration proxy, both of which already ship restructured tensors, and
  re-quantizing those must be a no-op.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register_virtual_tensors(index, config) -> int:
    """Give every known registrar a chance to claim tensors on ``index``.

    Returns the total number of registrations made. A registrar that raises is
    fatal by design: a checkpoint whose layout we half-understand must not be
    quantized silently.
    """
    if not config:
        return 0

    total = 0
    try:
        from .mimo_v2.virtual_qkv import register as register_mimo_v2_qkv
    except ImportError:
        logger.debug("mimo_v2 virtual-qkv registrar unavailable", exc_info=True)
    else:
        total += register_mimo_v2_qkv(index, config)

    return total
