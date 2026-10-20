# SPDX-License-Identifier: Apache-2.0
"""Optional bounded expert-read transport; no import-time native allocation."""

import logging
import threading
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ..quant_matmul.fast import _ext as _quant_ext

logger = logging.getLogger(__name__)

_native = getattr(_quant_ext, "moe_staging", None)
NATIVE_AVAILABLE = _native is not None

# Match the qualified 40-expert pool, independent of IO batch size. Refuse
# oversized layouts rather than silently increasing model memory admission.
EXPERTS = 40
MAX_BYTES = 128 * 1024 * 1024
_DTYPES = {
    np.dtype(np.uint16): mx.uint16,
    np.dtype(np.float16): mx.float16,
    np.dtype(np.float32): mx.float32,
    np.dtype(np.uint32): mx.uint32,
    np.dtype(np.int32): mx.int32,
    np.dtype(np.uint8): mx.uint8,
}


@dataclass
class StagedRead:
    plan: object
    lease: object


class Staging:
    """One shared pool per model. Initialize lazily on the inference owner.

    Workers only perform pread into pre-resolved addresses. MLX's shared Data
    retains each lease through lazy graphs and GPU completion, using a pure
    C++ deleter that never needs the GIL. Another calling thread uses bytes;
    it cannot convert arrays on the original owner's stream.
    """

    def __init__(self, sizes, read, to_mx):
        self.sizes = sizes
        self.original_read = read
        self.original_to_mx = to_mx
        self.pool = self.owner = None
        self.disabled = False
        self.lock = threading.Lock()

    def for_owner(self):
        with self.lock:
            if self.disabled:
                return None
            if self.owner is not None:
                return self if self.owner == threading.get_ident() else None
            try:
                self.pool = _native.Pool(self.sizes)
                self.owner = threading.get_ident()
            except Exception:
                self.disabled = True
                logger.warning(
                    "moe offload staging unavailable; using bytes transport",
                    exc_info=True,
                )
                return None
            logger.info("moe offload staging active: %s", self.pool.stats())
            return self

    def read(self, plan):
        lease = self.pool.read(plan.fd, plan.offset, plan.nbytes)
        return (
            StagedRead(plan, lease) if lease is not None else self.original_read(plan)
        )

    def to_mx(self, plan, payload):
        if not isinstance(payload, StagedRead):
            return self.original_to_mx(plan, payload)
        if payload.plan is not plan:
            raise ValueError("Staged payload belongs to a different plan")
        out = _native.to_array(
            payload.lease, list(plan.shape), _DTYPES[np.dtype(plan.np_dtype)]
        )
        return out.view(plan.mx_view) if plan.mx_view is not None else out
