# SPDX-License-Identifier: Apache-2.0
"""Use ordinary MLX projections for multi-request Qwen verification."""

from functools import wraps


def apply(language):
    if getattr(language, "_omlx_mtp_batch_linear", False):
        return
    original = language._use_target_verify_dense

    @wraps(original)
    def use_verify_dense(linear, x, target_verify):
        # Preserve dtype/quantization and let MLX reuse weights across B*T rows.
        # Singleton verification keeps its existing numerical path.
        if x.ndim == 3 and x.shape[0] > 1:
            return False
        return original(linear, x, target_verify)

    language._use_target_verify_dense = use_verify_dense
    language._omlx_mtp_batch_linear = True
