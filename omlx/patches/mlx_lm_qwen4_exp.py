# SPDX-License-Identifier: Apache-2.0
"""mlx-lm qwen4_exp fixes that must apply outside the mlx-vlm compat path."""

from __future__ import annotations

import bisect
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False


def _patch_qwen4_exp_sanitize(qwen4_exp_module: Any) -> None:
    """Ensure Model.sanitize folds 1.0 into norms for checkpoints prefixed with language_model."""
    model_cls = getattr(qwen4_exp_module, "Model", None)
    if model_cls is None or getattr(model_cls, "_omlx_sanitize_patched", False):
        return

    def sanitize(self, weights):
        fold = any(
            k.startswith(("model.language_model.", "language_model."))
            for k in weights
        )
        out = {}
        for k, v in weights.items():
            if k.startswith(
                ("mtp.", "model.mtp.", "model.visual.", "visual.", "vision_tower.")
            ):
                continue
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model.") :]
            elif k.startswith("language_model."):
                k = k[len("language_model.") :]

            if k.endswith("mlp.experts.gate_up_proj"):
                base = k[: -len("experts.gate_up_proj")]
                mid = v.shape[-2] // 2
                out[base + "switch_mlp.gate_proj.weight"] = v[..., :mid, :]
                out[base + "switch_mlp.up_proj.weight"] = v[..., mid:, :]
                continue
            if k.endswith("mlp.experts.down_proj"):
                base = k[: -len("experts.down_proj")]
                out[base + "switch_mlp.down_proj.weight"] = v
                continue

            if "ngram_embedding.shards." in k:
                k = k.replace("ngram_embedding.shards.", "ngram_embedding.shard_")

            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[1] == 1:
                v = v.transpose(0, 2, 1)

            if fold and k.endswith(self._FOLD_ONE):
                v = 1.0 + v

            out[k] = v
        return out

    model_cls.sanitize = sanitize
    model_cls._omlx_sanitize_patched = True
    logger.debug("Patched mlx_lm qwen4_exp Model.sanitize for language_model norm folding")


def apply_mlx_lm_qwen4_exp_patch(
    model_name: str | Path | None = None,
    model_settings: Any | None = None,
) -> bool:
    """Inject bisect_right, norm folding, and PLE runtime for mlx_lm.models.qwen4_exp."""
    global _APPLIED
    try:
        import mlx_lm.models.qwen4_exp as qwen4_exp  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return False

    if not hasattr(qwen4_exp, "bisect_right"):
        qwen4_exp.bisect_right = bisect.bisect_right
        logger.debug("Injected mlx-lm qwen4_exp.bisect_right (upstream missing import)")

    _patch_qwen4_exp_sanitize(qwen4_exp)

    if model_name:
        resolved_path = str(Path(model_name).expanduser().resolve())
        os.environ["OMLX_QWEN4_PLE_PATH"] = resolved_path
        mode = "mmap"
        if model_settings is not None and not getattr(
            model_settings, "qwen4_ple_ssd_offload", True
        ):
            mode = "resident"
        os.environ["OMLX_QWEN4_PLE_MODE"] = mode
        logger.debug(
            "Configured mlx-lm qwen4_exp PLE runtime: path=%s mode=%s",
            resolved_path,
            mode,
        )

    try:
        from .mlx_vlm_qwen4_exp_compat import _patch_prompt_loop

        _patch_prompt_loop()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to patch prompt loop for PLE lookahead: %s", exc)

    _APPLIED = True
    return True
