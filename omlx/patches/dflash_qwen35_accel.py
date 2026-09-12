# SPDX-License-Identifier: Apache-2.0
"""Extend oMLX's Qwen3.5/3.6/3.8 acceleration paths to the DFlash target model.

``BatchedEngine`` and ``VLMBatchedEngine`` install a stack of Qwen-specific
prefill accelerations before serving (native q4/q8 MLP and prefill linear tiles,
the FA-256 steel attention route, the GDN prework and chunked-prefill kernels,
and the private ANE/GPU prefill split).  ``DFlashEngine`` installs none of them,
so a DFlash-served target runs an unaccelerated prefill: measured 81 tok/s on an
8K prompt versus 105-108 tok/s for the same checkpoint on the batched path with
ANE prefill enabled.

Not every patch in that stack can reach a DFlash target.  dflash-mlx drives Qwen
GDN targets through the **mlx-lm** module tree
(``dflash_mlx/engine/target_qwen_gdn.py`` imports ``mlx_lm.models.{cache,
gated_delta,base}``), while several oMLX patches rebind classes in
``mlx_vlm.models.qwen3_5``:

* ``qwen35_q4_mlp`` patches both trees, so it applies.
* ``qwen35_fa256_attention`` patches both ``mlx_lm.models.base`` and
  ``mlx_vlm.models.base``, so it applies.
* ``enable_qwen35_ane_prefill`` walks the loaded model's modules, so it applies
  regardless of tree.
* ``qwen35_gdn_prework``, ``qwen35_gdn_chunked``, ``qwen35_verify_sdpa_split``,
  ``qwen35_ragged_decode`` and the mlx-vlm half of the prefill-linear patch are
  deliberately **not** installed here. They bind ``mlx_vlm`` seams that a DFlash
  target never executes -- even a VLM checkpoint runs its text backbone through
  dflash-mlx's bundled text-only mlx-lm module -- so installing them would
  mutate VLM globals with no path to run them. When dflash falls back to
  ``VLMBatchedEngine``, that engine installs the mlx-vlm patch set itself.

Each piece is gated by the same per-model setting the batched engines use --
``qwen35_q4_mlp_prefill_enabled``, ``fa256_steel_prefill_enabled``,
and
``qwen35_ane_prefill_enabled`` (itself default off) -- so a DFlash-served model
behaves like the same model on the batched path.  ``OMLX_DFLASH_QWEN35_ACCEL=0``
is a kill switch for the whole set.

**These are not bit-exact fast paths.**  The native qmm tiles reorder
accumulation and the ANE prefix re-quantizes selected weights to
per-output-channel INT8, so greedy output drifts from the unpatched path once a
generation runs long enough for one flipped argmax to compound.  Measured on
Qwen3.8-27B with a 2960-token prompt and 200 greedy tokens, all three regimes
produce different text: unpatched versus class-patches-only diverges 40% in,
unpatched versus the full stack 95% in, and the continuations are semantically
equivalent rather than corrupt.  A 128-token probe on the same prompt showed no
divergence at all, so short hash checks are not evidence of equality here.

This is the same class of behaviour the batched engines already ship by default
for these patches; what changes is that a DFlash-served target now shares it.
``OMLX_DFLASH_QWEN35_ACCEL=0`` stops this DFlash load from enabling the stack --
in a fresh process that is the previous path exactly, but it cannot un-install
patches a batched or VLM engine already placed in the same process, since they
are process-global with module-level idempotency flags.
``qwen35_ane_prefill_enabled=false`` drops only the INT8 prefix (keeping the
reordered-but-full-precision native tiles).  Task-level quality was not
evaluated; draft acceptance is unchanged at 76.4%.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def accel_enabled() -> bool:
    """False only when explicitly killed; otherwise per-setting gating applies."""
    return os.environ.get("OMLX_DFLASH_QWEN35_ACCEL", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _install_class_patches(model_settings: Any) -> dict[str, bool]:
    """Install the class-level Qwen prefill patches. Each is independently guarded."""
    results: dict[str, bool] = {}

    def wanted(name: str) -> bool:
        # Match the batched engines: these default on, and only an explicit
        # False in the per-model settings disables them.
        return getattr(model_settings, name, True) is not False if model_settings else True

    def attempt(name: str, fn) -> None:
        try:
            results[name] = bool(fn())
        except Exception:
            results[name] = False
            logger.debug("dflash accel: %s not applied", name, exc_info=True)

    if wanted("qwen35_q4_mlp_prefill_enabled"):
        try:
            from .qwen35_q4_mlp import (
                apply_qwen35_q4_lm_prefill_linear_patch,
                apply_qwen35_q4_mlp_patch,
            )

            # apply_qwen35_q4_mlp_patch rebinds the MLP in both trees in one
            # call, so a little mlx-vlm mutation is unavoidable; the mlx-lm half
            # is the one a DFlash target executes.
            attempt("q4_mlp", apply_qwen35_q4_mlp_patch)
            # The mlx-lm prefill-linear patch wraps mlx-lm Attention and
            # GatedDeltaNet projections -- the tree dflash actually drives.
            # BatchedEngine installs this same one (engine/batched.py). Its
            # mlx-vlm sibling is deliberately skipped: see the module docstring.
            attempt("q4_lm_prefill_linear", apply_qwen35_q4_lm_prefill_linear_patch)
        except Exception:
            logger.debug("dflash accel: q4 mlp patches unavailable", exc_info=True)

    if wanted("fa256_steel_prefill_enabled"):
        try:
            from .qwen35_fa256_attention import apply_qwen35_fa256_attention_patch

            attempt("fa256_attention", apply_qwen35_fa256_attention_patch)
        except Exception:
            logger.debug("dflash accel: fa256 patch unavailable", exc_info=True)

    return results


def _enable_ane(model: Any, model_settings: Any) -> int:
    from .qwen35_ane_prefill import enable_qwen35_ane_prefill

    def s(name: str, default: Any) -> Any:
        return getattr(model_settings, name, default) if model_settings else default

    return enable_qwen35_ane_prefill(
        model,
        sequence_length=int(s("qwen35_ane_prefill_sequence_length", 2048)),
        fraction=float(s("qwen35_ane_prefill_fraction", 0.53)),
        max_layers=int(s("qwen35_ane_prefill_max_layers", 64)),
        gdn=bool(s("qwen35_ane_prefill_gdn", True)),
        gdn_fraction=float(s("qwen35_ane_prefill_gdn_fraction", 0.50)),
        gdn_max_layers=int(s("qwen35_ane_prefill_gdn_max_layers", 48)),
        dual_ane=bool(s("qwen35_ane_prefill_dual_ane", True)),
    )


def install_dflash_qwen35_class_patches(model_settings: Any = None) -> dict[str, Any]:
    """Install the class/module-level patches.

    Must run **before** ``load_target_bundle`` and before dflash installs its own
    class-level ``__call__`` hooks. Installing afterwards would let these
    wrappers capture a dflash hook, and ``restore_dflash_class_patches`` would
    then restore the pre-dflash function and silently drop the wrapper while the
    patch modules' module-level ``_PATCHED`` flags still report installed, so a
    later load would not reinstall it. Installing first keeps dflash's snapshot
    of "pre-dflash state" inclusive of these wrappers.
    """
    if not accel_enabled():
        return {"enabled": False}
    return {"enabled": True, "class_patches": _install_class_patches(model_settings)}


def enable_dflash_qwen35_ane(
    model: Any,
    model_settings: Any = None,
    *,
    prefill_step_size: int = 2048,
) -> dict[str, Any]:
    """Enable the ANE/GPU prefill split on a loaded DFlash target.

    Instance-level work, so unlike the class patches this must run after
    ``load_target_bundle`` and on the MLX executor thread.
    """
    if not accel_enabled():
        return {"enabled": False}

    summary: dict[str, Any] = {"enabled": True}

    ane_requested = bool(
        getattr(model_settings, "qwen35_ane_prefill_enabled", False)
        if model_settings
        else False
    )
    if ane_requested:
        sequence_length = int(
            getattr(model_settings, "qwen35_ane_prefill_sequence_length", 2048)
        )
        # dflash prefills in fixed prefill_step_size chunks; a chunk narrower
        # than the compiled shape cannot tile onto it, so the ANE programs would
        # compile and never run.
        if sequence_length > prefill_step_size:
            logger.warning(
                "Qwen ANE prefill sequence_length=%d exceeds the dflash prefill "
                "step (%d tokens); chunks narrower than the compiled shape cannot "
                "tile onto it. Set sequence_length<=%d.",
                sequence_length,
                prefill_step_size,
                prefill_step_size,
            )
        try:
            count = _enable_ane(model, model_settings)
        except Exception:
            count = 0
            logger.warning("Qwen ANE prefill not enabled for dflash", exc_info=True)
        summary["ane_mlp_layers"] = count
        summary["ane_gdn_layers"] = int(
            getattr(model, "_omlx_ane_gdn_prefill_count", 0) or 0
        )
        if count or summary["ane_gdn_layers"]:
            # The ANE prefix is per-output-channel INT8, so target prefill is
            # not bit-exact with the unpatched path; measured greedy output
            # diverges late in long generations. Acceptance is unaffected.
            logger.warning(
                "Qwen ANE prefill active on a DFlash target (%d MLP / %d GDN layers): "
                "prefill uses an approximate INT8 route, so greedy output can drift "
                "from the unpatched path in long generations. Set "
                "OMLX_DFLASH_QWEN35_ACCEL=0 to keep this load off the stack.",
                count,
                summary["ane_gdn_layers"],
            )
    else:
        summary["ane_mlp_layers"] = 0
        summary["ane_gdn_layers"] = 0

    logger.info(
        "DFlash Qwen accel: ane_mlp=%d ane_gdn=%d (step=%d)",
        summary["ane_mlp_layers"],
        summary["ane_gdn_layers"],
        prefill_step_size,
    )
    return summary
