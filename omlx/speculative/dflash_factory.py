# SPDX-License-Identifier: Apache-2.0
"""Factory for attaching a DFlash drafter to an already-loaded VLM target.

The vanilla ``dflash_mlx.runtime.bundle.load_runtime_bundle`` invokes
``mlx_lm.utils.load(target_ref)`` internally — when the omlx DFlashEngine
also runs an embedded ``VLMBatchedEngine`` (Path A double-engine layout),
that means the target weights are materialized twice and Gemma 4 model
memory doubles. This module skips the second load: it accepts a target
model that the embedded VLM has already loaded (wrapped through
``DFlashVLMTargetWrapper`` so dflash's mlx_lm-shaped TargetOps can see it)
and only loads the small drafter.

The mlx_vlm Gemma 4 model never matches ``Gemma4TargetOps.supports_model``
(its inner module exposes ``model.embed_tokens`` rather than
``inner.embed_tokens``); we bypass ``resolve_target_ops`` dispatch by
constructing ``Gemma4TargetOps`` directly. For Qwen GDN targets we still
go through ``resolve_target_ops`` since the wrapper-equivalent isn't
needed there — Path A's first cut is Gemma 4 only, but we keep the
factory generic so a future Qwen wrapper drops in.

Mirrored side-effects from ``load_target_bundle`` that the dflash decode
path relies on (Gemma 4 specifics):

  * ``install_speculative_hooks`` — no-op on Gemma 4, kept for symmetry.
  * ``configure_full_attention_split`` — no-op on Gemma 4.
  * ``install_verify_linears`` — applied to the real mlx_vlm model
    (``wrapped._vlm``) since it walks ``leaf_modules()`` and the wrapper
    is not an ``nn.Module``. ``runtime_context.verify`` decides whether
    this runs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AttachedDFlashBundle:
    """Subset of ``dflash_mlx.runtime.bundle.RuntimeBundle`` that omits the
    target model (already owned by the embedded VLM engine).

    Holding the wrapper here (``target_wrapper``) keeps it alive for the
    lifetime of the bundle; dropping the only reference would let the proxy
    views inside it get GC'd while dflash is mid-decode.
    """

    target_ops: Any
    draft_model: Any
    draft_backend: Any
    draft_meta: dict[str, Any]
    runtime_context: Any
    target_wrapper: Any
    resolved_draft_ref: str
    effective_draft_quant: str | None


def _construct_target_ops_for(wrapped_target: Any) -> Any:
    """Construct TargetOps without going through ``resolve_target_ops``.

    ``DFlashVLMTargetWrapper`` makes the mlx_vlm Gemma 4 model look
    mlx_lm-shaped enough for ``Gemma4TargetOps`` runtime methods, but the
    dispatch predicate ``supports_model`` still returns False because it
    probes ``inner.embed_tokens`` which lives at ``model.embed_tokens`` on
    the vlm model. We side-step dispatch and instantiate the family ops
    directly based on the wrapped model's model_type.
    """
    # Walk through proxy to the real config; the wrapper exposes args view.
    model_type = None
    lm = getattr(wrapped_target, "language_model", None)
    if lm is not None:
        args = getattr(lm, "args", None)
        if args is not None:
            model_type = getattr(args, "model_type", None)
    if model_type is None:
        cfg = getattr(wrapped_target, "config", None)
        if cfg is not None:
            model_type = getattr(cfg, "model_type", None)

    if isinstance(model_type, str) and "gemma4" in model_type.lower():
        from dflash_mlx.engine.target_gemma4 import Gemma4TargetOps
        return Gemma4TargetOps()

    # Fall back to dispatch for any other family (covers Qwen GDN once a
    # wrapper exists for it). Will raise NotImplementedError if not
    # compatible — caller catches and falls back.
    from dflash_mlx.engine.target_ops import resolve_target_ops
    return resolve_target_ops(wrapped_target)


def attach_dflash_to_loaded_target(
    target_model: Any,
    draft_path: str,
    draft_quant: str | None,
    runtime_context: Any,
) -> AttachedDFlashBundle:
    """Attach a DFlash drafter to a pre-loaded (and wrapped) target.

    Args:
        target_model: An mlx_vlm model already loaded by the embedded VLM
            engine, wrapped through ``DFlashVLMTargetWrapper`` so dflash's
            TargetOps see an mlx_lm-shaped surface.
        draft_path: HF ref / local path of the DFlash drafter checkpoint.
        draft_quant: Draft quantization spec string (e.g. ``"w4"``,
            ``"w8"``) or None for the model default.
        runtime_context: dflash runtime context (carries verify config,
            cache config, etc.).

    Returns:
        ``AttachedDFlashBundle`` with the drafter, backend, ops, and
        wrapper attached. The wrapper is retained so its proxy lifetime
        ties to the bundle's.
    """
    from dflash_mlx.draft_backend import make_draft_backend
    from dflash_mlx.engine.target_ops import bind_draft_to_target
    from dflash_mlx.runtime.loading import load_draft_bundle
    from dflash_mlx.runtime.registry import (
        resolve_effective_draft_quant,
        resolve_model_support_spec,
    )

    target_ops = _construct_target_ops_for(target_model)

    # Mirror load_target_bundle's optional install_verify_linears step.
    # Gemma 4 reports supports_verify_linear=True; install_speculative_hooks
    # and configure_full_attention_split are no-ops on Gemma 4 but we still
    # call them for symmetry (and so a future non-Gemma family works).
    capabilities = target_ops.capabilities_for(target_model)
    target_ops.install_speculative_hooks(target_model)
    # Gemma 4's configure_full_attention_split takes the wrapped model and
    # does nothing; pass enabled=False to match the safe default when we
    # don't know quantize_kv_cache config from the embedded engine.
    target_ops.configure_full_attention_split(
        target_model, enabled=False, chunk_size=8,
    )

    verify_cfg = getattr(runtime_context, "verify", None)
    verify_mode = getattr(verify_cfg, "mode", "auto") if verify_cfg else "auto"
    verify_enabled = (
        bool(capabilities.supports_verify_linear)
        and verify_mode != "off"
    )
    if verify_enabled:
        # install_verify_linears walks model.leaf_modules(); the proxy
        # wrapper is not an nn.Module so swap on the underlying mlx_vlm
        # model. The proxy continues to see swapped layers via __getattr__.
        real_model = getattr(target_model, "_vlm", target_model)
        from dflash_mlx.verify_linear import install_verify_linears
        enable_qmm = bool(getattr(verify_cfg, "enable_qmm", True)) if verify_cfg else True
        n_swapped = install_verify_linears(real_model, enable_qmm=enable_qmm)
        logger.info(
            f"DFlash factory: installed verify_linear on {n_swapped} "
            f"QuantizedLinear modules of {type(real_model).__name__}"
        )

    # Resolve drafter via registry when a non-empty path was given (the
    # registry returns None for unknown bases; explicit paths flow through
    # unchanged).
    resolved_draft_ref = draft_path
    support_spec = None
    try:
        support_spec = resolve_model_support_spec(draft_path)
    except Exception:
        support_spec = None
    effective_draft_quant = resolve_effective_draft_quant(
        draft_quant=draft_quant,
        resolved_draft_ref=resolved_draft_ref,
        support_spec=support_spec,
    )

    draft_model, draft_meta = load_draft_bundle(
        resolved_draft_ref,
        lazy=True,
        draft_quant=effective_draft_quant,
    )
    draft_meta = dict(draft_meta)
    draft_meta["draft_quant_spec"] = effective_draft_quant
    draft_meta["draft_quant_source"] = (
        "explicit"
        if (draft_quant or "").strip()
        and (draft_quant or "").strip().lower() != "none"
        else "model_default"
        if effective_draft_quant is not None
        else "none"
    )

    draft_backend = make_draft_backend()
    bind_draft_to_target(draft_model, target_model, target_ops=target_ops)

    return AttachedDFlashBundle(
        target_ops=target_ops,
        draft_model=draft_model,
        draft_backend=draft_backend,
        draft_meta=draft_meta,
        runtime_context=runtime_context,
        target_wrapper=target_model,
        resolved_draft_ref=str(resolved_draft_ref),
        effective_draft_quant=effective_draft_quant,
    )
