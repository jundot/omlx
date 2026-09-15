# SPDX-License-Identifier: Apache-2.0
"""Per-family hook registry for the unified expert-streaming path.

Everything the shared converter/runtime needs that legitimately differs
per model family lives behind ``hooks_for(model_type)``: which decoder
classes need the per-layer eval-boundary wrapper, whether an adaptive
top-k truncation hook exists (and the patch that installs it), the fused
weighted-sum kernel used on the sorted-decode path, and the structural
spelling of the family's MoE container (attribute chain, checkpoint key
prefixes, MTP-stage owner chain, verify scope). Adding a family is one
table row — not another hardcoded branch inside ``streaming_switch`` or
``__init__``.

What deliberately stays shared (NOT per-family state): the walks
themselves (``find_moe_container`` / ``find_mtp_stages`` are single
implementations parameterized by the registry fields), slot bookkeeping,
and the cache/governor/backing machinery — those are layout-generic, and
keying them per model would only restrict them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .residency import normalize_model_type

logger = logging.getLogger(__name__)


def _import(path: str) -> Any | None:
    try:
        import importlib

        return importlib.import_module(path)
    except Exception:
        return None


def _qwen35_language() -> Any | None:
    return _import("mlx_vlm.models.qwen3_5_moe.language")


def _qwen4_exp_language() -> Any | None:
    """qwen4_exp decoder module, applying the mlx_vlm compat patch if the
    installed tree has not registered the vendored model yet."""
    mod = _import("mlx_vlm.models.qwen4_exp.language")
    if mod is not None:
        return mod
    try:
        from omlx.patches.mlx_vlm_qwen4_exp_compat import (
            apply_mlx_vlm_qwen4_exp_compat_patch,
        )

        if apply_mlx_vlm_qwen4_exp_compat_patch():
            return _import("mlx_vlm.models.qwen4_exp.language")
    except Exception:  # noqa: BLE001
        logger.debug("qwen4_exp compat patch unavailable", exc_info=True)
    return None


def _apply_qwen_topk() -> bool:
    from .adaptive_topk import apply_qwen35_moe_topk_patch

    return apply_qwen35_moe_topk_patch()


def _glm_weighted_sum() -> Any | None:
    mod = _import("omlx.custom_kernels.glm_moe_dsa.fast")
    return getattr(mod, "glm_moe_weighted_sum", None) if mod is not None else None


def _v41_verify_scope() -> Any | None:
    """DeepSeek V4.1 verify-scope context manager, resolved lazily.

    ``deepseek_v41.moe_offload`` imports ``expert_streaming.slot_cache``
    at module level, so a top-level import here would be a cycle — the
    resolver runs at verify time, when everything is already loaded.
    """
    mod = _import("omlx.patches.deepseek_v41.moe_offload")
    return getattr(mod, "verify_scope", None) if mod is not None else None


# Checkpoint key prefix templates for stacked expert banks, covering the
# LLM (``model.layers``) and every observed VLM wrapper spelling
# (``model.language_model``, ``language_model.model``, ``language_model``).
# The mlp spellings come first: for the empty-weight-map fallback the
# first candidate is the key the streaming linear gets stamped with, and
# mlp-nested families are the common case. ffn-nested families reorder so
# their real spelling leads. Extra templates are harmless — each is one
# dict probe — so the shared default stays the full superset.
_MLP_PREFIX_TEMPLATES = (
    "model.layers.{i}.mlp.switch_mlp",
    "model.language_model.layers.{i}.mlp.switch_mlp",
    "language_model.model.layers.{i}.mlp.switch_mlp",
    "language_model.layers.{i}.mlp.switch_mlp",
)
_FFN_PREFIX_TEMPLATES = tuple(
    t.replace(".mlp.", ".ffn.") for t in _MLP_PREFIX_TEMPLATES
)
DEFAULT_PREFIX_TEMPLATES = _MLP_PREFIX_TEMPLATES + _FFN_PREFIX_TEMPLATES

# Container attribute chain for DeepSeek-style layouts (vendored v32/v4/
# v41, glm_moe_dsa): the MoE nests under ``ffn``. Ordering is a lookup
# preference only — ``find_moe_container`` requires ``switch_mlp``, so a
# chain never matches the wrong container.
_FFN_FIRST_CHAIN = ("ffn", "mlp")
_FFN_FIRST_TEMPLATES = _FFN_PREFIX_TEMPLATES + _MLP_PREFIX_TEMPLATES


@dataclass(frozen=True)
class ModelHooks:
    """Adaptation points one model family contributes to the shared path.

    ``stream_eval_targets``: ``(module_resolver, class_name)`` pairs —
    decoder-layer classes that ignore ``_stream_eval`` and need the
    per-layer eval-boundary wrapper. Families whose decoder honors the
    flag inline (``_INLINE_STREAM_EVAL_ATTR``) list none.

    ``topk_supported``: an adaptive top-k truncation hook exists for this
    family — either a patchable MoE block or an inline vendored hook. Gates
    the threshold knob's "applicable" verdict.

    ``apply_topk``: installs the family's truncation patch (None when the
    hook is inline in the vendored model — nothing to apply).

    ``weighted_sum_kernel``: resolver returning the fused weighted-sum
    callable for sorted decode (``(x_sorted, inv_order, scores) -> out``),
    or None for the plain scatter-unsort path.

    ``moe_attr_chain``: attribute names tried in order on a decoder layer
    or MTP stage to locate the MoE container (the module holding
    ``switch_mlp``); ``find_moe_container`` then repeats the chain one
    level under ``.block`` for nested layouts. Order is a preference, not
    a filter — only containers holding ``switch_mlp`` match.

    ``prefix_templates``: ``model.layers.{i}.…switch_mlp`` spellings the
    family's checkpoints can use, primary first (the first template is the
    empty-weight-map fallback key). None → ``DEFAULT_PREFIX_TEMPLATES``.

    ``mtp_owner_chain``: attributes descended from each candidate owner
    (the layers owner, then the model root) when hunting ``.mtp`` stages.
    """

    stream_eval_targets: tuple[tuple[Callable[[], Any | None], str], ...] = ()
    topk_supported: bool = False
    apply_topk: Callable[[], bool] | None = None
    weighted_sum_kernel: Callable[[], Any | None] | None = None
    moe_attr_chain: tuple[str, ...] = ("mlp", "ffn")
    prefix_templates: tuple[str, ...] | None = None
    mtp_owner_chain: tuple[str, ...] = ("language_model", "model")
    verify_scope: Callable[[], Any | None] | None = None


_QWEN_STREAM_EVAL = (
    (_qwen35_language, "Qwen3_5MoeDecoderLayer"),
    (_qwen4_exp_language, "Qwen4ExpDecoderLayer"),
)

_HOOKS: dict[str, ModelHooks] = {
    # Qwen families: the installed decoder classes ignore _stream_eval and
    # need the wrapper; top-k truncation patches the shared
    # Qwen3_5MoeSparseMoeBlock.
    "qwen4_exp": ModelHooks(
        stream_eval_targets=_QWEN_STREAM_EVAL,
        topk_supported=True,
        apply_topk=_apply_qwen_topk,
    ),
    "qwen4_exp_text": ModelHooks(
        stream_eval_targets=_QWEN_STREAM_EVAL,
        topk_supported=True,
        apply_topk=_apply_qwen_topk,
    ),
    # GLM families: the vendored decoder honors _stream_eval inline
    # (no wrapper targets); top-k hook lives in the vendored Glm5NextMoE;
    # both use the fused glm_moe_weighted_sum kernel on sorted decode.
    "glm5_next": ModelHooks(
        topk_supported=True,
        weighted_sum_kernel=lambda: _glm_weighted_sum(),
    ),
    "glm5_next_text": ModelHooks(
        topk_supported=True,
        weighted_sum_kernel=lambda: _glm_weighted_sum(),
    ),
    "glm_moe_dsa": ModelHooks(
        weighted_sum_kernel=lambda: _glm_weighted_sum(),
        moe_attr_chain=_FFN_FIRST_CHAIN,
        prefix_templates=_FFN_FIRST_TEMPLATES,
    ),
    # DeepSeek V3.2 (the glm_moe_dsa patch's vendored model) calls
    # switch_mlp with weighted_sum=True the same way.
    "deepseek_v32": ModelHooks(
        weighted_sum_kernel=lambda: _glm_weighted_sum(),
        moe_attr_chain=_FFN_FIRST_CHAIN,
        prefix_templates=_FFN_FIRST_TEMPLATES,
    ),
    # DeepSeek V4 (unified path) and its MTP wrapper type: ffn-nested MoE,
    # DSpark stages under model.mtp via the default owner chain.
    "deepseek_v4": ModelHooks(
        moe_attr_chain=_FFN_FIRST_CHAIN,
        prefix_templates=_FFN_FIRST_TEMPLATES,
    ),
    "deepseek_v4_mtp": ModelHooks(
        moe_attr_chain=_FFN_FIRST_CHAIN,
        prefix_templates=_FFN_FIRST_TEMPLATES,
    ),
    # DeepSeek V4.1 runs its own upstream offload path — not a unified
    # family — but its checkpoint layout shares the ffn-first chain, so the
    # row documents the family wiring in one place.
    "deepseek_v41": ModelHooks(
        moe_attr_chain=_FFN_FIRST_CHAIN,
        prefix_templates=_FFN_FIRST_TEMPLATES,
        verify_scope=_v41_verify_scope,
    ),
}

_DEFAULT = ModelHooks()


def hooks_for(model_type: object) -> ModelHooks:
    """Hooks for *model_type* (empty hooks for unlisted families)."""
    return _HOOKS.get(normalize_model_type(model_type), _DEFAULT)


def resolve_verify_scope(model_type: object) -> Any | None:
    """The family's draft-verify context manager factory, or None."""
    resolver = hooks_for(model_type).verify_scope
    if resolver is None:
        return None
    try:
        return resolver()
    except Exception:
        return None


def all_stream_eval_targets() -> list[tuple[str, Any]]:
    """Every registered (class_name, class) needing the eval wrapper.

    Aggregation is process-global on purpose: wrapping is idempotent and
    the wrapper fires only on instances carrying ``_stream_eval``, so one
    pass covers every family the process has converted.
    """
    found: list[tuple[str, Any]] = []
    seen: set[int] = set()
    for hooks in _HOOKS.values():
        for resolver, class_name in hooks.stream_eval_targets:
            mod = resolver()
            if mod is None:
                continue
            cls = getattr(mod, class_name, None)
            if cls is not None and id(cls) not in seen:
                seen.add(id(cls))
                found.append((class_name, cls))
    return found


def resolve_weighted_sum_kernel(model_type: object) -> Any | None:
    """The family's fused weighted-sum callable, or None."""
    resolver = hooks_for(model_type).weighted_sum_kernel
    if resolver is None:
        return None
    try:
        return resolver()
    except Exception:
        return None


def find_moe_container(
    node: Any,
    attr_chain: tuple[str, ...] = ("mlp", "ffn"),
    *,
    descend_block: bool = True,
) -> Any | None:
    """The MoE container on a decoder layer / MTP stage, or None.

    Tries each attribute of *attr_chain* in order and returns the first
    that exists AND holds a ``switch_mlp`` member — the "first match with
    a switch" semantics every caller used inline. When the direct chain
    misses and *descend_block* is set, the chain repeats one level under
    ``node.block`` (legacy MTPBlock layouts nest ``block.mlp``/``block.ffn``).
    """
    if node is None:
        return None
    for attr in attr_chain:
        moe = getattr(node, attr, None)
        if moe is not None and getattr(moe, "switch_mlp", None) is not None:
            return moe
    if descend_block:
        block = getattr(node, "block", None)
        if block is not None:
            for attr in attr_chain:
                moe = getattr(block, attr, None)
                if moe is not None and getattr(moe, "switch_mlp", None) is not None:
                    return moe
    return None


def find_moe_owner(node: Any, moe: Any | None) -> Any:
    """The object owning *moe* — *node* or its ``.block`` child.

    ``_convert_switch_mlp_module`` stamps ``compile_ffn``/``_stream_eval``
    on this owner (a nested ``block.mlp`` GLU needs the flag on the block,
    not the stage). Complements find_moe_container for the one caller that
    needs the owner identity (the MTP-stage walk).
    """
    if moe is not None:
        block = getattr(node, "block", None)
        if block is not None and any(
            getattr(block, attr, None) is moe for attr in ("mlp", "ffn")
        ):
            return block
    return node


def find_mtp_stages(
    roots: Any,
    owner_chain: tuple[str, ...] = ("language_model", "model"),
) -> Any | None:
    """The first non-empty ``.mtp`` stage list reachable from *roots*.

    MTP stages live next to the decoder stack, but not always on the same
    owner that holds it: glm5_next VLM resolves layers through the root
    ``Model.layers`` property while the draft hangs off
    ``language_model.mtp``. Each root (typically the layers owner and the
    model root) is tried along with its *owner_chain* children — dedup'd,
    in order — before giving up.
    """
    candidates: list[Any] = []
    for root in roots or ():
        if root is None:
            continue
        for cand in [root] + [getattr(root, a, None) for a in owner_chain]:
            if cand is not None and all(cand is not seen for seen in candidates):
                candidates.append(cand)
    for owner in candidates:
        mtp = getattr(owner, "mtp", None)
        if isinstance(mtp, (list, tuple)) and len(mtp):
            return mtp
    return None
