# SPDX-License-Identifier: Apache-2.0
"""Conditional MTP dispatch inside ``mlx_lm.generate.GenerationBatch``.

This is the integration point that lets the existing oMLX scheduler /
paged cache / prefix cache / SSD cache stack drive MTP without touching any
of those layers. ``GenerationBatch`` is mlx-lm's per-step decoder for the
active set of sequences in continuous batching. We patch:

- ``GenerationBatch.__init__`` — leave the standard mlx-lm initialization
  untouched. Fresh singleton donor batches may still be merged into a larger
  continuous batch, so MTP must not mutate cache state in ``__init__``.

- ``GenerationBatch.next`` — when the batch holds exactly one MTP-capable
  sequence, lazily initialize MTP from the standard post-prefill state. We
  emit from the per-batch queue first; once empty, we run a 2-token verify
  forward over ``[next_main, draft]`` with ``n_confirmed=1`` and a single
  MTP-head forward at the bonus position (accept) or confirmed position
  (reject), refilling the queue from the verify outputs.

- ``GenerationBatch.extend`` / ``filter`` — drop MTP state whenever continuous
  batching reshapes ownership. MTP state belongs to one uid in one singleton
  timeline; it must not survive standard batched decoding.

The throughput math (greedy, accept rate p):
  - Cost per *cycle*: 1× backbone (2-token verify) + 1× MTP head ≈ 1.15
  - Tokens per cycle: 1 + p (accept emits draft+bonus; reject emits verify_pred only)
  - At p≈1: 0.575 cost/token → ~1.74× throughput
  - At p≈0.5: ~0.77 cost/token → ~1.30× throughput

Known limitation (compute-bound single-stream Apple Silicon):
  The cost model above assumes the 2-token verify forward is nearly free
  relative to a 1-token forward, which is the bandwidth-bound decode regime
  speculative decoding targets. On lower-end single-stream Apple Silicon
  (e.g. M1/M2 base/Pro) decode is compute-bound, so the verify forward costs
  ~2× a 1-token forward and MTP can be net-negative regardless of accept
  rate. Wins are expected on M3/M4 or higher-end parts, on MoE models with a
  smaller per-step backbone, or under continuous batching where spare
  compute exists. See #1097 / #1311 for measurements.

Greedy identity (sampler is None): the patched dispatch produces the same
tokens as the standard step. PR 990's ``test_mtp_generate_identity``
encodes this contract; the oMLX-side equivalent lives in
``tests/test_mlx_lm_mtp_patch.py``.

Stochastic acceptance (sampler is not None): we use ``min(1, p_target / p_draft)``
(Leviathan & Chen 2023). On rejection we sample from the residual
``max(p_target - p_draft, 0) / Z`` so the marginal output distribution
equals the target distribution exactly.

PagedCacheManager interaction
-----------------------------
``cache.trim(1)`` on a ``BatchKVCache`` only updates ``self._idx``; the
underlying paged blocks are untouched. ``ArraysCache.rollback_state``
holds ``(conv_snap, ssm_snap)`` snapshots produced by the patched
``GatedDeltaNet.__call__`` and is restored on reject. Because both code
paths only mutate cache *length* (not block ownership), oMLX's
``PagedCacheManager`` is oblivious to the trim — its block_table is
unaffected and prefix-cache lookups continue to work normally.

TokenBuffer interaction
-----------------------
``GenerationBatch._token_context[0]`` is a ``TokenBuffer`` accumulating
the prompt + every forward-input token. We update it in lock-step with
each forward-input position so that ``logits_processors`` see the same
token sequence the standard step would see. On reject we shrink the
buffer's ``_size`` by 1 to discard the rejected draft (mirroring PR 990's
``prev_tokens = prev_tokens[:-1]``).
"""

from __future__ import annotations

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply() -> bool:
    """Wrap ``GenerationBatch`` and ``BatchGenerator`` MTP hooks.

    One-shot by design: the wraps capture ``original_*`` in closures so
    re-applying would chain wraps and double-init. ``GenerationBatch`` is
    not touched by dflash so the leftover-class-patch risk that motivates
    self-healing elsewhere doesn't apply here.
    """
    try:
        from mlx_lm.generate import BatchGenerator, GenerationBatch
    except ImportError:
        logger.debug("mlx_lm.generate GenerationBatch/BatchGenerator not importable")
        return False

    if not hasattr(GenerationBatch, "_omlx_mtp_patched"):
        original_init = GenerationBatch.__init__
        original_next = GenerationBatch.next
        original_filter = GenerationBatch.filter
        original_extend = GenerationBatch.extend

        def patched_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            # Do not activate MTP here. Fresh singleton batches created by
            # PromptProcessingBatch.generate() may still be merged into a larger
            # continuous batch; mutating their cache in __init__ can corrupt the
            # later standard batched path. Activation is lazy in patched_next().
            uids = getattr(self, "uids", None)
            if uids:
                reason = _ineligibility_reason(self)
                if reason:
                    logger.debug("MTP path not active: %s", reason)

        def patched_next(self, *args, **kwargs):
            if _is_mtp_batch_eligible(self):
                try:
                    batch_state = _prepare_mtp_batch_state_for_next(self)
                    if batch_state is not None:
                        return _mtp_batch_next(self, batch_state)
                except _MtpStepFallback as exc:
                    logger.debug("MTP batch next() fallback to standard step: %s", exc)
                    _reconcile_mtp_batch_to_standard(self)
                    _drop_mtp_batch_state(self, "batch-step-fallback")
            elif getattr(self, "_omlx_mtp_batch_state", None) is not None:
                _reconcile_mtp_batch_to_standard(self)
                _drop_mtp_batch_state(self, "batch-ineligible")

            if _is_mtp_eligible(self):
                try:
                    state = _prepare_mtp_state_for_next(self)
                    if state is not None:
                        return _mtp_next(self, state)
                except _MtpStepFallback as exc:
                    logger.debug("MTP next() fallback to standard step: %s", exc)
                    _drop_mtp_state(self, "step-fallback")
            else:
                _drop_mtp_state(self, "non-singleton-or-ineligible")
            _mark_standard_multirow_decode(self)
            return original_next(self, *args, **kwargs)

        def patched_extend(self, batch, *args, **kwargs):
            # The host (self) may have active MTP about to gain a co-runner.
            # The MTP path never maintains mlx-lm's _next_tokens, so a plain
            # drop here would leave standard batched decode resuming from a
            # stale _next_tokens against an MTP-advanced cache. Reconcile
            # before merge while ownership is still well defined.
            _reconcile_mtp_batch_to_standard(self)
            _drop_mtp_batch_state(self, "extend-reconciled")
            _drop_mtp_batch_state(batch, "donor-extended")

            host_state = getattr(self, "_omlx_mtp_state", None)
            if host_state is not None and _mtp_state_valid_for_batch(self, host_state):
                _reconcile_mtp_to_standard(self, host_state)
                _drop_mtp_state(self, "extend-reconciled")
            result = original_extend(self, batch, *args, **kwargs)
            _drop_mtp_state(batch, "donor-extended")
            _drop_invalid_mtp_state(self, "extend")
            _drop_invalid_mtp_batch_state(self, "extend")
            return result

        def patched_filter(self, keep, *args, **kwargs):
            old_uids = list(getattr(self, "uids", []) or [])
            result = original_filter(self, keep, *args, **kwargs)
            _drop_invalid_mtp_state(self, "filter", log_empty=True)
            _drop_invalid_mtp_batch_state(
                self,
                "filter",
                old_uids=old_uids,
                log_empty=True,
            )
            return result

        GenerationBatch.__init__ = patched_init
        GenerationBatch.next = patched_next
        GenerationBatch.filter = patched_filter
        GenerationBatch.extend = patched_extend
        GenerationBatch._omlx_mtp_patched = True

    if not hasattr(BatchGenerator, "_omlx_mtp_patched"):
        original_bg_next = BatchGenerator._next

        def patched_bg_next(self, *args, **kwargs):
            gen_batch = getattr(self, "_generation_batch", None)
            if gen_batch is not None:
                gen_batch._omlx_mtp_activation_safe = (
                    _batch_generator_allows_mtp_activation(self)
                )
            return original_bg_next(self, *args, **kwargs)

        BatchGenerator._next = patched_bg_next
        BatchGenerator._omlx_mtp_patched = True
    return True


def _model_has_mtp_module(model: Any) -> bool:
    """Check whether the model actually has an MTP head attached.

    The ``mtp_forward`` method is added to the class unconditionally by
    the patch, but the per-instance ``mtp`` module is only attached when
    ``mtp_enabled`` was True at load time (see qwen35_model._patch_model
    and deepseek_v4_model._patch_model). Without the inner module the
    ``mtp_forward`` call would AttributeError, so we gate eligibility on
    the actual module's presence.
    """
    inner = getattr(model, "language_model", model)
    return hasattr(inner, "mtp") and getattr(inner, "mtp", None) is not None


def _batch_generator_allows_mtp_activation(batch_gen: Any) -> bool:
    """True when lazy MTP activation cannot race with a pending batch merge."""
    try:
        return (
            len(getattr(batch_gen, "_unprocessed_sequences", [])) == 0
            and len(getattr(batch_gen, "_prompt_batch", [])) == 0
            and len(getattr(batch_gen, "_currently_processing", [])) == 0
        )
    except Exception:
        return False


def _mtp_common_eligible(gen_batch: Any) -> bool:
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    try:
        from . import is_mtp_active

        if not is_mtp_active():
            return False
    except Exception:
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) == 0:
        return False
    if _has_grammar_processors(gen_batch):
        return False
    return True


def _allows_new_mtp_activation(gen_batch: Any, state_attr: str) -> bool:
    if getattr(gen_batch, state_attr, None) is not None:
        return True
    if getattr(gen_batch, "_omlx_mtp_saw_standard_multirow_decode", False):
        return False
    return bool(getattr(gen_batch, "_omlx_mtp_activation_safe", True))


def _mark_standard_multirow_decode(gen_batch: Any) -> None:
    """Remember that this batch has decoded with shared standard cache state.

    A row that survives a standard multi-row decode and later becomes singleton
    no longer satisfies the narrow invariant that singleton MTP initialization
    relies on. Existing row-wise MTP state may continue, but starting a fresh
    singleton MTP state after late-join/late-finish reshaping is unsafe.
    """
    try:
        if len(getattr(gen_batch, "uids", []) or []) > 1:
            gen_batch._omlx_mtp_saw_standard_multirow_decode = True
    except Exception:
        pass


def _batch_rows_aligned_for_mtp(gen_batch: Any) -> bool:
    """True when all batch rows share the same target-cache decode position."""
    prompt_cache = getattr(gen_batch, "prompt_cache", None) or []
    for cache in prompt_cache:
        offset = getattr(cache, "offset", None)
        if offset is None:
            continue
        try:
            if hasattr(offset, "tolist"):
                values = [int(v) for v in offset.tolist()]
                return len(set(values)) <= 1
            return True
        except Exception:
            return False
    return True


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward``, has an attached MTP head, and
    the process-wide ``mtp_active`` flag is on.

    The MTP head may be attached unconditionally (e.g. by the mlx-vlm
    runtime patches, which need it for weight-load matching even when
    inference-time MTP is off) — so head presence alone is not enough
    to decide whether to run the draft/verify cycle. ``is_mtp_active``
    reflects the per-load ``model_settings.mtp_enabled`` choice.
    """
    if not _mtp_common_eligible(gen_batch):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return False
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_state"):
        return False
    return True


def _is_mtp_batch_eligible(gen_batch: Any) -> bool:
    if not _mtp_common_eligible(gen_batch):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) <= 1:
        return False
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_batch_state"):
        return False
    if (
        getattr(gen_batch, "_omlx_mtp_batch_state", None) is None
        and not _batch_rows_aligned_for_mtp(gen_batch)
    ):
        return False
    return True


def _ineligibility_reason(gen_batch: Any) -> str:
    """Return a short human-readable reason for why the MTP path isn't active.

    Only used for debug logging — the patched_init / patched_next paths
    don't act on this string.
    """
    if not hasattr(gen_batch, "model"):
        return "GenerationBatch has no .model attribute"
    if not hasattr(gen_batch.model, "mtp_forward"):
        return (
            f"model {type(gen_batch.model).__module__}.{type(gen_batch.model).__name__} "
            "has no mtp_forward (qwen35 patch may not have applied to this class)"
        )
    if not _model_has_mtp_module(gen_batch.model):
        return "model has no attached mtp head"
    try:
        from . import is_mtp_active

        if not is_mtp_active():
            return "mtp_active flag is off (model_settings.mtp_enabled was False at load time)"
    except Exception:
        return "is_mtp_active import failed"
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        if not _batch_rows_aligned_for_mtp(gen_batch):
            return "row-wise MTP requires aligned target-cache positions"
        return ""
    if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_state"):
        return "pending prompt work may still merge into this singleton batch"
    if _has_grammar_processors(gen_batch):
        return "grammar-constrained decoding uses GenerationBatch._step hooks"
    return ""


class _MtpStepFallback(RuntimeError):
    """Raised inside the MTP path to signal a clean fallback to the standard step."""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


@dataclass
class _MtpStats:
    """Acceptance / throughput counters for one MTP-active sequence.

    Logged at INFO when the sequence finishes (length / stop / filter)
    so the operator can see whether the draft+verify cycle is actually
    productive on this model + sampler combo.
    """

    cycles: int = 0  # number of verify cycles run
    accepts: int = 0  # cycles where the draft was accepted
    rejects: int = 0  # cycles where the draft was rejected
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Component-level timings. Help diagnose where MTP overhead comes from
    # when accept rate is healthy but wall-clock throughput isn't.
    backbone_ms: float = 0.0  # cumulative time inside the 2-token verify forward
    mtp_head_ms: float = 0.0  # cumulative time inside MTP-head forwards
    sample_ms: float = 0.0  # cumulative time in sampling + acceptance check
    cache_ops_ms: float = 0.0  # cumulative time in trim / rollback restore


@dataclass
class _MtpState:
    """Per-batch MTP state stashed on the GenerationBatch instance."""

    # MTP state is valid only for this exact singleton uid. It must be dropped
    # across any standard batched step or batch reshape that breaks ownership.
    uid: Any = None

    # Pending tokens to emit in upcoming next() calls. Each entry is
    # (token_id_int, logprobs_1d, source_label). source_label is one of
    # "init", "draft", "bonus", "verify" — used to bucket stats correctly
    # when the queue is drained.
    queue: Deque[Tuple[int, Any, str]] = field(default_factory=deque)

    # Cache for the MTP head (separate from gen_batch.prompt_cache).
    mtp_cache: Optional[List[Any]] = None

    # First input token of the next verify forward. Tracked as a 1-element
    # mx.array (uint32) so it can be concatenated with `draft_tok` cheaply.
    next_main: Optional[Any] = None

    # Draft logprobs (vocab,) needed by stochastic acceptance / residual sampling.
    draft_tok: Optional[Any] = None  # (1,) uint32
    draft_lp: Optional[Any] = None  # (vocab,) float
    # Filtered (sampler-applied) draft logprobs reused by the next cycle's
    # acceptance ratio + residual sampling. Mirrors PR 990's accept_lp,
    # adapted to oMLX's callable-sampler contract via metadata-introspection.
    # None when the sampler exposes no metadata (raw-lp fallback path).
    draft_accept_lp: Optional[Any] = None  # (vocab,) float
    # Host-side int copy of draft_tok. Cached at draft creation time so the
    # verify cycle can compare draft vs verify ids without a separate
    # GPU→CPU sync (`int(draft_tok.tolist()[0])` would force a stall).
    draft_id: int = -1

    # Accept-rate / throughput counters. Surfaced via logger.info on finish.
    stats: _MtpStats = field(default_factory=_MtpStats)


@dataclass
class _MtpBatchState:
    """Experimental row-wise MTP state for a multi-sequence GenerationBatch."""

    states: Dict[Any, _MtpState] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_sampler(gen_batch: Any):
    """Match ``GenerationBatch._step``'s per-sequence sampler resolution (batch=1)."""
    if gen_batch.samplers and gen_batch.samplers[0] is not None:
        return gen_batch.samplers[0]
    return gen_batch.fallback_sampler


def _is_greedy(gen_batch):
    sampler = _resolve_sampler(gen_batch)
    if sampler is not None:
        return getattr(sampler, "temp", 0.0) == 0.0
    return True


def _proc_list(gen_batch: Any) -> Optional[List[Any]]:
    if gen_batch.logits_processors and gen_batch.logits_processors[0]:
        return gen_batch.logits_processors[0]
    return None


def _has_grammar_processors(gen_batch: Any) -> bool:
    """True when MTP would bypass grammar state advanced by scheduler._step."""
    processors_by_seq = getattr(gen_batch, "logits_processors", None)
    if not processors_by_seq:
        return False
    try:
        from omlx.api.grammar import GrammarConstraintProcessor
    except Exception:
        return False
    return any(
        isinstance(proc, GrammarConstraintProcessor)
        for processors in processors_by_seq
        for proc in (processors or [])
    )


def _mtp_state_valid_for_batch(gen_batch: Any, state: Optional[_MtpState]) -> bool:
    """MTP state may only represent one uid in one current singleton slot."""
    if state is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    return bool(uids is not None and len(uids) == 1 and uids[0] == state.uid)


def _drop_mtp_state(
    gen_batch: Any,
    reason: str,
    *,
    log_stats: bool = False,
) -> Optional[_MtpState]:
    """Delete attached MTP state, optionally surfacing stats for external finish."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None:
        return None
    if log_stats:
        try:
            _log_mtp_stats(
                getattr(state, "uid", "?"),
                state.stats,
                getattr(state, "_finish_reason", reason),
            )
        except Exception:
            pass
    try:
        delattr(gen_batch, "_omlx_mtp_state")
    except AttributeError:
        pass
    logger.debug("MTP state dropped: %s", reason)
    return state


def _drop_invalid_mtp_state(
    gen_batch: Any,
    reason: str,
    *,
    log_empty: bool = False,
) -> Optional[_MtpState]:
    """Drop state after a batch reshape unless ownership still matches."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if state is None:
        return None
    if _mtp_state_valid_for_batch(gen_batch, state):
        return state
    uids = getattr(gen_batch, "uids", None)
    return _drop_mtp_state(
        gen_batch,
        reason,
        log_stats=bool(log_empty and not uids),
    )


def _mtp_batch_state_valid_for_batch(
    gen_batch: Any, batch_state: Optional[_MtpBatchState]
) -> bool:
    if batch_state is None:
        return False
    uids = getattr(gen_batch, "uids", None)
    if not uids:
        return False
    return all(uid in batch_state.states for uid in uids)


def _drop_mtp_batch_state(
    gen_batch: Any,
    reason: str,
    *,
    log_stats: bool = False,
) -> Optional[_MtpBatchState]:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return None
    if log_stats:
        for state in list(batch_state.states.values()):
            try:
                _log_mtp_stats(
                    getattr(state, "uid", "?"),
                    state.stats,
                    getattr(state, "_finish_reason", reason),
                )
            except Exception:
                pass
    try:
        delattr(gen_batch, "_omlx_mtp_batch_state")
    except AttributeError:
        pass
    logger.debug("MTP batch state dropped: %s", reason)
    return batch_state


def _drop_invalid_mtp_batch_state(
    gen_batch: Any,
    reason: str,
    *,
    old_uids: Optional[List[Any]] = None,
    log_empty: bool = False,
) -> Optional[_MtpBatchState]:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return None
    uids = list(getattr(gen_batch, "uids", []) or [])
    if not uids:
        return _drop_mtp_batch_state(
            gen_batch,
            reason,
            log_stats=bool(log_empty),
        )

    keep = set(uids)
    removed = set(old_uids or []) - keep
    for uid in removed:
        state = batch_state.states.pop(uid, None)
        if state is not None and log_empty:
            try:
                _log_mtp_stats(uid, state.stats, reason)
            except Exception:
                pass
    batch_state.states = {
        uid: state for uid, state in batch_state.states.items() if uid in keep
    }
    if _mtp_batch_state_valid_for_batch(gen_batch, batch_state):
        if len(uids) == 1:
            gen_batch._omlx_mtp_state = batch_state.states[uids[0]]
            _drop_mtp_batch_state(gen_batch, "filter-to-singleton")
            return None
        return batch_state
    return _drop_mtp_batch_state(gen_batch, reason)


def _row_value(values: Optional[List[Any]], idx: int, default: Any = None) -> Any:
    if values is None:
        return default
    try:
        if len(values) == 0:
            return default
        return values[idx]
    except Exception:
        return default


def _make_row_batch(
    gen_batch: Any,
    idx: int,
    *,
    prompt_cache: Optional[List[Any]] = None,
    state: Optional[_MtpState] = None,
) -> Any:
    if prompt_cache is None:
        prompt_cache = gen_batch.extract_cache(idx)

    next_tokens = getattr(gen_batch, "_next_tokens", None)
    next_logprobs = getattr(gen_batch, "_next_logprobs", None)
    row = SimpleNamespace(
        model=gen_batch.model,
        uids=[gen_batch.uids[idx]],
        prompt_cache=prompt_cache,
        tokens=[gen_batch.tokens[idx]],
        samplers=[_row_value(getattr(gen_batch, "samplers", None), idx)],
        fallback_sampler=gen_batch.fallback_sampler,
        logits_processors=[
            _row_value(getattr(gen_batch, "logits_processors", None), idx, [])
        ],
        state_machines=[_row_value(getattr(gen_batch, "state_machines", None), idx)],
        max_tokens=[_row_value(getattr(gen_batch, "max_tokens", None), idx)],
        _next_tokens=next_tokens[idx : idx + 1] if next_tokens is not None else None,
        _next_logprobs=(
            [next_logprobs[idx]]
            if next_logprobs is not None and len(next_logprobs) > idx
            else []
        ),
        _token_context=[gen_batch._token_context[idx]],
        _num_tokens=[gen_batch._num_tokens[idx]],
        _matcher_states=[gen_batch._matcher_states[idx]],
    )
    if state is not None:
        row._omlx_mtp_state = state
    return row


def _merge_row_caches(row_caches: List[List[Any]]) -> List[Any]:
    if not row_caches:
        return []
    merged = []
    for layer_idx in range(len(row_caches[0])):
        per_row = [cache[layer_idx] for cache in row_caches]
        merge = getattr(per_row[0], "merge", None)
        if not callable(merge):
            raise _MtpStepFallback(
                f"cache {type(per_row[0]).__name__} cannot merge row caches"
            )
        merged.append(merge(per_row))
    return merged


def _replace_cache_rows(
    gen_batch: Any,
    replacements: Dict[int, List[Any]],
) -> None:
    if not replacements:
        return
    row_caches = [
        replacements.get(idx) or gen_batch.extract_cache(idx)
        for idx in range(len(gen_batch.uids))
    ]
    gen_batch.prompt_cache = _merge_row_caches(row_caches)


def _prepare_mtp_batch_state_for_next(gen_batch: Any) -> Optional[_MtpBatchState]:
    """Return a valid row-wise MTP state, lazily initializing every row."""
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if _mtp_batch_state_valid_for_batch(gen_batch, batch_state):
        return batch_state
    if batch_state is not None:
        _drop_mtp_batch_state(gen_batch, "stale-batch-owner")

    replacements: Dict[int, List[Any]] = {}
    token_context_updates: Dict[int, Any] = {}
    states: Dict[Any, _MtpState] = {}

    for idx, uid in enumerate(gen_batch.uids):
        row = _make_row_batch(gen_batch, idx)
        _set_singleton_mrope_delta(row)
        _post_init_mtp(row)
        state = getattr(row, "_omlx_mtp_state", None)
        if not _mtp_state_valid_for_batch(row, state):
            _drop_mtp_batch_state(gen_batch, "batch-post-init-invalid")
            return None
        states[uid] = state
        replacements[idx] = row.prompt_cache
        token_context_updates[idx] = row._token_context[0]

    _replace_cache_rows(gen_batch, replacements)
    for idx, token_context in token_context_updates.items():
        gen_batch._token_context[idx] = token_context

    batch_state = _MtpBatchState(states=states)
    gen_batch._omlx_mtp_batch_state = batch_state
    logger.info(
        "MTP row-wise batch path activated for %d sequences",
        len(gen_batch.uids),
    )
    return batch_state


def _reconcile_mtp_batch_to_standard(gen_batch: Any) -> bool:
    batch_state = getattr(gen_batch, "_omlx_mtp_batch_state", None)
    if batch_state is None:
        return True
    if not getattr(gen_batch, "uids", None):
        return True

    import mlx.core as mx

    row_caches: Dict[int, List[Any]] = {}
    next_tokens = []
    next_logprobs = []
    token_context_updates: Dict[int, Any] = {}

    try:
        for idx, uid in enumerate(gen_batch.uids):
            state = batch_state.states.get(uid)
            if state is None:
                row_caches[idx] = gen_batch.extract_cache(idx)
                if getattr(gen_batch, "_next_tokens", None) is not None:
                    next_tokens.append(gen_batch._next_tokens[idx : idx + 1])
                if len(getattr(gen_batch, "_next_logprobs", [])) > idx:
                    next_logprobs.append(gen_batch._next_logprobs[idx])
                continue

            row = _make_row_batch(gen_batch, idx, state=state)
            if not _reconcile_mtp_to_standard(row, state):
                return False
            row_caches[idx] = row.prompt_cache
            next_tokens.append(row._next_tokens)
            next_logprobs.extend(row._next_logprobs)
            token_context_updates[idx] = row._token_context[0]

        if row_caches:
            _replace_cache_rows(gen_batch, row_caches)
        if next_tokens:
            gen_batch._next_tokens = mx.concatenate(next_tokens)
            gen_batch._next_logprobs = next_logprobs
        for idx, token_context in token_context_updates.items():
            gen_batch._token_context[idx] = token_context
        return True
    except Exception as exc:
        logger.warning("MTP batch reconcile failed: %s", exc)
        return False


def _prepare_mtp_state_for_next(gen_batch: Any) -> Optional[_MtpState]:
    """Return a valid singleton MTP state, lazily initializing if needed."""
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if _mtp_state_valid_for_batch(gen_batch, state):
        return state
    if state is not None:
        _drop_mtp_state(gen_batch, "stale-owner")

    _set_singleton_mrope_delta(gen_batch)
    _post_init_mtp(gen_batch)
    state = getattr(gen_batch, "_omlx_mtp_state", None)
    if not _mtp_state_valid_for_batch(gen_batch, state):
        _drop_mtp_state(gen_batch, "post-init-invalid")
        return None

    logger.info(
        "MTP path activated for uid=%s (model has mtp_forward, batch=1)",
        state.uid,
    )
    return state


def _set_singleton_mrope_delta(gen_batch: Any) -> None:
    """Mirror scheduler._step's per-uid mRoPE setup for direct MTP forwards."""
    model = getattr(gen_batch, "model", None)
    uids = getattr(gen_batch, "uids", None)
    if (
        model is not None
        and getattr(model, "_uses_mrope", False)
        and getattr(model, "_uid_rope_deltas", None)
        and uids
        and len(uids) == 1
        and hasattr(model, "set_batch_rope_deltas")
    ):
        import mlx.core as mx

        delta = model._uid_rope_deltas.get(uids[0], 0.0)
        model.set_batch_rope_deltas(mx.array([delta]))


def _rebuild_singleton_cache(model: Any) -> Optional[List[Any]]:
    """Build a fresh single-sequence batch-aware cache (left_padding=[0]).

    Reuses mlx-lm's own ``_make_cache`` so the per-layer types match exactly
    what ``extend()`` / ``_extend_cache`` expects, keeping the subsequent merge
    type-compatible. Returns None if the converter is unavailable.
    """
    import sys

    try:
        make_cache = sys.modules["mlx_lm.generate"]._make_cache
        return make_cache(model, [0], None)
    except Exception as exc:
        logger.warning("MTP reconcile: cache rebuild unavailable: %s", exc)
        return None


def _reconcile_mtp_to_standard(gen_batch: Any, state: _MtpState) -> bool:
    """Rewind a to-be-dropped MTP singleton into a standard-resumable state.

    The MTP path never maintains mlx-lm's ``_next_tokens`` — it streams tokens
    from ``state.queue`` and advances the shared cache speculatively, and the
    GatedDeltaNet rollback snapshot is cleared on accept, so a partial rollback
    at an arbitrary drop point is not reliable. Instead, rebuild the cache by
    re-prefilling exactly the already-streamed tokens (``gen_batch.tokens[0]``)
    into a fresh cache (which deterministically reconstructs every layer state,
    KV and SSM), then set ``_next_tokens`` to the correct next-to-emit token:

    - if ``state.queue`` is non-empty, ``queue[0]`` is the correct, not-yet-
      streamed next token — reuse it (and its logprobs). The rest of the queue
      is discarded; standard decode re-derives those positions.
    - otherwise (cycle boundary) sample from the re-prefill's last-position
      logits, exactly as a standard ``_step`` would after feeding ``tokens[-1]``.

    Leaves ``tokens[0]`` / ``_num_tokens[0]`` untouched (they already reflect
    streamed tokens), so there is no duplicated or skipped token. Returns False
    (caller falls back to a plain drop) when reconcile cannot be done safely.
    """
    import mlx.core as mx

    tokens = gen_batch.tokens[0] if getattr(gen_batch, "tokens", None) else None
    if not tokens:
        return False
    try:
        new_cache = _rebuild_singleton_cache(gen_batch.model)
        if new_cache is None:
            return False
        procs = _proc_list(gen_batch)
        _set_singleton_mrope_delta(gen_batch)
        tok_arr = _ensure_uint32(mx.array(list(tokens)))
        # Inherits the per-engine stream from the enclosing BatchGenerator context.
        logits, _, _ = _call_backbone(gen_batch.model, tok_arr[None, :], new_cache)
        last_logits = logits[:, -1, :]  # (1, vocab) — dist after tokens[-1]

        if state.queue:
            next_id, next_lp_1d, _src = state.queue[0]
            next_tok = mx.array([int(next_id)], dtype=mx.uint32)
            next_lp = next_lp_1d
        else:
            prev_buf = gen_batch._token_context[0].tokens if procs is not None else None
            ll = _apply_processors(procs, prev_buf, last_logits)
            next_lp_2d = _logprobs(ll)
            next_tok = _ensure_uint32(_resolve_sampler(gen_batch)(next_lp_2d))
            next_lp = next_lp_2d.squeeze(0)

        mx.eval(next_tok)
        gen_batch.prompt_cache = new_cache
        gen_batch._next_tokens = next_tok
        gen_batch._next_logprobs = [next_lp]
        if procs is not None:
            from mlx_lm.models.cache import TokenBuffer

            gen_batch._token_context[0] = TokenBuffer(list(tokens))
        logger.debug(
            "MTP reconciled to standard on reshape (uid=%s tokens=%d queue=%d)",
            getattr(state, "uid", "?"),
            len(tokens),
            len(state.queue),
        )
        return True
    except Exception as exc:
        logger.warning("MTP reconcile failed, falling back to plain drop: %s", exc)
        return False


def _apply_processors(processors, prev_tokens, logits_2d):
    if not processors:
        return logits_2d
    for proc in processors:
        logits_2d = proc(prev_tokens, logits_2d)
    return logits_2d


def _logprobs(logits_2d):
    import mlx.core as mx

    return logits_2d - mx.logsumexp(logits_2d, axis=-1, keepdims=True)


def _accept_lp_for(sampler, lp):
    """Reproduce the sampler's filter+temperature pipeline on `lp` so the
    acceptance ratio (and residual distribution) match the distribution the
    sampler actually drew from.

    Reads sampling params off the callable as function attributes (set by
    ``omlx.utils.sampling.make_sampler``). For samplers without metadata —
    e.g. mlx-lm stock callables, fallback samplers — returns `lp` unchanged
    so behavior matches the pre-PR-990 raw-lp acceptance.
    """
    import mlx.core as mx

    from omlx.utils.sampling import apply_min_p, apply_top_k, apply_top_p

    temp = float(getattr(sampler, "temp", 0.0) or 0.0)
    if temp == 0.0:
        # Greedy / unknown sampler — raw lp is the acceptance distribution.
        return lp

    out = lp
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    if 0.0 < top_p < 1.0:
        out = apply_top_p(out, top_p)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    if min_p != 0.0:
        min_keep = int(getattr(sampler, "min_tokens_to_keep", 1) or 1)
        out = apply_min_p(out, min_p, min_keep)
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    if top_k > 0:
        out = apply_top_k(out, top_k)

    # Temperature scale + renormalize so the output is a proper logprob
    # distribution that can be indexed by token id for the acceptance check.
    scaled = out * (1.0 / temp)
    return scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


def _trim_token_buffer(gen_batch: Any, n: int) -> None:
    """Shrink ``_token_context[0]`` by ``n`` (mirrors PR 990 ``prev[:-n]``)."""
    if n <= 0:
        return
    procs = _proc_list(gen_batch)
    if procs is None:
        return
    buf = gen_batch._token_context[0]
    buf._size = max(0, buf._size - n)


def _restore_or_trim_caches(prompt_cache: List[Any]) -> bool:
    """Roll back one token from each layer cache after a draft rejection.

    SSM / linear-attention layers expose ``rollback_state`` populated by the
    patched ``GatedDeltaNet.__call__``; we restore that snapshot. Standard
    KV cache layers (full-attention) expose ``trim`` and ``is_trimmable``;
    we trim by 1. Layers that support neither cause the entire MTP step to
    fall back to the standard path.
    """
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            c.trim(1)
            continue
        return False
    return True


def _rollback_after_reject(
    model: Any,
    prompt_cache: List[Any],
    gdn_states: Optional[list],
    accepted: int = 0,
    block_size: int = 2,
) -> bool:
    """Roll back per-layer cache state after a rejected MTP draft token.

    Two mechanisms are supported, dispatched on the model's capability:

    1. **mlx-vlm path** — when the model exposes ``rollback_speculative_cache``
       (Qwen3.5 LanguageModel ships with it upstream) AND ``gdn_states`` is
       populated, we delegate to that method. It batches the per-layer SSM
       replay into a single ``gated_delta_update`` call and trims KV
       caches by ``block_size - (accepted + 1)``. The backbone forward was
       run with both confirmed and draft tokens; the rollback replays only
       the accepted prefix through the original pre-update state.

    2. **mlx-lm path** (PR 990) — per-layer ``cache.rollback_state`` snapshot
       written by the patched ``GatedDeltaNet.__call__`` during the
       confirmed/draft split. We restore the snapshot for SSM layers and
       trim KV layers by 1. ``gdn_states`` is None in this path.

    Returns True on success. False means a cache layer in the list supports
    neither mechanism, in which case the caller falls back to the standard
    non-MTP step.
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        model.rollback_speculative_cache(prompt_cache, gdn_states, accepted, block_size)
        return True
    return _restore_or_trim_caches(prompt_cache)


def _call_backbone(
    model: Any,
    inputs: Any,
    cache: List[Any],
    n_confirmed: int = 0,
) -> Tuple[Any, Any, Optional[list]]:
    """Run the backbone with ``return_hidden=True`` and normalise the result.

    Returns ``(logits, hidden_pre_norm, gdn_states_or_None)``:

    - mlx-lm path returns the 2-tuple ``(logits, hidden)``; ``gdn_states``
      is ``None`` and rollback uses ``cache.rollback_state``.
    - mlx-vlm path returns the 3-tuple ``(logits, hidden, gdn_states)`` so
      a rejected draft can be rolled back via
      ``rollback_speculative_cache``.

    ``n_confirmed`` is forwarded so the mlx-lm path can split its
    GatedDeltaNet forward into confirmed and draft chunks. mlx-vlm
    discards it (irrelevant — rollback is post-hoc, not splitwise).
    """
    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    result = model(inputs, **kwargs)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    raise TypeError(f"backbone returned unexpected shape: {type(result).__name__}")


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop ``rollback_state`` snapshots after a draft is accepted."""
    for c in prompt_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


# ---------------------------------------------------------------------------
# Post-init: run one extra backbone forward + MTP forward; queue the two
# emitted tokens; stash a draft for the first verify cycle.
# ---------------------------------------------------------------------------


def _post_init_mtp(gen_batch: Any) -> None:
    """Bridge from standard ``__init__``'s ``_step()`` into PR 990's cycle 1.

    State on entry (after standard ``__init__``):
      - cache contains the prompt up to ``prompt[-1]`` inclusive
      - ``_next_tokens`` = ``main_tok`` (token sampled from ``prompt[-1]``'s logits)
      - ``_next_logprobs[0]`` = main_tok's distribution
      - ``tokens[0]`` = original prompt list

    We perform one more 1-token backbone forward (so the cache also includes
    ``main_tok`` and we obtain the hidden state at that position), run the
    MTP head to produce a draft for the next verify cycle, and seed
    ``state.queue`` with two confirmed tokens — ``main_tok`` and the
    standard-sample at the next position. After this, the queue handles
    the first two emit calls and the third call enters the verify cycle.

    If the batch was empty when ``__init__`` ran, ``_next_tokens`` is
    ``None`` — we leave MTP inactive and the standard path runs unchanged.
    """
    import mlx.core as mx

    if gen_batch._next_tokens is None or not gen_batch.uids:
        # Nothing was sampled in the standard _step (empty batch). The
        # next() call will be a no-op anyway; leave the patch inert.
        return

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    main_tok = _ensure_uint32(gen_batch._next_tokens)  # (1,)
    main_lp = gen_batch._next_logprobs[0]  # (vocab,)

    if procs is not None:
        prev_buf = gen_batch._token_context[0].update_and_fetch(main_tok)
    else:
        prev_buf = None

    # 1-token backbone forward at main_tok with hidden state. No draft yet,
    # so no rollback is possible — discard gdn_states.
    # Inherits the per-engine stream from the enclosing BatchGenerator context.
    logits, hidden, _ = _call_backbone(
        gen_batch.model, main_tok[:, None], gen_batch.prompt_cache
    )

    next_main_logits = logits[:, -1, :]  # (1, vocab) — distribution after main_tok
    next_main_logits = _apply_processors(procs, prev_buf, next_main_logits)
    next_main_lp = _logprobs(next_main_logits)
    next_main_tok = sampler(next_main_lp)  # (1,)

    # MTP head sees (hidden_at_main, next_main_tok) and proposes the draft
    # that the *next* verify cycle will check against forward([next_main, draft]).
    mtp_cache = gen_batch.model.make_mtp_cache()
    hidden_at_main = hidden[:, -1:, :]  # (1, 1, H)
    next_ids = next_main_tok.reshape(1, 1)
    mtp_logits = gen_batch.model.mtp_forward(hidden_at_main, next_ids, mtp_cache)
    mtp_logits_2d = mtp_logits[:, -1, :]
    if procs is not None:
        prev_with_main_and_next = mx.concatenate(
            [prev_buf, _ensure_uint32(next_main_tok)]
        )
        mtp_logits_2d = _apply_processors(procs, prev_with_main_and_next, mtp_logits_2d)
    draft_lp_2d = _logprobs(mtp_logits_2d)
    draft_tok = sampler(draft_lp_2d)
    # Filtered draft lp — what the sampler actually drew from. The next
    # cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than the raw softmax.
    draft_accept_lp_2d = _accept_lp_for(sampler, draft_lp_2d)

    mx.eval(main_tok, next_main_tok, draft_tok)

    # Queue the two confirmed tokens (main_tok + next_main_tok); their
    # logprobs come from the standard / patched samplers. Cache draft_id
    # while the array is already evaluated to avoid re-syncing in cycle 1.
    state = _MtpState(uid=gen_batch.uids[0])
    state.mtp_cache = mtp_cache
    state.next_main = _ensure_uint32(next_main_tok)
    state.draft_tok = _ensure_uint32(draft_tok)
    state.draft_lp = draft_lp_2d.squeeze(0)
    state.draft_accept_lp = draft_accept_lp_2d.squeeze(0)
    state.draft_id = int(draft_tok.tolist()[0])
    state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
    state.queue.append(
        (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
    )

    gen_batch._omlx_mtp_state = state


# ---------------------------------------------------------------------------
# next() dispatch
# ---------------------------------------------------------------------------


def _mtp_batch_next(gen_batch: Any, batch_state: _MtpBatchState) -> Any:
    """Emit one token per row using independent MTP state per active uid.

    This is intentionally conservative: rows whose queues are empty are
    advanced through the proven singleton MTP cycle against extracted row
    caches, then the modified rows are merged back into the batched cache.
    That keeps continuous-batching ownership correct while enabling MTP in
    multi-request decode without sharing singleton state across rows.
    """
    if not getattr(gen_batch, "uids", None):
        return []

    replacements: Dict[int, List[Any]] = {}
    token_context_updates: Dict[int, Any] = {}

    for idx, uid in enumerate(list(gen_batch.uids)):
        state = batch_state.states.get(uid)
        if state is None:
            raise _MtpStepFallback(f"missing row state for uid={uid}")
        if state.queue:
            continue

        row = _make_row_batch(
            gen_batch,
            idx,
            prompt_cache=gen_batch.extract_cache(idx),
            state=state,
        )
        _set_singleton_mrope_delta(row)
        _run_verify_cycle(row, state)
        if not state.queue:
            raise _MtpStepFallback(f"row uid={uid} verify produced no tokens")
        replacements[idx] = row.prompt_cache
        token_context_updates[idx] = row._token_context[0]

    _replace_cache_rows(gen_batch, replacements)
    for idx, token_context in token_context_updates.items():
        gen_batch._token_context[idx] = token_context

    return _emit_batch_responses(gen_batch, batch_state)


def _emit_batch_responses(gen_batch: Any, batch_state: _MtpBatchState) -> List[Any]:
    Response = type(gen_batch).Response

    keep = []
    responses = []
    finished_uids = []

    for idx, uid in enumerate(list(gen_batch.uids)):
        state = batch_state.states.get(uid)
        if state is None or not state.queue:
            raise _MtpStepFallback(f"row uid={uid} has no queued token")

        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)

        finish_reason: Optional[str] = None
        match_sequence = None

        gen_batch.tokens[idx].append(token_id)
        gen_batch._num_tokens[idx] += 1
        if gen_batch._num_tokens[idx] >= gen_batch.max_tokens[idx]:
            finish_reason = "length"

        new_state, match_sequence, current_state = gen_batch.state_machines[idx].match(
            gen_batch._matcher_states[idx],
            token_id,
        )
        gen_batch._matcher_states[idx] = new_state
        if match_sequence is not None and current_state is None:
            finish_reason = "stop"

        if finish_reason is not None:
            responses.append(
                Response(
                    uid=uid,
                    token=token_id,
                    logprobs=logprobs_1d,
                    finish_reason=finish_reason,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=gen_batch.extract_cache(idx),
                    all_tokens=gen_batch.tokens[idx],
                )
            )
            _log_mtp_stats(uid, state.stats, finish_reason)
            finished_uids.append(uid)
        else:
            keep.append(idx)
            responses.append(
                Response(
                    uid=uid,
                    token=token_id,
                    logprobs=logprobs_1d,
                    finish_reason=None,
                    current_state=current_state,
                    match_sequence=match_sequence,
                    prompt_cache=None,
                    all_tokens=None,
                )
            )

    for uid in finished_uids:
        batch_state.states.pop(uid, None)

    if len(keep) < len(gen_batch.uids):
        gen_batch.filter(keep)

    return responses


def _mtp_next(gen_batch: Any, state: _MtpState) -> Any:
    """Emit one token; run a verify cycle if the queue is empty."""
    if state.queue:
        token_id, logprobs_1d, source = state.queue.popleft()
        _bump_emit_stat(state, source)
        return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)

    _run_verify_cycle(gen_batch, state)
    if not state.queue:
        # Verify cycle should always populate the queue with at least the
        # rejected-verify token; if it didn't, fall back to the standard
        # step rather than yield an undefined response.
        raise _MtpStepFallback("verify cycle produced no emit tokens")

    token_id, logprobs_1d, source = state.queue.popleft()
    _bump_emit_stat(state, source)
    return _emit_response(gen_batch, token_id, logprobs_1d, state.stats)


def _log_mtp_stats(uid: Any, stats: "_MtpStats", finish_reason: str) -> None:
    """Emit a one-line summary of MTP draft/verify activity for a finished sequence.

    Format chosen to match PR 990's headline metrics, plus component timings
    that make wall-clock vs. accept-rate gaps debuggable:
      MTP[<uid>] finish=<reason> tokens=<N> cycles=<C> accept=<A>/<C> (<rate>%)
        emits[init=<i>,draft=<d>,bonus=<b>,verify=<v>]
        timing[backbone=<X>ms mtp=<Y>ms sample=<S>ms cache=<C>ms]
    """
    total_emits = (
        stats.init_emits + stats.draft_emits + stats.bonus_emits + stats.verify_emits
    )
    if stats.cycles > 0:
        rate_str = f"{stats.accepts / stats.cycles * 100:.1f}%"
    else:
        rate_str = "n/a"
    logger.info(
        "MTP[%s] finish=%s tokens=%d cycles=%d accept=%d/%d (%s) "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        total_emits,
        stats.cycles,
        stats.accepts,
        stats.cycles,
        rate_str,
        stats.init_emits,
        stats.draft_emits,
        stats.bonus_emits,
        stats.verify_emits,
        stats.backbone_ms,
        stats.mtp_head_ms,
        stats.sample_ms,
        stats.cache_ops_ms,
    )


def _bump_emit_stat(state: _MtpState, source: str) -> None:
    if source == "init":
        state.stats.init_emits += 1
    elif source == "draft":
        state.stats.draft_emits += 1
    elif source == "bonus":
        state.stats.bonus_emits += 1
    elif source == "verify":
        state.stats.verify_emits += 1


# ---------------------------------------------------------------------------
# Verify cycle: 2-token forward + accept/reject + MTP forward for next draft.
# ---------------------------------------------------------------------------


def _run_verify_cycle(gen_batch: Any, state: _MtpState) -> None:
    """Run one verify cycle. Populates ``state.queue`` with 1 (reject) or 2
    (accept) tokens for upcoming emit calls. Updates ``state.next_main`` and
    ``state.draft_tok`` / ``state.draft_lp`` for the cycle after that.
    """
    import time

    import mlx.core as mx

    if state.next_main is None or state.draft_tok is None:
        raise _MtpStepFallback("verify cycle entered without next_main / draft")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)

    inputs = mx.concatenate([state.next_main, state.draft_tok])  # (2,)

    # Update the token buffer per-position (mirrors PR 990 _step_backbone).
    prev_main = None
    prev_draft = None
    if procs is not None:
        prev_main = gen_batch._token_context[0].update_and_fetch(state.next_main)
        prev_draft = gen_batch._token_context[0].update_and_fetch(state.draft_tok)

    # --- backbone forward (materialized before sampling) ---
    # Dispatch the backbone on the generation stream, then force ``mx.eval``
    # on the logits before the sampler runs. MLX is lazy, so without this the
    # later ``mx.eval(verify_tok, bonus_tok)`` barrier would resolve the whole
    # graph in one stall and the heavy verify forward would leak into
    # sample_ms (this is what made the sampler look like the bottleneck in
    # #1097 / #1311 / #1330). The extra eval costs one CPU<->GPU round-trip
    # per cycle (negligible vs the forward compute) and keeps the
    # backbone_ms / sample_ms split accurate.
    t0 = time.perf_counter()
    logits, hidden, gdn_states = _call_backbone(
        gen_batch.model,
        inputs[None, :],
        gen_batch.prompt_cache,
        n_confirmed=1,
    )
    verify_logits = logits[:, 0, :]
    bonus_logits = logits[:, 1, :]
    mx.eval(logits)
    state.stats.backbone_ms += (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    if procs is not None:
        verify_logits = _apply_processors(procs, prev_main, verify_logits)
        bonus_logits = _apply_processors(procs, prev_draft, bonus_logits)
    # Batched logprobs: one logsumexp over (2, vocab) instead of two over
    # (1, vocab). Shaves one reduction per cycle on the vocab dimension.
    combined_logits = mx.concatenate(
        [verify_logits, bonus_logits], axis=0
    )  # (2, vocab)
    combined_lp = combined_logits - mx.logsumexp(
        combined_logits, axis=-1, keepdims=True
    )
    verify_lp_2d = combined_lp[0:1]
    bonus_lp_2d = combined_lp[1:2]
    verify_tok = sampler(verify_lp_2d)
    bonus_tok = sampler(bonus_lp_2d)
    mx.eval(verify_tok, bonus_tok)

    # ``draft_id`` was cached when the draft was sampled (post_init or the
    # prior _step_mtp); skip the GPU→CPU sync that ``state.draft_tok.tolist()``
    # would impose on every cycle.
    draft_id = state.draft_id
    verify_id = int(verify_tok.tolist()[0])
    bonus_id = int(bonus_tok.tolist()[0])

    # Filtered logprobs — distribution the sampler actually drew from.
    # Used for acceptance ratio + residual sampling so they match the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    verify_accept_lp = _accept_lp_for(sampler, verify_lp_2d)
    draft_accept_lp = (
        state.draft_accept_lp
        if state.draft_accept_lp is not None
        else _accept_lp_for(sampler, state.draft_lp)
    )

    if is_greedy:
        accept = verify_id == draft_id
    else:
        log_accept = (
            verify_accept_lp[0, draft_id].item() - draft_accept_lp[draft_id].item()
        )
        # Draw the acceptance roll from mx.random so it follows the same
        # mx.random.seed the rest of the sampler uses (line ~962 residual
        # sampling). stdlib ``random`` was never seeded by oMLX, which made
        # stochastic acceptance irreproducible even with a fixed seed (#1330).
        accept = log_accept >= 0 or float(
            mx.random.uniform(shape=()).item()
        ) < math.exp(log_accept)
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    hidden_at_confirmed = hidden[:, 0:1, :]
    hidden_at_draft = hidden[:, 1:2, :]

    state.stats.cycles += 1
    if accept:
        state.stats.accepts += 1
        # --- cache cleanup (timed) ---
        t0 = time.perf_counter()
        _clear_rollback(gen_batch.prompt_cache)
        state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

        # --- MTP head forward for next draft (timed inside _step_mtp) ---
        new_draft, new_draft_lp = _step_mtp(
            gen_batch,
            hidden_at_draft,
            _ensure_uint32(bonus_tok),
            prev_buf=prev_draft if procs is not None else None,
            stats=state.stats,
        )
        # Queue the two emitted tokens. Per PR 990: the accepted draft uses
        # the *MTP head's* original draft distribution as its logprobs; the
        # bonus uses the verify forward's bonus distribution.
        state.queue.append((draft_id, state.draft_lp, "draft"))
        state.queue.append((bonus_id, bonus_lp_2d.squeeze(0), "bonus"))
        state.next_main = _ensure_uint32(bonus_tok)
        state.draft_tok = new_draft
        state.draft_lp = new_draft_lp
        return

    # Reject path.
    state.stats.rejects += 1
    t0 = time.perf_counter()
    # accepted=0 means only the confirmed token (verify position) is kept;
    # block_size=2 covers both the confirmed and the rejected draft.
    if not _rollback_after_reject(
        gen_batch.model,
        gen_batch.prompt_cache,
        gdn_states,
        accepted=0,
        block_size=2,
    ):
        if procs is not None:
            _trim_token_buffer(gen_batch, 1)
        raise _MtpStepFallback("cache layer rejects rollback")
    if procs is not None:
        _trim_token_buffer(gen_batch, 1)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # Pick the verify-position emit token: residual sample for stochastic.
    # Residual is computed on the *filtered* distributions so the sample
    # comes from `max(p_target_filt - p_draft_filt, 0)` — matching what the
    # sampler would have produced if it had drawn directly from the verify
    # position. emit_lp returned to the caller stays as the raw verify lp
    # so downstream logprobs reporting is consistent with non-MTP paths.
    if is_greedy:
        emit_id = verify_id
        emit_lp = verify_lp_2d.squeeze(0)
    else:
        emit_id, _ = _residual_sample(verify_accept_lp, draft_accept_lp)
        emit_lp = verify_lp_2d.squeeze(0)

    emit_tok = mx.array([emit_id], dtype=mx.uint32)
    new_draft, new_draft_lp = _step_mtp(
        gen_batch,
        hidden_at_confirmed,
        emit_tok,
        prev_buf=prev_main if procs is not None else None,
        stats=state.stats,
    )

    state.queue.append((emit_id, emit_lp, "verify"))
    state.next_main = emit_tok
    state.draft_tok = new_draft
    state.draft_lp = new_draft_lp


# ---------------------------------------------------------------------------
# Helpers used by the verify cycle.
# ---------------------------------------------------------------------------


def _step_mtp(
    gen_batch: Any,
    hidden_at_position: Any,
    next_main_tok: Any,
    prev_buf: Optional[Any],
    stats: Optional["_MtpStats"] = None,
) -> Tuple[Any, Any]:
    """Run one MTP-head forward + sample. Returns ``(draft_tok, draft_lp)``.

    Side effect: caches the host-side int copy of the new draft on
    ``gen_batch._omlx_mtp_state.draft_id`` so the next verify cycle's
    accept check is sync-free.
    """
    import time

    import mlx.core as mx

    state = gen_batch._omlx_mtp_state
    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)

    t0 = time.perf_counter()
    next_ids = next_main_tok.reshape(1, 1)
    mtp_logits = gen_batch.model.mtp_forward(
        hidden_at_position, next_ids, state.mtp_cache
    )
    mtp_logits_2d = mtp_logits[:, -1, :]
    if procs is not None and prev_buf is not None:
        prev_with_next = mx.concatenate([prev_buf, _ensure_uint32(next_main_tok)])
        mtp_logits_2d = _apply_processors(procs, prev_with_next, mtp_logits_2d)
    new_lp = _logprobs(mtp_logits_2d)
    new_tok = sampler(new_lp)
    # Filtered draft lp — what the sampler actually drew from. The next
    # verify cycle's acceptance ratio uses this so the math matches the
    # sampling distribution rather than raw softmax (PR 990 alignment).
    new_accept_lp = _accept_lp_for(sampler, new_lp)
    # ``.tolist()`` forces evaluation; replaces the explicit ``mx.eval`` and
    # piggybacks the host-side int caching on the same sync.
    draft_id_int = int(new_tok.tolist()[0])
    state.draft_id = draft_id_int
    state.draft_accept_lp = new_accept_lp.squeeze(0)
    if stats is not None:
        stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    return _ensure_uint32(new_tok), new_lp.squeeze(0)


def _residual_sample(verify_lp_2d: Any, draft_lp_1d: Any) -> Tuple[int, Any]:
    """Sample from ``max(p_target - p_draft, 0)`` (Leviathan et al. 2022).

    On degenerate input (residual all zero) falls back to the target
    distribution rather than the verify-position argmax — keeps the sample
    drawn from a proper distribution and stays in-graph (no host sync).
    Mirrors mlx-lm PR 990 commit 6594348.

    Returns ``(token_id_int, verify_lp_1d)``.
    """
    import mlx.core as mx

    p_target = mx.exp(verify_lp_2d.squeeze(0))
    p_draft = mx.exp(draft_lp_1d)
    residual = mx.maximum(p_target - p_draft, 0.0)
    # Keep z in graph; mx.where switches to the target distribution when
    # the residual mass is zero. ``categorical`` treats log(0) = -inf as
    # p=0 so no safety epsilon is needed.
    z = residual.sum(keepdims=True)
    dist = mx.where(z > 0, residual, p_target)
    sample = mx.random.categorical(mx.log(dist).reshape(1, -1))
    return int(sample.item()), verify_lp_2d.squeeze(0)


# ---------------------------------------------------------------------------
# Response builder — mirrors GenerationBatch.next()'s per-sequence epilogue.
# ---------------------------------------------------------------------------


def _emit_response(
    gen_batch: Any,
    token_id: int,
    logprobs_1d: Any,
    stats: Optional["_MtpStats"] = None,
) -> List[Any]:
    """Produce a single-element response list, applying the standard
    epilogue (token append + max_tokens / matcher checks) so external
    callers (BatchGenerator, scheduler, response stream) see the same
    contract as the unmodified next().
    """
    Response = type(gen_batch).Response

    finish_reason: Optional[str] = None
    match_sequence = None

    gen_batch.tokens[0].append(token_id)
    gen_batch._num_tokens[0] += 1
    if gen_batch._num_tokens[0] >= gen_batch.max_tokens[0]:
        finish_reason = "length"

    new_state, match_sequence, current_state = gen_batch.state_machines[0].match(
        gen_batch._matcher_states[0], token_id
    )
    gen_batch._matcher_states[0] = new_state
    if match_sequence is not None and current_state is None:
        finish_reason = "stop"

    if finish_reason is not None:
        prompt_cache = gen_batch.extract_cache(0)
        all_tokens = gen_batch.tokens[0]
        response = Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=finish_reason,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=prompt_cache,
            all_tokens=all_tokens,
        )
        if stats is not None:
            _log_mtp_stats(gen_batch.uids[0], stats, finish_reason)
        # Drop state *before* filter([]) so the patched_filter epilogue
        # doesn't double-log when the standard finish path already logged.
        if hasattr(gen_batch, "_omlx_mtp_state"):
            try:
                delattr(gen_batch, "_omlx_mtp_state")
            except AttributeError:
                pass
        gen_batch.filter([])
        return [response]

    return [
        Response(
            uid=gen_batch.uids[0],
            token=token_id,
            logprobs=logprobs_1d,
            finish_reason=None,
            current_state=current_state,
            match_sequence=match_sequence,
            prompt_cache=None,
            all_tokens=None,
        )
    ]
