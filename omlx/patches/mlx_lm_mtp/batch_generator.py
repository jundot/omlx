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
import os
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

from . import cache_rollback as _rollback_mod

logger = logging.getLogger(__name__)


def _set_verify_qmm_armed(flag: bool) -> None:
    """Arm the verify-shape qmm routing for the duration of an MTP forward.

    Import is deferred and failure-tolerant: the kernel module is optional
    and its absence must not affect the MTP path.
    """
    try:
        from ..qwen35_verify_qmm import set_verify_qmm_armed

        set_verify_qmm_armed(flag)
    except Exception:
        pass

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
            realign_rows = getattr(self, "_omlx_realign_rows", None)
            if callable(realign_rows):
                realign_rows()
            _maybe_clear_multirow_marker(self)

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
            _log_multirow_mtp_inactive_once(self)
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
            if _generation_batch_has_active_mtp(gen_batch):
                old_completion_batch_size = getattr(
                    self,
                    "completion_batch_size",
                    None,
                )
                had_completion_batch_size = hasattr(self, "completion_batch_size")
                # Force mlx-lm's "hands full" early return after generation,
                # even if an active row-wise MTP batch shrinks during next().
                self.completion_batch_size = 0
                try:
                    return original_bg_next(self, *args, **kwargs)
                finally:
                    if had_completion_batch_size:
                        self.completion_batch_size = old_completion_batch_size
                    elif hasattr(self, "completion_batch_size"):
                        delattr(self, "completion_batch_size")
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


def _model_mtp_decode_enabled(model: Any) -> bool:
    """Return the MTP decode decision captured on the loaded model instance.

    ``mlx_lm_mtp._MTP_ACTIVE`` is a construction-time switch. It is reset
    before each model load so patched ``__init__`` methods know whether to
    attach MTP heads, but decode-time eligibility must not read that global:
    a later non-MTP load would otherwise disable already-loaded MTP models.
    """
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    return any(
        bool(getattr(candidate, "_omlx_mtp_decode_enabled", False))
        for candidate in candidates
    )


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


def _generation_batch_has_active_mtp(gen_batch: Any) -> bool:
    """True while a generation batch owns Native MTP cache state.

    mlx-lm's ``BatchGenerator._next`` generates first and then may promote
    pending prompt work into the same ``GenerationBatch`` via ``extend()``. That
    merge path forces MTP reconciliation, which can re-prefill a long streamed
    context outside the scheduler's guarded prefill path. Treat active MTP as
    a temporary full generation batch so late-join requests wait instead.
    """
    if gen_batch is None:
        return False
    try:
        if len(gen_batch) == 0:
            return False
    except Exception:
        pass
    return (
        getattr(gen_batch, "_omlx_mtp_state", None) is not None
        or getattr(gen_batch, "_omlx_mtp_batch_state", None) is not None
    )


def _mtp_common_eligible(gen_batch: Any) -> bool:
    if not hasattr(gen_batch, "model"):
        return False
    if not hasattr(gen_batch.model, "mtp_forward"):
        return False
    if not _model_has_mtp_module(gen_batch.model):
        return False
    if not _model_mtp_decode_enabled(gen_batch.model):
        return False
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) == 0:
        return False
    if _has_grammar_processors(gen_batch):
        return False
    return True


_ROWWISE_BATCH_MTP_ENV = "OMLX_MTP_ROWWISE_BATCH"


def _rowwise_batch_mtp_enabled() -> bool:
    """Opt-in for row-wise MTP on multi-row batches (default off).

    The row-wise path runs one backbone forward per row per cycle, so its
    aggregate throughput is roughly single-stream MTP throughput regardless
    of batch size, while standard batched decode amortizes one forward over
    all rows. Measured on Qwen3.6-27B-oQ4e-mtp / M3 Ultra (pp1024/tg128):
    row-wise 53.3 / 52.5 tok/s aggregate at batch 2 / 4 versus 65.2 / 86.5
    for standard batched decode — despite 83-93% draft acceptance. It only
    pays off when tokens-per-cycle exceeds the row count, so it stays off
    unless explicitly requested.
    """
    return os.environ.get(_ROWWISE_BATCH_MTP_ENV, "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _allows_new_mtp_activation(gen_batch: Any, state_attr: str) -> bool:
    if getattr(gen_batch, state_attr, None) is not None:
        return True
    # The multirow-decode marker guards the singleton-init invariant only
    # (see _mark_standard_multirow_decode). Row-wise batch activation seeds
    # every row from a freshly extracted per-row cache, so a prior standard
    # multi-row decode is exactly the state it expects — blocking it here
    # would permanently lock batches out of MTP, because a batch's first
    # decode step is always standard.
    if state_attr == "_omlx_mtp_state" and getattr(
        gen_batch, "_omlx_mtp_saw_standard_multirow_decode", False
    ):
        return False
    return bool(getattr(gen_batch, "_omlx_mtp_activation_safe", True))


def _mark_standard_multirow_decode(gen_batch: Any) -> None:
    """Remember that this batch has decoded with shared standard cache state.

    A row that survives a standard multi-row decode and later becomes singleton
    no longer satisfies the narrow invariant that singleton MTP initialization
    relies on. Existing row-wise MTP state may continue, but starting a fresh
    singleton MTP state after late-join/late-finish reshaping is unsafe.

    The marker is not permanent: once the batch shrinks back to one row with a
    verifiably compact cache, ``_maybe_clear_multirow_marker`` lifts it so the
    surviving request regains MTP for the rest of its generation.
    """
    try:
        if len(getattr(gen_batch, "uids", []) or []) > 1:
            gen_batch._omlx_mtp_saw_standard_multirow_decode = True
    except Exception:
        pass


def _log_multirow_mtp_inactive_once(gen_batch: Any) -> None:
    """Say once, at INFO, why an MTP-capable batch is decoding without MTP.

    #2150 showed the silent fallback is easy to misread: the benchmark's
    batched phases report plain continuous-batching numbers while every log
    line about MTP inactivity hides at DEBUG. One line per batch keeps the
    signal visible without per-step spam.
    """
    if getattr(gen_batch, "_omlx_mtp_inactive_logged", False):
        return
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) <= 1:
        return
    if not _mtp_common_eligible(gen_batch):
        return
    gen_batch._omlx_mtp_inactive_logged = True
    if not _rowwise_batch_mtp_enabled():
        logger.info(
            "MTP inactive for %d-row batch: standard batched decode is faster "
            "at this batch size (set %s=1 to force row-wise MTP)",
            len(uids),
            _ROWWISE_BATCH_MTP_ENV,
        )
    else:
        logger.info(
            "MTP inactive for %d-row batch: %s",
            len(uids),
            _ineligibility_reason(gen_batch) or "activation deferred",
        )


def _maybe_clear_multirow_marker(gen_batch: Any) -> None:
    """Re-enable singleton MTP once a shrunken batch is verifiably safe again.

    The multirow marker exists because a row surviving batch reshaping may sit
    in a cache whose layout singleton MTP's raw backbone calls don't expect
    (left padding). ``BatchKVCache.filter()`` shifts out the minimum shared
    left padding, so a batch filtered down to one row is compact again in the
    common case — verify that per layer instead of assuming, and only then
    lift the marker. Without this, a request that ever shared a decode step
    with another request is locked out of MTP for the rest of its generation
    even once it is running alone (#2150).
    """
    if not getattr(gen_batch, "_omlx_mtp_saw_standard_multirow_decode", False):
        return
    uids = getattr(gen_batch, "uids", None)
    if uids is None or len(uids) != 1:
        return
    for cache in getattr(gen_batch, "prompt_cache", None) or []:
        left_padding = getattr(cache, "left_padding", None)
        if left_padding is None:
            continue
        try:
            if max(int(v) for v in left_padding.tolist()) > 0:
                return
        except Exception:
            return
    gen_batch._omlx_mtp_saw_standard_multirow_decode = False
    logger.info("MTP singleton recovery: multirow marker cleared (compact cache)")


def _is_mtp_eligible(gen_batch: Any) -> bool:
    """``__init__`` and ``next`` only engage MTP for single-sequence batches
    when the model exposes ``mtp_forward``, has an attached MTP head, and
    was loaded with MTP decode enabled.

    The MTP head may be attached unconditionally (e.g. by the mlx-vlm
    runtime patches, which need it for weight-load matching even when
    inference-time MTP is off) — so head presence alone is not enough
    to decide whether to run the draft/verify cycle. The per-instance
    ``_omlx_mtp_decode_enabled`` marker reflects the per-load
    ``model_settings.mtp_enabled`` choice without being affected by later
    model loads in the same process.
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
    if getattr(
        gen_batch, "_omlx_mtp_batch_state", None
    ) is None and not _rowwise_batch_mtp_enabled():
        return False
    # No cache-position alignment requirement: activation seeds each row from
    # its own extract_cache(idx) view and steady-state row cycles diverge the
    # per-row offsets immediately anyway (accept counts differ per row), so
    # the merge path already handles ragged rows. Under continuous batching
    # rows join at different times, so requiring aligned offsets at
    # activation kept this path from ever engaging (#2150).
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
    if not _model_mtp_decode_enabled(gen_batch.model):
        return (
            "model instance MTP decode flag is off "
            "(model_settings.mtp_enabled was False when this model was loaded)"
        )
    uids = getattr(gen_batch, "uids", None)
    if uids is None:
        return "GenerationBatch has no uids"
    if len(uids) != 1:
        if not _allows_new_mtp_activation(gen_batch, "_omlx_mtp_batch_state"):
            return "pending prompt work may still merge into this batch"
        if getattr(
            gen_batch, "_omlx_mtp_batch_state", None
        ) is None and not _rowwise_batch_mtp_enabled():
            return (
                f"row-wise batch MTP is opt-in ({_ROWWISE_BATCH_MTP_ENV}=1); "
                "standard batched decode is faster at batch >= 2"
            )
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
    accepts: int = 0  # accepted draft tokens (depth-k: sum over positions)
    rejects: int = 0  # cycles that ended in a rejection
    init_emits: int = 0  # tokens emitted from the post-init queue (always 2)
    draft_emits: int = 0  # tokens emitted as accepted drafts
    bonus_emits: int = 0  # tokens emitted as bonus (accepted + emit_bonus)
    verify_emits: int = 0  # tokens emitted as verify-position correction (reject path)
    # Per-depth accept telemetry for chained drafting: drafted[j] counts
    # cycles where a draft existed at depth j; accepted[j] counts how many
    # of those were verified. Depth-1 legacy path fills index 0 only.
    depth_drafted: List[int] = field(default_factory=list)
    depth_accepted: List[int] = field(default_factory=list)
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

    # --- depth-k chained drafting (Qwen3.5/3.6 only) ---
    # chain=True routes decode through _run_verify_cycle_chain; False keeps
    # the PR-990 depth-1 legacy cycle.
    chain: bool = False
    depth: int = 1
    # head_clone=True runs speculative head steps on a per-cycle cache clone
    # (models whose head cache can't be exactly trimmed once rotated).
    head_clone: bool = False
    # Pending draft tokens for the next verify forward: (depth,) uint32 array.
    # Host-side ids are read in the verify cycle's single sync, not here.
    drafts: Optional[Any] = None
    # Per-draft raw logprob rows (vocab,) — emitted as the accepted drafts'
    # logprobs (PR 990 contract) — and sampler-filtered rows for stochastic
    # acceptance. Lazy arrays; only evaluated if a consumer touches them.
    draft_lps: List[Any] = field(default_factory=list)
    draft_accept_lps: List[Any] = field(default_factory=list)
    # MTP-head cache offset at cycle start. Chain entries beyond this offset
    # are speculative and trimmed at commit; committed history is re-appended
    # from verify-forward hidden rows so the head sees a dense, committed-only
    # timeline.
    hist_offset: int = 0
    # Sampler for draft tokens (lazily resolved). For stochastic target
    # samplers this is a *sharper* distribution than the target (temp 0.6 /
    # top_p 0.95 / top_k 20) — the Leviathan/Chen acceptance ratio uses the
    # true draft distribution q, so any q keeps the output distribution
    # exact, and truncating the 1-layer head's noisy tail is what keeps
    # acceptance usable on high-entropy content (creative prose collapses to
    # ~10-20% with matched-temp drafts).
    draft_sampler: Optional[Any] = None
    # Adaptive depth controller (None = fixed depth). Chooses how many
    # drafts the next chain builds from rolling accept/latency estimates.
    controller: Optional[Any] = None

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

    All layers are checked before anything is mutated: a partial rollback
    (early layers trimmed, a later layer refusing) leaves per-layer KV
    lengths desynchronised by one position and corrupts every subsequent
    forward (the shared attention mask is built from the first layer's
    cache, so the mismatch surfaces as a broadcast error on DeepSeek-V4
    compressed-attention layers).
    """
    for c in prompt_cache:
        if getattr(c, "rollback_state", None) is not None:
            # A draft stash marks the chain-path *pre-forward* snapshot
            # semantics (qwen35_model unsplit verify); restoring it here
            # would drop the confirmed token too. Only mtp_partial_rollback
            # knows how to replay it — refuse so the caller falls back.
            if getattr(c, "_mtp_draft_stash", None) is not None:
                return False
            continue
        if hasattr(c, "is_trimmable") and c.is_trimmable():
            continue
        return False
    for c in prompt_cache:
        rollback = getattr(c, "rollback_state", None)
        if rollback is not None:
            conv_snap, ssm_snap = rollback
            c[0] = conv_snap
            c[1] = ssm_snap
            c.rollback_state = None
            continue
        c.trim(1)
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
    - mlx-vlm path returns a ``LanguageModelOutput`` or 3-tuple
      ``(logits, hidden, gdn_states)`` so a rejected draft can be rolled
      back via ``rollback_speculative_cache``.

    ``n_confirmed`` is forwarded so the mlx-lm path can split its
    GatedDeltaNet forward into confirmed and draft chunks. mlx-vlm
    discards it (irrelevant — rollback is post-hoc, not splitwise).

    The rotating-cache undo stash (cache_rollback) is armed for the
    duration of the forward so a rejected draft can be rolled back even on
    a rotated RotatingKVCache; non-MTP forwards keep stock trim semantics.
    """
    kwargs = {"cache": cache, "return_hidden": True}
    if n_confirmed:
        kwargs["n_confirmed"] = n_confirmed
    _rollback_mod.set_undo_armed(True)
    _set_verify_qmm_armed(True)
    try:
        result = model(inputs, **kwargs)
    finally:
        _set_verify_qmm_armed(False)
        _rollback_mod.set_undo_armed(False)

    # LanguageModelOutput (mlx-vlm dataclass)
    if hasattr(result, "logits") and hasattr(result, "hidden_states"):
        hidden = result.hidden_states
        if isinstance(hidden, list):
            hidden = hidden[-1] if hidden else None
        return result.logits, hidden, getattr(result, "gdn_states", None)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            return result[0], result[1], None
    raise TypeError(f"backbone returned unexpected shape: {type(result).__name__}")


def _clear_rollback(prompt_cache: List[Any]) -> None:
    """Drop rollback snapshots after a draft is accepted."""
    for c in prompt_cache:
        if hasattr(c, "rollback_state") and c.rollback_state is not None:
            c.rollback_state = None
        if getattr(c, "_mtp_draft_stash", None) is not None:
            c._mtp_draft_stash = None
        if getattr(c, "_mtp_undo", None) is not None:
            c._mtp_undo = None
        for sub in getattr(c, "caches", ()):
            if getattr(sub, "_mtp_undo", None) is not None:
                sub._mtp_undo = None


def _ensure_uint32(arr):
    """Ensure a 1-element mx.array is uint32 (cache update_and_fetch expects it)."""
    import mlx.core as mx

    if arr.dtype == mx.uint32:
        return arr
    return arr.astype(mx.uint32)


# ---------------------------------------------------------------------------
# Depth-k chained drafting helpers (Qwen3.5/3.6): a linear draft chain through
# the MTP head, one batched verify forward covering all drafts plus a free
# bonus row, offset-trim KV rollback with GDN prefix replay, and a
# committed-only MTP-head history rebuilt from verify hidden rows each cycle.
# ---------------------------------------------------------------------------


def _resolve_mtp_chain_depth(model: Any) -> Tuple[bool, int, bool]:
    """Read the chain capability markers stamped on the model at load.

    Returns ``(chain, depth, head_clone)``. ``head_clone`` marks models
    whose MTP-head cache cannot be exactly trimmed (e.g. DeepSeek-V4's
    RotatingKVCache head once rotated): the chain then runs its speculative
    draft steps on a per-cycle clone and keeps the persistent head cache
    committed-only, instead of trimming speculative entries afterwards.
    """
    candidates = [model]
    for attr in ("language_model", "_language_model"):
        inner = getattr(model, attr, None)
        if inner is not None and inner is not model:
            candidates.append(inner)
    for candidate in candidates:
        if getattr(candidate, "_omlx_mtp_chain", False):
            depth = int(getattr(candidate, "_omlx_mtp_depth", 1) or 1)
            head_clone = bool(getattr(candidate, "_omlx_mtp_head_clone", False))
            return True, max(1, min(8, depth)), head_clone
    return False, 1, False


def _clone_mtp_head_cache(mtp_cache: List[Any]) -> List[Any]:
    """Detached per-cycle copy of the MTP-head cache for speculative steps.

    ``copy.copy`` keeps scalars; mx.array attributes are detached with
    ``v + 0`` so the clone's in-place ring writes never mutate arrays the
    persistent cache still references; list attributes are shallow-copied.
    Container caches (``CacheList``-style, exposing ``.caches``) recurse.
    """
    import copy

    import mlx.core as mx

    def clone_one(c):
        if c is None:
            return None
        subs = getattr(c, "caches", None)
        if subs is not None:
            return type(c)(*[clone_one(sub) for sub in subs])
        new = copy.copy(c)
        for attr, val in vars(c).items():
            if isinstance(val, mx.array):
                setattr(new, attr, val + 0)
            elif isinstance(val, list):
                setattr(new, attr, list(val))
        return new

    return [clone_one(c) for c in mtp_cache]


def _trunk_norm_module(model: Any):
    """Final RMSNorm of the backbone (for post_norm head inputs).

    Walks both wrapper conventions: mlx-lm's outer ``Model.language_model``
    and oMLX's ``VLMModelAdapter._language_model`` (mlx-vlm path).
    """
    inner = model
    for attr in ("language_model", "_language_model"):
        candidate = getattr(model, attr, None)
        if candidate is not None:
            inner = candidate
            break
    return inner.model.norm


# The MTP head is fed the trunk's *post-norm* hidden and chains on its own
# post-norm output. Measured on Qwen3.6-27B this accepts a few points higher
# than PR 990's pre-norm at every depth. Draft-side only, so output identity
# is unaffected regardless.
_HEAD_HIDDEN_POST_NORM = True


def _mtp_head_trim_to(mtp_cache: List[Any], offset: int) -> None:
    """Trim speculative chain entries so the head cache ends at ``offset``."""
    for c in mtp_cache:
        current = int(getattr(c, "offset", 0))
        extra = current - offset
        if extra > 0:
            c.trim(extra)


class _DepthController:
    """Adaptive draft-depth selection.

    Pure host-side bookkeeping — no extra GPU syncs. Tracks an EMA of
    conditional acceptance per depth position and a wall-time EMA of cycle
    cost per depth used, then picks the depth with the best expected tokens
    per unit time:

        score(d) = (1 + p1 + p1*p2 + ... ) / t_est(d)

    Everything the decision uses is measured on this machine, on this model,
    under the current load — no hand-tuned per-chip or per-model value:

    - Cost: a warmup sweep runs each depth once so every ``t[d]`` starts from
      a real cycle; the marginal cost of an extra verify row is the measured
      slope between depths (``_marginal_est``), so a fine-grained MoE on a
      bandwidth-limited chip learns its true (large) marginal within the first
      few cycles. ``MARGINAL_MS`` is only the pre-measurement fallback.
    - Drift: the cost EMA horizon is wall-clock (``TAU_MS``), not a cycle
      count, so context growth, thermal throttling and external GPU contention
      are tracked at constant real-time responsiveness however long a cycle
      is; a one-off slow cycle is damped (``SPIKE_RATIO``).
    - Staleness: only the depth currently run gets fresh measurements, and a
      fresh-vs-stale cost comparison is systematically biased — e.g. a depth
      whose t was measured during the slow post-prefill cycles looks expensive
      forever, so the controller locks into its rival (measured as a depth-2
      lock costing prose ~2-4%). Probes are therefore BIDIRECTIONAL and
      staleness-directed: on a wall-clock cadence, re-run the best rival depth
      (shallower or deeper) when its score is within ``PROBE_MARGIN`` of the
      current choice, and periodically the most-stale depth, so every t[d] has
      bounded age. On heavy models a fixed wall-clock cadence would spend a
      large share of cycles probing, so probing is duty-bounded to
      ~``PROBE_DUTY`` of cycles — a scale-free ratio, not a per-model tuning.

    Content-adaptive by construction: prose/chat settles at depth 1,
    code/predictable text climbs. Rejected alternatives (interleaved
    in-process A/B, rotated order, paired per rep, on Qwen3.6-35B-A3B +
    GLM-5.2, M3 Ultra): a fixed shallow-bias constant (won earlier separate-
    process comparisons only by masking the staleness lock; this design beats
    it on 3 of 4 model x content cells and ties the 4th), a pure realized
    tok/s bandit (exploration tax), a live per-cycle learned correction
    (decision churn), a frozen cross-generation correction (content
    oscillation), and a base x shape cost decomposition (unidentifiable while
    one depth runs for long stretches).
    """

    ALPHA = 0.08  # acceptance EMA weight (token domain, content-driven)
    TAU_MS = 400.0  # cost EMA horizon in wall-clock ms (load/thermal/context)
    PROBE_PERIOD_MS = 1000.0  # min wall-time between probes (light models)
    PROBE_PERIOD_MAX_MS = 5000.0  # staleness-exploration cadence floor
    PROBE_LEN = 4
    PROBE_DUTY = 0.15  # probes never consume more than ~this share of cycles
    PROBE_MARGIN = 1.15  # a rival within this score ratio is worth re-measuring
    SPIKE_RATIO = 2.0  # a cycle above this * the EMA is treated as an outlier
    SPIKE_DAMP = 0.25  # ...and folded in at this fraction of the normal weight
    # Fallback prior for one extra verify token's cost, used only until two
    # depths have actually been measured; after that the marginal is the
    # measured slope between depths. 7 ms matches dense backbones (6-10 ms).
    MARGINAL_MS = 7.0
    HYSTERESIS = 1.03  # switch depth only for a >3% score gain

    def __init__(self, max_depth: int, marginal_ms: Optional[float] = None):
        if marginal_ms:
            self.MARGINAL_MS = float(marginal_ms)
        self.max_depth = max(1, int(max_depth))
        self.cur = self.max_depth  # first cycle drafts deep; warmup sweeps down
        self.p = [0.6] * self.max_depth
        self.t: Dict[int, float] = {}
        self.t_age: Dict[int, float] = {}  # ms since each depth was measured
        self.cycles = 0
        self.probe_left = 0
        self._ms_probe = 0.0  # wall-time since any probe burst
        self._ms_explore = 0.0  # wall-time since a staleness-exploration burst
        # Measure each depth once (max..1) before the score gate takes over, so
        # t[] and the marginal estimate are data-driven within max_depth cycles.
        self._warmup: List[int] = list(range(self.max_depth, 0, -1))

    def observe(self, used: int, accepted: int, cycle_ms: float) -> None:
        self.cycles += 1
        used = max(1, min(int(used), self.max_depth))
        accepted = max(0, min(int(accepted), used))
        # Acceptance: token-domain EMA (a property of model/content, not load).
        a = self.ALPHA
        for j in range(used):
            hit = 1.0 if j < accepted else 0.0
            self.p[j] = (1.0 - a) * self.p[j] + a * hit
            if j >= accepted:
                break
        # Cost: wall-time-domain EMA with a one-off-spike guard, plus per-depth
        # ages so probes can target the estimate that is most stale.
        cycle_ms = max(0.0, float(cycle_ms))
        self._update_time(used, cycle_ms)
        for d in list(self.t_age):
            self.t_age[d] += cycle_ms
        self.t_age[used] = 0.0
        self._ms_probe += cycle_ms
        self._ms_explore += cycle_ms

        # Warmup sweep: keep walking max..1 until every depth is measured once.
        if self._warmup:
            self._warmup.pop(0)
            if self._warmup:
                self.cur = self._warmup[0]
                return
            self.cur = self._best()
            self._ms_probe = 0.0
            return

        # Finishing a probe burst.
        if self.probe_left > 0:
            self.probe_left -= 1
            if self.probe_left == 0:
                self.cur = self._best()
                self._ms_probe = 0.0
            return

        # Re-decide every cycle (cheap); HYSTERESIS in _best prevents thrash.
        self.cur = self._best()

        # Probe scheduling: bounded-staleness re-measurement in either
        # direction, at a duty-bounded wall-clock cadence.
        if self.max_depth > 1:
            period = max(
                self.PROBE_PERIOD_MS,
                self.PROBE_LEN * cycle_ms / self.PROBE_DUTY,
            )
            if self._ms_probe >= period:
                explore_due = self._ms_explore >= max(
                    self.PROBE_PERIOD_MAX_MS, 2.0 * period
                )
                target = (
                    self._most_stale() if explore_due else self._best_rival()
                )
                if target is not None:
                    self.cur = target
                    self.probe_left = self.PROBE_LEN
                    self._ms_probe = 0.0
                    if explore_due:
                        self._ms_explore = 0.0

    def _time_alpha(self, cycle_ms: float) -> float:
        # EMA weight for a cycle of this wall-time: the memory horizon is
        # ~TAU_MS regardless of cycle duration, so responsiveness is constant
        # in real time whether a cycle is 8 ms (short) or 80 ms (128k context).
        return 1.0 - math.exp(-max(0.0, float(cycle_ms)) / self.TAU_MS)

    def _update_time(self, used: int, cycle_ms: float) -> None:
        cycle_ms = max(0.0, float(cycle_ms))
        prev = self.t.get(used)
        if prev is None:
            self.t[used] = cycle_ms
            return
        # Deliberately a per-cycle EMA, NOT an irregular-sampling EMA weighted
        # by staleness age. Age-weighting (nearly replacing a stale estimate at
        # the first probe cycle) is the textbook form, but it was measured
        # WORSE here: single-cycle noise is ~±10%, so replacing an estimate
        # from a 4-cycle probe burst injects that noise straight into the depth
        # decision every probe (~1s), and prose re-over-drafted (-1.6%). The
        # slow EMA is a variance shield; stale errors are corrected by probe
        # REPETITION instead (each ~1s rival probe moves the estimate ~10% of
        # the gap, converging within a few seconds).
        a = self._time_alpha(cycle_ms)
        if cycle_ms > self.SPIKE_RATIO * prev:
            a *= self.SPIKE_DAMP  # a one-off spike moves the estimate slowly
        self.t[used] = (1.0 - a) * prev + a * cycle_ms

    def _marginal_est(self) -> float:
        # Measured cost of one extra verify row: the slope between the cheapest
        # and priciest measured depths. Falls back to the prior until two
        # depths exist. This is what self-calibrates the controller to the real
        # (model x chip x context) marginal instead of a hardcoded value.
        if len(self.t) >= 2:
            depths = sorted(self.t)
            lo, hi = depths[0], depths[-1]
            if hi > lo:
                slope = (self.t[hi] - self.t[lo]) / (hi - lo)
                if slope > 0.0:
                    return slope
        return self.MARGINAL_MS

    def _t_est(self, d: int) -> float:
        if d in self.t:
            return self.t[d]
        if not self.t:
            return 30.0 + self.MARGINAL_MS * d
        ref = min(self.t, key=lambda x: abs(x - d))
        return self.t[ref] + self._marginal_est() * (d - ref)

    def _score(self, d: int) -> float:
        expected = 1.0
        run = 1.0
        for j in range(d):
            run *= self.p[j]
            expected += run
        return expected / max(1e-6, self._t_est(d))

    def _best_rival(self) -> Optional[int]:
        # The highest-scoring depth other than cur, if within PROBE_MARGIN —
        # i.e. a depth whose (possibly stale) estimate could flip the choice.
        # Bidirectional on purpose: re-measuring a SHALLOWER rival is what
        # breaks the depth-2 lock (a stale-high t[1] hides depth 1's true
        # advantage and nothing else would ever refresh it).
        best = self._score(self.cur)
        if best <= 0.0:
            return self._most_stale()
        rival = None
        rival_score = 0.0
        for d in range(1, self.max_depth + 1):
            if d == self.cur:
                continue
            s = self._score(d)
            if s > rival_score:
                rival, rival_score = d, s
        if rival is not None and rival_score >= best / self.PROBE_MARGIN:
            return rival
        return None

    def _most_stale(self) -> Optional[int]:
        # The depth whose cost estimate has gone longest unmeasured (never
        # measured counts as infinitely stale). Keeps every t[d] fresh enough
        # that fresh-vs-stale comparison bias stays bounded.
        cand = None
        worst = -1.0
        for d in range(1, self.max_depth + 1):
            if d == self.cur:
                continue
            age = self.t_age.get(d)
            age = float("inf") if age is None else age
            if age > worst:
                cand, worst = d, age
        return cand

    def _best(self) -> int:
        # argmax of measured score with switch hysteresis; the shallow-to-deep
        # scan with strict '>' keeps the lower depth on an exact tie.
        scores = [self._score(d) for d in range(1, self.max_depth + 1)]
        best_i = 0
        for i in range(1, self.max_depth):
            if scores[i] > scores[best_i]:
                best_i = i
        best_d = best_i + 1
        if best_d != self.cur and scores[best_i] < scores[self.cur - 1] * self.HYSTERESIS:
            return self.cur
        return best_d


# Draft sampler for stochastic (temp > 0) decoding. A sharper distribution
# than the target: the 1-layer head's noisy tail otherwise gets sampled and
# rejected, collapsing acceptance on high-entropy content. Exactness holds
# because the Leviathan/Chen ratio uses this sampler's own distribution as q.
_DRAFT_SAMPLER_TEMP = 0.6
_DRAFT_SAMPLER_TOP_P = 0.95
_DRAFT_SAMPLER_TOP_K = 20


def _resolve_draft_sampler(gen_batch: Any, state: _MtpState):
    """Sampler used to draw MTP draft tokens.

    Greedy target → greedy drafts (preserves the greedy-identity contract).
    Stochastic target → the sharper draft sampler above. The acceptance ratio
    and residual sampling use this sampler's filtered distribution as q, so
    the emitted token distribution still equals the target sampler's exactly.
    """
    if state.draft_sampler is not None:
        return state.draft_sampler
    if _is_greedy(gen_batch):
        state.draft_sampler = _resolve_sampler(gen_batch)
        return state.draft_sampler

    from omlx.utils.sampling import make_sampler

    state.draft_sampler = make_sampler(
        temp=_DRAFT_SAMPLER_TEMP,
        top_p=_DRAFT_SAMPLER_TOP_P,
        top_k=_DRAFT_SAMPLER_TOP_K,
    )
    return state.draft_sampler


def _chain_next_drafts(
    gen_batch: Any,
    state: _MtpState,
    hidden_rows: Any,
    committed: Any,
    prev_buf: Optional[Any],
) -> None:
    """Rebuild committed MTP-head history and draft the next chain.

    ``hidden_rows`` is the trunk hidden at the positions of the n tokens
    *preceding* each committed token — (1, n, H) pre-norm for Qwen (the
    final trunk norm is applied here), or the model's native 4D raw hidden
    for DeepSeek-V4 (passed through untouched); ``committed`` is the (n,)
    uint32 committed tokens. One batched head forward appends n committed
    history entries and yields the next cycle's first draft logits for free
    (its last entry pairs the newest committed token with the hidden of its
    predecessor — exactly the fused state that predicts the next-next token).
    The remaining drafts chain on the head's own output hidden; with
    ``state.head_clone`` those speculative steps run on a per-cycle clone so
    the persistent head cache stays committed-only.

    Populates ``state.drafts`` / ``draft_lps`` / ``draft_accept_lps`` and
    advances ``state.hist_offset`` by n. All arrays are dispatched with
    ``mx.async_eval`` and stay lazy on the host; the next verify cycle's
    single sync resolves them.
    """
    import mlx.core as mx

    model = gen_batch.model
    sampler = _resolve_draft_sampler(gen_batch, state)
    procs = _proc_list(gen_batch)

    if _HEAD_HIDDEN_POST_NORM and hidden_rows.ndim == 3:
        hidden_rows = _trunk_norm_module(model)(hidden_rows)

    n = committed.shape[0]
    logits, head_hidden = model.mtp_forward(
        hidden_rows,
        committed.reshape(1, n),
        state.mtp_cache,
        return_hidden=True,
        logits_keep=1,
    )
    state.hist_offset += int(n)

    draft_toks: List[Any] = []
    draft_lps: List[Any] = []
    draft_accept_lps: List[Any] = []

    chain_prefix = committed[-1:]
    h = head_hidden[:, -1:]
    depth = state.controller.cur if state.controller is not None else state.depth
    chain_cache = state.mtp_cache
    if state.head_clone and depth > 1:
        chain_cache = _clone_mtp_head_cache(state.mtp_cache)
    for j in range(depth):
        logits_2d = logits[:, -1, :]
        if procs is not None and prev_buf is not None:
            prev = mx.concatenate(
                [prev_buf.astype(mx.int32), chain_prefix.astype(mx.int32)]
                + [t.reshape(1).astype(mx.int32) for t in draft_toks]
            )
            logits_2d = _apply_processors(procs, prev, logits_2d)
        lp_2d = _logprobs(logits_2d)
        tok = _ensure_uint32(sampler(lp_2d))
        draft_toks.append(tok)
        draft_lps.append(lp_2d.squeeze(0))
        draft_accept_lps.append(_accept_lp_for(sampler, lp_2d).squeeze(0))
        if j + 1 == depth:
            break
        logits, head_hidden = model.mtp_forward(
            h,
            tok.reshape(1, 1),
            chain_cache,
            return_hidden=True,
        )
        h = head_hidden[:, -1:]

    state.drafts = mx.concatenate(draft_toks)
    state.draft_lps = draft_lps
    state.draft_accept_lps = draft_accept_lps
    # Fire-and-forget dispatch: the GPU evaluates the chain while the host
    # finishes emit bookkeeping; the next cycle's sync finds it materialized.
    mx.async_eval(state.drafts)


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

    chain, depth, head_clone = _resolve_mtp_chain_depth(gen_batch.model)

    if chain:
        # Depth-k seed: the history fold pairs hidden(main_tok) with
        # next_main_tok — the first committed history entry — and its logits
        # are the first draft's distribution; the rest of the chain follows.
        mx.eval(main_tok, next_main_tok)
        state = _MtpState(uid=gen_batch.uids[0])
        state.chain = True
        state.depth = depth
        state.head_clone = head_clone
        if depth > 1:
            state.controller = _DepthController(
                depth,
                marginal_ms=getattr(
                    gen_batch.model, "_omlx_mtp_marginal_ms", None
                ),
            )
        state.mtp_cache = gen_batch.model.make_mtp_cache()
        state.next_main = _ensure_uint32(next_main_tok)
        state.queue.append((int(main_tok.tolist()[0]), main_lp, "init"))
        state.queue.append(
            (int(next_main_tok.tolist()[0]), next_main_lp.squeeze(0), "init")
        )
        _chain_next_drafts(
            gen_batch,
            state,
            hidden[:, -1:],
            state.next_main,
            prev_buf,
        )
        gen_batch._omlx_mtp_state = state
        return

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
    total_drafted = sum(stats.depth_drafted) or stats.cycles
    if total_drafted > 0:
        rate_str = f"{stats.accepts / total_drafted * 100:.1f}%"
    else:
        rate_str = "n/a"
    if stats.depth_drafted:
        depth_str = " depth[" + ",".join(
            f"d{i + 1}={a}/{d}"
            for i, (a, d) in enumerate(
                zip(stats.depth_accepted, stats.depth_drafted)
            )
        ) + "]"
    else:
        depth_str = ""
    tpc = total_emits / stats.cycles if stats.cycles else 0.0
    logger.info(
        "MTP[%s] finish=%s tokens=%d cycles=%d tok/cycle=%.2f accept=%d/%d (%s)%s "
        "emits[init=%d,draft=%d,bonus=%d,verify=%d] "
        "timing[backbone=%.1fms mtp=%.1fms sample=%.1fms cache=%.1fms]",
        uid,
        finish_reason,
        total_emits,
        stats.cycles,
        tpc,
        stats.accepts,
        total_drafted,
        rate_str,
        depth_str,
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
    """Dispatch to the depth-k chain cycle or the PR-990 depth-1 legacy cycle."""
    if state.chain:
        return _run_verify_cycle_chain(gen_batch, state)
    return _run_verify_cycle_legacy(gen_batch, state)


def _run_verify_cycle_chain(gen_batch: Any, state: _MtpState) -> None:
    """One depth-k verify cycle.

    Verify ``[next_main, d1..dk]`` in a single backbone forward with
    ``n_confirmed=1``. Greedy acceptance is computed in-graph, so the whole
    cycle costs exactly ONE host sync (a ~2k-int ``tolist``); the next draft
    chain is dispatched with ``mx.async_eval`` and resolves inside the next
    cycle's sync. Emits ``m + 1`` tokens per cycle (m = accepted drafts, plus
    bonus on full accept or the verify-position correction on reject).
    """
    import time

    import mlx.core as mx

    if state.next_main is None or state.drafts is None:
        raise _MtpStepFallback("chain cycle entered without next_main / drafts")

    sampler = _resolve_sampler(gen_batch)
    procs = _proc_list(gen_batch)
    is_greedy = _is_greedy(gen_batch)
    # Adaptive depth: the chain may have drafted fewer than state.depth
    # tokens this cycle — the verify window follows the actual drafts.
    k = int(state.drafts.shape[0])
    cycle_t0 = time.perf_counter()

    inputs = mx.concatenate([state.next_main, state.drafts])  # (k+1,)

    # Token buffer per input position (mirrors PR 990 _step_backbone). Row j's
    # processor prefix is everything before that input position.
    prev_rows: List[Optional[Any]] = [None] * (k + 1)
    if procs is not None:
        buf = gen_batch._token_context[0]
        prev_rows[0] = buf.update_and_fetch(state.next_main)
        for j in range(k):
            prev_rows[j + 1] = buf.update_and_fetch(state.drafts[j : j + 1])

    # --- backbone verify forward + single host sync ---
    t0 = time.perf_counter()
    logits, hidden, gdn_states = _call_backbone(
        gen_batch.model,
        inputs[None, :],
        gen_batch.prompt_cache,
        n_confirmed=1,
    )
    rows = logits[0]  # (k+1, vocab)
    if procs is not None:
        rows = mx.stack(
            [
                _apply_processors(procs, prev_rows[j], rows[j : j + 1]).squeeze(0)
                for j in range(k + 1)
            ]
        )
    combined_lp = rows - mx.logsumexp(rows, axis=-1, keepdims=True)  # (k+1, V)

    if is_greedy:
        targets = mx.argmax(rows, axis=-1).astype(mx.int32)  # (k+1,)
        matches = (targets[:k] == state.drafts.astype(mx.int32)).astype(mx.int32)
        m_arr = mx.cumprod(matches).sum().reshape(1)
        host = mx.concatenate(
            [m_arr, targets, state.drafts.astype(mx.int32)]
        ).tolist()
        m = int(host[0])
        target_ids = host[1 : k + 2]
        draft_ids = host[k + 2 :]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        emit_last_id = target_ids[m] if m < k else target_ids[k]
        emit_last_lp = combined_lp[m if m < k else k]
    else:
        # Stochastic: batched Leviathan/Chen acceptance computed in-graph —
        # per-position ratios of the filtered target rows (p) against the
        # draft sampler's filtered rows (q), cumulative accept, residual
        # samples for every position, and the bonus draw, all resolved in
        # ONE host sync (mirrors the greedy path's sync structure).
        accept_rows = _accept_lp_for(sampler, combined_lp)  # (k+1, V)
        q_rows = mx.stack(state.draft_accept_lps)  # (k, V)
        idx = state.drafts.astype(mx.int32)[:, None]
        p_at = mx.take_along_axis(accept_rows[:k], idx, axis=-1).squeeze(-1)
        q_at = mx.take_along_axis(q_rows, idx, axis=-1).squeeze(-1)
        ratio = p_at - q_at  # (k,) log acceptance ratios
        u = mx.random.uniform(shape=(k,))
        acc = mx.logical_or(ratio >= 0, mx.log(u) < ratio)
        m_arr = mx.cumprod(acc.astype(mx.int32)).sum().reshape(1)
        # Residual distributions max(p - q, 0) per draft position. Only the
        # reject position's sample is used; computing all k keeps the cycle
        # single-sync and costs a few elementwise vocab ops on GPU.
        p_all = mx.exp(accept_rows[:k])
        res = mx.maximum(p_all - mx.exp(q_rows), 0.0)
        z = res.sum(axis=-1, keepdims=True)
        res_dist = mx.where(z > 0, res, p_all)
        res_samples = mx.random.categorical(mx.log(res_dist))  # (k,)
        bonus_tok = sampler(combined_lp[k : k + 1]).reshape(1)
        host = mx.concatenate(
            [
                m_arr.astype(mx.int32),
                state.drafts.astype(mx.int32),
                res_samples.astype(mx.int32),
                bonus_tok.astype(mx.int32),
            ]
        ).tolist()
        m = int(host[0])
        draft_ids = host[1 : k + 1]
        res_ids = host[k + 1 : 2 * k + 1]
        bonus_id = host[2 * k + 1]
        state.stats.backbone_ms += (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        if m < k:
            emit_last_id = res_ids[m]
            emit_last_lp = combined_lp[m]
        else:
            emit_last_id = bonus_id
            emit_last_lp = combined_lp[k]

    # Boundary-aligned commit: when the scheduler needs block-boundary
    # cache snapshots (hybrid models), land the committed run exactly on
    # the next block boundary whenever it falls inside the accepted
    # drafts. The emit-time snapshot capture can then observe the boundary
    # state (cache offset == emitted count), which the emit queue's
    # run-ahead otherwise makes rare. Emitting fewer verified drafts is
    # distribution-exact; the cost is at most depth-1 verified tokens once
    # per block.
    align = int(getattr(gen_batch.model, "_omlx_mtp_commit_align", 0) or 0)
    if align > 0 and m > 0:
        emitted = len(gen_batch.tokens[0])
        aligned = ((emitted // align) + 1) * align - emitted
        if 0 < aligned < m:
            m = aligned
            emit_last_id = draft_ids[m]
            emit_last_lp = combined_lp[m]

    # Clamp the accepted count to what every cache layer can roll back
    # (optional model hook — DeepSeek-V4 PoolingCache replay windows are
    # bounded). Emitting fewer verified drafts is always correct; position
    # ``m`` was itself accepted when the clamp lowers it, so its draft
    # token is a fair emit for the correction slot.
    if m < k:
        clamp = getattr(gen_batch.model, "mtp_clamp_accept", None)
        if callable(clamp):
            clamped = int(clamp(gen_batch.prompt_cache, m, k))
            if clamped < m:
                m = clamped
                emit_last_id = draft_ids[m]
                emit_last_lp = combined_lp[m]

    # --- stats ---
    state.stats.cycles += 1
    if len(state.stats.depth_drafted) < state.depth:
        pad = state.depth - len(state.stats.depth_drafted)
        state.stats.depth_drafted.extend([0] * pad)
        state.stats.depth_accepted.extend([0] * pad)
    for j in range(k):
        state.stats.depth_drafted[j] += 1
        if j < m:
            state.stats.depth_accepted[j] += 1
        else:
            break
    state.stats.accepts += m
    if m < k:
        state.stats.rejects += 1
    state.stats.sample_ms += (time.perf_counter() - t0) * 1000

    # --- commit: queue emits + cache rollback ---
    t0 = time.perf_counter()
    for j in range(m):
        state.queue.append((int(draft_ids[j]), state.draft_lps[j], "draft"))
    if m == k:
        state.queue.append((int(emit_last_id), emit_last_lp, "bonus"))
        _clear_rollback(gen_batch.prompt_cache)
    else:
        state.queue.append((int(emit_last_id), emit_last_lp, "verify"))
        if not _chain_rollback(
            gen_batch.model, gen_batch.prompt_cache, m, k, gdn_states
        ):
            if procs is not None:
                _trim_token_buffer(gen_batch, k - m)
            raise _MtpStepFallback("cache layer rejects chain rollback")
        if procs is not None:
            _trim_token_buffer(gen_batch, k - m)
    state.stats.cache_ops_ms += (time.perf_counter() - t0) * 1000

    # --- MTP-head history + next draft chain (async-dispatched) ---
    t0 = time.perf_counter()
    if not state.head_clone:
        _mtp_head_trim_to(state.mtp_cache, state.hist_offset)
    committed = mx.array(
        [int(d) for d in draft_ids[:m]] + [int(emit_last_id)], dtype=mx.uint32
    )
    next_main = committed[-1:]
    hidden_rows = hidden[:, : m + 1]
    prev_buf = None
    if procs is not None:
        prev_buf = gen_batch._token_context[0].tokens
    _chain_next_drafts(gen_batch, state, hidden_rows, committed, prev_buf)
    state.next_main = next_main
    state.stats.mtp_head_ms += (time.perf_counter() - t0) * 1000
    if state.controller is not None:
        state.controller.observe(
            k, m, (time.perf_counter() - cycle_t0) * 1000
        )


def _chain_rollback(
    model: Any,
    prompt_cache: List[Any],
    accepted: int,
    num_drafts: int,
    gdn_states: Optional[list] = None,
) -> bool:
    """Roll the backbone cache back to ``accepted`` drafts after a chain verify.

    mlx-vlm path (``gdn_states`` populated): delegate to the stock
    ``rollback_speculative_cache``, which natively supports partial accepts —
    it keeps ``accepted + 1`` positions of the ``num_drafts + 1``-token
    verify window and replays the accepted prefix through the captured GDN
    states. mlx-lm path: ``mtp_partial_rollback`` (qwen35_model patch).
    """
    if gdn_states is not None and hasattr(model, "rollback_speculative_cache"):
        try:
            model.rollback_speculative_cache(
                prompt_cache, gdn_states, accepted, num_drafts + 1
            )
            return True
        except Exception as exc:
            logger.debug("rollback_speculative_cache failed: %s", exc)
            return False
    rollback = getattr(model, "mtp_partial_rollback", None)
    if callable(rollback):
        try:
            return bool(rollback(prompt_cache, accepted, num_drafts))
        except Exception as exc:
            logger.debug("mtp_partial_rollback failed: %s", exc)
            return False
    if accepted == 0 and num_drafts == 1:
        return _restore_or_trim_caches(prompt_cache)
    return False


def _run_verify_cycle_legacy(gen_batch: Any, state: _MtpState) -> None:
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
