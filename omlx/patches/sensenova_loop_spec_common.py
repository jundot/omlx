# SPDX-License-Identifier: Apache-2.0
"""SenseNova-Loop generic speculative multi-visit adaptations.

The loop9b 56-slot execution plan visits the linear (SSM) layers inside the
Loop interval ``count`` times per block (``tuple(range(start, end)) * count``).
Standard speculative caches assume single-visit slots, so:

  * ``Qwen3_5BatchInvariantForward.verify`` needs a speculation window scaled
    by ``count`` (``_patch_loop_verify_capacity``);
  * ``ArraysCache.validate_speculation`` needs its strict length equality
    relaxed (multi-visit SSM slots record longer than single-visit FA slots),
    and ``commit_speculation`` needs the accepted-token prefix translated into
    the multi-visit slot's cumulative state index (``_patch_loop_validate_speculation``).

These are model-type-gated and drafter-independent: both DFlash2 and VLM MTP
install them. Kept in a shared module so neither drafter depends on the other.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

def _patch_loop_verify_capacity() -> None:
    """Make the exact verifier's speculative window fit the loop plan.

    ``Qwen3_5BatchInvariantForward.verify`` calls ``start_speculative_cache(
    cache or [], inputs.shape[1])``, so every ArraysCache slot gets
    ``history_capacity = inputs.shape[1]``. For SenseNova-Loop the linear
    (SSM) layers inside the Loop interval are visited ``count`` times per
    block (``tuple(range(start, end)) * count`` in ``execution_plan``), so one
    cache slot receives up to ``count`` window updates per verify block. The
    accumulated ``_recorded_length(index)`` then exceeds ``inputs.shape[1]``
    and ``record_speculative_window`` raises "Window update exceeds the cache
    history capacity".

    Adapt: when the target is SenseNova-Loop, multiply the speculation
    ``length`` by ``count`` (``loop_settings["eval_count"]``) so the bounded
    timeline fits every multi-visit slot. Non-loop models are unchanged.
    """
    try:
        from mlx_vlm.models.qwen3_5.speculative_verifier import (
            Qwen3_5BatchInvariantForward,
        )
        from mlx_vlm.speculative.cache_state import start_speculative_cache
    except Exception as e:  # pragma: no cover
        logger.debug("verifier verify capacity patch skipped: %s", e)
        return

    if getattr(Qwen3_5BatchInvariantForward, "_sl_verify_capacity_patched", False):
        return

    orig_verify = Qwen3_5BatchInvariantForward.verify

    def _loop_eval_count(language_model) -> int:
        config = getattr(language_model, "config", None)
        loop = getattr(config, "loop_settings", None)
        if isinstance(loop, dict):
            count = loop.get("eval_count")
            if isinstance(count, int) and count >= 1:
                return count
        return 1

    def _set_multi_visit(language_model, cache, count):
        """Annotate every ArraysCache with the per-slot visit multiplicity.

        The 56-slot execution plan repeats ``tuple(range(start, end)) * count``
        so the slots inside the Loop interval are visited ``count`` times.
        ``commit_speculation`` needs this map to translate the accepted-token
        prefix into the multi-visit slot's cumulative state index.
        """
        if count <= 1 or cache is None:
            return
        plan = getattr(getattr(language_model, "model", None),
                       "_execution_plan", None)
        if plan is None:
            plan = getattr(language_model, "_execution_plan", None)
        if plan is None:
            return
        from collections import Counter
        visits = Counter(plan)
        from mlx_vlm.models.cache import ArraysCache
        for c in cache:
            if isinstance(c, ArraysCache):
                mv = {i: v for i, v in enumerate(visits.values()) if v > 1}
                # visits is keyed by physical index; execution_plan maps slot
                # -> physical, and physical -> visit count is uniform across
                # the repeated range. Build slot->visit from the plan itself.
                mv = {}
                for si, pi in enumerate(plan):
                    if visits[pi] > 1:
                        mv[si] = visits[pi]
                c._sl_multi_visit = mv
                logger.warning(
                    "SL-VERIFY-CAP: multi-visit slots=%d count=%d",
                    len(mv), count,
                )

    def _patched_verify(self, language_model, inputs, *, cache=None, **kwargs):
        count = _loop_eval_count(language_model)
        length = inputs.shape[1] * count if count > 1 else inputs.shape[1]
        if count > 1:
            logger.warning(
                "SL-VERIFY-CAP: loop count=%d length=%d (orig=%d)",
                count, length, inputs.shape[1],
            )
            _set_multi_visit(language_model, cache or [], count)
        transaction = start_speculative_cache(cache or [], length)
        try:
            output = self(language_model, inputs, cache=cache, **kwargs)
            output.gdn_states = transaction
            return output
        except BaseException:
            transaction.abort()
            raise

    Qwen3_5BatchInvariantForward.verify = _patched_verify
    setattr(Qwen3_5BatchInvariantForward, "_sl_verify_capacity_patched", True)
    logger.debug("Qwen3.5 verifier verify capacity patch installed")


# ---------------------------------------------------------------------------
# 4c. ArraysCache.validate_speculation: accept multi-visit loop slots.
# ---------------------------------------------------------------------------


def _patch_loop_validate_speculation() -> None:
    """Relax the speculative commit validation for multi-visit loop slots.

    Upstream ``ArraysCache.validate_speculation`` requires every recorded
    slot's ``_recorded_length`` to equal ``transaction["length"]`` exactly.
    SenseNova-Loop's 56-slot execution plan visits linear (SSM) layers inside
    the Loop interval ``count`` times per block, so those slots record
    ``count * block_len`` while full-attention slots record ``block_len``.
    A single ``total`` cannot equal both, so the strict check fails for the
    multi-visit slots (SSM record is longer than ``total``) or for the
    single-visit slots (FA record is shorter).

    Adapt: relax the equality to a lower bound. ``_select_speculative_window``
    only indexes ``source[:, keep : keep + width]`` with ``keep`` derived from
    ``lengths`` (accepted tokens), so a slot merely needs ``_recorded_length``
    >= ``max(lengths)`` (the committed prefix must exist). Longer records
    (multi-visit SSM slots) are valid — their extra history is simply not
    selected by the commit prefix. Non-loop caches keep the strict check.
    """
    try:
        from mlx_vlm.models.cache import ArraysCache
    except Exception as e:  # pragma: no cover
        logger.debug("validate_speculation patch skipped: %s", e)
        return

    if getattr(ArraysCache, "_sl_validate_patched", False):
        return

    orig_validate = ArraysCache.validate_speculation

    def _patched_validate(self, lengths, generation=None):
        transaction = self._speculation
        if transaction is None:
            raise RuntimeError("No speculative cache transaction is active.")
        if generation is not None and generation != transaction["generation"]:
            raise RuntimeError("Attempted to commit a stale cache transaction.")
        from mlx_vlm.models.cache import _speculative_lengths

        total = transaction["length"]
        norm = _speculative_lengths(lengths, self.batch_size, total)
        records = transaction["records"]
        initial_state = transaction["initial_state"]
        missing = [
            index
            for index, (initial, current) in enumerate(
                zip(initial_state, self.cache)
            )
            if initial is not current and index not in records
        ]
        if missing:
            raise RuntimeError(
                "Speculative cache state changed without temporal records for "
                f"slots {missing}."
            )
        # Multi-visit loop slots may record longer than `total`; accept any
        # slot whose recorded length covers the committed prefix.
        need = max(norm) if norm else 0
        for index in records:
            rec = self._recorded_length(index)
            if rec < need:
                raise RuntimeError(
                    "Temporal cache history does not cover the full window."
                )
        return transaction, norm

    ArraysCache.validate_speculation = _patched_validate
    setattr(ArraysCache, "_sl_validate_patched", True)
    logger.debug("ArraysCache.validate_speculation relaxed for loop slots")

    # Also patch commit_speculation so multi-visit slots translate the
    # accepted-token prefix into the cumulative state index. The stock
    # ``_select_speculative_states`` uses ``keep`` (accepted tokens) to index
    # ``intermediate`` directly, but a multi-visit slot's ``intermediate``
    # holds ``count`` concatenated segments, so the correct index is
    # ``keep * count``. Without this, committing a partial accept picks the
    # wrong state (or hits a None intermediate for a single-step segment).
    if getattr(ArraysCache, "_sl_commit_patched", False):
        return
    orig_commit = ArraysCache.commit_speculation

    def _patched_commit(self, lengths, generation=None):
        transaction = self._speculation
        if transaction is None:
            return orig_commit(self, lengths, generation)
        mv = getattr(self, "_sl_multi_visit", None)
        if not mv:
            return orig_commit(self, lengths, generation)
        transaction, norm = self.validate_speculation(lengths, generation)
        total = transaction["length"]
        # The speculation length was scaled by ``count`` for the multi-visit
        # slots. Single-visit slots (outside the repeated Loop range) still
        # recorded only ``block_len`` steps, so their ``keep == total``
        # branch must use the un-scaled ``total // count``. Derive the
        # single-visit total once.
        counts = sorted(set(mv.values())) if mv else [1]
        base_total = total // max(counts) if counts else total
        initial_state = transaction["initial_state"]
        raw = norm
        for index, record in transaction["records"].items():
            count = mv.get(index)
            if count and count > 1:
                # Multi-visit SSM slot: intermediate holds ``count``
                # concatenated segments (one per visit, each of length
                # ``block_len``). ``raw`` (accepted tokens) is a global
                # prefix <= block_len, all inside the first visit. So a
                # partial accept indexes ``intermediate[:, raw - 1]``
                # directly; a full accept (raw == block_len) must return
                # ``final``, which requires ``keep == slot_total``.
                block_len = total // count
                sel = [total if k >= block_len else k for k in raw]
                slot_total = total
            else:
                sel = raw
                slot_total = base_total
            if record[0] == "states":
                self.cache[index] = self._select_speculative_states(
                    record, initial_state[index], sel, slot_total,
                )
            else:
                self.cache[index] = self._select_speculative_window(record, sel)
        self._restore_speculative_metadata(raw, transaction["initial_metadata"])
        self._speculation = None

    ArraysCache.commit_speculation = _patched_commit
    setattr(ArraysCache, "_sl_commit_patched", True)
    logger.debug("ArraysCache.commit_speculation handles multi-visit slots")




# ---------------------------------------------------------------------------
# Loop LanguageModel.__call__: bypass speculative_verify ValueError.
# ---------------------------------------------------------------------------
def _patch_loop_language_model_call(guard_flag: str) -> None:
    """Bypass loop9b ``LanguageModel.__call__`` speculative_verify rejection.

    Native ``sensenova_loop.language.LanguageModel.__call__`` raises
    ValueError when ``speculative_verify=True``. The MTP/DFlash verify path
    needs spec_verify (via qwen3_5 ``_EXACT_SPECULATIVE_VERIFIER.verify``),
    so it must be bypassed.

    Adapt: when ``speculative_verify=True`` (or ``capture_layer_ids``), call
    the qwen3_5 base ``Qwen3_5LanguageModel.__call__`` (natively supports
    spec_verify + return_hidden + capture_layer_ids) and force
    ``return_hidden=True`` to append the 5th feature (final_postnorm).
    Otherwise fall through to native loop9b logic (incl. ValueError).

    ``guard_flag`` is the class attribute that marks this patch as installed
    (idempotence). DFlash2 and MTP install the same logical patch on the
    same ``__call__`` slot but with independent flags so each can be applied
    without clobbering the other's application order.
    """
    try:
        from mlx_vlm.models.sensenova_loop import language as sl_lang
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except Exception as e:  # pragma: no cover
        logger.debug("loop LanguageModel call patch skipped: %s", e)
        return

    cls = sl_lang.LanguageModel
    if getattr(cls, guard_flag, False):
        return

    orig_call = cls.__call__

    def _patched_call(self, *args, **kwargs):
        # Both DFlash2 (speculative_verify) and the VLM DFlash
        # verify_forward path (capture_layer_ids w/o speculative_verify) need
        # the post-norm final hidden appended as the 5th feature
        # (final_postnorm). The qwen3_5 base appends it to hidden_sink only
        # when return_hidden=True, so force it on for any capture request.
        needs_capture = kwargs.get("speculative_verify", False) or bool(
            kwargs.get("capture_layer_ids")
        )
        if needs_capture:
            # Delegate to the qwen3_5 base, which natively handles
            # spec_verify + return_hidden + capture. Force return_hidden so
            # the post-norm final hidden becomes the 5th feature.
            base = q35_lang.LanguageModel.__call__
            kwargs.setdefault("return_hidden", True)
            # Force speculative_verify so the exact verifier's verify()
            # runs and returns a SpeculativeCacheTransaction (gdn_states).
            # _patch_loop_verify_capacity enlarges start_speculative_cache's
            # length by the loop eval_count so the 56-slot multi-visit SSM
            # window fits the bounded timeline (no "Window update exceeds the
            # cache history capacity"). The transaction enables the native
            # mlx-vlm rollback path (gdn_states not None).
            kwargs["speculative_verify"] = True
            logger.debug(
                "SL-SPECVERIFY: return_hidden=%s capture=%s spec_verify=%s",
                kwargs.get("return_hidden"),
                kwargs.get("capture_layer_ids"),
                kwargs.get("speculative_verify"),
            )
            return base(self, *args, **kwargs)
        return orig_call(self, *args, **kwargs)

    cls.__call__ = _patched_call
    setattr(cls, guard_flag, True)
    logger.debug("loop9b LanguageModel.__call__ speculative_verify bypassed (%s)", guard_flag)
