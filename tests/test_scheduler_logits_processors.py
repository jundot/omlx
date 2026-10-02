# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the logits_processors row contract (#934, #1823).

mlx-lm normalises every inserted row to a tuple (#1826) and reindexes rows on
``GenerationBatch.filter`` unconditionally (#1772), so empty and None slots no
longer crash ``GenerationBatch._step``. oMLX still pins:

1. **Caller-side**: the insert call site passes a per-row list, which the uid
   registry records as the row the request must run.
2. **Row realignment**: ``_patched_generation_batch_step`` rebuilds sampler
   and processor rows from the registry before the grammar-accept loop and
   the original step read them by position.
3. **Pattern matcher**: ``CACHE_CORRUPTION_PATTERNS`` includes
   ``"'NoneType' object is not iterable"`` so the scheduler recovers
   gracefully if a None slot ever sneaks through.
"""

from __future__ import annotations

import pytest

from omlx.exceptions import CACHE_CORRUPTION_PATTERNS, is_cache_corruption_error


class TestLogitsProcessorsCallShape:
    """Pin the caller-side contract: per-row list, never None."""

    def test_scheduler_source_uses_list_wrapper(self):
        """The insert call site must wrap logits_processors as a list.

        Source-level assertion; cheaper than spinning up a real engine.
        Catches accidental regressions where someone changes the
        ``per_row_lps = list(logits_processors) if logits_processors else []``
        line back to a raw passthrough.
        """
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        # The variable name and the wrapping pattern.
        assert (
            "per_row_lps = list(logits_processors) if logits_processors else []"
            in scheduler_src
        ), (
            "scheduler.py must wrap per-request logits_processors as a "
            "list before passing to BatchGenerator.insert. See #934."
        )
        assert "logits_processors=[per_row_lps]" in scheduler_src, (
            "scheduler.py must pass logits_processors=[per_row_lps] "
            "(per-row list, never None) to BatchGenerator.insert. See #934."
        )


class _RecordingMatcher:
    """xgrammar.GrammarMatcher stand-in with an explicit allow-set."""

    def __init__(self, allowed=None):
        self.allowed = allowed
        self.accepted = []

    def accept_token(self, token_id):
        if self.allowed is not None and token_id not in self.allowed:
            return False
        self.accepted.append(token_id)
        return True

    def is_terminated(self):
        return False


def _bare_grammar_processor(*, pending, allowed=None, vocab_size=64):
    """Build a GrammarConstraintProcessor via __new__, without xgrammar.

    ``__init__`` is the only part of the class that imports xgrammar, and CI
    does not install it (``.github/workflows/ci.yml`` installs ``.[mcp]``
    only), so the row-advance tests fill the instance state directly.
    """
    import numpy as np

    from omlx.api.grammar import GrammarConstraintProcessor

    proc = GrammarConstraintProcessor.__new__(GrammarConstraintProcessor)
    proc._matcher = _RecordingMatcher(allowed)
    proc._vocab_size = vocab_size
    proc._bitmask = np.full((1, (vocab_size + 31) // 32), -1, dtype=np.int32)
    proc._terminated = False
    proc._pending = pending
    return proc


def _grammar_batch(logits_processors, next_tokens):
    """Minimal GenerationBatch stand-in for the row-advance chokepoint."""
    from types import SimpleNamespace

    return SimpleNamespace(
        model=object(),
        uids=list(range(len(logits_processors))),
        logits_processors=logits_processors,
        _next_tokens=next_tokens,
    )


class TestGrammarRowAdvance:
    """Pin the accept point: top of the next step, once per sampled token.

    Grammar rows used to be advanced immediately after the wrapped step
    dispatched the sampled tokens, which forced ``mx.eval`` on them before
    any of the host work that follows a step could overlap with the GPU.
    ``_omlx_advance_grammar_rows`` moves the read to the top of the next
    step, where the ids are needed anyway; these tests pin the placement and
    the ``pending`` bookkeeping that makes it exact.
    """

    def test_sampled_token_is_accepted_at_the_next_step(self):
        import mlx.core as mx

        import omlx.scheduler as scheduler

        proc = _bare_grammar_processor(pending=True)
        batch = _grammar_batch([[proc]], mx.array([7], dtype=mx.uint32))

        scheduler._omlx_advance_grammar_rows(batch)

        assert proc._matcher.accepted == [7]
        assert proc.pending is False

    def test_priming_step_does_not_accept_the_prompt_token(self):
        """``GenerationBatch.__init__`` steps once with the prompt's last
        token in ``_next_tokens``; nothing was sampled yet, so the matcher
        must not see it. The guard is the processor's own ``pending`` flag,
        which ``extend()`` carries along with the row."""
        import mlx.core as mx

        import omlx.scheduler as scheduler

        proc = _bare_grammar_processor(pending=False)
        batch = _grammar_batch([[proc]], mx.array([7], dtype=mx.uint32))

        scheduler._omlx_advance_grammar_rows(batch)

        assert proc._matcher.accepted == []

    def test_only_grammar_rows_are_read(self):
        """A row without a grammar processor must not be touched, and a
        batch with no pending grammar row must not force an eval at all."""
        import mlx.core as mx

        import omlx.scheduler as scheduler

        def identity_processor(token_context, logits):
            return logits

        proc = _bare_grammar_processor(pending=True)
        batch = _grammar_batch(
            [[identity_processor], [proc]], mx.array([3, 9], dtype=mx.uint32)
        )

        scheduler._omlx_advance_grammar_rows(batch)

        assert proc._matcher.accepted == [9]

    def test_none_next_tokens_is_skipped(self):
        """``filter([])`` leaves ``_next_tokens`` as None (mlx-lm)."""
        import omlx.scheduler as scheduler

        proc = _bare_grammar_processor(pending=True)
        batch = _grammar_batch([[proc]], None)

        scheduler._omlx_advance_grammar_rows(batch)

        assert proc._matcher.accepted == []

    def test_row_count_mismatch_is_skipped(self):
        """Same defensive criterion as the row realignment: a token array
        that disagrees with ``uids`` cannot be attributed to rows."""
        import mlx.core as mx

        import omlx.scheduler as scheduler

        proc = _bare_grammar_processor(pending=True)
        batch = _grammar_batch([[proc]], mx.array([1, 2], dtype=mx.uint32))

        scheduler._omlx_advance_grammar_rows(batch)

        assert proc._matcher.accepted == []

    def test_advance_runs_before_the_wrapped_step(self, monkeypatch):
        """The accept must happen while ``_next_tokens`` still holds the
        previous samples — the original step promotes them to the model
        input on its first line."""
        import mlx.core as mx

        import omlx.scheduler as scheduler

        order = []

        def fake_original_step(self):
            order.append("step")
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        proc = _bare_grammar_processor(pending=True)
        original_accept = proc.accept_token

        def recording_accept(token_id):
            order.append("accept")
            original_accept(token_id)

        proc.accept_token = recording_accept

        batch = _grammar_batch([[proc]], mx.array([5], dtype=mx.uint32))
        assert scheduler._patched_generation_batch_step(batch) == "stepped"

        assert order == ["accept", "step"]
        assert proc._matcher.accepted == [5]

    def test_scheduler_source_orders_advance_between_realign_and_step(self):
        """Source-level guard on the call order inside the patched step.

        The advance reads ``logits_processors[e]`` as the row state for
        ``uids[e]``, which only holds after the realignment; and it must run
        before the original step, which consumes ``_next_tokens``.
        """
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        realign = scheduler_src.index("    _omlx_realign_generation_batch_rows(self)")
        advance = scheduler_src.index("    _omlx_advance_grammar_rows(self)")
        step = scheduler_src.index("return _original_generation_batch_step(self)")
        assert realign < advance < step, (
            "_omlx_advance_grammar_rows must run after the row realignment "
            "(it indexes logits_processors by row) and before the wrapped "
            "step (which promotes _next_tokens to the model input)."
        )


class TestCorruptionPatternRecovery:
    """Pin the recovery contract: 'not iterable' is a known corruption."""

    def test_not_iterable_pattern_in_list(self):
        assert "'NoneType' object is not iterable" in CACHE_CORRUPTION_PATTERNS

    def test_not_iterable_typeerror_recognized(self):
        """Raising the exact error mlx-lm produces should match recovery."""
        err = TypeError("'NoneType' object is not iterable")
        assert is_cache_corruption_error(err) is True

    def test_not_iterable_with_traceback_text(self):
        """Match should work even when the message has extra context
        (e.g., when re-raised with formatting)."""
        err = TypeError("in GenerationBatch._step: 'NoneType' object is not iterable")
        assert is_cache_corruption_error(err) is True


class TestRowRealignment:
    """Pin the uid-registry realignment (#1823).

    Stale or offset row slots left by batch extend/filter/split shift every
    row after them, so a request silently runs another request's — or no —
    sampler and logits processors. The #1799 normalisation makes the step
    crash-safe but cannot restore alignment; the chokepoint must realign
    the positional lists from the per-uid registry."""

    def test_patched_step_realigns_offset_rows_from_registry(self, monkeypatch):
        """The #1823 probe scenario: three processor slots for two uids.

        A stale leading slot (left by a finished request) offsets every row:
        the constrained request's processors sit in a slot nothing reads,
        and its row runs an empty one. Red before the registry realignment
        (the wrapped step sees the offset rows: uid 2 runs no processors);
        green after (uid 2's row runs its own sampler and processors).
        """
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())

        captured = {}

        def fake_original_step(self):
            captured["logits_processors"] = list(self.logits_processors)
            captured["samplers"] = list(self.samplers)
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        def budget_processor(token_context, logits):
            return logits

        def grammar_processor(token_context, logits):
            return logits

        sampler_uid2 = object()

        class FakeModel:
            pass

        class FakeBatch:
            model = FakeModel()
            uids = [1, 2]
            # Stale leading slot from a finished request: 3 slots, 2 uids.
            logits_processors = [[], [], [budget_processor, grammar_processor]]
            samplers = [None, None, sampler_uid2]
            _next_tokens = None

        # What the insert sites record: uid 1 is a plain request, uid 2 is
        # the constrained one (grammar + thinking budget).
        scheduler._register_uid_rows(FakeBatch.model, [1], [None], [[]])
        scheduler._register_uid_rows(
            FakeBatch.model,
            [2],
            [sampler_uid2],
            [[budget_processor, grammar_processor]],
        )

        batch = FakeBatch()
        result = scheduler._patched_generation_batch_step(batch)

        assert result == "stepped"
        # uid 2's row must run ITS processors and sampler, not the offset ones.
        assert captured["logits_processors"][1] == [
            budget_processor,
            grammar_processor,
        ]
        assert captured["samplers"][1] is sampler_uid2
        # Alignment restored: exactly one slot per uid.
        assert len(batch.logits_processors) == len(batch.uids)
        assert len(batch.samplers) == len(batch.uids)

    def test_registry_is_bounded(self):
        """A missed cleanup must never grow the registry unbounded."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model = object()
        try:
            for uid in range(scheduler._UID_ROW_REGISTRY_MAX + 100):
                scheduler._register_uid_rows(model, [uid], [None], [[]])
            assert len(registry) == scheduler._UID_ROW_REGISTRY_MAX
            # Oldest entries evicted first.
            assert (id(model), 0) not in registry
            assert (id(model), scheduler._UID_ROW_REGISTRY_MAX + 99) in registry
        finally:
            scheduler._uid_row_registry = original

    def test_scheduler_source_registers_rows_at_insert(self):
        """Source-level guard: both insert sites must record what each uid
        is supposed to run, or the chokepoint has nothing to realign from."""
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        assert scheduler_src.count("_register_uid_rows(self.model, uids") >= 2, (
            "every batch_generator.insert call site must register the "
            "per-uid sampler and logits processors; the step chokepoint "
            "realigns rows from that registry. See #1823."
        )

    def test_unregistered_uid_keeps_current_row_and_short_slots_pad(self, monkeypatch):
        """Realignment must not invent state: a uid missing from the registry
        keeps its current row, and missing trailing slots pad to empty
        instead of raising."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())

        captured = {}

        def fake_original_step(self):
            captured["logits_processors"] = list(self.logits_processors)
            captured["samplers"] = list(self.samplers)
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        def legacy_processor(token_context, legacy_logits):
            return legacy_logits

        class FakeModel:
            pass

        class FakeBatch:
            model = FakeModel()
            uids = [7, 8]
            # uid 7 is not registered but carries a live row: keep it.
            # uid 8 has no slot at all (shorter list): pad to [].
            logits_processors = [[legacy_processor]]
            samplers = [None]
            _next_tokens = None

        batch = FakeBatch()
        result = scheduler._patched_generation_batch_step(batch)

        assert result == "stepped"
        assert captured["logits_processors"][0] == [legacy_processor]
        assert captured["logits_processors"][1] == []
        assert len(batch.samplers) == len(batch.uids)

    def test_same_uid_on_two_models_does_not_cross_contaminate(self, monkeypatch):
        """mlx-lm numbers uids per BatchGenerator instance, so two engines
        serving concurrently produce colliding uid values. The registry must
        key by model so engine A's realignment never installs engine B's
        sampler and processors."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())

        captured = {}

        def fake_original_step(self):
            captured[id(self.model)] = list(self.logits_processors)
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        def qwen_processor(token_context, logits):
            return logits

        def gemma_processor(token_context, logits):
            return logits

        class FakeModel:
            pass

        model_a, model_b = FakeModel(), FakeModel()
        # SAME uid value on both engines, different processors.
        scheduler._register_uid_rows(model_a, [7], [None], [[qwen_processor]])
        scheduler._register_uid_rows(model_b, [7], [None], [[gemma_processor]])

        def make_batch(model):
            class FakeBatch:
                pass

            b = FakeBatch()
            b.model = model
            b.uids = [7]
            b.logits_processors = [[]]
            b.samplers = [None]
            b._next_tokens = None
            return b

        scheduler._patched_generation_batch_step(make_batch(model_a))
        scheduler._patched_generation_batch_step(make_batch(model_b))

        assert captured[id(model_a)][0] == [qwen_processor]
        assert captured[id(model_b)][0] == [gemma_processor]

    def test_unregister_drops_the_row(self):
        """Completion cleanup must release the row so heavy processors are
        not pinned until FIFO eviction."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model = object()
        try:
            scheduler._register_uid_rows(model, [3], [None], [[object()]])
            assert (id(model), 3) in registry
            scheduler._unregister_uid_row(model, 3)
            assert (id(model), 3) not in registry
            # Unregistering twice (or an unknown uid) is a no-op.
            scheduler._unregister_uid_row(model, 3)
        finally:
            scheduler._uid_row_registry = original

    def test_realigned_rows_rebuilds_in_uid_order(self):
        """Direct unit coverage of the pure rebuild: offset slots are
        replaced by the registered rows and the drift flag is set."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model = object()
        proc = object()
        sampler = object()
        try:
            scheduler._register_uid_rows(model, [1], [None], [[]])
            scheduler._register_uid_rows(model, [2], [sampler], [[proc]])
            # The #1823 probe shape: a stale leading slot, 3 slots for 2 uids.
            samplers, lps, drift = scheduler._realigned_rows(
                model, [1, 2], [None, None, sampler], [[], [], [proc]]
            )
            assert drift is True
            assert samplers == [None, sampler]
            assert lps == [[], [proc]]
        finally:
            scheduler._uid_row_registry = original

    def test_realigned_rows_steady_state_reports_no_drift(self):
        """Feeding the rebuilt lists back in (the post-realignment state)
        must report no drift: the identity fast path short-circuits."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model = object()
        proc = object()
        try:
            scheduler._register_uid_rows(model, [1], [None], [[proc]])
            samplers, lps, drift = scheduler._realigned_rows(model, [1], [], [])
            assert drift is True  # short slots on the first pass
            samplers, lps, drift = scheduler._realigned_rows(model, [1], samplers, lps)
            assert drift is False
            assert lps == [[proc]]
        finally:
            scheduler._uid_row_registry = original

    def test_realigned_rows_reports_sampler_only_drift(self):
        """A corrected sampler-only mismatch is still row-state drift."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model = object()
        expected_sampler = object()
        wrong_sampler = object()
        try:
            scheduler._register_uid_rows(model, [1], [expected_sampler], [[]])
            samplers, lps, drift = scheduler._realigned_rows(
                model, [1], [wrong_sampler], [[]]
            )
            assert drift is True
            assert samplers == [expected_sampler]
            assert lps == [[]]
        finally:
            scheduler._uid_row_registry = original

    def test_realign_hook_rebuilds_rows_for_non_step_callers(self, monkeypatch):
        """Native MTP calls GenerationBatch.next before _step, so the shared
        hook must realign rows independently of the patched step wrapper."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())

        model = object()
        expected_sampler = object()
        scheduler._register_uid_rows(model, [1], [expected_sampler], [[]])

        batch = type("FakeBatch", (), {})()
        batch.model = model
        batch.uids = [1]
        batch.logits_processors = [[], []]
        batch.samplers = [None, expected_sampler]

        scheduler._omlx_realign_generation_batch_rows(batch)

        assert batch.samplers == [expected_sampler]
        assert batch.logits_processors == [[]]

    def test_model_scoped_clear_drops_only_that_model(self):
        """Reset/recovery/shutdown release by model: every row of the reset
        engine goes, every row of the other engine stays."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        registry = OrderedDict()
        original = scheduler._uid_row_registry
        scheduler._uid_row_registry = registry
        model_a, model_b = object(), object()
        try:
            scheduler._register_uid_rows(model_a, [0, 1], [None, None], [[], []])
            scheduler._register_uid_rows(model_b, [0], [None], [[]])
            scheduler._unregister_uid_rows_for_model(model_a)
            assert (id(model_a), 0) not in registry
            assert (id(model_a), 1) not in registry
            assert (id(model_b), 0) in registry
            # Clearing an unknown model is a no-op.
            scheduler._unregister_uid_rows_for_model(object())
            assert (id(model_b), 0) in registry
        finally:
            scheduler._uid_row_registry = original

    def test_offset_rows_pass_through_without_registry(self, monkeypatch):
        """The pre-fix behavior, pinned through the fallback path: with
        nothing registered, the chokepoint cannot restore alignment, so the
        #1823 probe shape (three slots for two uids) reaches the step with
        uid 2 running no processors. This is the exact silent failure the
        registry realignment corrects in
        ``test_patched_step_realigns_offset_rows_from_registry``."""
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())

        captured = {}

        def fake_original_step(self):
            captured["logits_processors"] = list(self.logits_processors)
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        def grammar_processor(token_context, logits):
            return logits

        class FakeModel:
            pass

        class FakeBatch:
            model = FakeModel()
            uids = [1, 2]
            # Stale leading slot: uid 2's processors sit in slot 2, which the
            # two-uid loop never reads.
            logits_processors = [[], [], [grammar_processor]]
            samplers = [None, None, object()]
            _next_tokens = None

        scheduler._patched_generation_batch_step(FakeBatch())

        # Without registry rows the constrained request silently decodes
        # unconstrained — the pre-#1824 behavior.
        assert captured["logits_processors"][1] == []

    def test_drift_warning_is_rate_limited(self, monkeypatch, caplog):
        """One drift correction per window logs at WARNING; the rest go to
        DEBUG so a pathological merge pattern cannot flood the logs."""
        import logging
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())
        monkeypatch.setattr(scheduler, "_uid_row_drift_last_warning", float("-inf"))
        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", lambda self: "stepped"
        )

        def make_misaligned_batch():
            class FakeModel:
                pass

            class FakeBatch:
                pass

            batch = FakeBatch()
            batch.model = FakeModel()
            batch.uids = [1]
            # One stale slot too many: drift on every call.
            batch.logits_processors = [[], []]
            batch.samplers = [None, None]
            batch._next_tokens = None
            return batch

        with caplog.at_level(logging.DEBUG, logger=scheduler.logger.name):
            scheduler._patched_generation_batch_step(make_misaligned_batch())
            scheduler._patched_generation_batch_step(make_misaligned_batch())

        realign_levels = [
            record.levelno
            for record in caplog.records
            if "Realigned generation-batch row state" in record.getMessage()
        ]
        assert realign_levels == [logging.WARNING, logging.DEBUG]

    def test_tuple_row_matching_registry_list_is_not_drift(self, monkeypatch, caplog):
        """mlx-lm stores each row as a tuple; the registry keeps a list."""
        import logging
        from collections import OrderedDict

        import omlx.scheduler as scheduler

        monkeypatch.setattr(scheduler, "_uid_row_registry", OrderedDict())
        monkeypatch.setattr(scheduler, "_uid_row_drift_last_warning", float("-inf"))
        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", lambda self: "stepped"
        )

        def penalty_processor(token_context, logits):
            return logits

        class FakeModel:
            pass

        class FakeBatch:
            pass

        sampler = object()
        batch = FakeBatch()
        batch.model = FakeModel()
        batch.uids = [1]
        batch.logits_processors = [(penalty_processor,)]
        batch.samplers = [sampler]
        batch._next_tokens = None
        scheduler._register_uid_rows(batch.model, [1], [sampler], [[penalty_processor]])

        with caplog.at_level(logging.DEBUG, logger=scheduler.logger.name):
            scheduler._patched_generation_batch_step(batch)

        assert not [
            record
            for record in caplog.records
            if "Realigned generation-batch row state" in record.getMessage()
        ]


class TestRegistryCleanupPaths:
    """Every path that retires a uid — or the whole generator — must release
    its registry rows. A finished, aborted, or failed request that stays
    registered pins its (possibly heavy, stateful) processors until the FIFO
    backstop, and entries surviving a generator reset or engine unload are
    exactly the residue an ``id(model)`` recycle could later match.

    Structural AST checks: cheaper than spinning up a Scheduler per path,
    and immune to formatting churn (unlike substring counting)."""

    PER_UID_RELEASE_PATHS = [
        "_drain_pending_async_removes",
        "_do_abort_request",
        "_cleanup_finished",
    ]
    MODEL_WIDE_RELEASE_PATHS = [
        "fail_all_requests",
        "_recover_from_cache_error",
        "_recover_from_generation_overflow_error",
        "reset",
        "shutdown",
    ]

    @staticmethod
    def _called_names(func_name: str) -> set:
        import ast
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
            ):
                return {
                    (
                        call.func.id
                        if isinstance(call.func, ast.Name)
                        else getattr(call.func, "attr", None)
                    )
                    for call in ast.walk(node)
                    if isinstance(call, ast.Call)
                }
        raise AssertionError(f"{func_name} not found in scheduler.py")

    @pytest.mark.parametrize("func_name", PER_UID_RELEASE_PATHS)
    def test_per_uid_paths_release_the_row(self, func_name):
        assert "_unregister_uid_row" in self._called_names(func_name), (
            f"{func_name} retires a uid from the batch but does not release "
            "its registry row; the processors stay pinned until the FIFO "
            "backstop. See #1823."
        )

    @pytest.mark.parametrize("func_name", MODEL_WIDE_RELEASE_PATHS)
    def test_model_wide_paths_release_every_row(self, func_name):
        assert "_unregister_uid_rows_for_model" in self._called_names(func_name), (
            f"{func_name} clears the uid maps (or retires the generator) "
            "wholesale but leaves the registry rows behind; release by model "
            "so nothing survives a reset, recovery, or shutdown. See #1823."
        )
