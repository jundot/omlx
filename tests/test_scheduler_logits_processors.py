# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the logits_processors call shape contract (#934).

mlx-lm's ``GenerationBatch._step`` does
``for p in self.logits_processors[e]`` whenever
``any(self.logits_processors)`` is True. If any per-row slot is ``None``
(instead of an empty list), this raises
``TypeError: 'NoneType' object is not iterable``.

This crash escapes omlx's recovery path if ``CACHE_CORRUPTION_PATTERNS``
doesn't match it, and presents to users as a request hang. See
``vllm-mlx-patched`` commit ``8d4052b`` for the same root cause in a
sibling project.

The caller-side wrap is necessary but **not sufficient**: on a heterogeneous
continuous-batch merge, mlx-lm's ``GenerationBatch.extend()`` re-introduces
None slots via ``if not any(self.logits_processors): self.logits_processors =
[None] * len(self.uids)``. Because ``any([[], []])`` is False, the empty-list
slots written at insert time collapse back to None whenever a batch with no
*active* processor merges with a grammar-constrained one (a plain chat request
joining a batch that is already serving a structured ``json_schema`` request).
The crash then fires from the merge path, not the insert path, and only
reproduces under request concurrency.

Three levels of defense:

1. **Chokepoint**: ``_patched_generation_batch_step`` normalises the whole
   list AND every per-row slot to ``[]`` before each step, covering both the
   insert and the ``extend()`` merge origins. This is the load-bearing guard.
2. **Caller-side**: ``omlx/scheduler.py`` always wraps ``logits_processors``
   as a list (possibly empty), never None, at the insert call site.
3. **Pattern matcher**: ``CACHE_CORRUPTION_PATTERNS`` includes
   ``"'NoneType' object is not iterable"`` so the scheduler recovers
   gracefully if a None slot ever sneaks through.

These tests pin all three invariants.
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
        assert "per_row_lps = list(logits_processors) if logits_processors else []" in scheduler_src, (
            "scheduler.py must wrap per-request logits_processors as a "
            "list before passing to BatchGenerator.insert. See #934."
        )
        assert "logits_processors=[per_row_lps]" in scheduler_src, (
            "scheduler.py must pass logits_processors=[per_row_lps] "
            "(per-row list, never None) to BatchGenerator.insert. See #934."
        )


class TestChokepointNormalisation:
    """Pin the load-bearing guard: per-row None slots are normalised to []
    before the original step runs, covering the extend() merge origin."""

    def test_patched_step_normalises_none_row_slots(self, monkeypatch):
        """A None per-row slot (as extend() produces it) must be normalised
        to [] at the chokepoint, before the wrapped mlx-lm step is called.

        Fails before the fix: the raw None slot reaches the wrapped step (or
        crashes omlx's own grammar-accept loop). Passes after: the slot is [].
        No model required — the rope branch is skipped without ``_uses_mrope``
        and the grammar branch is skipped without GrammarConstraintProcessor.
        """
        import omlx.scheduler as scheduler

        captured = {}

        def fake_original_step(self):
            captured["logits_processors"] = list(self.logits_processors)
            return "stepped"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", fake_original_step
        )

        def identity_processor(token_context, logits):
            return logits

        class FakeModel:
            pass

        class FakeBatch:
            model = FakeModel()
            uids = [0, 1]
            # Row 0 has a real processor; row 1 is the None slot extend() leaves.
            logits_processors = [[identity_processor], None]
            _next_tokens = None

        batch = FakeBatch()
        result = scheduler._patched_generation_batch_step(batch)

        assert result == "stepped"
        # The wrapped step must never see a None slot.
        assert captured["logits_processors"][1] == []
        assert all(slot is not None for slot in batch.logits_processors)

    def test_scheduler_source_normalises_per_row_slots(self):
        """Source-level guard against silent removal of the per-row
        normalisation. Cheap; runs without a model in CI."""
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        assert "procs if procs is not None else []" in scheduler_src, (
            "scheduler.py must normalise every per-row logits_processors slot "
            "to [] at the _patched_generation_batch_step chokepoint, because "
            "GenerationBatch.extend() re-introduces None slots on a "
            "heterogeneous merge. See #934 / #1747."
        )


def _bare_generation_batch(uid, logits_processors):
    """Build a GenerationBatch via __new__ with plain-list state.

    ``filter()`` and ``extend()`` never touch the model, so a bare instance
    is enough to exercise the real mlx-lm bookkeeping without loading
    weights. Mirrors the ``__class__.__new__`` idiom of
    ``_patched_ppb_split`` in omlx/scheduler.py.
    """
    from mlx_lm.generate import GenerationBatch

    batch = GenerationBatch.__new__(GenerationBatch)
    batch.uids = [uid]
    batch.prompt_cache = []
    batch.tokens = [[1, 2, 3]]
    batch.samplers = [lambda x: x]
    batch.fallback_sampler = lambda x: x
    batch.logits_processors = logits_processors
    batch.state_machines = [object()]
    batch.max_tokens = [4]
    batch._current_tokens = None
    batch._current_logprobs = []
    batch._next_tokens = None
    batch._next_logprobs = [object()]
    batch._token_context = [object()]
    batch._num_tokens = [0]
    batch._matcher_states = [object()]
    return batch


class TestFilterStaleProcessorAlignment:
    """Pin the GenerationBatch.filter alignment patch.

    mlx-lm's ``GenerationBatch.filter`` reindexes ``logits_processors`` only
    when ``any(self.logits_processors)`` is True; there is no else branch
    (the prompt-batch class has one: ``[[]] * len(keep)``). After a request
    with no per-request processors finishes — every slot ``[]``, the shape
    omlx inserts — removal shrinks ``uids`` but leaves the stale processor
    list behind. The next request's row then ``extend()``s in BEHIND its own
    index: row 0 reads the leftover empty slot and its real processor
    (thinking budget, grammar constraint) is silently never applied. The
    misalignment self-heals when the affected request finishes (the orphan
    makes ``any()`` True again), so the symptom is an intermittently ignored
    thinking_budget / grammar that depends on request order.

    ``_patched_generation_batch_filter`` resets the list to one empty slot
    per surviving row whenever the original guard would have skipped the
    reindex.
    """

    def test_filter_resets_stale_list_when_all_slots_inert(self):
        """filter(keep=[]) on an all-empty-slot batch must empty the list.

        Fails before the fix: ``logits_processors`` stays ``[[]]`` while
        ``uids`` becomes ``[]``. Passes after: both are empty.
        """
        import omlx.scheduler  # noqa: F401  (installs the filter patch)

        batch = _bare_generation_batch(uid=0, logits_processors=[[]])
        batch.filter([])

        assert batch.uids == []
        assert batch.logits_processors == []

    def test_processor_lands_on_its_own_row_after_remove_then_extend(self):
        """End-to-end shape of the live reproduction (#1825 follow-up).

        Request A (no processors) finishes and is removed; request B (with a
        thinking-budget-style processor) joins via extend(). B's processor
        must sit at B's row index. Fails before the fix with
        ``logits_processors == [[], [processor]]`` against ``uids == [1]`` —
        row 0 reads the stale empty slot and the processor is never called.
        """
        import omlx.scheduler  # noqa: F401  (installs the filter patch)

        def budget_processor(tokens, logits):
            return logits

        survivor = _bare_generation_batch(uid=0, logits_processors=[[]])
        survivor.filter([])  # request A removed; batch now empty

        joiner = _bare_generation_batch(
            uid=1, logits_processors=[[budget_processor]]
        )
        survivor.extend(joiner)  # request B joins the long-lived batch

        assert survivor.uids == [1]
        assert len(survivor.logits_processors) == len(survivor.uids)
        assert survivor.logits_processors[0] == [budget_processor]

    def test_filter_preserves_active_processor_reindex(self):
        """When any slot is active the original reindex path runs; the patch
        must not clobber its (correct) result."""
        import omlx.scheduler  # noqa: F401  (installs the filter patch)

        def grammar_processor(tokens, logits):
            return logits

        batch = _bare_generation_batch(uid=0, logits_processors=None)
        batch.uids = [0, 1]
        batch.tokens = [[1], [2]]
        batch.samplers = [lambda x: x, lambda x: x]
        batch.logits_processors = [[], [grammar_processor]]
        batch.state_machines = [object(), object()]
        batch.max_tokens = [4, 4]
        batch._next_logprobs = [object(), object()]
        batch._token_context = [object(), object()]
        batch._num_tokens = [0, 0]
        batch._matcher_states = [object(), object()]
        import mlx.core as mx

        batch._next_tokens = mx.array([1, 2])
        batch.filter([1])

        assert batch.uids == [1]
        assert batch.logits_processors == [[grammar_processor]]

    def test_filter_normalises_none_list(self):
        """A None logits_processors list must not crash the original filter
        (``any(None)`` raises TypeError) and must come out aligned."""
        import omlx.scheduler  # noqa: F401  (installs the filter patch)

        batch = _bare_generation_batch(uid=0, logits_processors=None)
        batch.filter([])

        assert batch.logits_processors == []

    def test_scheduler_source_installs_filter_patch(self):
        """Source-level guard against silent removal of the patch
        installation. Cheap; runs without a model in CI."""
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        assert (
            "GenerationBatch.filter = _patched_generation_batch_filter"
            in scheduler_src
        ), (
            "scheduler.py must install _patched_generation_batch_filter on "
            "GenerationBatch.filter: mlx-lm's filter leaves a stale "
            "logits_processors list behind when every slot is empty, which "
            "silently drops the next request's processors after a "
            "remove-then-extend."
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
        err = TypeError(
            "in GenerationBatch._step: 'NoneType' object is not iterable"
        )
        assert is_cache_corruption_error(err) is True


@pytest.mark.integration
class TestHeterogeneousMergeReproduction:
    """End-to-end reproduction against real mlx-lm. Integration-gated.

    Run with::

        VLLM_MLX_INTEGRATION=1 pytest tests/test_scheduler_logits_processors.py -v -m integration

    Skipped by default because it instantiates a real (small) model.
    """

    @pytest.fixture
    def small_model(self):
        import os

        if os.environ.get("VLLM_MLX_INTEGRATION") != "1":
            pytest.skip("set VLLM_MLX_INTEGRATION=1 to run this test")

        try:
            from mlx_lm import load
        except ImportError:
            pytest.skip("mlx_lm not installed")

        # Tiny model — downloads on first run.
        return load("mlx-community/Qwen3-0.6B-8bit")

    def test_none_slot_per_row_raises_typeerror(self, small_model):
        """Negative test: confirm mlx-lm does crash on None per-row slot.

        If this test stops failing in a future mlx-lm version (e.g.,
        because they harden the loop with ``or []``), it's safe to
        relax our caller-side guard. Until then, the guard is required.
        """
        import mlx.core as mx
        from mlx_lm.generate import BatchGenerator

        model, tokenizer = small_model
        bg = BatchGenerator(model, max_tokens=4)

        # Mix: row 0 has a real processor, row 1 has None.
        def identity_processor(token_context, logits):
            return logits

        tok_a = tokenizer.encode("Hi ", add_special_tokens=False)
        tok_b = tokenizer.encode("There ", add_special_tokens=False)

        bg.insert([tok_a], logits_processors=[[identity_processor]])
        bg.insert([tok_b], logits_processors=[None])  # ← the bad slot

        with pytest.raises(TypeError, match="not iterable"):
            # Drain a few generation steps to trigger _step's loop.
            for _ in range(8):
                bg.next_generated()

        bg.close()

    def test_empty_list_slot_per_row_succeeds(self, small_model):
        """Positive test: empty list slot is the fix shape, must work."""
        from mlx_lm.generate import BatchGenerator

        model, tokenizer = small_model
        bg = BatchGenerator(model, max_tokens=4)

        def identity_processor(token_context, logits):
            return logits

        tok_a = tokenizer.encode("Hi ", add_special_tokens=False)
        tok_b = tokenizer.encode("There ", add_special_tokens=False)

        bg.insert([tok_a], logits_processors=[[identity_processor]])
        bg.insert([tok_b], logits_processors=[[]])  # ← the fix shape

        # Should not raise.
        for _ in range(8):
            bg.next_generated()

        bg.close()

    def test_extend_renones_empty_slots_but_chokepoint_survives(self, small_model):
        """The actual gap: extend() turns insert-time [] slots back into None.

        Importing ``omlx.scheduler`` installs ``_patched_generation_batch_step``
        on ``GenerationBatch._step``. With the per-row normalisation in place,
        a grammar batch merged with a no-active-processor batch (the empty-list
        shape #1747 ships) must decode without raising, even though extend()
        re-None-ifies the empty slots. Drop the chokepoint normalisation and
        this test raises ``TypeError: 'NoneType' object is not iterable``.
        """
        from mlx_lm.generate import BatchGenerator

        import omlx.scheduler  # noqa: F401  (installs the _step chokepoint patch)

        model, tokenizer = small_model
        bg = BatchGenerator(model, max_tokens=6)

        def identity_processor(token_context, logits):
            return logits

        tok_a = tokenizer.encode("Hi ", add_special_tokens=False)
        tok_b = tokenizer.encode("There ", add_special_tokens=False)

        # Start a grammar-constrained row decoding, then join a plain row
        # carrying the empty-list "fix shape" — the join routes through
        # GenerationBatch.extend(), which collapses [] back to None.
        bg.insert([tok_a], logits_processors=[[identity_processor]])
        bg.next_generated()
        bg.insert([tok_b], logits_processors=[[]])

        # Must not raise with the chokepoint normalisation in place.
        for _ in range(8):
            bg.next_generated()

        bg.close()


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
            FakeBatch.model, [2], [sampler_uid2], [[budget_processor, grammar_processor]]
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
        assert scheduler_src.count("_register_uid_rows(uids") >= 2, (
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
        not pinned until LRU eviction."""
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
