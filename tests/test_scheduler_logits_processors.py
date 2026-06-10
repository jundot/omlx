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
