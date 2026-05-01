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

Two levels of defense:

1. **Caller-side**: ``omlx/scheduler.py`` always wraps
   ``logits_processors`` as a list (possibly empty), never None.
2. **Pattern matcher**: ``CACHE_CORRUPTION_PATTERNS`` includes
   ``"'NoneType' object is not iterable"`` so the scheduler recovers
   gracefully if a None slot ever sneaks through.

These tests pin both invariants.
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
