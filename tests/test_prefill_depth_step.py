# SPDX-License-Identifier: Apache-2.0
"""The prefill chunk shrinks as the KV cache gets deeper (Metal GPU watchdog)."""

from types import SimpleNamespace

import pytest


def test_bounded_step_keeps_the_ceiling_shallow_and_shrinks_deep():
    from omlx.patches.prefill_depth_step import bounded_step

    assert bounded_step(2048, 0) == 2048
    assert bounded_step(2048, 50_000) == 2048
    assert bounded_step(1024, 124_000) == 1024
    assert bounded_step(1024, 150_000) == 1024
    assert bounded_step(1024, 245_000) == 512
    assert bounded_step(512, 0) == 512  # never above the configured step
    assert bounded_step(1024, 10**7) == 128  # floor


def test_prompt_processing_batch_reads_a_depth_bounded_step(monkeypatch):
    mlx_generate = pytest.importorskip("mlx_lm.generate")
    from omlx.patches import prefill_depth_step

    cls = mlx_generate.PromptProcessingBatch
    # Restore the unpatched attribute when the test ends.
    monkeypatch.setattr(
        cls, "prefill_step_size", cls.__dict__.get("prefill_step_size"), raising=False
    )
    assert prefill_depth_step.install() is True
    assert prefill_depth_step.install() is True  # idempotent

    batch = cls.__new__(cls)
    batch.prompt_cache = [SimpleNamespace(offset=0)]
    batch.prefill_step_size = 1024  # what __init__ does
    assert batch.prefill_step_size == 1024
    # Gated-delta layers carry no offset; the deepest KV layer decides.
    batch.prompt_cache = [SimpleNamespace(), SimpleNamespace(offset=245_000)]
    assert batch.prefill_step_size == 512
    batch.prompt_cache = []
    assert batch.prefill_step_size == 1024
