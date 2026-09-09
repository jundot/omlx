# SPDX-License-Identifier: Apache-2.0
"""Model-free tests for the fused-batch (ragged) MTP path.

Covers the parts that need no model weights: the ``OMLX_MTP_FUSED_BATCH`` env
gating and the multi-token ragged emit contract (``_emit_ragged_responses``),
which is what lets each row commit its own accepted run in a single
``GenerationBatch.next()`` call. The end-to-end draft/verify path is exercised
by the slow/integration suites with a real model.
"""
from __future__ import annotations

from types import SimpleNamespace

from omlx.patches.mlx_lm_mtp import batch_generator as bg


# ---------------------------------------------------------------------------
# Env gating
# ---------------------------------------------------------------------------


class TestFusedEnvGating:
    def test_fused_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("OMLX_MTP_FUSED_BATCH", raising=False)
        assert bg._fused_batch_mtp_enabled() is False

    def test_fused_enabled_truthy_values(self, monkeypatch):
        for val in ("1", "true", "TRUE", "yes", "on"):
            monkeypatch.setenv("OMLX_MTP_FUSED_BATCH", val)
            assert bg._fused_batch_mtp_enabled() is True

    def test_any_batch_combines_rowwise_and_fused(self, monkeypatch):
        monkeypatch.delenv("OMLX_MTP_FUSED_BATCH", raising=False)
        monkeypatch.delenv("OMLX_MTP_ROWWISE_BATCH", raising=False)
        assert bg._any_batch_mtp_enabled() is False
        monkeypatch.setenv("OMLX_MTP_FUSED_BATCH", "1")
        assert bg._any_batch_mtp_enabled() is True


# ---------------------------------------------------------------------------
# Ragged multi-token emit
# ---------------------------------------------------------------------------


def _fake_batch(n, max_tokens=None):
    """A minimal GenerationBatch stand-in for _emit_ragged_responses."""
    max_tokens = max_tokens or [1000] * n

    class _Batch:
        # ``_emit_ragged_responses`` builds ``type(gen_batch).Response(...)``.
        Response = SimpleNamespace

        def __init__(self):
            self.uids = list(range(n))
            self.tokens = [[] for _ in range(n)]
            self._num_tokens = [0] * n
            self.max_tokens = list(max_tokens)
            # matcher that never triggers a stop sequence
            self.state_machines = [
                SimpleNamespace(match=lambda st, tok: (0, None, None)) for _ in range(n)
            ]
            self._matcher_states = [0] * n
            self.filtered = None

        def extract_cache(self, idx):
            return f"cache-{idx}"

        def filter(self, keep):
            self.filtered = list(keep)

    return _Batch()


class TestEmitRagged:
    def test_each_row_emits_its_full_ragged_run(self):
        # Row 0 commits 3 tokens, row 1 commits 1 — different lengths (ragged).
        batch = _fake_batch(2)
        state = SimpleNamespace(states={})  # states absent -> stat bump skipped
        per_row = {
            0: [(101, None, "draft"), (102, None, "draft"), (103, None, "verify")],
            1: [(201, None, "verify")],
        }
        responses = bg._emit_ragged_responses(batch, state, per_row)

        # 3 + 1 = 4 Response objects, in order, all with finish_reason None
        assert len(responses) == 4
        assert [r.token for r in responses] == [101, 102, 103, 201]
        assert all(r.finish_reason is None for r in responses)
        # tokens appended to the right rows, counters advanced
        assert batch.tokens[0] == [101, 102, 103]
        assert batch.tokens[1] == [201]
        assert batch._num_tokens == [3, 1]
        # no row finished -> batch not filtered
        assert batch.filtered is None

    def test_length_finish_truncates_row_midrun(self):
        # Row 0 is allowed only 2 tokens but commits 3 -> stops at the 2nd (length)
        # and is filtered out; row 1 keeps going.
        batch = _fake_batch(2, max_tokens=[2, 1000])
        state = SimpleNamespace(states={})
        per_row = {
            0: [(101, None, "draft"), (102, None, "verify"), (103, None, "draft")],
            1: [(201, None, "verify")],
        }
        responses = bg._emit_ragged_responses(batch, state, per_row)

        # row 0 emits only 2 (the 3rd is never reached), row 1 emits 1 -> 3 total
        row0 = [r for r in responses if r.uid == 0]
        assert [r.token for r in row0] == [101, 102]
        assert row0[-1].finish_reason == "length"
        assert row0[-1].prompt_cache == "cache-0"  # finish path extracts cache
        assert batch.tokens[0] == [101, 102]
        # finished row 0 filtered out, kept = [row 1 index]
        assert batch.filtered == [1]
