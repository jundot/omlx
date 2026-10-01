# SPDX-License-Identifier: Apache-2.0
"""Tests for the full-vocab logprobs skip in ``_patched_generation_batch_step``.

mlx-lm's ``GenerationBatch._step`` materialises and retains one [vocab] fp32
array per row per step (``self._next_logprobs = list(logprobs)`` handed to
``mx.async_eval``), and omlx discards those arrays host-side whenever the
request did not ask for logprobs.  The patched step runs
``_omlx_generation_batch_step_no_logprobs`` — same forward, same processors,
same normalised sampler input, hence bit-identical tokens — but with the
per-row materialisation elided, gated on:

* no registered row asked for API logprobs (``wants_logprobs`` recorded at
  insert; unregistered uids keep mlx-lm's always-compute behaviour), and
* no speculative-decode path can read ``_next_logprobs`` (native MTP/dspark
  per-load model markers, live MTP batch state).

Consumer contracts pinned here:

* requested (or mixed) batches run the upstream step with unchanged values;
* unrequested batches return ``logprobs=None`` rows and never hand a
  vocab-sized array to ``async_eval``/``eval``;
* samplers still receive true log-probabilities (top-p/min-p math depends
  on it), so sampling is unchanged;
* grammar accept still runs ahead of whichever step variant executes.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import pytest


@pytest.fixture
def registry(monkeypatch):
    """Isolated uid-row registry, mirroring the sibling scheduler tests."""
    import omlx.scheduler as scheduler

    fresh = OrderedDict()
    monkeypatch.setattr(scheduler, "_uid_row_registry", fresh)
    return scheduler


def _register(scheduler, model, uids, wants, samplers=None, lps=None):
    n = len(uids)
    scheduler._register_uid_rows(
        model,
        uids,
        samplers if samplers is not None else [None] * n,
        lps if lps is not None else [[] for _ in uids],
        wants_logprobs_rows=list(wants),
    )


class _FixedLogitsModel:
    """Model stand-in: returns the same fixed [B, V] logits every call."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, inputs, cache=None):
        return self._logits[:, None, :]


def _fixed_logits(vocab=64, batch=2):
    """Deterministic per-row logits with distinct argmaxes and signs."""
    import mlx.core as mx

    rows = []
    for r in range(batch):
        row = mx.arange(vocab, dtype=mx.float32) * (0.01 * (r + 1))
        row = row - (0.005 * r * mx.arange(vocab, dtype=mx.float32) ** 2)
        # Distinct, unambiguous per-row argmax.
        row[r * 7 + 5] += 100.0
        rows.append(row - 3.0 * r)
    return mx.stack(rows, axis=0)


def _step_ready_batch(model, uids, next_tokens, samplers=None):
    """A real GenerationBatch (via __new__) with enough state to _step.

    No weights involved: the model is a fixed-logits callable and the
    fallback sampler is omlx's greedy argmax.  Mirrors the
    ``GenerationBatch.__new__`` idiom of ``_bare_generation_batch`` in
    tests/test_scheduler_logits_processors.py.
    """
    from mlx_lm.generate import GenerationBatch

    from omlx.utils.sampling import make_sampler

    n = len(uids)
    batch = GenerationBatch.__new__(GenerationBatch)
    batch.model = model
    batch.uids = list(uids)
    batch.prompt_cache = []
    batch.tokens = [[] for _ in uids]
    batch.samplers = list(samplers) if samplers is not None else [None] * n
    batch.fallback_sampler = make_sampler(temp=0)
    batch.logits_processors = [[] for _ in uids]
    batch.max_tokens = [8] * n
    batch.state_machines = [None] * n
    batch._current_tokens = None
    batch._current_logprobs = []
    batch._next_tokens = next_tokens
    batch._next_logprobs = []
    batch._token_context = [None] * n
    batch._num_tokens = [0] * n
    batch._matcher_states = [None] * n
    return batch


def _flatten_arrays(args):
    import mlx.core as mx

    for arg in args:
        if isinstance(arg, mx.array):
            yield arg
        elif isinstance(arg, (list, tuple)):
            yield from _flatten_arrays(arg)


class _DispatchRecorder:
    """Spies on both step variants to pin which one the gate selected."""

    def __init__(self, scheduler, monkeypatch):
        self.calls = []

        def original(self_batch):
            self.calls.append("original")
            return "original"

        def skip(self_batch):
            self.calls.append("skip")
            return "skip"

        monkeypatch.setattr(
            scheduler, "_original_generation_batch_step", original
        )
        monkeypatch.setattr(
            scheduler, "_omlx_generation_batch_step_no_logprobs", skip
        )


def _light_batch(model, uids):
    """Namespace batch good enough for realign + grammar advance + gate."""
    return SimpleNamespace(
        model=model,
        uids=list(uids),
        logits_processors=[[] for _ in uids],
        _next_tokens=None,
    )


class TestStepVariantDispatch:
    """Pin the gate: who runs — the upstream step or the skip variant."""

    def test_no_row_wanting_logprobs_runs_skip_variant(
        self, registry, monkeypatch
    ):
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace()
        _register(scheduler, model, [0, 1], wants=[False, False])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [0, 1])
        )

        assert result == "skip"
        assert recorder.calls == ["skip"]

    def test_any_row_wanting_logprobs_runs_original_variant(
        self, registry, monkeypatch
    ):
        """Mixed batch: one requesting row keeps the upstream step for all."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace()
        _register(scheduler, model, [0, 1], wants=[False, True])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [0, 1])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_unregistered_uid_runs_original_variant(
        self, registry, monkeypatch
    ):
        """Mid-insert singletons and foreign batches keep mlx-lm behaviour."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace()

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [7])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_registration_default_keeps_original_variant(
        self, registry, monkeypatch
    ):
        """Callers that omit wants_logprobs_rows keep always-compute."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace()
        scheduler._register_uid_rows(model, [3], [None], [[]])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [3])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_mtp_decode_enabled_model_runs_original_variant(
        self, registry, monkeypatch
    ):
        """Native MTP reads _next_logprobs at lazy activation — never skip."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace(_omlx_mtp_decode_enabled=True)
        _register(scheduler, model, [0], wants=[False])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [0])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_mtp_marker_on_inner_language_model_runs_original_variant(
        self, registry, monkeypatch
    ):
        """Wrapper models expose the marker on language_model (step3p7)."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        inner = SimpleNamespace(_omlx_mtp_decode_enabled=True)
        model = SimpleNamespace(language_model=inner)
        _register(scheduler, model, [0], wants=[False])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [0])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_dspark_decode_enabled_model_runs_original_variant(
        self, registry, monkeypatch
    ):
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace(_omlx_dspark_decode_enabled=True)
        _register(scheduler, model, [0], wants=[False])

        result = scheduler._patched_generation_batch_step(
            _light_batch(model, [0])
        )

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_live_mtp_batch_state_runs_original_variant(
        self, registry, monkeypatch
    ):
        """An active row-wise MTP state must keep consuming logprobs."""
        scheduler = registry
        recorder = _DispatchRecorder(scheduler, monkeypatch)
        model = SimpleNamespace()
        _register(scheduler, model, [0], wants=[False])
        batch = _light_batch(model, [0])
        batch._omlx_mtp_batch_state = object()

        result = scheduler._patched_generation_batch_step(batch)

        assert result == "original"
        assert recorder.calls == ["original"]

    def test_grammar_accept_runs_before_skip_variant(
        self, registry, monkeypatch
    ):
        """The grammar row advance still runs, ahead of whichever step."""
        import mlx.core as mx

        scheduler = registry
        calls = []

        def fake_skip(self_batch):
            calls.append("step")
            return "skip"

        monkeypatch.setattr(
            scheduler, "_omlx_generation_batch_step_no_logprobs", fake_skip
        )

        import numpy as np

        from omlx.api.grammar import GrammarConstraintProcessor

        class _RecordingMatcher:
            def __init__(self):
                self.accepted = []

            def accept_token(self, token_id):
                self.accepted.append(token_id)
                return True

            def is_terminated(self):
                return False

        proc = GrammarConstraintProcessor.__new__(GrammarConstraintProcessor)
        proc._matcher = _RecordingMatcher()
        proc._vocab_size = 64
        proc._bitmask = np.full((1, 2), -1, dtype=np.int32)
        proc._terminated = False
        proc._pending = True

        model = SimpleNamespace()
        # Register the grammar row too: the realignment rebuilds the
        # positional slots from the registry, so the proc must live there.
        _register(scheduler, model, [0], wants=[False], lps=[[proc]])
        batch = SimpleNamespace(
            model=model,
            uids=[0],
            logits_processors=[[proc]],
            _next_tokens=mx.array([11], dtype=mx.uint32),
        )

        original_accept = proc.accept_token

        def recording_accept(token_id):
            calls.append("accept")
            original_accept(token_id)

        proc.accept_token = recording_accept

        result = scheduler._patched_generation_batch_step(batch)

        assert result == "skip"
        assert calls == ["accept", "step"]
        assert proc._matcher.accepted == [11]


class TestSkipVariantSemantics:
    """Exercise the real step variants with fixed logits, no weights."""

    def test_skip_step_returns_none_logprobs_and_identical_tokens(
        self, registry
    ):
        """Skip vs upstream: same sampled tokens, but no logprobs anywhere."""
        import mlx.core as mx

        scheduler = registry
        vocab, uids = 64, [0, 1]
        logits = _fixed_logits(vocab=vocab, batch=len(uids))
        expected_tokens = mx.argmax(logits, axis=-1)

        skip_model = _FixedLogitsModel(logits)
        _register(scheduler, skip_model, uids, wants=[False, False])
        skip_batch = _step_ready_batch(
            skip_model, uids, mx.array([3, 4], dtype=mx.uint32)
        )

        inputs, lps = scheduler._patched_generation_batch_step(skip_batch)
        mx.eval(skip_batch._next_tokens)

        assert inputs == [3, 4]
        assert lps == [None, None]
        assert skip_batch._next_logprobs == [None, None]
        assert skip_batch._current_logprobs == [None, None]
        assert skip_batch._next_tokens.tolist() == expected_tokens.tolist()
        # Tokens still join the KV-cache history exactly as upstream does.
        assert skip_batch.tokens == [[3], [4]]

        # Control: identical initial state through the upstream step.
        ctrl_model = _FixedLogitsModel(logits)
        _register(scheduler, ctrl_model, uids, wants=[True, True])
        ctrl_batch = _step_ready_batch(
            ctrl_model, uids, mx.array([3, 4], dtype=mx.uint32)
        )

        ctrl_inputs, _ = scheduler._original_generation_batch_step(ctrl_batch)
        mx.eval(ctrl_batch._next_tokens)

        assert ctrl_inputs == inputs
        assert ctrl_batch._next_tokens.tolist() == expected_tokens.tolist()

    def test_skip_step_does_not_materialize_vocab_arrays(
        self, registry, monkeypatch
    ):
        """No vocab-sized array reaches async_eval/eval in the skip path."""
        import mlx.core as mx

        scheduler = registry
        vocab, uids = 64, [0, 1]
        logits = _fixed_logits(vocab=vocab, batch=len(uids))
        n = len(uids)

        eval_calls, async_eval_calls = [], []
        real_eval, real_async_eval = mx.eval, mx.async_eval

        def spy_eval(*args):
            eval_calls.append(args)
            return real_eval(*args)

        def spy_async_eval(*args):
            async_eval_calls.append(args)
            return real_async_eval(*args)

        monkeypatch.setattr(mx, "eval", spy_eval)
        monkeypatch.setattr(mx, "async_eval", spy_async_eval)

        model = _FixedLogitsModel(logits)
        _register(scheduler, model, uids, wants=[False, False])
        batch = _step_ready_batch(
            model, uids, mx.array([3, 4], dtype=mx.uint32)
        )

        scheduler._patched_generation_batch_step(batch)

        for calls in (eval_calls, async_eval_calls):
            assert calls, "the step must still evaluate its token arrays"
            for arr in _flatten_arrays(calls):
                assert arr.size <= n, (
                    "a vocab-sized array was handed to the evaluator; the "
                    "full-vocab logprobs materialisation was not skipped"
                )

        # Sanity: the spy DOES catch the materialisation on the upstream
        # path — this assertion is what makes the one above meaningful.
        eval_calls.clear()
        async_eval_calls.clear()
        ctrl_model = _FixedLogitsModel(logits)
        _register(scheduler, ctrl_model, uids, wants=[True, True])
        ctrl_batch = _step_ready_batch(
            ctrl_model, uids, mx.array([3, 4], dtype=mx.uint32)
        )
        scheduler._original_generation_batch_step(ctrl_batch)
        assert any(
            arr.size == vocab
            for arr in _flatten_arrays(async_eval_calls)
        )

    def test_skip_step_sampler_still_sees_normalized_logprobs(
        self, registry
    ):
        """top-p/min-p math needs true log-probabilities, not raw logits."""
        import mlx.core as mx

        scheduler = registry
        vocab, uids = 64, [0]
        logits = _fixed_logits(vocab=vocab, batch=1)

        seen = []

        def recording_sampler(row_logprobs):
            seen.append(row_logprobs)
            return mx.argmax(row_logprobs, axis=-1)

        model = _FixedLogitsModel(logits)
        _register(
            scheduler, model, uids, wants=[False], samplers=[recording_sampler]
        )
        batch = _step_ready_batch(
            model,
            uids,
            mx.array([3], dtype=mx.uint32),
            samplers=[recording_sampler],
        )

        scheduler._patched_generation_batch_step(batch)

        assert len(seen) == 1
        expected = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        assert mx.allclose(seen[0], expected[0:1]).item()

    def test_skip_step_drops_stale_current_logprobs(self, registry):
        """Arrays a previous computing step left behind are released."""
        import mlx.core as mx

        scheduler = registry
        vocab, uids = 64, [0, 1]
        logits = _fixed_logits(vocab=vocab, batch=len(uids))

        model = _FixedLogitsModel(logits)
        _register(scheduler, model, uids, wants=[False, False])
        batch = _step_ready_batch(
            model, uids, mx.array([3, 4], dtype=mx.uint32)
        )
        # Simulate a prior logprobs-computing step (e.g. before a requesting
        # row left the batch).
        batch._next_logprobs = [mx.zeros(vocab), mx.zeros(vocab)]

        _, lps = scheduler._patched_generation_batch_step(batch)

        assert lps == [None, None]
        assert batch._current_logprobs == [None, None]
        assert batch._next_logprobs == [None, None]

    def test_requested_step_values_are_unchanged(self, registry):
        """Requested logprobs are the exact per-row normalised distribution."""
        import mlx.core as mx

        scheduler = registry
        vocab, uids = 64, [0, 1]
        logits = _fixed_logits(vocab=vocab, batch=len(uids))
        expected = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        model = _FixedLogitsModel(logits)
        _register(scheduler, model, uids, wants=[False, True])
        batch = _step_ready_batch(
            model, uids, mx.array([3, 4], dtype=mx.uint32)
        )

        # First step: logprobs of the [3, 4] inputs land in _next_logprobs.
        scheduler._patched_generation_batch_step(batch)
        assert len(batch._next_logprobs) == 2
        for i in range(2):
            assert batch._next_logprobs[i] is not None
            assert batch._next_logprobs[i].shape == (vocab,)

        # Second step: they promote to _current_logprobs and come back as
        # the Response logprobs — the values omlx would keep for the
        # requesting row.
        _, lps = scheduler._patched_generation_batch_step(batch)
        for i in range(2):
            assert mx.allclose(lps[i], expected[i]).item()


class TestSourceGuard:
    """Cheap source-level pins against silent removal of the skip."""

    def test_skip_dispatch_is_installed(self):
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        assert "GenerationBatch._step = _patched_generation_batch_step" in scheduler_src
        assert "if _omlx_batch_wants_logprobs(self) or _omlx_spec_decode_reads_logprobs(self):" in scheduler_src
        assert "return _omlx_generation_batch_step_no_logprobs(self)" in scheduler_src

    def test_skip_variant_keeps_normalisation_but_elides_materialisation(self):
        from pathlib import Path

        scheduler_src = (
            Path(__file__).resolve().parents[1] / "omlx" / "scheduler.py"
        ).read_text()
        variant = scheduler_src[
            scheduler_src.index("def _omlx_generation_batch_step_no_logprobs") :
        ]
        # Samplers require true log-probabilities: the normalisation stays.
        assert "logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)" in variant
        # The per-row materialisation does not.
        assert "list(logprobs)" not in variant
        assert "mx.async_eval(self._next_tokens, token_context)" in variant
