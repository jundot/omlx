# SPDX-License-Identifier: Apache-2.0
"""Grammar constraints under speculative (MTP) decoding.

Covers the speculative mode of ``GrammarConstraintProcessor`` and its wiring:

1. The processor, driven the way the native verify walk drives it (per-position
   prefixes, per-row checkpoints, restore on partial accept), produces at every
   committed position the same bitmask a fresh matcher produces when advanced
   linearly along the emitted tokens. A mutation control (a processor that does
   not advance) fails the same property.
2. Dead and terminated prefixes: masks pass through, and a rewind revives them.
3. The hand-back to the deferred protocol (``end_speculative``) leaves the
   matcher on the streamed history with the row pending.
4. ``MTPProcessingSampler`` admits the grammar processor and every token it
   samples is grammar-legal, including across draft-rejection rewinds.
5. The native path: eligibility, activation hook order, the real
   ``_run_verify_cycle_chain`` with a grammar row, and the exit hook.

Everything runs on the real xgrammar matcher with a 256-entry mock vocabulary
(one token per printable ASCII character) and no model weights.
"""

from __future__ import annotations

import json
import random
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

xgr = pytest.importorskip("xgrammar")

from omlx.api.grammar import GrammarConstraintProcessor  # noqa: E402
from omlx.speculative.processing_sampler import (  # noqa: E402
    MTPProcessingSampler,
    supports_vlm_mtp_processing,
)

VOCAB_SIZE = 256
EOS = 2
PROMPT = [200, 201, 202, 203, 204]  # arbitrary ids; never part of the grammar
SCHEMA = {
    "type": "object",
    "properties": {
        "k": {"type": "string", "maxLength": 4},
        "n": {"type": "integer"},
    },
    "required": ["k", "n"],
    "additionalProperties": False,
}
TARGET_TEXT = '{"k": "ab", "n": 12}'


def _vocab() -> list[str]:
    vocab = [f"<tok_{i}>" for i in range(VOCAB_SIZE)]
    vocab[0] = "<unk>"
    vocab[1] = "<s>"
    vocab[EOS] = "</s>"
    for code in range(32, 127):
        vocab[code] = chr(code)
    return vocab


@pytest.fixture(scope="module")
def compiled():
    info = xgr.TokenizerInfo(_vocab(), stop_token_ids=[EOS])
    compiler = xgr.GrammarCompiler(info)
    cg = compiler.compile_json_schema(
        json.dumps(SCHEMA), any_whitespace=False, indent=None
    )
    ref = xgr.GrammarMatcher(cg)
    assert ref.accept_string(TARGET_TEXT), "fixture text must be grammar-legal"
    assert ref.accept_token(EOS) and ref.is_terminated()
    return cg


def _target_ids() -> list[int]:
    return [ord(c) for c in TARGET_TEXT] + [EOS]


def _allowed_from_matcher(matcher) -> set[int]:
    width = (VOCAB_SIZE + 31) // 32
    bitmask = np.full((1, width), -1, dtype=np.int32)
    matcher.fill_next_token_bitmask(bitmask)
    words = bitmask[0].astype(np.uint32)
    return {t for t in range(VOCAB_SIZE) if (int(words[t // 32]) >> (t % 32)) & 1}


def _allowed_from_logits(logits) -> set[int]:
    row = np.array(logits, dtype=np.float32).reshape(-1)[:VOCAB_SIZE]
    return {t for t in range(VOCAB_SIZE) if np.isfinite(row[t])}


def _reference_allowed(cg, generated: list[int]) -> set[int]:
    """Allowed set of a fresh matcher advanced linearly along ``generated``."""
    matcher = xgr.GrammarMatcher(cg)
    for token in generated:
        assert matcher.accept_token(token), f"reference rejected {token}"
    if matcher.is_terminated():
        return set(range(VOCAB_SIZE))  # processor passes logits through
    return _allowed_from_matcher(matcher)


def _zeros():
    return mx.zeros((1, VOCAB_SIZE))


def _first_illegal(cg, generated: list[int]) -> int:
    matcher = xgr.GrammarMatcher(cg)
    for token in generated:
        matcher.accept_token(token)
    allowed = _allowed_from_matcher(matcher)
    for candidate in (ord("#"), ord("!"), ord("Z"), ord("\\"), 3, 0):
        if candidate not in allowed:
            return candidate
    raise AssertionError("no illegal token found")


# ---------------------------------------------------------------------------
# 1. Verify-walk emulation against the linear reference
# ---------------------------------------------------------------------------


def _run_emulated_walk(cg, proc, *, k: int, seed: int) -> None:
    """Drive ``proc`` the way ``_run_verify_cycle_chain`` does and check masks.

    Mirrors the native path: ``begin_speculative`` at activation with the
    prompt-length history and the first token in flight; the seed's
    processor call on ``prompt + [t0]``; the draft chain (snapshot, one call
    per draft with the hypothetical prefix, restore); then cycles of k+1 rows
    with per-row checkpoints, restore to the last emitted row's checkpoint and
    a buffer trimmed to the committed prefix.
    """
    rng = random.Random(seed)
    target = _target_ids()
    prompt = list(PROMPT)

    proc.begin_speculative(len(prompt))
    committed = [target[0]]  # main_tok, sampled under the deferred mask
    # Seed: the distribution after main_tok is masked with prefix prompt+[t0].
    allowed = _allowed_from_logits(proc(mx.array(prompt + committed), _zeros()))
    assert allowed == _reference_allowed(cg, committed), "seed row mask"
    committed.append(target[1])  # next_main_tok
    assert not proc.pending

    def draft_chain(buffer: list[int], chain_prefix: int, n_committed: int):
        """One draft chain: masked drafts along the hypothetical prefix."""
        snap = proc.snapshot_state()
        drafts: list[int] = []
        for j in range(k):
            prefix = buffer + [chain_prefix] + drafts
            out = proc(mx.array(prefix), _zeros())
            legal = _allowed_from_logits(out)
            want = target[n_committed + j] if n_committed + j < len(target) else EOS
            if want not in legal or rng.random() < 0.35:
                # The drafter samples under the mask of its own hypothetical
                # prefix: after a wrong draft the target's next token may be
                # illegal there, and sometimes it just guesses another legal one.
                options = sorted(legal - {want}) or sorted(legal)
                tok = rng.choice(options)
            else:
                tok = want
            assert tok in legal, "the drafter samples under the mask"
            drafts.append(tok)
        proc.restore_state(snap)
        return drafts

    buffer = prompt + committed[:-1]
    drafts = draft_chain(buffer, committed[-1], len(committed))

    while True:
        # --- verify cycle ---
        next_main = committed[-1]
        n_before = len(committed)
        prev_rows = []
        row_allowed = []
        row_snaps = []
        inputs = [next_main] + drafts
        for j in range(k + 1):
            prev_rows.append(buffer + inputs[: j + 1])
            out = proc(mx.array(prev_rows[j]), _zeros())
            row_allowed.append(_allowed_from_logits(out))
            row_snaps.append(proc.snapshot_state())
        assert not proc.pending
        # Greedy acceptance against the target sequence.
        m = 0
        while (
            m < k and n_before + m < len(target) and drafts[m] == target[n_before + m]
        ):
            m += 1
        if m < k:
            proc.restore_state(row_snaps[m])
        pos_last = n_before + m
        emit_last = target[pos_last] if pos_last < len(target) else EOS
        emitted = drafts[:m] + [emit_last]
        if EOS in emitted:
            # The request finishes at EOS; the scheduler discards the rest.
            emitted = emitted[: emitted.index(EOS) + 1]
        # Every committed row's mask equals the linear reference at its position.
        for j in range(len(emitted)):
            assert row_allowed[j] == _reference_allowed(cg, committed + drafts[:j]), (
                f"seed={seed} cycle at {n_before} row {j}"
            )
            assert emitted[j] in row_allowed[j]
        committed.extend(emitted)
        if emitted[-1] == EOS:
            break
        buffer = prompt + committed[:-1]
        assert proc._seen == len(buffer), "buffer trimmed to the committed prefix"
        drafts = draft_chain(buffer, committed[-1], len(committed))

    assert committed == target
    # The final EOS is next_main of a cycle that never runs; accept it as the
    # standard epilogue would and the matcher terminates exactly there.
    proc(mx.array(prompt + committed), _zeros())
    assert proc.is_terminated


class TestSpeculativeModeMatchesLinearMatcher:
    @pytest.mark.parametrize("k", [1, 2, 3, 4])
    @pytest.mark.parametrize("seed", range(6))
    def test_masks_match_reference_along_emitted_tokens(self, compiled, k, seed):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        _run_emulated_walk(compiled, proc, k=k, seed=seed)

    def test_mutation_control_non_advancing_processor_fails(
        self, compiled, monkeypatch
    ):
        """The old processor (mask from a frozen state) fails the same property."""
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        monkeypatch.setattr(proc, "_sync_to_history", lambda tokens: None)
        with pytest.raises(AssertionError):
            _run_emulated_walk(compiled, proc, k=3, seed=0)

    def test_speculative_mode_never_sets_pending(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        proc(mx.array(PROMPT + [ord("{")]), _zeros())
        assert proc.pending is False
        assert proc.speculative is True


# ---------------------------------------------------------------------------
# 2. Dead and terminated prefixes
# ---------------------------------------------------------------------------


class TestDeadAndTerminated:
    def test_illegal_speculative_prefix_passes_through_until_restored(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        good = [ord("{"), ord('"')]
        proc(mx.array(PROMPT + good), _zeros())
        snap = proc.snapshot_state()
        bad = _first_illegal(compiled, good)
        out = proc(mx.array(PROMPT + good + [bad]), _zeros())
        # Dead: nothing masked (the row can never be committed).
        assert _allowed_from_logits(out) == set(range(VOCAB_SIZE))
        out = proc(mx.array(PROMPT + good + [bad, ord("k")]), _zeros())
        assert _allowed_from_logits(out) == set(range(VOCAB_SIZE))
        # Restore to the checkpoint before the illegal token revives the matcher.
        proc.restore_state(snap)
        out = proc(mx.array(PROMPT + good), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(compiled, good)

    def test_rewind_by_shorter_history_revives_dead_prefix(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        good = [ord("{"), ord('"'), ord("k")]
        bad = _first_illegal(compiled, good)
        proc(mx.array(PROMPT + good + [bad]), _zeros())
        # A shorter history (as after a buffer trim) rewinds past the dead token.
        out = proc(mx.array(PROMPT + good[:2]), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(compiled, good[:2])

    def test_terminated_then_rollback_revives(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        target = _target_ids()
        proc(mx.array(PROMPT + target[:-1]), _zeros())
        snap = proc.snapshot_state()
        out = proc(mx.array(PROMPT + target), _zeros())  # accepts EOS
        assert proc.is_terminated
        assert _allowed_from_logits(out) == set(range(VOCAB_SIZE))
        proc.restore_state(snap)
        assert not proc.is_terminated
        out = proc(mx.array(PROMPT + target[:-1]), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(compiled, target[:-1])
        assert EOS in _allowed_from_logits(out)

    def test_restore_cannot_move_forward(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        proc(mx.array(PROMPT + [ord("{"), ord('"')]), _zeros())
        later = proc.snapshot_state()
        proc(mx.array(PROMPT + [ord("{")]), _zeros())
        with pytest.raises(RuntimeError):
            proc.restore_state(later)


# ---------------------------------------------------------------------------
# 3. Hand-back to the deferred protocol
# ---------------------------------------------------------------------------


class TestEndSpeculative:
    def test_end_syncs_to_streamed_history_and_marks_pending(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        # Deferred prelude: the __init__ step masks position 0.
        proc(mx.array(PROMPT), _zeros())
        assert proc.pending
        proc.begin_speculative(len(PROMPT))
        assert not proc.pending
        target = _target_ids()
        # The walk got ahead of what was streamed (three speculative tokens).
        proc(mx.array(PROMPT + target[:6]), _zeros())
        streamed = PROMPT + target[:3]
        proc.end_speculative(streamed, pending=True)
        assert proc.speculative is False
        assert proc.pending is True
        # Deferred accept of the in-flight token, then the mask is the reference's.
        proc.accept_token(target[3])
        assert not proc.pending
        out = proc(mx.array(streamed + [target[3]]), _zeros())
        assert proc.pending
        assert _allowed_from_logits(out) == _reference_allowed(compiled, target[:4])

    def test_end_accepts_streamed_tokens_the_walk_had_not_seen(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        target = _target_ids()
        proc(mx.array(PROMPT + target[:2]), _zeros())
        # The queue drained one token further than the matcher (emit_last).
        proc.end_speculative(PROMPT + target[:3], pending=True)
        out = proc(mx.array(PROMPT + target[:3]), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(compiled, target[:3])

    def test_end_without_history_only_switches_mode(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        proc(mx.array(PROMPT + [ord("{")]), _zeros())
        proc.end_speculative(None, pending=False)
        assert proc.speculative is False
        assert proc.pending is False

    def test_end_is_noop_in_deferred_mode(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc(mx.array(PROMPT), _zeros())
        proc.end_speculative(PROMPT, pending=False)
        assert proc.pending is True  # untouched


# ---------------------------------------------------------------------------
# 4. vlm_mtp: MTPProcessingSampler with a grammar processor
# ---------------------------------------------------------------------------


def _argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _favor(token_ids: list[int]) -> mx.array:
    rows = np.zeros((len(token_ids), VOCAB_SIZE), dtype=np.float32)
    for r, t in enumerate(token_ids):
        rows[r, t] = 10.0
    return mx.array(rows)


class TestProcessingSampler:
    def test_grammar_processor_passes_the_gate(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        assert supports_vlm_mtp_processing(proc)

    def test_sampled_tokens_are_legal_across_rewinds(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
        assert proc.speculative and not proc.pending
        target = _target_ids()

        first = sampler.process_first_logits(_favor([_first_illegal(compiled, [])]))
        bonus = int(_argmax_sampler(first).item())
        assert bonus in _reference_allowed(compiled, [])
        assert bonus == target[0]  # the only legal opener is "{"
        sampler.note_first_bonus(bonus, position=1)

        # Verify walk 1 at positions 1..3: slot 2 favours an illegal token.
        favored = [target[1], _first_illegal(compiled, target[:2]), target[3]]
        out = sampler.sample_target(_favor(favored), positions=[1, 2, 3])
        sampled = [int(t) for t in out.tolist()]
        history = [target[0]]
        for tok in sampled:
            assert tok in _reference_allowed(compiled, history)
            history.append(tok)

        # Draft rejection: mlx-vlm re-samples the suffix from position 2.
        out = sampler.sample_target(_favor([target[2], target[3]]), positions=[2, 3])
        sampled = [int(t) for t in out.tolist()]
        assert sampled == [target[2], target[3]]
        assert _allowed_from_logits(proc(sampler._history, _zeros())) == (
            _reference_allowed(compiled, target[:4])
        )

    def test_reset_hands_back_a_pristine_deferred_processor(self, compiled):
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
        sampler.process_first_logits(_zeros())
        sampler.note_first_bonus(ord("{"), position=1)
        sampler.sample_target(_favor([ord('"')]), positions=[1])
        sampler.reset_processors()
        assert proc.speculative is False
        assert proc.pending is False
        out = proc(mx.array(PROMPT), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(compiled, [])


# ---------------------------------------------------------------------------
# 5. Native path wiring
# ---------------------------------------------------------------------------


class _MtpModel:
    def __init__(self):
        self.mtp = object()
        self._omlx_mtp_decode_enabled = True

    def mtp_forward(self, *_):
        pass


class TestNativePathWiring:
    def test_grammar_rows_are_eligible_for_singleton_mtp(self, compiled):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        batch = SimpleNamespace(model=_MtpModel(), uids=[1], logits_processors=[[proc]])
        assert bg._has_grammar_processors(batch) is True
        assert bg._is_mtp_eligible(batch) is True
        assert bg._ineligibility_reason(batch) == ""

    def test_grammar_rows_stay_off_rowwise_batch_mtp(self, compiled, monkeypatch):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        monkeypatch.setenv(bg._ROWWISE_BATCH_MTP_ENV, "1")
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        batch = SimpleNamespace(
            model=_MtpModel(), uids=[1, 2], logits_processors=[[proc], []]
        )
        assert bg._is_mtp_batch_eligible(batch) is False
        assert "grammar" in bg._ineligibility_reason(batch)
        batch.logits_processors = [[], []]
        assert bg._is_mtp_batch_eligible(batch) is True

    def test_enter_hook_baselines_on_the_token_buffer(self, compiled):
        from mlx_lm.models.cache import TokenBuffer

        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc(mx.array(PROMPT), _zeros())  # deferred prelude, main_tok in flight
        batch = SimpleNamespace(
            logits_processors=[[proc]], _token_context=[TokenBuffer(list(PROMPT))]
        )
        bg._grammar_enter_speculative(batch)
        assert proc.speculative and not proc.pending
        assert proc._base == len(PROMPT)

    def test_post_init_enters_speculative_before_any_forward(
        self, compiled, monkeypatch
    ):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        class MarkerError(Exception):
            pass

        def enter(_batch):
            raise MarkerError()

        monkeypatch.setattr(bg, "_grammar_enter_speculative", enter)
        monkeypatch.setattr(
            bg, "_call_backbone", lambda *a, **k: pytest.fail("forward before enter")
        )
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        batch = SimpleNamespace(
            uids=[1],
            model=_MtpModel(),
            _next_tokens=mx.array([ord("{")], dtype=mx.uint32),
            _next_logprobs=[mx.zeros((VOCAB_SIZE,))],
            samplers=[None],
            fallback_sampler=_argmax_sampler,
            logits_processors=[[proc]],
            _token_context=[SimpleNamespace(update_and_fetch=lambda t: t)],
        )
        with pytest.raises(MarkerError):
            bg._post_init_mtp(batch)

    def test_verify_cycle_emits_legal_tokens_and_keeps_matcher_in_step(
        self, compiled, monkeypatch
    ):
        from mlx_lm.models.cache import TokenBuffer

        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        target = _target_ids()
        k = 3
        # Cycle boundary inside the string value: committed = target[:7]
        # ('{"k": "'), buffer holds target[:6] (accepted), next_main = target[6]
        # is the opening quote not yet pushed. Rows 0..3 predict positions 7..10.
        base = 7
        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc.begin_speculative(len(PROMPT))
        proc(mx.array(PROMPT + target[: base - 1]), _zeros())
        buffer = TokenBuffer(PROMPT + target[: base - 1])
        # Drafts: the first is right ('a'), the second is legal but wrong
        # ('x' for 'b'), the third is illegal after it (a backslash: the
        # bounded string production has no escape branch). Greedy acceptance
        # must stop at m = 1 and the dead third row must never be committed.
        wrong_legal = sorted(
            _reference_allowed(compiled, target[: base + 1]) - {target[base + 1]}
        )[0]
        illegal = ord("\\")
        assert illegal not in _reference_allowed(compiled, target[:base] + [wrong_legal])
        drafts = [target[base], wrong_legal, illegal]

        def fake_backbone(_model, inputs, _cache, **_kwargs):
            width = int(inputs.shape[1])
            # Row j predicts the target token after input j; the dead row
            # favours the illegal token, which is fine: it is never emitted.
            rows = []
            for j in range(width):
                pos = base + j
                favored = target[pos] if pos < len(target) else EOS
                if j == 3:
                    favored = illegal
                row = [-100.0] * VOCAB_SIZE
                row[favored] = 0.0
                rows.append(row)
            return (
                mx.array([rows], dtype=mx.float32),
                mx.zeros((1, width, 8), dtype=mx.float32),
                None,
            )

        rollbacks = []
        monkeypatch.setattr(bg, "_call_backbone", fake_backbone)
        monkeypatch.setattr(
            bg,
            "_chain_rollback",
            lambda _m, _c, accepted, num_drafts, _g: (
                rollbacks.append((accepted, num_drafts)) or True
            ),
        )
        monkeypatch.setattr(bg, "_chain_next_drafts", lambda *a, **k: None)
        monkeypatch.setattr(bg, "_clear_rollback", lambda _c: None)

        state = bg._MtpState(
            uid=1,
            chain=True,
            depth=k,
            mtp_cache=[],
            next_main=mx.array([target[base - 1]], dtype=mx.uint32),
            drafts=mx.array(drafts, dtype=mx.uint32),
            draft_lps=[mx.zeros((VOCAB_SIZE,)) for _ in drafts],
        )
        batch = SimpleNamespace(
            model=SimpleNamespace(),
            prompt_cache=[SimpleNamespace(offset=len(PROMPT) + base - 1)],
            tokens=[PROMPT + target[:base]],
            samplers=[None],
            fallback_sampler=_argmax_sampler,
            logits_processors=[[proc]],
            _token_context=[buffer],
        )

        bg._run_verify_cycle_chain(batch, state)

        emitted = [tok for tok, _lp, _src in state.queue]
        # m = 1: the right draft plus the target's correction for the wrong one.
        assert emitted == [target[base], target[base + 1]], emitted
        assert rollbacks == [(1, k)]
        assert not proc.pending
        # Buffer and matcher both sit on prompt + committed + next_main + d1.
        assert buffer._size == len(PROMPT) + base + 1
        assert proc._seen == len(PROMPT) + base + 1
        out = proc(mx.array(PROMPT + target[: base + 1]), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(
            compiled, target[: base + 1]
        )

        # Exit: everything streamed, a fresh token in flight for the standard step.
        batch.tokens[0].extend(emitted)
        batch._next_tokens = mx.array([target[base + 2]], dtype=mx.uint32)
        batch._omlx_mtp_state = state
        bg._drop_mtp_state(batch, "test-exit")
        assert proc.speculative is False
        assert proc.pending is True
        proc.accept_token(target[base + 2])
        out = proc(mx.array(PROMPT + target[: base + 3]), _zeros())
        assert _allowed_from_logits(out) == _reference_allowed(
            compiled, target[: base + 3]
        )

    def test_leave_hook_is_a_noop_for_deferred_rows(self, compiled):
        from omlx.patches.mlx_lm_mtp import batch_generator as bg

        proc = GrammarConstraintProcessor(compiled, VOCAB_SIZE)
        proc(mx.array(PROMPT), _zeros())
        batch = SimpleNamespace(
            logits_processors=[[proc]], tokens=[list(PROMPT)], _next_tokens=None
        )
        bg._grammar_leave_speculative(batch)
        assert proc.pending is True
        assert proc.speculative is False
