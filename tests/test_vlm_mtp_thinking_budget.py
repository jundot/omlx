# SPDX-License-Identifier: Apache-2.0
"""Thinking budget enforcement on the vlm_mtp decode path.

Covers the three layers of the fix for #2399 (thinking-budget half):

1. ``ThinkingBudgetProcessor.snapshot_state`` / ``restore_state`` —
   position-keyed rewind support.
2. ``MTPProcessingSampler`` — processor application through mlx-vlm's
   positioned ``sample_target`` hook, including draft-rejection rewinds.
3. ``Scheduler._route_to_vlm_mtp`` gate — budget processors route through
   vlm_mtp; unsupported processors still fall back to BatchGenerator.

The end-to-end tests drive mlx-vlm's *real* ``_mtp_rounds`` loop with fake
target/drafter modules, so the full contract (verify walk, acceptance,
rollback, positioned sampling) is exercised without model weights.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import mlx.core as mx
import pytest

import omlx.scheduler as scheduler_mod
from omlx.api.thinking import ThinkingBudgetProcessor
from omlx.scheduler import Scheduler
from omlx.speculative.processing_sampler import (
    MTPProcessingSampler,
    supports_vlm_mtp_processing,
)
from omlx.speculative.vlm_mtp import (
    _VLMAdapterMTPProxy,
    vlm_mtp_positioned_sampling_available,
)

VOCAB = 32
THINK = 7  # the model's preferred "thinking filler" token
LEAD, END, TRAIL = 20, 21, 22  # forced close sequence \n </think> \n\n
PROMPT = [1, 2, 3]


def _argmax_sampler(logits):
    return mx.argmax(logits, axis=-1)


def _favor(token_id: int) -> mx.array:
    logits = mx.zeros((1, VOCAB))
    logits[0, token_id] = 10.0
    return logits


def _make_budget_processor(budget: int) -> ThinkingBudgetProcessor:
    return ThinkingBudgetProcessor(
        think_end_token_ids=[END],
        budget=budget,
        think_start_token_id=None,
        leading_token_ids=[LEAD],
        trailing_token_ids=[TRAIL],
    )


# ---------------------------------------------------------------------------
# 1. Processor snapshot / restore
# ---------------------------------------------------------------------------


class TestSnapshotRestore:
    def test_supports_protocol(self):
        assert supports_vlm_mtp_processing(_make_budget_processor(4))
        assert not supports_vlm_mtp_processing(lambda toks, logits: logits)

    def test_restore_rewinds_forcing(self):
        proc = _make_budget_processor(2)
        history = list(PROMPT)

        proc(history, _favor(THINK))  # baseline call: count=1
        snap = proc.snapshot_state()

        history.append(THINK)
        out = proc(history, _favor(THINK))  # count=2 >= budget -> forces LEAD
        assert int(mx.argmax(out, axis=-1).item()) == LEAD
        assert proc._forcing

        proc.restore_state(snap)
        assert not proc._forcing
        assert proc._thinking_tokens == 1

        # Replaying the same continuation reproduces the same decision.
        out = proc(history, _favor(THINK))
        assert int(mx.argmax(out, axis=-1).item()) == LEAD

    def test_restore_drops_lazy_baseline_attr(self):
        proc = _make_budget_processor(4)
        snap = proc.snapshot_state()  # before any call: no _accepted_up_to
        proc(list(PROMPT), _favor(THINK))
        assert hasattr(proc, "_accepted_up_to")
        proc.restore_state(snap)
        assert not hasattr(proc, "_accepted_up_to")


# ---------------------------------------------------------------------------
# 2. MTPProcessingSampler unit behaviour
# ---------------------------------------------------------------------------


def _positioned_logprobs(n_slots: int, favored: int = THINK) -> mx.array:
    logits = mx.zeros((n_slots, VOCAB))
    for i in range(n_slots):
        logits[i, favored] = 10.0
    return logits


class TestMTPProcessingSampler:
    def _fresh(self, budget: int):
        proc = _make_budget_processor(budget)
        sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
        logits = sampler.process_first_logits(_favor(THINK))
        bonus = int(mx.argmax(logits, axis=-1).item())
        sampler.note_first_bonus(bonus)
        return proc, sampler, bonus

    def test_forces_close_at_budget(self):
        proc, sampler, bonus = self._fresh(budget=4)
        assert bonus == THINK  # bonus counted as thinking token 1

        out = sampler.sample_target(
            _positioned_logprobs(4), row_ids=[0] * 4, positions=[1, 2, 3, 4]
        )
        # counts 2,3 free; count 4 hits the budget -> LEAD, then END.
        assert [int(t) for t in out.tolist()] == [THINK, THINK, LEAD, END]
        assert not sampler._degraded

    def test_rejection_rewind_replays_forcing_decision(self):
        proc, sampler, _ = self._fresh(budget=4)

        out_a = sampler.sample_target(
            _positioned_logprobs(3), row_ids=[0] * 3, positions=[1, 2, 3]
        )
        assert [int(t) for t in out_a.tolist()] == [THINK, THINK, LEAD]

        # Simulate the walk rejecting the draft at slot 2: only position 1
        # committed; mlx-vlm re-samples from position 2.
        out_b = sampler.sample_target(
            _positioned_logprobs(3), row_ids=[0] * 3, positions=[2, 3, 4]
        )
        assert [int(t) for t in out_b.tolist()] == [THINK, LEAD, END]

        # All three committed; forcing continues exactly where it left off.
        out_c = sampler.sample_target(
            _positioned_logprobs(1), row_ids=[0], positions=[5]
        )
        assert [int(t) for t in out_c.tolist()] == [TRAIL]

        out_d = sampler.sample_target(
            _positioned_logprobs(1), row_ids=[0], positions=[6]
        )
        assert [int(t) for t in out_d.tolist()] == [THINK]  # budget done
        assert proc._done
        assert not sampler._degraded

    def test_natural_close_disables_forcing(self):
        proc, sampler, _ = self._fresh(budget=10)
        logits = mx.zeros((3, VOCAB))
        logits[0, THINK] = 10.0
        logits[1, END] = 10.0  # model closes thinking naturally
        logits[2, THINK] = 10.0
        out = sampler.sample_target(logits, row_ids=[0] * 3, positions=[1, 2, 3])
        assert [int(t) for t in out.tolist()] == [THINK, END, THINK]
        assert proc._done
        assert not proc._forcing

    def test_reset_processors_restores_pristine_state(self):
        proc, sampler, _ = self._fresh(budget=4)
        assert proc._thinking_tokens == 1
        sampler.reset_processors()
        assert proc._thinking_tokens == 0
        assert not hasattr(proc, "_accepted_up_to")
        assert sampler._history == PROMPT

    def test_missing_positions_degrades_loudly(self, caplog):
        _, sampler, _ = self._fresh(budget=4)
        with caplog.at_level(
            logging.WARNING, logger="omlx.speculative.processing_sampler"
        ):
            out = sampler.sample_target(_positioned_logprobs(2))
        assert sampler._degraded
        assert "NOT enforced" in caplog.text
        assert [int(t) for t in out.tolist()] == [THINK, THINK]


# ---------------------------------------------------------------------------
# 3. End-to-end through mlx-vlm's real _mtp_rounds
# ---------------------------------------------------------------------------


class _FakeTargetLM:
    """Target language model driving mlx-vlm's verify path.

    Encodes the absolute generated-token position of every verify slot in
    ``hidden[..., 0]`` so ``speculative_logits_from_hidden`` can pick a
    deterministic favored token per position. Tracks a fake KV length with
    correct rollback semantics.
    """

    def __init__(self, favored_by_pos=None):
        self.gen_tokens = 0  # generated tokens currently in the fake cache
        self.rollbacks = []
        self._favored_by_pos = favored_by_pos or {}

    def speculative_verify_hidden(self, verify_input, prompt_cache):
        n = int(verify_input.shape[1])
        hidden = mx.zeros((1, n, 4))
        for i in range(n):
            # slot i predicts generated position gen_tokens + i + 1
            hidden[0, i, 0] = float(self.gen_tokens + i + 1)
        self.gen_tokens += n
        kv = mx.zeros((1, 1, self.gen_tokens, 2))
        return hidden, {"full": (kv, kv)}

    def speculative_logits_from_hidden(self, hidden):
        n = int(hidden.shape[1])
        logits = mx.zeros((1, n, VOCAB))
        for i in range(n):
            pos = int(hidden[0, i, 0].item())
            favored = self._favored_by_pos.get(pos, THINK)
            logits[0, i, favored] = 10.0
        return logits

    def rollback_speculative_cache(self, prompt_cache, gdn_states, accepted, bs):
        trimmed = bs - accepted - 1
        self.gen_tokens -= trimmed
        self.rollbacks.append((accepted, bs))


class _FakeDrafter:
    """Drafter proposing a fixed pattern (default: always THINK)."""

    supports_greedy_draft_argmax = False

    def __init__(self, pattern=None):
        self.config = SimpleNamespace(block_size=4)
        self.accept_lens = []
        self.pattern = pattern or [THINK]

    def reset(self, model):
        pass

    def set_shared_kv(self, states, kv_offset, position=None, kv_valid_len=None):
        pass

    def draft_block(self, b, hidden, x, bs, sampler, dtype):
        row = [self.pattern[i % len(self.pattern)] for i in range(bs - 1)]
        return mx.array([row], dtype=dtype)


def _run_rounds(sampler, lm=None, drafter=None, max_tokens=16, first_bonus=THINK):
    from mlx_vlm.speculative.mtp import _mtp_rounds

    lm = lm or _FakeTargetLM()
    drafter = drafter or _FakeDrafter()
    prompt_cache = [SimpleNamespace(offset=len(PROMPT))]
    hidden = mx.zeros((1, 1, 4))

    tokens = [first_bonus]
    for tok, _ in _mtp_rounds(
        lm,
        drafter,
        prompt_cache,
        hidden,
        {},
        first_bonus=first_bonus,
        max_tokens=max_tokens,
        sampler=sampler,
        draft_block_size=4,
        token_dtype=mx.int32,
    ):
        tokens.append(int(tok))
    return tokens, lm, drafter


def _wrapped_sampler(budget: int):
    proc = _make_budget_processor(budget)
    sampler = MTPProcessingSampler(_argmax_sampler, [proc], PROMPT)
    logits = sampler.process_first_logits(_favor(THINK))
    bonus = int(mx.argmax(logits, axis=-1).item())
    sampler.note_first_bonus(bonus)
    return proc, sampler, bonus


class TestEndToEndMtpRounds:
    def test_budget_forces_close_inside_speculation(self):
        proc, sampler, bonus = _wrapped_sampler(budget=5)
        tokens, lm, drafter = _run_rounds(sampler, first_bonus=bonus, max_tokens=12)

        # bonus + 3 free thinking tokens, then the forced close sequence,
        # then ordinary (fake) content until max_tokens.
        assert tokens[:4] == [THINK] * 4
        assert tokens[4:7] == [LEAD, END, TRAIL]
        assert all(t == THINK for t in tokens[7:])
        assert len(tokens) == 12
        assert proc._done
        assert not sampler._degraded
        # The forced tokens mismatch the drafter's proposals, so at least
        # one rejection/rollback must have occurred.
        assert lm.rollbacks

    def test_budget_holds_under_frequent_draft_rejection(self):
        # Drafter proposes a wrong token in slot 2 of every block, forcing
        # a rejection (and wrapper rewind) each round.
        proc, sampler, bonus = _wrapped_sampler(budget=5)
        drafter = _FakeDrafter(pattern=[THINK, 9, THINK])
        tokens, lm, _ = _run_rounds(
            sampler, drafter=drafter, first_bonus=bonus, max_tokens=12
        )
        assert tokens[:4] == [THINK] * 4
        assert tokens[4:7] == [LEAD, END, TRAIL]
        assert all(t == THINK for t in tokens[7:])
        assert proc._done
        assert not sampler._degraded

    def test_natural_close_before_budget_is_respected(self):
        proc, sampler, bonus = _wrapped_sampler(budget=10)
        lm = _FakeTargetLM(favored_by_pos={3: END})
        tokens, _, _ = _run_rounds(sampler, lm=lm, first_bonus=bonus, max_tokens=10)

        assert tokens[3] == END
        assert LEAD not in tokens
        assert TRAIL not in tokens
        assert proc._done
        assert not proc._forcing
        assert not sampler._degraded

    def test_without_wrapper_budget_is_dropped(self):
        # Control: a bare sampler (pre-fix behaviour) never closes thinking.
        tokens, _, _ = _run_rounds(_argmax_sampler, max_tokens=12)
        assert all(t == THINK for t in tokens)


# ---------------------------------------------------------------------------
# 4. _route_to_vlm_mtp gate
# ---------------------------------------------------------------------------


def _make_route_request():
    return SimpleNamespace(
        request_id="req-budget",
        sampling_params=SimpleNamespace(max_tokens=64, stop_token_ids=None),
        rope_deltas=0.0,
        prompt_token_ids=list(PROMPT),
    )


class TestRouteGate:
    def test_budget_processor_passes_gate(self, caplog):
        """A snapshot-capable processor must not trigger the fallback; the
        fake model lacks _language_model, so passing the gate surfaces as
        the later rollback-hook decline."""
        sched = SimpleNamespace(
            _vlm_mtp_drafter=object(),
            _vlm_mtp_active={},
            model=SimpleNamespace(),
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            uid = Scheduler._route_to_vlm_mtp(
                sched,
                _make_route_request(),
                [object()],
                [42],
                lambda x: x,
                object(),
                logits_processors=[_make_budget_processor(4)],
            )
        assert uid is None
        assert "logits processors" not in caplog.text
        assert "rollback_speculative_cache" in caplog.text

    def test_unsupported_processor_still_declines(self, caplog):
        sched = SimpleNamespace(_vlm_mtp_drafter=object())
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            uid = Scheduler._route_to_vlm_mtp(
                sched,
                _make_route_request(),
                [object()],
                [42],
                lambda x: x,
                object(),
                logits_processors=[lambda toks, logits: logits],
            )
        assert uid is None
        assert "without vlm_mtp support" in caplog.text

    def test_budget_requires_positioned_verify_hook(self, caplog):
        """When the language model lacks speculative_logits_from_hidden,
        mlx-vlm would sample verify tokens without consulting the wrapper —
        routing must decline instead of silently dropping the budget."""
        lm = SimpleNamespace(rollback_speculative_cache=lambda *a, **k: None)
        sched = SimpleNamespace(
            _vlm_mtp_drafter=object(),
            _vlm_mtp_active={},
            model=SimpleNamespace(_language_model=lm),
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            uid = Scheduler._route_to_vlm_mtp(
                sched,
                _make_route_request(),
                [object()],
                [42],
                lambda x: x,
                object(),
                logits_processors=[_make_budget_processor(4)],
            )
        assert uid is None
        assert "speculative_logits_from_hidden" in caplog.text

    def test_happy_path_builds_processing_sampler(self, monkeypatch):
        """Routing with a budget processor must hand run_vlm_mtp_decode an
        MTPProcessingSampler whose first-bonus state is initialized."""
        captured = {}

        def fake_decode(**kwargs):
            captured.update(kwargs)

            def gen():
                yield kwargs["first_bonus"]

            return gen()

        monkeypatch.setattr(scheduler_mod, "run_vlm_mtp_decode", fake_decode)

        class FakeVLM:
            _language_model = SimpleNamespace(
                rollback_speculative_cache=lambda *a, **k: None,
                speculative_logits_from_hidden=lambda h: h,
            )

            def __call__(self, tokens, cache=None, **kwargs):
                logits = mx.zeros((1, 1, VOCAB))
                logits[0, 0, THINK] = 10.0
                return SimpleNamespace(
                    logits=logits,
                    hidden_states=mx.zeros((1, 1, 4)),
                    shared_kv_states={},
                )

        sched = SimpleNamespace(
            _vlm_mtp_drafter=SimpleNamespace(model=object()),
            _vlm_mtp_active={},
            _vlm_mtp_next_uid=-1,
            _vlm_mtp_draft_block_size=None,
            _model_suppress_tokens=set(),
            _stream=mx.default_device(),
            model=FakeVLM(),
            _get_stop_tokens=lambda: set(),
        )
        proc = _make_budget_processor(4)

        uid = Scheduler._route_to_vlm_mtp(
            sched,
            _make_route_request(),
            [SimpleNamespace(state=mx.zeros(1), offset=3)],
            [42],
            _argmax_sampler,
            object(),
            logits_processors=[proc],
        )

        assert uid is not None
        sampler = captured["sampler"]
        assert isinstance(sampler, MTPProcessingSampler)
        # First bonus was processed and recorded: THINK counted, position 1
        # checkpointed, history extended past the prompt.
        assert captured["first_bonus"] == THINK
        assert proc._thinking_tokens == 1
        assert 1 in sampler._snapshots
        assert sampler._history == PROMPT + [THINK]
        assert sched._vlm_mtp_active[uid].sampler is sampler


# ---------------------------------------------------------------------------
# 5. Positioned-hook visibility through _VLMAdapterMTPProxy (mRoPE gate)
# ---------------------------------------------------------------------------


def _hook(hidden):
    return hidden


def _make_adapter(*, mrope: bool, adapter_hook: bool, lm_hook: bool):
    """Build a fake VLM adapter + inner language model pair."""
    lm_attrs = {"rollback_speculative_cache": lambda *a, **k: None}
    if lm_hook:
        lm_attrs["speculative_logits_from_hidden"] = _hook
    lm = SimpleNamespace(**lm_attrs)
    adapter_attrs = {"_language_model": lm, "_uses_mrope": mrope}
    if adapter_hook:
        adapter_attrs["speculative_logits_from_hidden"] = _hook
    return SimpleNamespace(**adapter_attrs), lm


class TestPositionedHookVisibility:
    """The routing gate must probe what mlx-vlm's round loop will actually
    see. For mRoPE adapters (Qwen VLMs) _VLMAdapterMTPProxy hides the inner
    language model's ``speculative_*`` fast paths, so a check against the
    inner model passes while the loop silently falls back to plain
    vectorized sampling and drops the processors (#2399)."""

    @pytest.mark.parametrize("mrope", [False, True])
    @pytest.mark.parametrize("adapter_hook", [False, True])
    @pytest.mark.parametrize("lm_hook", [False, True])
    def test_helper_matches_real_proxy_resolution(
        self, mrope, adapter_hook, lm_hook
    ):
        """vlm_mtp_positioned_sampling_available == what the round loop
        resolves through the real proxy, for every combination."""
        adapter, lm = _make_adapter(
            mrope=mrope, adapter_hook=adapter_hook, lm_hook=lm_hook
        )
        proxy = _VLMAdapterMTPProxy(adapter, lm)
        # mlx-vlm's resolution (mtp.py): lm = model.language_model if
        # present else model; positioned path gated on the hook's presence.
        loop_lm = (
            proxy.language_model
            if hasattr(proxy, "language_model")
            else proxy
        )
        loop_sees_hook = hasattr(loop_lm, "speculative_logits_from_hidden")
        assert (
            vlm_mtp_positioned_sampling_available(adapter) == loop_sees_hook
        )

    def test_mrope_hides_inner_hook(self):
        """The maintainer-reported case: inner LM has the hook, adapter is
        mRoPE — the proxy hides it, so availability must be False."""
        adapter, lm = _make_adapter(
            mrope=True, adapter_hook=False, lm_hook=True
        )
        assert hasattr(lm, "speculative_logits_from_hidden")  # naive check
        assert not vlm_mtp_positioned_sampling_available(adapter)

    def test_adapter_level_hook_survives_mrope(self):
        """An mRoPE-safe hook implemented on the adapter itself is visible
        to the loop and keeps the vlm_mtp route open."""
        adapter, _ = _make_adapter(
            mrope=True, adapter_hook=True, lm_hook=False
        )
        assert vlm_mtp_positioned_sampling_available(adapter)

    def test_no_adapter_falls_back_to_model_probe(self):
        bare = SimpleNamespace(speculative_logits_from_hidden=_hook)
        assert vlm_mtp_positioned_sampling_available(bare)
        assert not vlm_mtp_positioned_sampling_available(SimpleNamespace())

    def test_route_gate_declines_mrope_adapter(self, caplog):
        """Regression for the silent-drop report on Qwen VLM targets: the
        gate must decline (falling back to BatchGenerator) even though the
        inner language model carries the hook."""
        adapter, _ = _make_adapter(
            mrope=True, adapter_hook=False, lm_hook=True
        )
        sched = SimpleNamespace(
            _vlm_mtp_drafter=object(),
            _vlm_mtp_active={},
            model=adapter,
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            uid = Scheduler._route_to_vlm_mtp(
                sched,
                _make_route_request(),
                [object()],
                [42],
                _argmax_sampler,
                object(),
                logits_processors=[_make_budget_processor(4)],
            )
        assert uid is None
        assert "positioned verify sampling is unavailable" in caplog.text

    def test_route_gate_passes_non_mrope_adapter(self, caplog):
        """Same shape, mRoPE off: the hook is visible through the proxy, so
        the gate passes; routing then declines on the empty last_tokens —
        the check immediately after the positioned gate — proving the
        positioned gate itself let the request through."""
        adapter, _ = _make_adapter(
            mrope=False, adapter_hook=False, lm_hook=True
        )
        sched = SimpleNamespace(
            _vlm_mtp_drafter=object(),
            _vlm_mtp_active={},
            model=adapter,
        )
        with caplog.at_level(logging.INFO, logger="omlx.scheduler"):
            uid = Scheduler._route_to_vlm_mtp(
                sched,
                _make_route_request(),
                [object()],
                [],
                _argmax_sampler,
                object(),
                logits_processors=[_make_budget_processor(4)],
            )
        assert uid is None
        assert "positioned verify sampling is unavailable" not in caplog.text
        assert "last_tokens empty" in caplog.text
