"""Scheduler-native speculative drafting for oMLX.

The provider owns only drafter state and speculative verification.  Request
admission, cache policy, token processors, stop matching and output delivery
remain Scheduler responsibilities.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import mlx.core as mx

from .handlers import DSparkLoadOptions, LoadedDrafter, resolve_handler
from .native_sampling import sample_probs, truncate_probs
from .native_target import Target
from .native_verify import speculative_sample_accept
from .target_adapters import adapt_target

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeResponse:
    uid: int
    token: int
    logprobs: Any = None
    finish_reason: str | None = None
    current_state: Any = None
    match_sequence: Any = None
    prompt_cache: Any = None


@dataclass
class DSparkRequestState:
    request: Any
    cache: list[Any]
    drafter_context: Any
    sampler: Any
    logits_processors: list[Any]
    state_machine: Any
    matcher_state: Any
    history: list[int]
    pending: int
    pending_logprobs: Any
    max_tokens: int
    cap: int
    max_cap: int
    auto_cap: bool
    native_speculation: bool
    emitted: int = 0
    draft_tokens: int = 0
    accepted_tokens: int = 0
    verify_rounds: int = 0
    ready: list[tuple[int, Any]] = field(default_factory=list)
    finished: bool = False
    accounted: bool = False
    auto_round: int = 0
    selected_cap: int | None = None
    cap_perf: dict[int, list[int]] = field(default_factory=dict)
    rng_state: list[Any] = field(default_factory=list)


@runtime_checkable
class SpeculativeDraftProvider(Protocol):
    format: str
    pairing_mode: str

    def probe(self) -> dict[str, Any]: ...
    def load(self) -> None: ...
    def close(self) -> None: ...
    def begin_prefill(self, request: Any, cached_tokens: int) -> Any: ...
    def prefill_context(
        self, state: Any, tokens: Any, cache: list[Any], offset: int
    ) -> None: ...
    def create_request_state(
        self,
        request: Any,
        cache: list[Any],
        last_tokens: list[int],
        sampler: Any,
        logits_processors: list[Any],
        state_machine: Any,
        prefill_state: Any = None,
    ) -> DSparkRequestState: ...
    def step(
        self, uid: int, state: DSparkRequestState
    ) -> list[SpeculativeResponse]: ...
    def step_batch(
        self, states: list[tuple[int, DSparkRequestState]]
    ) -> list[SpeculativeResponse]: ...
    def propose(self, state: DSparkRequestState) -> tuple[list[int], Any, Any, int]: ...
    def snapshot(self, state: DSparkRequestState) -> dict[str, int]: ...
    def rollback(
        self,
        state: DSparkRequestState,
        snapshot: dict[str, int],
        rejected: int,
        accepted: list[int],
    ) -> None: ...
    def commit(
        self,
        state: DSparkRequestState,
        snapshot: dict[str, int],
        fused: Any,
        committed: int,
    ) -> None: ...
    def update_after_verify(
        self, state: DSparkRequestState, *, drafted: int, accepted: int, elapsed_ns: int
    ) -> None: ...
    def abort(self, state: DSparkRequestState) -> None: ...
    def memory_usage(self) -> int: ...
    def stats(self, active_states: Any = ()) -> dict[str, Any]: ...


@dataclass
class _DeepSpecPrefill:
    context: list[Any]
    offset: int = 0


@dataclass
class _DFlashPrefill:
    context: list[Any]
    offset: int = 0


class NativeDSparkProvider:
    """A format-neutral provider attached to one native oMLX Scheduler."""

    def __init__(
        self,
        *,
        target_model: Any,
        tokenizer: Any,
        drafter_path: str,
        requested_format: str,
        max_draft_tokens: str | int | None,
        markov_mode: str,
        pairing_mode: str,
    ) -> None:
        self.target = Target(target_model, tokenizer)
        self.target_adapter = adapt_target(self.target)
        self.target.verify_tap()
        handler, probe = resolve_handler(drafter_path, requested_format)
        loaded = handler.load(
            DSparkLoadOptions(
                drafter=__import__("pathlib").Path(drafter_path),
                target_model=target_model,
                tokenizer=tokenizer,
                max_draft_tokens=max_draft_tokens,
                markov_mode=markov_mode,
            )
        )
        self.loaded: LoadedDrafter = loaded
        self.drafter = loaded.model
        self.config = loaded.config
        self.format = loaded.format
        self.pairing_mode = pairing_mode
        self._probe = probe
        self.tap = list(self.config.target_layer_ids)
        width = int(
            getattr(self.config, "max_proposal_tokens", 0)
            or getattr(self.config, "block_size", 1)
        )
        if self.format != "deepspec":
            width = max(1, width - 1)
        self.auto_cap = max_draft_tokens in (None, "auto")
        if self.auto_cap:
            self.cap = width
        else:
            self.cap = max(0, min(int(max_draft_tokens), width))
        self._totals = {
            "draft_tokens": 0,
            "accepted_tokens": 0,
            "verify_rounds": 0,
            "requests": 0,
        }
        # Capture immutable weight accounting while we are on EngineCore's
        # single MLX executor.  Status polling then reads only a Python int and
        # never traverses a live MLX module from the event-loop thread.
        self._weight_bytes = self._measure_drafter_weight_bytes()

    def probe(self) -> dict[str, Any]:
        to_dict = getattr(self._probe, "to_dict", None)
        return to_dict() if callable(to_dict) else dict(vars(self._probe))

    def load(self) -> None:
        """The provider is eagerly loaded by EngineCore's MLX executor."""
        return None

    def close(self) -> None:
        self.drafter = None
        self.loaded = None  # type: ignore[assignment]
        self.target = None  # type: ignore[assignment]
        self.target_adapter = None  # type: ignore[assignment]

    def greedy_smoke(self, prompt_ids: list[int], max_tokens: int = 4) -> None:
        """Prove a configured pair commits the target's greedy token sequence."""
        from ..request import SamplingParams

        if not prompt_ids:
            raise ValueError("dSpark smoke prompt must contain at least one token")

        # Independent target-only reference cache.
        baseline_cache = self.target.make_cache()
        logits = self.target.plain(mx.array([prompt_ids]), baseline_cache)
        baseline: list[int] = []
        for _ in range(max_tokens):
            token = int(mx.argmax(logits[0, -1]).item())
            baseline.append(token)
            logits = self.target.plain(mx.array([[token]]), baseline_cache)

        class _NoStop:
            @staticmethod
            def make_state() -> None:
                return None

            @staticmethod
            def match(state: Any, token: int) -> tuple[Any, None, object]:
                del token
                return state, None, object()

        params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            top_k=0,
        )
        request = SimpleNamespace(
            prompt_token_ids=list(prompt_ids),
            sampling_params=params,
            specprefill_indices=None,
        )
        speculative_cache = self.target.make_cache()
        prefill = self.begin_prefill(request, 0)
        if len(prompt_ids) > 1:
            self.prefill_context(
                prefill,
                mx.array([prompt_ids[:-1]]),
                speculative_cache,
                0,
            )

        def sampler(rows):
            return mx.argmax(rows, axis=-1)

        state = self.create_request_state(
            request,
            speculative_cache,
            prompt_ids[-1:],
            sampler,
            [],
            _NoStop(),
            prefill,
        )
        before_totals = dict(self._totals)
        actual: list[int] = []
        while not state.finished and len(actual) < max_tokens:
            actual.extend(response.token for response in self.step(-1, state))
        self._totals = before_totals
        if actual[:max_tokens] != baseline:
            mismatch = next(
                (
                    index
                    for index, pair in enumerate(zip(baseline, actual))
                    if pair[0] != pair[1]
                ),
                min(len(baseline), len(actual)),
            )
            raise ValueError(
                "target-only and dSpark greedy tokens diverged at position "
                f"{mismatch} (target={baseline[mismatch : mismatch + 1]}, "
                f"dspark={actual[mismatch : mismatch + 1]})"
            )

    def begin_prefill(self, request: Any, cached_tokens: int) -> Any:
        if self.format == "deepspec":
            state: Any = _DeepSpecPrefill(self.drafter.make_ctx_cache(), 0)
        else:
            state = _DFlashPrefill(self.drafter.make_cache(), 0)
        # oMLX prefix entries contain target cache state.  Drafter context is
        # deliberately request-local, so reconstruct it from the cached token
        # prefix without replacing or mutating the native target cache.
        if cached_tokens > 0:
            from mlx.utils import tree_flatten

            prefix = list(request.prompt_token_ids or [])[:cached_tokens]
            scratch = self.target.make_cache()
            for start in range(0, len(prefix), 512):
                piece = mx.array([prefix[start : start + 512]])
                self.prefill_context(state, piece, scratch, start)
                context_arrays = [
                    leaf
                    for _, leaf in tree_flatten(state.context)
                    if hasattr(leaf, "shape")
                ]
                mx.eval([cache.state for cache in scratch], context_arrays)
                mx.clear_cache()
            scratch = None
        state.offset = cached_tokens
        return state

    def prefill_context(
        self, state: Any, tokens: Any, cache: list[Any], offset: int
    ) -> None:
        if tokens.shape[1] == 0:
            return
        _, fused = self.target.prefill(tokens, cache, self.tap, want_logits=False)
        if self.format == "deepspec":
            self.drafter.update_context(fused, offset, state.context)
        else:
            self.drafter.prefill_context(fused, state.context, offset)
        state.offset = offset + int(tokens.shape[1])

    @staticmethod
    def _apply_processors(
        history: list[int], logits: Any, processors: list[Any]
    ) -> Any:
        row = logits[None] if len(logits.shape) == 1 else logits
        if processors:
            tokens = mx.array(history)
            for processor in processors:
                row = processor(tokens, row)
        return row

    @staticmethod
    def _logprobs(logits: Any) -> Any:
        row = logits[None] if len(logits.shape) == 1 else logits
        return (row - mx.logsumexp(row, axis=-1, keepdims=True))[0]

    def _sample(
        self, history: list[int], logits: Any, sampler: Any, processors: list[Any]
    ) -> tuple[int, Any]:
        row = self._apply_processors(history, logits, processors)
        logprobs = self._logprobs(row)
        token = sampler(logprobs[None] if len(logprobs.shape) == 1 else logprobs)
        mx.eval(token, logprobs)
        return int(token.item()), logprobs

    def create_request_state(
        self,
        request: Any,
        cache: list[Any],
        last_tokens: list[int],
        sampler: Any,
        logits_processors: list[Any],
        state_machine: Any,
        prefill_state: Any = None,
    ) -> DSparkRequestState:
        if not last_tokens:
            raise ValueError("dSpark requires a non-empty final prompt token")
        if prefill_state is None:
            prefill_state = self.begin_prefill(request, 0)
        ids = mx.array(last_tokens)[None]
        logits, fused = self.target.verify(ids, cache, self.tap)
        if self.format == "deepspec":
            self.drafter.update_context(
                fused, prefill_state.offset, prefill_state.context
            )
        else:
            self.drafter.prefill_context(
                fused, prefill_state.context, prefill_state.offset
            )
        history = list(request.prompt_token_ids or [])
        pending, lp = self._sample(history, logits[0, -1], sampler, logits_processors)
        # Arbitrary processors, min-p and XTC are handled exactly by target-only
        # rounds inside this provider.  Basic temperature/top-p/top-k can use
        # exact rejection sampling.
        params = request.sampling_params
        basic_sampling = (
            not logits_processors
            and getattr(request, "specprefill_indices", None) is None
            and float(getattr(params, "min_p", 0.0) or 0.0) == 0.0
            and float(getattr(params, "xtc_probability", 0.0) or 0.0) == 0.0
        )
        return DSparkRequestState(
            request=request,
            cache=cache,
            drafter_context=prefill_state.context,
            sampler=sampler,
            logits_processors=list(logits_processors or ()),
            state_machine=state_machine,
            matcher_state=state_machine.make_state(),
            history=history,
            pending=pending,
            pending_logprobs=lp,
            max_tokens=int(params.max_tokens),
            cap=self.cap,
            max_cap=self.cap,
            auto_cap=self.auto_cap,
            native_speculation=basic_sampling,
            rng_state=list(mx.random.state),
        )

    @staticmethod
    def snapshot(state: DSparkRequestState) -> dict[str, int]:
        return {
            "history_len": len(state.history),
            "emitted": state.emitted,
        }

    def rollback(
        self,
        state: DSparkRequestState,
        snapshot: dict[str, int],
        rejected: int,
        accepted: list[int],
    ) -> None:
        del snapshot
        self.target.rollback(state.cache, rejected, accepted)

    def commit(
        self,
        state: DSparkRequestState,
        snapshot: dict[str, int],
        fused: Any,
        committed: int,
    ) -> None:
        offset = snapshot["history_len"] - 1
        if self.format == "deepspec":
            self.drafter.update_context(
                fused[:, :committed], offset, state.drafter_context
            )
        else:
            self.drafter.prefill_context(
                fused[:, :committed], state.drafter_context, offset
            )

    def propose(self, state: DSparkRequestState) -> tuple[list[int], Any, Any, int]:
        return (
            self._draft_deepspec(state)
            if self.format == "deepspec"
            else self._draft_dflash(state)
        )

    @staticmethod
    def _choose_auto_cap(state: DSparkRequestState) -> int:
        """Choose cap from measured committed-token throughput, not acceptance.

        Each request samples every legal cap twice, including cap=0, then uses
        the cap with the highest committed tokens per wall-clock nanosecond.
        It periodically re-samples so prompt/domain changes can move the
        optimum without changing engine or cache state.
        """
        if not state.auto_cap or not state.native_speculation:
            return state.cap if state.native_speculation else 0
        explore = [min(2, state.max_cap)] + [
            cap for cap in range(0, state.max_cap + 1) if cap != min(2, state.max_cap)
        ]
        for cap in explore:
            if state.cap_perf.get(cap, [0, 0, 0])[2] < 2:
                return cap
        if state.selected_cap is None:
            state.selected_cap = max(
                explore,
                key=lambda cap: (
                    state.cap_perf[cap][0] / max(1, state.cap_perf[cap][1]),
                    -cap,
                ),
            )
        if state.auto_round >= (2 * len(explore) + 64):
            state.auto_round = 0
            state.selected_cap = None
            state.cap_perf.clear()
            return explore[0]
        return state.selected_cap

    def update_after_verify(
        self,
        state: DSparkRequestState,
        *,
        drafted: int,
        accepted: int,
        elapsed_ns: int,
    ) -> None:
        used_cap = state.cap
        perf = state.cap_perf.setdefault(used_cap, [0, 0, 0])
        # Every verify round commits accepted drafts plus one target token.
        perf[0] += accepted + 1
        perf[1] += max(1, elapsed_ns)
        perf[2] += 1
        state.auto_round += 1
        state.draft_tokens += drafted
        state.accepted_tokens += accepted
        state.verify_rounds += int(drafted > 0)

    def _target_only(self, state: DSparkRequestState) -> list[tuple[int, Any]]:
        ids = mx.array([[state.pending]])
        logits, fused = self.target.verify(ids, state.cache, self.tap)
        if self.format == "deepspec":
            self.drafter.update_context(
                fused, len(state.history) - 1, state.drafter_context
            )
        else:
            self.drafter.prefill_context(
                fused, state.drafter_context, len(state.history) - 1
            )
        token, lp = self._sample(
            state.history + [state.pending],
            logits[0, -1],
            state.sampler,
            state.logits_processors,
        )
        return [(token, lp)]

    def _propose_deepspec(self, state: DSparkRequestState) -> tuple[list[int], Any]:
        cfg = self.config
        cap = max(1, state.cap)
        block = [state.pending] + [int(cfg.mask_token_id)] * (int(cfg.block_size) - 1)
        hidden = self.drafter.backbone(
            self.drafter.embed(mx.array([block])),
            len(state.history) - 1,
            state.drafter_context,
        )
        base = self.drafter.compute_logits(hidden[:, :cap])[0]
        temp = float(state.request.sampling_params.temperature)
        if temp > 0:
            draft_arr, q = self.drafter.sample_block_probs(
                base,
                state.pending,
                temp,
                float(state.request.sampling_params.top_p),
                int(state.request.sampling_params.top_k),
            )
            mx.eval(draft_arr, q)
            draft = [int(x) for x in draft_arr.tolist()]
        else:
            draft_arr = self.drafter.sample_block(base, state.pending)
            draft = [int(x) for x in draft_arr.tolist()]
            q = None
        return draft, q

    def _draft_deepspec(
        self, state: DSparkRequestState
    ) -> tuple[list[int], Any, Any, int]:
        draft, q = self._propose_deepspec(state)
        verify_ids = mx.array([[state.pending] + draft])
        logits, fused = self.target.verify(verify_ids, state.cache, self.tap)
        return draft, q, (logits, fused), len(draft)

    def _propose_dflash(self, state: DSparkRequestState) -> tuple[list[int], Any]:
        cfg = self.config
        cap = max(1, state.cap)
        block = mx.array(
            [[state.pending] + [int(cfg.mask_token_id)] * (int(cfg.block_size) - 1)]
        )
        base = self.drafter.propose(block, state.drafter_context)[0][:cap]
        markov = getattr(self.drafter, "markov_head", None)
        markov_on = markov is not None and getattr(self.drafter, "markov_enabled", True)
        draft_to_target = getattr(self.drafter, "draft_to_target", None)
        temp = float(state.request.sampling_params.temperature)
        draft, q_rows, previous = [], [], state.pending
        for i in range(cap):
            row = base[i]
            if markov_on:
                row = row + markov.step_bias(mx.array([previous]))[0]
            if temp > 0:
                q_compact = truncate_probs(
                    mx.softmax(row / temp, axis=-1),
                    float(state.request.sampling_params.top_p),
                    int(state.request.sampling_params.top_k),
                )
                compact_token = int(sample_probs(q_compact).item())
            else:
                compact_token = int(mx.argmax(row).item())
            if draft_to_target is not None:
                token = int(draft_to_target[compact_token].item())
                if temp > 0:
                    q = mx.zeros((int(self.config.vocab_size),), dtype=q_compact.dtype)
                    q[draft_to_target] = q_compact
                    q_rows.append(q)
            else:
                token = compact_token
                if temp > 0:
                    q_rows.append(q_compact)
            draft.append(token)
            previous = token
        return draft, (mx.stack(q_rows) if q_rows else None)

    def _draft_dflash(
        self, state: DSparkRequestState
    ) -> tuple[list[int], Any, Any, int]:
        draft, q = self._propose_dflash(state)
        verify_ids = mx.array([[state.pending] + draft])
        logits, fused = self.target.verify(verify_ids, state.cache, self.tap)
        return draft, q, (logits, fused), len(draft)

    def _propose_only(self, state: DSparkRequestState) -> tuple[list[int], Any]:
        return (
            self._propose_deepspec(state)
            if self.format == "deepspec"
            else self._propose_dflash(state)
        )

    def _finish_verified(
        self,
        state: DSparkRequestState,
        snapshot: dict[str, int],
        draft: list[int],
        q: Any,
        logits: Any,
        fused: Any,
        started_ns: int,
        *,
        capture: Any = None,
    ) -> list[tuple[int, Any]]:
        params = state.request.sampling_params
        temp = float(params.temperature)
        if temp > 0:
            n, replacement = speculative_sample_accept(
                logits, draft, q, temp, float(params.top_p), int(params.top_k)
            )
            committed = draft[:n] + [replacement]
        else:
            target_tokens = [int(x) for x in mx.argmax(logits, axis=-1).tolist()]
            n = 0
            while n < len(draft) and draft[n] == target_tokens[n]:
                n += 1
            committed = draft[:n] + [target_tokens[n]]
        self.target.rollback(
            state.cache,
            len(draft) - n,
            draft[:n],
            capture=capture,
            spec_width=len(draft) + 1,
        )
        self.commit(state, snapshot, fused, n + 1)
        mx.eval(logits, fused)
        self.update_after_verify(
            state,
            drafted=len(draft),
            accepted=n,
            elapsed_ns=time.perf_counter_ns() - started_ns,
        )
        return [
            (int(token), self._logprobs(logits[i])) for i, token in enumerate(committed)
        ]

    def _speculative(self, state: DSparkRequestState) -> list[tuple[int, Any]]:
        snapshot = self.snapshot(state)
        started_ns = time.perf_counter_ns()
        draft, q, (logits, fused), _ = self.propose(state)
        return self._finish_verified(
            state, snapshot, draft, q, logits[0], fused, started_ns
        )

    def step(self, uid: int, state: DSparkRequestState) -> list[SpeculativeResponse]:
        global_rng = list(mx.random.state)
        if state.rng_state:
            mx.random.state[:] = list(state.rng_state)
        try:
            return self._step_with_request_rng(uid, state)
        finally:
            state.rng_state = list(mx.random.state)
            mx.random.state[:] = global_rng

    def _step_with_request_rng(
        self, uid: int, state: DSparkRequestState
    ) -> list[SpeculativeResponse]:
        if state.finished:
            return []
        if state.ready:
            pairs = state.ready
            state.ready = []
        elif state.emitted == 0:
            pairs = [(state.pending, state.pending_logprobs)]
        else:
            state.cap = self._choose_auto_cap(state)
            started_ns = time.perf_counter_ns()
            if state.cap == 0 or not state.native_speculation:
                pairs = self._target_only(state)
                mx.eval(*[lp for _, lp in pairs])
                if state.auto_cap and state.native_speculation:
                    self.update_after_verify(
                        state,
                        drafted=0,
                        accepted=0,
                        elapsed_ns=time.perf_counter_ns() - started_ns,
                    )
            else:
                pairs = self._speculative(state)

        return self._emit_pairs(uid, state, pairs)

    def _emit_pairs(
        self,
        uid: int,
        state: DSparkRequestState,
        pairs: list[tuple[int, Any]],
    ) -> list[SpeculativeResponse]:
        responses: list[SpeculativeResponse] = []
        for pair_index, (token, lp) in enumerate(pairs):
            if state.emitted >= state.max_tokens:
                break
            state.emitted += 1
            state.history.append(int(token))
            state.pending = int(token)
            matcher, match_sequence, current = state.state_machine.match(
                state.matcher_state, int(token)
            )
            state.matcher_state = matcher
            for processor in state.logits_processors:
                accept = getattr(processor, "accept_token", None)
                if callable(accept):
                    accept(int(token))
            finish = None
            if match_sequence is not None and current is None:
                finish = "stop"
            elif state.emitted >= state.max_tokens:
                finish = "length"
            responses.append(
                SpeculativeResponse(
                    uid=uid,
                    token=int(token),
                    logprobs=lp,
                    finish_reason=finish,
                    current_state=current,
                    match_sequence=match_sequence,
                    prompt_cache=(
                        state.cache if finish and pair_index == len(pairs) - 1 else None
                    ),
                )
            )
            if finish:
                state.finished = True
                break
        if state.finished and not state.accounted:
            self._totals["draft_tokens"] += state.draft_tokens
            self._totals["accepted_tokens"] += state.accepted_tokens
            self._totals["verify_rounds"] += state.verify_rounds
            self._totals["requests"] += 1
            state.accounted = True
        return responses

    def step_batch(
        self, states: list[tuple[int, DSparkRequestState]]
    ) -> list[SpeculativeResponse]:
        """Advance native Scheduler rows, batching equal-width target verifies."""
        responses: list[SpeculativeResponse] = []
        groups: dict[int, list[tuple[int, DSparkRequestState, Any, Any, int, Any]]] = {}

        for uid, state in states:
            if state.finished or state.ready or state.emitted == 0:
                responses.extend(self.step(uid, state))
                continue
            state.cap = self._choose_auto_cap(state)
            if state.cap == 0 or not state.native_speculation:
                responses.extend(self.step(uid, state))
                continue

            global_rng = list(mx.random.state)
            if state.rng_state:
                mx.random.state[:] = list(state.rng_state)
            snapshot = self.snapshot(state)
            started_ns = time.perf_counter_ns()
            try:
                draft, q = self._propose_only(state)
            finally:
                state.rng_state = list(mx.random.state)
                mx.random.state[:] = global_rng
            groups.setdefault(len(draft), []).append(
                (uid, state, draft, q, started_ns, snapshot)
            )

        for _width, group in groups.items():
            if len(group) == 1:
                uid, state, draft, q, started_ns, snapshot = group[0]
                row_logits, row_fused = self.target.verify(
                    mx.array([[state.pending] + draft]), state.cache, self.tap
                )
                global_rng = list(mx.random.state)
                mx.random.state[:] = list(state.rng_state)
                try:
                    pairs = self._finish_verified(
                        state,
                        snapshot,
                        draft,
                        q,
                        row_logits[0],
                        row_fused,
                        started_ns,
                    )
                finally:
                    state.rng_state = list(mx.random.state)
                    mx.random.state[:] = global_rng
                responses.extend(self._emit_pairs(uid, state, pairs))
                continue
            ids = mx.array([[state.pending] + draft for _, state, draft, *_ in group])
            try:
                logits, fused, row_caches, captures = self.target.verify_batch(
                    ids, [state.cache for _, state, *_ in group], self.tap
                )
            except (TypeError, ValueError):
                # A model cache family without native merge/extract still remains
                # correct and inside DSparkEngine; capability stats expose this.
                self._totals["batch_verify_fallbacks"] = (
                    self._totals.get("batch_verify_fallbacks", 0) + 1
                )
                if self._totals["batch_verify_fallbacks"] == 1:
                    logger.warning(
                        "native batched target verify unavailable for %s; "
                        "using row verifies inside DSparkEngine",
                        type(group[0][1].cache[0]).__name__,
                    )
                for uid, state, draft, q, started_ns, snapshot in group:
                    row_logits, row_fused = self.target.verify(
                        mx.array([[state.pending] + draft]), state.cache, self.tap
                    )
                    global_rng = list(mx.random.state)
                    mx.random.state[:] = list(state.rng_state)
                    try:
                        pairs = self._finish_verified(
                            state,
                            snapshot,
                            draft,
                            q,
                            row_logits[0],
                            row_fused,
                            started_ns,
                        )
                    finally:
                        state.rng_state = list(mx.random.state)
                        mx.random.state[:] = global_rng
                    responses.extend(self._emit_pairs(uid, state, pairs))
                continue

            self._totals["batched_verify_rounds"] = (
                self._totals.get("batched_verify_rounds", 0) + 1
            )
            self._totals["batched_verify_rows"] = self._totals.get(
                "batched_verify_rows", 0
            ) + len(group)
            if self._totals["batched_verify_rounds"] == 1:
                logger.info(
                    "native batched target verify active: rows=%d proposal_width=%d",
                    len(group),
                    _width,
                )
            for row_index, (uid, state, draft, q, started_ns, snapshot) in enumerate(
                group
            ):
                state.cache = row_caches[row_index]
                global_rng = list(mx.random.state)
                mx.random.state[:] = list(state.rng_state)
                try:
                    pairs = self._finish_verified(
                        state,
                        snapshot,
                        draft,
                        q,
                        logits[row_index],
                        fused[row_index : row_index + 1],
                        started_ns,
                        capture=captures[row_index],
                    )
                finally:
                    state.rng_state = list(mx.random.state)
                    mx.random.state[:] = global_rng
                responses.extend(self._emit_pairs(uid, state, pairs))
        return responses

    def abort(self, state: DSparkRequestState) -> None:
        state.finished = True
        state.ready.clear()

    def memory_usage(self) -> int:
        return self._weight_bytes

    def _measure_drafter_weight_bytes(self) -> int:
        try:
            from mlx.utils import tree_flatten

            return sum(
                int(v.nbytes) for _, v in tree_flatten(self.drafter.parameters())
            )
        except Exception:
            return 0

    @staticmethod
    def _tree_bytes(value: Any) -> int:
        try:
            from mlx.utils import tree_flatten

            return sum(
                int(getattr(leaf, "nbytes", 0) or 0) for _, leaf in tree_flatten(value)
            )
        except Exception:
            return 0

    def stats(self, active_states: Any = ()) -> dict[str, Any]:
        totals = dict(self._totals)
        active = list(active_states or ())
        totals["draft_tokens"] += sum(state.draft_tokens for state in active)
        totals["accepted_tokens"] += sum(state.accepted_tokens for state in active)
        totals["verify_rounds"] += sum(state.verify_rounds for state in active)
        totals.update(
            {
                "format": self.format,
                "pairing_mode": self.pairing_mode,
                "cap": self.cap,
                "markov": bool(getattr(self.drafter, "markov_enabled", False)),
                "quantization": dict(self.loaded.quantization),
                "drafter_context_bytes": sum(
                    self._tree_bytes(state.drafter_context) for state in active
                ),
                "drafter_weight_bytes": self.memory_usage(),
                "active_caps": [state.cap for state in active],
                "target_adapter": self.target_adapter.family,
            }
        )
        return totals
