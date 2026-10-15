# SPDX-License-Identifier: Apache-2.0
"""Row-wise DFlash drafting against per-request context rings."""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ...patches.mlx_lm_mtp import batch_generator as bg
from ...patches.qwen35_verify_qmm import set_verify_qmm_armed
from . import kernels, selection, variants

# Finished requests whose drafter context stays available for prefix hits.
_KEPT_REQUESTS = 8
# Released rows waiting for the scheduler to name their tokens.
_RELEASED_ROWS = 32


@dataclass
class _RowContext:
    """Per-request drafter state: the context ring plus unfed hidden rows.

    ``keys``/``values`` hold one ``(n_kv, C, D)`` ring per draft layer with
    RoPE already applied, so slot order does not matter for the non-causal
    context attention. ``fed`` counts every token the ring has seen; the
    write slot is ``fed % C``. ``pos`` holds each slot's absolute position,
    and only the newest ``W - 1`` positions are attended.
    """

    pending: list[mx.array] = field(default_factory=list)
    pending_len: int = 0
    fed: int = 0
    keys: list[mx.array] | None = None
    values: list[mx.array] | None = None
    pos: mx.array | None = None
    # Absolute prompt position after the newest queued context row, or None
    # while the request's first position is unknown.
    end: int | None = None
    # Context restored for a prefix-cache hit, already projected by ``fc``
    # and ``hidden_norm``; it precedes ``pending``.
    restored: mx.array | None = None
    # The newest fed rows after projection, as chunks, oldest first, and the
    # absolute position after them. Rows still pending are not in history.
    history: list[mx.array] = field(default_factory=list)
    history_len: int = 0
    history_end: int | None = None


@dataclass
class _KeptContext:
    """Projected context of a finished request, for its prefix-cache hits."""

    tokens: list[int]
    context: mx.array
    end: int


@dataclass
class _Cohort:
    """Stacked rings for the rows of one generation batch, in row order."""

    uids: tuple
    keys: list[mx.array]
    values: list[mx.array]
    pos: mx.array


@dataclass
class _Predraft:
    """A block drafted before the host read how many verify rows committed."""

    uids: tuple
    keys: list[mx.array]
    values: list[mx.array]
    pos: mx.array
    bases: list[int]
    unfed: list[int]
    proposals: list[tuple]
    contexts: list[mx.array] = field(default_factory=list)


def _concat_captured(captured: Sequence[mx.array]) -> mx.array:
    """Join per-layer captures ``[(1, n, H), ...]`` into ``(1, n, K*H)``."""
    if len(captured) == 1:
        return captured[0]
    return mx.concatenate(list(captured), axis=-1)


class DFlashDrafter:
    """Row-wise DFlash drafting against per-request context caches."""

    def __init__(
        self,
        model: nn.Module,
        *,
        block_size: int,
        source_path: str,
        adaptive: bool = True,
        window_size: int | None = None,
        value_scale: float | None = None,
        mask_embedding: mx.array | None = None,
    ):
        self.model = model
        self.source_path = source_path
        self._window_size = window_size
        # MiMo scales attention values and embeds masked block positions with
        # a trained vector instead of the mask token's embedding.
        self.value_scale = value_scale
        self.mask_embedding = mask_embedding
        self.target_layer_ids: list[int] = [
            int(i) for i in model.config.target_layer_ids
        ]
        self.block_size = int(block_size)
        # Lightning MTP reads this as the maximum draft depth.
        self.depth = self.block_size - 1
        # Adaptive: Lightning MTP's depth controllers choose how many drafts
        # of each block to verify. Otherwise the whole block is verified.
        self.adaptive = bool(adaptive)
        attn = model.layers[0].self_attn
        # Laguna drafts with causal block attention; DFlash v1/v2 blocks
        # attend both ways unless the checkpoint says otherwise.
        self.causal = bool(getattr(attn, "causal", getattr(attn, "is_causal", False)))
        self._rows: dict[Any, _RowContext] = {}
        self._cohort: _Cohort | None = None
        # Prefill captures wait under the request id until the scheduler
        # learns the row uid at insert time.
        self._request_seeds: dict[str, list[mx.array]] = {}
        self._request_starts: dict[str, int] = {}
        self._request_restored: dict[str, tuple[mx.array, int]] = {}
        self._request_uids: dict[str, Any] = {}
        self._uid_requests: dict[Any, str] = {}
        # Contexts of rows the batch released, until the scheduler names
        # their tokens, and of finished requests kept for prefix hits.
        self._released: OrderedDict[Any, tuple[list[mx.array], int]] = OrderedDict()
        self._kept: OrderedDict[str, _KeptContext] = OrderedDict()
        # Row order of the generation batch whose ordinary decode step is
        # running, so a plain forward can be attributed to rows.
        self.scope_uids: tuple | None = None
        self._predraft: _Predraft | None = None

    @property
    def window(self) -> int:
        """Context tokens the drafter attends to; older captures are dropped."""
        if self._window_size:
            return int(self._window_size)
        config = self.model.config
        explicit = getattr(config, "draft_window_size", None)
        if explicit:
            return int(explicit)
        return int(getattr(config, "sliding_window", None) or 2048)

    @property
    def kind(self) -> str:
        return "dflash2" if hasattr(self.model, "candidate_selector") else "dflash"

    def _row(self, uid: Any) -> _RowContext:
        row = self._rows.get(uid)
        if row is None:
            row = _RowContext()
            self._rows[uid] = row
        return row

    @property
    def ring_slots(self) -> int:
        # mlx-vlm keeps window - 1 context tokens in front of the block.
        return self.window - 1

    @property
    def capacity(self) -> int:
        # A block of spare slots lets a predraft write every verify row
        # without overwriting a position that is still attended.
        return self.ring_slots + self.block_size

    def seed(self, uid: Any, captured: Sequence[mx.array]) -> None:
        """Queue prefill captures ``[(1, n, H), ...]`` as context for ``uid``."""
        if captured:
            row = self._row(uid)
            self._queue(row, _concat_captured(captured))
            self._bound_pending(row)

    @staticmethod
    def _queue(row: _RowContext, context: mx.array) -> None:
        row.pending.append(context)
        row.pending_len += int(context.shape[1])
        if row.end is not None:
            row.end += int(context.shape[1])

    @staticmethod
    def _take_pending(row: _RowContext) -> list[mx.array]:
        pending, row.pending, row.pending_len = row.pending, [], 0
        return pending

    def _bound_pending(self, row: _RowContext) -> None:
        # Only the newest ring_slots rows are ever attended, so rows queued
        # while drafting is off are dropped here; fed still counts them.
        drop = row.pending_len - self.ring_slots
        if row.restored is not None:
            drop += int(row.restored.shape[1])
            older = min(drop, int(row.restored.shape[1]))
            if older > 0:
                row.fed += older
                drop -= older
                kept = row.restored[:, older:]
                row.restored = kept if int(kept.shape[1]) else None
        if drop <= 0:
            return
        row.fed += drop
        row.pending_len -= drop
        while drop > 0:
            head = row.pending[0]
            if int(head.shape[1]) <= drop:
                drop -= int(head.shape[1])
                row.pending.pop(0)
            else:
                row.pending[0] = head[:, drop:]
                drop = 0

    def observe(self, uids: Iterable[Any], captured: Sequence[mx.array] | None) -> None:
        """Queue one committed position per row from a batched ``(B, 1, H)`` capture."""
        if not captured:
            return
        for index, uid in enumerate(uids):
            self.seed(uid, [layer[index : index + 1] for layer in captured])

    def seed_request(
        self,
        request_id: str,
        captured: Sequence[mx.array],
        position: int | None = None,
    ) -> None:
        """Queue prefill captures for a request that has no row uid yet.

        ``position`` is the prompt position of the first captured row.
        """
        if captured:
            self._request_seeds.setdefault(request_id, []).append(
                _concat_captured(captured)
            )
            if position is not None:
                self._request_starts.setdefault(request_id, int(position))

    def prepare_request(
        self, request_id: str, prompt_tokens: Sequence[int], cached: int
    ) -> int:
        """Restore the context of a prefix-cache hit from a finished request.

        Cached tokens skip prefill, so the drafter never sees their layer
        captures. A kept request whose tokens match the first ``cached``
        prompt tokens supplies the rows just before ``cached``. Returns the
        number of restored rows. Only the first call per request counts: a
        prefill resumed after a pause keeps what its first attempt captured.
        """
        cached = int(cached)
        if cached <= 0 or request_id in self._request_starts:
            return 0
        self._request_starts[request_id] = cached
        prefix = list(prompt_tokens[:cached])
        for key in reversed(self._kept):
            kept = self._kept[key]
            start = kept.end - int(kept.context.shape[1])
            if not start < cached <= kept.end or kept.tokens[:cached] != prefix:
                continue
            take = min(self.ring_slots, cached - start)
            offset = cached - start
            self._request_restored[request_id] = (
                kept.context[:, offset - take : offset],
                cached - take,
            )
            self._kept.move_to_end(key)
            return take
        return 0

    def bind_uid(self, request_id: str, uid: Any) -> None:
        self._request_uids[request_id] = uid
        self._uid_requests[uid] = request_id
        restored = self._request_restored.pop(request_id, None)
        start = self._request_starts.pop(request_id, None)
        seeds = self._request_seeds.pop(request_id, None)
        if restored is None and not seeds and start is None:
            return
        row = self._row(uid)
        if restored is not None:
            row.restored, start = restored
            row.end = start + int(row.restored.shape[1])
        else:
            row.end = start
        for seed in seeds or ():
            self._queue(row, seed)
        self._bound_pending(row)

    def release_request(
        self, request_id: str, tokens: Sequence[int] | None = None
    ) -> None:
        """Forget a request; with its ``tokens``, keep its context for hits."""
        self._request_seeds.pop(request_id, None)
        self._request_starts.pop(request_id, None)
        self._request_restored.pop(request_id, None)
        uid = self._request_uids.pop(request_id, None)
        if uid is None:
            return
        self._uid_requests.pop(uid, None)
        released = self._released.pop(uid, None)
        if tokens is None:
            return
        row = self._rows.get(uid)
        if row is not None:
            released = (row.history, row.history_end) if row.history else None
        if released is None:
            return
        history, end = released
        if end is None or end > len(tokens):
            return
        context = history[0] if len(history) == 1 else mx.concatenate(history, axis=1)
        if int(context.shape[1]) > self.ring_slots:
            # A tail view keeps the discarded history's buffer alive.
            context = mx.contiguous(context[:, -self.ring_slots :])
        # Realize per-row copies too: otherwise an idle snapshot can pin the
        # whole drafting batch until its next prefix hit.
        mx.async_eval(context)
        self._kept[request_id] = _KeptContext(
            tokens=list(tokens[:end]),
            context=context,
            end=end,
        )
        while len(self._kept) > _KEPT_REQUESTS:
            self._kept.popitem(last=False)

    @contextmanager
    def decode_scope(self, uids: Iterable[Any]):
        previous = self.scope_uids
        self.scope_uids = tuple(uids)
        try:
            yield
        finally:
            self.scope_uids = previous

    def release(self, uids: Iterable[Any]) -> None:
        self._predraft = None
        for uid in uids:
            row = self._rows.pop(uid, None)
            if row is None:
                continue
            self._detach_cohort()
            # Keep the context only while its request can still claim it.
            if row.history and uid in self._uid_requests:
                self._released[uid] = (row.history, row.history_end)
                while len(self._released) > _RELEASED_ROWS:
                    self._released.popitem(last=False)

    def reset_rows(self) -> None:
        """Forget every row after the batch generator was replaced.

        A new generator numbers rows from 0 again. Requests that have not
        bound a row yet and finished requests' contexts stay.
        """
        self._predraft = None
        self._rows.clear()
        self._cohort = None
        self._request_uids.clear()
        self._uid_requests.clear()
        self._released.clear()

    def clear(self) -> None:
        self.reset_rows()
        self._request_seeds.clear()
        self._request_starts.clear()
        self._request_restored.clear()
        self._kept.clear()

    def _remember(self, row: _RowContext, context: mx.array) -> None:
        """Keep the newest ring_slots projected rows of a fed context."""
        if row.end is None or not int(context.shape[1]):
            return
        # A slice of the batched projection would keep the whole buffer.
        row.history.append(mx.contiguous(context))
        row.history_end = row.end
        row.history_len += int(context.shape[1])
        while row.history_len - int(row.history[0].shape[1]) >= self.ring_slots:
            row.history_len -= int(row.history.pop(0).shape[1])

    def context_length(self, uid: Any) -> int:
        row = self._rows.get(uid)
        return 0 if row is None else row.fed

    def draft(self, jobs: Sequence[tuple]) -> None:
        """Draft the next block for every job after its cycle committed.

        Each job is ``(gen_batch, state, captured, committed, prev_buf)`` in the
        Lightning MTP draft-job layout: ``captured`` holds the per-layer hidden
        rows ``[(1, n, H), ...]`` of the positions just committed and
        ``committed[-1]`` is the newest committed token, which anchors the
        block. Writes ``state.drafts`` and clears the head-only logprob lists.
        """
        rows = []
        for gen_batch, state, captured, committed, _prev_buf in jobs:
            row = self._row(state.uid)
            if captured:
                self._queue(row, _concat_captured(captured))
            if not row.pending:
                raise RuntimeError(
                    f"DFlash drafter has no context for uid={state.uid!r}"
                )
            pending = self._take_pending(row)
            context = (
                pending[0] if len(pending) == 1 else mx.concatenate(pending, axis=1)
            )
            anchor = committed.reshape(-1)[-1:].astype(mx.int32)
            # Stochastic rows draw proposals from a per-row sampler; the
            # normalized draft distribution feeds Leviathan acceptance.
            sampler = None
            if gen_batch is not None and not bg._is_greedy(gen_batch):
                sampler = bg._resolve_draft_sampler(gen_batch, state)
            rows.append((state, row, context, anchor, sampler))
        # The drafter's projections see rows x block inputs like a verify
        # forward, so the same small-M kernels apply.
        started = time.perf_counter()
        set_verify_qmm_armed(True)
        try:
            proposals = self._draft_batched(rows)
        finally:
            set_verify_qmm_armed(False)
        for (state, *_), (tokens, accept_lps) in zip(rows, proposals):
            state.drafts = tokens.reshape(-1).astype(mx.uint32)
            mx.async_eval(state.drafts)
            state.draft_lps = []
            state.draft_accept_lps = accept_lps
        # Dispatch time shared across the rows; the GPU work overlaps the
        # next verify like the MTP head's async draft.
        share = (time.perf_counter() - started) * 1000 / max(1, len(rows))
        for state, *_ in rows:
            stats = getattr(state, "stats", None)
            if stats is not None:
                stats.mtp_head_ms += share

    def predraft(self, gen_batch, state, captured, count, anchor) -> bool:
        """Queue the next block before the host reads this cycle's acceptance.

        ``captured`` holds every verify row; ``count`` (a lazy ``(1,)`` array)
        of them commit and ``anchor`` is the lazy last emitted token. The GPU
        drafts while the host settles the cycle; ``adopt_predraft`` then keeps
        the block if the host commits the same count.
        """
        if not captured:
            return False
        row = self._row(state.uid)
        if row.restored is not None:
            return False
        unfed = row.pending_len
        context = mx.concatenate([*row.pending, _concat_captured(captured)], axis=1)
        if int(context.shape[1]) > self.ring_slots:
            return False
        sampler = None
        if gen_batch is not None and not bg._is_greedy(gen_batch):
            sampler = bg._resolve_draft_sampler(gen_batch, state)
        set_verify_qmm_armed(True)
        try:
            proposals = self._draft_batched(
                [(state, row, context, anchor.reshape(1).astype(mx.int32), sampler)],
                counts=[count.reshape(1) + unfed],
                commit=False,
            )
        finally:
            set_verify_qmm_armed(False)
        self._predraft.unfed = [unfed]
        tokens, accept_lps = proposals[0]
        mx.async_eval(tokens, *selection.q_arrays(accept_lps))
        return True

    def adopt_predraft(self, state, count: int) -> None:
        """Commit the queued block: ``count`` verify rows entered the context."""
        pre = self._predraft
        self._predraft = None
        cohort = self._cohort
        if pre is None or cohort is None or cohort.uids != pre.uids:
            raise RuntimeError("DFlash predraft does not match the drafting cohort")
        cohort.keys, cohort.values, cohort.pos = pre.keys, pre.values, pre.pos
        row = self._row(state.uid)
        row.fed = pre.bases[0] + pre.unfed[0] + int(count)
        self._take_pending(row)
        if row.end is not None:
            row.end += int(count)
        self._remember(row, pre.contexts[0][:, : pre.unfed[0] + int(count)])
        tokens, accept_lps = pre.proposals[0]
        state.drafts = tokens.reshape(-1).astype(mx.uint32)
        state.draft_lps = []
        state.draft_accept_lps = accept_lps

    def discard_predraft(self) -> None:
        self._predraft = None

    # --- batched forward -------------------------------------------------

    def _detach_cohort(self) -> None:
        cohort = self._cohort
        if cohort is None:
            return
        for index, uid in enumerate(cohort.uids):
            row = self._rows.get(uid)
            if row is not None:
                row.keys = [layer[index] for layer in cohort.keys]
                row.values = [layer[index] for layer in cohort.values]
                row.pos = cohort.pos[index]
        self._cohort = None

    def _assemble_cohort(self, uids: Sequence[Any], dtype) -> _Cohort:
        uids = tuple(uids)
        cohort = self._cohort
        if cohort is not None and cohort.uids == uids:
            return cohort
        self._detach_cohort()
        layers = self.model.layers
        attn = layers[0].self_attn
        shape = (attn.n_kv_heads, self.capacity, attn.head_dim)
        keys, values = [], []
        for layer_index in range(len(layers)):
            keys.append(
                mx.stack(
                    [
                        (
                            self._rows[uid].keys[layer_index]
                            if self._rows[uid].keys is not None
                            else mx.zeros(shape, dtype=dtype)
                        )
                        for uid in uids
                    ]
                )
            )
            values.append(
                mx.stack(
                    [
                        (
                            self._rows[uid].values[layer_index]
                            if self._rows[uid].values is not None
                            else mx.zeros(shape, dtype=dtype)
                        )
                        for uid in uids
                    ]
                )
            )
        empty = mx.full((self.capacity,), -1, dtype=mx.int32)
        pos = mx.stack(
            [
                self._rows[uid].pos if self._rows[uid].pos is not None else empty
                for uid in uids
            ]
        )
        self._cohort = _Cohort(uids=uids, keys=keys, values=values, pos=pos)
        return self._cohort

    def _draft_batched(self, rows: Sequence[tuple], counts=None, commit=True):
        """One draft forward for every row; mirrors DFlashAttention rowwise math.

        ``counts`` optionally gives each row's committed context rows as a lazy
        ``(1,)`` array; the rest of its context is written but never attended.
        Without ``commit`` the new rings wait in ``self._predraft``.
        """
        model = self.model
        uids = [state.uid for state, *_ in rows]
        slots = self.ring_slots
        capacity = self.capacity
        batch = len(rows)
        block = self.block_size
        fc = model.encoder.fc if hasattr(model, "encoder") else model.fc
        dtype = fc.weight.dtype if hasattr(fc, "weight") else mx.bfloat16
        if hasattr(fc, "scales"):
            dtype = fc.scales.dtype

        # Context segments: keep at most ``slots`` newest tokens per row and
        # count the rest as skipped so absolute positions stay aligned. A
        # restored context precedes the captures and is already projected.
        segments, restored, lengths, bases = [], [], [], []
        for _state, row, context, _anchor, _sampler in rows:
            prior = row.restored if commit else None
            if commit:
                row.restored = None
            held = 0 if prior is None else int(prior.shape[1])
            skip = max(0, held + int(context.shape[1]) - slots)
            if skip >= held:
                prior = None
                context = context[:, skip - held :]
            elif skip:
                prior = prior[:, skip:]
            segments.append(context)
            restored.append(prior)
            lengths.append(
                int(context.shape[1]) + (0 if prior is None else int(prior.shape[1]))
            )
            bases.append(row.fed + skip)
        # Metal projections need a nonempty shape even when the window keeps no context.
        width = max(1, max(int(seg.shape[1]) for seg in segments))
        padded = mx.concatenate(
            [
                (
                    seg
                    if seg.shape[1] == width
                    else mx.concatenate(
                        [
                            seg,
                            mx.zeros(
                                (1, width - seg.shape[1], seg.shape[2]), dtype=seg.dtype
                            ),
                        ],
                        axis=1,
                    )
                )
                for seg in segments
            ],
            axis=0,
        )
        h_ctx = variants.project_context(model, padded)
        if any(prior is not None for prior in restored):
            width = max(lengths)
            parts = []
            for index, (prior, seg) in enumerate(zip(restored, segments)):
                part = h_ctx[index : index + 1, : int(seg.shape[1])]
                if prior is not None:
                    part = mx.concatenate([prior.astype(part.dtype), part], axis=1)
                if int(part.shape[1]) < width:
                    pad = mx.zeros(
                        (1, width - int(part.shape[1]), part.shape[2]), dtype=part.dtype
                    )
                    part = mx.concatenate([part, pad], axis=1)
                parts.append(part)
            h_ctx = mx.concatenate(parts, axis=0)

        anchors = mx.concatenate([anchor for _, _, _, anchor, _ in rows]).astype(
            mx.int32
        )
        # DSpark's anchor row proposes the first token, so its block holds the
        # anchor plus block - 2 masks and keeps every row.
        markov = getattr(model, "markov_head", None)
        anchor_slot = markov is not None and bool(
            getattr(model.config, "sample_from_anchor", False)
        )
        if anchor_slot:
            block -= 1
        masks = mx.full(
            (batch, block - 1), int(model.config.mask_token_id), dtype=mx.int32
        )
        inputs = mx.concatenate([anchors[:, None], masks], axis=1)
        h = model._embed_input_tokens(inputs)
        if self.mask_embedding is not None:
            fill = self.mask_embedding.astype(h.dtype)
            h = mx.concatenate(
                [h[:, :1], mx.broadcast_to(fill, (batch, block - 1, h.shape[-1]))],
                axis=1,
            )

        cohort = self._assemble_cohort(uids, dtype)
        base_arr = mx.array(bases, dtype=mx.int32)
        if counts is None:
            totals = mx.array([b + n for b, n in zip(bases, lengths)], dtype=mx.int32)
        else:
            totals = base_arr + mx.concatenate(counts).astype(mx.int32)
        query_offsets = totals
        write_slots = [
            mx.array([(b + j) % capacity for j in range(n)], dtype=mx.int32)
            for b, n in zip(bases, lengths)
        ]
        # Fresh wrappers: an index assignment rebinds the wrapped array, and a
        # predraft must leave the cohort's rings untouched.
        pos = mx.stop_gradient(cohort.pos)
        for index, (b, n, slots_b) in enumerate(zip(bases, lengths, write_slots)):
            if not n:
                continue
            pos[index, slots_b] = mx.arange(b, b + n, dtype=mx.int32)
        # Attend the newest ``slots`` committed positions; block keys always.
        # Unwritten slots hold position -1 until the ring fills.
        ring_valid = (
            (pos >= 0) & (pos >= (totals - slots)[:, None]) & (pos < totals[:, None])
        )
        if self.causal:
            # Each block query sees the context and the block rows up to it.
            mask = mx.concatenate(
                [
                    mx.broadcast_to(ring_valid[:, None, :], (batch, block, capacity)),
                    mx.broadcast_to(
                        mx.tril(mx.ones((block, block), dtype=mx.bool_)),
                        (batch, block, block),
                    ),
                ],
                axis=2,
            )[:, None]
        else:
            mask = mx.concatenate(
                [ring_valid, mx.ones((batch, block), dtype=mx.bool_)], axis=1
            )[:, None, None, :]

        new_keys, new_values = [], []
        for layer_index, layer in enumerate(model.layers):
            attn = layer.self_attn
            # DFlash2 layers wrap attention and the MLP in dynamic convs;
            # DFlash v1 layers are plain pre-norm residual blocks.
            attention_conv = getattr(layer, "attention_conv", None)
            mlp_conv = getattr(layer, "mlp_conv", None)
            residual = h
            x = layer.input_layernorm(h)
            if attention_conv is not None:
                x, kernel = kernels.conv_prepare(attention_conv, x)

            queries, ctx_keys, ctx_values, prop_keys, prop_values = (
                variants.project_qkv(attn, x, h_ctx)
            )
            ctx_keys = attn.k_norm(
                ctx_keys.reshape(batch, width, attn.n_kv_heads, attn.head_dim)
            ).transpose(0, 2, 1, 3)
            ctx_values = ctx_values.reshape(
                batch, width, attn.n_kv_heads, attn.head_dim
            ).transpose(0, 2, 1, 3)
            if self.value_scale is not None:
                ctx_values = ctx_values * self.value_scale
            ctx_keys = model.rope(ctx_keys, offset=base_arr)

            ring_keys = mx.stop_gradient(cohort.keys[layer_index])
            ring_values = mx.stop_gradient(cohort.values[layer_index])
            for index, (n, slots_b) in enumerate(zip(lengths, write_slots)):
                if not n:
                    continue
                ring_keys[index, :, slots_b, :] = ctx_keys[index, :, :n, :]
                ring_values[index, :, slots_b, :] = ctx_values[index, :, :n, :]
            new_keys.append(ring_keys)
            new_values.append(ring_values)

            queries = attn.q_norm(
                queries.reshape(batch, block, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            prop_keys = attn.k_norm(
                prop_keys.reshape(batch, block, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            prop_values = prop_values.reshape(
                batch, block, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
            if self.value_scale is not None:
                prop_values = prop_values * self.value_scale
            queries = model.rope(queries, offset=query_offsets)
            prop_keys = model.rope(prop_keys, offset=query_offsets)
            keys = mx.concatenate([ring_keys, prop_keys], axis=2)
            values = mx.concatenate([ring_values, prop_values], axis=2)
            attended = mx.fast.scaled_dot_product_attention(
                queries,
                keys,
                values,
                scale=attn.scale,
                mask=mask,
                sinks=getattr(attn, "attention_sink_bias", None),
            )
            attended = attended.transpose(0, 2, 1, 3).reshape(batch, block, -1)
            attended = attn.o_proj(variants.gate_output(attn, x, attended))
            if attention_conv is not None:
                attended = kernels.conv_finish(attention_conv, attended, kernel)
            h = residual + attended

            residual = h
            x = layer.post_attention_layernorm(h)
            if mlp_conv is not None:
                x, kernel = kernels.conv_prepare(mlp_conv, x)
            x = variants.gated_mlp(layer.mlp, x)
            if mlp_conv is not None:
                x = kernels.conv_finish(mlp_conv, x, kernel)
            h = residual + x

        if commit:
            cohort.keys = new_keys
            cohort.values = new_values
            cohort.pos = pos
            for index, ((_state, row, *_), b, n) in enumerate(
                zip(rows, bases, lengths)
            ):
                row.fed = b + n
                self._remember(row, h_ctx[index : index + 1, :n])

        draft_hidden = model.norm(h) if anchor_slot else model.norm(h)[:, 1:]
        logits = model._logits(draft_hidden)
        samplers = [sampler for *_, sampler in rows]
        selector = getattr(model, "candidate_selector", None)
        if markov is not None:
            proposals = selection.select_markov(markov, logits, anchors, samplers)
        elif selector is not None and selection.fused_select_eligible(
            selector, draft_hidden
        ):
            proposals = selection.select_fused(
                selector, draft_hidden, logits, anchors, samplers
            )
        elif all(sampler is None for sampler in samplers):
            if selector is not None:
                tokens = selector.select(
                    draft_hidden, logits, anchors, selection.greedy_proposals
                )
            else:
                tokens = selection.greedy_proposals(logits)
            proposals = [(tokens[i : i + 1], []) for i in range(batch)]
        elif selector is not None:
            proposals = selection.select_sampled(
                selector, draft_hidden, logits, anchors, samplers
            )
        else:
            proposals = selection.sample_positions(logits, samplers)
        if not commit:
            self._predraft = _Predraft(
                uids=tuple(uids),
                keys=new_keys,
                values=new_values,
                pos=pos,
                bases=bases,
                unfed=[],
                proposals=proposals,
                contexts=[h_ctx[i : i + 1, :n] for i, n in enumerate(lengths)],
            )
        return proposals
