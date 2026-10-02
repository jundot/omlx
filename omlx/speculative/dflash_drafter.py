# SPDX-License-Identifier: Apache-2.0
"""DFlash block drafter for the batched Lightning MTP verify path.

The drafter replaces the embedded MTP head as the draft source inside
``omlx.patches.mlx_lm_mtp``: the shared verify forward, acceptance, cache
rollback and emission stay the same, and this module only turns captured
target hidden states into the next draft block for every row.

The draft model itself is mlx-vlm's ``DFlash2DraftModel`` (or the DFlash v1
``DFlashDraftModel``). Its ``draft_block`` takes the target hidden states of
the tokens committed since the previous call and appends them to a per-row
sliding-window context cache, so each row keeps its own cache here and the
verify path feeds it the accepted positions after every cycle.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.speculative.drafters import load_drafter

from ..model_settings import MAX_LIGHTNING_MTP_DRAFT_TOKENS
from ..patches import qwen35_packed_linear, qwen35_verify_qmm
from ..patches.mlx_lm_mtp import batch_generator as bg
from ..patches.qwen35_verify_qmm import set_verify_qmm_armed
from ..utils.sampling import top_k_indices

logger = logging.getLogger(__name__)


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
    fed: int = 0
    keys: list[mx.array] | None = None
    values: list[mx.array] | None = None
    pos: mx.array | None = None


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


# ``_grouped_dynamic_convolve`` for one of the two kernel sets in
# ``dynamic`` (B, L, 2, KS, G), one thread per output element, with the bf16
# rounding of each op in the composed version.
_CONV_SOURCE = """
    uint e = thread_position_in_grid.x;
    uint c = e % H;
    uint t = (e / H) % L;
    uint bl = e / H;
    uint g = c / GS;
    T acc = T(0);
    for (int o = 0; o < KS; ++o) {
        T xv = int(t) >= o ? x[e - uint(o) * H] : T(0);
        T k = static_cast<T>(
            float(base[(SET * KS + o) * H + c]) + float(dyn[((bl * 2 + SET) * KS + o) * G + g]));
        T prod = static_cast<T>(float(k) * float(xv));
        acc = static_cast<T>(float(acc) + float(prod));
    }
    out[e] = acc;
"""
_CONV_KERNEL = None


def _convolve(conv, hidden: mx.array, dynamic: mx.array, which: int) -> mx.array:
    global _CONV_KERNEL
    if _CONV_KERNEL is None:
        _CONV_KERNEL = mx.fast.metal_kernel(
            name="omlx_dflash2_conv",
            input_names=["x", "dyn", "base"],
            output_names=["out"],
            source=_CONV_SOURCE,
        )
    batch, length, width = hidden.shape
    (out,) = _CONV_KERNEL(
        inputs=[hidden, dynamic, conv.base_kernel],
        template=[
            ("T", hidden.dtype),
            ("H", width),
            ("L", length),
            ("GS", conv.group_size),
            ("G", width // conv.group_size),
            ("KS", conv.kernel_size),
            ("SET", which),
        ],
        grid=(hidden.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[hidden.shape],
        output_dtypes=[hidden.dtype],
    )
    return out


def _conv_prepare(conv, hidden: mx.array) -> tuple[mx.array, mx.array]:
    """``GroupedDynamicCausalConv.prepare``; returns both kernel sets."""
    groups = hidden.shape[-1] // conv.group_size
    dynamic = conv.kernel_projection(hidden).reshape(
        *hidden.shape[:-1], 2, conv.kernel_size, groups
    )
    return _convolve(conv, hidden, dynamic, 0), dynamic


def _conv_finish(conv, hidden: mx.array, dynamic: mx.array) -> mx.array:
    return _convolve(conv, hidden, dynamic, 1)


def _mlp(mlp, x: mx.array) -> mx.array:
    """Gated MLP; eight-row verify shapes fuse gate, up and swiglu."""
    gate = getattr(mlp, "gate_proj", None)
    up = getattr(mlp, "up_proj", None)
    if gate is not None and qwen35_verify_qmm.sg8_swiglu_eligible(gate, up, x):
        batch, length, width = x.shape
        hidden = qwen35_verify_qmm.vk_swiglu_sg8(
            x.reshape(batch * length, width), gate, up
        )
        return mlp.down_proj(hidden.reshape(batch, length, -1))
    return mlp(x)


def _greedy_proposals(logits: mx.array) -> mx.array:
    # DFlash v1 samples proposals from this callable; the DFlash2 selector
    # only looks for ``sample_proposal`` on it and otherwise takes argmax.
    return mx.argmax(logits, axis=-1)


def _concat_captured(captured: Sequence[mx.array]) -> mx.array:
    """Join per-layer captures ``[(1, n, H), ...]`` into ``(1, n, K*H)``."""
    if len(captured) == 1:
        return captured[0]
    return mx.concatenate(list(captured), axis=-1)


class DFlashDrafter:
    """Row-wise DFlash drafting against per-request context caches."""

    def __init__(self, model: nn.Module, *, block_size: int, source_path: str):
        self.model = model
        self.source_path = source_path
        self.target_layer_ids: list[int] = [
            int(i) for i in model.config.target_layer_ids
        ]
        self.block_size = int(block_size)
        # Lightning MTP reads this as the fixed draft depth.
        self.depth = self.block_size - 1
        self._rows: dict[Any, _RowContext] = {}
        self._cohort: _Cohort | None = None
        # Prefill captures wait under the request id until the scheduler
        # learns the row uid at insert time.
        self._request_seeds: dict[str, list[mx.array]] = {}
        # Row order of the generation batch whose ordinary decode step is
        # running, so a plain forward can be attributed to rows.
        self.scope_uids: tuple | None = None
        self._predraft: _Predraft | None = None

    @property
    def window(self) -> int:
        """Context tokens the drafter attends to; older captures are dropped."""
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
            row.pending.append(_concat_captured(captured))
            self._bound_pending(row)

    def _bound_pending(self, row: _RowContext) -> None:
        # Only the newest ring_slots rows are ever attended, so rows queued
        # while drafting is off are dropped here; fed still counts them.
        drop = sum(int(p.shape[1]) for p in row.pending) - self.ring_slots
        if drop <= 0:
            return
        row.fed += drop
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

    def seed_request(self, request_id: str, captured: Sequence[mx.array]) -> None:
        """Queue prefill captures for a request that has no row uid yet."""
        if captured:
            self._request_seeds.setdefault(request_id, []).append(
                _concat_captured(captured)
            )

    def bind_uid(self, request_id: str, uid: Any) -> None:
        seeds = self._request_seeds.pop(request_id, None)
        if seeds:
            row = self._row(uid)
            row.pending.extend(seeds)
            self._bound_pending(row)

    def release_request(self, request_id: str) -> None:
        self._request_seeds.pop(request_id, None)

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
            if self._rows.pop(uid, None) is not None:
                self._detach_cohort()

    def clear(self) -> None:
        self._predraft = None
        self._rows.clear()
        self._cohort = None
        self._request_seeds.clear()

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
                row.pending.append(_concat_captured(captured))
            if not row.pending:
                raise RuntimeError(
                    f"DFlash drafter has no context for uid={state.uid!r}"
                )
            context = (
                row.pending[0]
                if len(row.pending) == 1
                else mx.concatenate(row.pending, axis=1)
            )
            row.pending = []
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
        unfed = sum(int(p.shape[1]) for p in row.pending)
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
        mx.async_eval(tokens, *_q_arrays(accept_lps))
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
        row.pending = []
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
        dtype = model.fc.weight.dtype if hasattr(model.fc, "weight") else mx.bfloat16
        if hasattr(model.fc, "scales"):
            dtype = model.fc.scales.dtype

        # Context segments: keep at most ``slots`` newest tokens per row and
        # count the rest as skipped so absolute positions stay aligned.
        segments, lengths, bases = [], [], []
        for _state, row, context, _anchor, _sampler in rows:
            skip = max(0, int(context.shape[1]) - slots)
            if skip:
                context = context[:, skip:]
            segments.append(context)
            lengths.append(int(context.shape[1]))
            bases.append(row.fed + skip)
        width = max(lengths)
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
        h_ctx = model.hidden_norm(model.fc(padded))

        anchors = mx.concatenate([anchor for _, _, _, anchor, _ in rows]).astype(
            mx.int32
        )
        masks = mx.full(
            (batch, block - 1), int(model.config.mask_token_id), dtype=mx.int32
        )
        inputs = mx.concatenate([anchors[:, None], masks], axis=1)
        h = model._embed_input_tokens(inputs)

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
            pos[index, slots_b] = mx.arange(b, b + n, dtype=mx.int32)
        # Attend the newest ``slots`` committed positions; block keys always.
        # Unwritten slots hold position -1 until the ring fills.
        ring_valid = (
            (pos >= 0) & (pos >= (totals - slots)[:, None]) & (pos < totals[:, None])
        )
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
                x, kernel = _conv_prepare(attention_conv, x)

            ctx_keys, ctx_values = attn._project_kv(h_ctx)
            ctx_keys = attn.k_norm(
                ctx_keys.reshape(batch, width, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            ctx_values = ctx_values.reshape(
                batch, width, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
            ctx_keys = model.rope(ctx_keys, offset=base_arr)

            ring_keys = mx.stop_gradient(cohort.keys[layer_index])
            ring_values = mx.stop_gradient(cohort.values[layer_index])
            for index, (n, slots_b) in enumerate(zip(lengths, write_slots)):
                ring_keys[index, :, slots_b, :] = ctx_keys[index, :, :n, :]
                ring_values[index, :, slots_b, :] = ctx_values[index, :, :n, :]
            new_keys.append(ring_keys)
            new_values.append(ring_values)

            queries = attn.q_proj(x)
            prop_keys, prop_values = attn._project_kv(x)
            queries = attn.q_norm(
                queries.reshape(batch, block, attn.n_heads, -1)
            ).transpose(0, 2, 1, 3)
            prop_keys = attn.k_norm(
                prop_keys.reshape(batch, block, attn.n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            prop_values = prop_values.reshape(
                batch, block, attn.n_kv_heads, -1
            ).transpose(0, 2, 1, 3)
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
                sinks=attn.attention_sink_bias,
            )
            attended = attn.o_proj(
                attended.transpose(0, 2, 1, 3).reshape(batch, block, -1)
            )
            if attention_conv is not None:
                attended = _conv_finish(attention_conv, attended, kernel)
            h = residual + attended

            residual = h
            x = layer.post_attention_layernorm(h)
            if mlp_conv is not None:
                x, kernel = _conv_prepare(mlp_conv, x)
            x = _mlp(layer.mlp, x)
            if mlp_conv is not None:
                x = _conv_finish(mlp_conv, x, kernel)
            h = residual + x

        if commit:
            cohort.keys = new_keys
            cohort.values = new_values
            cohort.pos = pos
            for (_state, row, _context, _anchor, _sampler), b, n in zip(
                rows, bases, lengths
            ):
                row.fed = b + n

        draft_hidden = model.norm(h)[:, 1:]
        logits = model._logits(draft_hidden)
        samplers = [sampler for *_, sampler in rows]
        selector = getattr(model, "candidate_selector", None)
        if selector is not None and _fused_select_eligible(selector, draft_hidden):
            proposals = _select_fused(selector, draft_hidden, logits, anchors, samplers)
        elif all(sampler is None for sampler in samplers):
            if selector is not None:
                tokens = selector.select(
                    draft_hidden, logits, anchors, _greedy_proposals
                )
            else:
                tokens = _greedy_proposals(logits)
            proposals = [(tokens[i : i + 1], []) for i in range(batch)]
        elif selector is not None:
            proposals = _select_sampled(
                selector, draft_hidden, logits, anchors, samplers
            )
        else:
            proposals = _sample_positions(logits, samplers)
        if not commit:
            self._predraft = _Predraft(
                uids=tuple(uids),
                keys=new_keys,
                values=new_values,
                pos=pos,
                bases=bases,
                unfed=[],
                proposals=proposals,
            )
        return proposals


def _sample_positions(logits: mx.array, samplers: list) -> list[tuple]:
    """Independent per-position proposals (DFlash v1) with their q rows."""
    out = []
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    for index, sampler in enumerate(samplers):
        if sampler is None:
            out.append((mx.argmax(logits[index : index + 1], axis=-1), []))
            continue
        tokens, accept = [], []
        for position in range(lp.shape[1]):
            token, accept_lp = bg._sample_draft_with_logprobs(
                sampler, lp[index, position : position + 1]
            )
            tokens.append(token.reshape(1))
            accept.append(accept_lp.reshape(-1))
        out.append((mx.concatenate(tokens)[None], accept))
    return out


def _sample_candidates(scores: mx.array, sampler) -> tuple[mx.array, mx.array]:
    """Draw from ``sampler``'s filters applied to candidate scores ``(1, C)``.

    Returns the chosen column and the log density over the candidates. The
    density is the proposal q that acceptance reads, so the filters only have
    to match the request's in spirit, not bit for bit.
    """
    lp = scores.astype(mx.float32)
    lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
    count = lp.shape[-1]
    top_k = int(getattr(sampler, "top_k", 0) or 0)
    top_p = float(getattr(sampler, "top_p", 0.0) or 0.0)
    min_p = float(getattr(sampler, "min_p", 0.0) or 0.0)
    temp = float(getattr(sampler, "temp", 1.0) or 1.0)
    if 0 < top_k < count:
        kth = mx.sort(lp, axis=-1)[..., count - top_k : count - top_k + 1]
        lp = mx.where(lp >= kth, lp, -float("inf"))
    if 0.0 < top_p < 1.0:
        order = mx.argsort(-lp, axis=-1)
        ranked = mx.take_along_axis(lp, order, axis=-1)
        probs = mx.exp(ranked)
        keep = (mx.cumsum(probs, axis=-1) - probs) < top_p
        keep = mx.put_along_axis(
            mx.zeros(keep.shape, dtype=mx.bool_), order, keep, axis=-1
        )
        lp = mx.where(keep, lp, -float("inf"))
    if min_p > 0.0:
        floor = lp.max(axis=-1, keepdims=True) + math.log(min_p)
        lp = mx.where(lp >= floor, lp, -float("inf"))
    scaled = lp * (1.0 / temp)
    pick = mx.random.categorical(scaled)
    return pick, scaled - mx.logsumexp(scaled, axis=-1, keepdims=True)


# One threadgroup per row walks the positions: simdgroup c scores candidate
# c against the previous pick, then simdgroup 0 applies the row's filters
# (as ``_sample_candidates``) and draws by inverse CDF. Temperature <= 0
# takes the first argmax.
_SELECT_SOURCE = """
    constexpr int C = 16;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint b = threadgroup_position_in_grid.x;
    threadgroup float scores[C];
    threadgroup int pred_tg;
    if (sg == 0 && lane == 0)
        pred_tg = anchors[b];
    float temp = params[b * 4 + 0];
    float top_p = params[b * 4 + 1];
    int top_k = int(params[b * 4 + 2]);
    float min_p = params[b * 4 + 3];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int p = 0; p < P; ++p) {
        int row = int(b) * P + p;
        int pred = pred_tg;
        int cid = cand[row * C + int(sg)];
        auto pc = pcb + long(pred) * R;
        auto sc = scb + long(cid) * R;
        auto pj = proj + long(row) * R;
        float acc = 0.0f;
        for (int i = 0; i < R / 32; ++i) {
            int d = int(lane) + i * 32;
            acc += float(pc[d]) * float(pj[d]) * float(sc[d]);
        }
        acc = simd_sum(acc);
        if (lane == 0)
            scores[sg] = float(unary[row * C + int(sg)]) + acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            bool live = int(lane) < C;
            float s = live ? scores[lane] : -INFINITY;
            float m = simd_max(s);
            int pick;
            float q = -INFINITY;
            if (temp <= 0.0f) {
                pick = simd_min((live && s == m) ? int(lane) : C);
            } else {
                float lp = s - m - log(simd_sum(live ? exp(s - m) : 0.0f));
                if (!live)
                    lp = -INFINITY;
                if (top_k > 0 && top_k < C) {
                    int greater = 0;
                    for (int j = 0; j < C; ++j)
                        greater += simd_shuffle(lp, ushort(j)) > lp ? 1 : 0;
                    if (greater >= top_k)
                        lp = -INFINITY;
                }
                if (top_p > 0.0f && top_p < 1.0f) {
                    float prefix = 0.0f;
                    for (int j = 0; j < C; ++j) {
                        float o = simd_shuffle(lp, ushort(j));
                        if (o > lp || (o == lp && j < int(lane)))
                            prefix += exp(o);
                    }
                    if (!(prefix < top_p))
                        lp = -INFINITY;
                }
                if (min_p > 0.0f) {
                    float floor_ = simd_max(lp) + log(min_p);
                    if (lp < floor_)
                        lp = -INFINITY;
                }
                float scaled = lp / temp;
                float sm = simd_max(scaled);
                float e = scaled == -INFINITY ? 0.0f : exp(scaled - sm);
                float total = simd_sum(e);
                q = scaled == -INFINITY ? -INFINITY : scaled - sm - log(total);
                float cdf = simd_prefix_inclusive_sum(e) / total;
                float u = uniforms[row];
                int first = simd_min((e > 0.0f && cdf > u) ? int(lane) : C);
                int last = simd_max(e > 0.0f ? int(lane) : -1);
                pick = first < C ? first : last;
            }
            if (live)
                logq[row * C + int(lane)] = q;
            if (lane == 0) {
                int token = cand[row * C + pick];
                tokens[row] = token;
                pred_tg = token;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"""
_SELECT_KERNEL = None


def _select_kernel():
    global _SELECT_KERNEL
    if _SELECT_KERNEL is None:
        _SELECT_KERNEL = mx.fast.metal_kernel(
            name="omlx_dflash2_select",
            input_names=[
                "cand",
                "unary",
                "proj",
                "anchors",
                "pcb",
                "scb",
                "params",
                "uniforms",
            ],
            output_names=["tokens", "logq"],
            source=_SELECT_SOURCE,
        )
    return _SELECT_KERNEL


def _fused_select_eligible(selector, hidden) -> bool:
    pcb = getattr(selector.predecessor_codebook, "weight", None)
    scb = getattr(selector.successor_codebook, "weight", None)
    return (
        selector.top_k == 16
        and isinstance(pcb, mx.array)
        and isinstance(scb, mx.array)
        and not hasattr(selector.predecessor_codebook, "scales")
        and not hasattr(selector.successor_codebook, "scales")
        and pcb.shape[-1] % 32 == 0
        and hidden.shape[1] <= 32
    )


def _select_fused(selector, hidden, logits, anchors, samplers) -> list[tuple]:
    """``_select_sampled`` in one launch with sparse q over the candidates."""
    batch, length, vocab = logits.shape
    candidates = top_k_indices(logits, selector.top_k).astype(mx.int32)
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden)
    params = mx.array(
        [
            (
                [0.0, 0.0, 0.0, 0.0]
                if sampler is None
                else [
                    float(getattr(sampler, "temp", 1.0) or 1.0),
                    float(getattr(sampler, "top_p", 0.0) or 0.0),
                    float(int(getattr(sampler, "top_k", 0) or 0)),
                    float(getattr(sampler, "min_p", 0.0) or 0.0),
                ]
            )
            for sampler in samplers
        ],
        dtype=mx.float32,
    )
    pcb = selector.predecessor_codebook.weight
    tokens, logq = _select_kernel()(
        inputs=[
            candidates,
            unary,
            projected,
            anchors.reshape(-1).astype(mx.int32),
            pcb,
            selector.successor_codebook.weight,
            params,
            mx.random.uniform(shape=(batch, length)),
        ],
        template=[("P", length), ("R", pcb.shape[-1])],
        grid=(32 * selector.top_k * batch, 1, 1),
        threadgroup=(32 * selector.top_k, 1, 1),
        output_shapes=[(batch, length), (batch, length, selector.top_k)],
        output_dtypes=[mx.int32, mx.float32],
    )
    return [
        (
            tokens[index : index + 1],
            (
                []
                if sampler is None
                else bg.SparseDraftQ(candidates[index], logq[index], vocab)
            ),
        )
        for index, sampler in enumerate(samplers)
    ]


def _q_arrays(accept_lps) -> list:
    if isinstance(accept_lps, bg.SparseDraftQ):
        return accept_lps.arrays()
    return list(accept_lps)


def _select_sampled(selector, hidden, logits, anchors, samplers) -> list[tuple]:
    """DFlash2 selector path with per-row sampling over the candidate set.

    Mirrors ``CandidateSelector.select``: top-k candidates per position, a
    predecessor/successor edge score, then the next predecessor is the
    chosen token. Sampled rows draw from their sampler's filters over the
    candidate scores and expose that sparse distribution as q.
    """
    batch, length, vocab = logits.shape
    candidates = top_k_indices(logits, selector.top_k)
    unary = mx.take_along_axis(logits, candidates, axis=-1)
    projected = selector.hidden_projection(hidden)
    predecessor = anchors.reshape(-1)
    tokens = [[] for _ in range(batch)]
    densities = [[] for _ in range(batch)]
    for position in range(length):
        edges = mx.sum(
            selector.predecessor_codebook(predecessor)[:, None]
            * projected[:, position, None]
            * selector.successor_codebook(candidates[:, position]),
            axis=-1,
        )
        scores = unary[:, position] + edges  # (batch, top_k)
        chosen = []
        for index, sampler in enumerate(samplers):
            if sampler is None:
                pick = mx.argmax(scores[index], keepdims=True)
            else:
                pick, density = _sample_candidates(scores[index : index + 1], sampler)
                densities[index].append(density)
            token = candidates[index, position][pick]
            chosen.append(token.reshape(1))
            tokens[index].append(token.reshape(1))
        predecessor = mx.concatenate(chosen).astype(mx.int32)
    accept = [[] for _ in range(batch)]
    for index in range(batch):
        if not densities[index]:
            continue
        rows = mx.concatenate(densities[index])  # (length, top_k)
        q = mx.full((length, vocab), -float("inf"), dtype=mx.float32)
        q = mx.put_along_axis(q, candidates[index], rows, axis=-1)
        accept[index] = [q[position] for position in range(length)]
    return [
        (mx.concatenate(tokens[index])[None].astype(mx.int32), accept[index])
        for index in range(batch)
    ]


def resolve_block_size(model: nn.Module, requested: int | None) -> int:
    """Clamp the block (anchor + proposals) to what Lightning MTP verifies."""
    trained = int(getattr(model.config, "block_size", 0) or 0)
    block = int(requested or trained or 0)
    if block <= 1:
        raise ValueError("DFlash block size must cover the anchor and one proposal")
    limit = MAX_LIGHTNING_MTP_DRAFT_TOKENS + 1
    if trained and block > trained:
        block = trained
    return min(block, limit)


def quantize_drafter(model: nn.Module, *, bits: int, group_size: int) -> int:
    """Affine-quantize the drafter's linear layers; returns the layer count."""
    quantized = 0

    def predicate(path: str, module: nn.Module) -> bool:
        nonlocal quantized
        if not isinstance(module, nn.Linear):
            return False
        if module.weight.shape[-1] % group_size:
            return False
        quantized += 1
        return True

    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=predicate)
    return quantized


def load_dflash_drafter(
    path: str,
    target_model: Any,
    *,
    block_size: int | None = None,
    quant_enabled: bool = False,
    quant_bits: int = 4,
    quant_group_size: int = 64,
) -> DFlashDrafter:
    """Load a DFlash checkpoint through mlx-vlm and bind it to ``target_model``.

    ``target_model`` is the mlx-vlm model whose ``language_model`` owns the
    embeddings and lm_head the drafter borrows.
    """
    model, kind = load_drafter(path, kind="dflash")
    if kind != "dflash":
        raise ValueError(f"{path} resolved to drafter kind {kind!r}, expected 'dflash'")
    if quant_enabled and not getattr(model.config, "quantization", None):
        count = quantize_drafter(model, bits=quant_bits, group_size=quant_group_size)
        logger.info(
            "DFlash drafter quantized: %d linear layers at %d bits (group %d)",
            count,
            quant_bits,
            quant_group_size,
        )
    model.bind(target_model)
    block = resolve_block_size(model, block_size)
    mx.eval(model.parameters())
    if qwen35_packed_linear.enabled(target_model):
        packed = qwen35_packed_linear.pack_drafter(model)
        logger.info("DFlash drafter packed 4-bit projections: %d layers", packed)
    drafter = DFlashDrafter(model, block_size=block, source_path=path)
    logger.info(
        "DFlash drafter loaded: path=%s kind=%s block=%d target_layers=%s",
        path,
        drafter.kind,
        block,
        drafter.target_layer_ids,
    )
    return drafter


def attach_drafter(language_model: Any, drafter: DFlashDrafter) -> None:
    """Expose the drafter to Lightning MTP through the language model markers."""
    language_model._omlx_drafter = drafter
    language_model._omlx_mtp_decode_enabled = True
    language_model._omlx_mtp_multi_request = True
    language_model._omlx_mtp_batch_rollback = True
    language_model._omlx_mtp_chain = True
    language_model._omlx_mtp_depth = drafter.depth
    language_model._omlx_mtp_head_clone = False


def detach_drafter(language_model: Any) -> DFlashDrafter | None:
    drafter = getattr(language_model, "_omlx_drafter", None)
    if drafter is None:
        return None
    drafter.clear()
    language_model._omlx_drafter = None
    language_model._omlx_mtp_decode_enabled = False
    return drafter
