"""A pure, ``mx.compile``-able batched (B, 1) decode step for Qwen4-Exp.

The eager multi-row decode step of the flash pool spends ~36 ms per forward in
pure Python dispatch: 48 decoder layers, each re-traced from Python, with the
cache classes mutating *Python integer* bookkeeping between every layer and
every step.  ``mx.compile`` captures Python scalars as trace constants, so a
graph built at write column ``N`` would keep using ``N`` forever -- the same
reason MTPLX's ``laguna_compiled_step`` refuses to capture the stock caches.

This module removes the blocker the same way laguna did: every piece of
dynamic decode state becomes an explicit tensor leaf that goes in as an
argument and comes out as a result, and every Python-int read that changes
per step is replaced by an array computation.

State leaves (in layer order)
----------------------------
* linear (GDN) layer: ``conv`` [B, K-1, conv_dim] and recurrent ``state``
  [B, Hv, Dv, Dk] float32;
* the one PLE layer additionally carries its short-conv state and its int64
  token-history window (ArraysCache slots 2 and 3; slots 0/1 stay unused);
* full-attention (QSA) layer: pinned-capacity ``keys``/``values``
  [B, Hkv, cap, D] plus the indexer's raw ``index_keys`` [B, cap, Di] and
  ``index_positions`` [B, cap] int32.

Two shared scalars
------------------
* ``step`` -- an int32 [1] array counting decode steps since the snapshot.
  The absolute write column is ``write_col0 + step`` and each row's rope
  position is ``pos_base + step``; both stay graph inputs, never constants;
* per-row ``pos_base`` [B] int32 -- the position the row's next token would
  take, captured from the eager cache exactly the way the eager
  ``Qwen3_5LanguageModel.__call__`` derives it (cache offset + rope delta).

At B >= 2 every row writes the SAME KV column per step (the eager
``BatchKVCache`` pads rows on the left and appends at a shared ``_idx``), so
one shared scalar serves every layer -- the batched twin of laguna's
lockstep invariant.

Attention
---------
The eager dense path branches three ways at decode (ragged Metal kernel for
left-padded rows, exact-width masked sdpa for uniform rows, QSA sparse mask
merged into both).  The lane uses ONE path: ``mx.fast`` sdpa over the
pinned-capacity buffers with an in-graph bool mask that admits exactly the
columns the eager path admits -- per-row left padding, the causal frontier
``col <= write_col0 + step``, and the QSA indexer's selected blocks when it
is active.  Exact-width and ragged attention are numerically equivalent over
the same admitted set; only accumulation order differs.

Everything else -- hyper-connections (via ``_forward`` directly, so the
B=1-only pre-compiled inner graphs never nest inside this one), the Gated
Delta Net step with its fused Metal kernel, the PLE layer, the sparse MoE
and the final mixer -- runs the SHIPPED module code, so every installed
optimization keeps applying.  The GDN Metal kernel is compile-eligible: it
was probed to trace and replay bit-comparably under ``mx.compile`` on MLX
0.32.2.

Scope: T = 1, greedy-capable logits out, prefill stays eager.  The lane is
a prototype: it is not wired into the scheduler and changes no behavior
unless constructed explicitly.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import mlx.core as mx

from .qsa_fast import pool_completed_index_keys


class _VirtualSlotCache:
    """Duck-typed ``ArraysCache`` whose slots are the lane's leaf arrays."""

    left_padding = None
    lengths = None

    def __init__(self, size: int):
        self.cache = [None] * size

    def __getitem__(self, idx):
        return self.cache[idx]

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    @property
    def state(self):
        return self.cache


def _pad_to_cap(buf: mx.array, width: int, cap: int, axis: int) -> mx.array:
    index = [slice(None)] * buf.ndim
    index[axis] = slice(0, width)
    live = buf[tuple(index)]
    if width >= cap:
        index[axis] = slice(0, cap)
        return buf[tuple(index)]
    pad_shape = list(live.shape)
    pad_shape[axis] = cap - width
    pad = mx.zeros(tuple(pad_shape), dtype=live.dtype)
    return mx.concatenate([live, pad], axis=axis)


def snapshot_from_caches(
    lm: Any,
    caches: Sequence[Any],
    cap: int,
    pos_base: Optional[mx.array] = None,
) -> tuple[mx.array, int, list[mx.array]]:
    """Turn post-prefill eager serving caches into the lane's leaf state.

    ``caches`` must be the batched serving caches (``BatchQSAKVCache`` for
    full-attention layers, ``ArraysCache`` with left padding for linear
    ones) exactly as ``mlx_lm.generate._make_cache`` builds them through the
    model-owned ``to_batch`` conversion.
    """
    layers = lm.model.layers
    if len(caches) != len(layers):
        raise ValueError(f"expected {len(layers)} caches, got {len(caches)}")

    widths = set()
    pads = None  # per-row dead-prefix columns (joined rows), zeros otherwise
    for cache in caches:
        inner = getattr(cache, "kv_cache", None)
        if inner is None:
            continue
        left = getattr(inner, "left_padding", None)
        if isinstance(left, mx.array) and left.ndim > 0:
            pads = left.astype(mx.int32)
        widths.add(int(inner.keys.shape[2]))
    if len(widths) != 1:
        raise ValueError(f"KV caches are not width-aligned: {sorted(widths)}")
    alloc_width = widths.pop()

    write_col0 = None
    for cache in caches:
        inner = getattr(cache, "kv_cache", None)
        if inner is None:
            continue
        idx = getattr(inner, "_idx", None)
        if idx is not None:
            idx = int(idx)
            if write_col0 is not None and write_col0 != idx:
                raise ValueError("caches disagree on the write column")
            write_col0 = idx
    if write_col0 is None:
        write_col0 = alloc_width
    # The live prefix is _idx long; the allocation beyond it is a BatchKVCache
    # growth detail, not state.
    width = write_col0
    if width > cap:
        raise ValueError(f"cap {cap} is below the live KV width {width}")

    offsets = None
    for cache in caches:
        inner = getattr(cache, "kv_cache", None)
        if inner is None:
            continue
        offset = getattr(inner, "offset", None)
        if isinstance(offset, mx.array) and offset.ndim > 0:
            offsets = offset
            break
    if offsets is None:
        raise ValueError("no batched QSA cache carried per-row offsets")

    if pos_base is None:
        # Mirror Qwen3_5LanguageModel.__call__: text position = cache offset
        # (+ rope delta, which is zero for text-only rows).  The snapshot is
        # taken right after a forward, so the NEXT token sits at this offset.
        rope_deltas = getattr(lm, "_rope_deltas", None)
        base = mx.maximum(offsets, 0).astype(mx.int32)
        if isinstance(rope_deltas, mx.array) and rope_deltas.size == base.size:
            base = base + rope_deltas.reshape(base.shape).astype(mx.int32)
        pos_base = base

    leaves: list[mx.array] = []
    for layer, cache in zip(layers, caches):
        if layer.is_linear:
            if cache[0] is None or cache[1] is None:
                raise ValueError("GDN cache slots are empty; snapshot after prefill")
            leaves.append(cache[0])
            leaves.append(cache[1].astype(mx.float32))
            if "ple" in layer:
                if cache[3] is None:
                    raise ValueError("PLE history slot is empty")
                leaves.append(cache[2])
                leaves.append(cache[3])
        else:
            inner = cache.kv_cache
            if inner.keys is None or inner.values is None:
                raise ValueError("QSA cache is empty; snapshot after prefill")
            leaves.append(_pad_to_cap(inner.keys, width, cap, axis=2))
            leaves.append(_pad_to_cap(inner.values, width, cap, axis=2))
            index_width = 0 if cache.index_keys is None else int(cache.index_keys.shape[1])
            if cache.index_keys is None:
                batch = inner.keys.shape[0]
                di = layer.self_attn.indexer.head_dim
                idx = mx.zeros((batch, cap, di), dtype=inner.keys.dtype)
                pos = mx.zeros((batch, cap), dtype=mx.int32)
            else:
                idx = _pad_to_cap(cache.index_keys, index_width, cap, axis=1)
                pos = cache.index_position_ids
                if pos.ndim == 3:
                    pos = pos[0]
                pos = _pad_to_cap(pos, index_width, cap, axis=1).astype(mx.int32)
            leaves.append(idx)
            leaves.append(pos)
            # The pooled indexer bank, seeded once from the live prefix.  The
            # eager batch path re-pools every completed block on EVERY step;
            # the lane pools a block once, when it completes.
            indexer = layer.self_attn.indexer
            pooled = pool_completed_index_keys(
                idx,
                pos,
                compress_ratio=indexer.compress_ratio,
                index_key_norm=indexer.k_layernorm,
                apply_index_rope=indexer._apply_rope,
                stop_block=width // indexer.compress_ratio,
            )
            n_blocks = cap // indexer.compress_ratio
            if int(pooled.shape[1]) < n_blocks:
                pad_blocks = mx.zeros(
                    (pooled.shape[0], n_blocks - int(pooled.shape[1]), pooled.shape[-1]),
                    dtype=pooled.dtype,
                )
                pooled = mx.concatenate([pooled, pad_blocks], axis=1)
            leaves.append(pooled)

    if pads is None:
        batch = leaves[0].shape[0] if leaves else 1
        pads = mx.zeros((batch,), dtype=mx.int32)
    mx.eval(pos_base, pads, *leaves)
    return pos_base, pads, write_col0, leaves


def build_step(
    lm: Any,
    cap: int,
    write_col0: int,
    *,
    compiled: bool = True,
) -> Callable[..., tuple[mx.array, ...]]:
    """Build the pure decode step for a loaded ``LanguageModel``.

    The returned callable is::

        step(tokens, step_counter, pos_base, *leaves)
            -> (logits, step_counter_next, *leaves_next)

    ``pos_base`` is constant for a lane instance (captured, not updated);
    the step counter drives write columns and rope positions.
    """

    inner = lm.model
    layers = list(inner.layers)
    args = inner.args
    hc_count = args.hc_count
    mixer = inner.hyper_connection_mixer
    tied = bool(args.tie_word_embeddings)
    cols = mx.arange(cap, dtype=mx.int32)

    n_qsa_leaves = sum(5 for layer in layers if not layer.is_linear)
    n_gdn_leaves = sum(2 for layer in layers if layer.is_linear)
    n_ple_leaves = sum(2 for layer in layers if "ple" in layer)
    expected_leaves = n_qsa_leaves + n_gdn_leaves + n_ple_leaves

    def _qsa_decode(
        attn,
        x,
        tokens,
        positions,
        write_col,
        pads,
        k_leaf,
        v_leaf,
        idx_leaf,
        pos_leaf,
        pooled_leaf,
    ):
        """The dense QSA decode branch, compile-safe.

        Mirrors ``Qwen3_5Attention.__call__`` at T=1 plus the indexer mask
        tail of ``Qwen4ExpQSAIndexer.from_projected``, with the cache
        replaced by capacity-pinned leaves and every per-step scalar read
        replaced by an array computation.  The pooled indexer bank is
        maintained incrementally: at T=1 at most ONE block completes per
        step, and it is pooled once (dynamic take_along_axis gather) and
        written back under a completion guard, instead of re-pooling every
        completed block like the eager batch path does.
        """
        batch, length, _ = x.shape
        q_proj_output = attn.q_proj(x)
        keys = attn.k_proj(x)
        values = attn.v_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(batch, length, attn.num_attention_heads, -1),
            2,
            axis=-1,
        )
        gate = gate.reshape(batch, length, -1)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(
            keys.reshape(batch, length, attn.num_key_value_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(
            batch, length, attn.num_key_value_heads, -1
        ).transpose(0, 2, 1, 3)

        rotary_positions = mx.broadcast_to(positions, (3, batch, length))
        queries, keys = attn.rotary_emb.apply_rotary(
            queries, keys, rotary_positions, unsqueeze_dim=1
        )

        k_leaf = mx.slice_update(k_leaf, keys, write_col, axes=(2,))
        v_leaf = mx.slice_update(v_leaf, values, write_col, axes=(2,))

        indexer = attn.indexer
        projected = indexer.index_qk_proj(x).reshape(
            batch,
            length,
            indexer.n_heads + indexer.kv_heads,
            indexer.head_dim,
        )
        index_queries = indexer.q_layernorm(
            projected[:, :, : indexer.n_heads]
        ).transpose(0, 2, 1, 3)
        raw_index_keys = projected[:, :, indexer.n_heads :].squeeze(2)
        idx_leaf = mx.slice_update(idx_leaf, raw_index_keys, write_col, axes=(1,))
        pos_leaf = mx.slice_update(
            pos_leaf, positions.astype(pos_leaf.dtype), write_col, axes=(1,)
        )

        key_len = cap
        past_len = write_col  # scalar [1] array; buffer width minus one
        max_complete_blocks = key_len // indexer.compress_ratio
        ratio = indexer.compress_ratio

        index_queries = indexer._apply_rope(index_queries, positions)

        # --- incremental pooled-bank maintenance -------------------------
        # Blocks [0, write_col // ratio) are already pooled; a new block
        # completes exactly when (write_col + 1) % ratio == 0.  The gather
        # below always reads the would-be new block's rows (static width
        # ``ratio``, dynamic base) and the completion guard writes either
        # the fresh pooled value or the bank's current contents back.
        have = write_col // ratio  # [1] block index that may complete now
        want = (write_col + 1) // ratio
        completed = want > have
        start_col = have * ratio
        gather_cols = start_col + mx.arange(ratio, dtype=mx.int32)  # [ratio]
        gather_cols = mx.broadcast_to(
            gather_cols[None, :, None], (batch, ratio, 1)
        )
        rows = mx.take_along_axis(idx_leaf, gather_cols, axis=1)  # [B,ratio,Di]
        pooled_new = mx.mean(rows.astype(mx.float32), axis=1).astype(idx_leaf.dtype)
        pooled_new = indexer.k_layernorm(pooled_new[:, None])[:, 0]
        block_pos = mx.take_along_axis(
            pos_leaf, mx.broadcast_to(start_col, (batch, 1)), axis=1
        )  # [B, 1]
        pooled_new = indexer._apply_rope(
            pooled_new[:, None, None], block_pos
        )[:, 0, 0]
        have_idx = mx.broadcast_to(have[None, None], (batch, 1, 1))
        old_block = mx.take_along_axis(pooled_leaf, have_idx, axis=1)  # [B,1,Di]
        update = mx.where(
            mx.broadcast_to(completed[:, None], (batch, 1, 1)),
            pooled_new[:, None],
            old_block,
        )
        pooled_leaf = mx.slice_update(pooled_leaf, update, have, axes=(1,))

        pooled_keys = mx.expand_dims(pooled_leaf, axis=1)

        scores = index_queries.astype(mx.float32) @ pooled_keys.astype(
            mx.float32
        ).transpose(0, 1, 3, 2)
        scores = mx.sum(mx.maximum(scores, 0), axis=1)
        scores = scores / (indexer.head_dim**0.5)

        query_ends = past_len + mx.arange(length) + 1  # [length]
        complete_counts = query_ends // indexer.compress_ratio

        if max_complete_blocks > indexer.block_topk:
            # Sparse selection is structurally possible in this graph (the
            # cap holds more blocks than the budget).  The DYNAMIC gate --
            # whether this step is past the budget yet -- stays the array
            # ``use_sparse`` below, exactly like the eager tail.  When the
            # cap cannot ever hold a budget's worth of blocks the eager
            # indexer returns None and the layer attends densely; mirror
            # that by skipping the machinery entirely (argpartition with
            # kth=-topk would be invalid on fewer blocks than topk).
            valid_blocks = (
                mx.arange(max_complete_blocks)[None, None, :]
                < complete_counts[None, :, None]
            )
            scores = mx.where(valid_blocks, scores, -mx.inf)
            selected_blocks = mx.argpartition(
                scores, kth=-indexer.block_topk, axis=-1
            )[..., -indexer.block_topk :]

            block_hits = mx.put_along_axis(
                mx.zeros((batch, length, max_complete_blocks), dtype=mx.bool_),
                selected_blocks,
                mx.array(True),
                axis=-1,
            )
            selected_tokens = mx.repeat(
                block_hits, indexer.compress_ratio, axis=-1
            )
            complete_key_len = max_complete_blocks * indexer.compress_ratio
            if complete_key_len < key_len:
                # Block-selected columns stop at the last complete block; the
                # eager path pads the ragged remainder so the mask spans the
                # fetched width. Both widths are cap constants, so the pad is
                # compile-stable.
                selected_tokens = mx.concatenate(
                    [
                        selected_tokens,
                        mx.zeros(
                            (batch, length, key_len - complete_key_len),
                            dtype=mx.bool_,
                        ),
                    ],
                    axis=-1,
                )

            token_indices = cols
            tail_starts = complete_counts * indexer.compress_ratio
            tail = (
                token_indices[None, None, :] >= tail_starts[None, :, None]
            ) & (token_indices[None, None, :] < query_ends[None, :, None])
            causal = token_indices[None, None, :] < query_ends[None, :, None]
            use_sparse = complete_counts > indexer.block_topk
            selected_tokens = mx.where(
                use_sparse[None, :, None], selected_tokens | tail, causal
            )
        else:
            # Below the budget for the whole lane: dense causal decode, the
            # mask the eager indexer's None return produces.
            selected_tokens = cols[None, None, :] < query_ends[None, :, None]
        qsa_mask = selected_tokens[:, None]  # [B, 1, 1, cap]

        # Admission: the columns update_and_fetch would return for each row
        # -- [pad_row, write_col].  A joined row's dead prefix is zeros in
        # the shared buffer; eager serves mixed batches through the dense
        # per-row ragged path (the QSA mask only applies when every row is
        # unpadded), so an ANY-pad batch attends densely over live columns.
        live = (
            (cols[None, None, None, :] >= pads[:, None, None, None])
            & (cols[None, None, None, :] <= mx.reshape(write_col, (1, 1, 1)))
        )
        if max_complete_blocks > indexer.block_topk:
            any_pad = mx.max(pads) > 0
            mask = mx.where(any_pad, live, qsa_mask & live)
        else:
            mask = live
        bias = mx.where(mask, 0.0, -mx.inf).astype(k_leaf.dtype)

        output = mx.fast.scaled_dot_product_attention(
            queries, k_leaf, v_leaf, scale=attn.scale, mask=bias
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return (
            attn.o_proj(output * mx.sigmoid(gate)),
            (k_leaf, v_leaf, idx_leaf, pos_leaf, pooled_leaf),
        )

    def step(tokens, step_counter, pos_base, pads, *leaves):
        if len(leaves) != expected_leaves:
            raise ValueError(
                f"expected {expected_leaves} leaves, got {len(leaves)}"
            )

        write_col = mx.reshape(step_counter + write_col0, (1,))
        # pos_base is evolving state: each row's NEXT-token absolute position,
        # advanced by one per step.  A joined row enters with its own value
        # at join time -- a global column clock would miscount it by the
        # steps taken before it joined.
        positions = pos_base[:, None]  # [B, 1]

        # Layer-order virtual caches over the flat leaf sequence.
        it = iter(leaves)
        per_layer: list[tuple[Any, ...]] = []
        for layer in layers:
            if layer.is_linear:
                slots = _VirtualSlotCache(4 if "ple" in layer else 2)
                slots[0] = next(it)
                slots[1] = next(it)
                if "ple" in layer:
                    slots[2] = next(it)
                    slots[3] = next(it)
                per_layer.append((slots,))
            else:
                per_layer.append(
                    (next(it), next(it), next(it), next(it), next(it))
                )

        hidden = inner.embed_tokens(tokens)
        hidden = mx.tile(hidden, (1, 1, hc_count))

        updated: list[mx.array] = []
        for layer, state in zip(layers, per_layer):
            if "ple" in layer:
                (slots,) = state
                hidden = hidden + layer.ple(
                    hidden, tokens, slots, None, target_verify=False
                )
            mixed, hyper_input, injection_weights = layer.attn_hyper_connection._forward(
                hidden
            )
            if layer.is_linear:
                (slots,) = state
                branch = layer.linear_attn(
                    mixed, mask=None, cache=slots, gdn_sink=None, target_verify=False
                )
                updated.extend([slots[0], slots[1]])
                if "ple" in layer:
                    updated.extend([slots[2], slots[3]])
            else:
                k_leaf, v_leaf, idx_leaf, pos_leaf, pooled_leaf = state
                branch, (k_new, v_new, idx_new, pos_new, pooled_new) = (
                    _qsa_decode(
                        layer.self_attn,
                        mixed,
                        tokens,
                        positions,
                        write_col,
                        pads,
                        k_leaf,
                        v_leaf,
                        idx_leaf,
                        pos_leaf,
                        pooled_leaf,
                    )
                )
                updated.extend([k_new, v_new, idx_new, pos_new, pooled_new])
            injection = branch[..., None, :] * injection_weights[..., None]
            hidden = hyper_input + injection.reshape(*hyper_input.shape)

            mixed, hyper_input, injection_weights = (
                layer.mlp_hyper_connection._forward(hidden)
            )
            branch = layer.mlp(mixed, target_verify=False)
            injection = branch[..., None, :] * injection_weights[..., None]
            hidden = hyper_input + injection.reshape(*hyper_input.shape)

        out = mixer._forward(hidden)
        if tied:
            logits = inner.embed_tokens.as_linear(out)
        else:
            logits = lm.lm_head(out)
        return (logits[:, 0, :], step_counter + 1, pos_base + 1, *updated)

    return mx.compile(step) if compiled else step


class Qwen4ExpCompiledDecodeLane:
    """Drives the compiled step against leaf state captured from eager caches.

    Usage: prefill eagerly with the serving caches, then::

        lane = Qwen4ExpCompiledDecodeLane(lm, caches, cap)
        lane.seed()
        for _ in range(n):
            logits = lane.advance()
    """

    def __init__(self, lm, caches, cap: int, *, compiled: bool = True, pos_base=None):
        self.lm = lm
        # Align the cap to the indexer compress ratio so the QSA mask never
        # needs the ragged tail pad, and to a coarse column boundary so the
        # number of distinct graph shapes stays bounded in serving.
        ratio = 1
        for layer in lm.model.layers:
            if not layer.is_linear:
                ratio = layer.self_attn.indexer.compress_ratio
                break
        grain = ratio * 64
        self.cap = ((int(cap) + grain - 1) // grain) * grain
        self.compiled = bool(compiled)
        # The serving rank of QSA index positions (3 for MRoPE models);
        # the lane works on the text plane and the flush must restore it.
        self.pos_ndim = 2
        for cache in caches:
            positions = getattr(cache, "index_position_ids", None)
            if positions is not None:
                self.pos_ndim = int(positions.ndim)
                break
        self.pos_base, self.pads, self.write_col0, self.leaves = (
            snapshot_from_caches(lm, caches, self.cap, pos_base=pos_base)
        )
        self.step_fn = build_step(
            lm, self.cap, self.write_col0, compiled=self.compiled
        )
        self.tokens: Optional[mx.array] = None
        self.step_counter = mx.array(0, dtype=mx.int32).reshape(1)
        self._position = 0

    @property
    def width(self) -> int:
        """The next write column: every row's next token lands here."""
        return self.write_col0 + int(self.step_counter.item())

    def remaining_steps(self) -> int:
        return self.cap - self.write_col0 - self._position

    def seed(self, tokens: mx.array) -> None:
        self.tokens = mx.array(tokens, dtype=mx.uint32).reshape(-1, 1)
        mx.eval(self.tokens)

    def filter_rows(self, indices) -> None:
        """Keep only the given rows, mirroring eager ``cache.filter``.

        A pure batch-axis gather over every leaf plus the per-row scalars;
        the shared column clock (write_col0, step counter) is untouched.
        The next step runs under a new batch shape and compiles a new graph
        (~1s, cached per shape for the process lifetime).
        """
        idx = mx.array(list(indices), dtype=mx.int32)
        self.pos_base = self.pos_base[idx]
        self.pads = self.pads[idx]
        self.leaves = [leaf[idx] for leaf in self.leaves]
        if self.tokens is not None:
            self.tokens = self.tokens[idx]
        mx.eval(self.pos_base, self.pads, *self.leaves)

    def extend_rows(self, row_caches, lm=None) -> None:
        """Join prefilled singleton rows at the batch's write column.

        ``row_caches`` is one serving-style cache list PER ROW (the caches
        its own prefill produced, converted with ``to_batch([0])``).  The
        row's live prefix is left-padded with zeros to the batch's current
        column so its next token writes where every other row's does --
        the leaf-side equivalent of ``BatchQSAKVCache.extend`` /
        ``ArraysCache.extend``.  Pads become nonzero and the QSA mask
        falls back to dense-live admission, exactly like the eager batch
        path for mixed-width batches.
        """
        lm = lm or self.lm
        layers = lm.model.layers
        width = self.width
        new_pos_base = []
        new_pads = []
        new_leaves: list[list[mx.array]] = [[] for _ in row_caches]
        for row_index, caches in enumerate(row_caches):
            # QSA geometry comes from the first full-attention cache
            for layer, cache in zip(layers, caches):
                if layer.is_linear:
                    if cache[0] is None or cache[1] is None:
                        raise ValueError("GDN slots empty; prefill the row first")
                    new_leaves[row_index].append(cache[0])
                    new_leaves[row_index].append(cache[1].astype(mx.float32))
                    if "ple" in layer:
                        if cache[3] is None:
                            raise ValueError("PLE history empty; prefill first")
                        new_leaves[row_index].append(cache[2])
                        new_leaves[row_index].append(cache[3])
                else:
                    # Accept a to_batch([0])-converted cache or a raw
                    # singleton QSAKVCache alike.
                    inner = getattr(cache, "kv_cache", cache)
                    row_len = int(inner.offset.reshape(-1)[0].item()) if isinstance(
                        inner.offset, mx.array
                    ) else int(inner.offset)
                    pad = width - row_len
                    if pad < 0:
                        raise ValueError(
                            f"joining row is longer ({row_len}) than the batch "
                            f"position ({width}); snapshot a wider batch instead"
                        )
                    # left-pad the live prefix to the batch column
                    def _lp(buf, axis):
                        shape = list(buf.shape)
                        shape[axis] = pad
                        z = mx.zeros(tuple(shape), dtype=buf.dtype)
                        return mx.concatenate([z, buf], axis=axis)

                    keys = inner.keys[..., : width - pad, :]
                    values = inner.values[..., : width - pad, :]
                    k_new = _lp(mx.array(keys), axis=2)
                    k_new = _pad_to_cap(k_new, width, self.cap, axis=2)
                    v_new = _lp(mx.array(values), axis=2)
                    v_new = _pad_to_cap(v_new, width, self.cap, axis=2)
                    if cache.index_keys is None:
                        idx = mx.zeros(
                            (1, self.cap, layer.self_attn.indexer.head_dim),
                            dtype=k_new.dtype,
                        )
                        pos = mx.zeros((1, self.cap), dtype=mx.int32)
                    else:
                        idx = _lp(mx.array(cache.index_keys), axis=1)
                        idx = _pad_to_cap(idx, width, self.cap, axis=1)
                        pos_arr = cache.index_position_ids
                        if pos_arr.ndim == 3:
                            pos_arr = pos_arr[0]
                        pos = _lp(pos_arr.astype(mx.int32), axis=1)
                        pos = _pad_to_cap(pos, width, self.cap, axis=1)
                    indexer = layer.self_attn.indexer
                    pooled = pool_completed_index_keys(
                        idx,
                        pos,
                        compress_ratio=indexer.compress_ratio,
                        index_key_norm=indexer.k_layernorm,
                        apply_index_rope=indexer._apply_rope,
                        stop_block=width // indexer.compress_ratio,
                    )
                    n_blocks = self.cap // indexer.compress_ratio
                    if int(pooled.shape[1]) < n_blocks:
                        pad_b = mx.zeros(
                            (1, n_blocks - int(pooled.shape[1]), pooled.shape[-1]),
                            dtype=pooled.dtype,
                        )
                        pooled = mx.concatenate([pooled, pad_b], axis=1)
                    new_leaves[row_index].extend([k_new, v_new, idx, pos, pooled])
                    new_pos_base.append(row_len)
                    new_pads.append(pad)

        # interleave the new rows' leaves into the lane's flat leaf order
        merged: list[mx.array] = []
        it_new = [iter(row) for row in new_leaves]
        for leaf in self.leaves:
            stacked = [leaf] + [next(it) for it in it_new]
            merged.append(mx.concatenate(stacked, axis=0))
        self.leaves = merged
        self.pos_base = mx.concatenate(
            [self.pos_base, mx.array(new_pos_base, dtype=mx.int32)]
        )
        self.pads = mx.concatenate(
            [self.pads, mx.array(new_pads, dtype=mx.int32)]
        )
        # The caller supplies every row's token on the next advance(); a
        # stale self.tokens of the old width is simply dropped.
        self.tokens = None
        mx.eval(self.pos_base, self.pads, *self.leaves)

    def advance(self, tokens: Optional[mx.array] = None) -> mx.array:
        """Run one step; returns [B, V] logits (unevaluated).

        ``tokens`` replaces the lane's input token before stepping -- the
        driver samples the previous step's logits and feeds the winners back.
        """
        if tokens is not None:
            self.tokens = mx.array(tokens, dtype=mx.uint32).reshape(-1, 1)
            mx.eval(self.tokens)
        if self.tokens is None:
            raise ValueError("lane has no seed token; call seed() first")
        if self.remaining_steps() <= 0:
            raise ValueError(
                f"leaves are full at cap {self.cap}; re-snapshot with a larger cap"
            )
        logits, next_counter, next_pos_base, *updated = self.step_fn(
            self.tokens, self.step_counter, self.pos_base, self.pads, *self.leaves
        )
        self.step_counter = next_counter
        self.pos_base = next_pos_base
        self.leaves = updated
        self._position += 1
        return logits

# ---------------------------------------------------------------------------
# serving shim: route GenerationBatch decode steps through the lane
# ---------------------------------------------------------------------------
_LANE_MIN_BATCH = 2
_LANE_CAP_MARGIN = 1024


def _live_width(caches) -> Optional[int]:
    """The eager caches' write column, or None if no QSA cache is present."""
    for cache in caches:
        inner = getattr(cache, "kv_cache", None)
        if inner is None:
            continue
        return int(inner._idx)
    return None


def flush_to_caches(lm, lane: "Qwen4ExpCompiledDecodeLane", caches) -> None:
    """Write the lane's state back into the eager serving caches.

    The inverse of ``snapshot_from_caches``: after k lane steps the eager
    caches sit at the snapshot width while the lane holds k more tokens.
    Every reshape (filter/extend) and every eager forward (prefill overlap,
    MTP singleton activation) needs the eager caches current, so the shim
    flushes before any of those run.  Buffers are only grown, never
    shrunk, to keep BatchKVCache's own allocation discipline.
    """
    layers = lm.model.layers
    if len(caches) != len(layers):
        raise ValueError(f"expected {len(layers)} caches, got {len(caches)}")
    width = lane.width
    li = 0
    for layer, cache in zip(layers, caches):
        if layer.is_linear:
            slots = lane.leaves[li : li + (4 if "ple" in layer else 2)]
            cache.cache = list(slots)
            li += 4 if "ple" in layer else 2
        else:
            k_leaf, v_leaf, idx_leaf, pos_leaf, _pooled = lane.leaves[li : li + 5]
            li += 5
            inner = cache.kv_cache
            batch, heads = k_leaf.shape[0], k_leaf.shape[1]
            alloc = max(256, ((width + 255) // 256) * 256)
            if (
                inner.keys is None
                or inner.keys.shape[0] != batch
                or inner.keys.shape[2] < width
            ):
                inner.keys = mx.zeros(
                    (batch, heads, alloc, k_leaf.shape[-1]), dtype=k_leaf.dtype
                )
                inner.values = mx.zeros_like(inner.keys)
            inner.keys[..., :width, :] = k_leaf[:, :, :width, :]
            inner.values[..., :width, :] = v_leaf[:, :, :width, :]
            inner._idx = width
            inner.offset = lane.pos_base.astype(mx.int32)
            inner.left_padding = lane.pads
            cache.index_keys = idx_leaf[:, :width, :]
            pos_back = pos_leaf[:, :width]
            if getattr(lane, "pos_ndim", 2) == 3:
                # Serving keeps MRoPE-shaped [3, B, W] positions; the lane
                # works on the text plane.  Restore the rank so a later
                # extend concatenates like-with-like (_pad_index promotes
                # singletons the same way).
                pos_back = mx.broadcast_to(
                    pos_back[None], (3, *pos_back.shape)
                )
            cache.index_position_ids = pos_back
            cache.index_offset = width
    mx.eval(
        [c for cache in caches for c in (
            cache.cache
            if hasattr(cache, "cache")
            else [cache.kv_cache.keys, cache.kv_cache.values]
        )]
    )


class Qwen4ExpLaneShim:
    """Model-call shim for ``GenerationBatch`` decode steps.

    Wrap the language model handed to ``GenerationBatch``: single-token
    multi-row forwards (the continuous-batching decode step) run through
    the compiled lane; everything else -- prefill chunks, MTP singleton
    rounds, verification shapes -- is flushed to the eager caches and run
    eagerly.  Reshape events (filter/extend on the batch) must call
    :meth:`before_reshape` first and :meth:`after_reshape` afterwards; the
    omlx batch_generator patch wires those.

    The eager caches are scratch: while the lane runs they sit at the
    snapshot width and only become current through a flush.
    """

    def __init__(self, lm, *, enabled: bool = True):
        self.lm = lm
        # The wrapper may be the serving adapter; the lane machinery needs
        # the LanguageModel underneath it.
        self._language = getattr(lm, "_language_model", None) or lm
        self.enabled = bool(enabled)
        self.lane: Optional[Qwen4ExpCompiledDecodeLane] = None
        self._cache_ids: Optional[tuple] = None
        self.stats = {"lane_steps": 0, "eager_steps": 0, "flushes": 0, "resnapshots": 0}

    def __getattr__(self, name):
        # Transparent proxy: the batch machinery (MTP probes, prompt-priming
        # drop_ctx, stats walkers) reads attributes off ``self.model`` -- now
        # this shim -- and must see the wrapped model's surface unchanged.
        # Only reached for attributes not found normally.
        return getattr(self.__dict__["lm"], name)

    def _owns(self, caches) -> bool:
        """Whether these are the caches the lane shadows.

        Every prefill completion constructs a transient donor
        GenerationBatch sharing this shim; its singleton caches are NOT the
        lane's, and flushing onto them would corrupt both sides.
        """
        return self._cache_ids is not None and caches is not None and (
            tuple(id(c) for c in caches) == self._cache_ids
        )

    # -- lifecycle hooks for reshape events ----------------------------
    def before_reshape(self, caches) -> None:
        if self.lane is not None and self._owns(caches):
            self.stats["flushes"] += 1
            flush_to_caches(self._language, self.lane, caches)

    def after_reshape(self, caches) -> None:
        if self._owns(caches):
            self.lane = None
            self._cache_ids = None

    # -- model-call interception ---------------------------------------
    def _lane_for(self, caches) -> Optional[Qwen4ExpCompiledDecodeLane]:
        if not self.enabled or caches is None or len(caches) < _LANE_MIN_BATCH:
            return None
        lm = self._language
        if getattr(lm, "model_type", "") != "qwen4_exp_text":
            return None
        ids = tuple(id(c) for c in caches)
        if self.lane is None or ids != self._cache_ids:
            width = _live_width(caches) or 0
            self.lane = Qwen4ExpCompiledDecodeLane(
                lm, caches, cap=width + _LANE_CAP_MARGIN
            )
            self._cache_ids = ids
            self.stats["resnapshots"] += 1
        return self.lane

    def __call__(self, inputs, cache=None, **kwargs):
        shape = getattr(inputs, "shape", None)
        if (
            cache is not None
            and shape is not None
            and len(shape) == 2
            and shape[1] == 1
            and shape[0] >= _LANE_MIN_BATCH
            and not kwargs
        ):
            lane = self._lane_for(cache)
            if lane is not None:
                if lane.remaining_steps() <= 8:
                    # Grow the cap: flush, re-snapshot with a wider margin.
                    self.stats["flushes"] += 1
                    flush_to_caches(self._language, lane, cache)
                    self.lane = None
                    self._cache_ids = None
                    lane = self._lane_for(cache)
                if not getattr(self, "_lane_stepped_once", False):
                    self._lane_stepped_once = True
                    import logging

                    logging.getLogger(__name__).info(
                        "Qwen4Exp compiled decode lane engaged at B=%d",
                        shape[0],
                    )
                self.stats["lane_steps"] += 1
                if self.stats["lane_steps"] % 500 == 0:
                    import logging

                    logging.getLogger(__name__).info(
                        "Qwen4Exp lane stats: %s", self.stats
                    )
                try:
                    return lane.advance(inputs[:, 0])[:, None, :]
                except Exception:
                    # Fail closed to the eager path; the flush below makes
                    # the eager caches current either way.
                    self.stats["flushes"] += 1
                    flush_to_caches(self._language, lane, cache)
                    self.lane = None
                    self._cache_ids = None
        if cache is not None and self.lane is not None and self._owns(cache):
            self.stats["flushes"] += 1
            flush_to_caches(self._language, self.lane, cache)
            self.lane = None
            self._cache_ids = None
        if cache is not None:
            self.stats["eager_steps"] += 1
        return self.lm(inputs, cache=cache, **kwargs)

_LANE_PREWARM_ENV = "OMLX_QWEN4_LANE_PREWARM"
# Conservative default set: the common agent shapes. Transient leaf memory
# scales with B * cap (the 16384 bucket is the largest at ~1.6 GB).
_LANE_PREWARM_DEFAULT = "4096:2,4096:3,4096:4,8192:2,8192:3,8192:4,16384:2"


def _prewarm_shapes(spec: str):
    shapes = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        cap_s, _, b_s = item.partition(":")
        try:
            shapes.append((int(cap_s), int(b_s or 2)))
        except ValueError:
            continue
    return shapes


def prewarm_lane(lm, shapes=None) -> int:
    """Trace the compiled graphs for common (cap, B) shapes ahead of traffic.

    The graph depends only on shapes, never values, so one step over
    zero-filled leaves forces mx.compile's shape-keyed trace and keeps it
    for the process lifetime -- the first live request at that shape then
    skips the ~1s compile.  Shapes run sequentially with a cache clear in
    between; transient leaf memory is released after each.

    Returns the number of shapes traced.
    """
    import logging

    if shapes is None:
        import os

        spec = os.environ.get(_LANE_PREWARM_ENV, "")
        if not spec:
            return 0
        shapes = _prewarm_shapes(spec or _LANE_PREWARM_DEFAULT)

    language = getattr(lm, "_language_model", None) or lm
    inner = language.model
    layers = list(inner.layers)
    traced = 0
    for cap, batch in shapes:
        ratio = 1
        for layer in layers:
            if not layer.is_linear:
                ratio = layer.self_attn.indexer.compress_ratio
                break
        grain = ratio * 64
        cap = ((cap + grain - 1) // grain) * grain
        leaves = []
        for layer in layers:
            if layer.is_linear:
                gdn = layer.linear_attn
                conv = mx.zeros(
                    (batch, gdn.conv_kernel_size - 1, gdn.conv_dim),
                    dtype=mx.bfloat16,
                )
                state = mx.zeros(
                    (
                        batch,
                        gdn.num_v_heads,
                        gdn.head_v_dim,
                        gdn.head_k_dim,
                    ),
                    dtype=mx.float32,
                )
                leaves.extend([conv, state])
                if "ple" in layer:
                    ple = layer.ple
                    state_len = getattr(ple, "short_conv_state_len", 2)
                    # The PLE short conv runs on the hyper-connection
                    # expanded plane (hc_count * hidden), matching the
                    # decoder layer's hidden_states.
                    ple_dim = inner.args.hc_count * inner.args.hidden_size
                    leaves.append(
                        mx.zeros(
                            (batch, state_len, ple_dim), dtype=mx.bfloat16
                        )
                    )
                    ctx = getattr(
                        ple.ple_embedding, "context_len", 2
                    )
                    leaves.append(
                        mx.zeros((batch, ctx), dtype=mx.int64)
                    )
            else:
                attn = layer.self_attn
                kv = mx.zeros(
                    (batch, attn.num_key_value_heads, cap, attn.head_dim),
                    dtype=mx.bfloat16,
                )
                leaves.extend([kv, mx.zeros_like(kv)])
                di = attn.indexer.head_dim
                leaves.append(
                    mx.zeros((batch, cap, di), dtype=mx.bfloat16)
                )
                leaves.append(mx.zeros((batch, cap), dtype=mx.int32))
                leaves.append(
                    mx.zeros(
                        (batch, cap // attn.indexer.compress_ratio, di),
                        dtype=mx.bfloat16,
                    )
                )
        step = build_step(language, cap, 0, compiled=True)
        out = None
        try:
            out = step(
                mx.zeros((batch, 1), dtype=mx.uint32),
                mx.zeros((1,), dtype=mx.int32),
                mx.zeros((batch,), dtype=mx.int32),
                mx.zeros((batch,), dtype=mx.int32),
                *leaves,
            )
            mx.eval(out[0])
            traced += 1
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning(
                "lane prewarm shape (cap=%d, B=%d) failed: %s", cap, batch, exc
            )
        del leaves, out
        mx.clear_cache()
    if traced:
        logging.getLogger(__name__).info(
            "Qwen4Exp lane prewarmed %d shapes", traced
        )
    return traced
