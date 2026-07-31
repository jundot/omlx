"""Family-aware target wrapper: KV cache + a hidden-state tap at given layers.

Adapted in part from ARahim3/mlx-dspark (MIT); see THIRD_PARTY_NOTICES.md.

- gemma4 (mlx-vlm): uses the built-in ``capture_layer_ids`` / ``hidden_sink`` hook.
- qwen3  (mlx-lm):  no hook exists, so we replicate the model's forward loop and
  capture the residual stream after the tapped layers.
- qwen3_5 (mlx-lm, **hybrid** linear+full attention):
  same replicated-loop tap with per-layer fa/ssm masks, plus exact spec rollback
  (see :meth:`verify`) because the linear-attention layers carry recurrent state
  that cannot be trimmed like a KV cache.

All expose: make_cache(), run(ids, cache, tap)->(logits, fused_hidden),
plain(ids, cache)->logits (no capture), and the spec-round pair verify()/rollback()
(identical to run()+trim for dense targets).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any

import mlx.core as mx


class Target:
    def __init__(
        self, model, tokenizer, *, kv_bits: int | None = None, kv_group_size: int = 64
    ):
        self.model = model
        self.tokenizer = tokenizer
        # mlx-vlm targets (gemma4) vs mlx-lm; mlx-lm also ships text-only modules for some
        # multimodal checkpoints (qwen3_5), which expose .language_model too — so route by
        # the class's package, not by attribute shape.
        self.is_vlm = type(model).__module__.startswith("mlx_vlm")
        # mlx-lm text stack: model.model for plain families, model.language_model.model for
        # wrapped multimodal-text families (qwen3_5).
        self._inner = (
            model.language_model
            if (not self.is_vlm and hasattr(model, "language_model"))
            else model
        )
        self.family = (
            "gemma4"
            if self.is_vlm
            else getattr(getattr(self._inner, "args", None), "model_type", "qwen3")
        )
        inner_layers = getattr(getattr(self._inner, "model", None), "layers", [])
        # hybrid = some layers hold recurrent (linear-attention) state instead of KV.
        self.is_hybrid = not self.is_vlm and any(
            getattr(layer, "is_linear", False) for layer in inner_layers
        )
        if kv_bits and self.is_vlm:
            raise ValueError(
                "--kv-bits is supported for mlx-lm text targets only "
                "(the mlx-vlm/gemma-4 cache layout is managed by mlx-vlm)"
            )
        if kv_bits and self.is_hybrid:
            raise ValueError(
                "--kv-bits is unsupported for hybrid linear-attention targets "
                "(their recurrent-state caches are not KV caches; only 16 of "
                "64 layers even hold KV). Run without --kv-bits."
            )
        self.kv_bits = int(kv_bits) if kv_bits else None
        self.kv_group_size = int(kv_group_size)
        if not self.is_vlm:
            # mlx-lm convention: tied models simply don't define lm_head (more reliable
            # than args.tie_word_embeddings, which some families omit)
            self._tied = not hasattr(self._inner, "lm_head")
        # hybrid targets: the linear-attention modules, in layer order. verify() hooks
        # their recurrence + conv calls to capture per-round inputs so rollback() can
        # rebuild the state at the accept point exactly (see the section comment below).
        if self.is_hybrid:
            self._gdn_modules = [
                layer.linear_attn
                for layer in inner_layers
                if getattr(layer, "is_linear", False)
            ]
            for m in self._gdn_modules:
                m.conv1d._mlx_dspark_capture = True
        # hybrid spec-verify state (see verify()/rollback()); per-generation, reset by
        # reset_spec() at loop start so a served Target never leaks state across requests.
        self.reset_spec()

    # -- cache --
    def make_cache(self):
        if self.is_vlm:
            return self.model.language_model.make_cache()
        if self.kv_bits:
            # Quantized KV from token 0: trimmable (spec rollback + prefix reuse work
            # unchanged), halves-or-quarters the KV bandwidth bill on long contexts.
            # Output is the greedy decoding of the KV-quantized target (a quality knob of
            # the same class as target quantization, not a spec-decoding approximation).
            from mlx_lm.models.cache import QuantizedKVCache

            layers = getattr(getattr(self._inner, "model", None), "layers", [])
            return [QuantizedKVCache(self.kv_group_size, self.kv_bits) for _ in layers]
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    # -- forward with hidden-state tap --
    def run(self, ids: mx.array, cache, tap: list[int]):
        """ids [1,L] -> (logits [1,L,V], fused_hidden [1,L,n_tap*H])."""
        if self.is_vlm:
            out = self.model.language_model(
                inputs=ids, cache=cache, capture_layer_ids=tap
            )
            return out.logits, mx.concatenate(out.hidden_states, axis=-1)
        if self.is_hybrid:
            return self._run_hybrid(ids, cache, tap)
        return self._run_mlxlm(ids, cache, tap)

    # The mlx-lm paths are split body/head so a prefill chunk can skip the vocab
    # projection it would throw away (see :meth:`prefill`); run() is body+full head,
    # byte-identical to the single-function version it replaced.
    def _body_mlxlm(self, ids, cache, tap):
        """Replicated mlx-lm forward through the final norm -> (hidden, fused|None)."""
        from mlx_lm.models.base import create_attention_mask

        mm = self._inner.model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        mask = create_attention_mask(h, cache[0])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask, c)
            if i in tapset:
                captured.append(h)
        return mm.norm(h), (
            mx.concatenate(captured, axis=-1) if tap is not None else None
        )

    def _body_hybrid(self, ids, cache, tap):
        """Replicates mlx-lm's hybrid text forward (qwen3_5-style: per-layer fa/ssm mask
        selection) with the residual-stream capture. Faithfulness is proven by
        :meth:`verify_tap` at load, exactly like the dense path."""
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask

        mm = self._inner.model
        tapset = set(tap) if tap is not None else ()
        h = mm.embed_tokens(ids)
        fa_mask = create_attention_mask(h, cache[mm.fa_idx])
        ssm_mask = create_ssm_mask(h, cache[mm.ssm_idx])
        captured = []
        for i, (layer, c) in enumerate(zip(mm.layers, cache)):
            h = layer(h, mask=(ssm_mask if layer.is_linear else fa_mask), cache=c)
            if i in tapset:
                captured.append(h)
        return mm.norm(h), (
            mx.concatenate(captured, axis=-1) if tap is not None else None
        )

    def _head_mlxlm(self, hn):
        # mlx-lm convention: tied models have no lm_head, they project through the embedding
        return (
            self._inner.model.embed_tokens.as_linear(hn)
            if self._tied
            else self._inner.lm_head(hn)
        )

    def _run_mlxlm(self, ids, cache, tap):
        hn, fused = self._body_mlxlm(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    def _run_hybrid(self, ids, cache, tap):
        hn, fused = self._body_hybrid(ids, cache, tap)
        return self._head_mlxlm(hn), fused

    # -- prefill forward (skips the vocab projection it would discard) ------------------
    #
    # A prefill chunk's logits are read at exactly ONE row — the last one of the last
    # chunk, which becomes the first sampled token. Projecting the other rows to vocab is
    # pure waste: measured ~7-8% of prefill FLOPs on an 8-bit Qwen3-8B, plus a
    # [chunk, vocab] transient (622 MB at chunk 2048) that is the reason PREFILL_CHUNK
    # has to be small at all. See NOTES "Prefill pass".

    def prefill(
        self,
        ids: mx.array,
        cache,
        tap: list[int] | None = None,
        *,
        want_logits: bool = False,
        head_last_row: bool = True,
    ):
        """One prefill chunk -> ``(logits | None, fused | None)``.

        ``want_logits=False`` (any non-final chunk) skips the vocab projection entirely
        and is **bit-identical** to :meth:`run` in everything it keeps — the KV cache and
        the tapped hidden states are untouched. ``head_last_row`` additionally projects
        only the final row, which moves that row from the quantized matmul path to the
        matvec path (the one every decode step already takes): the usual fp-tie class,
        measured max|Δlogit| = 0.06 on Qwen3-8B-8bit = 1-2 bf16 ulps. Set
        ``generate.PREFILL_LAST_ROW_HEAD = False`` for the bit-identical-but-slower form.
        ``tap=None`` skips the hidden-state capture."""
        if self.is_vlm:
            lm = self.model.language_model
            sink = [] if tap is not None else None
            hn = lm.model(ids, cache=cache, capture_layer_ids=tap, hidden_sink=sink)
            fused = mx.concatenate(sink, axis=-1) if tap is not None else None
            head = lm.logits_from_hidden
        else:
            hn, fused = (self._body_hybrid if self.is_hybrid else self._body_mlxlm)(
                ids, cache, tap
            )
            head = self._head_mlxlm
        if not want_logits:
            return None, fused
        return head(hn[:, -1:, :] if head_last_row else hn), fused

    # -- plain forward (no capture) for the greedy baseline --
    def plain(self, ids: mx.array, cache):
        if self.is_vlm:
            return self.model.language_model(inputs=ids, cache=cache).logits
        return self.model(ids, cache=cache)

    # -- speculative verify + rollback -------------------------------------------------
    #
    # Dense targets: verify == run, rollback == KV trim of the rejected tail (the prior
    # inline behavior, byte-identical).
    #
    # Hybrid targets: linear-attention state advances through *every* token a forward
    # touches and has no trim, so a partially-rejected verify forward would poison it.
    # verify() therefore records, per linear layer, references to the round's recurrence
    # inputs (the exact ``gated_delta_update`` args, post-conv) and the conv input — free:
    # they are intermediates of the round's graph, held one round. On a partial accept,
    # rollback() rebuilds each linear cache AT the accept point:
    #
    #   delta state  → re-run the recurrence over the accepted prefix from the pre-round
    #                  state. The kernel consumes tokens strictly sequentially, so the
    #                  state after ``keep`` tokens is independent of the rejected tail —
    #                  the re-run reproduces the original round's intermediate state
    #                  bit-for-bit (same inputs, same op order).
    #   conv window  → a slice of the recorded conv input (rows [keep, keep+k-1)) — the
    #                  exact values a committed forward of the prefix would have kept.
    #   KV layers    → trim only the rejected tail (their rows for accepted tokens are
    #                  already correct).
    #
    # So rejects cost ~48 tiny recurrence kernels (lazy, folded into the next round's
    # graph) instead of re-forwarding the accepted tokens through the whole model — the
    # earlier "replay backlog" design paid one full-model row per accepted token on every
    # partial accept, which made past-optimum caps and low-acceptance content decay much
    # faster than dense targets. (A two-phase commit+spec variant measured 0.66× baseline
    # — two full weight reads per round; still don't resurrect it.)
    #
    # The capture is two scoped hooks active only while the forward's graph is built:
    # the defining module's ``gated_delta_update`` and the conv class ``__call__`` (only
    # modules flagged ``_mlx_dspark_capture`` record). Both call straight through — no
    # numeric change — and they sit on mlx-lm's own modules, so the capture works for
    # the tap path (_run_hybrid) and the model's native forward (lookup mode) alike.

    def reset_spec(self) -> None:
        """Clear per-generation spec state (hybrid capture). Loops call this at start so
        a long-lived served Target never carries state across generations."""
        self._stash = None  # ([per-layer gated_delta args], [per-layer conv input])
        self._spec_width = 0

    @contextmanager
    def _capture_linear(self):
        """Scoped hooks recording each linear layer's recurrence args + conv input for
        the forward built inside the context. Pure pass-through (records references,
        calls the originals) — verify_tap()'s bit-exactness probe runs with the hooks
        active, so a semantic drift in this plumbing fails loudly at load."""
        gdn = self._gdn_modules[0]
        mod = sys.modules[type(gdn).__module__]
        conv_cls = type(gdn.conv1d)
        orig_gdu = mod.gated_delta_update
        orig_conv = conv_cls.__call__
        rec_delta, rec_conv = [], []

        def rec_gdu(
            q, k, v, a, b, a_log, dt_bias, state=None, mask=None, use_kernel=True
        ):
            # use_kernel is recorded so the rollback re-run takes the SAME
            # implementation path the forward took (bit-exact prefix re-run)
            rec_delta.append((q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel))
            return orig_gdu(
                q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel=use_kernel
            )

        def rec_conv_call(conv_self, x, *args, **kwargs):
            if getattr(conv_self, "_mlx_dspark_capture", False):
                rec_conv.append(x)
            return orig_conv(conv_self, x, *args, **kwargs)

        from .target_adapters import target_capture_lock

        with target_capture_lock():
            mod.gated_delta_update = rec_gdu
            conv_cls.__call__ = rec_conv_call
            try:
                yield rec_delta, rec_conv
            finally:
                mod.gated_delta_update = orig_gdu
                conv_cls.__call__ = orig_conv

    def verify(self, ids: mx.array, cache, tap: list[int] | None):
        """Spec-round verify forward: ``ids`` [1, 1+m] = [anchor] + m draft tokens.
        Returns (logits [1, 1+m, V], fused [1, 1+m, n_tap*H]) at those rows.
        ``tap=None`` skips the hidden-state capture (drafter-free lookup mode) and
        returns ``fused=None``."""

        def fwd(x):
            if tap is None:
                return self.plain(x, cache), None
            return self.run(x, cache, tap)

        if not self.is_hybrid:
            return fwd(ids)
        self._stash = None
        want = int(ids.shape[1])
        if want == 1:
            # no drafts (lookup miss / plain step): fully committed, nothing to capture
            return fwd(ids)
        with self._capture_linear() as (rec_delta, rec_conv):
            logits, fused = fwd(ids)
        self._stash = (rec_delta, rec_conv)
        self._spec_width = want
        return logits, fused

    @staticmethod
    def _merge_cache_rows(row_caches: list[list[Any]]) -> list[Any]:
        """Merge independent request caches into the model's native batch cache.

        oMLX installs merge/extract support for its normal KV, TurboQuant and
        recurrent cache families.  dSpark deliberately uses that contract
        instead of maintaining a second cache representation.
        """
        if not row_caches:
            return []
        layer_count = len(row_caches[0])
        if any(len(cache) != layer_count for cache in row_caches):
            raise ValueError("cannot batch target caches with different layer counts")
        merged = []
        for layer_index in range(layer_count):
            rows = [cache[layer_index] for cache in row_caches]
            merge = getattr(rows[0], "merge", None)
            if not callable(merge):
                raise TypeError(
                    f"target cache {type(rows[0]).__name__} does not support merge()"
                )
            merged.append(merge(rows))
        return merged

    @staticmethod
    def _extract_cache_rows(merged: list[Any], batch_size: int) -> list[list[Any]]:
        rows: list[list[Any]] = [[] for _ in range(batch_size)]
        for layer in merged:
            extract = getattr(layer, "extract", None)
            if not callable(extract):
                raise TypeError(
                    f"batched target cache {type(layer).__name__} does not support extract()"
                )
            for row_index in range(batch_size):
                rows[row_index].append(extract(row_index))
        return rows

    @staticmethod
    def _slice_hybrid_capture(capture: tuple[Any, Any], row_index: int):
        rec_delta, rec_conv = capture
        row_delta = []
        for q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel in rec_delta:
            row_delta.append(
                (
                    q[row_index : row_index + 1],
                    k[row_index : row_index + 1],
                    v[row_index : row_index + 1],
                    a[row_index : row_index + 1],
                    b[row_index : row_index + 1],
                    a_log,
                    dt_bias,
                    state[row_index : row_index + 1] if state is not None else None,
                    mask[row_index : row_index + 1] if mask is not None else None,
                    use_kernel,
                )
            )
        row_conv = [value[row_index : row_index + 1] for value in rec_conv]
        return row_delta, row_conv

    def verify_batch(
        self,
        ids: mx.array,
        caches: list[list[Any]],
        tap: list[int] | None,
    ) -> tuple[Any, Any, list[list[Any]], list[Any]]:
        """Verify equal-width proposals with one native target forward.

        The returned caches are extracted back into independent request rows.
        Hybrid recurrent captures are also split per row so a rejection only
        rewinds that request.
        """
        batch_size = int(ids.shape[0])
        if batch_size != len(caches):
            raise ValueError("verify batch size does not match target cache rows")
        merged = self._merge_cache_rows(caches)

        def fwd():
            if tap is None:
                return self.plain(ids, merged), None
            return self.run(ids, merged, tap)

        captures: list[Any] = [None] * batch_size
        if self.is_hybrid and int(ids.shape[1]) > 1:
            with self._capture_linear() as capture:
                logits, fused = fwd()
            captures = [
                self._slice_hybrid_capture(capture, row_index)
                for row_index in range(batch_size)
            ]
        else:
            logits, fused = fwd()
        row_caches = self._extract_cache_rows(merged, batch_size)
        return logits, fused, row_caches, captures

    def rollback(
        self,
        cache,
        n_rejected: int,
        accepted: list[int],
        *,
        capture: Any = None,
        spec_width: int | None = None,
    ) -> None:
        """Undo the rejected tail of the last verify. ``accepted`` = the accepted draft
        tokens (unused — kept for call-site symmetry; the hybrid rebuild slices the
        recorded arrays lazily, so the on-GPU greedy path never adds a device sync)."""
        if not self.is_hybrid:
            if n_rejected > 0:
                for c in cache:
                    if c is not None and hasattr(c, "trim"):
                        c.trim(n_rejected)
            return
        if n_rejected > 0:
            stash = self._stash if capture is None else capture
            width = self._spec_width if spec_width is None else spec_width
            if stash is None:
                raise RuntimeError(
                    "rollback() with rejected tokens but no preceding verify() capture "
                    "on a hybrid target"
                )
            from mlx_lm.models.gated_delta import gated_delta_update

            rec_delta, rec_conv = stash
            keep = width - n_rejected  # anchor + accepted drafts, >= 1
            li = 0
            for c in cache:
                if self._is_trimmable(c):
                    c.trim(n_rejected)
                    continue
                q, k, v, a, b, a_log, dt_bias, state, mask, use_kernel = rec_delta[li]
                conv_input = rec_conv[li]
                m = mask[:, :keep] if mask is not None else None
                _, new_state = gated_delta_update(
                    q[:, :keep],
                    k[:, :keep],
                    v[:, :keep],
                    a[:, :keep],
                    b[:, :keep],
                    a_log,
                    dt_bias,
                    state,
                    m,
                    use_kernel=use_kernel,
                )
                n_keep = conv_input.shape[1] - width  # conv kernel size - 1
                c[0] = mx.contiguous(conv_input[:, keep : keep + n_keep, :])
                c[1] = new_state
                li += 1
        if capture is None:
            self._stash = None

    @staticmethod
    def _is_trimmable(c) -> bool:
        fn = getattr(c, "is_trimmable", None)
        return bool(fn()) if callable(fn) else hasattr(c, "trim")

    # -- tap sanity probe (drafter modes) --
    def verify_tap(self) -> None:
        """Prove the manual mlx-lm tap is faithful for THIS model, or fail loudly.

        ``_run_mlxlm`` / ``_run_hybrid`` replicate the model's forward (embed → layers →
        norm → head). A family whose forward does more — embedding scaling (gemma),
        per-layer sliding-window masks, extra streams — would draft from a silently-wrong
        hidden stream, which wastes far more user time than an error. Two checks:
        (1) structural: refuse windowed/sliding attention (a short probe can't exercise a
        window, so it must be refused, not probed) — hybrid *linear* attention is fine,
        it has its own replicated path; (2) numeric: the replicated loop must reproduce
        the model's own logits on a tiny input (identical ops/widths → should match
        bit-for-bit; 1e-3 is generous headroom). Costs two 4-token forwards, once per
        load. VLM targets use the native capture hook — nothing replicated, nothing to
        verify."""
        if self.is_vlm:
            return
        args = getattr(self.model, "args", None)
        mt = getattr(args, "model_type", "?")
        layer_types = getattr(args, "layer_types", None) or []
        windowed = any(
            t not in ("full_attention", "linear_attention") for t in layer_types
        ) or (
            getattr(args, "use_sliding_window", False)
            and getattr(args, "sliding_window", None)
        )
        if windowed:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: it uses "
                f"windowed/alternating attention layers, which the generic tap's single "
                f"causal mask would get wrong past the window."
            )
        ids = mx.array([[1, 2, 3, 4]])
        try:
            ref = self.plain(ids, self.make_cache())
            if self.is_hybrid:
                # route through verify() so the capture hooks are active during the
                # probe — proves the recording plumbing is a pure pass-through too
                got, _ = self.verify(ids, self.make_cache(), [0])
                self.reset_spec()
            else:
                got, _ = self._run_mlxlm(ids, self.make_cache(), [0])
            diff = float(mx.abs(ref - got).max())
        except Exception as e:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the generic forward "
                f"loop failed ({e.__class__.__name__}: {e}). baseline/lookup modes still work."
            ) from e
        if diff > 1e-3:
            raise ValueError(
                f"hidden-state tap unsupported for model_type {mt!r}: the replicated forward "
                f"diverges from the model's own (max |Δlogit| = {diff:.4g}) — this family's "
                f"forward does more than embed→layers→norm. baseline/lookup modes still work."
            )
