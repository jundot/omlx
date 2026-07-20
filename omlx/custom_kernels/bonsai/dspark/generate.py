# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding loop for Bonsai DSpark (batch=1, greedy and sampled).

Adapted from mlx-dspark's ``speculative_generate`` to use:
  - BonsaiTarget.run() for the hidden-state tap (handles hybrid SSM+FA model)
  - BonsaiDSparkDrafter for drafting (with log-SNR conditioning)
  - spec_decode_verify Metal kernel from the bonsai fast module (when available)

Generation flow (per round):
  1. Draft K tokens from the cross-attention backbone + Markov head.
  2. Run one target verify forward over [pending] + draft (K+1 tokens).
  3. Accept the matching prefix + 1 bonus token (≥1 token/round guaranteed).
  4. Trim the target KV / SSM cache and update the drafter context.

Output is exact greedy (temperature=0) or exact speculative sampling (temp>0)
of the target distribution — drafter quality only affects throughput (acceptance
rate), never output quality.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx

from .drafter import BonsaiDSparkDrafter, _CtxCache
from .target import BonsaiTarget

try:
    from ..fast import spec_decode_verify as _metal_verify
    _HAS_METAL_VERIFY = True
except Exception:
    _HAS_METAL_VERIFY = False


# ---------------------------------------------------------------------------
# Re-export streaming / sampling helpers from mlx-dspark when available
# ---------------------------------------------------------------------------

try:
    from mlx_dspark.generate import (
        GenResult,
        StopStreaming,
        _Penalizer,
        _Streamer,
        _finish_reason,
        _logprobs_for_block,
        _spec_sample_accept,
        eos_token_ids,
        encode_messages,
        encode_prompt,
    )
    from mlx_dspark.sampling import sample_probs, truncate_probs
    _MLXDSPARK_AVAILABLE = True
except ImportError:
    _MLXDSPARK_AVAILABLE = False
    # Minimal local fallbacks so the module is importable even without mlx-dspark
    @dataclass
    class GenResult:
        text: str
        token_ids: list[int]
        num_tokens: int
        num_rounds: int
        accept_lengths: list[int]
        target_forwards: int
        seconds: float
        finish_reason: str = "stop"
        logprobs: list | None = None

        @property
        def mean_accept_len(self) -> float:
            return self.num_tokens / max(self.num_rounds, 1)

        @property
        def tokens_per_sec(self) -> float:
            return self.num_tokens / max(self.seconds, 1e-9)

    class StopStreaming(Exception):
        pass

    def eos_token_ids(tokenizer) -> set[int]:
        ids: set[int] = set()
        for attr in ("eos_token_id", "eos_token_ids"):
            v = getattr(tokenizer, attr, None)
            if isinstance(v, int):
                ids.add(v)
            elif v:
                ids.update(int(x) for x in v)
        return ids

    def encode_prompt(tokenizer, prompt: str, use_chat: bool = True) -> list[int]:
        if use_chat and getattr(tokenizer, "chat_template", None):
            r = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], add_generation_prompt=True)
            if isinstance(r, list):
                return r
        return list(tokenizer.encode(prompt))


PREFILL_CHUNK = 2048


# ---------------------------------------------------------------------------
# Prefill helpers
# ---------------------------------------------------------------------------

def _prefill_tapped(
    target: BonsaiTarget,
    ids: list[int],
    cache,
    tap: list[int],
    drafter: BonsaiDSparkDrafter,
    ctx_caches: list[_CtxCache],
    ctx_offset: int = 0,
    log_snr: float | None = None,
    chunk: int = PREFILL_CHUNK,
):
    """Chunked prefill with hidden-state capture fed into drafter context."""
    logits = fused = None
    pos = ctx_offset
    many = len(ids) > chunk
    for i in range(0, len(ids), chunk):
        piece = ids[i:i + chunk]
        logits, fused = target.run(mx.array([piece]), cache, tap)
        drafter.update_context(fused, ctx_offset=pos, ctx_caches=ctx_caches, log_snr=log_snr)
        pos += len(piece)
        if many:
            mx.eval(logits, *[c.k for c in ctx_caches if c.k is not None])
            mx.clear_cache()
    return logits, fused


def _prefill_plain(target: BonsaiTarget, ids: list[int], cache, chunk: int = PREFILL_CHUNK):
    logits = None
    many = len(ids) > chunk
    for i in range(0, len(ids), chunk):
        logits = target.plain(mx.array([ids[i:i + chunk]]), cache)
        if many:
            mx.eval(logits)
            mx.clear_cache()
    return logits


# ---------------------------------------------------------------------------
# Verify helpers
# ---------------------------------------------------------------------------

def _greedy_verify(target: BonsaiTarget, verify_ids: mx.array, cache, tap: list[int]):
    """Run target verify forward; return (v_logits, v_fused)."""
    return target.run(verify_ids, cache, tap)


def _trim_cache(cache, trim: int) -> None:
    """Roll back target KV cache by ``trim`` tokens after a partial acceptance."""
    if trim <= 0:
        return
    for c in cache:
        if c is not None and hasattr(c, "trim"):
            c.trim(trim)


# ---------------------------------------------------------------------------
# Sampling (temperature=0 path uses metal verify when available)
# ---------------------------------------------------------------------------

def _sample(logits_row: mx.array, temperature: float, top_p: float = 1.0,
            top_k: int = 0) -> int:
    if temperature > 0.0 and _MLXDSPARK_AVAILABLE:
        probs = truncate_probs(mx.softmax(logits_row / temperature, axis=-1), top_p, top_k)
        return int(sample_probs(probs).item())
    return int(mx.argmax(logits_row).item())


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def speculative_generate(
    target: BonsaiTarget,
    tokenizer,
    drafter: BonsaiDSparkDrafter,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    cache=None,
    ctx_caches: list[_CtxCache] | None = None,
    reuse_len: int = 0,
    max_new_tokens: int = 512,
    max_draft_tokens: int | None = 2,
    confidence_threshold: float = 0.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    logprobs: int | None = None,
    seed: int | None = None,
    apply_chat_template: bool = True,
    stop: list[str] | None = None,
    on_text=None,
    log_snr: float | None = None,
) -> GenResult:
    """Speculative decoding for Bonsai 27B with DSpark drafter.

    Parameters
    ----------
    target:
        BonsaiTarget wrapping the loaded Qwen3.5 VLM.
    tokenizer:
        Model tokenizer.
    drafter:
        BonsaiDSparkDrafter with weights loaded and target embedding bound.
    prompt:
        Text prompt (encoded via chat template if ``apply_chat_template`` is True).
    prompt_ids:
        Pre-tokenized prompt token ids (overrides ``prompt``).
    max_draft_tokens:
        Max draft tokens per verify round (``cap``). Default 2 (Apple Silicon optimum).
        None = full block_size-1.
    log_snr:
        Log-SNR conditioning scalar. None = use ``drafter.config.log_snr_inference``.
    """
    if seed is not None:
        mx.random.seed(seed)

    if log_snr is None:
        log_snr = drafter.config.log_snr_inference

    cfg = drafter.config
    tap = list(cfg.target_layer_ids)
    k = cfg.block_size
    mask_id = cfg.mask_token_id
    cap_ceiling = k if max_draft_tokens is None else max(1, min(max_draft_tokens, k))
    cap = cap_ceiling

    eos_ids = eos_token_ids(tokenizer)

    if prompt_ids is not None:
        ids = prompt_ids
    elif _MLXDSPARK_AVAILABLE:
        ids = encode_prompt(tokenizer, prompt, use_chat=apply_chat_template)
    else:
        ids = list(tokenizer.encode(prompt)) if prompt else []

    if cache is None:
        cache = target.make_cache()
        ctx_caches = drafter.make_ctx_cache()
        reuse_len = 0

    # Streaming detokenizer
    if _MLXDSPARK_AVAILABLE:
        st = _Streamer(tokenizer, eos_ids, on_text, stop)
    else:
        st = _MinimalStreamer(tokenizer, eos_ids, on_text, stop)

    pen = _Penalizer(presence_penalty, frequency_penalty) if _MLXDSPARK_AVAILABLE else None

    t0 = time.time()

    # Prefill
    suffix = ids[reuse_len:] if reuse_len else ids
    logits, _ = _prefill_tapped(
        target, suffix, cache, tap, drafter, ctx_caches,
        ctx_offset=reuse_len, log_snr=log_snr,
    )
    n_cached = len(ids)

    pending = _sample(logits[0, -1], temperature, top_p, top_k)
    mx.async_eval([c.k for c in ctx_caches if c.k is not None])

    if pen is not None:
        pen.add([pending])
    lp_list: list | None = [] if (logprobs is not None and _MLXDSPARK_AVAILABLE) else None
    if lp_list is not None:
        lp_list.extend(_logprobs_for_block(logits[0, -1][None, :], [pending], logprobs))

    out_ids: list[int] = [pending]
    accept_lengths: list[int] = []
    target_forwards = 1

    if hasattr(st, "update"):
        st.update(out_ids)

    while len(out_ids) < max_new_tokens and pending not in eos_ids:
        if hasattr(st, "stopped") and st.stopped:
            break

        # ---- 1. Draft ----
        block_ids = [pending] + [mask_id] * (k - 1)
        block_offset = n_cached + len(out_ids) - 1

        base_logits = drafter._draft_logits(block_ids, ctx_caches, cap, block_offset)

        n_drafted = cap  # number of draft tokens proposed this round
        if temperature > 0.0 and _MLXDSPARK_AVAILABLE:
            draft_arr, q_probs = drafter.sample_block_probs(
                base_logits, pending, temperature, top_p, top_k)
            mx.eval(draft_arr, q_probs)
            drafted = [int(x) for x in draft_arr.tolist()]
            n_drafted = len(drafted)

            verify_ids = mx.array([[pending] + drafted])
            v_logits, v_fused = _greedy_verify(target, verify_ids, cache, tap)
            if pen is not None and pen.active:
                pen_delta = pen.block_penalty(v_logits.shape[-1], drafted, v_logits.dtype)
                v_logits = v_logits - pen_delta[None]
            mx.eval(v_logits, v_fused)

            n, repl = _spec_sample_accept(
                v_logits[0], drafted, q_probs, temperature, top_p, top_k)
            committed = drafted[:n] + [repl]
        else:
            # Greedy path — fused argmax verify
            draft_arr = drafter.sample_block(base_logits, pending)

            verify_ids = mx.concatenate(
                [mx.array([pending], dtype=draft_arr.dtype), draft_arr]
            ).reshape(1, -1)
            v_logits, v_fused = _greedy_verify(target, verify_ids, cache, tap)
            tt_arr = mx.argmax(v_logits[0], axis=-1)

            if pen is not None and pen.active:
                v_logits_pen = pen.apply(v_logits[0], draft_arr.tolist())
                tt_arr = mx.argmax(v_logits_pen, axis=-1)

            # Accept/reject using the Metal spec_decode_verify kernel when available
            if _HAS_METAL_VERIFY:
                n_arr, committed_arr = _metal_verify(
                    draft_arr.reshape(1, -1).astype(mx.int32),
                    v_logits.astype(mx.float32),
                )
                mx.eval(n_arr, committed_arr)
                n = int(n_arr[0].item())
                # committed_arr is [1, k+1] with accepted prefix + correction token + zeros
                committed = committed_arr[0].tolist()[:n + 1]
            else:
                mx.eval(tt_arr, draft_arr)
                n_drafted = int(draft_arr.shape[0])
                match = (draft_arr == tt_arr[:n_drafted]).astype(mx.int32)
                n_arr = mx.cumprod(match).sum()
                mx.eval(n_arr)
                n = int(n_arr.item())
                tt = tt_arr.tolist()
                drafted_list = draft_arr.tolist()
                committed = drafted_list[:n] + [tt[n]]

        target_forwards += 1
        accept_lengths.append(len(committed))

        # ---- 2. Trim rejected tail from target cache ----
        trim = n_drafted - n
        _trim_cache(cache, trim)

        # ---- 3. Update drafter context from accepted positions ----
        # v_fused is [1, K+1, n_tap*H] — keep [anchor, accepted] = n+1 positions
        pending_ctx = v_fused[:, : n + 1, :]
        drafter.update_context(pending_ctx, ctx_offset=block_offset, ctx_caches=ctx_caches,
                               log_snr=log_snr)

        # ---- 4. Commit tokens ----
        for tok in committed:
            out_ids.append(tok)
            if pen is not None:
                pen.add([tok])
            if tok in eos_ids:
                break
        pending = out_ids[-1]

        if hasattr(st, "update"):
            st.update(out_ids)

    if hasattr(st, "flush"):
        st.flush()

    secs = time.time() - t0
    if hasattr(st, "stopped") and st.stopped and hasattr(st, "text"):
        text = st.text
    else:
        text = tokenizer.decode([t for t in out_ids if t not in eos_ids])

    finish = "stop"
    if len(out_ids) >= max_new_tokens and pending not in eos_ids:
        finish = "length"

    return GenResult(
        text=text,
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(accept_lengths),
        accept_lengths=accept_lengths,
        target_forwards=target_forwards,
        seconds=secs,
        finish_reason=finish,
        logprobs=lp_list,
    )


def greedy_generate(
    target: BonsaiTarget,
    tokenizer,
    prompt: str = "",
    *,
    prompt_ids: list[int] | None = None,
    cache=None,
    reuse_len: int = 0,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    stop: list[str] | None = None,
    on_text=None,
) -> GenResult:
    """Plain (non-speculative) greedy decoding for Bonsai — matches target speed baseline."""
    if prompt_ids is not None:
        ids = prompt_ids
    elif _MLXDSPARK_AVAILABLE:
        ids = encode_prompt(tokenizer, prompt, use_chat=True)
    else:
        ids = list(tokenizer.encode(prompt))

    eos_ids = eos_token_ids(tokenizer)
    if cache is None:
        cache = target.make_cache()
        reuse_len = 0

    if _MLXDSPARK_AVAILABLE:
        st = _Streamer(tokenizer, eos_ids, on_text, stop)
    else:
        st = _MinimalStreamer(tokenizer, eos_ids, on_text, stop)

    t0 = time.time()
    suffix = ids[reuse_len:] if reuse_len else ids
    logits = _prefill_plain(target, suffix, cache)
    out_ids: list[int] = []

    y = mx.argmax(logits[0, -1])
    mx.async_eval(y)
    while True:
        logits = target.plain(y.reshape(1, 1), cache)
        y_next = mx.argmax(logits[0, -1])
        mx.async_eval(y_next)
        nxt = int(y.item())
        out_ids.append(nxt)
        if hasattr(st, "update"):
            st.update(out_ids)
        if len(out_ids) >= max_new_tokens or nxt in eos_ids:
            break
        if hasattr(st, "stopped") and st.stopped:
            break
        y = y_next

    if hasattr(st, "flush"):
        st.flush()

    secs = time.time() - t0
    return GenResult(
        text=tokenizer.decode([t for t in out_ids if t not in eos_ids]),
        token_ids=out_ids,
        num_tokens=len(out_ids),
        num_rounds=len(out_ids),
        accept_lengths=[1] * len(out_ids),
        target_forwards=len(out_ids),
        seconds=secs,
        finish_reason="stop" if out_ids and out_ids[-1] in eos_ids else "length",
    )


# ---------------------------------------------------------------------------
# Minimal fallback streamer (used when mlx-dspark is not installed)
# ---------------------------------------------------------------------------

class _MinimalStreamer:
    def __init__(self, tokenizer, eos_ids, on_text, stop):
        self._tok = tokenizer
        self._eos = eos_ids
        self._on_text = on_text
        self._stop = [s for s in (stop or []) if s]
        self._ids: list[int] = []
        self.stopped = False
        self.text = ""

    def update(self, out_ids: list[int]) -> None:
        self._ids = list(out_ids)
        self.text = self._tok.decode([t for t in self._ids if t not in self._eos])
        if self._stop:
            for s in self._stop:
                if s in self.text:
                    cut = self.text.find(s)
                    self.text = self.text[:cut]
                    self.stopped = True
                    break
        if self._on_text:
            try:
                self._on_text(self.text)
            except Exception:
                pass

    def flush(self) -> None:
        pass
