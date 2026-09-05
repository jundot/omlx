# SPDX-License-Identifier: Apache-2.0
"""An eviction pause MID external prefill must not lose the progress made.

The external prefill loop sizes each chunk through the adaptive throttle,
which may raise ``_PrefillEvictionNeeded`` — including AFTER chunks have
already entered the cache (the ``prompt_cache`` object is advanced in
place). The pause re-queued the request with the ``remaining_tokens`` and
``cached_tokens`` it started with, so on retry the same tokens were fed
again on top of the advanced cache: a DUPLICATED span in the KV, boundary
snapshots labelled one block behind what the cache actually held
(``tc=7680 ... kv=8192``, measured on GLM-5.3-Flash), and every block
stored from there on misaligned — the next request that hit that prefix
generated garbage ("NoNo", "DEDEDE"). It only showed up with the MTP head
enabled, because only then did memory pressure push the throttle far
enough to pause a chunk.
"""
from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillEvictionNeeded


class _CountingModel:
    """Only pushes positions into a KVCache: the offset is what matters."""

    def __init__(self):
        self.layers = [SimpleNamespace()]
        self.args = SimpleNamespace(num_hidden_layers=1)

    def __call__(self, inputs, cache=None, **kwargs):
        S = inputs.shape[1]
        cache[0].update_and_fetch(mx.zeros((1, 1, S, 4)), mx.zeros((1, 1, S, 4)))
        return mx.zeros((1, S, 8))

    def make_cache(self):
        return [KVCache()]

    def parameters(self):
        return {}


def test_pause_mid_external_prefill_commits_the_progress(mock_tokenizer):
    model = _CountingModel()
    scheduler = Scheduler(
        model=model, tokenizer=mock_tokenizer, config=SchedulerConfig(prefill_step_size=4)
    )
    # restored prefix: 4 tokens already in the cache
    cache = model.make_cache()
    model(mx.zeros((1, 4), dtype=mx.int32), cache=cache)
    prompt = list(range(100, 116))  # 16 tokens; 12 to go (the last is the generator's)
    request = Request(request_id="req-pause", prompt=prompt, sampling_params=SamplingParams())
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.cached_tokens = 4
    request.remaining_tokens = prompt[4:]
    request.prompt_cache = cache
    scheduler.requests[request.request_id] = request

    calls = {"n": 0}
    original = scheduler._adaptive_chunk_size

    def throttle_on_the_second(requested, **kw):
        calls["n"] += 1
        if calls["n"] == 2:
            raise _PrefillEvictionNeeded(
                SimpleNamespace(reason="adaptive_prefill_throttle", request_id=request.request_id)
            )
        return original(requested, **kw)

    scheduler._adaptive_chunk_size = throttle_on_the_second

    with pytest.raises(_PrefillEvictionNeeded):
        scheduler._do_external_prefill(request, request.remaining_tokens, cache)

    # the cache advanced by ONE chunk (4 tokens) before the pause
    assert cache[0].offset == 8
    # and the request has to know it: what is left starts AFTER what went in
    assert request.cached_tokens == 8, request.cached_tokens
    assert request.remaining_tokens == prompt[8:], request.remaining_tokens
    assert request.prompt_cache is cache


def test_pause_on_a_cold_prompt_commits_the_progress_too(mock_tokenizer):
    """The case the first fix missed: with no restored prefix, the pause restarted at token zero.

    Measured 05/09 on GLM-5.3-Flash oQ2e: a cold 229,923-token prompt was paused at
    206,336 tokens for headroom (111.76 GB against a 112.48 GB target), the eviction
    reclaimed 1.56 GB, and the request was re-admitted as new (new=229923, kv_exact=0)
    — 19 minutes of prefill thrown away, with nothing preventing the same at 206k again.
    The locally built cache has to go onto the request, which is where the retry reads it.
    """
    model = _CountingModel()
    scheduler = Scheduler(
        model=model, tokenizer=mock_tokenizer, config=SchedulerConfig(prefill_step_size=4)
    )
    prompt = list(range(100, 116))
    request = Request(request_id="req-cold", prompt=prompt, sampling_params=SamplingParams())
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.cached_tokens = 0
    request.remaining_tokens = prompt
    request.prompt_cache = None
    scheduler.requests[request.request_id] = request

    calls = {"n": 0}
    original = scheduler._adaptive_chunk_size

    def throttle_on_third(requested, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise _PrefillEvictionNeeded(
                SimpleNamespace(reason="adaptive_prefill_throttle", request_id=request.request_id)
            )
        return original(requested, **kw)

    scheduler._adaptive_chunk_size = throttle_on_third
    with pytest.raises(_PrefillEvictionNeeded):
        scheduler._do_external_prefill(request, prompt, None)

    # two chunks of 4 went in before the pause; the cache must be ON THE REQUEST
    assert request.prompt_cache is not None
    assert request.prompt_cache[0].offset == 8
    assert request.cached_tokens == 8, request.cached_tokens
    assert request.remaining_tokens == prompt[8:], request.remaining_tokens
