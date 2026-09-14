# SPDX-License-Identifier: Apache-2.0
"""External prefill preserves KV and token progress across eviction pauses."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler, SchedulerConfig, _PrefillEvictionNeeded


class _RecordingModel:
    """Store input tokens in KV so retries expose duplication or recomputation."""

    def __init__(self):
        self.layers = [SimpleNamespace()]
        self.args = SimpleNamespace(num_hidden_layers=1)
        self.seen = []

    def __call__(self, inputs, cache=None, **kwargs):
        self.seen.extend(inputs[0].tolist())
        values = inputs[:, None, :, None].astype(mx.float32)
        cache[0].update_and_fetch(values, values)
        return mx.zeros((1, inputs.shape[1], 8))

    def make_cache(self):
        return [KVCache()]

    def parameters(self):
        return {}


@pytest.mark.parametrize("cached_tokens", [0, 4], ids=["cold", "warm"])
@pytest.mark.parametrize("pause_after_chunks", [0, 1, 2])
@pytest.mark.parametrize("pause_count", [1, 2])
@pytest.mark.parametrize("route", ["adaptive", "guard"])
def test_external_prefill_resumes_without_replaying_tokens(
    mock_tokenizer, monkeypatch, cached_tokens, pause_after_chunks, pause_count, route
):
    model = _RecordingModel()
    scheduler = Scheduler(
        model=model,
        tokenizer=mock_tokenizer,
        config=SchedulerConfig(prefill_step_size=4),
    )
    prompt = list(range(100, 132))
    request = Request(
        request_id="req-pause", prompt=prompt, sampling_params=SamplingParams()
    )
    request.prompt_token_ids = prompt
    request.num_prompt_tokens = len(prompt)
    request.cached_tokens = cached_tokens
    request.remaining_tokens = prompt[cached_tokens:]
    if cached_tokens:
        request.prompt_cache = model.make_cache()
        model(mx.array(prompt[:cached_tokens])[None], cache=request.prompt_cache)
    scheduler.requests[request.request_id] = request

    scheduler._memory_limit_bytes = 80
    scheduler._memory_hard_limit_bytes = 100
    scheduler._memory_abort_limit_bytes = 100
    scheduler._prefill_abort_margin = 0.9
    scheduler._prefill_min_chunk_tokens = 4
    pause_at = cached_tokens + 4 * pause_after_chunks

    def current_usage():
        return 60 if len(model.seen) >= pause_at else 0

    monkeypatch.setattr(scheduler, "_current_usage_bytes", current_usage)
    monkeypatch.setattr(scheduler, "_reclaim_prefill_headroom", current_usage)
    # The guard also charges observed peaks, which can exceed the throttle's estimate.
    monkeypatch.setattr(
        scheduler,
        "_predicted_chunk_transient",
        lambda *args, **kwargs: 50 if route == "adaptive" else 4,
    )
    monkeypatch.setattr(scheduler, "_admission_transient_bound", lambda *a, **kw: 50)

    for _ in range(pause_count):
        with pytest.raises(_PrefillEvictionNeeded) as exc:
            scheduler._do_external_prefill(
                request, request.remaining_tokens, request.prompt_cache
            )
        expected_reason = (
            "adaptive_prefill_throttle" if route == "adaptive" else "prefill_safety_cap"
        )
        assert exc.value.request.reason == expected_reason
        assert request.cached_tokens == pause_at
        assert request.remaining_tokens == prompt[pause_at:]
        if pause_at:
            assert request.prompt_cache[0].offset == pause_at
        else:
            assert request.prompt_cache is None
        scheduler._pause_for_prefill_eviction(request, exc.value.request)
        assert scheduler.waiting.popleft() is request
        pause_at += 4

    pause_at = len(prompt) + 1
    cache, last_token = scheduler._do_external_prefill(
        request, request.remaining_tokens, request.prompt_cache
    )
    assert last_token == prompt[-1:]
    assert cache[0].offset == len(prompt) - 1
    model(mx.array(last_token)[None], cache=cache)
    assert model.seen == prompt
    assert cache[0].keys[0, 0, : cache[0].offset, 0].tolist() == prompt
