"""Uno transitions preserve emitted tokens and per-request KV state."""

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator, GenerationBatch

from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs
from omlx.patches.k2_horizon.uno_batch import install_cache_hooks, step
from omlx.utils.sampling import make_sampler
from test_k2_horizon import small_config


@pytest.mark.parametrize("late_join", [False, True])
def test_uno_handoff_preserves_eight_greedy_sequences(monkeypatch, late_join):
    mx.random.seed(41)
    model = Model(ModelArgs.from_dict(small_config()))
    model._uno_adapter_loaded = True
    model._omlx_uno_eos = []
    prompts = [[1, 2, 3] + [4 + i] * (i + 1) for i in range(8)]
    sampler = make_sampler(temp=0)

    def generate(uno):
        model._omlx_uno_enabled = uno
        model._omlx_uno_singleton = uno and late_join
        result = {}
        batch = BatchGenerator(
            model,
            max_tokens=24,
            sampler=sampler,
            completion_batch_size=8,
            prefill_batch_size=1,
        )
        uids = batch.insert(
            prompts[:1] if late_join else prompts,
            max_tokens=[24] * (1 if late_join else 8),
        )
        for uid in uids:
            result[uid] = []
        joins = not late_join
        while True:
            responses = batch.next_generated()
            if not responses:
                break
            for response in responses:
                result[response.uid].append(response.token)
                if response.prompt_cache is not None:
                    expected = prompts[response.uid] + result[response.uid]
                    for cache in response.prompt_cache:
                        assert cache.offset == len(expected)
                    resumed = model(mx.array([[19]]), cache=response.prompt_cache)[
                        :, -1
                    ]
                    reference = model(mx.array([expected + [19]]))[:, -1]
                    assert mx.allclose(resumed, reference, atol=1e-4).item()
            if late_join and not joins and len(result[0]) >= 1:
                if uno:
                    state = batch._generation_batch._omlx_uno_state
                    assert state.queued, "The join must interrupt buffered speculation"
                    snapshots = batch._generation_batch.extract_cache(0)
                    expected_tokens = prompts[0] + result[0]
                    assert all(c.offset == len(expected_tokens) for c in snapshots)
                    actual = model(mx.array([[19]]), cache=snapshots)[:, -1]
                    reference = model(mx.array([expected_tokens + [19]]))[:, -1]
                    assert mx.allclose(actual, reference, atol=1e-4).item()
                model._omlx_uno_singleton = False
                new_uids = batch.insert(prompts[1:], max_tokens=[24] * 7)
                result.update({uid: [] for uid in new_uids})
                joins = True
        return result

    expected = generate(False)
    original_step = GenerationBatch._step
    monkeypatch.setattr(
        GenerationBatch, "_step", lambda self: step(self, original_step)
    )
    install_cache_hooks()
    assert generate(True) == expected
