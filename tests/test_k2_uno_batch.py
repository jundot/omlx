"""Uno transitions preserve emitted tokens and per-request KV state."""

import mlx.core as mx
import pytest
from mlx_lm.generate import BatchGenerator, GenerationBatch

from omlx.patches.k2_horizon.k2_horizon_model import Model, ModelArgs
from omlx.patches.k2_horizon.uno_batch import install_cache_hooks
from omlx.patches.k2_horizon.uno_adapter import ConditionalLoRALinear, TARGETS
from omlx.patches.k2_horizon.compiled import install_compiled_blocks
from omlx.scheduler import Scheduler, SchedulerConfig
from omlx.utils.sampling import make_sampler
from test_k2_horizon import small_config


@pytest.mark.parametrize("late_join", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("constrained", [False, True])
def test_uno_handoff_preserves_eight_greedy_sequences(late_join, compiled, constrained):
    mx.random.seed(41)
    model = Model(ModelArgs.from_dict(small_config()))
    for layer in model.layers:
        for scope, names in TARGETS.items():
            owner = getattr(layer, scope)
            for name in names:
                linear = getattr(owner, name)
                n, k = linear.weight.shape
                setattr(
                    owner,
                    name,
                    ConditionalLoRALinear(
                        linear,
                        mx.random.normal((4, k)) * 0.02,
                        mx.random.normal((n, 4)) * 0.02,
                        2.0,
                    ),
                )
    if compiled:
        install_compiled_blocks(model)
    model._uno_adapter_loaded = True
    model._omlx_uno_eos = []
    prompts = [[1, 2, 3] + [4 + i] * (i + 1) for i in range(8)]
    sampler = make_sampler(temp=0)
    grammars = []
    expected_text = ["read_file:" + chr(65 + i) * 14 for i in range(8)]
    if constrained:
        xgr = pytest.importorskip("xgrammar")

        info = xgr.TokenizerInfo(
            [bytes([i]) for i in range(128)], xgr.VocabType.RAW, stop_token_ids=[0]
        )
        compiler = xgr.GrammarCompiler(info)
        grammars = [
            compiler.compile_grammar('root ::= "' + value + '"')
            for value in expected_text
        ]

    def generate(uno):
        model._omlx_uno_enabled = uno
        model._omlx_uno_singleton = uno and late_join
        result = {}
        from omlx.api.grammar import GrammarConstraintProcessor

        processors = (
            [[GrammarConstraintProcessor(g, 128)] for g in grammars]
            if constrained
            else [[] for _ in range(8)]
        )
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
            logits_processors=processors[:1] if late_join else processors,
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
                new_uids = batch.insert(
                    prompts[1:], max_tokens=[24] * 7, logits_processors=processors[1:]
                )
                result.update({uid: [] for uid in new_uids})
                joins = True
        return result

    expected = generate(False)
    install_cache_hooks()
    assert generate(True) == expected
    if constrained:
        assert [bytes(row).decode() for row in expected.values()] == expected_text


@pytest.mark.parametrize("temperature", [0.4, 1.0, 1.7])
@pytest.mark.parametrize("top_p", [0.6, 1.0])
@pytest.mark.parametrize("top_k", [0, 3])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_uno_target_matches_ordinary_sampler(
    monkeypatch, temperature, top_p, top_k, dtype
):
    from omlx.patches.k2_horizon.uno_decode import probabilities

    logits = mx.array([[0.1, 0.4, 0.7, 1.1, -1.2]], dtype=dtype)
    logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    monkeypatch.setattr(
        mx.random,
        "categorical",
        lambda values: mx.softmax(values.astype(mx.float32), axis=-1),
    )
    ordinary = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)(logprobs)
    uno = probabilities(logits, temperature, top_p, top_k or None)
    assert mx.array_equal(ordinary, uno).item()
