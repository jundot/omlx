"""Exercise CSA2 rollback and the shared DSpark generation loop."""

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41 import load_reference_weights, tiny

from omlx.patches.deepseek_v41.language import LanguageModel


def mtp_config(**kwargs):
    return tiny(
        preserve_mtp=True,
        n_mtp_layers=3,
        dspark_block_size=3,
        dspark_noise_token_id=2,
        dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2,
        dspark_n_activated_experts=1,
        dspark_markov_rank=32,
        compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        **kwargs,
    )


@pytest.mark.parametrize("ratio", [2, 4, 8])
@pytest.mark.parametrize("prefix", [1, 3, 4, 5, 7, 8])
@pytest.mark.parametrize("accepted", [0, 1, 2, 3])
def test_verify_rollback_restores_all_csa2_slots(prefix, accepted, ratio, monkeypatch):
    config = replace(mtp_config(), compress_ratios=(0, ratio, ratio, 1, 1, 0, 0, 0))
    model = LanguageModel(config)
    load_reference_weights(model)
    model.configure_mtp(True, 3)
    cache, expected = model.make_cache(), model.make_cache()
    prompt = mx.array([[3 + i % 20 for i in range(prefix)]])
    model(prompt, cache=cache)
    model(prompt, cache=expected)
    block = mx.array([[21, 22, 23, 24]])
    model(block, cache=cache, return_hidden=True, n_confirmed=1)

    def fail_replay(*args, **kwargs):
        raise AssertionError("Rollback must not execute the target backbone")

    with monkeypatch.context() as patch:
        patch.setattr(model, "_forward", fail_replay)
        assert model.mtp_partial_rollback(cache, accepted, 3)
    assert cache[0]._mtp_draft_stash is None
    model(block[:, : accepted + 1], cache=expected, return_hidden=True)
    for actual, wanted in zip(cache, expected):
        assert actual.size() == wanted.size() == prefix + accepted + 1
        np.testing.assert_array_equal(actual.left_padding, wanted.left_padding)
        for a, b in zip(actual.state, wanted.state):
            np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)
    following = mx.array([[25]])
    a = model(following, cache=cache)
    b = model(following, cache=expected)
    np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-5)


def test_rollback_restores_engram_history():
    from omlx.patches.deepseek_v41.language import LanguageModel

    model = LanguageModel(
        tiny(
            engram_layer_ids=(1,),
            engram_num_embeddings=(72,),
            engram_vocab_size=5,
            engram_max_ngram_size=4,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_compressed_vocab_size=64,
        )
    )
    model.set_token_map(np.arange(64))
    cache, expected = model.make_cache(), model.make_cache()
    for state in (cache, expected):
        model(mx.array([[3, 4, 5, 6, 7]]), cache=state)
    model(mx.array([[8, 9, 10, 11]]), cache=cache, n_confirmed=1)
    assert model.mtp_partial_rollback(cache, 1, 3)
    model(mx.array([[8, 9]]), cache=expected)
    np.testing.assert_array_equal(cache[0][6], expected[0][6])
    for a, b in zip(cache, expected):
        for x, y in zip(a.state, b.state):
            np.testing.assert_allclose(x, y, atol=1e-5)


def test_shared_batch_generator_activates_and_matches_greedy(monkeypatch):
    from mlx_lm.generate import BatchGenerator

    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    model = LanguageModel(mtp_config())
    load_reference_weights(model)
    prompt = [3, 4, 5, 6, 7, 8, 9]
    expected, cache = [], model.make_cache()
    ids = mx.array([prompt])
    for _ in range(16):
        logits = model(ids, cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        expected.append(token)
        ids = mx.array([[token]])
    model.configure_mtp(True, 3)
    calls = []
    original = type(model).dspark_forward

    def tracked(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "dspark_forward", tracked)
    controllers = []
    original_factory = model.make_mtp_depth_controller

    def make_controller(depth):
        controller = original_factory(depth)
        controllers.append(controller)
        return controller

    monkeypatch.setattr(model, "make_mtp_depth_controller", make_controller)
    generator = BatchGenerator(
        model,
        max_tokens=16,
        prefill_step_size=3,
        sampler=lambda logits: mx.argmax(logits, -1),
    )
    result = []
    try:
        generator.insert([prompt])
        for _ in range(64):
            _, responses = generator.next()
            result.extend(response.token for response in responses)
            if len(result) == 16:
                break
    finally:
        generator.close()
    assert controllers, "The shared loop must use the model depth policy"
    assert calls, "The test must exercise actual DSpark proposal execution"
    assert result == expected


def test_mtp_load_flag_is_instance_owned(monkeypatch):
    from omlx.patches import mlx_lm_mtp
    from omlx.patches.deepseek_v41.language import LanguageModel

    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", True)
    enabled = LanguageModel(mtp_config())
    monkeypatch.setattr(mlx_lm_mtp, "_MTP_ACTIVE", False)
    disabled = LanguageModel(mtp_config())
    assert enabled._omlx_dspark_decode_enabled
    assert not disabled._omlx_dspark_decode_enabled
    assert disabled.mtp  # Weight preservation is independent of execution.
    assert enabled._omlx_mtp_rowwise_unsupported


def test_v41_dspark_config_is_eligible():
    from omlx.utils.model_loading import _is_mtp_compatible

    assert _is_mtp_compatible(
        {"dspark_block_size": 5, "dspark_target_layer_ids": [37, 38, 39]},
        "deepseek_v41",
    )
    assert not _is_mtp_compatible({"n_mtp_layers": 0}, "deepseek_v41")


def test_acceptance_depth_is_independent_of_timing():
    from omlx.patches.deepseek_v41.mtp import AcceptanceDepthController

    a, b = AcceptanceDepthController(3), AcceptanceDepthController(3)
    depths = []
    for accepted in [0, 1, 2, 3, 1, 0, 1, 2, 3]:
        assert a.cur == b.cur
        used = a.cur
        a.observe(used, accepted, 0.01)
        b.observe(used, accepted, 100000, time_sample=False)
        assert a.cur == b.cur
        assert not a.should_exit() and not b.should_exit()
        depths.append(a.cur)
    assert depths == [1, 2, 3, 3, 2, 1, 2, 3, 3]
