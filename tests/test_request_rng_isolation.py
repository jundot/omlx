# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import mlx.core as mx
import pytest

from omlx.patches.mlx_lm_mtp.batch_generator import _stochastic_verify_tokens
from omlx.patches.mlx_lm_mtp.batched_head import _sample_rows
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler


def build_request_sampler(
    request_id, seed, xtc_probability=0.0, temperature=0.8, **scheduler_settings
):
    scheduler = SimpleNamespace(
        _xtc_special_tokens=[], _model_suppress_tokens=[], **scheduler_settings
    )
    params = SamplingParams(
        seed=seed,
        temperature=temperature,
        top_p=1.0,
        top_k=3,
        xtc_probability=xtc_probability,
        xtc_threshold=0.19,
    )
    request = Request(request_id=request_id, prompt="sample", sampling_params=params)
    sampler, _ = Scheduler._build_sampler_and_processors(scheduler, params, request)
    return scheduler, request, sampler


def request_sampler(request_id, seed, xtc_probability):
    return build_request_sampler(request_id, seed, xtc_probability)[2]


@pytest.mark.parametrize("xtc_probability", [0.0, 0.4])
def test_request_seed_survives_other_draws(xtc_probability):
    scores = mx.log(mx.array([[0.26, 0.24, 0.20, 0.19, 0.11]]))

    def run(peer_seed):
        mx.random.seed(7)
        target = request_sampler("target", 7, xtc_probability)
        first = target(scores).item()
        # A legacy speculative request can still reseed the shared generator.
        mx.random.seed(peer_seed)
        peer = request_sampler("peer", peer_seed, xtc_probability)
        result = [first]
        for _ in range(95):
            peer(scores).item()
            result.append(target(scores).item())
        return result

    baseline = run(11)
    assert len(set(baseline)) > 1
    assert baseline == run(97)


def test_unseeded_request_does_not_clone_global_stream():
    scores = mx.log(mx.array([[0.26, 0.24, 0.20, 0.19, 0.11]]))

    def run():
        mx.random.seed(7)
        draw = request_sampler("unseeded", None, 0.0)
        return [draw(scores).item() for _ in range(96)]

    assert run() != run()


def rng_state():
    return mx.random.state[0].tolist()


@pytest.mark.parametrize("temperature", [-0.5, 0.0, 0.8])
def test_ordinary_sampling_leaves_legacy_global_rng_untouched(temperature):
    scheduler, request, sampler = build_request_sampler(
        "target", 7, temperature=temperature
    )
    mx.random.seed(19)
    expected = rng_state()
    Scheduler._seed_legacy_request_sampling(scheduler, request)
    sampler(mx.log(mx.array([[0.26, 0.24, 0.20, 0.19, 0.11]]))).item()
    assert rng_state() == expected


@pytest.mark.parametrize(
    "scheduler_settings",
    [
        {"model": SimpleNamespace(_omlx_mtp_decode_enabled=True)},
        {
            "model": SimpleNamespace(
                language_model=SimpleNamespace(_omlx_mtp_decode_enabled=True)
            )
        },
        {"_vlm_mtp_drafter": object()},
    ],
)
def test_speculative_sampler_preserves_legacy_admission_seed(scheduler_settings):
    scores = mx.log(mx.array([[0.2, 0.3, 0.3, 0.2], [0.3, 0.25, 0.25, 0.2]]))
    drafts = mx.array([0], dtype=mx.int32)
    draft_density = [mx.log(mx.array([0.6, 0.15, 0.15, 0.1]))]
    samples = []
    for ambient_seed in (11, 97, 388, 440, 1097, 1759):
        mx.random.seed(ambient_seed)
        scheduler, request, sampler = build_request_sampler(
            "legacy", 7, **scheduler_settings
        )
        assert sampler._mtp_batch_sampling_key is not None
        Scheduler._seed_legacy_request_sampling(scheduler, request)
        samples.append(
            _stochastic_verify_tokens(sampler, scores, drafts, draft_density).tolist()
        )
    assert all(result == samples[0] for result in samples)


def test_owned_sampler_is_not_shared_in_batched_draft_sampling():
    scores = mx.log(mx.array([[0.26, 0.24, 0.20, 0.19, 0.11]]))
    expected = [build_request_sampler(str(seed), seed)[2] for seed in (7, 97)]
    test = [build_request_sampler(str(seed), seed)[2] for seed in (7, 97)]
    for _ in range(32):
        actual, _ = _sample_rows(test, mx.concatenate([scores, scores]))
        assert actual.tolist() == [
            sampler.sample_with_logprobs(scores)[0].item() for sampler in expected
        ]


def test_requeued_legacy_sampler_does_not_set_global_seed_again():
    scheduler, request, sampler = build_request_sampler(
        "requeued", 7, model=SimpleNamespace(_omlx_mtp_decode_enabled=True)
    )
    Scheduler._seed_legacy_request_sampling(scheduler, request)
    sampler(mx.log(mx.array([[0.26, 0.24, 0.20, 0.19, 0.11]]))).item()
    expected = rng_state()
    request.requeued_output_tokens = 1
    Scheduler._seed_legacy_request_sampling(scheduler, request)
    assert rng_state() == expected
