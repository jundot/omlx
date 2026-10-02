# SPDX-License-Identifier: Apache-2.0
"""Exercise Lightning MTP eligibility through real scheduler-created batches."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.llama import Model, ModelArgs

from omlx.patches.mlx_lm_mtp import batch_generator as mtp
from omlx.request import SamplingParams
from omlx.scheduler import Scheduler
from omlx.utils.sampling import make_sampler


@pytest.fixture
def scheduler_probe(monkeypatch):
    mtp.apply()
    monkeypatch.setenv("OMLX_MTP_ROWWISE_BATCH", "1")
    model = Model(
        ModelArgs(
            model_type="llama",
            hidden_size=32,
            num_hidden_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            rms_norm_eps=1e-5,
            vocab_size=64,
        )
    )
    # Run real prefill/standard decode, observing the MTP dispatch boundary.
    # This small model has no trained MTP head.
    model.mtp = object()
    model.mtp_forward = lambda *args, **kwargs: None
    model._omlx_mtp_decode_enabled = True
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.model = model
    scheduler.config = SimpleNamespace(completion_batch_size=8, prefill_step_size=32)
    scheduler._xtc_special_tokens = []
    scheduler._model_suppress_tokens = []
    scheduler._get_stop_tokens = lambda: set()
    scheduler._stream = mx.new_stream(mx.gpu)
    scheduler.batch_generator = None
    observed = []
    original = mtp._is_mtp_eligible
    original_batch = mtp._is_mtp_batch_eligible

    def observe(batch):
        observed.append((list(batch.uids), original(batch), original_batch(batch)))
        return False

    monkeypatch.setattr(mtp, "_is_mtp_eligible", observe)
    monkeypatch.setattr(mtp, "_is_mtp_batch_eligible", lambda batch: False)
    try:
        yield scheduler, observed
    finally:
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


def insert(scheduler, probability, temperature=1.0, max_tokens=8):
    params = SamplingParams(
        temperature=temperature,
        xtc_probability=probability,
        xtc_threshold=0.1,
        max_tokens=max_tokens,
    )
    scheduler._ensure_batch_generator(params)
    sampler, processors = scheduler._build_sampler_and_processors(params)
    return scheduler.batch_generator.insert(
        [[1, 2, 3]],
        max_tokens=[max_tokens],
        samplers=[sampler],
        logits_processors=[processors],
    )[0]


@pytest.mark.parametrize("first,second", [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)])
def test_reused_generator_follows_request_sampler(scheduler_probe, first, second):
    scheduler, observed = scheduler_probe
    owner = None
    for probability in (first, second):
        uid = insert(scheduler, probability)
        generator = scheduler.batch_generator
        if owner is None:
            owner = generator
        assert generator is owner
        observed.clear()
        for _ in range(32):
            generator.next()
            if observed and not generator._generation_batch.uids:
                break
        else:
            pytest.fail("request did not finish")
        assert observed
        assert all(rows == [uid] for rows, _, _ in observed)
        assert all(single == (probability == 0.0) for _, single, _ in observed)


@pytest.mark.parametrize("xtc_first", [False, True])
def test_late_join_mixed_batch_and_filter(scheduler_probe, xtc_first):
    scheduler, observed = scheduler_probe
    first = insert(scheduler, float(xtc_first), max_tokens=24)
    generator = scheduler.batch_generator
    for _ in range(8):
        generator.next()
        if observed:
            break
    assert observed[-1][1] == (not xtc_first)
    second = insert(scheduler, float(not xtc_first), max_tokens=24)
    for _ in range(8):
        generator.next()
        if len(generator._generation_batch.uids) == 2:
            break
    batch = generator._generation_batch
    assert batch.uids == [first, second]
    generator.next()
    assert observed[-1] == ([first, second], False, False)
    assert "XTC" in mtp._ineligibility_reason(batch)
    # Removing the XTC row must not leave a sticky generator-wide veto.
    remaining = second if xtc_first else first
    batch.filter([batch.uids.index(remaining)])
    generator.next()
    assert observed[-1] == ([remaining], True, False)


def test_greedy_ignores_xtc(scheduler_probe):
    scheduler, observed = scheduler_probe
    uid = insert(scheduler, 1.0, temperature=0.0)
    for _ in range(8):
        scheduler.batch_generator.next()
        if observed:
            break
    assert observed[-1] == ([uid], True, False)


@pytest.mark.parametrize("override", [None, 0.0, 1.0])
@pytest.mark.parametrize("fallback", [0.0, 1.0])
def test_sampler_override_and_fallback(override, fallback):
    batch = SimpleNamespace(
        uids=[1],
        samplers=[
            (
                None
                if override is None
                else make_sampler(temp=1.0, xtc_probability=override)
            )
        ],
        fallback_sampler=make_sampler(temp=1.0, xtc_probability=fallback),
    )
    assert mtp._has_xtc_sampler(batch) == (
        (fallback if override is None else override) > 0.0
    )
