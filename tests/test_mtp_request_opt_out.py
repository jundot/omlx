# SPDX-License-Identifier: Apache-2.0
"""Per-request Lightning MTP opt-out (``X-oMLX-MTP: off``)."""

from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx_lm.models.llama import Model, ModelArgs

from omlx.patches.mlx_lm_mtp import batch_generator as mtp
from omlx.request import Request, SamplingParams
from omlx.scheduler import Scheduler
from omlx.server import _mtp_kwargs


@pytest.fixture
def scheduler_probe(monkeypatch):
    mtp.apply()
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
    # Same probe as test_mtp_xtc_sampling: real scheduler batches, no trained head.
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

    def observe(batch):
        observed.append((list(batch.uids), original(batch)))
        return False

    monkeypatch.setattr(mtp, "_is_mtp_eligible", observe)
    monkeypatch.setattr(mtp, "_is_mtp_batch_eligible", lambda batch: False)
    mtp.MTP_OPT_OUT_UIDS.clear()
    try:
        yield scheduler, observed
    finally:
        mtp.MTP_OPT_OUT_UIDS.clear()
        if scheduler.batch_generator is not None:
            scheduler.batch_generator.close()


def insert(scheduler, max_tokens=8):
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    scheduler._ensure_batch_generator(params)
    sampler, processors = scheduler._build_sampler_and_processors(params)
    return scheduler.batch_generator.insert(
        [[1, 2, 3]],
        max_tokens=[max_tokens],
        samplers=[sampler],
        logits_processors=[processors],
    )[0]


def run_until_observed(scheduler, observed):
    for _ in range(8):
        scheduler.batch_generator.next()
        if observed:
            return observed[-1]
    pytest.fail("MTP eligibility was never evaluated")


@pytest.mark.parametrize("opted_out", [False, True])
def test_opt_out_uid_disables_mtp_for_that_request(scheduler_probe, opted_out):
    scheduler, observed = scheduler_probe
    uid = insert(scheduler)
    if opted_out:
        mtp.set_request_mtp_opt_out(uid, True)
    rows, eligible = run_until_observed(scheduler, observed)
    assert rows == [uid]
    assert eligible is (not opted_out)
    if opted_out:
        reason = mtp._ineligibility_reason(scheduler.batch_generator._generation_batch)
        assert "X-oMLX-MTP" in reason


def test_opt_out_of_another_uid_does_not_leak(scheduler_probe):
    scheduler, observed = scheduler_probe
    uid = insert(scheduler)
    mtp.MTP_OPT_OUT_UIDS.add(uid + 1000)
    _, eligible = run_until_observed(scheduler, observed)
    assert eligible is True


@pytest.mark.parametrize(
    "value,expected",
    [
        ("off", True),
        ("OFF", True),
        (" 0 ", True),
        ("false", True),
        ("on", False),
        ("", False),
        (None, False),
    ],
)
def test_mtp_header_parsing(value, expected):
    headers = {} if value is None else {"X-oMLX-MTP": value}
    req = SimpleNamespace(headers=headers)
    assert _mtp_kwargs(req) == ({"mtp_opt_out": True} if expected else {})
    assert _mtp_kwargs(None) == {}


def test_reused_uid_resets_opt_out():
    mtp.MTP_OPT_OUT_UIDS.clear()
    mtp.set_request_mtp_opt_out(7, True)
    assert 7 in mtp.MTP_OPT_OUT_UIDS
    mtp.set_request_mtp_opt_out(7, False)
    assert 7 not in mtp.MTP_OPT_OUT_UIDS


def test_request_defaults_to_mtp_allowed():
    req = Request(request_id="r", prompt="hi", sampling_params=SamplingParams())
    assert req.mtp_opt_out is False
