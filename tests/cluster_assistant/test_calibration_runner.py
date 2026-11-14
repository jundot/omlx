# SPDX-License-Identifier: Apache-2.0
"""The real-model calibration runner (fake-driven).

``calibrate_node`` drives one warmup + one measured completion through an
injected seam — the same completion shape ``CoordinatorOps.observe`` uses —
and converts oMLX's own usage fields into a ``ComputeProfile``. No real
Mac/model/SSH here (AGENTS.md): the seam is a fake that records bodies and
replays usage replies. Live evidence belongs in PHASE-2B-RUN.md.
"""

from __future__ import annotations

import pytest

from omlx.cluster.assistant.calibration import (
    CALIBRATION_MAX_TOKENS,
    CALIBRATION_MODEL_DIR,
    CALIBRATION_PROMPT,
    CALIBRATION_WARMUP_PROMPT,
    CALIBRATION_WARMUP_TOKENS,
    CompletionTimings,
    calibrate_node,
    calibration_body,
    calibration_weight_bytes,
    timings_from_reply,
)
from omlx.cluster.planner import ModelLayout, synthetic_model_layout

# dev-1.5b is ~1 GB of weights; a round number keeps rates readable as GB/s.
WEIGHT_BYTES = 1_000_000_000


def usage_reply(**overrides) -> dict:
    """One oMLX completion reply, in the shape the observe seam returns."""

    usage = {
        "generation_tokens_per_second": 500.0,
        "prompt_tokens_per_second": 4000.0,
        "time_to_first_token": 0.02,
        "prompt_tokens": 10,
    }
    usage.update(overrides)
    return {"usage": usage}


class FakeSeam:
    """An async completion seam: records bodies, replays scripted replies.

    A reply that is an ``Exception`` instance is raised instead of returned,
    so a failed completion can be faked without a real transport.
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.bodies: list[dict] = []

    async def __call__(self, body):
        self.bodies.append(dict(body))
        reply = self.replies[len(self.bodies) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply


# ---------------------------------------------------------------------------
# The runner: warmup discarded, measured completion converted
# ---------------------------------------------------------------------------


async def test_calibrate_node_runs_warmup_then_measured_completion() -> None:
    seam = FakeSeam(usage_reply(), usage_reply())

    profile = await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES)

    assert profile is not None
    # decode reads the whole model once per token: 500 tok/s * 1 GB.
    assert profile.decode_weight_bytes_per_second == pytest.approx(500e9)
    # prefill reads the weights once per prompt: 4000 tok/s * 1 GB / 10 tokens.
    assert profile.prefill_weight_bytes_per_second == pytest.approx(400e9)
    assert profile.samples == 1

    # Two completions: the warmup first (discarded), the measured one second.
    assert len(seam.bodies) == 2
    warmup, measured = seam.bodies
    assert warmup["messages"][0]["content"] == CALIBRATION_WARMUP_PROMPT
    assert warmup["max_tokens"] == CALIBRATION_WARMUP_TOKENS
    assert measured["model"] == CALIBRATION_MODEL_DIR
    assert measured["messages"][0]["content"] == CALIBRATION_PROMPT
    assert measured["max_tokens"] == CALIBRATION_MAX_TOKENS
    # The measured request streams with include_usage so TTFT/prefill exist.
    assert measured["stream"] is True
    assert measured["stream_options"] == {"include_usage": True}


async def test_calibrate_node_accepts_a_sync_seam() -> None:
    bodies: list[dict] = []

    def seam(body):
        bodies.append(dict(body))
        return usage_reply()

    profile = await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES)

    assert profile is not None
    assert profile.decode_weight_bytes_per_second == pytest.approx(500e9)
    assert len(bodies) == 2


async def test_calibrate_node_honours_model_id_backend_and_timestamp() -> None:
    seam = FakeSeam(usage_reply(), usage_reply())

    profile = await calibrate_node(
        seam,
        model_weight_bytes=WEIGHT_BYTES,
        model_id="dev-1.5b",
        backend="jaccl-ring",
        measured_at="2026-11-13T00:00:00+00:00",
    )

    assert profile is not None
    assert profile.backend == "jaccl-ring"
    assert profile.measured_at == "2026-11-13T00:00:00+00:00"
    assert all(body["model"] == "dev-1.5b" for body in seam.bodies)


async def test_missing_prefill_reuses_decode_as_the_documented_stand_in() -> None:
    seam = FakeSeam(
        usage_reply(),
        {"usage": {"generation_tokens_per_second": 500.0, "prompt_tokens": 10}},
    )

    profile = await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES)

    assert profile is not None
    # The converter's documented conservative stand-in, not a fabricated rate.
    assert profile.prefill_weight_bytes_per_second == (
        profile.decode_weight_bytes_per_second
    )


# ---------------------------------------------------------------------------
# Unknown compute: a missing/failed reply is None, never a raise or a rate
# ---------------------------------------------------------------------------


async def test_missing_decode_is_unknown_compute() -> None:
    seam = FakeSeam(
        usage_reply(),
        {"usage": {"prompt_tokens_per_second": 4000.0, "prompt_tokens": 10}},
    )
    assert await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES) is None


@pytest.mark.parametrize("reply", [{}, {"usage": None}, None, "garbage", []])
async def test_a_reply_without_usage_is_unknown_compute(reply) -> None:
    seam = FakeSeam(usage_reply(), reply)
    assert await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES) is None


@pytest.mark.parametrize(
    "decode", [0.0, -5.0, float("nan"), float("inf"), True, "fast", None]
)
async def test_an_unusable_decode_rate_is_unknown_compute(decode) -> None:
    seam = FakeSeam(
        usage_reply(), usage_reply(generation_tokens_per_second=decode)
    )
    assert await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES) is None


async def test_a_failed_measured_completion_is_unknown_compute() -> None:
    seam = FakeSeam(usage_reply(), RuntimeError("the node did not answer"))

    assert await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES) is None
    # The warmup ran; the measured request failed and nothing was invented.
    assert len(seam.bodies) == 2


async def test_a_failed_warmup_is_unknown_compute() -> None:
    seam = FakeSeam(RuntimeError("model not loaded"), usage_reply())

    assert await calibrate_node(seam, model_weight_bytes=WEIGHT_BYTES) is None
    # A failed warmup cannot prove a warm measurement: no second request.
    assert len(seam.bodies) == 1


async def test_invalid_weight_bytes_are_a_caller_error() -> None:
    seam = FakeSeam(usage_reply(), usage_reply())
    with pytest.raises(ValueError):
        await calibrate_node(seam, model_weight_bytes=0)


# ---------------------------------------------------------------------------
# The usage read mirrors start.observed_from_reply
# ---------------------------------------------------------------------------


def test_timings_from_reply_reads_the_observe_usage_fields() -> None:
    timings = timings_from_reply(usage_reply())
    assert timings == CompletionTimings(
        decode_tok_per_s=500.0,
        prefill_tok_per_s=4000.0,
        ttft_seconds=0.02,
        prompt_tokens=10,
    )


def test_timings_from_reply_keeps_missing_optionals_unknown() -> None:
    timings = timings_from_reply(
        {"usage": {"generation_tokens_per_second": 500.0}}
    )
    assert timings == CompletionTimings(decode_tok_per_s=500.0)


def test_timings_from_reply_requires_decode() -> None:
    assert timings_from_reply({"usage": {"prompt_tokens_per_second": 4000.0}}) is None
    assert timings_from_reply({}) is None
    assert timings_from_reply(None) is None


@pytest.mark.parametrize("prompt_tokens", [0, -3, True, "ten", None])
def test_an_unusable_prompt_token_count_stays_unknown(prompt_tokens) -> None:
    timings = timings_from_reply(
        usage_reply(prompt_tokens=prompt_tokens)
    )
    assert timings is not None
    assert timings.prompt_tokens is None


# ---------------------------------------------------------------------------
# The weight-bytes helper and the body shape
# ---------------------------------------------------------------------------


def test_weight_bytes_helper_sums_fixed_layers_and_optional_mtp() -> None:
    layout = ModelLayout(
        source="test",
        fixed_weight_bytes=100,
        layer_weight_bytes=(10, 20, 30),
        mtp_weight_bytes=7,
    )
    assert calibration_weight_bytes(layout) == 160
    assert calibration_weight_bytes(layout, include_mtp=True) == 167


def test_weight_bytes_helper_matches_the_layout_total() -> None:
    layout = synthetic_model_layout(total_weight_bytes=WEIGHT_BYTES, layer_count=28)
    assert calibration_weight_bytes(layout) == layout.total_weight_bytes
    assert calibration_weight_bytes(layout) == WEIGHT_BYTES


def test_calibration_body_is_a_streaming_usage_request() -> None:
    body = calibration_body("dev-1.5b", prompt="hi", max_tokens=8)
    assert body == {
        "model": "dev-1.5b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
