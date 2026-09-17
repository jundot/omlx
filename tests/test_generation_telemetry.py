import pytest

from omlx.request import Request, SamplingParams


def _request() -> Request:
    return Request(
        request_id="chatcmpl-test",
        prompt=[1],
        sampling_params=SamplingParams(max_tokens=256),
    )


def test_recent_generation_tps_uses_trailing_hundred_emitted_tokens():
    request = _request()
    for token_id in range(150):
        request.append_output_token(token_id)
        request.record_generation_timestamp(token_id * 0.5)

    assert len(request.generation_timestamps) == 101
    assert request.generation_tps_recent == pytest.approx(2.0)
    assert request.generation_timestamps[0] == 24.5


def test_speculative_efficiency_tracks_total_and_output_window():
    request = _request()
    for position in range(1, 121):
        request.append_output_token(position)
        request.record_speculative_cycle(
            accepted=1 if position % 2 else 0,
            proposed=1,
        )

    assert request.speculative_efficiency == pytest.approx(0.5)
    assert request.speculative_efficiency_recent == pytest.approx(0.5)
    assert request.speculative_accepted_tokens == 60
    assert request.speculative_proposed_tokens == 120
    assert request.speculative_events[0][0] == 21


def test_non_speculative_request_reports_null_efficiency():
    request = _request()

    assert request.speculative_efficiency is None
    assert request.speculative_efficiency_recent is None
    assert request.speculative_active is False


def test_zero_proposal_cycle_marks_backend_active_without_fake_percentage():
    request = _request()
    request.append_output_token(1)
    request.record_speculative_cycle(accepted=0, proposed=0)

    assert request.speculative_active is True
    assert request.speculative_efficiency is None
    assert request.speculative_efficiency_recent is None
