"""Pure arithmetic gates for the DS4 indexer taper profiler."""

import pytest

from benchmarks.bench_ds4_indexer_threshold import build_taper_attribution


def test_taper_attribution_converts_rates_and_exposes_slope_share():
    measurements = [
        {"context_tokens": 30_000, "parallel_layers_ms": 80.0},
        {"context_tokens": 100_000, "parallel_layers_ms": 256.0},
        {"context_tokens": 250_000, "parallel_layers_ms": 636.0},
    ]
    report = build_taper_attribution(
        measurements,
        (870.0, 784.0, 619.0),
        logical_chunk_tokens=1024,
    )

    assert [point["context_tokens"] for point in report["points"]] == [
        30_000,
        100_000,
        250_000,
    ]
    assert report["points"][0]["observed_chunk_ms"] == pytest.approx(
        1024_000 / 870
    )
    assert report["points"][2]["indexer_wall_fraction"] == pytest.approx(
        636 / (1024_000 / 619)
    )
    assert report["intervals"][0]["indexer_share_of_wall_growth"] > 1.0
    assert report["intervals"][1]["indexer_share_of_wall_growth"] > 1.0


@pytest.mark.parametrize(
    "rates,chunk",
    [((870.0,), 1024), ((870.0, 0.0), 1024), ((870.0, 784.0), 0)],
)
def test_taper_attribution_rejects_misaligned_or_nonpositive_inputs(rates, chunk):
    measurements = [
        {"context_tokens": 30_000, "parallel_layers_ms": 80.0},
        {"context_tokens": 100_000, "parallel_layers_ms": 256.0},
    ]
    with pytest.raises(ValueError):
        build_taper_attribution(
            measurements,
            rates,
            logical_chunk_tokens=chunk,
        )
