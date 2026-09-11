# SPDX-License-Identifier: Apache-2.0
"""Streamed per-layer sensitivity: parity with the resident measurement.

The layer-streaming calibration sweep also measures quantization sensitivity
on the block it already holds resident: forward in float, temporarily
quantize-dequantize the block with the active predicate configuration,
forward again, score the relative MSE, restore the weights, and propagate the
float output to the next layer. That is exactly the walk
_measure_sensitivity_from_model does on a whole loaded model, so on a model
both sides can hold the per-layer scores must match, and the extra qdq
forwards must not move the imatrix statistic by a single bit.

The sensitivity boundary is deliberately separate from the imatrix boundary
(its own calibration corpus, sample count, and sequence length), which is
what these tests pin alongside the scores themselves.

Reuses the truncated bf16 MiniMax-M3 fixture from test_oq_stream_collect
(original layers 0..3: three dense layers plus the first MoE layer). The
module skips when neither the cached fixture nor the source checkpoint is
present. The imported fixtures shadow their own names as test parameters,
hence the F811 suppressions.
"""

import mlx.core as mx
import numpy as np
import omlx.oq as oq
import pytest
from omlx.oq import (
    _OQE_CALIB_DATASET,
    _collect_imatrix_streaming,
    _measure_sensitivity_from_model,
    quantize_oq_streaming,
)
from test_oq_stream_collect import (  # noqa: F401  (imported fixtures)
    CALIB_SEED,
    FIXTURE_DONE,
    KEEP_LAYERS,
    M3_DIR,
    MICRO_BATCH,
    NUM_SAMPLES,
    SEQ_LENGTH,
    _load_resident_model,
    _pinned_micro_batch,
    fixture_config,
    fixture_dir,
    resident_result,
    tokenizer,
)

pytestmark = pytest.mark.skipif(
    not M3_DIR.is_dir() and not FIXTURE_DONE.exists(),
    reason="neither MiniMax-M3-MXFP8 nor the cached truncated fixture is present",
)

OQ_LEVEL = 3
SENS_DATASET = "code_multilingual"
SENS_SAMPLES = 32
SENS_SEQ = 256


@pytest.fixture(scope="module")
def resident_sensitivity(fixture_dir, fixture_config, tokenizer):  # noqa: F811
    """Per-layer scores from the whole-model measurement, seeded draw."""
    model = _load_resident_model(fixture_dir)
    try:
        # The subsample permutation inside _load_calibration_data is
        # unseeded; seeding right before the measurement pins the draw to
        # the one the streamed sweep makes with sensitivity_calib_seed.
        mx.random.seed(CALIB_SEED)
        scores = _measure_sensitivity_from_model(
            model,
            tokenizer,
            fixture_config,
            OQ_LEVEL,
            calib_dataset=SENS_DATASET,
            num_samples=SENS_SAMPLES,
            seq_length=SENS_SEQ,
        )
    finally:
        del model
        mx.synchronize()
        mx.clear_cache()
    return scores


@pytest.fixture(scope="module")
def fused_result(fixture_dir, fixture_config, tokenizer):  # noqa: F811
    """Entries and metadata from one fused imatrix + sensitivity sweep."""
    with _pinned_micro_batch(MICRO_BATCH):
        entries, metadata = _collect_imatrix_streaming(
            fixture_dir,
            tokenizer,
            fixture_config,
            calib_dataset=_OQE_CALIB_DATASET,
            num_samples=NUM_SAMPLES,
            seq_length=SEQ_LENGTH,
            calib_seed=CALIB_SEED,
            measure_sensitivity=True,
            sensitivity_oq_level=OQ_LEVEL,
            sensitivity_calib_dataset=SENS_DATASET,
            sensitivity_num_samples=SENS_SAMPLES,
            sensitivity_seq_length=SENS_SEQ,
            sensitivity_calib_seed=CALIB_SEED,
        )
    mx.synchronize()
    mx.clear_cache()
    return entries, metadata


def test_streamed_sensitivity_matches_resident(fused_result, resident_sensitivity):
    """THE GATE: fused-sweep sensitivity must equal the resident scores."""
    _, metadata = fused_result
    streamed = metadata["sensitivity_map"]

    expected_layers = set(range(KEEP_LAYERS))
    assert set(resident_sensitivity) == expected_layers, (
        f"resident measurement is missing layers: {resident_sensitivity}"
    )
    assert set(streamed) == expected_layers, (
        f"streamed measurement is missing layers: {streamed}"
    )
    # A degenerate all-zero map would satisfy equality while carrying no
    # ranking signal; qdq at oQ3 must produce a real error on every layer.
    assert all(v > 0.0 for v in streamed.values()), streamed

    diffs = {i: abs(streamed[i] - resident_sensitivity[i]) for i in expected_layers}
    max_diff = max(diffs.values())
    exact = all(streamed[i] == resident_sensitivity[i] for i in expected_layers)
    print(
        f"\nsensitivity parity: {'bitwise' if exact else 'approximate'}, "
        f"max abs diff {max_diff:.3e} "
        f"(resident={resident_sensitivity}, streamed={streamed})"
    )
    if not exact:
        for i in sorted(expected_layers):
            assert diffs[i] < 1e-6, (
                f"layer {i}: streamed {streamed[i]!r} vs resident "
                f"{resident_sensitivity[i]!r}, abs diff {diffs[i]:.3e}"
            )


def test_imatrix_gate_still_bitwise(resident_result, fused_result):  # noqa: F811
    """The fused qdq forwards must not perturb the imatrix statistic.

    Same comparison as test_streaming_matches_resident_imatrix, now with
    the sensitivity pass riding the sweep: every entry must still equal
    the resident collector bit for bit, and the metadata contract must
    only grow by the sensitivity map.
    """
    res_entries, res_meta = resident_result
    fused_entries, fused_meta = fused_result

    assert fused_meta["load_kind"] == "streaming"
    assert set(fused_meta) == set(res_meta) | {"load_kind", "sensitivity_map"}
    assert fused_meta["processed_samples"] == res_meta["processed_samples"]

    assert set(fused_entries) == set(res_entries)
    for name in sorted(res_entries):
        res, fused = res_entries[name], fused_entries[name]
        assert np.array_equal(res.counts, fused.counts), (
            f"{name}: routing counts diverge with fused sensitivity"
        )
        assert np.array_equal(res.in_sum2, fused.in_sum2), (
            f"{name}: in_sum2 diverges with fused sensitivity"
        )


def test_streaming_build_uses_no_proxy_for_sensitivity(
    fixture_dir,  # noqa: F811
    tmp_path,
    monkeypatch,
):
    """stream_calibration=True must never touch the proxy or a whole model.

    Both wiring branches are exercised: a fresh collection measures
    sensitivity inside the fused sweep, and a second run that hits the
    imatrix cache falls back to the standalone streamed sweep. Every
    proxy-building and whole-model-loading entry point is patched to blow
    up, so a regression on either branch cannot pass silently. The build
    is aborted right after the sensitivity map lands in the config; the
    quantization loop itself is out of scope here.
    """
    for entry_point in (
        "_build_proxy_for_sensitivity",
        "_build_streaming_proxy_for_sensitivity",
        "_measure_sensitivity_from_quantized_model",
        "_measure_sensitivity",
        "_collect_imatrix",
    ):

        def _forbidden(*args, _name=entry_point, **kwargs):
            raise AssertionError(f"{_name} must not run when calibration streams")

        monkeypatch.setattr(oq, entry_point, _forbidden)

    standalone_calls = []
    orig_standalone = oq._measure_sensitivity_streaming

    def spy_standalone(*args, **kwargs):
        standalone_calls.append(kwargs)
        return orig_standalone(*args, **kwargs)

    monkeypatch.setattr(oq, "_measure_sensitivity_streaming", spy_standalone)

    captured = {}
    orig_non_quantizable = oq._build_non_quantizable_set

    def spy_non_quantizable(config):
        captured["config"] = config
        return orig_non_quantizable(config)

    monkeypatch.setattr(oq, "_build_non_quantizable_set", spy_non_quantizable)

    class _StopBuildError(Exception):
        pass

    def stop(*args, **kwargs):
        raise _StopBuildError

    monkeypatch.setattr(oq, "_collect_named_weight_shapes_from_weights", stop)

    cache_path = tmp_path / "imatrix.npz"

    def run(out_name: str) -> dict:
        captured.clear()
        with pytest.raises(_StopBuildError):
            quantize_oq_streaming(
                str(fixture_dir),
                str(tmp_path / out_name),
                OQ_LEVEL,
                enhanced=True,
                stream_calibration=True,
                imatrix_cache_path=str(cache_path),
                imatrix_num_samples=NUM_SAMPLES,
                imatrix_seq_length=SEQ_LENGTH,
            )
        sensitivity_map = captured["config"].get("_oq_sensitivity_map")
        assert sensitivity_map, "no sensitivity map reached the config"
        assert set(sensitivity_map) == {str(i) for i in range(KEEP_LAYERS)}
        assert all(v > 0.0 for v in sensitivity_map.values())
        return sensitivity_map

    # Fresh collection: sensitivity comes out of the fused sweep.
    run("out-fused")
    assert standalone_calls == [], (
        "a fresh streamed collection must fuse sensitivity, not run a second sweep"
    )
    assert cache_path.exists(), "imatrix cache was not written by the first run"

    # Cache hit: the fused sweep never runs, the standalone streamed
    # measurement covers sensitivity instead.
    run("out-cached")
    assert len(standalone_calls) == 1, (
        "an imatrix cache hit must measure sensitivity with the standalone "
        "streamed sweep"
    )
