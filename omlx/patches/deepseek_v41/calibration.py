# SPDX-License-Identifier: MIT
"""Collect oQe statistics through V4.1's complete cache and Engram forward."""

import copy

import mlx.core as mx


def collect_imatrix(
    model,
    tokenizer,
    *,
    num_samples=32,
    seq_length=512,
    progress=None,
    initial=None,
    calib_dataset="oqe_code_multilingual",
    adaptive=False,
):
    from ...oq import (
        _OQE_MAX_ADAPTIVE_SAMPLES,
        _OQE_MAX_SAMPLE_MULTIPLIER,
        OQImatrixCollector,
        _imatrix_expert_coverage_stats,
        _imatrix_expert_coverage_sufficient,
        _load_calibration_data,
    )

    maximum = num_samples
    if adaptive:
        maximum = max(
            num_samples,
            min(num_samples * _OQE_MAX_SAMPLE_MULTIPLIER, _OQE_MAX_ADAPTIVE_SAMPLES),
        )
    samples = _load_calibration_data(tokenizer, calib_dataset, maximum, seq_length)
    if samples is None or samples.shape[0] == 0:
        raise ValueError("V4.1 oQe requires nonempty calibration data")
    collector = OQImatrixCollector()
    start = 0
    if initial is not None:
        expected = {
            "model_type": "deepseek_v41",
            "dataset": calib_dataset,
            "seq_length": seq_length,
        }
        if any(initial.metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("V4.1 calibration resume metadata does not match")
        start = int(initial.metadata.get("processed_samples", -1))
        if not 0 <= start < len(samples):
            raise ValueError("V4.1 calibration resume has no remaining samples")
        collector.entries = copy.deepcopy(initial.entries)
    core = model.language_model
    installed = collector.install(model)
    processed = start
    rounds = []
    try:
        for index, tokens in enumerate(samples[start:], start=start):
            cache = core.make_cache()
            logits = core(tokens[None], cache=cache)
            mx.eval(logits[:, -1], [entry.state for entry in cache])
            if progress is not None:
                progress(index + 1, len(samples), collector.entries)
            del logits, cache
            processed = index + 1
            if adaptive and processed >= num_samples and processed % num_samples == 0:
                coverage = _imatrix_expert_coverage_stats(collector.entries)
                sufficient = _imatrix_expert_coverage_sufficient(
                    coverage, require_expert_counts=collector.switch_capture_modules > 0
                )
                rounds.append(dict(processed_samples=processed, coverage=coverage))
                if sufficient:
                    break
        if not collector.entries:
            raise ValueError("V4.1 oQe captured no projection inputs")
        metadata = {
            "model_type": "deepseek_v41",
            "dataset": calib_dataset,
            "processed_samples": processed,
            "requested_samples": num_samples,
            "adaptive_max_samples": len(samples),
            "rounds": rounds,
            "seq_length": seq_length,
            "capture_modules": installed,
            "captured_modules": len(collector.entries),
            "expert_coverage": _imatrix_expert_coverage_stats(collector.entries),
            "forward": "full_model_with_request_local_cache_and_engram",
        }
        return collector.entries, metadata
    finally:
        collector.restore(model)


def collect_checkpoint_imatrix(
    model_path,
    *,
    calib_dataset,
    num_samples,
    seq_length,
    progress_callback=None,
    progress_start=13.0,
    progress_end=18.0,
):
    """Adapter for the official oQ collector, including cancellation cleanup."""
    from ...oq import _emit_progress, _imatrix_expert_coverage_sufficient
    from .loading import load

    model, processor = load(model_path, engram_ssd_offload=True)
    try:

        def progress(done, total, entries):
            _emit_progress(
                progress_callback,
                "imatrix",
                progress_start + (progress_end - progress_start) * done / total,
                f"V4.1 calibration {done}/{total}",
                {"entry_count": len(entries), "processed_samples": done},
            )

        entries, metadata = collect_imatrix(
            model,
            processor.tokenizer,
            calib_dataset=calib_dataset,
            num_samples=num_samples,
            seq_length=seq_length,
            progress=progress,
            adaptive=True,
        )
        coverage = metadata["expert_coverage"]
        sufficient = _imatrix_expert_coverage_sufficient(
            coverage, require_expert_counts=True
        )
        metadata.update(
            coverage=coverage,
            requires_expert_counts=True,
            coverage_sufficient=sufficient,
            collection_sufficient=sufficient,
            uncalibrated_policy="preserve_source_precision",
        )
        return entries, metadata
    finally:
        model.close()
        del model, processor
        mx.synchronize()
        mx.clear_cache()
