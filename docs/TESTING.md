# QSA reservation tests

Run `python -m pytest -q tests/test_qwen4_qsa_reservation_integration.py tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

# Prefill memory accounting tests

Run `python -m pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `python -m pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Prefill reclaim and restored-context regressions

The prefill accounting suites also check repeated versus contiguous footprint releases, partial repayment on skipped chunks, and growth between chunks. A one-off spike followed by a normal sample must retain main's EWMA recovery; repeated large allocations must remain visible to the scheduler. These are controlled observations, not actual OOM reproductions.

Run `python -m pytest -q tests/test_scheduler.py tests/test_qwen4_qsa_reservation_integration.py` to check that restored context reaches chunk pricing even when boundary snapshots are disabled, for both external and chunked prefill.

GLM projection preparation is covered by `tests/test_mlx_vlm_glm5_next_compat.py`: lazy versus load-time preparation, floating-point and quantized prefill/decode parity, idempotence, and mixed-quantization fallback. `tests/test_memory_monitor.py` checks that Qwen4 short-query estimates follow the runtime threshold, including environment overrides.

Generic context-priority partials stop updating the representative EWMA and last representative sample once a full step has been observed. The largest pending partial allocation is kept separately: larger candidates never multiply it by their width, while smaller candidates retain the existing proportional reduction. Floor observations and reclaim accounting remain active. A full sample retires this bound only when its per-token rate covers it at every width; cheap partial/full reuse and process-wide releases cannot erase it. Routes and model resets remain independent.

`TestPartialCostIsolation` covers partial-spike progress, cheap partials, repeated large allocations, transfer to a covering representative sample, and avoiding a new global floor. Unseeded generic partials retain the existing calibration: these tests do not claim to solve first-short-request poisoning or all long-context GLM failures. A pending partial bound can still reject work when the observed cost itself does not fit.
