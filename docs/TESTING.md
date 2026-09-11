# Claude launcher tests

Run `python -m pytest -q tests/test_cli.py tests/test_integrations.py` to check model selection, saved tier precedence, and the final Claude Code environment. Mixed-context cases verify that the selected initial model is passed through `ANTHROPIC_MODEL` and that the process-wide context and auto-compact limits use the smallest known context window across the initial model and configured tiers. Larger models consequently compact at that shared limit; models without reported limits cannot contribute a limit.

# ModernBERT embedding tests

Run `python -m pytest -q tests/test_modernbert_attention.py tests/test_embedding.py tests/test_mlx_embeddings_compat.py` to check finite padded attention, single-input equivalence, local-window masking, and embedding integration. The attention regression covers fp16, bf16, and fp32 at lengths around the affected SDPA tile boundaries.

# QSA reservation tests

Run `python -m pytest -q tests/test_qwen4_qsa_reservation_integration.py tests/test_qwen4_qsa_reserved_capacity.py` to check QSA capacity reservations.

The integration tests cover restored-prefix lengths with boundary snapshots enabled and disabled, the first allocation after cache restoration, and prefill/decode output equivalence using a small Qwen4 model.

Related regression suites are `test_qwen4_qsa_incremental_cache.py`, `test_qwen4_qsa_decode_gather.py`, and `test_prefill_oom_graceful.py`.

# Prefill memory accounting tests

Run `python -m pytest -q tests/test_prefill_transient_tracker.py tests/test_prefill_oom_graceful.py` to check retained versus reclaimed overhead, configured chunk sizes, and abort-cap enforcement. The loop tests run a small initialized MLX model with controlled footprint readings through external and chunked prefill; they do not load a checkpoint.

# Prefix cache completion tests

Run `python -m pytest -q tests/test_scheduler.py tests/test_scheduler_boundary_completion.py tests/test_prefix_cache_gdn_split.py` to check cache-freshness admission and completed boundary recovery. The completion tests use a small initialized Qwen3.5 hybrid model and the real BatchGenerator, then compare restored-prefix logits with a fresh forward pass. They cover embedded snapshots, GDN sidecars, off-boundary completion, and unknown or inconsistent cache positions.

# Cluster join recovery tests

Run `python -m pytest -q tests/test_cluster_pairing_session.py tests/test_cluster_pairing.py tests/test_cluster_ui_integration.py tests/ui/test_cluster_v2_wizard.py` to check joining, cancellation, and approval. Session tests recreate a manager with the same base path to verify that the original code and cancellation proof survive a restart. They also cover offline cancellation, switching peers while cleanup is pending, rejected requests versus lost responses, and storage failures. Wizard tests check that delayed responses cannot restore a cancelled join and that pending cleanup leaves new pairing controls available.

For a two-Mac smoke test, start isolated servers with separate base paths. Request a join, restart only the joining server, then cancel and retry; the coordinator must remove the original pending request. Repeat with the coordinator offline: cancellation must return `state: idle` with `cleanup_pending: true`, and joining a different reachable Mac must work. Restore the coordinator and poll the join endpoint to verify cleanup. Keep existing approval/cancel race and token-ownership tests in the run; never relax the coordinator's token check to make a stale request disappear.

# DeepSeek V4.1 offline tests

Run `python -m pytest -q tests/test_deepseek_v41.py` for the text/vision port. Synthetic weights and small recorded official outputs cover prefill/decode, DSpark, Engram and vision numerics without a checkpoint download or vendored reference implementation. See `tests/fixtures/deepseek_v41_expected.md` for provenance and reference corrections. PyTorch is optional for the independent FP8/FP4 arithmetic checks. Other cases cover BatchGenerator admission, cache restoration, DSML and VLM engine execution; usage and limits are in `docs/deepseek-v41.md`.

FP8 activation boundary and layout checks: `python -m pytest tests/test_deepseek_v41_activation.py -q`. These compare fused FP8 and weighted SwiGLU kernels against reference arithmetic across rounding ties, scale transitions, clipping, intermediate casts and floating-point dtypes.

MoE activation reuse and per-expert reference checks: `python -m pytest tests/test_deepseek_v41_moe.py -q`. These cover sorted prefill, decode, independent routed/shared activation policies, and repeated outputs with small quantized weights.

DeepSeek V4.1 Metal arithmetic and sparse-addressing checks: `python -m pytest tests/test_deepseek_v41_kernels.py -q` (no checkpoint required).

Engram storage and prefetch lifecycle: `python -m pytest tests/test_deepseek_v41_offload.py -q`. Modal state, forced-toggle behavior, and save payload: `node --test tests/deepseek_v41_offload_ui.test.cjs`. Both use synthetic fixtures and require no checkpoint download.
