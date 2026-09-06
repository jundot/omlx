# SPDX-License-Identifier: Apache-2.0
"""Benchmark execution logic for oMLX admin panel.

Provides single-request and continuous-batching benchmarks with
real-time progress reporting via SSE events.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator

from ..model_discovery import model_display_name
from ..utils.proc_memory import get_lifetime_max_phys_footprint
from ..utils.system_sampler import SystemSampler
from .external_api import ExternalAPIClient, ExternalEndpointConfig

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

logger = logging.getLogger(__name__)

# Module-level storage for active benchmark runs
_benchmark_runs: dict[str, "BenchmarkRun"] = {}

# Valid prompt lengths for single request tests
VALID_PROMPT_LENGTHS = [1024, 4096, 8192, 16384, 32768, 65536, 131072, 200000]

# Valid batch sizes for continuous batching tests
VALID_BATCH_SIZES = [2, 4, 8]


class BenchmarkContextProfile(StrEnum):
    """Stable identifiers for the bundled throughput-benchmark corpora."""

    CODE_PYTHON = "code_python"
    CODE_MIXED = "code_mixed"
    NOVEL_KO = "novel_ko"
    NOVEL_EN = "novel_en"
    NOVEL_JA = "novel_ja"


class BenchmarkWarmupMode(StrEnum):
    """How much of the local inference path to compile before measurement."""

    QUICK = "quick"
    ANE_2048 = "ane_2048"


def _warmup_prompt_tokens(mode: BenchmarkWarmupMode | str) -> int:
    """Return prompt length needed to execute the selected prefill warm-up.

    Generation reserves the final prompt token for the first decode step, so a
    2,048-token prefill requires a 2,049-token prompt.  Keeping that detail in
    one helper prevents the UI's "2,048-token block" option from accidentally
    warming only the 2,047-token GPU fallback shape.
    """
    normalized = BenchmarkWarmupMode(mode)
    return 2049 if normalized is BenchmarkWarmupMode.ANE_2048 else 32


@dataclass(frozen=True)
class BenchmarkCorpusSpec:
    """Metadata needed to build local and tokenizer-less prompts."""

    filename: str
    label: str
    chars_per_token: float
    start_marker: str | None = None


BENCHMARK_CONTEXT_PROFILES: dict[BenchmarkContextProfile, BenchmarkCorpusSpec] = {
    BenchmarkContextProfile.CODE_PYTHON: BenchmarkCorpusSpec(
        "code_python.txt", "Code (Python)", 4.0
    ),
    BenchmarkContextProfile.CODE_MIXED: BenchmarkCorpusSpec(
        "code_mixed.txt", "Code (Mixed)", 3.5
    ),
    BenchmarkContextProfile.NOVEL_KO: BenchmarkCorpusSpec(
        "novel_ko.txt", "Novel (Korean)", 1.35
    ),
    BenchmarkContextProfile.NOVEL_EN: BenchmarkCorpusSpec(
        "novel_en.txt", "Novel (English)", 4.0, "Call me Ishmael."
    ),
    BenchmarkContextProfile.NOVEL_JA: BenchmarkCorpusSpec(
        "novel_ja.txt", "Novel (Japanese)", 1.6
    ),
}


class BenchmarkRequest(BaseModel):
    """Request model for starting a benchmark."""

    model_id: str
    prompt_lengths: list[int]
    generation_length: int = 128
    batch_sizes: list[int] = []
    context_profile: BenchmarkContextProfile = BenchmarkContextProfile.CODE_PYTHON
    warmup_mode: BenchmarkWarmupMode = BenchmarkWarmupMode.QUICK
    align_prompt_to_ane: bool = False
    force_lm_engine: bool = False
    # When set, the benchmark runs against a remote OpenAI-compatible
    # endpoint instead of a local engine and model_id is the remote
    # model name (not validated against the local catalog).
    external: Optional[ExternalEndpointConfig] = None

    @field_validator("prompt_lengths")
    @classmethod
    def validate_prompt_lengths(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("At least one prompt length is required")
        for pl in v:
            if pl not in VALID_PROMPT_LENGTHS:
                raise ValueError(
                    f"Invalid prompt length {pl}. Must be one of {VALID_PROMPT_LENGTHS}"
                )
        return sorted(v)

    @field_validator("batch_sizes")
    @classmethod
    def validate_batch_sizes(cls, v: list[int]) -> list[int]:
        for bs in v:
            if bs not in VALID_BATCH_SIZES:
                raise ValueError(
                    f"Invalid batch size {bs}. Must be one of {VALID_BATCH_SIZES}"
                )
        return sorted(v)


def _single_prompt_lengths(request: BenchmarkRequest) -> list[int]:
    """Return the actual prompt lengths used by single-request trials.

    Generation reserves the last prompt token for first-token logits. Adding
    one to the standard PP sizes therefore turns PP4096 into exactly 4096
    prefill rows. Results intentionally report the actual length (PP4097), so
    aligned local diagnostics cannot be mistaken for standard leaderboard
    measurements.
    """
    adjustment = 1 if request.align_prompt_to_ane else 0
    return [length + adjustment for length in request.prompt_lengths]


@dataclass
class BenchmarkRun:
    """Tracks the state of a running benchmark.

    SSE delivery model: events are appended to `events` (append-only
    log) under `cond`. Subscribers replay `events` from offset 0 then
    wait on `cond` for new entries. `terminal` is set once the final
    event (`upload_done` / `error`) has been published so subscribers
    know to close their stream rather than wait for a follow-up.
    """

    bench_id: str
    request: BenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    events: list[dict] = field(default_factory=list)
    cond: asyncio.Condition = field(default_factory=asyncio.Condition)
    terminal: bool = False
    task: Optional[asyncio.Task] = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    # Acceleration features active when the benchmark started. Results are
    # uploaded either way; the flags ride along so the leaderboard can mark
    # and filter them instead of silently mixing them in.
    experimental_features: list[str] = field(default_factory=list)
    # Same snapshot in the upload payload's shape: [{key, label, detail?}].
    feature_flags: list[dict] = field(default_factory=list)
    # Performance-relevant subset of the model's settings at run start.
    model_settings_snapshot: Optional[dict] = None
    # Host telemetry sampler, running for the duration of the tests.
    sampler: Optional[Any] = None
    # Lifetime footprint high-water mark before the tests began, so the run's
    # own peak can be told apart from a larger one set earlier in the process.
    lifetime_footprint_at_start: int = 0
    # Mirror of the upload SSE events so REST consumers (e.g. native Swift
    # app polling /results) can render leaderboard status without opening
    # the stream. Phases: "idle" → "uploading" → "done" | "skipped". The
    # browser HTML still consumes the SSE stream directly; this is purely
    # additive state that lives alongside it.
    upload_state: dict = field(
        default_factory=lambda: {
            "phase": "idle",
            "results": [],  # per-context-length: {context_length, id?, url?, duplicate?, error?}
            "total": 0,
            "success_count": 0,
            "failed_count": 0,
            "owner_hash": None,  # display hash, populated on upload_done
            "skipped_reason": None,  # only "external_endpoint" reaches this now
            # Always empty. Kept because BenchDTO.swift declares it non-optional,
            # so dropping the key would fail decoding on every app build that has
            # not been updated — which turns into a per-second error loop while the
            # results poller runs.
            "skipped_features": [],
            "feature_flags": [],  # [{key, label, detail?}]
        }
    )


# Event types that close the SSE stream for a bench run. `done` is NOT
# terminal — it marks "tests finished, upload starting"; the real end of
# stream is `upload_done` (or `error`). `upload_skipped` is the external
# endpoint's last event: without it here, subscribers to an external run would
# wait for an `upload_done` that never comes.
_BENCH_TERMINAL_TYPES = frozenset({"upload_done", "upload_skipped", "error"})


@dataclass(frozen=True)
class _FeatureFlagSpec:
    """One acceleration toggle, in both the legacy and upload projections."""

    attr: str
    legacy: str
    key: str
    label: str
    detail_attr: Optional[str] = None
    detail_key_fmt: Optional[str] = None
    detail_label_fmt: Optional[str] = None


# `mtp_enabled` is surfaced as "Lightning MTP" everywhere in the UI, so the
# upload key follows the user-facing name rather than the settings field.
_FEATURE_FLAG_SPECS = (
    _FeatureFlagSpec("dflash_enabled", "dflash", "dflash", "DFlash"),
    _FeatureFlagSpec(
        "specprefill_enabled", "specprefill", "specprefill", "SpecPrefill"
    ),
    _FeatureFlagSpec(
        "turboquant_kv_enabled",
        "turboquant",
        "turboquant_kv",
        "TurboQuant KV",
        detail_attr="turboquant_kv_bits",
        detail_key_fmt="_{}bit",
        detail_label_fmt=" {}-bit",
    ),
    _FeatureFlagSpec("mtp_enabled", "mtp", "lightning_mtp", "Lightning MTP"),
    _FeatureFlagSpec("vlm_mtp_enabled", "vlm_mtp", "vlm_mtp", "VLM MTP"),
    _FeatureFlagSpec(
        "qwen35_ane_prefill_enabled",
        "qwen35_ane_prefill",
        "qwen35_ane_prefill",
        "Qwen ANE Prefill",
    ),
)


def _sample_window(run: "BenchmarkRun", window_start: float) -> Optional[dict]:
    """Aggregate host telemetry for the interval a single test occupied."""
    if run.sampler is None:
        return None
    try:
        return run.sampler.window(window_start, time.monotonic())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Benchmark: system metrics unavailable: {e}")
        return None


def _detect_experimental_features(model_settings: Any) -> list[str]:
    """Return benchmark-skewing model features enabled in settings."""
    return [
        spec.legacy
        for spec in _FEATURE_FLAG_SPECS
        if getattr(model_settings, spec.attr, False)
    ]


def _format_bits(value: Any) -> Optional[str]:
    """Render a bit-width for display, dropping a trailing .0 (4.0 -> "4")."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number:g}"


def _derive_feature_flags(model_settings: Any) -> list[dict]:
    """Build the upload projection of the active acceleration features.

    Objects rather than bare keys: the app and omlx.ai ship independently, so
    carrying the display label means a newly added feature renders correctly on
    the site from day one instead of showing a raw snake_case key until the
    next site deploy. Only active features are included — the site derives
    "this run was accelerated" from the list being non-empty.
    """
    flags: list[dict] = []
    for spec in _FEATURE_FLAG_SPECS:
        if not getattr(model_settings, spec.attr, False):
            continue
        key, label = spec.key, spec.label
        if spec.detail_attr:
            bits = _format_bits(getattr(model_settings, spec.detail_attr, None))
            if bits:
                # Keys must stay [a-z0-9_], so 2.5 becomes 2_5.
                key += spec.detail_key_fmt.format(bits.replace(".", "_"))
                label += spec.detail_label_fmt.format(bits)
        flags.append({"key": key, "label": label})
    return flags


# Performance-relevant settings only, as an allowlist rather than a denylist:
# ModelSettings gains fields regularly, and a denylist would ship every future
# addition to a public endpoint by default.
#
# Excluded on purpose: display_name / description / model_alias (user-authored
# free text), is_pinned / is_default / is_hidden / is_favorite /
# active_profile_name / ttl_seconds (local organization), the guided_grammar
# body (unbounded; the boolean is kept), chat_template_kwargs and
# forced_ct_kwargs (arbitrary user dicts), and trust_remote_code (security
# posture, not performance). The *_draft_model fields are included but reduced
# to a basename — the drafter's identity explains an MTP/DFlash result, while
# the full path would leak the local filesystem layout and the OS username.
_UPLOADED_SETTING_FIELDS = (
    "max_context_window",
    "max_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "force_sampling",
    "enable_thinking",
    "thinking_budget_enabled",
    "thinking_budget_tokens",
    "reasoning_parser",
    "guided_grammar_enabled",
    "model_type_override",
    "index_cache_freq",
    "turboquant_kv_enabled",
    "turboquant_kv_bits",
    "turboquant_skip_last",
    "specprefill_enabled",
    "specprefill_draft_model",
    "specprefill_keep_pct",
    "specprefill_threshold",
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_draft_quant_enabled",
    "dflash_draft_quant_weight_bits",
    "dflash_draft_quant_activation_bits",
    "dflash_draft_quant_group_size",
    "dflash_max_ctx",
    "dflash_in_memory_cache",
    "dflash_in_memory_cache_max_entries",
    "dflash_ssd_cache",
    "dflash_draft_window_size",
    "dflash_draft_sink_size",
    "dflash_block_size",
    "dflash_verify_mode",
    "mtp_enabled",
    "mtp_num_draft_tokens",
    "vlm_mtp_enabled",
    "vlm_mtp_draft_model",
    "vlm_mtp_draft_block_size",
    "qwen35_ane_prefill_enabled",
    "qwen35_ane_prefill_sequence_length",
    "qwen35_ane_prefill_tail_padding_min_tokens",
    "qwen35_ane_prefill_fraction",
    "qwen35_ane_prefill_fused_down",
    "qwen35_ane_prefill_max_layers",
    "qwen35_ane_prefill_dual_ane",
    "qwen35_ane_prefill_gdn",
    "qwen35_ane_prefill_gdn_fraction",
    "qwen35_ane_prefill_gdn_max_layers",
    "qwen35_ane_prefill_cpu_enabled",
    "qwen35_ane_prefill_cpu_fraction",
    "qwen35_ane_prefill_cpu_down_fraction",
    "qwen35_ane_prefill_cpu_gdn_fraction",
    "qwen35_ane_prefill_cpu_threads",
    "qwen35_ane_prefill_cpu_shared_resource",
)

_PATH_VALUED_SETTING_FIELDS = frozenset(
    {
        "specprefill_draft_model",
        "dflash_draft_model",
        "vlm_mtp_draft_model",
    }
)

_MAX_UPLOADED_SETTINGS_BYTES = 4096


def _filter_uploaded_settings(model_settings: Any) -> Optional[dict]:
    """Project model settings onto the uploadable allowlist."""
    to_dict = getattr(model_settings, "to_dict", None)
    if not callable(to_dict):
        return None
    try:
        raw = to_dict()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Benchmark: failed to serialize model settings: {e}")
        return None

    filtered: dict = {}
    for key in _UPLOADED_SETTING_FIELDS:
        if key not in raw:
            continue
        value = raw[key]
        if key in _PATH_VALUED_SETTING_FIELDS and isinstance(value, str):
            value = os.path.basename(value.rstrip("/")) or value
        filtered[key] = value

    if len(json.dumps(filtered, separators=(",", ":"))) > _MAX_UPLOADED_SETTINGS_BYTES:
        logger.warning(
            "Benchmark: model settings snapshot exceeded "
            f"{_MAX_UPLOADED_SETTINGS_BYTES} bytes, uploading accelerator flags only"
        )
        filtered = {
            spec.attr: filtered[spec.attr]
            for spec in _FEATURE_FLAG_SPECS
            if spec.attr in filtered
        }
    return filtered


def _with_benchmark_context(
    context_profile: BenchmarkContextProfile | str,
    model_settings: dict | None,
) -> dict:
    """Prepend the benchmark context to the uploaded settings snapshot."""
    settings = dict(model_settings or {})
    settings.pop("benchmark_context", None)
    return {
        "benchmark_context": benchmark_context_label(context_profile),
        **settings,
    }


def get_run(bench_id: str) -> Optional[BenchmarkRun]:
    """Get a benchmark run by ID."""
    return _benchmark_runs.get(bench_id)


def get_active_run() -> Optional[BenchmarkRun]:
    """Return the currently-running throughput benchmark, if any.

    Discovery surface for clients that need to attach to an in-progress
    run without knowing the bench_id upfront (page refresh, second tab).
    Returns the first run with status == "running"; throughput benches
    are 1-at-a-time so there's never more than one.
    """
    for run in _benchmark_runs.values():
        if run.status == "running":
            return run
    return None


def create_run(request: BenchmarkRequest) -> BenchmarkRun:
    """Create and register a new benchmark run."""
    bench_id = f"bench-{uuid.uuid4().hex[:12]}"
    run = BenchmarkRun(bench_id=bench_id, request=request)
    _benchmark_runs[bench_id] = run
    return run


def cleanup_old_runs(max_runs: int = 10) -> None:
    """Remove old completed runs to prevent memory leaks."""
    completed = [
        (bid, r)
        for bid, r in _benchmark_runs.items()
        if r.status in ("completed", "cancelled", "error")
    ]
    if len(completed) > max_runs:
        for bid, _ in completed[:-max_runs]:
            del _benchmark_runs[bid]


# Bundled corpora for benchmark prompts. They contain long-form code or prose,
# never a short filler sentence. A whole corpus may repeat for a tokenizer that
# compresses it unusually well, but the repeated unit is hundreds of thousands
# of natural tokens rather than a predictable one-line loop.
_BENCH_CORPUS_DIR = Path(__file__).parent / "bench_corpora"
_PROMPT_BUILD_MAX_ATTEMPTS = 16


def benchmark_context_label(profile: BenchmarkContextProfile | str) -> str:
    """Return the user-facing label for a benchmark context profile."""
    normalized = BenchmarkContextProfile(profile)
    return BENCHMARK_CONTEXT_PROFILES[normalized].label


@lru_cache(maxsize=len(BENCHMARK_CONTEXT_PROFILES))
def _load_bench_corpus(
    context_profile: (
        BenchmarkContextProfile | str
    ) = BenchmarkContextProfile.CODE_PYTHON,
) -> str:
    profile = BenchmarkContextProfile(context_profile)
    spec = BENCHMARK_CONTEXT_PROFILES[profile]
    path = _BENCH_CORPUS_DIR / spec.filename
    corpus = path.read_text(encoding="utf-8")
    if spec.start_marker:
        start = corpus.find(spec.start_marker)
        if start < 0:
            raise RuntimeError(
                f"Benchmark corpus at {path} is missing the content start marker"
            )
        corpus = corpus[start:]
    if not corpus:
        raise RuntimeError(f"Benchmark corpus at {path} is empty")
    return corpus


def _generate_prompt(
    tokenizer: Any,
    target_tokens: int,
    context_profile: (
        BenchmarkContextProfile | str
    ) = BenchmarkContextProfile.CODE_PYTHON,
) -> list[int]:
    """Generate exactly ``target_tokens`` benchmark-corpus token IDs.

    Uses a unique UUID prefix to prevent SSD cache hits from previous sessions.
    The prefix and corpus are encoded together so tokenizer boundary merges and
    special-token insertion happen exactly once. The token IDs are passed to the
    engine directly; decoding them back to text would make exact length depend
    on tokenizer round-trip behavior.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    unique_prefix = f"BENCH-{uuid.uuid4().hex} "
    profile = BenchmarkContextProfile(context_profile)
    spec = BENCHMARK_CONTEXT_PROFILES[profile]
    corpus = _load_bench_corpus(profile)

    target_chars = max(round(target_tokens * spec.chars_per_token), 1)
    for _ in range(_PROMPT_BUILD_MAX_ATTEMPTS):
        repeats = (target_chars + len(corpus) - 1) // len(corpus)
        body = (corpus * repeats)[:target_chars]
        tokens = [int(token) for token in tokenizer.encode(unique_prefix + body)]
        if len(tokens) >= target_tokens:
            return tokens[:target_tokens]
        if not tokens:
            raise RuntimeError(
                f"Benchmark corpus {profile.value} tokenized to 0 tokens"
            )

        # Scale by the observed tokenizer ratio, rounding up. The +1 guarantees
        # progress even for tokenizers with coarse or unusual segmentation.
        target_chars = max(
            target_chars + 1,
            (target_chars * target_tokens + len(tokens) - 1) // len(tokens) + 1,
        )

    raise RuntimeError(
        f"Could not build an exact {target_tokens}-token benchmark prompt "
        f"after {_PROMPT_BUILD_MAX_ATTEMPTS} attempts"
    )


def _generate_external_prompt(
    target_tokens: int,
    context_profile: (
        BenchmarkContextProfile | str
    ) = BenchmarkContextProfile.CODE_PYTHON,
) -> str:
    """Generate an approximately target_tokens-long prompt without a tokenizer.

    Uses a unique UUID prefix so remote prefix caches cannot skew results.
    """
    unique_prefix = f"BENCH-{uuid.uuid4().hex} "
    profile = BenchmarkContextProfile(context_profile)
    spec = BENCHMARK_CONTEXT_PROFILES[profile]
    corpus = _load_bench_corpus(profile)
    target_chars = max(
        0,
        round(target_tokens * spec.chars_per_token) - len(unique_prefix),
    )
    repeats = (target_chars + len(corpus) - 1) // len(corpus)
    return unique_prefix + (corpus * repeats)[:target_chars]


def _compute_single_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float,
    end_time: float,
    peak_memory: int,
    cached_tokens: int,
    prefill_duration_s: float | None = None,
    generation_duration_s: float | None = None,
    generation_measured: bool = True,
    timing_observed: bool = True,
) -> dict:
    """Compute all metrics for a single request benchmark."""
    ttft_s = first_token_time - start_time
    prefill_duration = prefill_duration_s if prefill_duration_s is not None else ttft_s
    gen_duration = (
        generation_duration_s
        if generation_duration_s is not None
        else end_time - first_token_time
    )
    e2e_duration = end_time - start_time

    ttft_ms: float | None = ttft_s * 1000
    if generation_measured and completion_tokens > 1 and gen_duration > 0:
        tpot_ms: float | None = (gen_duration / (completion_tokens - 1)) * 1000
        gen_tps: float | None = completion_tokens / gen_duration
    else:
        # Generation timing could not be measured (e.g. all content arrived
        # in a single burst with no measurable inter-token span) — report
        # unmeasured rather than a misleading 0.0.
        tpot_ms = None
        gen_tps = None
    processing_tps: float | None = prompt_tokens / max(prefill_duration, 1e-9)
    total_throughput = (prompt_tokens + completion_tokens) / max(e2e_duration, 1e-9)

    if not timing_observed:
        # The first-token timestamp was never observed and fell back to the
        # end of the response, so TTFT covers the whole response and the
        # prefill rate derived from it is not a prefill rate at all. Only
        # e2e latency and total throughput survive.
        ttft_ms = None
        processing_tps = None

    return {
        "ttft_ms": round(ttft_ms, 1) if ttft_ms is not None else None,
        "tpot_ms": round(tpot_ms, 2) if tpot_ms is not None else None,
        "gen_tps": round(gen_tps, 1) if gen_tps is not None else None,
        "processing_tps": (
            round(processing_tps, 1) if processing_tps is not None else None
        ),
        "e2e_latency_s": round(e2e_duration, 3),
        "total_throughput": round(total_throughput, 1),
        "peak_memory_bytes": peak_memory,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


_EXPERT_STREAMING_DELTA_FIELDS = (
    "route_lookups",
    "pinned_hits",
    "cache_hits",
    "cache_misses",
    "evictions",
    "loads",
    "pinned_loads",
    "cold_loads",
    "scratch_loads",
    "scratch_prefetch_requests",
    "warm_start_loads",
    "expert_major_calls",
    "qmm_calls",
    "sorted_prefill_groups",
    "sorted_prefill_routes",
    "sorted_qmm_calls",
    "route_materialize_calls",
    "just_in_time_loads",
    "native_demand_calls",
    "native_demand_callbacks",
    "native_demand_positions",
    "native_demand_destination_fence_skips",
    "elastic_decode_cache_activations",
    "elastic_decode_cache_demotions",
    "hotness_decays",
    "ssd_bytes_read",
    "ssd_read_operations",
    "ssd_preload_bytes_read",
    "ssd_preload_read_operations",
    "ssd_cold_bytes_read",
    "ssd_cold_read_operations",
    "ssd_io_seconds",
    "ssd_decode_seconds",
    "bank_bind_seconds",
    "bank_materialize_seconds",
    "scratch_prefetch_wait_seconds",
    "scratch_call_seconds",
    "scratch_groups",
    "scratch_load_seconds",
    "scratch_gather_seconds",
    "scratch_compute_seconds",
    "scratch_scatter_seconds",
    "scratch_submit_seconds",
    "scratch_mlx_materialize_seconds",
    "route_materialize_seconds",
    "native_demand_submit_seconds",
    "native_demand_callback_seconds",
)

_EXPERT_STREAMING_CONFIG_FIELDS = (
    "cache_budget_bytes",
    "cache_slots_per_layer",
    "scratch_budget_bytes",
    "scratch_slots_per_layer",
    "layer_count",
    "resident_experts",
    "resident_capacity",
    "execution_bank_slots",
    "execution_banks_per_layer",
    "fused_gate_up",
    "sorted_prefill",
    "streaming_mode",
    "cache_policy",
    "native_demand",
    "native_demand_decode_only",
    "decode_scratch_as_cache",
    "fast_resource_loading",
    "fast_resource_loading_scope",
    "fast_resource_max_gap_bytes",
    "direct_io",
    "hotlist_profile",
    "hotlist_preloaded",
    "optimistic_preloaded",
    "hotlist_profile_error",
)


def _expert_streaming_snapshot(engine: Any) -> dict[str, Any] | None:
    """Read the engine's cumulative streaming counters when available."""
    getter = getattr(engine, "get_stats", None)
    if not callable(getter):
        return None
    try:
        stats = getter()
    except Exception:
        logger.debug(
            "Could not snapshot benchmark expert streaming stats", exc_info=True
        )
        return None
    if not isinstance(stats, dict):
        return None
    streaming = stats.get("expert_streaming")
    return dict(streaming) if isinstance(streaming, dict) else None


def _expert_streaming_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return per-trial counters without copying the large per-layer snapshot."""
    if after is None:
        return None
    before = before or {}
    result = {
        key: after.get(key) for key in _EXPERT_STREAMING_CONFIG_FIELDS if key in after
    }
    for key in _EXPERT_STREAMING_DELTA_FIELDS:
        if key not in after:
            continue
        current = after.get(key, 0)
        previous = before.get(key, 0)
        try:
            result[key] = max(0, current - previous)
        except TypeError:
            continue
    attempts = sum(
        int(result.get(key, 0) or 0)
        for key in ("pinned_hits", "cache_hits", "cache_misses")
    )
    result["hit_rate"] = (
        (int(result.get("pinned_hits", 0) or 0) + int(result.get("cache_hits", 0) or 0))
        / attempts
        if attempts
        else 1.0
    )
    capacity = int(result.get("resident_capacity", 0) or 0)
    result["resident_fill_rate"] = (
        int(result.get("resident_experts", 0) or 0) / capacity
        if capacity
        else 1.0
    )
    hot_capacity = (
        int(result.get("cache_slots_per_layer", 0) or 0)
        * int(result.get("layer_count", 0) or 0)
    )
    result["hotlist_preload_fill_rate"] = (
        min(1.0, int(result.get("hotlist_preloaded", 0) or 0) / hot_capacity)
        if hot_capacity
        else 1.0
    )
    result["startup_preload_fill_rate"] = (
        min(
            1.0,
            (
                int(result.get("hotlist_preloaded", 0) or 0)
                + int(result.get("optimistic_preloaded", 0) or 0)
            )
            / hot_capacity,
        )
        if hot_capacity
        else 1.0
    )
    return result


def _active_metal_memory() -> int:
    try:
        return int(mx.get_active_memory())
    except Exception:
        return 0


def _pin_speed_priority(engine_pool: Any) -> bool | None:
    """Force prefill speed priority for the benchmark's duration.

    Throughput numbers measured while the memory-guard throttle shrinks
    prefill chunks are not comparable, so the bench always runs in speed
    mode. The pin lands on the pool's stored scheduler config — the bench
    model is loaded fresh after unload-all, so that is where its Scheduler
    reads the flag from. Returns the previous value for restoration, or
    None when the pool exposes no config (nothing to restore).
    """
    config = getattr(engine_pool, "_scheduler_config", None)
    if config is None:
        return None
    previous = bool(getattr(config, "prefill_speed_priority", False))
    config.prefill_speed_priority = True
    return previous


def _restore_speed_priority(engine_pool: Any, previous: bool | None) -> None:
    """Undo _pin_speed_priority (no-op when the pin never landed)."""
    if previous is None:
        return
    config = getattr(engine_pool, "_scheduler_config", None)
    if config is not None:
        config.prefill_speed_priority = previous


def _get_batch_benchmark_core(engine: Any) -> Any | None:
    """Return the scheduler core when this engine supports batch benchmarks."""
    engine_core = getattr(engine, "_engine", None)
    if engine_core is None:
        return None
    if not callable(getattr(engine_core, "add_request", None)):
        return None
    if not callable(getattr(engine_core, "stream_outputs", None)):
        return None
    return engine_core


async def _send_event(run: BenchmarkRun, event: dict) -> None:
    """Append an event to the run's log and wake any subscribers.

    Sets `run.terminal` when the event ends the stream so subscribers
    can return rather than wait for an event that will never come.
    """
    async with run.cond:
        run.events.append(event)
        if event.get("type") in _BENCH_TERMINAL_TYPES:
            run.terminal = True
        run.cond.notify_all()


async def _run_single_test(
    engine: Any,
    prompt: list[int],
    max_tokens: int,
    pp_len: int,
    ane_trace_config: dict[str, Any] | None = None,
) -> dict:
    """Run a single request benchmark test and return metrics."""
    if len(prompt) != pp_len:
        raise RuntimeError(
            f"Benchmark prompt length mismatch before pp{pp_len}: "
            f"built {len(prompt)} tokens"
        )

    streaming_before = _expert_streaming_snapshot(engine)
    active_memory_start = _active_metal_memory()

    # Reset peak memory tracking
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    start_time = time.perf_counter()
    first_token_time = None
    last_generated_token_time = None
    last_output = None
    prev_completion_tokens = 0
    ane_profile_enabled = False
    ane_profile: dict[str, dict[str, float]] = {}

    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        ane_profile_enabled = fast.qwen35_ane_profile_set_enabled(True)
        if ane_profile_enabled:
            fast.qwen35_ane_profile_reset()
    except Exception:
        logger.debug("Benchmark ANE profiler unavailable", exc_info=True)

    try:
        async for output in engine.stream_generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            skip_cache_store=True,
            benchmark_trace=True,
            benchmark_ane_sequence_length=int(
                (ane_trace_config or {}).get("sequence_length", 0) or 0
            ),
        ):
            # Detect first generated token via completion_tokens count,
            # not new_text. Some models (e.g. Harmony/gpt-oss) produce
            # protocol tokens that don't yield visible new_text.
            completion_delta = output.completion_tokens - prev_completion_tokens
            if completion_delta > 0:
                generated_at = getattr(output, "generated_at", None)
                generated_until = getattr(output, "generated_until", None)
                output_first_token_time = (
                    float(generated_at)
                    if generated_at is not None
                    else time.perf_counter()
                )
                if first_token_time is None:
                    first_token_time = output_first_token_time
                if generated_until is not None:
                    last_generated_token_time = float(generated_until)
                elif completion_delta == 1:
                    last_generated_token_time = output_first_token_time
            prev_completion_tokens = output.completion_tokens
            last_output = output
    finally:
        if ane_profile_enabled:
            try:
                ane_profile = fast.qwen35_ane_profile_snapshot()
            finally:
                fast.qwen35_ane_profile_set_enabled(False)

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    # Get peak memory
    try:
        peak_memory = mx.get_peak_memory()
    except Exception:
        peak_memory = 0
    active_memory_end = _active_metal_memory()
    streaming_after = _expert_streaming_snapshot(engine)

    if last_output is None:
        raise RuntimeError(f"Benchmark pp{pp_len} produced no engine output")

    prompt_tokens = last_output.prompt_tokens
    if prompt_tokens != pp_len:
        raise RuntimeError(
            f"Benchmark prompt length mismatch after pp{pp_len}: "
            f"engine reported {prompt_tokens} tokens"
        )

    completion_tokens = last_output.completion_tokens
    cached_tokens = last_output.cached_tokens

    if cached_tokens > 0:
        logger.warning(
            f"Benchmark test pp{pp_len} had {cached_tokens} cached tokens "
            f"(expected 0). Results may not reflect true prefill performance."
        )

    prefill_duration_s = None
    generation_duration_s = None
    producer_generation_duration_s = None
    metric_completion_tokens = completion_tokens
    if first_token_time is not None and last_generated_token_time is not None:
        measured_duration = last_generated_token_time - first_token_time
        if measured_duration > 0:
            producer_generation_duration_s = measured_duration
    if last_output is not None:
        prompt_tps = float(getattr(last_output, "prompt_tps", 0.0) or 0.0)
        if prompt_tps > 0 and prompt_tokens > 0:
            prefill_duration_s = prompt_tokens / prompt_tps

        canvas_tps = float(getattr(last_output, "diffusion_canvas_tps", 0.0) or 0.0)
        canvas_tokens = int(getattr(last_output, "diffusion_canvas_tokens", 0) or 0)
        if canvas_tps > 0 and canvas_tokens > 0:
            metric_completion_tokens = canvas_tokens
            generation_duration_s = canvas_tokens / canvas_tps
        else:
            generation_tps = float(getattr(last_output, "generation_tps", 0.0) or 0.0)
            if generation_tps > 0 and completion_tokens > 0:
                generation_duration_s = completion_tokens / generation_tps

    if generation_duration_s is None:
        generation_duration_s = producer_generation_duration_s

    generation_measured = generation_duration_s is not None
    trace_prefill_duration_s = prefill_duration_s
    if trace_prefill_duration_s is None and first_token_time is not None:
        trace_prefill_duration_s = max(0.0, first_token_time - start_time)

    ane_trace = _log_ane_benchmark_trace(
        pp_len=pp_len,
        prefill_duration_s=trace_prefill_duration_s,
        config=ane_trace_config,
        profile=ane_profile,
        # An ABI-skewed extension can enable the profiler yet return an empty
        # snapshot; that must read as unknown, not as an idle ANE.
        profiling_available=ane_profile_enabled and bool(ane_profile),
        scheduler_trace=(
            {
                "chunk_tokens": list(
                    getattr(last_output, "benchmark_prefill_chunks", [])
                ),
                "requested_steps": list(
                    getattr(last_output, "benchmark_requested_steps", [])
                ),
                "boundary_enabled": bool(
                    getattr(last_output, "benchmark_boundary_enabled", False)
                ),
                "cache_block_size": int(
                    getattr(last_output, "benchmark_cache_block_size", 0) or 0
                ),
            }
            if last_output is not None
            else None
        ),
    )

    metrics = _compute_single_metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=metric_completion_tokens,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        peak_memory=peak_memory,
        cached_tokens=cached_tokens,
        prefill_duration_s=prefill_duration_s,
        generation_duration_s=generation_duration_s,
        generation_measured=generation_measured,
    )
    if ane_trace_config is not None:
        metrics["ane_trace"] = ane_trace
    metrics["metal_memory"] = {
        "active_start_bytes": active_memory_start,
        "active_end_bytes": active_memory_end,
        "peak_bytes": peak_memory,
    }
    streaming_delta = _expert_streaming_delta(streaming_before, streaming_after)
    if streaming_delta is not None:
        metrics["expert_streaming"] = streaming_delta
    return metrics


def _log_ane_benchmark_trace(
    *,
    pp_len: int,
    prefill_duration_s: float | None,
    config: dict[str, Any] | None,
    profile: dict[str, dict[str, float]],
    profiling_available: bool = True,
    scheduler_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Log an offline-comparable ANE/scheduler summary for one PP trial.

    Returns the same per-category counters that are logged, so callers can
    tell whether the ANE actually executed rather than only whether its
    programs compiled.
    """
    config = config or {}
    scheduler_trace = scheduler_trace or {}
    sequence_length = int(config.get("sequence_length", 0) or 0)
    prefill_tokens = max(0, pp_len - 1)
    chunk_tokens = [
        int(value)
        for value in scheduler_trace.get("chunk_tokens", [])
        if int(value) > 0
    ]
    requested_steps = [
        int(value)
        for value in scheduler_trace.get("requested_steps", [])
        if int(value) > 0
    ]
    accounting_widths = chunk_tokens or [prefill_tokens]
    full_shapes, tail = (0, prefill_tokens)
    if sequence_length > 0:
        divisions = [divmod(width, sequence_length) for width in accounting_widths]
        full_shapes = sum(full for full, _ in divisions)
        tail = sum(remainder for _, remainder in divisions)

    def width_histogram(values: list[int]) -> str:
        counts: dict[int, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return ",".join(
            f"{width}x{count}" for width, count in sorted(counts.items(), reverse=True)
        ) or "none"

    logger.info(
        "[benchmark-ane-summary] pp=%d prefill_tokens=%d sequence_length=%d "
        "model_calls=%d model_call_widths=%s requested_steps=%s "
        "boundary_enabled=%s cache_block_size=%d accounting=%s "
        "full_ane_tiles=%d gpu_tail_tokens=%d measured_prefill_ms=%s",
        pp_len,
        prefill_tokens,
        sequence_length,
        len(chunk_tokens),
        width_histogram(chunk_tokens),
        width_histogram(requested_steps),
        bool(scheduler_trace.get("boundary_enabled", False)),
        int(scheduler_trace.get("cache_block_size", 0) or 0),
        "observed" if chunk_tokens else "prompt_estimate",
        full_shapes,
        tail,
        (
            f"{prefill_duration_s * 1000.0:.3f}"
            if prefill_duration_s is not None
            else "unknown"
        ),
    )

    summary: dict[str, Any] = {
        # Without the profiler the operation counts below are all zero
        # regardless of what the ANE did, so callers must not read them as
        # evidence that it stayed idle.
        "profiling_available": bool(profiling_available),
        "sequence_length": sequence_length,
        "expected_full_shapes": full_shapes,
        "gpu_tail_tokens": tail,
        "categories": {},
    }

    for category, layer_key, compiled_key in (
        ("mlp", "mlp_layers", "compiled_mlp_layers"),
        ("gdn", "gdn_layers", "compiled_gdn_layers"),
    ):
        values = profile.get(category, {})
        operations = int(values.get("operations", 0) or 0)
        configured_layers = int(config.get(layer_key, 0) or 0)
        compiled_value = config.get(compiled_key)
        compiled_layers = (
            None if compiled_value is None else int(compiled_value)
        )
        # Expectations follow what the patch actually compiled; the settings
        # value is only the fallback when the runtime state is unknown.
        layers = (
            compiled_layers if compiled_layers is not None else configured_layers
        )
        observed_shapes = operations / layers if layers > 0 else 0.0
        expected_operations = full_shapes * layers
        per_op = max(operations, 1)
        elapsed_ns = (
            prefill_duration_s * 1e9
            if prefill_duration_s is not None and prefill_duration_s > 0
            else 0.0
        )
        logger.info(
            "[benchmark-ane-profile] pp=%d category=%s operations=%d "
            "configured_layers=%d compiled_layers=%s expected_operations=%d "
            "observed_ane_tiles_per_layer=%.3f input_ready_ms_per_op=%.3f "
            "ane_region_ms_per_op=%.3f ane0_eval_ms_per_op=%.3f "
            "ane1_eval_ms_per_op=%.3f gpu_qmm_ms_per_op=%.3f "
            "gap_before_ms_per_op=%.3f ane0_duty=%.4f ane1_duty=%.4f",
            pp_len,
            category,
            operations,
            configured_layers,
            "unknown" if compiled_layers is None else compiled_layers,
            expected_operations,
            observed_shapes,
            float(values.get("pack_ns", 0.0) or 0.0) / per_op / 1e6,
            float(values.get("ane_region_ns", 0.0) or 0.0) / per_op / 1e6,
            float(values.get("ane0_eval_ns", 0.0) or 0.0) / per_op / 1e6,
            float(values.get("ane1_eval_ns", 0.0) or 0.0) / per_op / 1e6,
            float(values.get("gpu_qmm_ns", 0.0) or 0.0) / per_op / 1e6,
            float(values.get("gap_before_ns", 0.0) or 0.0) / per_op / 1e6,
            (
                float(values.get("ane0_eval_ns", 0.0) or 0.0) / elapsed_ns
                if elapsed_ns
                else 0.0
            ),
            (
                float(values.get("ane1_eval_ns", 0.0) or 0.0) / elapsed_ns
                if elapsed_ns
                else 0.0
            ),
        )

        summary["categories"][category] = {
            "operations": operations,
            "expected_operations": expected_operations,
            "observed_shapes": observed_shapes,
            "configured_layers": configured_layers,
            "compiled_layers": compiled_layers,
        }

    return summary


async def _run_batch_test(
    engine: Any,
    prompts: list[list[int]],
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Run a continuous batching benchmark test.

    Submits batch_size concurrent requests via the engine core and measures
    aggregate throughput including pp TPS and tg TPS.

    Args:
        prompts: List of prompts (one per request). For same-prompt tests,
                 all entries are identical. For different-prompt tests, each
                 has a unique UUID prefix.
        prompt_tokens: Number of prompt tokens per request (for pp TPS calc).
    """
    from ..request import SamplingParams

    engine_core = _get_batch_benchmark_core(engine)
    if engine_core is None:
        raise ValueError("Engine does not support batch benchmarks")
    if len(prompts) < batch_size:
        raise RuntimeError(
            f"Benchmark batch requires {batch_size} prompts, got {len(prompts)}"
        )
    invalid_lengths = [
        len(prompt) for prompt in prompts[:batch_size] if len(prompt) != prompt_tokens
    ]
    if invalid_lengths:
        raise RuntimeError(
            f"Benchmark batch prompt length mismatch: expected {prompt_tokens}, "
            f"got {invalid_lengths}"
        )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    streaming_before = _expert_streaming_snapshot(engine)
    active_memory_start = _active_metal_memory()

    async def _single_request(prompt: list[int]) -> dict:
        """Run a single request within the batch."""
        start = time.perf_counter()
        first_token = None
        tokens = 0
        prev_tokens = 0
        reported_prompt_tokens = 0

        request_id = await engine_core.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
            skip_cache_store=True,
        )

        async for output in engine_core.stream_outputs(request_id):
            if first_token is None and output.completion_tokens > prev_tokens:
                first_token = time.perf_counter()
            prev_tokens = output.completion_tokens
            if output.finished:
                tokens = output.completion_tokens
                reported_prompt_tokens = output.prompt_tokens

        end = time.perf_counter()
        if first_token is None:
            first_token = end
        if reported_prompt_tokens != prompt_tokens:
            raise RuntimeError(
                f"Benchmark batch prompt length mismatch after submission: "
                f"expected {prompt_tokens}, engine reported "
                f"{reported_prompt_tokens}"
            )

        return {
            "ttft_s": first_token - start,
            "first_token_abs": first_token,
            "end_abs": end,
            "completion_tokens": tokens,
        }

    # Submit all requests concurrently. Nothing else generates during the
    # gather, so the process-global peak counter belongs to this batch.
    if HAS_MLX:
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_single_request(prompts[i]) for i in range(batch_size)]
    )
    wall_end = time.perf_counter()
    peak_memory = 0
    if HAS_MLX:
        try:
            peak_memory = mx.get_peak_memory()
        except Exception:
            peak_memory = 0
    active_memory_end = _active_metal_memory()
    streaming_after = _expert_streaming_snapshot(engine)

    # Aggregate metrics
    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = prompt_tokens * batch_size
    wall_time = wall_end - wall_start
    avg_ttft_ms = (sum(r["ttft_s"] for r in results) / batch_size) * 1000

    # pp TPS: total prompt tokens / time until ALL requests finish prefill
    max_first_token = max(r["first_token_abs"] for r in results)
    prefill_wall_time = max_first_token - wall_start
    pp_tps = total_prompt_tokens / max(prefill_wall_time, 1e-9)

    # tg TPS: total generated tokens / generation wall time
    # Generation starts when the last request finishes prefill
    gen_wall_time = wall_end - max_first_token
    tg_tps = total_gen_tokens / max(gen_wall_time, 1e-9)

    metrics = {
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "e2e_latency_s": round(wall_time, 3),
        "peak_memory_bytes": peak_memory,
        "total_gen_tokens": total_gen_tokens,
        "batch_size": batch_size,
        "metal_memory": {
            "active_start_bytes": active_memory_start,
            "active_end_bytes": active_memory_end,
            "peak_bytes": peak_memory,
        },
    }
    streaming_delta = _expert_streaming_delta(streaming_before, streaming_after)
    if streaming_delta is not None:
        metrics["expert_streaming"] = streaming_delta
    return metrics


async def _run_external_single_test(
    client: ExternalAPIClient,
    prompt: str,
    max_tokens: int,
) -> dict:
    """Run a single-request benchmark against an external endpoint.

    Token counts come from the endpoint's streamed usage payload, never
    from counting SSE chunks (providers batch multiple tokens per chunk).
    Prefill duration is not observable remotely, so pp TPS falls back to
    prompt_tokens / TTFT (network latency included). Peak memory is not
    measurable for a remote host.
    """
    stats = await client.stream_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    gen_duration = stats.last_content_time - stats.first_content_time
    metrics = _compute_single_metrics(
        prompt_tokens=stats.prompt_tokens,
        completion_tokens=stats.completion_tokens,
        start_time=stats.start_time,
        first_token_time=stats.first_content_time,
        end_time=stats.end_time,
        peak_memory=0,
        cached_tokens=stats.cached_tokens,
        prefill_duration_s=None,
        generation_duration_s=gen_duration if gen_duration > 0 else None,
        generation_measured=gen_duration > 0,
        timing_observed=stats.content_observed,
    )
    metrics["peak_memory_bytes"] = None
    return metrics


async def _run_external_batch_test(
    client: ExternalAPIClient,
    prompts: list[str],
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Run a concurrent-requests benchmark against an external endpoint.

    Mirrors _run_batch_test aggregation, with actual per-request token
    counts taken from each stream's usage payload.
    """
    wall_start = time.perf_counter()
    stats_list = await asyncio.gather(
        *[
            client.stream_chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.0,
            )
            for prompt in prompts
        ]
    )
    wall_end = time.perf_counter()

    total_gen_tokens = sum(s.completion_tokens for s in stats_list)
    prompt_tokens_per_request = [s.prompt_tokens for s in stats_list]
    total_prompt_tokens = sum(prompt_tokens_per_request)
    wall_time = wall_end - wall_start

    # Every aggregate below is derived from the per-request content
    # timestamps, so a single stream that never reported content poisons all
    # of them: its fallback timestamp sits at the end of the response and
    # drags max_first_token along with it.
    timing_observed = all(s.content_observed for s in stats_list)
    decode_observed = all(
        s.last_content_time > s.first_content_time for s in stats_list
    )

    max_first_token = max(s.first_content_time for s in stats_list)
    gen_window = wall_end - max_first_token

    avg_ttft_ms: float | None = None
    pp_tps: float | None = None
    tg_tps: float | None = None
    if timing_observed:
        total_ttft_s = sum(s.first_content_time - s.start_time for s in stats_list)
        avg_ttft_ms = round((total_ttft_s / batch_size) * 1000, 1)
        # pp TPS: total prompt tokens / time until ALL requests emit content
        prefill_window = max(max_first_token - wall_start, 1e-9)
        pp_tps = round(total_prompt_tokens / prefill_window, 1)
        # tg TPS needs a real decode span. wall_end is sampled after
        # asyncio.gather returns, so gen_window stays positive even when
        # every per-request timestamp collapsed onto the end of the
        # response, and the window alone cannot tell a genuine decode phase
        # from a single-chunk dump.
        if decode_observed and gen_window > 0:
            tg_tps = round(total_gen_tokens / gen_window, 1)

    return {
        "pp_tps": pp_tps,
        "tg_tps": tg_tps,
        "avg_ttft_ms": avg_ttft_ms,
        "e2e_latency_s": round(wall_time, 3),
        "total_gen_tokens": total_gen_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "prompt_tokens": round(total_prompt_tokens / batch_size),
        "prompt_tokens_min": min(prompt_tokens_per_request),
        "prompt_tokens_max": max(prompt_tokens_per_request),
        "batch_size": batch_size,
    }


OMLX_AI_API_URL = "https://omlx.ai/api/benchmarks"

# The leaderboard accepts model_name up to 150 characters.
_MAX_MODEL_NAME_LEN = 150


def _detect_quantization(model_path: str) -> str:
    """Detect model quantization from config.json or directory name.

    Fallback chain: config.json → directory name → "unknown"
    """
    config_path = Path(model_path) / "config.json"
    if config_path.exists():
        try:
            with open(config_path) as f:
                config = json.load(f)
            qconfig = config.get("quantization_config", {})
            bits = qconfig.get("bits")
            if bits is not None:
                return f"{bits}bit"
        except Exception:
            pass

    # Fallback: extract from directory name
    dirname = Path(model_path).name
    match = re.search(
        r"(2bit|3bit|4bit|6bit|8bit|fp16|bf16|MXFP4|NVFP4)", dirname, re.IGNORECASE
    )
    if match:
        return match.group(1).lower()

    return "unknown"


def _upload_model_name(model_id: str) -> str:
    """Model name to publish: exactly what oMLX shows and its copy button copies.

    Quantization and MLX suffixes used to be stripped here, which lost the one
    detail that distinguishes two builds of the same model on the leaderboard.
    The trailing path component is taken defensively — discovery registers ids
    as a single path component today, so this is a no-op for local runs.
    """
    name = model_id.rstrip("/").split("/")[-1]
    return name[:_MAX_MODEL_NAME_LEN]


def _upload_model_repo(
    model_id: str, entry: Any = None, model_dirs: Any = None
) -> Optional[str]:
    """Org-qualified repo id to publish alongside the model name (#1808).

    Uses the same derivation as the models UI display name: the HF repo id
    when known, otherwise the org/leaf relative path under a configured
    model dir. Returns None when nothing beyond the bare model id can be
    derived (flat layouts), so the site simply shows no repo for those rows.
    """
    if entry is None:
        return None
    try:
        derived = model_display_name(
            model_id,
            getattr(entry, "model_path", None),
            list(model_dirs or []),
            source_repo_id=getattr(entry, "source_repo_id", None),
        )
    except Exception as e:  # noqa: BLE001
        logger.debug(f"Benchmark: repo derivation failed: {e}")
        return None
    if not isinstance(derived, str) or derived == model_id or "/" not in derived:
        return None
    return derived[:_MAX_MODEL_NAME_LEN]


def _sanitize_upload_error(resp: Any) -> str:
    """Extract a user-presentable error string from a failed upload response.

    Avoids dumping raw HTML bodies (e.g. Cloudflare's "Just a moment..."
    challenge interstitial) into the dashboard's red-x error column.
    Detects CF mitigation specifically so users get actionable context
    instead of a 5KB markup blob.

    Resolution order:
    1. Cloudflare challenge — header ``cf-mitigated: challenge`` is
       authoritative; a body sniff for "just a moment" / "cf-chl" covers
       edge transports that strip the header.
    2. JSON envelope — the omlx.ai API's normal error shape; extract
       ``error`` / ``detail`` / ``message`` if present, truncated.
    3. Plain-text body — short responses only; HTML-looking bodies are
       collapsed to a one-line "non-JSON response (N bytes)" hint.
    4. Fallback to the bare HTTP status code.
    """
    headers = getattr(resp, "headers", {}) or {}
    cf_mitigated = str(headers.get("cf-mitigated", "")).lower()
    body = getattr(resp, "text", "") or ""
    status = getattr(resp, "status_code", "?")

    body_head = body[:512].lower()
    if (
        cf_mitigated == "challenge"
        or "just a moment" in body_head
        or "cf-chl" in body_head
    ):
        return (
            f"Upload blocked by Cloudflare (HTTP {status}). "
            f"This is a server-side issue with omlx.ai — retry later or "
            f"report it to the maintainer."
        )

    try:
        data = resp.json()
        msg = data.get("error") or data.get("detail") or data.get("message")
        if msg:
            return str(msg)[:300]
    except Exception:
        pass

    text = body.strip()
    if "<" in text and ">" in text:
        return f"HTTP {status} — unexpected non-JSON response ({len(body)} bytes)"
    return text[:300] or f"HTTP {status}"


async def _upload_to_omlx_ai(run: BenchmarkRun, engine_pool: Any) -> None:
    """Upload benchmark results to omlx.ai community benchmarks.

    Sends each single-request result as a separate submission,
    grouped by submission_group. Upload failures don't affect
    the benchmark run status.
    """
    import requests

    from .._version import __version__
    from ..utils.hardware import (
        compute_owner_hash,
        get_chip_name,
        get_gpu_core_count,
        get_io_platform_uuid,
        get_os_version,
        get_total_memory_gb,
        parse_chip_info,
    )

    # Accelerated runs upload too. They carry their flags so the leaderboard
    # can mark and filter them, which is more useful than withholding the one
    # set of numbers people most want to see.
    run.upload_state["feature_flags"] = list(run.feature_flags)
    if run.feature_flags:
        logger.info(
            "Benchmark upload tagged with acceleration flags: "
            f"{[f['key'] for f in run.feature_flags]}"
        )

    run.upload_state["phase"] = "uploading"
    await _send_event(
        run,
        {
            "type": "progress",
            "phase": "upload",
            "message": "Uploading to community benchmarks...",
            "current": 0,
            "total": 0,
        },
    )

    # Collect hardware info
    chip_string = get_chip_name()
    chip_name, chip_variant = parse_chip_info(chip_string)
    memory_gb = round(get_total_memory_gb())
    gpu_cores = get_gpu_core_count()
    os_version = get_os_version()
    omlx_version = __version__

    # Compute owner_hash
    owner_hash_full = None
    owner_hash_display = None
    io_uuid = get_io_platform_uuid()
    if io_uuid:
        owner_hash_full = compute_owner_hash(io_uuid, chip_name, gpu_cores, memory_gb)
        # Display hash is without the verify character
        owner_hash_display = owner_hash_full[:-1]

    # Get model info
    entry = engine_pool.get_entry(run.request.model_id)
    model_path = entry.model_path if entry else ""
    quantization = _detect_quantization(model_path)
    model_name = _upload_model_name(run.request.model_id)

    # Generate submission group
    submission_group = str(uuid.uuid4())

    # Peak process memory for the run. ri_lifetime_max_phys_footprint is a
    # high-water mark since process start, so it only describes this benchmark
    # when the benchmark actually set a new maximum — a server that previously
    # held a larger model would otherwise report that older peak. Fall back to
    # the sampler's own maximum, which is scoped to the run.
    peak_footprint_gb = None
    lifetime_end = get_lifetime_max_phys_footprint()
    peak_bytes = 0
    if lifetime_end and lifetime_end > run.lifetime_footprint_at_start:
        peak_bytes = lifetime_end
    elif run.sampler is not None:
        peak_bytes = run.sampler.run_peak_footprint()
    if peak_bytes > 0:
        peak_footprint_gb = round(peak_bytes / (1024**3), 2)

    # Collect single results and batch results
    single_results = [r for r in run.results if r.get("test_type") == "single"]
    uploadable_single_results = [
        r for r in single_results if float(r.get("gen_tps", 0.0) or 0.0) > 0.0
    ]
    skipped_count = len(single_results) - len(uploadable_single_results)
    batch_results = [r for r in run.results if r.get("test_type") == "batch"]

    # Build batching_results from batch data
    batching_results = []
    pp1024_single = next((r for r in single_results if r.get("pp") == 1024), None)
    if (
        pp1024_single
        and float(pp1024_single.get("gen_tps", 0.0) or 0.0) > 0.0
        and batch_results
    ):
        baseline_tps = pp1024_single["gen_tps"]
        batching_results.append(
            {
                "batch_size": 1,
                "tg_tps": baseline_tps,
                "speedup": 1.0,
            }
        )
        for br in batch_results:
            speedup = round(br["tg_tps"] / baseline_tps, 2) if baseline_tps > 0 else 1.0
            batching_results.append(
                {
                    "batch_size": br["batch_size"],
                    "tg_tps": br["tg_tps"],
                    "speedup": speedup,
                }
            )

    success_count = 0
    failed_count = 0

    if skipped_count:
        logger.info(
            f"Benchmark upload skipped {skipped_count} result(s) without "
            f"measurable generation throughput"
        )

    for result in uploadable_single_results:
        context_length = result["pp"]
        peak_mem_gb = None
        if result.get("peak_memory_bytes") and result["peak_memory_bytes"] > 0:
            peak_mem_gb = round(result["peak_memory_bytes"] / (1024**3), 2)

        payload = {
            "chip_name": chip_name,
            "chip_variant": chip_variant,
            "memory_gb": memory_gb,
            "gpu_cores": gpu_cores,
            "omlx_version": omlx_version,
            "os_version": os_version,
            "model_name": model_name,
            "quantization": quantization,
            "context_length": context_length,
            "context_profile": run.request.context_profile.value,
            "pp_tps": result["processing_tps"],
            "tg_tps": result["gen_tps"],
            "ttft_ms": result.get("ttft_ms"),
            "peak_memory_gb": peak_mem_gb,
            "submission_group": submission_group,
            "peak_footprint_gb": peak_footprint_gb,
            "feature_flags": run.feature_flags,
            "model_settings": _with_benchmark_context(
                run.request.context_profile,
                run.model_settings_snapshot,
            ),
            # Per-row: each context length has its own load window. Stays None
            # when sampling was unavailable, so the site does not average
            # fabricated zeros in as measurements.
            "system_metrics": result.get("system_metrics"),
        }

        if owner_hash_full:
            payload["owner_hash"] = owner_hash_full

        # Attach batching_results only to the first submission (lowest context_length)
        if context_length == uploadable_single_results[0]["pp"] and batching_results:
            payload["batching_results"] = batching_results

        try:
            resp = await asyncio.to_thread(
                requests.post,
                OMLX_AI_API_URL,
                json=payload,
                timeout=15,
            )

            if resp.status_code == 201:
                data = resp.json()
                success_count += 1
                result_dict = {
                    "context_length": context_length,
                    "id": data.get("id"),
                    "url": data.get("url"),
                }
                run.upload_state["results"].append(result_dict)
                await _send_event(
                    run,
                    {
                        "type": "upload",
                        "data": result_dict,
                    },
                )
            elif resp.status_code == 409:
                data = resp.json()
                success_count += 1  # Duplicate is still ok
                result_dict = {
                    "context_length": context_length,
                    "id": data.get("existing_id"),
                    "url": data.get("existing_url"),
                    "duplicate": True,
                }
                run.upload_state["results"].append(result_dict)
                await _send_event(
                    run,
                    {
                        "type": "upload",
                        "data": result_dict,
                    },
                )
            else:
                failed_count += 1
                error_msg = _sanitize_upload_error(resp)
                result_dict = {
                    "context_length": context_length,
                    "error": error_msg,
                }
                run.upload_state["results"].append(result_dict)
                await _send_event(
                    run,
                    {
                        "type": "upload",
                        "data": result_dict,
                    },
                )
                # Surface the sanitized message to ops; the full body
                # (truncated) goes to debug so it can still be retrieved
                # from the log file if needed.
                logger.warning(
                    f"Benchmark upload failed for pp{context_length}: "
                    f"{resp.status_code} {error_msg}"
                )
                if (resp.text or "")[:1] not in ("{", "["):
                    logger.debug(
                        "Benchmark upload non-JSON body (truncated): %r",
                        (resp.text or "")[:500],
                    )

        except Exception as e:
            failed_count += 1
            result_dict = {
                "context_length": context_length,
                "error": str(e),
            }
            run.upload_state["results"].append(result_dict)
            await _send_event(
                run,
                {
                    "type": "upload",
                    "data": result_dict,
                },
            )
            logger.warning(f"Benchmark upload error for pp{context_length}: {e}")

    run.upload_state["phase"] = "done"
    run.upload_state["total"] = len(uploadable_single_results)
    run.upload_state["success_count"] = success_count
    run.upload_state["failed_count"] = failed_count
    run.upload_state["skipped_count"] = skipped_count
    run.upload_state["owner_hash"] = owner_hash_display
    await _send_event(
        run,
        {
            "type": "upload_done",
            "data": {
                "owner_hash": owner_hash_display,
                "total": len(uploadable_single_results),
                "success": success_count,
                "failed": failed_count,
                "skipped": skipped_count,
                # Also on the event so SSE-only consumers (the HTML dashboard) get
                # the flags without polling /results.
                "feature_flags": run.feature_flags,
            },
        },
    )

    logger.info(
        f"Benchmark upload complete: {success_count}/"
        f"{len(uploadable_single_results)} succeeded, skipped={skipped_count}"
    )


async def run_benchmark(run: BenchmarkRun, engine_pool: Any) -> None:
    """Execute a complete benchmark run.

    Phases:
    1. Unload all loaded models
    2. Load the target model
    3. Run single request tests
    4. Run batch tests
    5. Unload the benchmark model
    """
    request = run.request
    if request.external is not None:
        await _run_external_benchmark(run)
        return
    total_tests = len(request.prompt_lengths) + len(request.batch_sizes)
    current_test = 0
    overall_start = time.perf_counter()

    # Throughput measurements must not be skewed by the memory-guard
    # throttle shrinking chunks; pin speed priority for the run.
    previous_speed_priority = _pin_speed_priority(engine_pool)

    try:
        run.model_settings_snapshot = _with_benchmark_context(
            request.context_profile,
            None,
        )
        # Snapshot experimental flags at run start. Settings can change mid-run,
        # and the produced numbers are tied to whatever was active when
        # generation actually ran.
        model_settings = None
        sm = getattr(engine_pool, "_settings_manager", None)
        if sm is not None:
            try:
                model_settings = sm.get_settings(request.model_id)
                run.experimental_features.extend(
                    _detect_experimental_features(model_settings)
                )
                run.feature_flags = _derive_feature_flags(model_settings)
                run.model_settings_snapshot = _with_benchmark_context(
                    request.context_profile,
                    _filter_uploaded_settings(model_settings),
                )
            except Exception as e:
                logger.warning(
                    f"Benchmark: failed to read experimental flags for "
                    f"{request.model_id}: {e}"
                )

        # Phase 1: Unload all loaded models
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "unload",
                    "message": f"Unloading {len(loaded_ids)} model(s)...",
                    "current": 0,
                    "total": total_tests,
                },
            )
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                    logger.info(f"Benchmark: unloaded {model_id}")
                except Exception as e:
                    logger.warning(f"Benchmark: failed to unload {model_id}: {e}")

        # Phase 2: Load the target model
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "load",
                "message": f"Loading {request.model_id}...",
                "current": 0,
                "total": total_tests,
            },
        )
        # Both external VLM MTP and Lightning MTP on merged VLM checkpoints
        # require VLMBatchedEngine. Text-only Lightning MTP models still
        # resolve to BatchedEngine through their natural engine type.
        external_vlm_mtp_active = (
            model_settings is not None
            and getattr(model_settings, "vlm_mtp_enabled", False)
            and getattr(model_settings, "vlm_mtp_draft_model", None)
        )
        lightning_mtp_active = model_settings is not None and getattr(
            model_settings, "mtp_enabled", False
        )
        force_lm = request.force_lm_engine or not (
            external_vlm_mtp_active or lightning_mtp_active
        )
        engine = await engine_pool.get_engine(
            request.model_id,
            force_lm=force_lm,
        )
        logger.info(f"Benchmark: loaded {request.model_id}")

        # Generate prompts for all needed lengths
        tokenizer = engine.tokenizer
        prompts: dict[int, list[int]] = {}
        single_prompt_lengths = _single_prompt_lengths(request)
        for pp_len in single_prompt_lengths:
            prompts[pp_len] = _generate_prompt(
                tokenizer,
                pp_len,
                request.context_profile,
            )

        # Ensure pp1024 prompt exists for batch tests
        if request.batch_sizes and 1024 not in prompts:
            prompts[1024] = _generate_prompt(
                tokenizer,
                1024,
                request.context_profile,
            )

        # Warmup: run a short request to trigger JIT compilation,
        # Metal shader compilation, and KV cache initialization.
        # Without this, the first real benchmark test absorbs all
        # one-time overhead and shows artificially low pp TPS.
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "warmup",
                "message": "Warming up (JIT compile)...",
                "current": 0,
                "total": total_tests,
            },
        )
        warmup_started = time.perf_counter()
        warmup_prompt_tokens = _warmup_prompt_tokens(request.warmup_mode)
        warmup_prefill_tokens = warmup_prompt_tokens - 1
        warmup_prompt = _generate_prompt(
            tokenizer,
            warmup_prompt_tokens,
            request.context_profile,
        )
        warmup_max_tokens = (
            request.generation_length
            if getattr(engine, "is_diffusion_model", False)
            else 8
        )
        async for _ in engine.stream_generate(
            prompt=warmup_prompt, max_tokens=warmup_max_tokens, temperature=0.0
        ):
            pass
        logger.info(
            "Benchmark: warmup complete "
            f"(mode={request.warmup_mode.value}, "
            f"prompt_tokens={warmup_prompt_tokens}, "
            f"prefill_tokens={warmup_prefill_tokens}, "
            f"wall_ms={(time.perf_counter() - warmup_started) * 1000.0:.3f})"
        )

        # Start host sampling after warmup: Metal shader and JIT compilation
        # would otherwise be folded into the CPU aggregates.
        run.lifetime_footprint_at_start = get_lifetime_max_phys_footprint()
        try:
            run.sampler = SystemSampler()
            run.sampler.start()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Benchmark: host sampling unavailable: {e}")
            run.sampler = None

        # Phase 3: Single request tests
        single_pp1024_gen_tps = None
        ane_trace_config: dict[str, Any] | None = None
        if model_settings is not None and getattr(
            model_settings, "qwen35_ane_prefill_enabled", False
        ):
            gdn_enabled = bool(
                getattr(model_settings, "qwen35_ane_prefill_gdn", False)
            )
            # The settings flag only records intent. The load-time patch stores
            # what it actually compiled on the model, so the trace and the
            # uploaded metadata reflect the runtime state (the patch can find
            # no eligible layers or drop layers at the program budget).
            loaded_model = getattr(engine, "_model", None)
            compiled_mlp = getattr(
                loaded_model, "_omlx_ane_mlp_prefill_count", None
            )
            compiled_gdn = getattr(
                loaded_model, "_omlx_ane_gdn_prefill_count", None
            )
            ane_active = bool(compiled_mlp or compiled_gdn)
            ane_trace_config = {
                "sequence_length": int(
                    getattr(
                        model_settings,
                        "qwen35_ane_prefill_sequence_length",
                        2048,
                    )
                ),
                "mlp_layers": int(
                    getattr(model_settings, "qwen35_ane_prefill_max_layers", 0)
                ),
                "gdn_layers": (
                    int(
                        getattr(
                            model_settings,
                            "qwen35_ane_prefill_gdn_max_layers",
                            0,
                        )
                    )
                    if gdn_enabled
                    else 0
                ),
                "compiled_mlp_layers": (
                    None if compiled_mlp is None else int(compiled_mlp)
                ),
                "compiled_gdn_layers": (
                    None if compiled_gdn is None else int(compiled_gdn)
                ),
                "active": ane_active,
            }
            if not ane_active:
                run.feature_flags = [
                    flag
                    for flag in run.feature_flags
                    if flag.get("key") != "qwen35_ane_prefill"
                ]
                run.experimental_features = [
                    feature
                    for feature in run.experimental_features
                    if feature != "qwen35_ane_prefill"
                ]
                logger.info(
                    "Qwen ANE prefill is enabled in settings but inactive at "
                    "runtime; benchmark metadata reports it as off"
                )
        logger.info(
            "[benchmark-ane-config] model=%s enabled=%s config=%s",
            request.model_id,
            ane_trace_config is not None,
            ane_trace_config,
        )
        configured_scheduler = getattr(engine_pool, "_scheduler_config", None)
        runtime_scheduler = getattr(
            getattr(getattr(engine, "_engine", None), "engine", None),
            "scheduler",
            None,
        )
        effective_scheduler = getattr(runtime_scheduler, "config", None)
        logger.info(
            "[benchmark-scheduler-config] configured_step=%s effective_step=%s "
            "effective_qwen_floor=%s configured_block_size=%s "
            "effective_block_size=%s boundary_cache_available=%s "
            "configured_chunked_prefill=%s effective_chunked_prefill=%s "
            "speed_priority=%s max_num_batched_tokens=%s",
            getattr(configured_scheduler, "prefill_step_size", None),
            getattr(effective_scheduler, "prefill_step_size", None),
            getattr(runtime_scheduler, "_qwen35_prefill_floor", None),
            getattr(configured_scheduler, "paged_cache_block_size", None),
            getattr(effective_scheduler, "paged_cache_block_size", None),
            bool(getattr(runtime_scheduler, "block_aware_cache", None)),
            getattr(configured_scheduler, "chunked_prefill", None),
            getattr(effective_scheduler, "chunked_prefill", None),
            getattr(effective_scheduler, "prefill_speed_priority", None),
            getattr(effective_scheduler, "max_num_batched_tokens", None),
        )

        for pp_len in single_prompt_lengths:
            current_test += 1
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "single",
                    "message": f"Single: pp{pp_len}/tg{request.generation_length}",
                    "current": current_test,
                    "total": total_tests,
                },
            )

            # time.monotonic only — the test internals use perf_counter, and
            # the two clocks have different epochs.
            window_start = time.monotonic()
            metrics = await _run_single_test(
                engine=engine,
                prompt=prompts[pp_len],
                max_tokens=request.generation_length,
                pp_len=pp_len,
                ane_trace_config=ane_trace_config,
            )
            metrics["system_metrics"] = _sample_window(run, window_start)
            logger.info(
                "[benchmark-pp-result] pp=%d processing_tps=%.3f "
                "ttft_ms=%.3f e2e_ms=%.3f cached_tokens=%d",
                pp_len,
                float(metrics.get("processing_tps", 0.0) or 0.0),
                float(metrics.get("ttft_ms", 0.0) or 0.0),
                float(metrics.get("e2e_latency_s", 0.0) or 0.0) * 1000.0,
                int(metrics.get("cached_tokens", 0) or 0),
            )
            streaming_metrics = metrics.get("expert_streaming")
            if isinstance(streaming_metrics, dict):
                logger.info(
                    "[benchmark-expert-streaming] pp=%d slots=%s "
                    "scratch=%s execution_bank=%s banks=%s fused_gate_up=%s "
                    "hit_rate=%.4f hits=%d "
                    "misses=%d evictions=%d loads=%d ssd_bytes=%d "
                    "ssd_io_ms=%.3f decode_ms=%.3f bind_ms=%.3f "
                    "materialize_ms=%.3f scratch_wait_ms=%.3f "
                    "scratch_mlx_ms=%.3f "
                    "expert_major=%d qmm_calls=%d sorted_groups=%d "
                    "sorted_routes=%d sorted_qmm=%d "
                    "resident_fill=%.4f hotlist_preloaded=%s "
                    "optimistic_preloaded=%s hotlist_fill=%.4f "
                    "startup_fill=%.4f metal_start_gib=%.3f "
                    "metal_end_gib=%.3f metal_peak_gib=%.3f",
                    pp_len,
                    streaming_metrics.get("cache_slots_per_layer"),
                    streaming_metrics.get("scratch_slots_per_layer"),
                    streaming_metrics.get("execution_bank_slots"),
                    streaming_metrics.get("execution_banks_per_layer"),
                    streaming_metrics.get("fused_gate_up"),
                    float(streaming_metrics.get("hit_rate", 1.0) or 0.0),
                    int(streaming_metrics.get("pinned_hits", 0) or 0)
                    + int(streaming_metrics.get("cache_hits", 0) or 0),
                    int(streaming_metrics.get("cache_misses", 0) or 0),
                    int(streaming_metrics.get("evictions", 0) or 0),
                    int(streaming_metrics.get("loads", 0) or 0),
                    int(streaming_metrics.get("ssd_bytes_read", 0) or 0),
                    float(streaming_metrics.get("ssd_io_seconds", 0.0) or 0.0)
                    * 1000.0,
                    float(
                        streaming_metrics.get("ssd_decode_seconds", 0.0) or 0.0
                    )
                    * 1000.0,
                    float(
                        streaming_metrics.get("bank_bind_seconds", 0.0) or 0.0
                    )
                    * 1000.0,
                    float(
                        streaming_metrics.get("bank_materialize_seconds", 0.0)
                        or 0.0
                    )
                    * 1000.0,
                    float(
                        streaming_metrics.get("scratch_prefetch_wait_seconds", 0.0)
                        or 0.0
                    )
                    * 1000.0,
                    float(
                        streaming_metrics.get(
                            "scratch_mlx_materialize_seconds", 0.0
                        )
                        or 0.0
                    )
                    * 1000.0,
                    int(streaming_metrics.get("expert_major_calls", 0) or 0),
                    int(streaming_metrics.get("qmm_calls", 0) or 0),
                    int(streaming_metrics.get("sorted_prefill_groups", 0) or 0),
                    int(streaming_metrics.get("sorted_prefill_routes", 0) or 0),
                    int(streaming_metrics.get("sorted_qmm_calls", 0) or 0),
                    float(streaming_metrics.get("resident_fill_rate", 1.0) or 0.0),
                    streaming_metrics.get("hotlist_preloaded"),
                    streaming_metrics.get("optimistic_preloaded"),
                    float(
                        streaming_metrics.get("hotlist_preload_fill_rate", 1.0)
                        or 0.0
                    ),
                    float(
                        streaming_metrics.get("startup_preload_fill_rate", 1.0)
                        or 0.0
                    ),
                    float(metrics["metal_memory"]["active_start_bytes"]) / 1024**3,
                    float(metrics["metal_memory"]["active_end_bytes"]) / 1024**3,
                    float(metrics["metal_memory"]["peak_bytes"]) / 1024**3,
                )
                if (
                    streaming_metrics.get("streaming_mode") == "cache_only"
                    and float(
                        streaming_metrics.get("hotlist_preload_fill_rate", 0.0)
                        or 0.0
                    )
                    < 1.0
                ):
                    logger.warning(
                        "[benchmark-expert-streaming] pp=%d analytical cache "
                        "hotlist was only %.1f%% prefilled; compare only with a "
                        "run having the same warm-start coverage",
                        pp_len,
                        float(
                            streaming_metrics.get(
                                "hotlist_preload_fill_rate", 0.0
                            )
                            or 0.0
                        )
                        * 100.0,
                    )

            result = {
                "test_type": "single",
                "pp": pp_len,
                "tg": request.generation_length,
                **metrics,
            }
            run.results.append(result)

            await _send_event(run, {"type": "result", "data": result})

            # Store pp1024 gen_tps for speedup calculation
            if pp_len == 1024:
                single_pp1024_gen_tps = metrics["gen_tps"]

        # Phase 4: Batch tests
        # Each request has a unique UUID prefix (no cache hits)
        max_batch = max(request.batch_sizes) if request.batch_sizes else 0
        batch_prompts = [
            _generate_prompt(tokenizer, 1024, request.context_profile)
            for _ in range(max_batch)
        ]

        # Skip batch tests for engines without scheduler core (e.g. VLM/Diffusion)
        batch_core = _get_batch_benchmark_core(engine)
        if request.batch_sizes and batch_core is None:
            logger.info(
                "Batch test skipped: engine does not support concurrent batching"
            )
            current_test += len(request.batch_sizes)

        for batch_size in request.batch_sizes if batch_core is not None else []:
            current_test += 1
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "batch",
                    "message": f"Batch {batch_size}x: pp1024/tg{request.generation_length}",
                    "current": current_test,
                    "total": total_tests,
                },
            )

            window_start = time.monotonic()
            batch_metrics = await _run_batch_test(
                engine=engine,
                prompts=batch_prompts[:batch_size],
                prompt_tokens=1024,
                max_tokens=request.generation_length,
                batch_size=batch_size,
            )
            batch_metrics["system_metrics"] = _sample_window(run, window_start)
            batch_streaming = batch_metrics.get("expert_streaming")
            if isinstance(batch_streaming, dict):
                logger.info(
                    "[benchmark-expert-streaming-batch] batch=%d slots=%s "
                    "hit_rate=%.4f misses=%d evictions=%d ssd_bytes=%d "
                    "qmm_calls=%d metal_peak_gib=%.3f",
                    batch_size,
                    batch_streaming.get("cache_slots_per_layer"),
                    float(batch_streaming.get("hit_rate", 1.0) or 0.0),
                    int(batch_streaming.get("cache_misses", 0) or 0),
                    int(batch_streaming.get("evictions", 0) or 0),
                    int(batch_streaming.get("ssd_bytes_read", 0) or 0),
                    int(batch_streaming.get("qmm_calls", 0) or 0),
                    float(batch_metrics["metal_memory"]["peak_bytes"]) / 1024**3,
                )

            result = {
                "test_type": "batch",
                "pp": 1024,
                "tg": request.generation_length,
                **batch_metrics,
            }
            run.results.append(result)
            await _send_event(run, {"type": "result", "data": result})

        # Phase 5: Unload benchmark model
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "cleanup",
                "message": f"Unloading {request.model_id}...",
                "current": total_tests,
                "total": total_tests,
            },
        )
        try:
            await engine_pool._unload_engine(request.model_id)
            logger.info(f"Benchmark: unloaded {request.model_id} after benchmark")
        except Exception as e:
            logger.warning(f"Benchmark: failed to unload {request.model_id}: {e}")

        # Done
        overall_duration = time.perf_counter() - overall_start
        run.status = "completed"
        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "context_profile": request.context_profile.value,
                    "total_time": round(overall_duration, 1),
                    "total_tests": total_tests,
                },
            },
        )

        # Aligned prompts intentionally use non-standard PP4097/8193/etc.
        # Keep them local rather than mixing them into PP4096 leaderboard
        # buckets or depending on the remote service accepting arbitrary PP.
        if request.align_prompt_to_ane:
            run.upload_state["phase"] = "skipped"
            run.upload_state["skipped_reason"] = "ane_aligned_prompt"
            await _send_event(
                run,
                {
                    "type": "upload_skipped",
                    "reason": "ane_aligned_prompt",
                    "features": run.feature_flags,
                },
            )
            return

        # Upload results to omlx.ai (failures don't affect benchmark status)
        try:
            await _upload_to_omlx_ai(run, engine_pool)
        except Exception as e:
            logger.warning(f"Benchmark upload to omlx.ai failed: {e}")
            await _send_event(
                run,
                {
                    "type": "upload_done",
                    "data": {
                        "owner_hash": None,
                        "total": 0,
                        "success": 0,
                        "failed": 0,
                        "error": str(e),
                    },
                },
            )

    except asyncio.CancelledError:
        run.status = "cancelled"
        await _send_event(
            run,
            {
                "type": "error",
                "message": "Benchmark cancelled by user",
            },
        )
        # Try to unload the model on cancellation
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        run.status = "error"
        run.error_message = str(e)
        await _send_event(
            run,
            {
                "type": "error",
                "message": str(e),
            },
        )
        # Try to unload the model on error
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

    finally:
        if run.sampler is not None:
            try:
                run.sampler.stop()
            except Exception as e:  # noqa: BLE001
                logger.debug(f"Benchmark: sampler stop failed: {e}")
        _restore_speed_priority(engine_pool, previous_speed_priority)


async def _run_external_benchmark(run: BenchmarkRun) -> None:
    """Execute a benchmark run against an external OpenAI-compatible endpoint.

    No local model phases (unload/load/JIT warmup) and no community
    upload — external numbers measure someone else's hardware.
    """
    request = run.request
    total_tests = len(request.prompt_lengths) + len(request.batch_sizes)
    current_test = 0
    overall_start = time.perf_counter()
    client = ExternalAPIClient(request.external)

    try:
        # Warmup doubles as preflight: fail fast on bad URL/key and on
        # endpoints that do not return streamed usage (hard requirement
        # for accurate token counts) before any long test runs.
        await _send_event(
            run,
            {
                "type": "progress",
                "phase": "warmup",
                "message": "Warming up external endpoint...",
                "current": 0,
                "total": total_tests,
            },
        )
        await client.stream_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": _generate_external_prompt(
                        _warmup_prompt_tokens(request.warmup_mode),
                        request.context_profile,
                    ),
                }
            ],
            max_tokens=8,
            temperature=0.0,
        )
        logger.info(
            "Benchmark: external endpoint warmup complete "
            f"(mode={request.warmup_mode.value})"
        )

        # Single request tests
        for pp_len in _single_prompt_lengths(request):
            current_test += 1
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "single",
                    "message": f"Single: pp{pp_len}/tg{request.generation_length}",
                    "current": current_test,
                    "total": total_tests,
                },
            )

            metrics = await _run_external_single_test(
                client=client,
                prompt=_generate_external_prompt(pp_len, request.context_profile),
                max_tokens=request.generation_length,
            )

            result = {
                "test_type": "single",
                "pp": metrics["prompt_tokens"],
                "requested_pp": pp_len,
                "tg": request.generation_length,
                **metrics,
            }
            run.results.append(result)
            await _send_event(run, {"type": "result", "data": result})

        # Batch tests: concurrent requests with unique pp1024 prompts
        for batch_size in request.batch_sizes:
            current_test += 1
            await _send_event(
                run,
                {
                    "type": "progress",
                    "phase": "batch",
                    "message": f"Batch {batch_size}x: pp1024/tg{request.generation_length}",
                    "current": current_test,
                    "total": total_tests,
                },
            )

            batch_metrics = await _run_external_batch_test(
                client=client,
                prompts=[
                    _generate_external_prompt(1024, request.context_profile)
                    for _ in range(batch_size)
                ],
                max_tokens=request.generation_length,
                batch_size=batch_size,
            )

            result = {
                "test_type": "batch",
                "pp": batch_metrics["prompt_tokens"],
                "requested_pp": 1024,
                "tg": request.generation_length,
                **batch_metrics,
            }
            run.results.append(result)
            await _send_event(run, {"type": "result", "data": result})

        # Done
        overall_duration = time.perf_counter() - overall_start
        run.status = "completed"
        await _send_event(
            run,
            {
                "type": "done",
                "summary": {
                    "model_id": request.model_id,
                    "context_profile": request.context_profile.value,
                    "total_time": round(overall_duration, 1),
                    "total_tests": total_tests,
                },
            },
        )

        # External results measure remote hardware — never upload them to
        # the omlx.ai community leaderboard. Mirrors the experimental-
        # features skip so REST pollers see the same upload_state shape.
        run.upload_state["phase"] = "skipped"
        run.upload_state["skipped_reason"] = "external_endpoint"
        await _send_event(
            run,
            {
                "type": "upload_skipped",
                "reason": "external_endpoint",
                "features": [],
            },
        )

    except asyncio.CancelledError:
        run.status = "cancelled"
        await _send_event(
            run,
            {
                "type": "error",
                "message": "Benchmark cancelled by user",
            },
        )
    except Exception as e:
        logger.error(f"External benchmark error: {e}", exc_info=True)
        run.status = "error"
        run.error_message = str(e)
        await _send_event(
            run,
            {
                "type": "error",
                "message": str(e),
            },
        )
    finally:
        await client.aclose()
