# SPDX-License-Identifier: Apache-2.0
"""Benchmark execution logic for oMLX admin panel.

Provides single-request and continuous-batching benchmarks with
real-time progress reporting via SSE events.
"""

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, field_validator

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


class BenchmarkRequest(BaseModel):
    """Request model for starting a benchmark."""

    model_id: str
    prompt_lengths: list[int]
    generation_length: int = 128
    batch_sizes: list[int] = []
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


@dataclass
class BenchmarkRun:
    """Tracks the state of a running benchmark."""

    bench_id: str
    request: BenchmarkRequest
    status: str = "running"  # running, completed, cancelled, error
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    task: Optional[asyncio.Task] = None
    results: list[dict] = field(default_factory=list)
    error_message: str = ""
    # Experimental flags active when the benchmark started. When non-empty
    # the run's results are not uploaded to omlx.ai community benchmarks
    # because experimental features skew the numbers.
    experimental_features: list[str] = field(default_factory=list)


def get_run(bench_id: str) -> Optional[BenchmarkRun]:
    """Get a benchmark run by ID."""
    return _benchmark_runs.get(bench_id)


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



def _generate_prompt(tokenizer: Any, target_tokens: int) -> str:
    """Generate a prompt with exactly target_tokens tokens.

    Uses a unique UUID prefix to prevent SSD cache hits from previous sessions.
    """
    unique_prefix = f"BENCH-{uuid.uuid4().hex} "
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "In the realm of artificial intelligence, large language models "
        "have demonstrated remarkable capabilities across diverse tasks. "
    )

    # Build a large enough text
    text = unique_prefix + filler * (target_tokens // 10 + 1)
    tokens = tokenizer.encode(text)

    if len(tokens) < target_tokens:
        # Need more tokens, repeat more
        text = unique_prefix + filler * (target_tokens // 5 + 1)
        tokens = tokenizer.encode(text)

    # Truncate to exact target length
    tokens = tokens[:target_tokens]
    return tokenizer.decode(tokens)


def _compute_single_metrics(
    prompt_tokens: int,
    completion_tokens: int,
    start_time: float,
    first_token_time: float,
    end_time: float,
    peak_memory: int,
    cached_tokens: int,
) -> dict:
    """Compute all metrics for a single request benchmark."""
    ttft_s = first_token_time - start_time
    gen_duration = end_time - first_token_time
    e2e_duration = end_time - start_time

    ttft_ms = ttft_s * 1000
    tpot_ms = (gen_duration / max(completion_tokens - 1, 1)) * 1000
    gen_tps = completion_tokens / max(gen_duration, 1e-9)
    # wall_tg_tps: gen tokens over total wall (includes prefill). Used as the
    # 1x baseline for Continuous Batching speedup so the ratio is symmetric
    # with batch tg_tps (which is also total_gen / wall_time).
    wall_tg_tps = completion_tokens / max(e2e_duration, 1e-9)
    processing_tps = prompt_tokens / max(ttft_s, 1e-9)
    total_throughput = (prompt_tokens + completion_tokens) / max(e2e_duration, 1e-9)

    return {
        "ttft_ms": round(ttft_ms, 1),
        "tpot_ms": round(tpot_ms, 2),
        "gen_tps": round(gen_tps, 1),
        "wall_tg_tps": round(wall_tg_tps, 1),
        "processing_tps": round(processing_tps, 1),
        "e2e_latency_s": round(e2e_duration, 3),
        "total_throughput": round(total_throughput, 1),
        "peak_memory_bytes": peak_memory,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
    }


async def _send_event(run: BenchmarkRun, event: dict) -> None:
    """Send an SSE event to the run's queue."""
    await run.queue.put(event)


async def _run_single_test(
    engine: Any,
    prompt: str,
    max_tokens: int,
    pp_len: int,
) -> dict:
    """Run a single request benchmark test and return metrics."""
    # Reset peak memory tracking
    try:
        mx.reset_peak_memory()
    except Exception:
        pass

    start_time = time.perf_counter()
    first_token_time = None
    last_output = None
    prev_completion_tokens = 0

    async for output in engine.stream_generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    ):
        # Detect first generated token via completion_tokens count,
        # not new_text. Some models (e.g. Harmony/gpt-oss) produce
        # protocol tokens that don't yield visible new_text.
        if first_token_time is None and output.completion_tokens > prev_completion_tokens:
            first_token_time = time.perf_counter()
        prev_completion_tokens = output.completion_tokens
        last_output = output

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    # Get peak memory
    try:
        peak_memory = mx.get_peak_memory()
    except Exception:
        peak_memory = 0

    prompt_tokens = last_output.prompt_tokens if last_output else 0
    completion_tokens = last_output.completion_tokens if last_output else 0
    cached_tokens = last_output.cached_tokens if last_output else 0

    if cached_tokens > 0:
        logger.warning(
            f"Benchmark test pp{pp_len} had {cached_tokens} cached tokens "
            f"(expected 0). Results may not reflect true prefill performance."
        )

    return _compute_single_metrics(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        peak_memory=peak_memory,
        cached_tokens=cached_tokens,
    )



async def _run_batch_test_via_stream_generate(
    engine: Any,
    prompts: list[str],
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Batch test path for engines that only expose high-level stream_generate
    (no engine_core). Used for DFlashEngine where concurrent requests
    serialize through _active_request; metrics will reflect serial behavior.
    """

    async def _single_request(prompt: str) -> dict:
        start = time.perf_counter()
        first_token = None
        completion_tokens = 0
        async for output in engine.stream_generate(
            prompt=prompt, max_tokens=max_tokens, temperature=0.0
        ):
            ct = getattr(output, "completion_tokens", None)
            if ct is not None and ct > 0 and first_token is None:
                first_token = time.perf_counter()
            if ct is not None:
                completion_tokens = ct
            if getattr(output, "finished", False):
                break
        end = time.perf_counter()
        if first_token is None:
            first_token = end
        return {
            "ttft_s": first_token - start,
            "first_token_abs": first_token,
            "completion_tokens": completion_tokens,
            "wall_s": end - start,
        }

    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_single_request(prompts[i]) for i in range(batch_size)]
    )
    wall_end = time.perf_counter()

    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = prompt_tokens * batch_size
    wall_time = wall_end - wall_start
    avg_ttft_ms = (sum(r["ttft_s"] for r in results) / batch_size) * 1000

    # pp TPS: total prompt tokens / time until ALL requests finish prefill
    max_first_token = max(r["first_token_abs"] for r in results)
    prefill_wall_time = max_first_token - wall_start
    pp_tps = total_prompt_tokens / max(prefill_wall_time, 1e-9)

    # tg TPS: wall-aggregate (total_gen / wall_time). Same formula used in
    # the engine_core batch path so DFlash ↔ BatchedEngine ratios are
    # symmetric. The alternative (gen_wall_time = wall_end - max_first_token)
    # is inflated for any engine that serializes prefill or decode, and
    # makes cross-engine speedup columns meaningless. See _run_batch_test.
    tg_tps = total_gen_tokens / max(wall_time, 1e-9)

    return {
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "e2e_latency_s": round(wall_time, 3),
        "total_gen_tokens": total_gen_tokens,
        "batch_size": batch_size,
    }


async def _run_batch_test(
    engine: Any,
    prompts: list[str],
    prompt_tokens: int,
    max_tokens: int,
    batch_size: int,
) -> dict:
    """Run a continuous batching benchmark test.

    Submits batch_size concurrent requests via the engine and measures
    aggregate throughput including pp TPS and tg TPS.

    Args:
        prompts: List of prompts (one per request). For same-prompt tests,
                 all entries are identical. For different-prompt tests, each
                 has a unique UUID prefix.
        prompt_tokens: Number of prompt tokens per request (for pp TPS calc).
    """
    # Dispatch: engines exposing engine_core (BatchedEngine, VLMBatchedEngine)
    # use add_request/stream_outputs; DFlashEngine uses stream_generate (high
    # level) and serializes requests via its _active_request lock.
    if not hasattr(engine, "_engine"):
        return await _run_batch_test_via_stream_generate(
            engine=engine,
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            max_tokens=max_tokens,
            batch_size=batch_size,
        )

    from ..request import SamplingParams

    engine_core = engine._engine

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )

    async def _single_request(prompt: str) -> dict:
        """Run a single request within the batch."""
        start = time.perf_counter()
        first_token = None
        tokens = 0
        prev_tokens = 0

        request_id = await engine_core.add_request(
            prompt=prompt,
            sampling_params=sampling_params,
        )

        async for output in engine_core.stream_outputs(request_id):
            if first_token is None and output.completion_tokens > prev_tokens:
                first_token = time.perf_counter()
            prev_tokens = output.completion_tokens
            if output.finished:
                tokens = output.completion_tokens

        end = time.perf_counter()
        if first_token is None:
            first_token = end

        return {
            "ttft_s": first_token - start,
            "first_token_abs": first_token,
            "end_abs": end,
            "completion_tokens": tokens,
        }

    # Submit all requests concurrently
    wall_start = time.perf_counter()
    results = await asyncio.gather(
        *[_single_request(prompts[i]) for i in range(batch_size)]
    )
    wall_end = time.perf_counter()

    # Aggregate metrics
    total_gen_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = prompt_tokens * batch_size
    wall_time = wall_end - wall_start
    avg_ttft_ms = (sum(r["ttft_s"] for r in results) / batch_size) * 1000

    # pp TPS: total prompt tokens / time until ALL requests finish prefill
    max_first_token = max(r["first_token_abs"] for r in results)
    prefill_wall_time = max_first_token - wall_start
    pp_tps = total_prompt_tokens / max(prefill_wall_time, 1e-9)

    # tg TPS: wall-aggregate (total_gen / wall_time), same as the DFlash
    # path. Honest cross-engine ratio with the Single Request wall_tg_tps
    # baseline — see _run_batch_test_via_stream_generate.
    tg_tps = total_gen_tokens / max(wall_time, 1e-9)

    return {
        "pp_tps": round(pp_tps, 1),
        "tg_tps": round(tg_tps, 1),
        "avg_ttft_ms": round(avg_ttft_ms, 1),
        "e2e_latency_s": round(wall_time, 3),
        "total_gen_tokens": total_gen_tokens,
        "batch_size": batch_size,
    }


OMLX_AI_API_URL = "https://omlx.ai/api/benchmarks"

# Quantization patterns to strip from model directory names
_QUANT_SUFFIXES = re.compile(
    r"[-_](2bit|3bit|4bit|6bit|8bit|fp16|bf16|fp32|MXFP4|NVFP4)$", re.IGNORECASE
)
_MLX_SUFFIXES = re.compile(r"[-_]?MLX[-_]?", re.IGNORECASE)


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


def _clean_model_name(model_id: str, quantization: str) -> str:
    """Clean model directory name for display as model_name.

    Strips quantization suffixes and MLX markers.
    e.g. "Qwen3-30B-A3B-4bit" → "Qwen3-30B-A3B"
    """
    name = model_id
    name = _QUANT_SUFFIXES.sub("", name)
    name = _MLX_SUFFIXES.sub("", name)
    return name.strip("-_ ")


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
    if cf_mitigated == "challenge" or "just a moment" in body_head or "cf-chl" in body_head:
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

    # Skip upload when experimental features were active during the run.
    # These features (DFlash, SpecPrefill, TurboQuant KV) skew throughput
    # and would pollute the community leaderboard if mixed in unmarked.
    if run.experimental_features:
        await _send_event(run, {
            "type": "upload_skipped",
            "reason": "experimental_features",
            "features": list(run.experimental_features),
        })
        logger.info(
            f"Benchmark upload skipped: experimental features active: "
            f"{run.experimental_features}"
        )
        return

    await _send_event(run, {
        "type": "progress",
        "phase": "upload",
        "message": "Uploading to community benchmarks...",
        "current": 0,
        "total": 0,
    })

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
    model_name = _clean_model_name(run.request.model_id, quantization)

    # Generate submission group
    submission_group = str(uuid.uuid4())

    # Collect single results and batch results
    single_results = [r for r in run.results if r.get("test_type") == "single"]
    batch_results = [r for r in run.results if r.get("test_type") == "batch"]

    # Build batching_results from batch data
    batching_results = []
    pp1024_single = next(
        (r for r in single_results if r.get("pp") == 1024), None
    )
    if pp1024_single and batch_results:
        # Use wall-aggregate baseline so 1x ↔ Nx ratios stay honest across
        # engine types. Single Request table keeps showing gen_tps (peak
        # decode rate) for context-length comparisons; Continuous Batching
        # column needs the symmetric metric.
        baseline_tps = pp1024_single["wall_tg_tps"]
        batching_results.append({
            "batch_size": 1,
            "tg_tps": baseline_tps,
            "speedup": 1.0,
        })
        for br in batch_results:
            speedup = round(br["tg_tps"] / baseline_tps, 2) if baseline_tps > 0 else 1.0
            batching_results.append({
                "batch_size": br["batch_size"],
                "tg_tps": br["tg_tps"],
                "speedup": speedup,
            })

    success_count = 0
    failed_count = 0

    for result in single_results:
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
            "pp_tps": result["processing_tps"],
            # Community board metric: peak decode rate (gen_tps, gen-only),
            # NOT the wall-aggregate used for in-UI Continuous Batching
            # speedup. Do not "tidy" this to wall_tg_tps.
            "tg_tps": result["gen_tps"],
            "ttft_ms": result.get("ttft_ms"),
            "peak_memory_gb": peak_mem_gb,
            "submission_group": submission_group,
        }

        if owner_hash_full:
            payload["owner_hash"] = owner_hash_full

        # Attach batching_results only to the first submission (lowest context_length)
        if context_length == single_results[0]["pp"] and batching_results:
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
                await _send_event(run, {
                    "type": "upload",
                    "data": {
                        "context_length": context_length,
                        "id": data.get("id"),
                        "url": data.get("url"),
                    },
                })
            elif resp.status_code == 409:
                data = resp.json()
                success_count += 1  # Duplicate is still ok
                await _send_event(run, {
                    "type": "upload",
                    "data": {
                        "context_length": context_length,
                        "id": data.get("existing_id"),
                        "url": data.get("existing_url"),
                        "duplicate": True,
                    },
                })
            else:
                failed_count += 1
                error_msg = _sanitize_upload_error(resp)
                await _send_event(run, {
                    "type": "upload",
                    "data": {
                        "context_length": context_length,
                        "error": error_msg,
                    },
                })
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
            await _send_event(run, {
                "type": "upload",
                "data": {
                    "context_length": context_length,
                    "error": str(e),
                },
            })
            logger.warning(f"Benchmark upload error for pp{context_length}: {e}")

    await _send_event(run, {
        "type": "upload_done",
        "data": {
            "owner_hash": owner_hash_display,
            "total": len(single_results),
            "success": success_count,
            "failed": failed_count,
        },
    })

    logger.info(
        f"Benchmark upload complete: {success_count}/{len(single_results)} succeeded"
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
    total_tests = len(request.prompt_lengths) + len(request.batch_sizes)
    current_test = 0
    overall_start = time.perf_counter()

    try:
        # Snapshot experimental flags at run start. Settings can change mid-run
        # (user toggling DFlash/SpecPrefill/TurboQuant), and the produced
        # numbers are tied to whatever was active when generation actually ran.
        sm = getattr(engine_pool, "_settings_manager", None)
        if sm is not None:
            try:
                s = sm.get_settings(request.model_id)
                if getattr(s, "dflash_enabled", False):
                    run.experimental_features.append("dflash")
                if getattr(s, "specprefill_enabled", False):
                    run.experimental_features.append("specprefill")
                if getattr(s, "turboquant_kv_enabled", False):
                    run.experimental_features.append("turboquant")
            except Exception as e:
                logger.warning(
                    f"Benchmark: failed to read experimental flags for "
                    f"{request.model_id}: {e}"
                )

        # Phase 1: Unload all loaded models
        loaded_ids = engine_pool.get_loaded_model_ids()
        if loaded_ids:
            await _send_event(run, {
                "type": "progress",
                "phase": "unload",
                "message": f"Unloading {len(loaded_ids)} model(s)...",
                "current": 0,
                "total": total_tests,
            })
            for model_id in loaded_ids:
                try:
                    await engine_pool._unload_engine(model_id)
                    logger.info(f"Benchmark: unloaded {model_id}")
                except Exception as e:
                    logger.warning(f"Benchmark: failed to unload {model_id}: {e}")

        # Phase 2: Load the target model
        await _send_event(run, {
            "type": "progress",
            "phase": "load",
            "message": f"Loading {request.model_id}...",
            "current": 0,
            "total": total_tests,
        })
        engine = await engine_pool.get_engine(request.model_id, force_lm=True)
        logger.info(f"Benchmark: loaded {request.model_id}")

        # Generate prompts for all needed lengths
        tokenizer = engine.tokenizer
        prompts: dict[int, str] = {}
        for pp_len in request.prompt_lengths:
            prompts[pp_len] = _generate_prompt(tokenizer, pp_len)

        # Ensure pp1024 prompt exists for batch tests
        if request.batch_sizes and 1024 not in prompts:
            prompts[1024] = _generate_prompt(tokenizer, 1024)

        # Warmup: run a short request to trigger JIT compilation,
        # Metal shader compilation, and KV cache initialization.
        # Without this, the first real benchmark test absorbs all
        # one-time overhead and shows artificially low pp TPS.
        await _send_event(run, {
            "type": "progress",
            "phase": "warmup",
            "message": "Warming up (JIT compile)...",
            "current": 0,
            "total": total_tests,
        })
        warmup_prompt = _generate_prompt(tokenizer, 32)
        async for _ in engine.stream_generate(
            prompt=warmup_prompt, max_tokens=8, temperature=0.0
        ):
            pass
        logger.info("Benchmark: warmup complete")

        # Phase 3: Single request tests
        single_pp1024_gen_tps = None

        for pp_len in request.prompt_lengths:
            current_test += 1
            await _send_event(run, {
                "type": "progress",
                "phase": "single",
                "message": f"Single: pp{pp_len}/tg{request.generation_length}",
                "current": current_test,
                "total": total_tests,
            })

            metrics = await _run_single_test(
                engine=engine,
                prompt=prompts[pp_len],
                max_tokens=request.generation_length,
                pp_len=pp_len,
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
        batch_prompts = [_generate_prompt(tokenizer, 1024) for _ in range(max_batch)]

        # Skip batch tests for engines without scheduler core (e.g. DFlashEngine)
        # NOTE: DFlashEngine doesn't expose engine_core (`_engine`) but does
        # support concurrent requests by serializing them through its
        # _active_request lock. Running batch test on DFlash yields valid
        # metrics that show serialization behavior (4x wall time, no
        # aggregate gain) — strictly more useful than silently skipping.
        for batch_size in request.batch_sizes:
            current_test += 1
            await _send_event(run, {
                "type": "progress",
                "phase": "batch",
                "message": f"Batch {batch_size}x: pp1024/tg{request.generation_length}",
                "current": current_test,
                "total": total_tests,
            })

            batch_metrics = await _run_batch_test(
                engine=engine,
                prompts=batch_prompts[:batch_size],
                prompt_tokens=1024,
                max_tokens=request.generation_length,
                batch_size=batch_size,
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
        await _send_event(run, {
            "type": "progress",
            "phase": "cleanup",
            "message": f"Unloading {request.model_id}...",
            "current": total_tests,
            "total": total_tests,
        })
        try:
            await engine_pool._unload_engine(request.model_id)
            logger.info(f"Benchmark: unloaded {request.model_id} after benchmark")
        except Exception as e:
            logger.warning(f"Benchmark: failed to unload {request.model_id}: {e}")

        # Done
        overall_duration = time.perf_counter() - overall_start
        run.status = "completed"
        await _send_event(run, {
            "type": "done",
            "summary": {
                "model_id": request.model_id,
                "total_time": round(overall_duration, 1),
                "total_tests": total_tests,
            },
        })

        # Upload results to omlx.ai (failures don't affect benchmark status)
        try:
            await _upload_to_omlx_ai(run, engine_pool)
        except Exception as e:
            logger.warning(f"Benchmark upload to omlx.ai failed: {e}")
            await _send_event(run, {
                "type": "upload_done",
                "data": {
                    "owner_hash": None,
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "error": str(e),
                },
            })

    except asyncio.CancelledError:
        run.status = "cancelled"
        await _send_event(run, {
            "type": "error",
            "message": "Benchmark cancelled by user",
        })
        # Try to unload the model on cancellation
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        run.status = "error"
        run.error_message = str(e)
        await _send_event(run, {
            "type": "error",
            "message": str(e),
        })
        # Try to unload the model on error
        try:
            await engine_pool._unload_engine(request.model_id)
        except Exception:
            pass
