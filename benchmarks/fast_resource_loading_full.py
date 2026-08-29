#!/usr/bin/env python3
"""Guarded full-model A/B benchmark for expert Fast Resource Loading."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from pathlib import Path

import mlx.core as mx
import psutil


def gib(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fast-resource-loading", action="store_true")
    parser.add_argument("--frl-scope", choices=("off", "scratch", "all"), default=None)
    parser.add_argument("--cache-experts", type=int, default=48)
    parser.add_argument("--scratch-experts", type=int, default=48)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt-repeat", type=int, default=48)
    parser.add_argument("--memory-limit-gib", type=int, default=72)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"Model directory does not exist: {model_path}")
    mx.set_memory_limit(args.memory_limit_gib * 1024**3)
    with contextlib.suppress(AttributeError, RuntimeError):
        mx.set_wired_limit(args.memory_limit_gib * 1024**3)
    mx.reset_peak_memory()

    from mlx_vlm import generate
    from mlx_vlm.utils import load as vlm_load

    from omlx.engine.vlm import _force_qwen4_exp_sanitize_on_load
    from omlx.expert_streaming import install_expert_streaming
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import (
        materialize_lazy_state,
        maybe_apply_pre_load_patches,
    )

    settings = ModelSettings(
        expert_streaming_enabled=True,
        expert_streaming_mode="cache_only",
        expert_streaming_cache_experts=args.cache_experts,
        expert_streaming_scratch_experts=args.scratch_experts,
        expert_streaming_execution_policy="checked",
        qwen4_ple_ssd_offload=True,
        mtp_enabled=True,
    )
    frl_scope = args.frl_scope or ("all" if args.fast_resource_loading else "off")
    report: dict = {
        "variant": "baseline" if frl_scope == "off" else f"frl_{frl_scope}",
        "pid": os.getpid(),
        "model": str(model_path),
        "cache_experts": args.cache_experts,
        "scratch_experts": args.scratch_experts,
        "memory_limit_gib": args.memory_limit_gib,
    }

    maybe_apply_pre_load_patches(model_path, model_settings=settings, for_vlm=True)
    load_started = time.perf_counter()
    with _force_qwen4_exp_sanitize_on_load(model_path):
        model, processor = vlm_load(str(model_path), lazy=True)
    runtime = install_expert_streaming(
        model,
        model_path,
        None,
        cache_experts=args.cache_experts,
        scratch_experts=args.scratch_experts,
        execution_policy="checked",
        streaming_mode="cache_only",
        hotlist_profile_dir=None,
        fast_resource_loading=frl_scope,
    )
    materialize_lazy_state(model)
    report["load_seconds"] = round(time.perf_counter() - load_started, 3)
    report["loaded_memory"] = {
        "mlx_active_gib": gib(mx.get_active_memory()),
        "mlx_peak_gib": gib(mx.get_peak_memory()),
        "process_rss_gib": gib(psutil.Process().memory_info().rss),
    }
    print(json.dumps({"stage": "loaded", **report}), flush=True)

    user_prompt = (
        "Explain the performance trade-offs in a memory-limited mixture-of-experts "
        "inference server, including storage latency, routing locality, cache eviction, "
        "prefill parallelism, and token-generation latency. "
    ) * args.prompt_repeat
    tokenizer = getattr(processor, "tokenizer", processor)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    result = generate(
        model,
        processor,
        prompt,
        max_tokens=args.max_tokens,
        temp=0.0,
        verbose=False,
    )
    report["generation"] = {
        "prompt_tokens": int(result.prompt_tokens),
        "generation_tokens": int(result.generation_tokens),
        "prompt_tps": round(float(result.prompt_tps), 4),
        "generation_tps": round(float(result.generation_tps), 4),
        "output": result.text,
    }
    report["final_memory"] = {
        "mlx_active_gib": gib(mx.get_active_memory()),
        "mlx_peak_gib": gib(mx.get_peak_memory()),
        "process_rss_gib": gib(psutil.Process().memory_info().rss),
    }
    streaming = runtime.stats()
    report["streaming"] = {
        key: streaming[key]
        for key in (
            "fast_resource_loading",
            "fast_resource_loading_scope",
            "ssd_bytes_read",
            "ssd_read_operations",
            "ssd_io_seconds",
            "ssd_decode_seconds",
            "frl_loads",
            "frl_bytes_read",
            "frl_read_operations",
            "frl_io_wait_seconds",
            "frl_copy_seconds",
            "scratch_prefetch_wait_seconds",
            "scratch_mlx_materialize_seconds",
            "bank_bind_seconds",
            "bank_materialize_seconds",
            "cache_hits",
            "cache_misses",
            "scratch_loads",
            "qmm_calls",
        )
    }
    print(json.dumps({"stage": "complete", **report}), flush=True)
    runtime.close()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
