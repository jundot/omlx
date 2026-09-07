#!/usr/bin/env python3
"""Serve deterministic, fixed-length admin throughput benchmarks.

Usage: python benchmarks/serve_affine4_benchmark.py serve [omlx serve arguments]

Single-request admin trials suppress termination tokens; ordinary requests and
warmup retain normal stopping. Results include prompt-token and output-text
hashes. Use aligned prompts to keep the existing benchmark upload disabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import uuid
from functools import wraps
from pathlib import Path
from types import SimpleNamespace


def _install_benchmark_overrides():
    import omlx.admin.benchmark as bench
    from omlx.scheduler import Scheduler, _make_suppress_logits_processor

    logger = logging.getLogger("omlx.benchmark.fixed_length")
    build_sampler = Scheduler._build_sampler_and_processors
    generate_prompt = bench._generate_prompt
    run_single = bench._run_single_test

    @wraps(build_sampler)
    def build(self, sampling_params, request=None):
        sampler, processors = build_sampler(self, sampling_params, request)
        if not getattr(request, "benchmark_trace", False):
            return sampler, processors
        if self._vlm_mtp_drafter is not None:
            raise RuntimeError(
                "Fixed-length benchmarks support standard/Lightning MTP decode, "
                "not external VLM MTP, which bypasses this suppression processor."
            )
        stop_ids = set(self._get_stop_tokens())
        stop_ids.update(sampling_params.stop_token_ids or [])
        if self._output_parser_factory is not None:
            stop_ids.update(self._output_parser_factory.stop_token_ids)
        suppress = _make_suppress_logits_processor(stop_ids)
        if suppress is not None:
            processors = [*processors, suppress]
        logger.info(
            "Benchmark-only termination suppression active: model=%s "
            "request=%s ids=%s; normal requests unchanged",
            self.config.model_name,
            request.request_id,
            sorted(stop_ids),
        )
        return sampler, processors

    @wraps(generate_prompt)
    def prompt(tokenizer, target_tokens, context_profile="code_python"):
        profile = bench.BenchmarkContextProfile(context_profile).value
        salt = uuid.uuid5(uuid.NAMESPACE_URL, f"omlx-benchmark:{profile}:{target_tokens}")
        previous_uuid = bench.uuid
        bench.uuid = SimpleNamespace(uuid4=lambda: salt)
        try:
            return generate_prompt(tokenizer, target_tokens, context_profile)
        finally:
            bench.uuid = previous_uuid

    @wraps(run_single)
    async def single(engine, prompt, max_tokens, pp_len, ane_trace_config=None):
        last_output = None

        async def stream_generate(**kwargs):
            nonlocal last_output
            async for output in engine.stream_generate(**kwargs):
                last_output = output
                yield output

        metrics = await run_single(
            SimpleNamespace(stream_generate=stream_generate),
            prompt,
            max_tokens,
            pp_len,
            ane_trace_config,
        )
        if (
            last_output is None
            or not last_output.finished
            or last_output.finish_reason != "length"
            or last_output.completion_tokens != max_tokens
            or metrics["completion_tokens"] != max_tokens
        ):
            raise RuntimeError(
                f"Fixed-length benchmark pp{pp_len}: requested {max_tokens}, "
                f"generated {metrics['completion_tokens']}, "
                f"finish_reason={getattr(last_output, 'finish_reason', None)}"
            )
        if metrics["cached_tokens"] != 0:
            raise RuntimeError("Fixed-prompt benchmark hit prefix cache; clear caches")
        text = last_output.text
        metrics["prompt_sha256"] = hashlib.sha256(
            json.dumps(prompt, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        metrics["output_sample"] = {
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "head": text[:256],
            "tail": text[-256:],
        }
        metrics["finish_reason"] = last_output.finish_reason
        metrics["generation_policy"] = "suppress_termination"
        return metrics

    Scheduler._build_sampler_and_processors = build
    bench._generate_prompt = prompt
    bench._run_single_test = single


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    _install_benchmark_overrides()
    from omlx.cli import main as omlx_main

    omlx_main()


if __name__ == "__main__":
    main()
