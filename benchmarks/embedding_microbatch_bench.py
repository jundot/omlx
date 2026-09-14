#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Benchmark embedding cross-request microbatching against the legacy path.

The legacy runner submits one ``model.embed`` call per HTTP-like request to
oMLX's single MLX executor, matching the pre-microbatch execution pattern.
The microbatch runner submits the same requests through ``EmbeddingEngine``.

Example::

    PYTHONPATH=. python benchmarks/embedding_microbatch_bench.py \
        ~/.omlx/models/Qwen/Qwen3-Embedding-0.6B \
        --scenarios 8x1,32x1,8x8,1x64 --batch-size 64 --rounds 3
"""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial

import mlx.core as mx

from omlx.engine.embedding import EmbeddingEngine
from omlx.engine_core import get_mlx_executor
from omlx.models.embedding import EmbeddingOutput


@dataclass(frozen=True)
class Scenario:
    requests: int
    inputs_per_request: int

    @property
    def label(self) -> str:
        return f"{self.requests}x{self.inputs_per_request}"

    @property
    def input_count(self) -> int:
        return self.requests * self.inputs_per_request


def parse_scenarios(value: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for raw in value.split(","):
        try:
            requests, inputs = (int(part) for part in raw.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "scenarios must look like 8x1,8x8,1x64"
            ) from exc
        if requests <= 0 or inputs <= 0:
            raise argparse.ArgumentTypeError("scenario values must be positive")
        scenarios.append(Scenario(requests, inputs))
    if not scenarios:
        raise argparse.ArgumentTypeError("at least one scenario is required")
    return scenarios


def make_requests(scenario: Scenario) -> list[list[str]]:
    requests: list[list[str]] = []
    input_index = 0
    for _request_index in range(scenario.requests):
        request: list[str] = []
        for _ in range(scenario.inputs_per_request):
            repeats = 1 + input_index % 12
            request.append(
                f"embedding benchmark document {input_index}: "
                + "semantic retrieval sample text " * repeats
            )
            input_index += 1
        requests.append(request)
    return requests


async def run_legacy_requests(
    engine: EmbeddingEngine,
    requests: list[list[str]],
    *,
    max_length: int,
) -> list[EmbeddingOutput]:
    """Run the pre-microbatch pattern: one executor job per request."""
    model = engine._model
    if model is None:
        raise RuntimeError("embedding engine is not started")
    loop = asyncio.get_running_loop()

    async def run_one(inputs: list[str]) -> EmbeddingOutput:
        def embed_sync() -> EmbeddingOutput:
            try:
                return model.embed(
                    inputs=inputs,
                    max_length=max_length,
                    padding=True,
                    truncation=True,
                )
            finally:
                mx.synchronize()
                mx.clear_cache()

        return await loop.run_in_executor(get_mlx_executor(), embed_sync)

    return await asyncio.gather(*(run_one(inputs) for inputs in requests))


async def run_microbatched_requests(
    engine: EmbeddingEngine,
    requests: list[list[str]],
    *,
    max_length: int,
) -> list[EmbeddingOutput]:
    return await asyncio.gather(
        *(
            engine.embed(
                inputs,
                max_length=max_length,
                padding=True,
                truncation=True,
            )
            for inputs in requests
        )
    )


def embedding_agreement(
    expected: list[EmbeddingOutput], actual: list[EmbeddingOutput]
) -> tuple[float, float]:
    if len(expected) != len(actual):
        raise RuntimeError("benchmark modes returned different request counts")
    maximum = 0.0
    minimum_cosine = 1.0
    for expected_output, actual_output in zip(expected, actual):
        if len(expected_output.embeddings) != len(actual_output.embeddings):
            raise RuntimeError("benchmark modes returned different input counts")
        for expected_vector, actual_vector in zip(
            expected_output.embeddings, actual_output.embeddings
        ):
            if len(expected_vector) != len(actual_vector):
                raise RuntimeError("benchmark modes returned different dimensions")
            maximum = max(
                maximum,
                max(
                    (
                        abs(expected_value - actual_value)
                        for expected_value, actual_value in zip(
                            expected_vector, actual_vector
                        )
                    ),
                    default=0.0,
                ),
            )
            dot_product = sum(
                expected_value * actual_value
                for expected_value, actual_value in zip(
                    expected_vector, actual_vector
                )
            )
            expected_norm = math.sqrt(sum(value * value for value in expected_vector))
            actual_norm = math.sqrt(sum(value * value for value in actual_vector))
            cosine = (
                dot_product / (expected_norm * actual_norm)
                if expected_norm > 0 and actual_norm > 0
                else float(expected_vector == actual_vector)
            )
            minimum_cosine = min(minimum_cosine, cosine)
    return maximum, minimum_cosine


async def time_call(
    call: Callable[[], Awaitable[list[EmbeddingOutput]]],
) -> tuple[float, list[EmbeddingOutput]]:
    started = time.perf_counter()
    outputs = await call()
    return time.perf_counter() - started, outputs


async def benchmark(args: argparse.Namespace) -> None:
    scenarios = parse_scenarios(args.scenarios)
    largest_concurrency = max(scenario.requests for scenario in scenarios)
    engine = EmbeddingEngine(
        args.model,
        batch_size=args.batch_size,
        microbatch_wait_ms=args.wait_ms,
        max_pending_requests=largest_concurrency,
    )
    await engine.start()
    try:
        print(
            f"model={args.model} batch_size={args.batch_size} "
            f"wait_ms={args.wait_ms:g} rounds={args.rounds}"
        )
        print()
        print(
            "| scenario | legacy median | microbatch median | speedup "
            "| max delta | min cosine |"
        )
        print("|---:|---:|---:|---:|---:|---:|")

        for scenario in scenarios:
            requests = make_requests(scenario)
            legacy_call = partial(
                run_legacy_requests,
                engine,
                requests,
                max_length=args.max_length,
            )
            microbatch_call = partial(
                run_microbatched_requests,
                engine,
                requests,
                max_length=args.max_length,
            )

            legacy_reference: list[EmbeddingOutput] = []
            microbatch_reference: list[EmbeddingOutput] = []
            for _ in range(args.warmup):
                legacy_reference = await legacy_call()
                microbatch_reference = await microbatch_call()

            legacy_times: list[float] = []
            microbatch_times: list[float] = []
            for round_index in range(args.rounds):
                modes = (
                    (("legacy", legacy_call), ("microbatch", microbatch_call))
                    if round_index % 2 == 0
                    else (("microbatch", microbatch_call), ("legacy", legacy_call))
                )
                for mode, call in modes:
                    elapsed, outputs = await time_call(call)
                    if mode == "legacy":
                        legacy_times.append(elapsed)
                        legacy_reference = outputs
                    else:
                        microbatch_times.append(elapsed)
                        microbatch_reference = outputs

            legacy_median = statistics.median(legacy_times)
            microbatch_median = statistics.median(microbatch_times)
            speedup = legacy_median / microbatch_median
            delta, cosine = embedding_agreement(
                legacy_reference, microbatch_reference
            )
            print(
                f"| {scenario.label} ({scenario.input_count} inputs) "
                f"| {legacy_median:.3f}s "
                f"| {microbatch_median:.3f}s "
                f"| {speedup:.2f}x "
                f"| {delta:.3g} "
                f"| {cosine:.8f} |"
            )
    finally:
        await engine.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local embedding model path")
    parser.add_argument("--scenarios", default="8x1,32x1,8x8,1x64")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--wait-ms", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_length <= 0:
        parser.error("batch-size and max-length must be positive")
    if args.wait_ms < 0 or args.warmup < 0 or args.rounds <= 0:
        parser.error("wait-ms and warmup cannot be negative; rounds must be positive")
    asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
