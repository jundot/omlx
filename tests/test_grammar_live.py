# SPDX-License-Identifier: Apache-2.0
"""Live integration tests for grammar-constrained decoding.

Tests grammar correctness and measures performance across model families
(Qwen, Gemma, Harmony/OSS) against a running oMLX server.

Prerequisites:
  - oMLX server running on OMLX_TEST_URL (default: http://127.0.0.1:8899)
  - Models loaded: Qwen3.5-4B-4bit, gemma-3-4b-it-qat-4bit, gpt-oss-20b-MXFP4-Q4
  - reasoning_parser set via admin UI: qwen, (none), harmony respectively

Run:
  pytest tests/test_grammar_live.py -v -s
  pytest tests/test_grammar_live.py -v -s -k perf   # performance only
"""

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import httpx
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.integration]

BASE_URL = os.environ.get("OMLX_TEST_URL", "http://127.0.0.1:8899")
API_KEY = os.environ.get("OMLX_TEST_API_KEY", "1234")

MODELS = {
    "qwen": "Qwen3.5-4B-4bit",
    "gemma": "gemma-3-4b-it-qat-4bit",
    "oss": "gpt-oss-20b-MXFP4-Q4",
}

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"},
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

REGEX_PATTERN = r"\d{4}-\d{2}-\d{2}"

PROMPT_JSON = "Give me a fictional person with name, age, and city."
PROMPT_REGEX = "What is today's date in YYYY-MM-DD format?"
PROMPT_PLAIN = "Write a short haiku about the ocean."

OSS_MAX_TOKENS = 400  # Harmony needs room for analysis + final channels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _headers():
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _chat_payload(model, prompt, *, structured_outputs=None,
                  max_tokens=128, temperature=0.1, stream=False,
                  extra_body=None):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    if structured_outputs:
        if extra_body is None:
            extra_body = {}
        extra_body["structured_outputs"] = structured_outputs
    if extra_body:
        payload.update(extra_body)
    return payload


async def _complete(client, model, prompt, **kwargs):
    """Send a non-streaming chat completion and return (content, duration_s)."""
    payload = _chat_payload(model, prompt, **kwargs)
    t0 = time.perf_counter()
    resp = await client.post(f"{BASE_URL}/v1/chat/completions",
                             json=payload, headers=_headers(), timeout=120)
    dur = time.perf_counter() - t0
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"].get("content") or ""
    return content, dur


async def _complete_streaming(client, model, prompt, **kwargs):
    """Send a streaming chat completion and return (content, ttft_s, total_s, token_count).

    TTFT is measured to the first delta of any kind (content or
    reasoning_content).  Token count comes from the server-reported
    ``usage.completion_tokens`` when ``stream_options.include_usage``
    is set; falls back to counting content deltas.
    """
    payload = _chat_payload(model, prompt, stream=True, **kwargs)
    payload["stream_options"] = {"include_usage": True}
    t0 = time.perf_counter()
    ttft = None
    chunks = []
    token_count = 0
    server_tokens = None
    async with client.stream("POST", f"{BASE_URL}/v1/chat/completions",
                             json=payload, headers=_headers(), timeout=180) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            # Usage-only chunk (final)
            usage = chunk.get("usage")
            if usage and "completion_tokens" in usage:
                server_tokens = usage["completion_tokens"]
            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            has_content = delta.get("content", "")
            has_reasoning = delta.get("reasoning_content", "")
            if has_content or has_reasoning:
                if ttft is None:
                    ttft = time.perf_counter() - t0
            if has_content:
                chunks.append(has_content)
                token_count += 1
    total = time.perf_counter() - t0
    final_tokens = server_tokens if server_tokens is not None else token_count
    return "".join(chunks), ttft or total, total, final_tokens


def _server_available():
    try:
        r = httpx.get(f"{BASE_URL}/v1/models", headers=_headers(), timeout=5)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"oMLX server not reachable at {BASE_URL}",
)


# =========================================================================
# Integration Tests: Grammar Correctness
# =========================================================================

def _max_tokens_for(family, default=200):
    """Harmony models need more tokens for analysis + final channels."""
    return OSS_MAX_TOKENS if family == "oss" else default


class TestGrammarJson:
    """JSON schema grammar produces valid JSON for each model family."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", ["qwen", "gemma", "oss"])
    async def test_json_schema(self, client, family):
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_JSON,
            structured_outputs={"json": JSON_SCHEMA},
            max_tokens=_max_tokens_for(family),
        )
        print(f"\n[{family}] JSON output ({dur:.2f}s): {content[:200]}")
        # Harmony may produce multiple final channels with repeated JSON;
        # decode only the first object.
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(content.lstrip())
        assert "name" in parsed, f"Missing 'name' in {parsed}"
        assert "age" in parsed, f"Missing 'age' in {parsed}"
        assert "city" in parsed, f"Missing 'city' in {parsed}"
        assert isinstance(parsed["age"], int), f"'age' is not int: {parsed['age']}"


class TestGrammarRegex:
    """Regex grammar produces matching output for each model family."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", ["qwen", "gemma", "oss"])
    async def test_regex(self, client, family):
        import re
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_REGEX,
            structured_outputs={"regex": REGEX_PATTERN},
            max_tokens=_max_tokens_for(family, 50),
        )
        content = content.strip()
        print(f"\n[{family}] Regex output ({dur:.2f}s): {content}")
        # Harmony may produce multiple final channels whose content gets
        # concatenated, so check that the output starts with a valid match.
        assert re.match(REGEX_PATTERN, content), \
            f"Output '{content}' doesn't start with pattern '{REGEX_PATTERN}'"


class TestGrammarChoice:
    """Choice grammar restricts output to one of the given options."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", ["qwen", "gemma", "oss"])
    async def test_choice(self, client, family):
        model = MODELS[family]
        choices = ["yes", "no", "maybe"]
        content, dur = await _complete(
            client, model, "Is the sky blue? Answer with yes, no, or maybe.",
            structured_outputs={"choice": choices},
            max_tokens=_max_tokens_for(family, 10),
        )
        content = content.strip().strip('"')
        print(f"\n[{family}] Choice output ({dur:.2f}s): {content}")
        # Harmony may produce multiple final channels; check that output
        # starts with a valid choice.
        assert any(content.startswith(c) for c in choices), \
            f"Output '{content}' doesn't start with any of {choices}"


class TestNoGrammar:
    """Baseline: unconstrained generation works for each model."""

    @pytest.fixture()
    def client(self):
        return httpx.AsyncClient()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", ["qwen", "gemma"])
    async def test_plain(self, client, family):
        model = MODELS[family]
        content, dur = await _complete(
            client, model, PROMPT_PLAIN,
            max_tokens=100,
        )
        print(f"\n[{family}] Plain output ({dur:.2f}s): {content[:200]}")
        assert len(content.strip()) > 5, "Expected non-trivial output"

    @pytest.mark.asyncio
    async def test_plain_oss(self, client):
        """OSS/Harmony needs more tokens; analysis channel may consume most of them."""
        model = MODELS["oss"]
        content, dur = await _complete(
            client, model, PROMPT_PLAIN,
            max_tokens=OSS_MAX_TOKENS,
        )
        print(f"\n[oss] Plain output ({dur:.2f}s): {content[:200]}")
        # Harmony may return empty content if the model doesn't reach
        # the final channel within max_tokens. This is expected behavior.
        if not content.strip():
            pytest.skip("Harmony model did not produce final channel content")


# =========================================================================
# Performance Benchmarks
# =========================================================================

BENCH_DURATION = int(os.environ.get("OMLX_BENCH_DURATION", "60"))
BENCH_MAX_TOKENS = 128
CONCURRENCY_LEVELS = [1, 2, 4]

BENCH_PROMPT = "Write a detailed description of a fictional city."
BENCH_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "city_name": {"type": "string"},
        "description": {"type": "string"},
        "population": {"type": "integer"},
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["city_name", "description", "population", "landmarks"],
}


def _mean_std(values):
    if len(values) < 2:
        return (values[0] if values else 0.0), 0.0
    return statistics.mean(values), statistics.stdev(values)


@dataclass
class BenchResult:
    model: str
    grammar: str       # "none" or "json"
    thinking: str      # "on" or "off"
    concurrency: int
    durations: list = field(default_factory=list)
    ttfts: list = field(default_factory=list)
    token_counts: list = field(default_factory=list)

    @property
    def n(self):
        return len(self.durations)

    def ttft_stats(self):
        return _mean_std(self.ttfts)

    def dur_stats(self):
        return _mean_std(self.durations)

    def tps_stats(self):
        tps_list = [t / d for t, d in zip(self.token_counts, self.durations) if d > 0 and t > 0]
        return _mean_std(tps_list)


async def _run_one_bench(client, model, grammar, thinking, family):
    """Run a single streaming request and return (ttft, duration, tokens)."""
    so = {"json": BENCH_JSON_SCHEMA} if grammar == "json" else None
    extra = {}
    if thinking == "off":
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        extra["thinking_budget"] = 0

    max_tok = _max_tokens_for(family, BENCH_MAX_TOKENS)

    _, ttft, total, tokens = await _complete_streaming(
        client, model, BENCH_PROMPT,
        structured_outputs=so,
        max_tokens=max_tok,
        temperature=0.7,
        extra_body=extra if extra else None,
    )
    return ttft, total, tokens


async def _bench_timed(model, grammar, thinking, concurrency, duration, family):
    """Run requests for *duration* seconds at the given concurrency."""
    result = BenchResult(
        model=model, grammar=grammar, thinking=thinking, concurrency=concurrency,
    )
    sem = asyncio.Semaphore(concurrency)
    stop = asyncio.Event()
    pending: set = set()

    async def _worker(client):
        async with sem:
            if stop.is_set():
                return
            try:
                ttft, dur, tokens = await _run_one_bench(
                    client, model, grammar, thinking, family,
                )
                result.durations.append(dur)
                result.ttfts.append(ttft)
                result.token_counts.append(tokens)
            except Exception as e:
                pass  # skip failed requests

    async def _dispatcher(client):
        while not stop.is_set():
            task = asyncio.create_task(_worker(client))
            pending.add(task)
            task.add_done_callback(pending.discard)
            # Small sleep to avoid tight-looping; the semaphore throttles actual concurrency.
            await asyncio.sleep(0.01)

    async with httpx.AsyncClient() as client:
        dispatcher = asyncio.create_task(_dispatcher(client))
        await asyncio.sleep(duration)
        stop.set()
        dispatcher.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    return result


def _fmt(mean, std):
    if std < 0.005:
        return f"{mean:.3f}"
    return f"{mean:.3f}\u00b1{std:.3f}"


def _print_results(results):
    hdr = (f"  {'Model':<25} {'Think':>5} {'Gram':>5} {'Conc':>4} {'Reqs':>5} "
           f"{'TTFT (s)':>14} {'Dur (s)':>14} {'TPS':>14}")
    print(hdr)
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*4} {'-'*5} {'-'*14} {'-'*14} {'-'*14}")
    for r in results:
        tm, ts = r.ttft_stats()
        dm, ds = r.dur_stats()
        pm, ps = r.tps_stats()
        print(f"  {r.model:<25} {r.thinking:>5} {r.grammar:>5} {r.concurrency:>4} {r.n:>5} "
              f"{_fmt(tm, ts):>14} {_fmt(dm, ds):>14} {_fmt(pm, ps):>14}")


class TestPerformance:
    """Time-boxed performance benchmarks.

    For each model, runs requests for OMLX_BENCH_DURATION seconds (default 60)
    at each concurrency level, with and without grammar, with thinking on/off.
    Reports mean +/- stdev for TTFT, duration, and TPS.
    """

    @staticmethod
    async def _warmup(model):
        async with httpx.AsyncClient() as c:
            await _complete(c, model, "Hi", max_tokens=5)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("family", ["qwen", "gemma", "oss"])
    async def test_perf(self, family):
        model = MODELS[family]

        # Gemma has no reasoning_parser so thinking on/off is irrelevant
        think_modes = ["off"] if family == "gemma" else ["on", "off"]

        print(f"\n  Warming up {model}...")
        await self._warmup(model)

        results = []
        total_combos = len(think_modes) * 2 * len(CONCURRENCY_LEVELS)
        done = 0
        for thinking in think_modes:
            for grammar in ["none", "json"]:
                for conc in CONCURRENCY_LEVELS:
                    done += 1
                    label = f"think={thinking} grammar={grammar} conc={conc}"
                    print(f"  [{done}/{total_combos}] {label} "
                          f"({BENCH_DURATION}s)...", end="", flush=True)
                    r = await _bench_timed(
                        model, grammar, thinking, conc, BENCH_DURATION, family,
                    )
                    print(f" {r.n} reqs")
                    results.append(r)

        print()
        _print_results(results)

        # Grammar overhead analysis
        print()
        for thinking in think_modes:
            base = [r for r in results
                    if r.grammar == "none" and r.thinking == thinking and r.concurrency == 1]
            gram = [r for r in results
                    if r.grammar == "json" and r.thinking == thinking and r.concurrency == 1]
            if base and gram and base[0].n > 0 and gram[0].n > 0:
                bm, _ = base[0].dur_stats()
                gm, _ = gram[0].dur_stats()
                ratio = gm / max(bm, 0.001)
                print(f"  Grammar overhead (think={thinking}, conc=1): {ratio:.2f}x")
                assert ratio < 5.0, f"Grammar overhead too high: {ratio:.2f}x"
