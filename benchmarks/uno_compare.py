"""Measure a configured server. Repeat with Uno off and on."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import httpx


async def request(client, args, prompt, delay=0):
    await asyncio.sleep(delay)
    started = time.perf_counter()
    first = None
    content, reasoning = [], []
    usage = finish = None
    completed = False
    async with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": args.temperature,
            "top_p": 0.95,
            "seed": 42,
            "max_tokens": args.max_tokens,
            "chat_template_kwargs": {"reasoning_effort": "high"},
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                completed = True
                break
            event = json.loads(payload)
            if event.get("error"):
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                visible = delta.get("content") or delta.get("reasoning_content")
                if visible and first is None:
                    first = time.perf_counter() - started
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    if not completed or usage is None or finish is None:
        raise RuntimeError("Incomplete response or missing usage")
    return {
        "response_seconds": time.perf_counter() - started,
        "first_delta_seconds": first,
        "usage": usage,
        "finish_reason": finish,
        "content": "".join(content),
        "reasoning": "".join(reasoning),
    }


async def measure(client, args, prompts):
    samples = []
    done = asyncio.Event()

    async def memory():
        from omlx.utils.proc_memory import get_phys_footprint

        while not done.is_set():
            footprint = get_phys_footprint(args.server_pid)
            if footprint <= 0:
                raise RuntimeError("Cannot read the server process footprint")
            samples.append(footprint)
            await asyncio.sleep(0.05)

    monitor = asyncio.create_task(memory()) if args.server_pid else None
    started = time.perf_counter()
    try:
        responses = await asyncio.gather(
            *(request(client, args, prompt, delay) for prompt, delay in prompts)
        )
        elapsed = time.perf_counter() - started
    finally:
        done.set()
        if monitor:
            await monitor
    return {
        "seconds": elapsed,
        "requests": responses,
        "aggregate_tokens_per_second": sum(
            r["usage"]["completion_tokens"] for r in responses
        )
        / elapsed,
        "sampled_peak_phys_footprint_bytes": max(samples) if samples else None,
    }


async def run(args):
    prompt = (
        args.prompt.read_text()
        if args.prompt
        else (
            "Write a Python function that merges overlapping half-open integer intervals. "
            "Explain its time complexity and include three tests."
        )
    )
    long_prompt = "\n".join(
        f"Record {i}: the reference value is {i % 17}." for i in range(300)
    )
    workloads = {
        "short": [("What is 17 times 19? Reply with just the number.", 0)],
        "coding": [(prompt, 0)],
        "long_context": [
            (long_prompt + "\nSummarize the pattern in these records.", 0)
        ],
        "overlap": [
            (prompt, 0),
            ("What is 17 times 19? Reply with just the number.", 0),
        ],
        "staggered": [(prompt, 0), ("What is 17 times 19?", 0.25)],
    }
    report = {
        "label": args.label,
        "platform": platform.platform(),
        "temperature": args.temperature,
        "top_p": 0.95,
        "max_tokens": args.max_tokens,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "results": [],
    }
    headers = {"Authorization": "Bearer " + os.environ["OMLX_API_KEY"]}
    async with httpx.AsyncClient(
        base_url=args.url, headers=headers, timeout=600
    ) as client:
        response = await client.get("/api/models")
        response.raise_for_status()
        report["model"] = next(
            m for m in response.json()["models"] if m["id"] == args.model
        )
        for trial in range(args.trials + 1):
            for name, prompts in workloads.items():
                result = await measure(client, args, prompts)
                report["results"].append(
                    {"workload": name, "trial": trial, "warmup": trial == 0, **result}
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{args.label} {name} trial={trial}: {result['seconds']:.3f}s",
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--prompt", type=Path)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--trials", type=int, default=3)
    asyncio.run(run(parser.parse_args()))
