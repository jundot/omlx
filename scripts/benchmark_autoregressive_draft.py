# SPDX-License-Identifier: Apache-2.0
"""Compare greedy target decode and full-model draft decode on local models."""

import argparse
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

from omlx.patches.mlx_lm_mtp.qwen35_model import apply
from omlx.speculative.autoregressive import GreedyQwenDraft, load_qwen_draft, prefill


def benchmark(cache_state, step, limit, prefill_seconds):
    output = []
    accepted = proposed = 0
    mx.reset_peak_memory()
    start = time.perf_counter()
    while len(output) < limit:
        tokens, num_accepted, num_proposed = step(limit - len(output))
        output.extend(tokens)
        accepted += num_accepted
        proposed += num_proposed
    mx.eval(cache_state())
    elapsed = time.perf_counter() - start
    print(
        f"{len(output) / elapsed:.2f} decode tok/s, "
        f"{prefill_seconds + elapsed:.2f} s including prefill, "
        f"accepted {accepted}/{proposed}, peak {mx.get_peak_memory() / 1e9:.2f} GB",
        flush=True,
    )
    return output


def baseline(target, prompt, limit):
    start = time.perf_counter()
    cache = prefill(target, prompt)
    prefill_seconds = time.perf_counter() - start
    token = prompt[-1]

    def step(_remaining):
        nonlocal token
        logits = target(mx.array([[token]]), cache=cache)
        token = mx.argmax(logits[0, -1]).item()
        return [token], 0, 0

    return benchmark(
        lambda: [layer.state for layer in cache], step, limit, prefill_seconds
    )


def speculative(target, draft, prompt, limit, depth):
    start = time.perf_counter()
    decode = GreedyQwenDraft(target, draft, prompt)
    prefill_seconds = time.perf_counter() - start

    def step(remaining):
        result = decode.step(min(depth, remaining - 1))
        return result.tokens, result.accepted, result.proposed

    return benchmark(
        lambda: [layer.state for layer in (*decode.target_cache, *decode.draft_cache)],
        step,
        limit,
        prefill_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--depth", type=int, nargs="+", default=[2])
    args = parser.parse_args()
    if args.tokens < 1 or any(depth < 1 for depth in args.depth):
        parser.error("--tokens and every --depth must be positive")

    apply()
    target, tokenizer = load(args.target)
    if target.model_type != "qwen3_5":
        parser.error("Expected a dense qwen3_5 target checkpoint")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for prompt in args.prompt
    ]

    expected = []
    for i, prompt in enumerate(prompts, start=1):
        print(f"Prompt {i} ({len(prompt)} tokens), target only", flush=True)
        expected.append(baseline(target, prompt, args.tokens))

    draft = load_qwen_draft(args.draft, args.target)
    matched = True
    for i, (prompt, reference) in enumerate(zip(prompts, expected, strict=True), 1):
        for depth in args.depth:
            print(f"Prompt {i}, depth {depth}", flush=True)
            actual = speculative(target, draft, prompt, args.tokens, depth)
            mismatch = next(
                (
                    index
                    for index, (a, b) in enumerate(zip(actual, reference))
                    if a != b
                ),
                None,
            )
            if mismatch is None:
                print(f"Matched all {len(reference)} greedy tokens", flush=True)
            else:
                print(f"First token mismatch at index {mismatch}", flush=True)
                matched = False
    return 0 if matched else 1


if __name__ == "__main__":
    raise SystemExit(main())
